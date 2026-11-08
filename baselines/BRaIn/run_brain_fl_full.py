"""
Full BRaIn FL Implementation
Proper implementation with Elasticsearch BM25, method-level parsing, query expansion, and rescoring
"""
import json
import argparse
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import math
from dotenv import load_dotenv
import signal
from contextlib import contextmanager

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from elasticsearch import Elasticsearch
from langchain_openai import ChatOpenAI
import networkx as nx
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Import LLM provider utility
from utilities.llm_provider import get_llm

# Load environment variables from root .env file
load_dotenv(dotenv_path="../../.env")

# Download NLTK data if needed
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)


@contextmanager
def timeout_context(seconds):
    """Context manager for timeouts (Unix only)"""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class FullBRaInFL:
    """Full BRaIn Fault Localization"""

    def __init__(self, model_name: str, es_host: str = "localhost", es_port: int = 9200):
        self.model_name = model_name
        # Use the llm_provider utility to support multiple models (GPT, DeepSeek, etc.)
        self.llm = get_llm(model_name)
        self.es = Elasticsearch([f"http://{es_host}:{es_port}"])
        self.index_name = "ci_repair_repos"
        self.stop_words = set(stopwords.words('english'))

        # Timeout settings
        self.llm_timeout = 30  # seconds per LLM call
        self.max_methods_per_instance = 100  # limit methods to prevent timeout
        self.min_relevant_methods = 10  # early stop if we find enough relevant methods

    def run_fl(self, log_details_entry: Dict, index: int) -> Dict:
        """Run full BRaIn FL on a single entry"""
        entry_id = log_details_entry.get("id", str(index))
        sha_fail = log_details_entry.get("sha_fail", "")

        print(f"\n[{index+1}] Processing ID={entry_id}, SHA={sha_fail[:7]}...")

        # Extract ALL information from log_details
        failed_job = log_details_entry.get("failed_job", [{}])[0] if log_details_entry.get("failed_job") else {}
        error_context = log_details_entry.get("error_context", [])
        error_types = log_details_entry.get("error_types", [])

        # Build comprehensive bug description using ALL available info
        bug_title = f"CI Failure in {failed_job.get('job', 'unknown')}"
        bug_description_parts = []

        # 1. Add error context
        if error_context:
            bug_description_parts.extend(error_context[:5])

        # 2. Add error type information
        for error_type in error_types:
            category = error_type.get("category", "")
            subcategory = error_type.get("subcategory", "")
            if category:
                bug_description_parts.append(f"{category}: {subcategory}")

        # 3. Add failed command
        failed_command = failed_job.get("command", "")
        if failed_command:
            bug_description_parts.append(f"Command: {failed_command}")

        bug_description = " ".join(bug_description_parts) if bug_description_parts else "CI build failed"

        # Extract baseline's relevant files with line numbers and reasons
        baseline_relevant_files = []
        baseline_file_details = {}
        for rf in log_details_entry.get("relevant_files", []):
            file_path = rf.get("file", "")
            if file_path:
                # Normalize path (remove leading ./)
                file_path = file_path.lstrip("./")
                baseline_relevant_files.append(file_path)
                baseline_file_details[file_path] = {
                    "line_number": rf.get("line_number"),
                    "reason": rf.get("reason", "")
                }

        # Stage 1: BM25 Search on Elasticsearch (enhanced with baseline files)
        candidates = self.bm25_search(sha_fail, bug_description, baseline_relevant_files)
        print(f"  Stage 1: BM25 returned {len(candidates)} candidate files")

        # Ensure baseline files are in candidates (critical fix!)
        candidate_paths = set(c['file_path'].lstrip("./") for c in candidates)
        for baseline_file in baseline_relevant_files:
            normalized = baseline_file.lstrip("./")
            if normalized not in candidate_paths:
                # Add baseline file as high-priority candidate
                candidates.insert(0, {
                    "file_path": f"./{normalized}",
                    "file_content": "",
                    "bm25_score": 999.0  # High score to ensure it's considered
                })
                print(f"  [FIX] Added baseline file to candidates: {normalized}")

        if not candidates:
            print(f"  [WARN] No candidates found in index for {sha_fail}")
            return {
                "id": entry_id,
                "sha_fail": sha_fail,
                "fault_localization_data": []
            }

        # Stage 2: Method-level LLM Relevance Feedback
        methods_data = self.parse_and_assess_methods(candidates, bug_title, bug_description)
        relevant_methods = [m for m in methods_data if m['relevant']]
        brain_relevant_files = list(set([m['file_path'] for m in relevant_methods]))

        print(f"  Stage 2: Parsed {len(methods_data)} methods from {len(candidates)} files")
        print(f"           BRaIn: {len(relevant_methods)} methods in {len(brain_relevant_files)} files")
        print(f"           Baseline: {len(baseline_relevant_files)} files")
        if error_types or failed_command:
            print(f"           Context: {len(error_types)} error types, {1 if failed_command else 0} commands")

        # Stage 3: Query Expansion with PageRank
        expanded_query = self.expand_query_with_pagerank(bug_description, relevant_methods)
        print(f"  Stage 3: Query expanded with {len(expanded_query.split()) - len(bug_description.split())} new terms")

        # Stage 4: BM25 Rescoring + LLM Combination (with baseline details)
        ranked_files = self.rescore_and_rank(sha_fail, candidates, expanded_query,
                                            brain_relevant_files, baseline_relevant_files,
                                            baseline_file_details)
        print(f"  Stage 4: Ranked {len(ranked_files)} files")

        # Show top 3
        if ranked_files:
            print(f"\n  Top 3 files:")
            for i, item in enumerate(ranked_files[:3], 1):
                print(f"    {i}. {item['file_path']} (score: {item['score']:.3f})")

        return {
            "id": entry_id,
            "sha_fail": sha_fail,
            "fault_localization_data": ranked_files
        }

    def bm25_search(self, sha_fail: str, query: str, baseline_files: List[str] = None, top_k: int = 50) -> List[Dict]:
        """Stage 1: BM25 search on Elasticsearch (enhanced with baseline file boost)"""
        try:
            # Build query with baseline file boost
            must_clauses = [{"match": {"file_content": query}}]

            # Add file name boost if baseline files provided
            if baseline_files:
                for bf in baseline_files:
                    # Extract just the file name for better matching
                    file_name = bf.split('/')[-1].replace('.py', '')
                    if file_name:
                        must_clauses.append({
                            "match": {
                                "file_path": {
                                    "query": file_name,
                                    "boost": 2.0  # Boost files matching baseline names
                                }
                            }
                        })

            # Search for files in this commit
            search_body = {
                "query": {
                    "bool": {
                        "should": must_clauses,  # Changed from must to should for flexibility
                        "minimum_should_match": 1,
                        "filter": [
                            {"term": {"sha_fail": sha_fail}}
                        ]
                    }
                },
                "size": top_k,
                "_source": ["file_path", "file_content", "sha_fail"]
            }

            response = self.es.search(index=self.index_name, body=search_body)

            candidates = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                candidates.append({
                    "file_path": source['file_path'],
                    "file_content": source['file_content'],
                    "bm25_score": hit['_score']
                })

            return candidates

        except Exception as e:
            print(f"    [WARN] Elasticsearch search failed: {e}")
            return []

    def parse_and_assess_methods(self, candidates: List[Dict], bug_title: str, bug_description: str) -> List[Dict]:
        """Stage 2: Parse files into methods and assess relevance with LLM"""
        methods_data = []
        total_methods_checked = 0
        relevant_count = 0

        for candidate in candidates:
            file_path = candidate['file_path']
            file_content = candidate.get('file_content', '')

            # Parse methods from file
            methods = self.parse_methods_from_file(file_content, file_path)

            # Assess each method with LLM (with limit and timeout)
            for method in methods:
                # Stop if we've checked too many methods
                if total_methods_checked >= self.max_methods_per_instance:
                    print(f"    [LIMIT] Reached max methods ({self.max_methods_per_instance}), stopping assessment")
                    break

                # Early stop if we found enough relevant methods
                if relevant_count >= self.min_relevant_methods and total_methods_checked >= 30:
                    print(f"    [EARLY STOP] Found {relevant_count} relevant methods, stopping early")
                    break

                try:
                    is_relevant = self.assess_method_relevance(
                        method['code'],
                        method['name'],
                        file_path,
                        bug_title,
                        bug_description
                    )
                    if is_relevant:
                        relevant_count += 1
                except TimeoutError:
                    print(f"    [TIMEOUT] LLM call timed out for {file_path}:{method['name']}")
                    is_relevant = False
                except Exception as e:
                    print(f"    [ERROR] LLM call failed for {file_path}:{method['name']}: {e}")
                    is_relevant = False

                methods_data.append({
                    "file_path": file_path,
                    "method_name": method['name'],
                    "code": method['code'],
                    "relevant": is_relevant,
                    "bm25_score": candidate.get('bm25_score', 0.0)
                })

                total_methods_checked += 1

            # Break outer loop if limits reached
            if total_methods_checked >= self.max_methods_per_instance:
                break
            if relevant_count >= self.min_relevant_methods and total_methods_checked >= 30:
                break

        return methods_data

    def parse_methods_from_file(self, content: str, file_path: str) -> List[Dict]:
        """Parse Python file into methods/functions"""
        methods = []

        if not file_path.endswith('.py'):
            # For non-Python files, treat entire file as one "method"
            return [{"name": file_path, "code": content[:1000]}]

        # Simple Python method parsing using regex
        # Pattern: def function_name(...):
        pattern = r'(def\s+(\w+)\s*\([^)]*\)\s*(?:->.*?)?\s*:(?:\s*"""[^"]*""")?\s*\n(?:.*?\n)*?(?=\ndef\s|\nclass\s|$))'

        matches = re.finditer(pattern, content, re.MULTILINE | re.DOTALL)

        for match in matches:
            method_code = match.group(1)
            method_name = match.group(2)

            # Limit code length
            if len(method_code) > 800:
                method_code = method_code[:800] + "..."

            methods.append({
                "name": method_name,
                "code": method_code
            })

        # If no methods found, use entire file (truncated)
        if not methods:
            methods.append({
                "name": file_path,
                "code": content[:1000]
            })

        return methods

    def assess_method_relevance(self, code: str, method_name: str, file_path: str,
                                bug_title: str, bug_description: str) -> bool:
        """Use LLM to assess if method is relevant to bug"""

        system_prompt = """You are analyzing code methods to find bugs.
Determine if this method is likely responsible for the CI failure.
Output only JSON: {"relevant": true} or {"relevant": false}"""

        user_prompt = f"""Bug: {bug_title}
Description: {bug_description[:300]}

File: {file_path}
Method: {method_name}

Code:
```python
{code[:500]}
```

Is this method relevant to the bug?"""

        try:
            # Use timeout to prevent hanging
            with timeout_context(self.llm_timeout):
                response = self.llm.invoke([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ])

                response_text = response.content.lower()
                return 'true' in response_text or '"relevant": true' in response_text

        except TimeoutError:
            # Re-raise timeout to be handled by caller
            raise
        except Exception as e:
            # On other errors, check if file is in error context (conservative)
            return file_path in bug_description

    def expand_query_with_pagerank(self, original_query: str, relevant_methods: List[Dict]) -> str:
        """Stage 3: Expand query using PageRank on term co-occurrence graph"""

        # Extract terms from relevant methods
        all_terms = []
        for method in relevant_methods:
            code = method.get('code', '')
            terms = self.extract_terms(code)
            all_terms.extend(terms)

        if not all_terms:
            return original_query

        # Build term co-occurrence graph
        term_graph = self.build_term_cooccurrence_graph(all_terms, relevant_methods)

        # Apply PageRank
        try:
            pagerank_scores = nx.pagerank(term_graph, alpha=0.85, max_iter=100)

            # Get top terms
            sorted_terms = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
            top_terms = [term for term, score in sorted_terms[:10] if term not in original_query]

            # Expand query
            expanded_query = original_query + " " + " ".join(top_terms[:5])
            return expanded_query

        except Exception as e:
            # Fallback: use most common terms
            term_counts = Counter(all_terms)
            top_terms = [term for term, _ in term_counts.most_common(5) if term not in original_query]
            return original_query + " " + " ".join(top_terms)

    def extract_terms(self, text: str) -> List[str]:
        """Extract meaningful terms from text"""
        # Tokenize
        tokens = re.findall(r'\b[a-z_][a-z0-9_]*\b', text.lower())

        # Filter stopwords and short terms
        terms = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        return terms

    def build_term_cooccurrence_graph(self, terms: List[str], relevant_methods: List[Dict]) -> nx.Graph:
        """Build term co-occurrence graph from relevant methods"""
        graph = nx.Graph()

        # Add nodes
        unique_terms = set(terms)
        graph.add_nodes_from(unique_terms)

        # Add edges based on co-occurrence in same method
        for method in relevant_methods:
            method_terms = self.extract_terms(method.get('code', ''))

            # Connect terms that appear together
            for i, term1 in enumerate(method_terms):
                for term2 in method_terms[i+1:i+6]:  # Window of 5
                    if term1 != term2:
                        if graph.has_edge(term1, term2):
                            graph[term1][term2]['weight'] += 1
                        else:
                            graph.add_edge(term1, term2, weight=1)

        return graph

    def rescore_and_rank(self, sha_fail: str, candidates: List[Dict],
                        expanded_query: str, brain_relevant_files: List[str],
                        baseline_relevant_files: List[str] = None,
                        baseline_file_details: Dict = None) -> List[Dict]:
        """Stage 4: Rescore with expanded query and combine with LLM feedback + baseline"""

        # Re-search with expanded query (no baseline boost in re-search)
        expanded_candidates = self.bm25_search(sha_fail, expanded_query, baseline_files=None, top_k=50)

        # Build score maps
        bm25_scores = {c['file_path']: c.get('bm25_score', 0.0) for c in expanded_candidates}

        # Normalize baseline relevant files paths
        if baseline_relevant_files is None:
            baseline_relevant_files = []
        if baseline_file_details is None:
            baseline_file_details = {}
        baseline_set = set(f.lstrip("./") for f in baseline_relevant_files)

        # Combine scores
        ranked = []
        all_files = set([c['file_path'] for c in candidates])

        for file_path in all_files:
            bm25_score = bm25_scores.get(file_path, 0.0)
            normalized_path = file_path.lstrip("./")

            # Check relevance from both sources
            brain_relevant = normalized_path in [f.lstrip("./") for f in brain_relevant_files]
            baseline_relevant = normalized_path in baseline_set

            # Get baseline details if available
            baseline_details = baseline_file_details.get(normalized_path, {})
            has_line_number = baseline_details.get("line_number") is not None
            has_reason = bool(baseline_details.get("reason"))

            # Enhanced scoring: combine BRaIn + Baseline + Details
            if brain_relevant and baseline_relevant:
                # Both agree - highest confidence
                combined_score = bm25_score * 2.0
                # Extra boost if baseline has specific line number
                if has_line_number:
                    combined_score *= 1.2
                confidence = "high"
            elif brain_relevant or baseline_relevant:
                # One agrees - medium confidence
                combined_score = bm25_score * 1.5
                # Slight boost if baseline has detailed reason
                if has_reason:
                    combined_score *= 1.1
                confidence = "medium"
            else:
                # Neither - low confidence
                combined_score = bm25_score * 0.5
                confidence = "low"

            ranked.append({
                "file_path": file_path,
                "line_number": baseline_details.get("line_number"),
                "rank": 0,  # Will be updated
                "score": combined_score,
                "bm25_score": bm25_score,
                "brain_relevant": brain_relevant,
                "baseline_relevant": baseline_relevant,
                "confidence": confidence,
                "baseline_reason": baseline_details.get("reason", "")
            })

        # Sort by combined score
        ranked.sort(key=lambda x: x['score'], reverse=True)

        # Normalize scores to 0-1 range
        if ranked:
            max_score = max(r['score'] for r in ranked)
            if max_score > 0:
                for r in ranked:
                    r['score'] = r['score'] / max_score

        # Update ranks
        for idx, item in enumerate(ranked):
            item['rank'] = idx + 1

        return ranked


def find_log_details(model: str, tool: str) -> Path:
    """Find log_details.json for given model and tool"""
    possible_paths = [
        f"../results/{model}_{tool}/log_details.json",
        f"../results/{model}/log_details.json",
        f"../../results/{model}_{tool}/log_details.json",
    ]

    for path_str in possible_paths:
        path = Path(path_str)
        if path.exists():
            return path

    raise FileNotFoundError(f"Could not find log_details.json for model={model}, tool={tool}")


def main():
    parser = argparse.ArgumentParser(description="Full BRaIn FL with Elasticsearch")
    parser.add_argument("combined", nargs="?", help="Combined model_tool name (e.g., gpt-5-mini_llm)")
    parser.add_argument("--model", default="gpt-5-mini", help="Model name")
    parser.add_argument("--tool", default="llm", help="FL tool name")
    parser.add_argument("--limit", type=int, default=150, help="Number of samples to process")

    args = parser.parse_args()

    # Parse combined name
    if args.combined:
        if "_" in args.combined:
            parts = args.combined.rsplit("_", 1)
            args.model = parts[0]
            args.tool = parts[1]
        else:
            args.model = args.combined

    print(f"[START] Full BRaIn Fault Localization")
    print(f"   Model: {args.model}")
    print(f"   Tool:  {args.tool}")
    print(f"   Limit: {args.limit}")
    print()

    # Find log_details
    try:
        log_details_path = find_log_details(args.model, args.tool)
        print(f"[FILE] Found log_details: {log_details_path}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    # Check Elasticsearch
    try:
        es = Elasticsearch(["http://localhost:9200"])
        es.info()
        print(f" Connected to Elasticsearch")
    except Exception as e:
        print(f" Elasticsearch not running: {e}")
        print(f"   Run: brew services start elastic/tap/elasticsearch-full")
        return

    # Load data
    print(f"[LOAD] Loading data...")
    with open(log_details_path) as f:
        log_details = json.load(f)

    print(f"   Total entries: {len(log_details)}")
    print(f"   Processing: [0:{args.limit}]")

    # Initialize FL
    print(f"🤖 Initializing LLM: {args.model}")
    fl = FullBRaInFL(args.model)

    # Load existing results if file exists (RESUME capability)
    output_file = f"brain_fl_full_{args.model}_{args.tool}_{args.limit}.json"
    results = []
    processed_ids = set()

    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
            processed_ids = {r.get('id') for r in results}
            print(f"[RESUME] Loaded {len(results)} existing results, will skip already processed")
        except Exception as e:
            print(f"[WARN] Could not load existing results: {e}")
            results = []

    # Process entries with INCREMENTAL SAVING
    PER_INSTANCE_TIMEOUT = 600  # 10 minutes per instance

    for i, entry in enumerate(log_details[:args.limit]):
        entry_id = entry.get('id', i + 1)

        # SKIP if already processed
        if entry_id in processed_ids:
            print(f"\n[{i + 1}] ID={entry_id} - SKIPPED (already processed)")
            continue

        try:
            # Add per-instance timeout
            with timeout_context(PER_INSTANCE_TIMEOUT):
                result = fl.run_fl(entry, i)
                results.append(result)
                processed_ids.add(entry_id)

                # SAVE INCREMENTALLY after each instance
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"  ✓ Saved (total: {len(results)} results)")

        except TimeoutError:
            print(f"[ERROR] BRaIn FL timed out (>{PER_INSTANCE_TIMEOUT}s)")
            # Save empty result to skip this instance on resume
            results.append({
                "id": entry_id,
                "sha_fail": entry.get("sha_fail", ""),
                "fault_localization_data": [],
                "error": "timeout"
            })
            processed_ids.add(entry_id)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            continue
        except Exception as e:
            print(f"  [ERROR] Error processing entry {i+1}: {e}")
            continue

    # Final summary save
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f" ✓ COMPLETE: Saved {len(results)} results to: {output_file}")
    print()

    # Summary
    total_files_ranked = sum(len(r['fault_localization_data']) for r in results)
    avg_files = total_files_ranked / len(results) if results else 0
    print(f" Summary:")
    print(f"   Processed: {len(results)}/{args.limit}")
    print(f"   Total files ranked: {total_files_ranked}")
    print(f"   Average files per entry: {avg_files:.1f}")


if __name__ == "__main__":
    main()
