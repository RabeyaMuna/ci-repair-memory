"""
LLM-Based Patch Verification

Provides LLM verification as:
1. Fallback when environment setup fails
2. Pre-check before actual validation
3. Standalone verification when no validation command available

Helps when environment is unable to be set up for validation commands.
"""

import json
import logging
import re
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _call_llm(llm: Any, prompt: str) -> str:
    """Universal LLM caller."""
    if llm is None:
        return ""
    try:
        try:
            from langchain_core.messages import HumanMessage
            result = llm.invoke([HumanMessage(content=prompt)])
            return (getattr(result, "content", None) or "").strip()
        except (ImportError, AttributeError):
            pass
        result = llm.invoke(prompt)
        if hasattr(result, "content"):
            return (result.content or "").strip()
        return str(result).strip()
    except Exception:
        pass
    try:
        return str(llm(prompt)).strip()
    except Exception:
        return ""


def llm_verify_patch(
    problem: Dict[str, Any],
    patch: str,
    llm: Any
) -> Dict[str, Any]:
    """
    LLM verifies if patch correctly solves the problem.

    Args:
        problem: Problem details (description, root_cause, repair_plan)
        patch: Generated patch (git diff format)
        llm: LLM instance

    Returns:
        {
            "correct": bool,
            "confidence": "high" | "medium" | "low",
            "reason": str,
            "issues": [str] or None
        }
    """

    # Build verification prompt
    prompt = f"""Verify if this patch correctly solves the problem.

**PROBLEM DESCRIPTION:**
{problem.get('description', 'N/A')}

**ROOT CAUSE:**
{problem.get('root_cause', 'N/A')}

**REPAIR PLAN:**
{problem.get('repair_plan', {}).get('approach', 'N/A')}

**GENERATED PATCH:**
```diff
{patch[:2000]}  {f"... (truncated, total {len(patch)} chars)" if len(patch) > 2000 else ""}
```

**YOUR TASK:**

Analyze if the patch CORRECTLY addresses the problem:

1. **Does it fix the root cause?**
   - Check if changes address the root cause
   - Not just symptoms

2. **Does it follow the repair plan?**
   - Check if approach matches plan
   - Are the right files modified?

3. **Are the changes correct?**
   - No obvious errors (syntax, logic)
   - Changes make sense
   - No unrelated changes

4. **Completeness:**
   - All necessary changes included?
   - Nothing missing?

**OUTPUT FORMAT:**

Return JSON:
{{
  "correct": true/false,
  "confidence": "high" | "medium" | "low",
  "reason": "Why correct or incorrect",
  "issues": ["issue1", "issue2"] or null
}}

**RULES:**
- If patch FIXES root cause → correct: true
- If patch is INCOMPLETE or WRONG → correct: false
- confidence: "high" if very sure, "medium" if somewhat sure, "low" if unsure
- issues: List specific problems if incorrect

Return ONLY the JSON, no markdown fences.
"""

    try:
        response = _call_llm(llm, prompt)

        # Parse JSON
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            result = json.loads(match.group())

            logger.info(
                f"[LLM Verification] "
                f"correct={result.get('correct')}, "
                f"confidence={result.get('confidence')}, "
                f"reason={result.get('reason', '')[:100]}"
            )

            return result
        else:
            logger.warning("[LLM Verification] Failed to parse JSON")
            return {
                "correct": False,
                "confidence": "low",
                "reason": "LLM verification failed to parse",
                "issues": ["Parse error"]
            }

    except Exception as e:
        logger.warning(f"[LLM Verification] Error: {e}")
        return {
            "correct": False,
            "confidence": "low",
            "reason": f"LLM verification error: {e}",
            "issues": ["Exception"]
        }


def verify_patch_hybrid(
    problem: Dict[str, Any],
    patch: str,
    testbed_path: str,
    llm: Any,
    installation_cmd: Optional[str] = None,
    install_cache: Optional[set] = None
) -> Dict[str, Any]:
    """
    Hybrid verification: Try actual validation, fallback to LLM.

    This is the RECOMMENDED approach when environment setup might fail.

    Flow:
    1. Try actual validation command
    2. If fails to run (env issues) → Use LLM verification
    3. Return result with method used

    Args:
        problem: Problem details
        patch: Generated patch
        testbed_path: Path to repository
        llm: LLM instance
        installation_cmd: Optional installation command
        install_cache: Cache to track installations

    Returns:
        {
            "verified": bool,
            "method": "actual" | "llm" | "none",
            "confidence": "high" | "medium" | "low",
            "reason": str
        }
    """

    validation_cmd = problem.get("validation_cmd")

    # ── Try actual validation first ──
    if validation_cmd:
        logger.info("[Hybrid Verification] Attempting actual validation...")

        # Run installation if needed
        if installation_cmd and install_cache is not None:
            if installation_cmd not in install_cache:
                logger.info(f"[Installation] Running: {installation_cmd}")
                try:
                    subprocess.run(
                        installation_cmd,
                        shell=True,
                        cwd=testbed_path,
                        timeout=300,
                        check=False,
                        capture_output=True
                    )
                    install_cache.add(installation_cmd)
                except Exception as e:
                    logger.warning(f"[Installation] Failed: {e}")

        # Try validation
        try:
            result = subprocess.run(
                validation_cmd,
                shell=True,
                cwd=testbed_path,
                timeout=120,
                capture_output=True,
                check=False
            )

            if result.returncode == 0:
                logger.info("[Actual Validation] PASSED")
                return {
                    "verified": True,
                    "method": "actual",
                    "confidence": "high",
                    "reason": "Validation command passed"
                }
            else:
                logger.warning("[Actual Validation] FAILED")
                stderr = result.stderr.decode('utf-8', errors='ignore')[:500]
                return {
                    "verified": False,
                    "method": "actual",
                    "confidence": "high",
                    "reason": f"Validation failed: {stderr}"
                }

        except subprocess.TimeoutExpired:
            logger.warning("[Actual Validation] TIMEOUT")
            # Fallback to LLM
            pass

        except Exception as e:
            logger.warning(f"[Actual Validation] Error: {e}")
            # Fallback to LLM
            pass

    # ── Fallback: LLM verification ──
    logger.info("[Hybrid Verification] Using LLM verification (fallback)")

    llm_result = llm_verify_patch(problem, patch, llm)

    return {
        "verified": llm_result.get("correct", False),
        "method": "llm",
        "confidence": llm_result.get("confidence", "medium"),
        "reason": llm_result.get("reason", "LLM verification"),
        "llm_issues": llm_result.get("issues")
    }


def verify_patch_with_precheck(
    problem: Dict[str, Any],
    patch: str,
    testbed_path: str,
    llm: Any,
    installation_cmd: Optional[str] = None,
    install_cache: Optional[set] = None
) -> Dict[str, Any]:
    """
    Two-stage verification: LLM pre-check + actual validation.

    This is the MOST ROBUST approach:
    1. LLM pre-check (fast, catches obvious errors)
    2. Actual validation (if LLM approves and env works)

    Flow:
    1. LLM reviews patch
    2. If LLM rejects → Return rejected (agent should retry)
    3. If LLM approves → Try actual validation
    4. If actual validation unavailable → Trust LLM

    Args:
        problem: Problem details
        patch: Generated patch
        testbed_path: Path to repository
        llm: LLM instance
        installation_cmd: Optional installation command
        install_cache: Cache to track installations

    Returns:
        {
            "verified": bool,
            "method": "double" | "llm_only" | "llm_reject" | ...,
            "confidence": "high" | "medium" | "low",
            "reason": str,
            "should_retry": bool  # If true, agent should try again
        }
    """

    # ── Stage 1: LLM Pre-Check ──
    logger.info("[Stage 1] LLM pre-check...")

    llm_result = llm_verify_patch(problem, patch, llm)

    if not llm_result.get("correct", False):
        logger.warning(f"[FAIL] [LLM] REJECTED: {llm_result.get('reason')}")

        return {
            "verified": False,
            "method": "llm_reject",
            "confidence": llm_result.get("confidence", "medium"),
            "reason": llm_result.get("reason", "LLM rejected patch"),
            "issues": llm_result.get("issues"),
            "should_retry": True  # Agent should try again
        }

    logger.info(f"[OK] [LLM] APPROVED: {llm_result.get('reason', '')[:100]}")

    # ── Stage 2: Actual Validation ──
    validation_cmd = problem.get("validation_cmd")

    if not validation_cmd:
        logger.info("[Stage 2] No validation command, trusting LLM")
        return {
            "verified": True,
            "method": "llm_only",
            "confidence": llm_result.get("confidence", "medium"),
            "reason": "LLM approved, no validation command",
            "should_retry": False
        }

    logger.info("[Stage 2] Attempting actual validation...")

    # Run installation if needed
    if installation_cmd and install_cache is not None:
        if installation_cmd not in install_cache:
            try:
                subprocess.run(
                    installation_cmd,
                    shell=True,
                    cwd=testbed_path,
                    timeout=300,
                    check=False,
                    capture_output=True
                )
                install_cache.add(installation_cmd)
            except Exception as e:
                logger.warning(f"[Installation] Failed: {e}")

    # Try validation
    try:
        result = subprocess.run(
            validation_cmd,
            shell=True,
            cwd=testbed_path,
            timeout=120,
            capture_output=True,
            check=False
        )

        if result.returncode == 0:
            logger.info("[DOUBLE VERIFIED] LLM approved + Validation passed")
            return {
                "verified": True,
                "method": "double",
                "confidence": "high",
                "reason": "Both LLM and validation passed",
                "should_retry": False
            }
        else:
            logger.warning("[CONFLICT] LLM approved but validation FAILED")
            stderr = result.stderr.decode('utf-8', errors='ignore')[:500]

            # Decision: Trust actual validation over LLM
            return {
                "verified": False,
                "method": "llm_approved_validation_failed",
                "confidence": "medium",
                "reason": f"LLM approved but validation failed: {stderr}",
                "llm_reason": llm_result.get("reason"),
                "should_retry": True
            }

    except Exception as e:
        logger.warning(f"[Validation] Failed to run: {e}")
        logger.info("Using LLM approval (validation unavailable)")

        return {
            "verified": True,
            "method": "llm_only",
            "confidence": llm_result.get("confidence", "medium"),
            "reason": f"LLM approved, validation unavailable: {e}",
            "should_retry": False
        }
