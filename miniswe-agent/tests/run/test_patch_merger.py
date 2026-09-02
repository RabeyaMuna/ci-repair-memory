"""
Tests for patch_merger utility - merging duplicate file patches.
"""

import pytest
from pathlib import Path
from minisweagent.run.benchmarks.utils.patch_merger import (
    parse_unified_diff,
    group_patches_by_file,
    merge_duplicate_patches,
    detect_duplicate_patches,
    validate_patch,
)


def test_parse_unified_diff_single_file():
    """Test parsing a simple single-file diff."""
    diff = """diff --git a/file.py b/file.py
index 123..456 789
--- a/file.py
+++ b/file.py
@@ -1,3 +1,3 @@
 line1
-line2
+line2_modified
 line3
"""
    patches = parse_unified_diff(diff)

    assert len(patches) == 1
    assert patches[0]['file'] == 'file.py'
    assert '@@' in patches[0]['hunks']


def test_parse_unified_diff_multiple_files():
    """Test parsing diff with multiple different files."""
    diff = """diff --git a/file1.py b/file1.py
index 111..222 333
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old1
+new1

diff --git a/file2.py b/file2.py
index 444..555 666
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old2
+new2
"""
    patches = parse_unified_diff(diff)

    assert len(patches) == 2
    assert patches[0]['file'] == 'file1.py'
    assert patches[1]['file'] == 'file2.py'


def test_parse_unified_diff_duplicate_files():
    """Test parsing diff with multiple patches for the same file."""
    diff = """diff --git a/file.py b/file.py
index 111..222 333
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new

diff --git a/file.py b/file.py
index 333..444 555
--- a/file.py
+++ b/file.py
@@ -5 +5 @@
-old2
+new2
"""
    patches = parse_unified_diff(diff)

    assert len(patches) == 2
    assert patches[0]['file'] == 'file.py'
    assert patches[1]['file'] == 'file.py'


def test_group_patches_by_file():
    """Test grouping patches by file path."""
    patches = [
        {'file': 'a.py', 'full_patch': 'patch1'},
        {'file': 'b.py', 'full_patch': 'patch2'},
        {'file': 'a.py', 'full_patch': 'patch3'},
    ]

    grouped = group_patches_by_file(patches)

    assert len(grouped) == 2
    assert len(grouped['a.py']) == 2
    assert len(grouped['b.py']) == 1


def test_detect_duplicate_patches_no_duplicates():
    """Test detecting duplicates when there are none."""
    diff = """diff --git a/file1.py b/file1.py
index 111..222 333
--- a/file1.py
+++ b/file1.py
@@ -1 +1 @@
-old
+new

diff --git a/file2.py b/file2.py
index 333..444 555
--- a/file2.py
+++ b/file2.py
@@ -1 +1 @@
-old2
+new2
"""
    duplicates = detect_duplicate_patches(diff)

    assert duplicates['file1.py'] == 1
    assert duplicates['file2.py'] == 1


def test_detect_duplicate_patches_with_duplicates():
    """Test detecting duplicates when they exist."""
    diff = """diff --git a/file.py b/file.py
index 111..222 333
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new

diff --git a/file.py b/file.py
index 333..444 555
--- a/file.py
+++ b/file.py
@@ -5 +5 @@
-old2
+new2

diff --git a/other.py b/other.py
index 666..777 888
--- a/other.py
+++ b/other.py
@@ -1 +1 @@
-x
+y
"""
    duplicates = detect_duplicate_patches(diff)

    assert duplicates['file.py'] == 2  # Duplicate!
    assert duplicates['other.py'] == 1


def test_validate_patch_clean():
    """Test validating a clean patch (no duplicates)."""
    diff = """diff --git a/file.py b/file.py
index 111..222 333
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
    is_valid, message = validate_patch(diff)

    assert is_valid
    assert message == "OK"


def test_validate_patch_with_duplicates():
    """Test validating a patch with duplicates (should fail)."""
    diff = """diff --git a/file.py b/file.py
index 111..222 333
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new

diff --git a/file.py b/file.py
index 333..444 555
--- a/file.py
+++ b/file.py
@@ -5 +5 @@
-old2
+new2
"""
    is_valid, message = validate_patch(diff)

    assert not is_valid
    assert 'Duplicate patches' in message
    assert 'file.py' in message


def test_validate_patch_empty():
    """Test validating an empty patch."""
    is_valid, message = validate_patch("")

    assert not is_valid
    assert message == "Empty patch"


def test_merge_duplicate_patches_no_duplicates():
    """Test merging when there are no duplicates (should return original)."""
    diff = """diff --git a/file.py b/file.py
index 111..222 333
--- a/file.py
+++ b/file.py
@@ -1 +1 @@
-old
+new
"""
    merged = merge_duplicate_patches(diff)

    # Should be essentially unchanged (maybe minor formatting)
    assert 'diff --git a/file.py b/file.py' in merged
    assert '-old' in merged
    assert '+new' in merged
