from utilities.snippet_extractor import clamp_line_range, extract_snippet_from_line_range


def test_clamps_expanded_range_to_last_file_line():
    content = "\n".join(f"line {number}" for number in range(1, 83))

    assert clamp_line_range([1, 87], 82) == (1, 82)
    snippet = extract_snippet_from_line_range(content, [1, 87])
    assert snippet.splitlines()[0] == "line 1"
    assert snippet.splitlines()[-1] == "line 82"
    assert len(snippet.splitlines()) == 82


def test_clamps_both_ends_and_normalizes_reversed_range():
    assert clamp_line_range([-5, 100], 82) == (1, 82)
    assert clamp_line_range([87, 80], 82) == (80, 82)


def test_rejects_non_integer_ranges():
    assert clamp_line_range(["1", 82], 82) is None
    assert clamp_line_range([True, 82], 82) is None
