"""
Tests for text_utils module.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.text_utils import (
    truncate_text,
    sanitize_string,
    count_tokens_approx,
    format_code_block,
    extract_all_urls,
    normalize_whitespace,
    wrap_text,
    safe_get,
    is_valid_json_string,
    remove_comments,
    calculate_similarity,
    format_list,
    parse_key_value_pairs,
)


def test_truncate_text():
    assert truncate_text("hello world", 20) == "hello world"
    assert truncate_text("hello world this is a long text", 15) == "hello world ..."
    assert truncate_text("", 10) == ""


def test_sanitize_string():
    assert sanitize_string("hello\x00world") == "helloworld"
    assert sanitize_string("hello   world") == "hello world"


def test_count_tokens_approx():
    assert count_tokens_approx("") == 0
    assert count_tokens_approx("abcd") == 1
    assert count_tokens_approx("a" * 100) == 25


def test_format_code_block():
    assert format_code_block("print('hello')", "python") == "```python\nprint('hello')\n```"


def test_extract_all_urls():
    text = "Visit https://example.com and www.test.org"
    urls = extract_all_urls(text)
    assert "https://example.com" in urls
    assert "www.test.org" in urls


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"
    assert normalize_whitespace("hello\n\nworld") == "hello world"


def test_wrap_text():
    result = wrap_text("hello world this is a test", width=10)
    assert "\n" in result


def test_safe_get():
    d = {"key": "value"}
    assert safe_get(d, "key") == "value"
    assert safe_get(d, "missing", "default") == "default"
    assert safe_get(None, "key", "default") == "default"


def test_is_valid_json_string():
    assert is_valid_json_string('{"key": "value"}') is True
    assert is_valid_json_string('[1, 2, 3]') is True
    assert is_valid_json_string('not json') is False


def test_remove_comments():
    code = "x = 1  # this is a comment"
    assert "#" not in remove_comments(code, "python")


def test_calculate_similarity():
    assert calculate_similarity("hello", "hello") == 1.0
    assert calculate_similarity("", "hello") == 0.0
    assert 0 < calculate_similarity("hello", "world") < 1


def test_format_list():
    items = ["a", "b", "c"]
    result = format_list(items)
    assert result == "- a\n- b\n- c"


def test_parse_key_value_pairs():
    text = "name=John\nage=30"
    result = parse_key_value_pairs(text)
    assert result == {"name": "John", "age": "30"}


if __name__ == "__main__":
    test_truncate_text()
    test_sanitize_string()
    test_count_tokens_approx()
    test_format_code_block()
    test_extract_all_urls()
    test_normalize_whitespace()
    test_wrap_text()
    test_safe_get()
    test_is_valid_json_string()
    test_remove_comments()
    test_calculate_similarity()
    test_format_list()
    test_parse_key_value_pairs()
    print("All tests passed!")
