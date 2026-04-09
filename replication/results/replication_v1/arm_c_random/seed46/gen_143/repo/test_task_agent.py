"""Test suite for task_agent improvements."""

from task_agent import TaskAgent, _extract_jsons, _extract_json_fallback


def test_extract_jsons():
    """Test the primary JSON extraction function."""
    # Test 1: Standard <json> tags
    result = _extract_jsons('<json>{"response": 8}</json>')
    assert result == [{"response": 8}], f"Test 1 failed: {result}"
    print("Test 1 passed: Standard <json> tags")

    # Test 2: Markdown code blocks
    result = _extract_jsons('```json\n{"response": 9}\n```')
    assert result == [{"response": 9}], f"Test 2 failed: {result}"
    print("Test 2 passed: Markdown code blocks")

    # Test 3: Plain JSON (should not be found by primary)
    result = _extract_jsons('{"response": 7}')
    assert result is None, f"Test 3 failed: {result}"
    print("Test 3 passed: Plain JSON not found by primary (expected)")

    print("All _extract_jsons tests passed!")


def test_extract_json_fallback():
    """Test the fallback JSON extraction function."""
    # Test 1: Plain JSON
    result = _extract_json_fallback('{"response": 7}')
    assert result == [{"response": 7}], f"Test 1 failed: {result}"
    print("Test 1 passed: Plain JSON")

    # Test 2: Single-quoted JSON
    result = _extract_json_fallback("{'response': 6}")
    assert result == [{"response": 6}], f"Test 2 failed: {result}"
    print("Test 2 passed: Single-quoted JSON")

    # Test 3: Plain number
    result = _extract_json_fallback('The answer is 5')
    assert result == [{"response": 5}], f"Test 3 failed: {result}"
    print("Test 3 passed: Plain number extraction")

    # Test 4: Response pattern
    result = _extract_json_fallback('response: 6')
    assert result == [{"response": 6}], f"Test 4 failed: {result}"
    print("Test 4 passed: Response pattern")

    print("All _extract_json_fallback tests passed!")


def test_normalize_prediction():
    """Test the prediction normalization."""
    agent = TaskAgent()

    # Test clamping to max
    result = agent._normalize_prediction(15, "", 10)
    assert result == "10", f"Clamping max failed: {result}"
    print("Test 1 passed: Clamping to max")

    # Test clamping to min
    result = agent._normalize_prediction(-3, "", 10)
    assert result == "0", f"Clamping min failed: {result}"
    print("Test 2 passed: Clamping to min")

    # Test integer
    result = agent._normalize_prediction(7, "", 10)
    assert result == "7", f"Integer failed: {result}"
    print("Test 3 passed: Integer normalization")

    # Test float
    result = agent._normalize_prediction(7.5, "", 10)
    assert result == "7.5", f"Float failed: {result}"
    print("Test 4 passed: Float normalization")

    # Test string number
    result = agent._normalize_prediction("8", "", 10)
    assert result == "8", f"String number failed: {result}"
    print("Test 5 passed: String number normalization")

    print("All _normalize_prediction tests passed!")


def test_extract_max_score():
    """Test the max score extraction from grading guidelines."""
    agent = TaskAgent()

    # Test various patterns
    assert agent._extract_max_score("out of 25 points") == 25, "Pattern 1 failed"
    print("Test 1 passed: 'out of X points'")

    assert agent._extract_max_score("maximum 15 points") == 15, "Pattern 2 failed"
    print("Test 2 passed: 'maximum X points'")

    assert agent._extract_max_score("total 20 points") == 20, "Pattern 3 failed"
    print("Test 3 passed: 'total X points'")

    assert agent._extract_max_score("worth 30 points") == 30, "Pattern 4 failed"
    print("Test 4 passed: 'worth X points'")

    # Test default
    assert agent._extract_max_score("") == 10, "Default failed"
    print("Test 5 passed: Default value")

    print("All _extract_max_score tests passed!")


def test_is_valid_number():
    """Test the number validation."""
    agent = TaskAgent()

    assert agent._is_valid_number(5) is True, "Integer failed"
    assert agent._is_valid_number(5.5) is True, "Float failed"
    assert agent._is_valid_number("5") is True, "String int failed"
    assert agent._is_valid_number("5.5") is True, "String float failed"
    assert agent._is_valid_number("abc") is False, "Invalid string failed"
    assert agent._is_valid_number(None) is False, "None failed"

    print("All _is_valid_number tests passed!")


if __name__ == "__main__":
    print("=== Running Task Agent Tests ===\n")

    print("Testing _extract_jsons:")
    test_extract_jsons()
    print()

    print("Testing _extract_json_fallback:")
    test_extract_json_fallback()
    print()

    print("Testing _normalize_prediction:")
    test_normalize_prediction()
    print()

    print("Testing _extract_max_score:")
    test_extract_max_score()
    print()

    print("Testing _is_valid_number:")
    test_is_valid_number()
    print()

    print("=== All tests passed! ===")
