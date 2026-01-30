#!/usr/bin/env python3
"""
Test cases for Fibonacci functions
"""

import sys
sys.path.insert(0, '.')

from main import fibonacci_iterative, fibonacci_recursive, fibonacci_memoized

def test_fibonacci_iterative():
    """Test iterative Fibonacci function."""
    test_cases = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (6, 8),
        (7, 13),
        (8, 21),
        (9, 34),
        (10, 55),
        (15, 610),
        (20, 6765),
    ]
    
    for n, expected in test_cases:
        result = fibonacci_iterative(n)
        assert result == expected, f"fibonacci_iterative({n}) = {result}, expected {expected}"
        print(f"✓ fibonacci_iterative({n}) = {result}")

def test_fibonacci_recursive():
    """Test recursive Fibonacci function (small numbers only)."""
    test_cases = [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (6, 8),
        (7, 13),
        (8, 21),
    ]
    
    for n, expected in test_cases:
        result = fibonacci_recursive(n)
        assert result == expected, f"fibonacci_recursive({n}) = {result}, expected {expected}"
        print(f"✓ fibonacci_recursive({n}) = {result}")

def test_fibonacci_memoized():
    """Test memoized Fibonacci function."""
    test_cases = [
        (0, 0),
        (1, 1),
        (5, 5),
        (10, 55),
        (15, 610),
        (20, 6765),
        (25, 75025),
        (30, 832040),
    ]
    
    for n, expected in test_cases:
        result = fibonacci_memoized(n)
        assert result == expected, f"fibonacci_memoized({n}) = {result}, expected {expected}"
        print(f"✓ fibonacci_memoized({n}) = {result}")

def test_consistency():
    """Test that all methods give the same results."""
    test_numbers = list(range(0, 15))
    
    for n in test_numbers:
        iterative = fibonacci_iterative(n)
        memoized = fibonacci_memoized(n)
        
        # Only test recursive for small numbers
        if n <= 10:
            recursive = fibonacci_recursive(n)
            assert recursive == iterative, f"Mismatch at n={n}: recursive={recursive}, iterative={iterative}"
        
        assert memoized == iterative, f"Mismatch at n={n}: memoized={memoized}, iterative={iterative}"
    
    print("✓ All methods are consistent")

def run_all_tests():
    """Run all test functions."""
    print("Running Fibonacci function tests...")
    print("=" * 40)
    
    test_fibonacci_iterative()
    print()
    
    test_fibonacci_recursive()
    print()
    
    test_fibonacci_memoized()
    print()
    
    test_consistency()
    print()
    
    print("=" * 40)
    print("All tests passed! ✅")

if __name__ == "__main__":
    run_all_tests()