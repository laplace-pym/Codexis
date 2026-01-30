#!/usr/bin/env python3
"""
Fibonacci Sequence Calculator
This program calculates Fibonacci numbers using different methods.
"""

def fibonacci_iterative(n):
    """Calculate Fibonacci number using iterative method."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def fibonacci_recursive(n):
    """Calculate Fibonacci number using recursive method."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def fibonacci_memoized(n, memo=None):
    """Calculate Fibonacci number using memoization."""
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    
    if n <= 0:
        result = 0
    elif n == 1:
        result = 1
    else:
        result = fibonacci_memoized(n - 1, memo) + fibonacci_memoized(n - 2, memo)
    
    memo[n] = result
    return result

def main():
    """Main function to demonstrate Fibonacci calculations."""
    print("Fibonacci Sequence Calculator")
    print("=" * 30)
    
    # Test different methods
    test_numbers = [0, 1, 5, 10, 15, 20]
    
    print("\nIterative method:")
    for n in test_numbers:
        result = fibonacci_iterative(n)
        print(f"F({n}) = {result}")
    
    print("\nRecursive method (small numbers only):")
    for n in test_numbers[:4]:  # Only test small numbers due to recursion depth
        result = fibonacci_recursive(n)
        print(f"F({n}) = {result}")
    
    print("\nMemoized method:")
    for n in test_numbers:
        result = fibonacci_memoized(n)
        print(f"F({n}) = {result}")
    
    # Generate Fibonacci sequence
    print("\nFirst 15 Fibonacci numbers:")
    sequence = [fibonacci_iterative(i) for i in range(15)]
    print(sequence)

if __name__ == "__main__":
    main()