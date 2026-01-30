# Fibonacci Sequence Calculator

A Python program that calculates Fibonacci numbers using three different methods:
1. **Iterative method** - Efficient O(n) time complexity
2. **Recursive method** - Simple but inefficient O(2^n) time complexity
3. **Memoized method** - Optimized recursive approach with O(n) time complexity

## Features

- Calculate Fibonacci numbers using different algorithms
- Compare performance and results of each method
- Comprehensive test suite
- Clean, well-documented code

## Installation

No special installation required. Just ensure you have Python 3.6 or higher.

## Usage

Run the main program:

```bash
python main.py
```

Run the tests:

```bash
python test_fibonacci.py
```

## Example Output

```
Fibonacci Sequence Calculator
==============================

Iterative method:
F(0) = 0
F(1) = 1
F(5) = 5
F(10) = 55
F(15) = 610
F(20) = 6765

First 15 Fibonacci numbers:
[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
```

## Functions

### `fibonacci_iterative(n)`
Calculates the nth Fibonacci number using an iterative approach.
- **Time Complexity**: O(n)
- **Space Complexity**: O(1)

### `fibonacci_recursive(n)`
Calculates the nth Fibonacci number using a recursive approach.
- **Time Complexity**: O(2^n) (exponential)
- **Space Complexity**: O(n) (call stack)

### `fibonacci_memoized(n, memo=None)`
Calculates the nth Fibonacci number using recursion with memoization.
- **Time Complexity**: O(n)
- **Space Complexity**: O(n)

## Performance Comparison

| Method | Time Complexity | Best For |
|--------|----------------|----------|
| Iterative | O(n) | Large numbers, general use |
| Recursive | O(2^n) | Small numbers, educational |
| Memoized | O(n) | Medium numbers, recursive style with optimization |

## Testing

The test suite verifies:
- Correctness of each method
- Consistency between methods
- Edge cases (n=0, n=1)
- Performance with various input sizes

## License

This project is for educational purposes.