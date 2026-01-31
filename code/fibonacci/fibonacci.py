"""斐波那契数列实现"""


def fibonacci_recursive(n: int) -> int:
    """递归方式计算第n个斐波那契数"""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci_iterative(n: int) -> int:
    """迭代方式计算第n个斐波那契数（更高效）"""
    if n <= 0:
        return 0
    if n == 1:
        return 1

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fibonacci_sequence(n: int) -> list[int]:
    """生成前n个斐波那契数列"""
    if n <= 0:
        return []
    if n == 1:
        return [0]

    seq = [0, 1]
    for _ in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq


if __name__ == "__main__":
    # 演示
    print("斐波那契数列前20个数:")
    print(fibonacci_sequence(20))

    print("\n第10个斐波那契数:")
    print(f"  递归方式: {fibonacci_recursive(10)}")
    print(f"  迭代方式: {fibonacci_iterative(10)}")
