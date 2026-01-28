#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
九九乘法表生成器
可以很规则地输出九九乘法表
"""

def print_multiplication_table_simple():
    """简单格式的九九乘法表"""
    print("简单格式九九乘法表:")
    print("=" * 40)
    for i in range(1, 10):
        for j in range(1, i + 1):
            print(f"{j}×{i}={i*j:2d}", end="  ")
        print()
    print()

def print_multiplication_table_aligned():
    """对齐格式的九九乘法表"""
    print("对齐格式九九乘法表:")
    print("=" * 40)
    for i in range(1, 10):
        for j in range(1, 10):
            if j <= i:
                print(f"{j}×{i}={i*j:2d}", end="  ")
            else:
                print(" " * 8, end="  ")
        print()
    print()

def print_multiplication_table_full():
    """完整格式的九九乘法表（9×9）"""
    print("完整格式九九乘法表 (9×9):")
    print("=" * 50)
    print("   ", end="")
    for i in range(1, 10):
        print(f"{i:8d}", end="")
    print()
    print("   " + "-" * 72)
    
    for i in range(1, 10):
        print(f"{i} |", end="")
        for j in range(1, 10):
            print(f"{i*j:8d}", end="")
        print()
    print()

def print_multiplication_table_beautiful():
    """美观格式的九九乘法表"""
    print("美观格式九九乘法表:")
    print("=" * 60)
    print("┌" + "─" * 58 + "┐")
    
    for i in range(1, 10):
        print("│", end="")
        for j in range(1, 10):
            if j <= i:
                result = i * j
                if result < 10:
                    print(f" {j}×{i}= {result} ", end="")
                else:
                    print(f" {j}×{i}={result} ", end="")
            else:
                print("       ", end="")
        print("│")
    
    print("└" + "─" * 58 + "┘")
    print()

def print_multiplication_table_compact():
    """紧凑格式的九九乘法表"""
    print("紧凑格式九九乘法表:")
    print("=" * 40)
    for i in range(1, 10):
        line = ""
        for j in range(1, i + 1):
            line += f"{j}×{i}={i*j} "
        print(line.center(40))
    print()

def print_multiplication_table_vertical():
    """垂直格式的九九乘法表"""
    print("垂直格式九九乘法表:")
    print("=" * 40)
    for j in range(1, 10):
        for i in range(j, 10):
            print(f"{j}×{i}={i*j:2d}", end="  ")
        print()
    print()

def main():
    """主函数"""
    print("九九乘法表生成器")
    print("=" * 60)
    
    # 显示所有格式的乘法表
    print_multiplication_table_simple()
    print_multiplication_table_aligned()
    print_multiplication_table_compact()
    print_multiplication_table_vertical()
    print_multiplication_table_full()
    print_multiplication_table_beautiful()
    
    print("所有格式的九九乘法表已显示完毕！")

if __name__ == "__main__":
    main()