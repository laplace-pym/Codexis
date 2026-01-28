#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简洁版九九乘法表
"""

# 方法1：使用嵌套循环
def multiplication_table_simple():
    """最简单的九九乘法表"""
    print("九九乘法表（简单版）：")
    for i in range(1, 10):
        for j in range(1, i + 1):
            print(f"{j}×{i}={i*j}", end="\t")
        print()

# 方法2：使用列表推导式
def multiplication_table_list_comprehension():
    """使用列表推导式的九九乘法表"""
    print("\n九九乘法表（列表推导式版）：")
    table = [[f"{j}×{i}={i*j}" for j in range(1, i + 1)] for i in range(1, 10)]
    for row in table:
        print("\t".join(row))

# 方法3：使用while循环
def multiplication_table_while():
    """使用while循环的九九乘法表"""
    print("\n九九乘法表（while循环版）：")
    i = 1
    while i <= 9:
        j = 1
        while j <= i:
            print(f"{j}×{i}={i*j}", end="\t")
            j += 1
        print()
        i += 1

# 方法4：使用递归
def multiplication_table_recursive(i=1):
    """使用递归的九九乘法表"""
    if i == 1:
        print("\n九九乘法表（递归版）：")
    
    if i > 9:
        return
    
    for j in range(1, i + 1):
        print(f"{j}×{i}={i*j}", end="\t")
    print()
    multiplication_table_recursive(i + 1)

# 方法5：使用生成器
def multiplication_table_generator():
    """使用生成器的九九乘法表"""
    print("\n九九乘法表（生成器版）：")
    def table_generator():
        for i in range(1, 10):
            row = []
            for j in range(1, i + 1):
                row.append(f"{j}×{i}={i*j}")
            yield row
    
    for row in table_generator():
        print("\t".join(row))

# 主函数
def main():
    """主函数"""
    print("多种方法实现九九乘法表")
    print("=" * 50)
    
    # 调用各种方法
    multiplication_table_simple()
    multiplication_table_list_comprehension()
    multiplication_table_while()
    multiplication_table_recursive()
    multiplication_table_generator()
    
    # 额外：打印所有乘法结果
    print("\n所有乘法结果（1-9）：")
    print("-" * 30)
    for i in range(1, 10):
        for j in range(1, 10):
            print(f"{i}×{j}={i*j:2d}", end="  ")
        print()

if __name__ == "__main__":
    main()