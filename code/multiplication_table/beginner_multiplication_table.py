#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初学者版九九乘法表
适合Python初学者学习
"""

# 版本1：最基本的九九乘法表
print("版本1：最基本的九九乘法表")
for i in range(1, 10):  # i从1到9
    for j in range(1, i + 1):  # j从1到i
        print(f"{j}×{i}={i*j}", end="\t")
    print()  # 换行

print("\n" + "="*50 + "\n")

# 版本2：带格式化的九九乘法表
print("版本2：带格式化的九九乘法表")
for i in range(1, 10):
    for j in range(1, i + 1):
        # 使用format格式化，保持对齐
        result = i * j
        print(f"{j}×{i}={result:2d}", end="   ")
    print()

print("\n" + "="*50 + "\n")

# 版本3：使用while循环
print("版本3：使用while循环")
i = 1
while i <= 9:
    j = 1
    while j <= i:
        print(f"{j}×{i}={i*j}", end="\t")
        j += 1
    print()
    i += 1

print("\n" + "="*50 + "\n")

# 版本4：打印完整的9×9表格
print("版本4：完整的9×9乘法表格")
print("   |", end="")
for i in range(1, 10):
    print(f"  {i}  ", end="")
print("\n---+" + "-"*45)

for i in range(1, 10):
    print(f" {i} |", end="")
    for j in range(1, 10):
        print(f" {i*j:2d} ", end="")
    print()

print("\n" + "="*50 + "\n")

# 版本5：函数封装版
def print_multiplication_table(style="simple"):
    """打印九九乘法表
    style: 'simple' - 简单版, 'formatted' - 格式化版, 'full' - 完整表格
    """
    if style == "simple":
        print("函数版：简单九九乘法表")
        for i in range(1, 10):
            for j in range(1, i + 1):
                print(f"{j}×{i}={i*j}", end="\t")
            print()
    
    elif style == "formatted":
        print("函数版：格式化九九乘法表")
        for i in range(1, 10):
            line = ""
            for j in range(1, i + 1):
                line += f"{j}×{i}={i*j:2d}  "
            print(line)
    
    elif style == "full":
        print("函数版：完整乘法表格")
        # 打印表头
        print("   |", end="")
        for i in range(1, 10):
            print(f"  {i:2d}  ", end="")
        print("\n---+" + "-"*50)
        
        # 打印表格内容
        for i in range(1, 10):
            print(f" {i:2d}|", end="")
            for j in range(1, 10):
                print(f" {i*j:3d} ", end="")
            print()

# 测试函数
print_multiplication_table("simple")
print()
print_multiplication_table("formatted")
print()
print_multiplication_table("full")

print("\n" + "="*50 + "\n")

# 练习：让用户输入数字查询乘法
print("练习：查询特定乘法")
print("输入两个1-9之间的数字，程序会显示乘法结果")

try:
    num1 = int(input("请输入第一个数字（1-9）："))
    num2 = int(input("请输入第二个数字（1-9）："))
    
    if 1 <= num1 <= 9 and 1 <= num2 <= 9:
        print(f"{num1} × {num2} = {num1 * num2}")
        
        # 显示相关的乘法行
        print(f"\n相关的乘法行：")
        for i in range(1, 10):
            if i == num1 or i == num2:
                for j in range(1, i + 1):
                    print(f"{j}×{i}={i*j}", end="\t")
                print()
    else:
        print("请输入1-9之间的数字！")
except ValueError:
    print("请输入有效的数字！")

print("\n程序结束！")