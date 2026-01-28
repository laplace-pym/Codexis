#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
九九乘法表
"""

def print_multiplication_table():
    """打印九九乘法表"""
    print("九九乘法表：")
    print("=" * 50)
    
    # 使用嵌套循环打印乘法表
    for i in range(1, 10):
        for j in range(1, i + 1):
            # 格式化输出，保持对齐
            print(f"{j} × {i} = {i*j:2d}", end="   ")
        print()  # 换行
    
    print("=" * 50)

def print_multiplication_table_formatted():
    """打印格式化的九九乘法表（更美观）"""
    print("\n格式化的九九乘法表：")
    print("=" * 60)
    
    for i in range(1, 10):
        line = ""
        for j in range(1, i + 1):
            # 更紧凑的格式
            line += f"{j}×{i}={i*j:2d}  "
        print(line.center(60))
    
    print("=" * 60)

def print_multiplication_table_reverse():
    """打印倒序的九九乘法表"""
    print("\n倒序九九乘法表：")
    print("=" * 50)
    
    for i in range(9, 0, -1):
        for j in range(1, i + 1):
            print(f"{j} × {i} = {i*j:2d}", end="   ")
        print()
    
    print("=" * 50)

def print_multiplication_table_full():
    """打印完整的九九乘法表（9×9矩阵）"""
    print("\n完整的九九乘法表（9×9矩阵）：")
    print("=" * 90)
    
    # 打印表头
    header = "   |"
    for i in range(1, 10):
        header += f"   {i:2d}   "
    print(header)
    print("-" * 90)
    
    # 打印表格内容
    for i in range(1, 10):
        row = f"{i:2d} |"
        for j in range(1, 10):
            row += f"  {i*j:3d}  "
        print(row)
    
    print("=" * 90)

def main():
    """主函数"""
    print("九九乘法表程序")
    print("=" * 50)
    
    # 打印不同版本的乘法表
    print_multiplication_table()
    print_multiplication_table_formatted()
    print_multiplication_table_reverse()
    print_multiplication_table_full()
    
    # 额外功能：查询特定乘法
    print("\n额外功能：查询特定乘法")
    print("输入两个数字（1-9），用空格分隔，例如：3 4")
    
    try:
        while True:
            user_input = input("\n输入数字（或输入 'q' 退出）: ").strip()
            if user_input.lower() == 'q':
                print("程序结束！")
                break
            
            try:
                nums = user_input.split()
                if len(nums) != 2:
                    print("请输入两个数字，用空格分隔")
                    continue
                
                a, b = int(nums[0]), int(nums[1])
                
                if 1 <= a <= 9 and 1 <= b <= 9:
                    print(f"{a} × {b} = {a*b}")
                else:
                    print("请输入1-9之间的数字")
            except ValueError:
                print("请输入有效的数字")
    except KeyboardInterrupt:
        print("\n程序被用户中断")

if __name__ == "__main__":
    main()