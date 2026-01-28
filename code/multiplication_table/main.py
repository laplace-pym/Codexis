def print_multiplication_table():
    for i in range(1, 10):
        row = []
        for j in range(1, i + 1):
            row.append(f"{j}×{i}={i*j}")
        print("\t".join(row))


if __name__ == "__main__":
    print_multiplication_table()
