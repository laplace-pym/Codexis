def quicksort(arr):
    """快速排序算法"""
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def quicksort_inplace(arr, low=0, high=None):
    """原地快速排序（不使用额外空间）"""
    if high is None:
        high = len(arr) - 1

    if low < high:
        pivot_idx = partition(arr, low, high)
        quicksort_inplace(arr, low, pivot_idx - 1)
        quicksort_inplace(arr, pivot_idx + 1, high)


def partition(arr, low, high):
    """分区函数"""
    pivot = arr[high]
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1


if __name__ == "__main__":
    # 测试
    test_arr = [64, 34, 25, 12, 22, 11, 90, 5]

    print("原始数组:", test_arr)
    print("排序结果:", quicksort(test_arr))

    # 测试原地排序
    arr2 = [64, 34, 25, 12, 22, 11, 90, 5]
    quicksort_inplace(arr2)
    print("原地排序:", arr2)
