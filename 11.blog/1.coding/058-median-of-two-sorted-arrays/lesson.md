> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第58课:两个正序数组的中位数

> **模块**:二分查找 | **难度**:Hard ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/median-of-two-sorted-arrays/
> **前置知识**:第54课(二分查找)
> **预计学习时间**:40分钟

---

## 🎯 题目描述

给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的中位数。

要求算法的时间复杂度为 O(log(m+n))。

**示例:**
```
输入:nums1 = [1,3], nums2 = [2]
输出:2.0
解释:合并后 [1,2,3],中位数是 2

输入:nums1 = [1,2], nums2 = [3,4]
输出:2.5
解释:合并后 [1,2,3,4],中位数是 (2+3)/2 = 2.5
```

**约束条件:**
- nums1.length == m, nums2.length == n
- 0 ≤ m ≤ 1000, 0 ≤ n ≤ 1000
- 1 ≤ m + n ≤ 2000
- -10^6 ≤ nums1[i], nums2[i] ≤ 10^6
- **核心约束**:时间复杂度必须是 O(log(m+n))

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 数组长度不等 | nums1=[1,3], nums2=[2] | 2.0 | 基本功能 |
| 总长度为偶数 | nums1=[1,2], nums2=[3,4] | 2.5 | 偶数中位数 |
| 一个数组为空 | nums1=[], nums2=[1] | 1.0 | 空数组处理 |
| 两数组无交集 | nums1=[1,2], nums2=[5,6] | 3.5 | 不相交情况 |
| 负数 | nums1=[-5,-3], nums2=[-2,-1] | -2.5 | 负数处理 |
| 极端大小 | nums1=[1], nums2=[2,3,4,5,6] | 3.5 | 一个很小一个很大 |

---

## 💡 思路引导

### 生活化比喻
> 想象你和朋友各自拿着一叠已排好序的扑克牌,现在要找出"合并后的中位数"牌是哪张。
>
> 🐌 **笨办法**:把两叠牌真的合并成一叠,重新排序,然后数到中间位置。这需要翻所有的牌。
>
> 🚀 **聪明办法**:其实你不需要真的合并!你只需要找到"第 k 小"的牌(k = 总数/2)。关键洞察是:如果你从两叠牌的中间位置各抽一张比较,你就能排除掉一半不可能是答案的牌!比如你的牌堆中间是 5,朋友的中间是 3,你要找第 7 小的牌,那么朋友的前半部分(1,2,3...)肯定都太小了,可以直接扔掉!这样每次都能排除一半,只需要翻 log(总数) 次牌。

### 关键洞察
**中位数的本质是将数组分成左右两半,左半部分的最大值 ≤ 右半部分的最小值。我们不需要真的合并数组,只需要在两个数组上找到正确的"分割线",使得左边所有元素 ≤ 右边所有元素,且两边元素数量相等(或左边多1个)。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:两个有序数组 nums1 (长度m), nums2 (长度n)
- **输出**:中位数(浮点数)
  - 如果总长度为奇数,返回中间元素
  - 如果总长度为偶数,返回中间两个元素的平均值
- **限制**:时间复杂度必须是 O(log(m+n)) → 提示不能遍历所有元素

### Step 2:先想笨办法(暴力法)
直接合并两个数组,排序,然后找中位数。
- 时间复杂度:O(m+n) — 需要遍历所有元素
- 瓶颈在哪:没有利用"两个数组已经有序"的特性

### Step 3:瓶颈分析 → 优化方向
观察发现:
- 核心问题:要求 O(log(m+n)) 意味着不能遍历所有元素,必须用类似二分的方法
- 优化思路:中位数的本质是"第 k 小的数"(k = (m+n+1)/2)。我们可以在两个有序数组上做二分查找,每次排除一半不可能是答案的元素。

更进一步的洞察:
- **分割线思想**:在 nums1 和 nums2 上各画一条分割线,使得:
  - 左边元素总数 = 右边元素总数(或左边多1个)
  - 左边最大值 ≤ 右边最小值
- 这样左边的最大值(或左边最大的两个值的平均)就是中位数!

### Step 4:选择武器
- 选用:**在较短数组上进行二分查找**
- 理由:
  1. 在 nums1 上二分确定分割位置 i
  2. 根据"左右元素数量相等"推导出 nums2 的分割位置 j
  3. 检查分割是否合法(左边最大 ≤ 右边最小)
  4. 根据检查结果调整二分搜索范围
  5. 时间复杂度 O(log(min(m,n)))

> 🔑 **模式识别提示**:当题目出现"两个有序数组"、"O(log n)"时,优先考虑"二分查找"

---

## 🔑 解法一:合并数组找中位数(直觉法)

### 思路
直接合并两个数组(利用双指针归并),然后找中位数。虽然不满足 O(log) 要求,但容易理解。

### 图解过程

```
nums1 = [1,3], nums2 = [2]

Step 1: 双指针归并
  p1 → 1, p2 → 2
  比较: 1 < 2, 取 1, merged = [1]

Step 2:
  p1 → 3, p2 → 2
  比较: 3 > 2, 取 2, merged = [1,2]

Step 3:
  p1 → 3, p2 已结束
  取 3, merged = [1,2,3]

Step 4: 找中位数
  总长度 n = 3 (奇数)
  中位数索引 = 3//2 = 1
  返回 merged[1] = 2
```

### Python代码

```python
from typing import List


def findMedianSortedArrays_merge(nums1: List[int], nums2: List[int]) -> float:
    """
    解法一:合并数组找中位数
    思路:双指针归并,然后取中位数
    """
    # 归并两个有序数组
    merged = []
    i, j = 0, 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            merged.append(nums1[i])
            i += 1
        else:
            merged.append(nums2[j])
            j += 1

    # 处理剩余元素
    merged.extend(nums1[i:])
    merged.extend(nums2[j:])

    # 计算中位数
    n = len(merged)
    if n % 2 == 1:  # 奇数
        return float(merged[n // 2])
    else:  # 偶数
        return (merged[n // 2 - 1] + merged[n // 2]) / 2.0


# ✅ 测试
print(findMedianSortedArrays_merge([1,3], [2]))     # 期望输出:2.0
print(findMedianSortedArrays_merge([1,2], [3,4]))   # 期望输出:2.5
print(findMedianSortedArrays_merge([], [1]))        # 期望输出:1.0
```

### 复杂度分析
- **时间复杂度**:O(m+n) — 需要遍历两个数组的所有元素
  - 具体地说:如果 m=500, n=500, 需要 1000 次操作
- **空间复杂度**:O(m+n) — 需要额外数组存储合并结果

### 优缺点
- ✅ 代码简单,容易理解
- ✅ 逻辑直观,不容易出错
- ❌ 时间复杂度 O(m+n),不满足题目要求 O(log(m+n))
- ❌ 需要额外的 O(m+n) 空间

---

## ⚡ 解法二:双指针找第k小(优化)

### 优化思路
我们不需要真的合并数组!只需要用双指针"走"到中位数位置即可。

> 💡 **关键想法**:中位数是"第 k 小的数",其中 k = (m+n+1)//2。我们只需要移动指针 k 次,最后指向的就是中位数。

### 图解过程

```
nums1 = [1,3], nums2 = [2,4]
总长度 = 4 (偶数),需要找第 2 和第 3 小的数

初始:
nums1: [1, 3]
       ↑
       p1
nums2: [2, 4]
       ↑
       p2

Step 1: 比较 nums1[0]=1, nums2[0]=2
  1 < 2, 移动 p1, count=1

Step 2: 比较 nums1[1]=3, nums2[0]=2
  3 > 2, 移动 p2, count=2 ← 记录为 median1

Step 3: 比较 nums1[1]=3, nums2[1]=4
  3 < 4, 移动 p1, count=3 ← 记录为 median2

中位数 = (median1 + median2) / 2 = (2 + 3) / 2 = 2.5
```

### Python代码

```python
def findMedianSortedArrays_kth(nums1: List[int], nums2: List[int]) -> float:
    """
    解法二:双指针找第k小
    思路:不合并数组,只用双指针走到中位数位置
    """
    m, n = len(nums1), len(nums2)
    total = m + n

    # 需要找到的位置
    if total % 2 == 1:
        # 奇数:只需找第 k 小
        k = total // 2
        return float(find_kth_element(nums1, nums2, k))
    else:
        # 偶数:需找第 k 和 k+1 小
        k1 = total // 2 - 1
        k2 = total // 2
        return (find_kth_element(nums1, nums2, k1) +
                find_kth_element(nums1, nums2, k2)) / 2.0


def find_kth_element(nums1: List[int], nums2: List[int], k: int) -> int:
    """找第 k 小的元素(k 从 0 开始)"""
    i, j = 0, 0
    count = 0

    while i < len(nums1) and j < len(nums2):
        if nums1[i] <= nums2[j]:
            if count == k:
                return nums1[i]
            i += 1
        else:
            if count == k:
                return nums2[j]
            j += 1
        count += 1

    # 处理剩余元素
    if i < len(nums1):
        return nums1[i + k - count]
    else:
        return nums2[j + k - count]


# ✅ 测试
print(findMedianSortedArrays_kth([1,3], [2]))     # 期望输出:2.0
print(findMedianSortedArrays_kth([1,2], [3,4]))   # 期望输出:2.5
```

### 复杂度分析
- **时间复杂度**:O(m+n) — 虽然不合并,但仍需走 k 步
- **空间复杂度**:O(1) — 只用了常数变量

---

## 🏆 解法三:二分查找分割线(最优解)

### 优化思路
这是本题的精髓!关键洞察:
1. **分割线思想**:在两个数组上画分割线,使得左边元素数量 = 右边元素数量
2. **合法性检查**:左边最大值 ≤ 右边最小值
3. **二分搜索**:在较短数组上二分确定分割位置

> 💡 **核心想法**:不需要找"第几小",直接找"正确的分割位置"!

### 图解过程

```
nums1 = [1,3,8,9,15], nums2 = [7,11,18,19,21,25]
总长度 = 11 (奇数),左半部分需要 6 个元素

在 nums1 上二分查找分割位置 i:

尝试 i=2 (在 nums1 的 8 前面分割):
nums1: [1, 3 | 8, 9, 15]
        left1   right1
       maxLeft1=3, minRight1=8

根据 i=2 推导 j (使得左边总数=6):
j = (11+1)//2 - 2 = 6 - 2 = 4

nums2: [7, 11, 18, 19 | 21, 25]
        left2           right2
       maxLeft2=19, minRight2=21

检查是否合法:
  左边最大值 max(3, 19) = 19
  右边最小值 min(8, 21) = 8
  19 > 8 ✗ 不合法! (左边太大了)

→ 说明 i=2 太小了,需要增大 i (让 nums1 左边包含更多大的数)


尝试 i=1:
nums1: [1 | 3, 8, 9, 15]
       maxLeft1=1, minRight1=3

j = 6 - 1 = 5

nums2: [7, 11, 18, 19, 21 | 25]
       maxLeft2=21, minRight2=25

检查:
  max(1, 21) = 21
  min(3, 25) = 3
  21 > 3 ✗ 仍不合法


尝试 i=0:
nums1: [| 1, 3, 8, 9, 15]
       maxLeft1=-∞, minRight1=1

j = 6 - 0 = 6

nums2: [7, 11, 18, 19, 21, 25 |]
       maxLeft2=25, minRight2=+∞

检查:
  max(-∞, 25) = 25
  min(1, +∞) = 1
  25 > 1 ✗ 仍不合法!


回到 i=3:
nums1: [1, 3, 8 | 9, 15]
       maxLeft1=8, minRight1=9

j = 6 - 3 = 3

nums2: [7, 11, 18 | 19, 21, 25]
       maxLeft2=18, minRight2=19

检查:
  max(8, 18) = 18
  min(9, 19) = 9
  18 ≤ 9 ? 不对...


正确的分割:i=4, j=2
nums1: [1, 3, 8, 9 | 15]
       maxLeft1=9, minRight1=15

nums2: [7, 11 | 18, 19, 21, 25]
       maxLeft2=11, minRight2=18

检查:
  max(9, 11) = 11
  min(15, 18) = 15
  11 ≤ 15 ✓ 合法!

因为总长度是奇数,中位数 = 左边最大值 = max(9, 11) = 11
```

更清晰的图解示意:

```
分割线的本质:

nums1:  1   3   8 | 9  15        (分割位置 i=3)
nums2:  7  11  18  19 | 21  25   (分割位置 j=4)
        ←─ 左半部分 ─→  ←─ 右半部分 ─→

左半部分: [1,3,8,7,11,18,19]  → 最大值 max(8, 19) = 19
右半部分: [9,15,21,25]        → 最小值 min(9, 21) = 9

检查: 19 ≤ 9 ? 否 → 不合法,需要调整 i
```

### Python代码

```python
def findMedianSortedArrays(nums1: List[int], nums2: List[int]) -> float:
    """
    解法三:二分查找分割线(最优解)
    思路:在较短数组上二分,找到正确的分割位置
    """
    # 确保 nums1 是较短的数组(优化性能)
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    total = m + n
    half = (total + 1) // 2  # 左半部分需要的元素数量

    # 在 nums1 上进行二分查找
    left, right = 0, m

    while left <= right:
        i = (left + right) // 2  # nums1 的分割位置
        j = half - i              # nums2 的分割位置

        # 获取分割线左右两侧的值
        nums1_left = nums1[i - 1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j - 1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')

        # 检查分割是否合法
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # 找到正确的分割!
            if total % 2 == 1:
                # 奇数:返回左半部分最大值
                return float(max(nums1_left, nums2_left))
            else:
                # 偶数:返回中间两个值的平均
                return (max(nums1_left, nums2_left) +
                        min(nums1_right, nums2_right)) / 2.0
        elif nums1_left > nums2_right:
            # nums1 左边太大,减小 i
            right = i - 1
        else:
            # nums1 左边太小,增大 i
            left = i + 1

    return 0.0  # 不会到达这里


# ✅ 测试
print(findMedianSortedArrays([1,3], [2]))           # 期望输出:2.0
print(findMedianSortedArrays([1,2], [3,4]))         # 期望输出:2.5
print(findMedianSortedArrays([], [1]))              # 期望输出:1.0
print(findMedianSortedArrays([1,3,8,9,15], [7,11,18,19,21,25]))  # 期望输出:11.0
```

### 复杂度分析
- **时间复杂度**:O(log(min(m,n))) — 在较短数组上进行二分查找
  - 具体地说:如果 m=100, n=900, 只需要 log2(100) ≈ 7 次比较
  - 相比 O(m+n) 的解法快了 1000/7 ≈ 143 倍!
- **空间复杂度**:O(1) — 只用了常数变量

**为什么这是最优解**:
- 时间复杂度 O(log(min(m,n))) 已经是最优(符合题目要求 O(log(m+n)))
- 空间复杂度 O(1),没有额外开销
- 不需要真的合并数组,直接找到答案

---

## 🐍 Pythonic 写法

Python 内置的归并虽然简洁,但时间复杂度是 O(m+n):

```python
# 简洁但不满足 O(log) 要求
def findMedianSortedArrays_pythonic(nums1: List[int], nums2: List[int]) -> float:
    merged = sorted(nums1 + nums2)  # O((m+n)log(m+n))
    n = len(merged)
    return merged[n//2] if n % 2 else (merged[n//2-1] + merged[n//2]) / 2
```

> ⚠️ **面试建议**:先写🏆最优解(解法三)展示算法功底。如果时间紧张,可以先讲思路再写代码。
> 面试官更看重你的**分割线思想**和**二分查找应用能力**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:合并数组 | 解法二:双指针找k小 | 🏆 解法三:二分分割线(最优) |
|------|--------------|-----------------|---------------------|
| 时间复杂度 | O(m+n) | O(m+n) | **O(log min(m,n))** ← 最优 |
| 空间复杂度 | O(m+n) | O(1) | **O(1)** |
| 代码难度 | 简单 | 中等 | 较难 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 学习理解 | 过渡思路 | **Hard题标准答案** |

**面试建议**:
1. 先用 1 分钟讲暴力法(合并数组 O(m+n)),表明你理解题意
2. 提出优化方向:"题目要求 O(log),提示使用二分查找"
3. **重点讲解🏆最优解的核心思想**:"分割线"+"二分搜索"
4. 画图解释分割线的含义和合法性检查
5. 手动模拟一个小例子,展示二分过程
6. 强调边界处理:空数组、分割在端点时用 ±∞

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题 1 分钟)好的,这道题要求在两个有序数组中找中位数,时间复杂度必须是 O(log(m+n))。让我先想一下...

我的第一个想法是合并两个数组,然后找中位数,时间复杂度是 O(m+n),但不满足要求。

题目要求 O(log) 级别,提示应该用二分查找。关键洞察是:中位数的本质是"将数组分成左右两半,左边最大值 ≤ 右边最小值"。

我可以在两个数组上各画一条分割线,使得:
1. 左半部分元素数量 = 右半部分元素数量
2. 左边最大值 ≤ 右边最小值

这样左边的最大值就是中位数(或中间两个值的平均)。

具体做法是:在较短的数组上进行二分查找,确定分割位置 i,然后根据"左右数量相等"推导另一个数组的分割位置 j。检查是否满足"左边最大 ≤ 右边最小",如果不满足就调整 i 的位置。

**面试官**:很好,请写一下代码。

**你**:(边写边说关键步骤)

```python
def findMedianSortedArrays(nums1, nums2):
    # 确保在较短数组上二分
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2

    left, right = 0, m

    while left <= right:
        i = (left + right) // 2  # nums1 的分割位置
        j = half - i              # nums2 的分割位置

        # 获取分割线左右的值
        nums1_left = nums1[i-1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j-1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')

        # 检查合法性
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # 找到正确的分割
            if (m + n) % 2 == 1:
                return max(nums1_left, nums2_left)
            else:
                return (max(nums1_left, nums2_left) +
                        min(nums1_right, nums2_right)) / 2.0
        elif nums1_left > nums2_right:
            right = i - 1  # nums1 左边太大
        else:
            left = i + 1   # nums1 左边太小

    return 0.0
```

**面试官**:能解释一下为什么用 float('-inf') 和 float('inf') 吗?

**你**:这是边界处理技巧。当分割线在数组端点时(i=0 或 i=m):
- 如果 i=0,说明 nums1 的左半部分为空,用 -∞ 表示"没有左边界"
- 如果 i=m,说明 nums1 的右半部分为空,用 +∞ 表示"没有右边界"

这样可以简化代码,不需要额外的 if 判断。比如 max(3, -∞) = 3, min(5, +∞) = 5。

**面试官**:测试一下?

**你**:用示例 [1,3], [2] 走一遍...
- m=2, n=1, half=2, 在 nums1 上二分
- 第一轮:i=1, j=1, nums1_left=1, nums2_left=2, nums1_right=3, nums2_right=+∞
- 检查:max(1,2)=2 ≤ min(3,+∞)=3 ✓ 合法
- 总长度为3(奇数),返回 max(1,2) = 2

结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么在较短数组上二分?" | "因为二分的时间复杂度是 O(log m),在短数组上二分更快。而且能保证 j = half - i 不会越界(如果 m > n,可能算出 j < 0)" |
| "如果两个数组长度相等呢?" | "可以在任意一个上二分,结果一样。但为了代码统一,我先判断长度做了交换" |
| "能否用递归实现?" | "可以,但递归需要 O(log m) 栈空间,不如迭代的 O(1)。面试推荐迭代版本" |
| "实际应用场景?" | "大数据处理中合并两个已排序的数据流,需要实时计算中位数(如监控系统的延迟中位数)" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:交换确保 nums1 更短
if len(nums1) > len(nums2):
    nums1, nums2 = nums2, nums1
# 原因:在短数组上二分更快,且能保证计算 j 时不越界

# 技巧2:边界处理用 ±∞
nums1_left = nums1[i-1] if i > 0 else float('-inf')
nums1_right = nums1[i] if i < m else float('inf')
# 原因:简化边界判断,max(x, -∞)=x, min(x, +∞)=x

# 技巧3:左半部分数量计算
half = (m + n + 1) // 2
# 原因:+1 确保奇数时左边多一个元素,方便统一处理
```

### 💡 底层原理(选读)

> **为什么分割线思想是正确的?**
>
> 中位数的定义:将数组排序后,位于中间位置的数(或中间两数的平均)。
>
> 如果我们将两个数组"虚拟合并"后的结果看作:
> ```
> 左半部分 | 右半部分
> ```
>
> 那么:
> - 左半部分包含前 (m+n+1)//2 个最小的数
> - 右半部分包含剩余的数
> - 左边最大值 ≤ 右边最小值(因为排序)
>
> 我们不需要真的合并!只需要在两个数组上分别找到分割位置 i 和 j,使得:
> - i + j = (m+n+1)//2 (左边元素总数)
> - nums1[i-1] ≤ nums2[j] 且 nums2[j-1] ≤ nums1[i] (交叉比较确保整体有序)
>
> 这样,左边的最大值 max(nums1[i-1], nums2[j-1]) 就是中位数(或中间较小的那个)!
>
> **为什么二分搜索 i 是对的?**
>
> - 如果 nums1[i-1] > nums2[j],说明 nums1 的左边包含了太多大的数,应该减小 i
> - 如果 nums2[j-1] > nums1[i],说明 nums1 的左边包含了太多小的数,应该增大 i
>
> 这样每次都能排除一半的搜索空间,保证找到正确的分割位置!

### 算法模式卡片 📐
- **模式名称**:二分查找 - 分割线法
- **适用条件**:两个有序数组的合并问题,需要 O(log) 时间
- **识别关键词**:"两个有序数组"、"中位数"、"第k小"、"O(log)"
- **模板代码**:
```python
def find_partition(nums1, nums2):
    # 在较短数组上二分
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    half = (m + n + 1) // 2
    left, right = 0, m

    while left <= right:
        i = (left + right) // 2
        j = half - i

        # 边界处理
        nums1_left = nums1[i-1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j-1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')

        # 检查分割合法性
        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            return (nums1_left, nums1_right, nums2_left, nums2_right)
        elif nums1_left > nums2_right:
            right = i - 1
        else:
            left = i + 1

    return None
```

### 易错点 ⚠️
1. **错误**:在较长数组上二分
   - **为什么错**:可能导致 j = half - i < 0 越界
   - **正确做法**:先判断长度,在短数组上二分

2. **错误**:边界处理时用 0 代替 ±∞
   - **为什么错**:0 可能是数组中的有效值,会导致比较错误
   - **正确做法**:用 float('-inf') 和 float('inf')

3. **错误**:计算 half 时忘记 +1
   - **为什么错**:奇数长度时,左半部分应该多一个元素,否则无法正确返回中位数
   - **正确做法**:half = (m + n + 1) // 2

4. **错误**:只检查 nums1_left ≤ nums2_right,忘记检查另一侧
   - **为什么错**:必须交叉检查两侧,确保整体有序
   - **正确做法**:同时检查 nums1_left ≤ nums2_right 和 nums2_left ≤ nums1_right

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:数据库查询优化
  - 两个已排序的数据表(如按时间戳排序的日志表),需要找中位数响应时间
  - 使用分割线法在 O(log n) 时间内完成,避免全表扫描

- **场景2**:实时数据流分析
  - 监控系统收集两个数据中心的延迟数据(都是排序的)
  - 需要实时计算全局延迟中位数,用于告警

- **场景3**:分布式系统
  - MapReduce 中合并多个已排序的中间结果
  - 需要高效找到中位数用于负载均衡

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 2386. 找出数组的第K大和 | Hard | 二分查找+堆 | 类似思想:第k大问题 |
| LeetCode 719. 找出第K小的数对距离 | Hard | 二分答案 | 二分答案+双指针判定 |
| LeetCode 378. 有序矩阵中第K小的元素 | Medium | 二分查找 | 二维有序结构的第k小 |
| LeetCode 668. 乘法表中第K小的数 | Hard | 二分答案 | 虚拟有序结构的二分 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定两个有序数组 nums1 和 nums2,以及整数 k,找出合并后的数组中第 k 小的元素(k 从 1 开始)。要求 O(log(m+n)) 时间。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

这题和中位数本质一样!中位数就是"第 (m+n+1)//2 小"。用同样的分割线思想,只是 half = k 而不是 (m+n+1)//2。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def findKthElement(nums1: List[int], nums2: List[int], k: int) -> int:
    """找第 k 小的元素(k 从 1 开始)"""
    # 确保 nums1 是较短的数组
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1

    m, n = len(nums1), len(nums2)
    left, right = 0, m

    while left <= right:
        i = (left + right) // 2
        j = k - i  # 注意这里是 k,不是 (m+n+1)//2

        # 边界检查:确保 j 不越界
        if j < 0:
            right = i - 1
            continue
        if j > n:
            left = i + 1
            continue

        nums1_left = nums1[i-1] if i > 0 else float('-inf')
        nums1_right = nums1[i] if i < m else float('inf')
        nums2_left = nums2[j-1] if j > 0 else float('-inf')
        nums2_right = nums2[j] if j < n else float('inf')

        if nums1_left <= nums2_right and nums2_left <= nums1_right:
            # 找到第 k 小,就是左边的最大值
            return max(nums1_left, nums2_left)
        elif nums1_left > nums2_right:
            right = i - 1
        else:
            left = i + 1

    return -1


# 测试
print(findKthElement([1,3], [2], 2))           # 输出:2 (合并后[1,2,3],第2小是2)
print(findKthElement([1,2], [3,4], 3))         # 输出:3 (合并后[1,2,3,4],第3小是3)
print(findKthElement([1,3,5,7], [2,4,6,8], 5)) # 输出:5
```

**核心思路**:
- 分割位置:i + j = k (而不是 (m+n+1)//2)
- 第 k 小的元素 = max(nums1_left, nums2_left)
- 需要额外检查 j 是否越界(0 ≤ j ≤ n)

</details>

### 📱 关注微信公众号「汀丶人工智能」
🔥 精选AI前沿资讯 | 📚 深度技术解读 | 💡 实战案例分享

### 🤝 欢迎加入AI Compass知识星球
🎯 **更深入的内容** - 独家教程、项目实战、面试指导  
⚡ **更高的更新频率** - 高频资讯推送、专家答疑、技术交流  
🎁 **限时优惠** - 与数千名AI学习者一起成长！
  * [AI Compass知识星球](https://t.zsxq.com/Tj1eS)
  * [🎫 AI Compass知识星球优惠券](https://github.com/tingaicompass/AI-Compass/blob/main/picture/minor/KnowledgePlanet.md)
>星球支持三天内免费退款，请放心订阅。

<table>
<tr>
<td width="50%" valign="top">

## 💬技术博客
* [CSDN](https://blog.csdn.net/sinat_39620217?type=blog)  
* [掘金](https://juejin.cn/user/4020284493662029)
* [知乎](https://www.zhihu.com/people/tingaicompass)
* [公众号](https://github.com/tingaicompass/AI-Compass/blob/main/picture/main/wx.png)
* [知识星球](https://github.com/tingaicompass/AI-Compass/blob/main/picture/minor/KnowledgePlanet.md)

</td>
<td width="50%" valign="top">

## 📍社交媒体
* [头条📬](https://profile.zjurl.cn/rogue/ugc/profile/?active_tab=dongtai&app_name=news_article&device_id=65&media_id=1719833587832835&request_source=1&share_token=b744b824-20ff-420e-b4f7-6080ad127720&tt_from=copy_link&user_id=3287673762&utm_campaign=client_share&utm_medium=toutiao_android&utm_source=copy_link&version_code=120900&version_name=0)
* [抖音🎶](https://v.douyin.com/ZbvqNyHo61I/)
* [小红书📕](https://www.xiaohongshu.com/user/profile/605c395e000000000100108b?xsec_token=YBq0UxPBd23DZ-rGp87wTY2qVctMuK7wWKQU9LsMEaGnw%3D&xsec_source=app_share&xhsshare=CopyLink&appuid=605c395e000000000100108b&apptime=1752306657&share_id=38c139d8155e4692b37a6316559ae8b3&share_channel=copy_link)

</td>
</tr>
</table>

---

> 如果这篇内容对你有帮助，推荐收藏 AI Compass：https://github.com/tingaicompass/AI-Compass
> 更多系统化题解、编程基础和 AI 学习资料都在这里，后续复习和拓展会更省时间。
