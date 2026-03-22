# 📖 第56课:在排序数组中查找元素的首末位置

> **模块**:二分查找 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/
> **前置知识**:第54课 二分查找、第55课 搜索插入位置
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个按照升序排列的整数数组 nums,和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 target,返回 [-1, -1]。

你必须设计并实现时间复杂度为 O(log n) 的算法解决此问题。

**示例:**
```
输入:nums = [5,7,7,8,8,10], target = 8
输出:[3,4]
解释:8 在数组中首次出现在索引3,最后出现在索引4

输入:nums = [5,7,7,8,8,10], target = 6
输出:[-1,-1]
解释:6 不存在于数组中
```

**约束条件:**
- 0 ≤ nums.length ≤ 10^5
- -10^9 ≤ nums[i] ≤ 10^9
- nums 是一个非递减数组(可能有重复元素)
- -10^9 ≤ target ≤ 10^9

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空数组 | nums=[], target=0 | [-1,-1] | 空数组处理 |
| 单元素匹配 | nums=[1], target=1 | [0,0] | 单元素边界 |
| 单元素不匹配 | nums=[1], target=2 | [-1,-1] | 不存在的情况 |
| 全部相同 | nums=[2,2,2,2], target=2 | [0,3] | 整个数组都是target |
| 首尾各一个 | nums=[1,2,3], target=2 | [1,1] | target只出现一次 |
| 在开头 | nums=[8,8,8,9], target=8 | [0,2] | 左边界在开头 |
| 在末尾 | nums=[1,8,8,8], target=8 | [1,3] | 右边界在末尾 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在图书馆找某个作家的所有作品,书按作者姓氏字母排序,这个作家有多本书放在一起。
>
> 🐌 **笨办法**:从头到尾扫描书架,记录这个作家第一次出现的位置和最后一次出现的位置。如果有10万本书,可能需要全部检查一遍。
>
> 🚀 **聪明办法**:
> 1. 用二分查找找到"第一本"这个作家的书(左边界)
> 2. 再用二分查找找到"最后一本"这个作家的书(右边界)
> 3. 两次二分,每次只需检查log n本书,10万本书只需要约17次检查!

### 关键洞察
**本题的核心是"左边界二分 + 右边界二分"的组合应用 — 两次二分查找分别找首尾位置!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:有序数组nums(可能有重复) + 目标值target
- **输出**:返回[首位置, 末位置],找不到返回[-1, -1]
- **限制**:必须O(log n)时间 → 提示用二分查找

### Step 2:先想笨办法(暴力法)
一次遍历,记录target第一次和最后一次出现的位置:
```python
first = last = -1
for i in range(len(nums)):
    if nums[i] == target:
        if first == -1:
            first = i
        last = i
return [first, last]
```
- 时间复杂度:O(n)
- 瓶颈在哪:没有利用"有序"特性,最坏情况(全部是target)需要扫描整个数组

### Step 3:瓶颈分析 → 优化方向
暴力法中即使找到target也要继续遍历。
- 核心问题:"有序 + 重复元素"的信息没有利用
- 优化思路:能不能用二分查找分别定位首尾位置?

### Step 4:选择武器
- 选用:**左边界二分 + 右边界二分**
- 理由:
  - 左边界二分找"第一个 >= target 的位置"
  - 右边界二分找"最后一个 <= target 的位置"
  - 两次O(log n)仍然是O(log n)

> 🔑 **模式识别提示**:当题目出现"有序数组 + 查找范围 + O(log n)",优先考虑"左右边界二分"组合

---

## 🔑 解法一:两次标准二分(直觉法)

### 思路
先用标准二分找到target的任意一个位置,然后从这个位置向左右扩展找边界。

### 图解过程

```
示例: nums = [5,7,7,8,8,10], target = 8

Step 1: 标准二分找到8的任意位置
  L           M           R
  ↓           ↓           ↓
[ 5,    7,    7,    8,    8,    10 ]
mid=2, nums[2]=7 < 8, 往右找

              L     M     R
              ↓     ↓     ↓
[ 5,    7,    7,    8,    8,    10 ]
mid=4, nums[4]=8 == 8, 找到了! 位置4

Step 2: 从位置4向左扩展找首位置
← ← ←
[ 5,    7,    7,    8,    8,    10 ]
                    ↑     ↑
                    3     4
首位置 = 3

Step 3: 从位置4向右扩展找末位置
                          → → →
[ 5,    7,    7,    8,    8,    10 ]
                    ↑     ↑
                    3     4
末位置 = 4

返回 [3, 4]
```

### Python代码

```python
from typing import List


def searchRange(nums: List[int], target: int) -> List[int]:
    """
    解法一:标准二分 + 线性扩展
    思路:先二分找到target,再向左右扩展找边界
    """
    if not nums:
        return [-1, -1]

    # 步骤1:标准二分找到target
    left, right = 0, len(nums) - 1
    found_idx = -1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            found_idx = mid
            break
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    if found_idx == -1:
        return [-1, -1]  # 没找到

    # 步骤2:向左扩展找首位置
    first = found_idx
    while first > 0 and nums[first - 1] == target:
        first -= 1

    # 步骤3:向右扩展找末位置
    last = found_idx
    while last < len(nums) - 1 and nums[last + 1] == target:
        last += 1

    return [first, last]


# ✅ 测试
print(searchRange([5, 7, 7, 8, 8, 10], 8))  # 期望输出:[3,4]
print(searchRange([5, 7, 7, 8, 8, 10], 6))  # 期望输出:[-1,-1]
print(searchRange([2, 2, 2, 2], 2))         # 期望输出:[0,3]
```

### 复杂度分析
- **时间复杂度**:O(n) — 最坏情况下,如果整个数组都是target,扩展步骤需要O(n)
  - 例如:nums=[8,8,8,...,8] (10万个8),扩展需要遍历10万次
- **空间复杂度**:O(1) — 只用了几个指针变量

### 优缺点
- ✅ 思路直观,容易想到
- ❌ **致命缺陷**:最坏情况退化为O(n),不满足题目要求的O(log n)!

---

## 🏆 解法二:左右边界二分(最优解)

### 优化思路
直接用两次二分查找分别找左边界和右边界,每次都是O(log n),总时间仍然是O(log n)。

> 💡 **关键想法**:
> - 左边界 = 第一个 >= target 的位置
> - 右边界 = 最后一个 <= target 的位置
> - 如果左边界位置的值不等于target,说明不存在

### 图解过程

```
示例: nums = [5,7,7,8,8,10], target = 8

========== 第一次二分:找左边界(第一个 >= 8) ==========

  L                                   R
  ↓                                   ↓
[ 5,    7,    7,    8,    8,    10 ] (范围:[0,6))
              M=3
nums[3]=8 >= 8, 可能是答案, 往左继续找

  L                 R
  ↓                 ↓
[ 5,    7,    7,    8,    8,    10 ] (范围:[0,3))
        M=1
nums[1]=7 < 8, 答案在右边

              L     R
              ↓     ↓
[ 5,    7,    7,    8,    8,    10 ] (范围:[2,3))
              M=2
nums[2]=7 < 8, 答案在右边

                    L
                    R
                    ↓
[ 5,    7,    7,    8,    8,    10 ] (范围:[3,3))
left == right, 结束, 左边界 = 3


========== 第二次二分:找右边界(最后一个 <= 8) ==========

  L                                   R
  ↓                                   ↓
[ 5,    7,    7,    8,    8,    10 ] (范围:[0,6))
              M=3
nums[3]=8 <= 8, 继续往右找

                    L                 R
                    ↓                 ↓
[ 5,    7,    7,    8,    8,    10 ] (范围:[4,6))
                          M=5
nums[5]=10 > 8, 答案在左边

                    L     R
                    ↓     ↓
[ 5,    7,    7,    8,    8,    10 ] (范围:[4,5))
                    M=4
nums[4]=8 <= 8, 继续往右找

                          L
                          R
                          ↓
[ 5,    7,    7,    8,    8,    10 ] (范围:[5,5))
left == right, 结束, 右边界 = 5-1 = 4

最终答案:[3, 4]
```

### Python代码

```python
def searchRange_v2(nums: List[int], target: int) -> List[int]:
    """
    解法二:左右边界二分 (最优解)
    思路:分别用二分查找左边界和右边界
    """
    if not nums:
        return [-1, -1]

    # 辅助函数:左边界二分(第一个 >= target)
    def find_left_bound(nums, target):
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left

    # 辅助函数:右边界二分(最后一个 <= target)
    def find_right_bound(nums, target):
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left - 1  # 注意:返回 left-1

    # 找左边界
    left_idx = find_left_bound(nums, target)

    # 检查是否存在
    if left_idx >= len(nums) or nums[left_idx] != target:
        return [-1, -1]

    # 找右边界
    right_idx = find_right_bound(nums, target)

    return [left_idx, right_idx]


# ✅ 测试
print(searchRange_v2([5, 7, 7, 8, 8, 10], 8))  # 期望输出:[3,4]
print(searchRange_v2([5, 7, 7, 8, 8, 10], 6))  # 期望输出:[-1,-1]
print(searchRange_v2([2, 2, 2, 2], 2))         # 期望输出:[0,3]
print(searchRange_v2([], 0))                   # 期望输出:[-1,-1]
print(searchRange_v2([1], 1))                  # 期望输出:[0,0]
```

### 复杂度分析
- **时间复杂度**:O(log n) — 两次二分查找,每次O(log n)
  - 具体地说:如果输入规模 n=100000,每次二分约需17次比较,总共34次
- **空间复杂度**:O(1) — 只用了几个指针变量

**为什么是最优解**:
- 时间O(log n)已经是理论最优,满足题目严格要求
- 即使整个数组都是target,仍然是O(log n),不会退化
- 代码模板清晰,可以复用左右边界二分的标准模板

---

## ⚡ 解法三:一次二分 + 优化扩展(折中方案)

### 优化思路
如果target数量不多,可以先二分找到一个位置,然后二分扩展而非线性扩展。

### Python代码

```python
def searchRange_v3(nums: List[int], target: int) -> List[int]:
    """
    解法三:标准二分 + 二分扩展
    思路:找到target后,用二分法在左右两侧继续查找边界
    """
    if not nums:
        return [-1, -1]

    # 步骤1:标准二分找到任意一个target
    left, right = 0, len(nums) - 1
    found_idx = -1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            found_idx = mid
            break
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    if found_idx == -1:
        return [-1, -1]

    # 步骤2:在[0, found_idx]用二分找左边界
    left, right = 0, found_idx
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    first = left

    # 步骤3:在[found_idx, len-1]用二分找右边界
    left, right = found_idx, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    last = left - 1

    return [first, last]


# ✅ 测试
print(searchRange_v3([5, 7, 7, 8, 8, 10], 8))  # 期望输出:[3,4]
print(searchRange_v3([2, 2, 2, 2], 2))         # 期望输出:[0,3]
```

### 复杂度分析
- **时间复杂度**:O(log n) — 三次二分查找
- **空间复杂度**:O(1)

### 优缺点
- ✅ 时间复杂度符合要求
- ⚠️ 代码比解法二复杂,且性能没有提升

---

## 🐍 Pythonic 写法

利用 Python 的 `bisect` 模块:

```python
import bisect


def searchRange_pythonic(nums: List[int], target: int) -> List[int]:
    """
    Pythonic写法:使用bisect模块
    bisect_left(nums, target) = 第一个 >= target 的位置
    bisect_right(nums, target) = 第一个 > target 的位置
    """
    left_idx = bisect.bisect_left(nums, target)

    # 检查是否存在
    if left_idx >= len(nums) or nums[left_idx] != target:
        return [-1, -1]

    # bisect_right 返回第一个 > target 的位置,所以要减1
    right_idx = bisect.bisect_right(nums, target) - 1

    return [left_idx, right_idx]


# ✅ 测试
print(searchRange_pythonic([5, 7, 7, 8, 8, 10], 8))  # 期望输出:[3,4]
print(searchRange_pythonic([5, 7, 7, 8, 8, 10], 6))  # 期望输出:[-1,-1]
```

**说明**:
- `bisect_left(nums, target)`:第一个 >= target 的位置(左边界)
- `bisect_right(nums, target)`:第一个 > target 的位置,所以右边界是它减1
- 一行代码解决,但面试时仍需手写展示算法理解

> ⚠️ **面试建议**:先手写解法二展示对左右边界二分的深刻理解,通过后再提`bisect`展示Python功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:二分+线性扩展 | 🏆 解法二:左右边界二分(最优) | 解法三:二分+二分扩展 | Pythonic:bisect |
|------|------------------|------------------------|------------------|----------------|
| 时间复杂度 | **O(n)** ← 致命缺陷 | **O(log n)** ← 最优 | O(log n) | O(log n) |
| 空间复杂度 | O(1) | O(1) | O(1) | O(1) |
| 代码难度 | 简单 | **中等(需理解模板)** | 较复杂 | 极简 |
| 面试推荐 | ⭐(不满足要求) | **⭐⭐⭐** ← 首选 | ⭐⭐ | ⭐(辅助) |
| 适用场景 | target数量极少 | **所有情况通用** | 折中方案 | Python快速实现 |

**为什么解法二是最优**:
- 时间O(log n)严格满足题目要求,即使全数组都是target也不退化
- 代码模板清晰,复用了左右边界二分的标准写法
- 逻辑独立:左右边界的查找互不干扰,易于理解和调试

**面试建议**:
1. 先用30秒口述暴力法思路(O(n)遍历找边界),表明理解题意
2. 提到解法一(二分+扩展)的思路,并指出其O(n)的缺陷
3. 🏆 重点讲解解法二(左右边界二分),强调"复用第55课的左边界模板"
4. 手动测试边界用例:空数组、单元素、全部相同、target不存在
5. 时间充裕时提一下bisect模块

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求在有序数组中找target的首末位置,并且要求O(log n)时间。让我先想一下...

最直接的想法是遍历一遍数组,记录第一次和最后一次出现的位置,但这是O(n),不满足要求。

题目要求O(log n),提示我们要用二分查找。我的思路是:
1. 用左边界二分找到"第一个 >= target"的位置
2. 用右边界二分找到"最后一个 <= target"的位置
3. 检查左边界位置的值是否等于target,如果不等于说明不存在

这样两次二分,每次O(log n),总时间仍然是O(log n)。

**面试官**:很好,左边界和右边界二分有什么区别?

**你**:核心区别在于更新策略:
- **左边界二分**:当`nums[mid] >= target`时,`right = mid`保留mid,继续往左找
- **右边界二分**:当`nums[mid] <= target`时,`left = mid + 1`,继续往右找,最后返回`left - 1`

左边界返回的是"第一个不小于target"的位置,右边界返回的是"最后一个不大于target"的位置。

**面试官**:请写一下代码。

**你**:(边写边说关键步骤,写出解法二的代码)

**面试官**:测试一下?

**你**:用示例[5,7,7,8,8,10], target=8走一遍:
- 左边界二分:在[0,6)区间找,最终定位到索引3(第一个8)
- 右边界二分:在[0,6)区间找,最终定位到索引4(最后一个8)
- 返回[3,4] ✅

再测target=6(不存在的情况):
- 左边界二分:会定位到索引2(第一个 >= 6的位置是7)
- 检查nums[2]=7 ≠ 6,返回[-1,-1] ✅

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间O(log n)已经是理论最优。两次二分是必须的,因为左右边界的位置独立,不能用一次二分同时确定。 |
| "能只用一次二分吗?" | 可以先二分找到任意位置,再向左右扩展,但扩展步骤最坏O(n),不满足题目要求。 |
| "为什么右边界要返回left-1?" | 因为右边界二分的循环条件是`nums[mid] <= target`时`left = mid + 1`,循环结束时left指向"第一个 > target"的位置,所以要减1。 |
| "如果target不存在怎么判断?" | 找到左边界后,检查`left_idx >= len(nums)` 或 `nums[left_idx] != target`,任一成立就说明不存在。 |
| "能用Python标准库吗?" | 可以用`bisect.bisect_left`和`bisect.bisect_right`,但手写更能展示对二分查找的理解。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:左右边界二分的区别
def left_bound(nums, target):
    """第一个 >= target"""
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid  # 保留mid,继续往左
    return left

def right_bound(nums, target):
    """最后一个 <= target"""
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1  # 继续往右
        else:
            right = mid
    return left - 1  # 返回left-1

# 技巧2:bisect模块的四个函数
import bisect
bisect.bisect_left(nums, x)   # 第一个 >= x 的位置(左边界)
bisect.bisect_right(nums, x)  # 第一个 > x 的位置
bisect.bisect(nums, x)        # 同bisect_right
bisect.insort(nums, x)        # 插入x并保持有序
```

### 💡 底层原理(选读)

> **左右边界二分的本质区别**
>
> 两者的区别在于"保留策略":
> - **左边界**:当找到候选时,保留它(`right=mid`),继续往左找更小的
> - **右边界**:当找到候选时,跳过它(`left=mid+1`),继续往右找更大的,最后回退一步
>
> **为什么右边界返回left-1?**
> - 循环不变量:[left, right)区间内所有元素 <= target
> - 当`nums[mid] <= target`时,`left = mid + 1`意味着"mid及左边都 <= target,继续看右边"
> - 循环结束时,left指向"第一个 > target"的位置,所以右边界是`left - 1`
>
> **记忆技巧**:
> - 左边界:往左找 → 保留候选 → `right = mid` → 返回`left`
> - 右边界:往右找 → 跳过候选 → `left = mid+1` → 返回`left-1`

### 算法模式卡片 📐
- **模式名称**:左右边界二分组合
- **适用条件**:有序数组 + 可能有重复元素 + 查找范围区间
- **识别关键词**:"有序"、"首末位置"、"范围"、"起始结束"、"O(log n)"
- **模板代码**:
```python
def search_range(nums: List[int], target: int) -> List[int]:
    """查找target的[首位置, 末位置]"""
    if not nums:
        return [-1, -1]

    # 左边界:第一个 >= target
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    left_idx = left

    # 检查是否存在
    if left_idx >= len(nums) or nums[left_idx] != target:
        return [-1, -1]

    # 右边界:最后一个 <= target
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    right_idx = left - 1

    return [left_idx, right_idx]
```

### 易错点 ⚠️
1. **右边界返回值错误**
   - 错误:返回`left`而不是`left - 1`
   - 原因:右边界的`left`指向"第一个 > target",要减1才是"最后一个 <= target"
   - 正确:`return left - 1`

2. **忘记检查target是否存在**
   - 错误:直接返回左右边界,没检查`nums[left_idx]`是否等于`target`
   - 后果:target不存在时会返回错误的范围
   - 正确:先检查`left_idx >= len(nums) or nums[left_idx] != target`

3. **左右边界条件混淆**
   - 错误:两个二分的条件写成一样的
   - 正确:左边界用`<`,右边界用`<=`

4. **空数组边界**
   - 错误:没有处理`nums`为空的情况
   - 正确:函数开头加`if not nums: return [-1, -1]`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:日志系统 — 在按时间戳排序的海量日志中,快速定位某个时间段的所有日志(左右边界查找时间戳)
- **场景2**:数据库索引 — MySQL的InnoDB引擎用B+树索引,范围查询`WHERE age BETWEEN 20 AND 30`本质就是左右边界查找
- **场景3**:版本管理 — 在Git的commit历史中,快速找到某个功能首次引入和最后修改的版本
- **场景4**:搜索引擎 — 在倒排索引中,查找包含某个关键词的文档ID范围

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 35. 搜索插入位置 | Easy | 左边界二分 | 第55课,本题的简化版 |
| LeetCode 278. 第一个错误的版本 | Easy | 左边界二分 | 找第一个返回true的版本号 |
| LeetCode 162. 寻找峰值 | Medium | 二分变体 | 局部有序,用二分找峰值 |
| LeetCode 540. 有序数组中的单一元素 | Medium | 二分 + 奇偶性 | 利用下标奇偶性判断答案在哪边 |
| LeetCode 74. 搜索二维矩阵 | Medium | 二分查找 | 将二维矩阵看作一维有序数组 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定有序数组nums和目标值target,统计target在数组中出现的次数。要求O(log n)时间复杂度。例如nums=[5,7,7,8,8,10], target=8,返回2。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

复用本题的左右边界二分!出现次数 = 右边界 - 左边界 + 1。注意处理target不存在的情况。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def countTarget(nums: List[int], target: int) -> int:
    """
    统计target出现次数 = 右边界 - 左边界 + 1
    """
    if not nums:
        return 0

    # 左边界:第一个 >= target
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid
    left_idx = left

    # 检查是否存在
    if left_idx >= len(nums) or nums[left_idx] != target:
        return 0

    # 右边界:最后一个 <= target
    left, right = 0, len(nums)
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1
        else:
            right = mid
    right_idx = left - 1

    return right_idx - left_idx + 1


# 测试
print(countTarget([5, 7, 7, 8, 8, 10], 8))  # 输出:2
print(countTarget([5, 7, 7, 8, 8, 10], 6))  # 输出:0
print(countTarget([2, 2, 2, 2], 2))         # 输出:4
```

核心思路:直接复用左右边界二分的代码,最后返回`right_idx - left_idx + 1`即可。时间复杂度仍然是O(log n)。

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
