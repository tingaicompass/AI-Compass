# 📖 第57课：搜索旋转排序数组

> **模块**：二分查找 | **难度**：Medium ⭐⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/search-in-rotated-sorted-array/
> **前置知识**：第54课（二分查找）
> **预计学习时间**：30分钟

---

## 🎯 题目描述

给你一个升序排列的整数数组 `nums`，它在某个未知的位置被旋转了（例如 `[0,1,2,4,5,6,7]` 可能变成 `[4,5,6,7,0,1,2]`）。现在给你一个目标值 `target`，请你在这个旋转数组中搜索它。如果存在返回下标，否则返回 `-1`。

**示例：**
```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4
解释：target = 0 在数组中的下标是 4
```

**约束条件：**
- 1 ≤ nums.length ≤ 5000
- -10^4 ≤ nums[i] ≤ 10^4
- nums 中的值互不相同（无重复）
- 数组原本是升序排列的，但在某个位置被旋转
- 要求时间复杂度必须是 O(log n)

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 未旋转 | nums=[1,2,3,4,5], target=3 | 2 | 基本二分查找 |
| 旋转1位 | nums=[5,1,2,3,4], target=1 | 1 | 旋转点在开头 |
| target在左半边 | nums=[4,5,6,7,0,1,2], target=5 | 1 | 有序区间判断 |
| target在右半边 | nums=[4,5,6,7,0,1,2], target=1 | 5 | 跨旋转点查找 |
| target不存在 | nums=[4,5,6,7,0,1,2], target=3 | -1 | 查找失败 |
| 单元素 | nums=[1], target=1 | 0 | 最小规模 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在一个特殊的图书馆找书，书架上的书原本是按编号 1-100 顺序排列的，但某天管理员把前 60 本书搬到了后面，现在书架变成了：61, 62...100, 1, 2...60。
>
> 🐌 **笨办法**：从头到尾逐本翻看，找到目标书为止。这样需要检查最多 100 本书。
>
> 🚀 **聪明办法**：虽然整体被"折断"了，但你注意到一个关键规律——无论你从中间随便抽一本书，它左边或右边至少有一侧仍然是完全有序的！比如你抽到编号 80，那么 61-80 这段一定是有序的；如果你抽到编号 30，那么 30-60 这段一定是有序的。于是你可以判断：如果目标书在有序的那一侧，就在那边二分查找；否则去另一侧继续同样的策略。这样只需要检查 log2(100) ≈ 7 本书！

### 关键洞察
**旋转数组虽然整体无序，但被 mid 分割后，左右两部分必有一部分完全有序！只要先判断哪边有序，再判断 target 是否在有序区间内，就能决定搜索方向。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出
- **输入**：旋转后的数组 `nums`（整数，无重复），目标值 `target`
- **输出**：目标值的下标（整数），不存在返回 -1
- **限制**：必须 O(log n) 时间 → 提示使用二分查找

### Step 2：先想笨办法（暴力法）
直接遍历数组，逐个比较元素是否等于 `target`。
- 时间复杂度：O(n)
- 瓶颈在哪：没有利用数组"部分有序"的特性，只是当成普通数组暴力查找

### Step 3：瓶颈分析 → 优化方向
虽然数组被旋转了，但它不是完全无序的：
- 核心问题：如何在旋转数组上使用二分查找？
- 优化思路：观察发现，用 mid 将数组分成两段后，**至少有一段是完全有序的**。比如 `[4,5,6,7,0,1,2]` 的 mid=3 (值为7)，左半边 `[4,5,6,7]` 有序，右半边 `[0,1,2]` 也有序（虽然小于左边）。

### Step 4：选择武器
- 选用：**改进的二分查找**
- 理由：
  1. 每次 mid 将数组分成两段，至少有一段有序
  2. 判断 target 是否在有序段范围内，决定搜索方向
  3. 保持 O(log n) 时间复杂度

> 🔑 **模式识别提示**：当题目出现"有序数组被旋转"、"O(log n) 查找"，优先考虑"二分查找变体"

---

## 🔑 解法一：线性扫描（直觉法）

### 思路
不管三七二十一，直接遍历数组找 target。虽然简单但不满足 O(log n) 要求。

### 图解过程

```
数组: [4,5,6,7,0,1,2]  target = 0

Step 1: 检查索引 0, nums[0] = 4 ≠ 0
Step 2: 检查索引 1, nums[1] = 5 ≠ 0
Step 3: 检查索引 2, nums[2] = 6 ≠ 0
Step 4: 检查索引 3, nums[3] = 7 ≠ 0
Step 5: 检查索引 4, nums[4] = 0 = 0 ✓ 找到！返回 4
```

### Python代码

```python
from typing import List


def search_linear(nums: List[int], target: int) -> int:
    """
    解法一：线性扫描
    思路：逐个检查每个元素
    """
    # 遍历数组
    for i in range(len(nums)):
        if nums[i] == target:  # 找到目标
            return i
    return -1  # 未找到


# ✅ 测试
print(search_linear([4,5,6,7,0,1,2], 0))  # 期望输出：4
print(search_linear([4,5,6,7,0,1,2], 3))  # 期望输出：-1
print(search_linear([1], 0))              # 期望输出：-1
```

### 复杂度分析
- **时间复杂度**：O(n) — 最坏情况需要遍历整个数组
  - 具体地说：如果输入规模 n=5000，最坏需要 5000 次比较
- **空间复杂度**：O(1) — 只用了常数变量

### 优缺点
- ✅ 代码简单，易于理解
- ❌ 时间复杂度 O(n)，不满足题目要求（需要 O(log n)）
- ❌ 没有利用数组"部分有序"的特性

---

## 🏆 解法二：一次二分查找（最优解）

### 优化思路
旋转数组的关键性质是：用任意位置 mid 切分后，**左半部分和右半部分至少有一个是完全有序的**。我们可以：
1. 先判断哪半边有序（通过比较 nums[left] 和 nums[mid]）
2. 再判断 target 是否在有序区间内
3. 根据判断结果决定搜索左半边还是右半边

> 💡 **关键想法**：不需要先找旋转点！直接在二分过程中利用"必有一侧有序"的性质即可。

### 图解过程

```
数组: [4,5,6,7,0,1,2]  target = 0

初始状态:
  L               M               R
  ↓               ↓               ↓
  4   5   6   7   0   1   2
索引0   1   2   3   4   5   6

Step 1: left=0, right=6, mid=3, nums[mid]=7
  判断哪边有序？
  nums[left]=4 < nums[mid]=7 → 左半边 [4,5,6,7] 有序
  target=0 在 [4,7] 范围内吗？ 否
  → 搜索右半边，left = mid+1 = 4

当前状态:
                  L       M       R
                  ↓       ↓       ↓
  4   5   6   7   0   1   2
索引0   1   2   3   4   5   6

Step 2: left=4, right=6, mid=5, nums[mid]=1
  判断哪边有序？
  nums[left]=0 < nums[mid]=1 → 左半边 [0,1] 有序
  target=0 在 [0,1] 范围内吗？ 是
  → 搜索左半边，right = mid-1 = 4

当前状态:
                  L=M=R
                  ↓
  4   5   6   7   0   1   2
索引0   1   2   3   4   5   6

Step 3: left=4, right=4, mid=4, nums[mid]=0
  nums[mid] = target ✓ 返回 4
```

再看一个例子（target 在左半边）：

```
数组: [4,5,6,7,0,1,2]  target = 5

Step 1: left=0, right=6, mid=3, nums[mid]=7
  左半边有序 [4,5,6,7]
  target=5 在 [4,7] 内吗？ 是
  → 搜索左半边，right = mid-1 = 2

Step 2: left=0, right=2, mid=1, nums[mid]=5
  nums[mid] = target ✓ 返回 1
```

### Python代码

```python
def search(nums: List[int], target: int) -> int:
    """
    解法二：一次二分查找（最优解）
    思路：利用"必有一侧有序"的性质，在二分过程中判断搜索方向
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        # 找到目标
        if nums[mid] == target:
            return mid

        # 判断哪边有序
        if nums[left] <= nums[mid]:  # 左半边有序
            # 判断 target 是否在左半边的有序区间内
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # 在左半边搜索
            else:
                left = mid + 1   # 在右半边搜索
        else:  # 右半边有序
            # 判断 target 是否在右半边的有序区间内
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # 在右半边搜索
            else:
                right = mid - 1  # 在左半边搜索

    return -1  # 未找到


# ✅ 测试
print(search([4,5,6,7,0,1,2], 0))  # 期望输出：4
print(search([4,5,6,7,0,1,2], 3))  # 期望输出：-1
print(search([1], 0))              # 期望输出：-1
print(search([1,3], 3))            # 期望输出：1
```

### 复杂度分析
- **时间复杂度**：O(log n) — 标准二分查找，每次排除一半
  - 具体地说：如果输入规模 n=5000，只需要 log2(5000) ≈ 13 次比较
  - 相比线性扫描快了 5000/13 ≈ 385 倍！
- **空间复杂度**：O(1) — 只用了 left、right、mid 三个变量

**为什么这是最优解**：
- 时间复杂度 O(log n) 已经是查找问题的理论最优（不能不看元素就知道答案）
- 空间复杂度 O(1)，没有额外开销
- 只需一次二分遍历，代码简洁高效

---

## ⚡ 解法三：先找旋转点再二分（两步法）

### 优化思路
另一种思路是先用二分查找找到旋转点（最小值位置），然后判断 target 在哪个有序子数组中，再用标准二分查找。

> 💡 **关键想法**：虽然也能达到 O(log n)，但需要两次二分查找，略微复杂。

### 图解过程

```
数组: [4,5,6,7,0,1,2]  target = 0

Phase 1: 找旋转点（最小值）
  使用二分查找找到 nums[i] < nums[i-1] 的位置
  找到旋转点索引 = 4 (值为0)

Phase 2: 判断 target 在哪个子数组
  target=0 < nums[0]=4 → 在右半边
  在 [4,5,6] 中用标准二分查找 target=0
  找到索引 4
```

### Python代码

```python
def search_two_pass(nums: List[int], target: int) -> int:
    """
    解法三：先找旋转点再二分（两步法）
    思路：第一次二分找旋转点，第二次标准二分查找
    """
    n = len(nums)
    if n == 1:
        return 0 if nums[0] == target else -1

    # Phase 1: 找旋转点（最小值位置）
    left, right = 0, n - 1
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] > nums[right]:
            left = mid + 1  # 旋转点在右边
        else:
            right = mid     # 旋转点在左边或就是 mid

    pivot = left  # 旋转点索引

    # Phase 2: 判断 target 在哪个有序子数组中
    left, right = 0, n - 1
    if target >= nums[pivot] and target <= nums[right]:
        left = pivot  # 在右半边
    else:
        right = pivot - 1  # 在左半边

    # Phase 3: 标准二分查找
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


# ✅ 测试
print(search_two_pass([4,5,6,7,0,1,2], 0))  # 期望输出：4
print(search_two_pass([4,5,6,7,0,1,2], 3))  # 期望输出：-1
```

### 复杂度分析
- **时间复杂度**：O(log n) — 两次二分查找，仍然是 O(log n)
- **空间复杂度**：O(1) — 常数空间

---

## 🐍 Pythonic 写法

利用 Python 的 `in` 操作符虽然简洁，但时间复杂度是 O(n)，不推荐：

```python
# 简洁但不满足 O(log n) 要求
def search_pythonic(nums: List[int], target: int) -> int:
    return nums.index(target) if target in nums else -1
```

> ⚠️ **面试建议**：先写🏆最优解（解法二）展示算法功底，如果有时间可以提解法三作为备选思路。
> 面试官更看重你的**二分查找思维**和**边界处理能力**。

---

## 📊 解法对比

| 维度 | 解法一：线性扫描 | 🏆 解法二：一次二分（最优） | 解法三：两步二分 |
|------|--------------|----------------------|--------------|
| 时间复杂度 | O(n) | **O(log n)** ← 最优 | O(log n) |
| 空间复杂度 | O(1) | **O(1)** | O(1) |
| 代码难度 | 简单 | 中等 | 中等偏难 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 不符合题目要求 | **面试标准答案** | 备用思路 |

**面试建议**：
1. 先用 30 秒口述暴力法（O(n)），表明你理解题意
2. 立即优化到🏆最优解（解法二：一次二分），展示对二分变体的掌握
3. **重点讲解关键技巧**："先判断哪边有序，再判断 target 是否在有序区间内"
4. 手动测试边界用例：未旋转数组、单元素、target不存在
5. 如果有时间，可以提解法三作为另一种思路

---

## 🎤 面试现场

> 模拟面试中的完整对话流程，帮你练习"边想边说"。

**面试官**：请你解决一下这道题。

**你**：（审题 30 秒）好的，这道题要求在旋转排序数组中查找目标值，时间复杂度必须是 O(log n)。让我先想一下...

我的第一个想法是直接遍历数组，时间复杂度是 O(n)，但不满足要求。

观察旋转数组的性质，我发现一个关键点：虽然整体被旋转了，但用 mid 分割后，**左右两半必有一边是完全有序的**。比如 `[4,5,6,7,0,1,2]` 的 mid=3，左半边 `[4,5,6,7]` 有序。

所以我可以这样做：
1. 每次二分时先判断哪边有序（比较 nums[left] 和 nums[mid]）
2. 再判断 target 是否在有序区间内
3. 根据判断结果决定搜索方向

这样可以保持 O(log n) 时间复杂度。

**面试官**：很好，请写一下代码。

**你**：（边写边说）

```python
def search(nums, target):
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid

        # 关键：判断哪边有序
        if nums[left] <= nums[mid]:  # 左半边有序
            # 判断 target 是否在左半边的范围内
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:  # 右半边有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1

    return -1
```

**面试官**：测试一下？

**你**：用示例 `[4,5,6,7,0,1,2]`, target=0 走一遍...
- 第一轮：mid=3, nums[mid]=7, 左边有序但 target 不在 [4,7] 内，搜索右边
- 第二轮：mid=5, nums[mid]=1, 左边有序且 target 在 [0,1] 内，搜索左边
- 第三轮：mid=4, nums[mid]=0, 找到！返回 4

再测一个边界情况：未旋转的 `[1,2,3,4,5]`, target=3
- 第一轮：mid=2, nums[mid]=3, 直接找到！返回 2

结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果数组有重复元素怎么办？" | "有重复元素时，无法通过 nums[left] 和 nums[mid] 直接判断哪边有序（可能相等），需要额外处理：当 nums[left] == nums[mid] 时，left++ 跳过重复元素，最坏退化到 O(n)" |
| "能否用递归实现？" | "可以，但递归需要 O(log n) 栈空间，不如迭代的 O(1) 空间。面试中推荐迭代版本" |
| "如果要找最小值而不是查找 target 呢？" | "那就是解法三的第一步，持续二分缩小范围直到 left == right，此时 nums[left] 就是最小值" |
| "这个算法的实际应用场景？" | "日志系统中的时间戳查询、循环有序数据结构的搜索（如循环队列）" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1：二分查找防溢出写法
mid = left + (right - left) // 2  # 而不是 (left + right) // 2
# 原因：防止 left + right 溢出（虽然 Python 整数不溢出，但这是好习惯）

# 技巧2：边界条件的等号处理
if nums[left] <= nums[mid]:  # 注意这里的 <=
# 原因：当数组只有两个元素时，mid == left，必须加等号才能正确判断

# 技巧3：区间范围判断
if nums[left] <= target < nums[mid]:  # 左闭右开
# 原因：target == nums[mid] 的情况已经在前面处理了
```

### 💡 底层原理（选读）

> **为什么旋转数组"必有一侧有序"？**
>
> 旋转数组本质上是将一个有序数组从某个位置切一刀，把前半段接到后半段后面。例如：
> - 原数组：`[0,1,2,4,5,6,7]`
> - 在索引 4 处切割：前半 `[0,1,2,4]`，后半 `[5,6,7]`
> - 旋转后：`[5,6,7,0,1,2,4]`
>
> 当你用 mid 再次切割旋转后的数组时，有两种情况：
> 1. **mid 落在后半段**（值较大）：`[left, mid]` 跨越了旋转点，但 `[mid, right]` 仍然有序
> 2. **mid 落在前半段**（值较小）：`[mid, right]` 跨越了旋转点，但 `[left, mid]` 仍然有序
>
> 无论哪种情况，至少有一侧不包含旋转点，因此必有一侧有序！
>
> **如何判断哪边有序？**
> - 如果 `nums[left] <= nums[mid]`，说明 `[left, mid]` 不包含旋转点，必然有序
> - 否则，`[mid, right]` 不包含旋转点，必然有序

### 算法模式卡片 📐
- **模式名称**：二分查找变体 - 旋转数组搜索
- **适用条件**：有序数组在某个位置被旋转，需要 O(log n) 查找
- **识别关键词**："旋转排序数组"、"升序数组旋转"、"O(log n)"
- **模板代码**：
```python
def search_rotated(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = left + (right - left) // 2
        if nums[mid] == target:
            return mid

        # 判断哪边有序
        if nums[left] <= nums[mid]:  # 左边有序
            if nums[left] <= target < nums[mid]:
                right = mid - 1  # target 在左边
            else:
                left = mid + 1
        else:  # 右边有序
            if nums[mid] < target <= nums[right]:
                left = mid + 1   # target 在右边
            else:
                right = mid - 1
    return -1
```

### 易错点 ⚠️
1. **错误**：判断有序时写成 `if nums[left] < nums[mid]`（缺少等号）
   - **为什么错**：当数组只有两个元素时，mid == left，不加等号会走到 else 分支，导致判断错误
   - **正确做法**：写成 `nums[left] <= nums[mid]`

2. **错误**：判断 target 范围时写成 `if nums[left] <= target <= nums[mid]`
   - **为什么错**：如果 `target == nums[mid]`，这个条件会进入 if 分支，但前面已经判断过 `nums[mid] == target` 了，所以这里应该排除相等的情况
   - **正确做法**：写成 `nums[left] <= target < nums[mid]`（右边用 `<`）

3. **错误**：忘记处理单元素数组
   - **为什么错**：单元素数组没有旋转概念，但代码仍需正确处理
   - **正确做法**：while 条件用 `left <= right`，自然包含单元素情况

---

## 🏗️ 工程实战（选读）

> 这个算法思想在真实项目中的应用，让你知道"学了有什么用"。

- **场景1**：日志系统中的时间戳查询
  - 日志文件按时间排序，但跨天时会从头开始（形成"旋转"）
  - 使用旋转数组搜索快速定位某个时间点的日志

- **场景2**：循环缓冲区（Ring Buffer）
  - 固定大小的缓冲区，写满后从头开始覆盖（逻辑上形成"旋转"有序结构）
  - 需要快速查找某个值是否在缓冲区中

- **场景3**：任务调度系统
  - 任务按优先级排序，但高优先级任务可能被"旋转"到队列尾部
  - 使用旋转数组搜索快速定位任务位置

---

## 🏋️ 举一反三

完成本课后，试试这些同类题目来巩固知识：

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 81. 搜索旋转排序数组 II | Medium | 二分查找变体 | 与本题区别：数组可能有重复元素，需要处理 nums[left] == nums[mid] 的情况 |
| LeetCode 153. 寻找旋转排序数组中的最小值 | Medium | 二分查找 | 不需要查找 target，只需找最小值（旋转点） |
| LeetCode 154. 寻找旋转排序数组中的最小值 II | Hard | 二分查找变体 | 有重复元素的最小值查找 |
| LeetCode 162. 寻找峰值 | Medium | 二分查找 | 类似思想：判断 mid 与相邻元素的关系决定搜索方向 |

---

## 📝 课后小测

试试这道变体题，不要看答案，自己先想5分钟！

**题目**：给定一个旋转排序数组，不给 target，直接返回数组中的最小值。例如 `[4,5,6,7,0,1,2]` 返回 `0`。

<details>
<summary>💡 提示（实在想不出来再点开）</summary>

最小值就是旋转点！判断 nums[mid] 与 nums[right] 的大小关系，决定最小值在左边还是右边。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def find_min(nums: List[int]) -> int:
    """找旋转排序数组的最小值"""
    left, right = 0, len(nums) - 1

    while left < right:  # 注意这里是 < 不是 <=
        mid = left + (right - left) // 2

        # 如果 mid 比 right 大，说明最小值在右边
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            # 最小值在左边或就是 mid
            right = mid

    return nums[left]  # left == right 时就是最小值


# 测试
print(find_min([4,5,6,7,0,1,2]))  # 输出：0
print(find_min([3,4,5,1,2]))      # 输出：1
print(find_min([1,2,3,4,5]))      # 输出：1（未旋转）
```

**核心思路**：
- 比较 nums[mid] 和 nums[right]（不是 nums[left]）
- 如果 nums[mid] > nums[right]，说明旋转点在右半边
- 否则旋转点在左半边或就是 mid
- 最终 left == right 时就是最小值位置

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
