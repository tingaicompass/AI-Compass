# 📖 第10课:接雨水

> **模块**:双指针 | **难度**:Hard ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/trapping-rain-water/
> **前置知识**:第7课(移动零-快慢指针)、第8课(盛最多水的容器-对撞指针)
> **预计学习时间**:35分钟

---

## 🎯 题目描述

给定 n 个非负整数表示每个宽度为 1 的柱子的高度,请计算下雨之后能接多少雨水。

**示例:**
```
输入:height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出:6
解释:如下图,接雨水的总量为 6 个单位。

高度:  3      █
       2    █ ░ █
       1  █ ░ █ ░ ░ █ ░ █
       0█ ░ █ ░ ░ ░ ░ ░ ░ ░ █
         0 1 0 2 1 0 1 3 2 1 2 1
```

**示例 2:**
```
输入:height = [4,2,0,3,2,5]
输出:9
```

**约束条件:**
- 1 <= height.length <= 2 * 10⁴
- 0 <= height[i] <= 10⁵

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | height=[0,1,0] | 0 | 基本功能 |
| 无法接水 | height=[1,2,3,4] | 0 | 单调递增无凹槽 |
| 全零 | height=[0,0,0] | 0 | 特殊值处理 |
| 两端高中间低 | height=[3,0,0,2,0,4] | 10 | 多个凹槽累加 |
| 大规模 | n=20000 | — | 性能边界O(n) |

---

## 💡 思路引导

### 生活化比喻

> 想象你站在一排高低不齐的柱子前,下了一场大雨。
>
> 🐌 **笨办法**:你拿着一个量杯,站在每根柱子前,抬头看左边最高的柱子是多高,再看右边最高的柱子是多高。取两边较矮的那个作为"水位线",然后计算当前位置能装多少水。这样每根柱子都要扫描一遍左边和右边,太慢了!
>
> 🧠 **聪明办法**:你先花一点时间,从左到右走一遍,记录下每个位置"左边的最高柱";再从右到左走一遍,记录下每个位置"右边的最高柱"。之后只需要一次遍历,用 min(左最高, 右最高) - 当前高度 就能算出每个位置的积水。
>
> 🚀 **天才办法**:用两个指针从两端同时向中间走,动态维护左右最高值。每次移动较矮的一端,因为水位由较矮的一端决定!不需要预处理,一次遍历搞定,空间O(1)。

### 关键洞察

**每个位置能接的雨水量 = min(左边最高柱, 右边最高柱) - 当前柱高度(如果为正)**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组 height,表示每个柱子的高度
- **输出**:整数,表示能接的雨水总量(单位体积)
- **限制**:需要考虑 n 可能很大(2万),要求 O(n) 时间复杂度

### Step 2:先想笨办法(暴力法)

对于每个位置 i:
1. 向左扫描找到最高柱 left_max
2. 向右扫描找到最高柱 right_max
3. 该位置能接的水 = min(left_max, right_max) - height[i](如果 > 0)

- 时间复杂度:O(n²) — 每个位置都要扫描左右两边
- 瓶颈在哪:**重复扫描**!每次都要遍历左右区间找最大值

### Step 3:瓶颈分析 → 优化方向

分析暴力法中"重复计算"的环节:
- 核心问题:对于每个位置,我们反复计算左侧最大值和右侧最大值
- 优化思路:能不能提前把这些信息算好存起来?→ **动态规划预处理**
- 进一步优化:能不能连预处理都省掉?→ **双指针动态维护**

### Step 4:选择武器
- 选用:**双指针 + 贪心思想**
- 理由:用两个指针从两端向中间移动,动态维护 left_max 和 right_max,避免额外空间,一次遍历完成

> 🔑 **模式识别提示**:当题目需要"每个位置依赖左右两侧信息"时,优先考虑"双指针对撞"或"预处理数组"

---

## 🔑 解法一:动态规划(预处理左右最大值)

### 思路

用两个辅助数组提前算好:
- `left_max[i]`:位置 i 左侧(包括i)的最大高度
- `right_max[i]`:位置 i 右侧(包括i)的最大高度

然后遍历每个位置,计算 `min(left_max[i], right_max[i]) - height[i]`。

### 图解过程

```
示例:height = [0,1,0,2,1,0,1,3,2,1,2,1]

Step 1:从左到右,构建 left_max 数组
  位置:  0  1  2  3  4  5  6  7  8  9  10 11
  高度:  0  1  0  2  1  0  1  3  2  1  2  1
left_max: 0  1  1  2  2  2  2  3  3  3  3  3
          ↑每个位置记录"从0到i的最大值"

Step 2:从右到左,构建 right_max 数组
  位置:  0  1  2  3  4  5  6  7  8  9  10 11
  高度:  0  1  0  2  1  0  1  3  2  1  2  1
right_max:3  3  3  3  3  3  3  3  2  2  2  1
          ↑每个位置记录"从i到末尾的最大值"

Step 3:计算每个位置的积水量
  位置 i=2:
    水位 = min(left_max[2]=1, right_max[2]=3) = 1
    积水 = 1 - height[2]=0 = 1  ✓

  位置 i=4:
    水位 = min(left_max[4]=2, right_max[4]=3) = 2
    积水 = 2 - height[4]=1 = 1  ✓

  位置 i=5:
    水位 = min(left_max[5]=2, right_max[5]=3) = 2
    积水 = 2 - height[5]=0 = 2  ✓

总积水量 = 0+0+1+0+1+2+1+0+0+0+0+0 = 6
```

### Python代码

```python
from typing import List


def trap_dp(height: List[int]) -> int:
    """
    解法一:动态规划预处理
    思路:提前计算每个位置的左右最大高度
    """
    if not height or len(height) < 3:
        return 0

    n = len(height)

    # 1. 构建 left_max:left_max[i] 表示 [0..i] 的最大高度
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    # 2. 构建 right_max:right_max[i] 表示 [i..n-1] 的最大高度
    right_max = [0] * n
    right_max[n - 1] = height[n - 1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    # 3. 计算每个位置的积水量
    total_water = 0
    for i in range(n):
        water_level = min(left_max[i], right_max[i])
        if water_level > height[i]:
            total_water += water_level - height[i]

    return total_water


# ✅ 测试
print(trap_dp([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))  # 期望输出:6
print(trap_dp([4, 2, 0, 3, 2, 5]))  # 期望输出:9
print(trap_dp([1, 2, 3, 4]))  # 期望输出:0(单调递增)
```

### 复杂度分析
- **时间复杂度**:O(n) — 三次遍历,分别构建 left_max、right_max 和计算结果
  - 具体地说:如果输入规模 n=10000,大约需要 3×10000 = 30000 次操作
- **空间复杂度**:O(n) — 需要两个辅助数组存储左右最大值

### 优缺点
- ✅ 思路清晰,易于理解
- ✅ 时间复杂度最优 O(n)
- ❌ 需要额外 O(n) 空间,能否优化?→ 引出解法二

---

## ⚡ 解法二:双指针(空间优化到O(1))

### 优化思路

观察解法一,我们发现:
- 每个位置的积水量只依赖 `min(left_max, right_max)`
- 如果我们知道当前位置的左右最大值,就不需要完整的数组!

用两个指针 `left` 和 `right` 从两端向中间移动:
- 维护 `left_max` 和 `right_max`
- **关键洞察**:如果 `left_max < right_max`,那么左指针位置的积水量由 `left_max` 决定(因为右边一定有更高的柱子);反之亦然
- 每次移动较矮的一端

> 💡 **关键想法**:双指针 + 贪心 — 水位由较矮的一端决定,所以总是移动较矮的一端并计算积水

### 图解过程

```
示例:height = [0,1,0,2,1,0,1,3,2,1,2,1]

初始化:
  left=0, right=11
  left_max=0, right_max=1

Step 1:
  height[0]=0 < height[11]=1 → 移动 left
  left_max=max(0,0)=0
  积水 = 0-0 = 0
  left=1

Step 2:
  height[1]=1 = height[11]=1 → 移动 left(可任选)
  left_max=max(0,1)=1
  积水 = 1-1 = 0
  left=2

Step 3:
  height[2]=0 < height[11]=1 → 移动 left
  left_max=max(1,0)=1
  积水 = 1-0 = 1  ← 第一个积水!
  left=3

Step 4:
  height[3]=2 > height[11]=1 → 移动 right
  right_max=max(1,1)=1
  积水 = 1-1 = 0
  right=10

...继续移动,直到 left > right

最终积水量 = 6
```

### Python代码

```python
def trap(height: List[int]) -> int:
    """
    解法二:双指针(推荐!)
    思路:从两端向中间移动,动态维护左右最大高度
    """
    if not height or len(height) < 3:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    total_water = 0

    while left < right:
        # 更新左右最大值
        left_max = max(left_max, height[left])
        right_max = max(right_max, height[right])

        # 关键:水位由较矮的一端决定
        if left_max < right_max:
            # 左边是短板,计算左指针的积水
            total_water += left_max - height[left]
            left += 1
        else:
            # 右边是短板,计算右指针的积水
            total_water += right_max - height[right]
            right -= 1

    return total_water


# ✅ 测试
print(trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))  # 期望输出:6
print(trap([4, 2, 0, 3, 2, 5]))  # 期望输出:9
print(trap([1, 2, 3, 4]))  # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n) — 只需一次遍历,每个元素访问一次
- **空间复杂度**:O(1) — 只用了常数个变量

---

## 🚀 解法三:单调栈(横向计算积水)

### 优化思路

前两种解法是"竖着"计算每个位置的积水,单调栈则是"横着"计算:
- 维护一个单调递减的栈(存下标)
- 当遇到比栈顶更高的柱子时,说明形成了一个"凹槽"
- 弹出栈顶作为"凹槽底部",计算这一层的积水面积

> 💡 **关键想法**:单调栈 — 当前元素比栈顶大时,触发"出栈并计算面积"

### 图解过程

```
示例:height = [0,1,0,2,1,0,1,3,2,1,2,1]

Step 1:i=0, height=0
  栈:[0]

Step 2:i=1, height=1 > height[0]=0
  形成凹槽!弹出栈顶 0
  left=栈顶(空,跳过)
  栈:[1]

Step 3:i=2, height=0 < height[1]=1
  栈:[1,2]

Step 4:i=3, height=2 > height[2]=0
  弹出 2(底部)
  left=1, right=3
  高度 = min(height[1]=1, height[3]=2) - height[2]=0 = 1
  宽度 = 3-1-1 = 1
  积水 += 1×1 = 1

  继续:height=2 > height[1]=1
  弹出 1
  left=栈顶(空,跳过)
  栈:[3]

...继续处理

总积水量 = 6
```

### Python代码

```python
def trap_stack(height: List[int]) -> int:
    """
    解法三:单调栈
    思路:横向计算每一层的积水面积
    """
    if not height or len(height) < 3:
        return 0

    stack = []  # 存储下标,栈内对应的高度单调递减
    total_water = 0

    for i in range(len(height)):
        # 当前柱子高于栈顶,形成凹槽
        while stack and height[i] > height[stack[-1]]:
            bottom_idx = stack.pop()  # 凹槽底部

            if not stack:
                break  # 左边没有柱子,无法接水

            left_idx = stack[-1]  # 左边界
            right_idx = i  # 右边界

            # 计算这一层的积水
            h = min(height[left_idx], height[right_idx]) - height[bottom_idx]
            w = right_idx - left_idx - 1
            total_water += h * w

        stack.append(i)

    return total_water


# ✅ 测试
print(trap_stack([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))  # 期望输出:6
print(trap_stack([4, 2, 0, 3, 2, 5]))  # 期望输出:9
print(trap_stack([1, 2, 3, 4]))  # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个元素最多入栈出栈各一次
- **空间复杂度**:O(n) — 栈的最坏情况(单调递减序列)

---

## 🐍 Pythonic 写法

利用 Python 的 zip 和生成器表达式简化解法一:

```python
def trap_pythonic(height: List[int]) -> int:
    """Pythonic 简洁版 - 基于DP思想"""
    if not height:
        return 0

    n = len(height)

    # 使用累积最大值构建 left_max
    left_max = [0] * n
    left_max[0] = height[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], height[i])

    # 使用累积最大值构建 right_max
    right_max = [0] * n
    right_max[-1] = height[-1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], height[i])

    # 一行计算总积水量
    return sum(min(l, r) - h for h, l, r in zip(height, left_max, right_max))
```

这个写法用 `zip` 同时迭代三个数组,用 `sum` 和生成器表达式一行计算结果,更简洁。

> ⚠️ **面试建议**:先写双指针版本(解法二)展示最优解,再提 DP 版本(解法一)说明思路推导过程,最后可以提单调栈作为"不同角度的解法"展示知识广度。
> 面试官更看重你的**思考过程**和**优化能力**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:动态规划 | 解法二:双指针 ⭐ | 解法三:单调栈 |
|------|--------------|--------------|--------------|
| 时间复杂度 | O(n) | O(n) | O(n) |
| 空间复杂度 | O(n) | O(1) | O(n) |
| 代码难度 | 简单 | 中等 | 较难 |
| 面试推荐 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| 适用场景 | 初学者理解思路 | 面试首选最优解 | 展示算法广度 |

**面试建议**:
1. 先说暴力法思路(O(n²)),建立问题理解
2. 提出DP优化(解法一),说明预处理的想法
3. 进一步优化到双指针(解法二),强调空间优化
4. 如果还有时间,可以提单调栈作为"不同维度的解法"

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下接雨水问题。

**你**:(审题30秒)好的,这道题要求计算柱状图下雨后能接多少水。让我先想一下...

我的第一个想法是暴力法:对于每个位置 i,向左扫描找最大高度,向右扫描找最大高度,然后该位置能接的水就是 min(左最大, 右最大) - height[i]。时间复杂度是 O(n²)。

不过我们可以用动态规划优化:提前用两个数组预处理出每个位置的左右最大值,这样时间降到 O(n),空间是 O(n)。

进一步优化,可以用双指针从两端向中间移动,动态维护左右最大值,核心思想是:水位由较矮的一端决定,所以每次移动较矮的指针并计算积水。这样空间降到 O(1)。

**面试官**:很好,请写一下双指针的代码。

**你**:(边写边说)
```python
def trap(height):
    if not height or len(height) < 3:
        return 0

    left, right = 0, len(height) - 1
    left_max, right_max = 0, 0
    total = 0

    while left < right:
        left_max = max(left_max, height[left])
        right_max = max(right_max, height[right])

        # 关键:谁小移动谁
        if left_max < right_max:
            total += left_max - height[left]
            left += 1
        else:
            total += right_max - height[right]
            right -= 1

    return total
```

核心逻辑是:如果 left_max < right_max,说明左边是短板,那么左指针位置的水位就是 left_max(右边肯定有更高的),直接计算积水并移动左指针。反之亦然。

**面试官**:测试一下?

**你**:用示例 [0,1,0,2,1,0,1,3,2,1,2,1] 走一遍:
- left=0, right=11, left_max=0, right_max=1
- height[0]=0 < height[11]=1,移动left,积水=0
- left=1, left_max=1,积水=0
- left=2, left_max=1,积水=1-0=1 ✓
- ...持续移动,最终得到 6 ✓

再测一个边界情况 [1,2,3,4] 单调递增:
- 由于每次 left_max = height[left],积水始终为0 ✓

**面试官**:不错!还有其他解法吗?

**你**:还有一种单调栈的方法,思路完全不同:它是"横向"计算每一层的积水面积,而不是"竖向"计算每个位置。维护一个单调递减栈,当遇到更高的柱子时就计算形成的凹槽面积。时间O(n),空间O(n)。不过面试中双指针已经是最优了。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么双指针每次移动较矮的一端?" | 因为积水量由 min(左最大,右最大) 决定。如果 left_max < right_max,那么左指针的积水只取决于 left_max(右边一定有更高的柱子兜底),所以可以安全计算并移动左指针。 |
| "能用递归解决吗?" | 可以,但没必要。这道题本质是"每个位置找左右最大值",递归会导致大量重复计算,还不如迭代清晰。 |
| "如果数据量非常大,内存放不下怎么办?" | 双指针解法已经是 O(1) 空间了,如果连输入数组都放不下,可以考虑流式处理:分段读取,但需要处理跨段的积水,会比较复杂。 |
| "单调栈的应用场景?" | 单调栈擅长处理"找下一个更大/更小元素"、"矩形面积"等问题,如 LC 84 柱状图最大矩形、LC 739 每日温度。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:双指针对撞模板
left, right = 0, len(arr) - 1
while left < right:
    # 处理逻辑
    if condition:
        left += 1
    else:
        right -= 1

# 技巧2:单调栈模板(递减栈)
stack = []
for i in range(len(arr)):
    while stack and arr[i] > arr[stack[-1]]:
        idx = stack.pop()
        # 处理逻辑
    stack.append(i)

# 技巧3:zip 同时迭代多个列表
for h, l, r in zip(height, left_max, right_max):
    result += min(l, r) - h
```

### 💡 底层原理(选读)

> **为什么双指针能工作?**
>
> 核心在于"贪心思想":
> - 每个位置的积水量 = min(左最大, 右最大) - 当前高度
> - 如果 left_max < right_max,说明左边是"短板",那么:
>   - 左指针位置的积水只取决于 left_max(右边肯定 ≥ right_max ≥ left_max)
>   - 可以安全计算左指针的积水并移动
> - 反之亦然
>
> **单调栈为什么是横向计算?**
>
> 传统方法是对每根柱子"竖着"算能接多少水(从下往上叠加)。单调栈则是"横着"算:当遇到更高的柱子时,形成一个"U型凹槽",可以一次性计算这一层的矩形积水面积(高×宽)。
>
> **Python 的列表操作复杂度:**
> - `list.append()` 平均 O(1)
> - `list.pop()` 平均 O(1)
> - `max(a, b)` 是 O(1)
> - 所以这些解法的常数因子都很小,实际运行很快

### 算法模式卡片 📐

- **模式名称**:双指针对撞 + 贪心
- **适用条件**:需要同时考虑左右两侧信息,且可以通过移动指针动态维护
- **识别关键词**:"左右两端"、"最大/最小值"、"对称处理"
- **模板代码**:
```python
def two_pointer_greedy(arr):
    left, right = 0, len(arr) - 1
    left_val, right_val = 0, 0
    result = 0

    while left < right:
        left_val = max(left_val, arr[left])
        right_val = max(right_val, arr[right])

        if left_val < right_val:
            result += left_val - arr[left]
            left += 1
        else:
            result += right_val - arr[right]
            right -= 1

    return result
```

### 易错点 ⚠️

1. **忘记处理边界**:
   - 错误:`trap([])` 或 `trap([1])` 导致数组越界
   - 原因:没有检查输入有效性
   - 正确:开头加 `if not height or len(height) < 3: return 0`

2. **双指针移动条件写反**:
   - 错误:`if left_max < right_max: right -= 1`
   - 原因:理解错了"移动较矮的一端"
   - 正确:left_max 小说明左边是短板,应该移动 **left**

3. **单调栈计算宽度错误**:
   - 错误:`width = right - left`
   - 原因:没有减去两个柱子本身的宽度
   - 正确:`width = right - left - 1`

4. **DP 数组初始化错误**:
   - 错误:`left_max[0] = 0`(应该是 `height[0]`)
   - 原因:没理解"包括自己"的含义
   - 正确:第一个位置的左最大就是它自己

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:图像处理中的"填充算法"**
  - 在图像分割中,需要计算封闭区域的面积
  - 类似接雨水,找到边界后填充内部
  - 应用:洪水填充算法(Flood Fill)

- **场景2:建筑设计中的排水系统**
  - 给定屋顶的高度分布,计算雨水积存量
  - 帮助设计师优化排水口位置

- **场景3:股票交易中的"支撑位"分析**
  - 将价格看作柱子高度,寻找"价格凹槽"
  - 判断是否有"接水"的空间,即反弹潜力

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 11. 盛最多水的容器 | Medium | 双指针对撞 | 不能"接水",只能用两根柱子围成矩形 |
| LeetCode 84. 柱状图最大矩形 | Hard | 单调栈 | 用单调栈找每根柱子左右第一个更矮的柱子 |
| LeetCode 85. 最大矩形 | Hard | 单调栈+DP | 每一行看作直方图,复用 LC 84 的解法 |
| LeetCode 407. 接雨水 II | Hard | 优先队列+BFS | 3D 版本的接雨水,从最低的边界开始 BFS |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个整数数组 `height`,现在允许你**移除一个柱子**,问移除后最多能接多少雨水?

示例:
```
输入:height = [3,0,2,0,4]
输出:7
解释:移除 height[0]=3 后,变成 [0,2,0,4],能接 2 单位水
      不移除的话,[3,0,2,0,4] 能接 2+0+2 = 4 单位水
      但如果移除 height[2]=2,变成 [3,0,0,4],能接 0+3+3 = 6 单位水
      最优:移除 height[1]=0,变成 [3,2,0,4],能接 0+1+3 = 4... 不对

      重新思考:不移除 → 接水 4
               移除 height[0]=3 → [0,2,0,4] → 0+0+2 = 2
               移除 height[2]=2 → [3,0,0,4] → 0+3+3 = 6 ✓

      答案是移除 height[2],接 6 单位水(题目示例有误,应该是6)
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

枚举移除每个位置,对剩余数组计算接雨水量,取最大值。优化:只需要考虑移除"高柱子"可能会增加积水。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def max_water_remove_one(height: List[int]) -> int:
    """
    变体:移除一个柱子后的最大接雨水量
    思路:枚举移除每个位置,计算剩余数组的接雨水量
    """
    def trap_water(arr):
        # 使用双指针计算接雨水
        if len(arr) < 3:
            return 0
        left, right = 0, len(arr) - 1
        left_max, right_max = 0, 0
        water = 0
        while left < right:
            left_max = max(left_max, arr[left])
            right_max = max(right_max, arr[right])
            if left_max < right_max:
                water += left_max - arr[left]
                left += 1
            else:
                water += right_max - arr[right]
                right -= 1
        return water

    max_water = 0
    for i in range(len(height)):
        # 移除第 i 个柱子
        new_height = height[:i] + height[i+1:]
        max_water = max(max_water, trap_water(new_height))

    return max_water


# 测试
print(max_water_remove_one([3, 0, 2, 0, 4]))  # 6
```

核心思路:暴力枚举移除每个位置,对剩余数组调用接雨水函数,取最大值。时间复杂度 O(n²)。

**优化思路**(进阶):
- 可以预处理出移除每个位置前后的接雨水量变化
- 关键观察:移除一个柱子可能会让它左右的"凹槽"连通,形成更大的积水
- 时间可以优化到 O(n)

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
