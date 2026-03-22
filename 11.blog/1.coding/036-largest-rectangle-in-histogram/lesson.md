# 📖 第36课:柱状图最大矩形

> **模块**:栈与队列 | **难度**:Hard ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/largest-rectangle-in-histogram/
> **前置知识**:第33课(有效的括号)、第34课(最小栈)、第35课(每日温度)
> **预计学习时间**:35分钟

---

## 🎯 题目描述

给定一个整数数组 `heights`,表示直方图中每个柱子的高度。每个柱子的宽度为 1,要求找出这个直方图中能够勾勒出的最大矩形面积。

**示例 1:**
```
输入:heights = [2,1,5,6,2,3]
输出:10
解释:最大矩形是从下标 2 到 3(即高度 [5,6]),高度为 5,宽度为 2,面积 = 5 × 2 = 10
```

**示例 2:**
```
输入:heights = [2,4]
输出:4
解释:可以取高度为 2,宽度为 2 的矩形,面积 = 2 × 2 = 4
```

**约束条件:**
- 1 ≤ heights.length ≤ 10^5
- 0 ≤ heights[i] ≤ 10^4

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单柱子 | [5] | 5 | 基本功能 |
| 递增序列 | [1,2,3,4,5] | 9 | 最大矩形可能在左侧 |
| 递减序列 | [5,4,3,2,1] | 9 | 最大矩形可能在右侧 |
| 全相同 | [3,3,3,3] | 12 | 最优解为全部柱子 |
| 含零 | [2,0,2] | 2 | 零高度打断连续性 |
| 大规模 | n=10^5 | — | 必须 O(n) 复杂度 |

---

## 💡 思路引导

### 生活化比喻

> 想象你是一名城市规划师,要在一排不同高度的建筑之间找一块最大的矩形空地来建广场。
>
> 🐌 **笨办法**:拿着卷尺,从每栋楼出发,一个个试:"以这栋楼高度为标准,能向左右延伸多远?" 这样需要尝试每一栋楼,每栋楼又要向两边扫描,耗时 O(n²)。
>
> 🚀 **聪明办法**:你拿着一张纸条,从左往右走。遇到更矮的楼时,你就知道"之前那些高楼的地盘到此为止了",立刻可以计算出它们能围出的最大矩形。这样只需走一遍,耗时 O(n)!

### 关键洞察

**对于每个柱子,如果我们能快速知道它向左右两边能延伸到哪里,就能算出以它为高度的最大矩形。**

核心问题转化为:**如何找到每个柱子左右两侧第一个比它矮的柱子?** → 单调栈模式!

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:整数数组 `heights`,每个元素代表柱子高度
- **输出**:整数,表示最大矩形面积
- **限制**:柱子数量最多 10^5,意味着必须在 O(n) 或 O(n log n) 内解决

### Step 2:先想笨办法(暴力法)

最直接的想法:枚举所有可能的矩形。
- 对于每个柱子 `i`,尝试以它的高度为矩形高度
- 向左右两边扩展,找到能延伸的最大宽度
- 计算面积并更新最大值

时间复杂度:O(n²) — 对于每个柱子 i,需要向两边扫描寻找边界
瓶颈在哪:**对每个柱子都要线性扫描左右边界**

### Step 3:瓶颈分析 → 优化方向

暴力法中,对于每个柱子,我们重复计算了左右边界:
- 核心问题:**如何快速找到左侧第一个比当前柱子矮的位置?**
- 核心问题:**如何快速找到右侧第一个比当前柱子矮的位置?**

这不正是单调栈擅长的"下一个更小元素"问题吗?

### Step 4:选择武器

- 选用:**单调递增栈**
- 理由:
  1. 栈中维护递增序列,遇到更矮柱子时,栈顶的高柱子就找到了右边界
  2. 栈中前一个元素就是左边界
  3. 一次遍历即可完成,时间 O(n)

> 🔑 **模式识别提示**:当题目要求"每个元素左/右第一个更小/更大的元素",优先考虑**单调栈**

---

## 🔑 解法一:暴力枚举(教学用)

### 思路

对于每个柱子 i,以它的高度为矩形高度,向左右两边扩展,直到遇到比它矮的柱子为止。计算这个矩形的面积,取所有矩形的最大值。

### 图解过程

```
示例:heights = [2,1,5,6,2,3]

枚举柱子 0(高度 2):
  向左:无柱子
  向右:遇到 1(高度 < 2),停止
  宽度 = 1,面积 = 2 × 1 = 2

枚举柱子 1(高度 1):
  向左:无更小
  向右:无更小,可延伸到末尾
  宽度 = 6,面积 = 1 × 6 = 6

枚举柱子 2(高度 5):
  向左:遇到 1(高度 < 5),停止
  向右:高度 6 可以,但遇到 2(高度 < 5),停止
  宽度 = 2(从索引 2 到 3),面积 = 5 × 2 = 10  ← 最大

枚举柱子 3(高度 6):
  向左:高度 5 可以,但遇到 1(高度 < 6),停止
  向右:遇到 2(高度 < 6),停止
  宽度 = 1,面积 = 6 × 1 = 6

... 依次类推

最大面积 = 10
```

### Python代码

```python
from typing import List


def largest_rectangle_brute_force(heights: List[int]) -> int:
    """
    解法一:暴力枚举
    思路:对每个柱子向左右扩展找边界
    """
    n = len(heights)
    max_area = 0

    for i in range(n):
        h = heights[i]

        # 找左边界:第一个比 h 小的位置
        left = i
        while left > 0 and heights[left - 1] >= h:
            left -= 1

        # 找右边界:第一个比 h 小的位置
        right = i
        while right < n - 1 and heights[right + 1] >= h:
            right += 1

        # 计算以 heights[i] 为高度的矩形面积
        width = right - left + 1
        area = h * width
        max_area = max(max_area, area)

    return max_area


# ✅ 测试
print(largest_rectangle_brute_force([2, 1, 5, 6, 2, 3]))  # 期望输出:10
print(largest_rectangle_brute_force([2, 4]))              # 期望输出:4
print(largest_rectangle_brute_force([5, 4, 3, 2, 1]))     # 期望输出:9
```

### 复杂度分析

- **时间复杂度**:O(n²) — 对于每个柱子 i,向左右扫描最坏需要 O(n) 时间
  - 具体地说:如果输入规模 n = 1000,最坏需要约 1000 × 1000 = 1,000,000 次操作
- **空间复杂度**:O(1) — 只用了常量级变量

### 优缺点

- ✅ 思路直观,容易理解
- ❌ 时间复杂度过高,对于 n = 10^5 的数据会超时(需要 10^10 次操作)

---

## ⚡ 解法二:预处理左右边界(优化)

### 优化思路

暴力法的瓶颈在于重复计算左右边界。我们可以预处理出两个数组:
- `left[i]`:柱子 i 左侧第一个比它矮的位置
- `right[i]`:柱子 i 右侧第一个比它矮的位置

然后对每个柱子,直接通过 `left[i]` 和 `right[i]` 计算宽度,避免重复扫描。

> 💡 **关键想法**:用空间换时间,预处理边界数组,将单次查询从 O(n) 降为 O(1)

### 图解过程

```
heights = [2,1,5,6,2,3]
索引      0 1 2 3 4 5

预处理 left[i](左侧第一个更小元素的索引):
  left[0] = -1  (无更小)
  left[1] = -1  (无更小)
  left[2] = 1   (索引 1 的高度 1 < 5)
  left[3] = 2   (索引 2 的高度 5 < 6,但需要继续向左找,最终 left[2]=1 导致 left[3]=1... 实际上应该是 1)
  left[4] = 1   (索引 1 的高度 1 < 2)
  left[5] = 4   (索引 4 的高度 2 < 3)

预处理 right[i](右侧第一个更小元素的索引):
  right[0] = 1  (索引 1 的高度 1 < 2)
  right[1] = 6  (无更小,设为 n)
  right[2] = 4  (索引 4 的高度 2 < 5)
  right[3] = 4  (索引 4 的高度 2 < 6)
  right[4] = 6  (无更小)
  right[5] = 6  (无更小)

计算每个柱子的最大面积:
  i=2:width = right[2] - left[2] - 1 = 4 - 1 - 1 = 2
      area = 5 × 2 = 10  ← 最大
```

### Python代码

```python
def largest_rectangle_preprocess(heights: List[int]) -> int:
    """
    解法二:预处理左右边界
    思路:先用两次遍历计算每个柱子的左右边界,再计算面积
    """
    n = len(heights)
    if n == 0:
        return 0

    # 预处理左边界:left[i] = 左侧第一个比 heights[i] 小的索引
    left = [-1] * n
    for i in range(1, n):
        p = i - 1
        while p >= 0 and heights[p] >= heights[i]:
            p = left[p]  # 跳跃优化:利用已计算的结果
        left[i] = p

    # 预处理右边界:right[i] = 右侧第一个比 heights[i] 小的索引
    right = [n] * n
    for i in range(n - 2, -1, -1):
        p = i + 1
        while p < n and heights[p] >= heights[i]:
            p = right[p]  # 跳跃优化
        right[i] = p

    # 计算最大面积
    max_area = 0
    for i in range(n):
        width = right[i] - left[i] - 1
        area = heights[i] * width
        max_area = max(max_area, area)

    return max_area


# ✅ 测试
print(largest_rectangle_preprocess([2, 1, 5, 6, 2, 3]))  # 期望输出:10
print(largest_rectangle_preprocess([2, 4]))              # 期望输出:4
print(largest_rectangle_preprocess([5, 4, 3, 2, 1]))     # 期望输出:9
```

### 复杂度分析

- **时间复杂度**:O(n) — 每个元素最多被访问 2 次(一次主循环,一次跳跃)
- **空间复杂度**:O(n) — 需要两个长度为 n 的辅助数组

---

## 🏆 解法三:单调栈(最优解)

### 优化思路

解法二虽然已经优化到 O(n),但代码较长,需要三次遍历。单调栈可以在**一次遍历**中同时计算左右边界并求出最大面积。

核心思想:
1. 维护一个单调递增栈,栈中存储柱子的索引
2. 遍历每个柱子时:
   - 如果当前柱子高度 ≥ 栈顶柱子,入栈(保持递增)
   - 如果当前柱子高度 < 栈顶柱子,说明栈顶柱子找到了右边界,出栈并计算面积
3. 栈中前一个元素就是左边界

> 💡 **关键想法**:单调栈不仅能找边界,还能在找到边界的同时立即计算面积

### 图解过程

```
heights = [2,1,5,6,2,3]
在末尾添加哨兵 0:[2,1,5,6,2,3,0]

初始:栈 = [],max_area = 0

i=0,h=2:
  栈空,入栈 → 栈=[0]

i=1,h=1:
  1 < heights[0]=2,栈顶出栈
    弹出索引 0,高度 2
    右边界 = 1,左边界 = -1(栈空)
    宽度 = 1 - (-1) - 1 = 1
    面积 = 2 × 1 = 2
  入栈 1 → 栈=[1]

i=2,h=5:
  5 ≥ heights[1]=1,入栈 → 栈=[1,2]

i=3,h=6:
  6 ≥ heights[2]=5,入栈 → 栈=[1,2,3]

i=4,h=2:
  2 < heights[3]=6,弹出索引 3
    高度 6,右边界 4,左边界 2
    宽度 = 4 - 2 - 1 = 1
    面积 = 6 × 1 = 6
  2 < heights[2]=5,弹出索引 2
    高度 5,右边界 4,左边界 1
    宽度 = 4 - 1 - 1 = 2
    面积 = 5 × 2 = 10  ← 更新最大值
  2 ≥ heights[1]=1,入栈 → 栈=[1,4]

i=5,h=3:
  3 ≥ heights[4]=2,入栈 → 栈=[1,4,5]

i=6,h=0(哨兵):
  依次弹出所有元素:
    弹出 5:3 × (6-4-1) = 3
    弹出 4:2 × (6-1-1) = 8
    弹出 1:1 × (6-(-1)-1) = 6

最大面积 = 10
```

### Python代码

```python
def largest_rectangle_area(heights: List[int]) -> int:
    """
    解法三:单调栈(最优解)
    思路:维护单调递增栈,遇到更小元素时计算面积
    """
    # 在末尾添加哨兵 0,确保所有柱子都能出栈
    heights = heights + [0]
    stack = []  # 存储索引
    max_area = 0

    for i, h in enumerate(heights):
        # 当前高度小于栈顶,说明栈顶柱子的右边界确定了
        while stack and h < heights[stack[-1]]:
            height_idx = stack.pop()  # 弹出栈顶
            height = heights[height_idx]

            # 左边界:栈中前一个元素(如果栈空,说明左边全部 ≥ 当前高度)
            left = stack[-1] if stack else -1
            # 右边界:当前索引 i
            width = i - left - 1

            area = height * width
            max_area = max(max_area, area)

        stack.append(i)

    return max_area


# ✅ 测试
print(largest_rectangle_area([2, 1, 5, 6, 2, 3]))  # 期望输出:10
print(largest_rectangle_area([2, 4]))              # 期望输出:4
print(largest_rectangle_area([5, 4, 3, 2, 1]))     # 期望输出:9
print(largest_rectangle_area([1]))                 # 期望输出:1
print(largest_rectangle_area([3, 3, 3, 3]))        # 期望输出:12
```

### 复杂度分析

- **时间复杂度**:O(n) — 每个元素最多入栈一次、出栈一次,总共 2n 次操作
  - 具体地说:如果 n = 100,000,大约需要 200,000 次操作
- **空间复杂度**:O(n) — 栈最多存储 n 个元素

### 为什么是最优解

1. **时间最优**:O(n) 已经是理论下限(至少要遍历一遍所有柱子)
2. **一次遍历**:比解法二的三次遍历更简洁
3. **代码清晰**:单调栈模式代码结构标准,面试易写对
4. **实际性能**:常数因子小,比暴力法快 5000 倍以上

---

## 🐍 Pythonic 写法

利用 Python 的枚举和列表推导式,可以让代码更简洁:

```python
def largest_rectangle_pythonic(heights: List[int]) -> int:
    """单调栈的 Pythonic 简化写法"""
    heights = heights + [0]
    stack = []
    max_area = 0

    for i, h in enumerate(heights):
        while stack and h < heights[stack[-1]]:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area
```

**改进点**:
- 用 `enumerate` 同时获取索引和值
- 简化宽度计算:`width = i if not stack else i - stack[-1] - 1`

> ⚠️ **面试建议**:先写标准版本展示思路,通过测试后再提简洁写法展示 Python 功底。

---

## 📊 解法对比

| 维度 | 解法一:暴力枚举 | 解法二:预处理边界 | 🏆 解法三:单调栈(最优) |
|------|--------------|-----------------|---------------------|
| 时间复杂度 | O(n²) | O(n) | **O(n)** ← 时间最优 |
| 空间复杂度 | O(1) | O(n) | **O(n)** ← 可接受 |
| 代码难度 | 简单 | 中等 | 中等 |
| 遍历次数 | n 次(每次 O(n)) | 3 次 | **1 次** ← 最简洁 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 只适合小规模数据 | 教学理解用 | **面试标准答案** |

**为什么单调栈是最优解**:
1. 时间复杂度 O(n) 已经达到理论下限
2. 一次遍历比预处理方法更高效
3. 单调栈是处理"下一个更小/更大元素"的标准模式
4. 代码结构清晰,面试中容易写对且不易出错

**面试建议**:
1. 先用 30 秒口述暴力法思路(O(n²)),表明你能想到基本解法
2. 立即优化到🏆单调栈解法,重点讲解核心思想
3. 强调单调栈的三个关键点:
   - 栈中存索引而非值
   - 遇到更小元素时出栈计算
   - 左边界是栈中前一个元素
4. 手动测试边界用例(递增、递减、全相同),展示对解法的深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请解决柱状图最大矩形问题。

**你**:(审题 30 秒)好的,这道题要求在一个直方图中找出最大矩形面积。让我先想一下...

我的第一个想法是暴力法:对每个柱子,向左右扩展找边界,计算以它为高度的矩形面积。时间复杂度是 O(n²),对于 10^5 的数据量会超时。

优化思路:这个问题本质是"对每个柱子,找左右两侧第一个比它矮的位置"。这正是单调栈擅长的场景!我可以用单调递增栈,在 O(n) 时间内解决。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def largest_rectangle_area(heights):
    # 添加哨兵 0,确保所有元素都能出栈
    heights = heights + [0]
    stack = []  # 单调递增栈,存索引
    max_area = 0

    for i, h in enumerate(heights):
        # 当前高度 < 栈顶,栈顶柱子的右边界确定
        while stack and h < heights[stack[-1]]:
            height_idx = stack.pop()
            height = heights[height_idx]
            # 左边界是栈中前一个元素
            left = stack[-1] if stack else -1
            width = i - left - 1
            max_area = max(max_area, height * width)
        stack.append(i)

    return max_area
```

**面试官**:测试一下?

**你**:用示例 `[2,1,5,6,2,3]` 走一遍:
- i=2 时入栈索引 2(高度 5)和 3(高度 6)
- i=4 时高度 2 < 栈顶 6,弹出 3,计算 6×1=6
- 继续弹出 2,计算 5×2=10(从索引 2 到 3)
- 最终返回 10,正确!

再测一个边界:全相同 `[3,3,3,3]` → 最终会计算 3×4=12,符合预期。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么要加哨兵 0?" | 确保所有柱子都能出栈计算面积。否则遍历结束时栈中剩余元素无法触发计算。 |
| "能不能不加哨兵?" | 可以,但需要遍历结束后再处理栈中剩余元素,代码会复杂一些。加哨兵是更优雅的做法。 |
| "空间能不能 O(1)?" | 理论上无法做到。单调栈需要 O(n) 空间,这是这类问题的固有开销。暴力法虽然 O(1) 空间,但 O(n²) 时间无法接受。 |
| "如果柱子宽度不为 1 呢?" | 输入增加一个 widths 数组,计算面积时改为 `height * sum(widths[left+1:i])`。 |
| "实际工程中怎么用?" | 图像处理中的最大矩形检测、仓库货物摆放优化、数据可视化中的柱状图分析等。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:哨兵技巧 — 简化边界处理
heights = heights + [0]  # 末尾加 0 确保所有元素出栈

# 技巧2:栈的非空判断
left = stack[-1] if stack else -1  # 栈空时左边界为 -1

# 技巧3:enumerate 同时获取索引和值
for i, h in enumerate(heights):
    # i 是索引,h 是值

# 技巧4:单行条件表达式
width = i if not stack else i - stack[-1] - 1
```

### 💡 底层原理(选读)

> **为什么单调栈能高效找到左右边界?**
>
> 单调栈维护的是一个"未来可能有用"的候选集合。当遇到更小元素时:
> 1. 栈顶元素的右边界确定(就是当前元素)
> 2. 栈中前一个元素就是左边界(因为栈单调递增,前一个一定比当前小)
> 3. 出栈后,栈中剩余元素仍保持单调性,可继续使用
>
> **为什么每个元素只会入栈/出栈一次?**
> - 每个元素遍历到时入栈一次
> - 只有在遇到更小元素时才出栈,出栈后不会再入栈
> - 所以总操作数 = 2n,时间复杂度 O(n)

### 算法模式卡片 📐

- **模式名称**:单调栈
- **适用条件**:
  - 需要找每个元素左/右第一个更大/更小的元素
  - 需要在线性时间内处理柱状图、温度等"相邻关系"问题
- **识别关键词**:
  - "下一个更大/更小"
  - "柱状图""直方图"
  - "每个元素向左/右能延伸多远"
- **模板代码**:
```python
def monotonic_stack_template(arr):
    stack = []  # 存索引
    result = []

    for i, val in enumerate(arr):
        # 维护单调性:递增栈用 <,递减栈用 >
        while stack and val < arr[stack[-1]]:
            idx = stack.pop()
            # 在这里处理弹出元素(已找到右边界)
            left = stack[-1] if stack else -1
            # 计算区间 [left+1, i-1]
        stack.append(i)

    # 处理栈中剩余元素(如果需要)
    return result
```

### 易错点 ⚠️

1. **栈中存值还是存索引?**
   - ❌ 错误:存值无法计算宽度
   - ✅ 正确:存索引,计算面积时 `width = right - left - 1`

2. **左边界怎么算?**
   - ❌ 错误:`left = stack[-1]`(忘记检查栈空)
   - ✅ 正确:`left = stack[-1] if stack else -1`

3. **是否需要哨兵?**
   - ❌ 错误:不加哨兵,导致栈中剩余元素未处理
   - ✅ 正确:末尾加 0 哨兵,确保所有元素出栈

4. **单调栈的方向?**
   - ❌ 错误:这题要用单调递减栈
   - ✅ 正确:要用单调递增栈,因为要找"更小"的元素作为边界

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:图像处理** — 在二值图像中检测最大矩形区域(如 OCR 预处理)
- **场景2:仓库优化** — 在不同高度的货架中找最大可用存储空间
- **场景3:数据可视化** — 柱状图中自动标注最大矩形辅助分析
- **场景4:建筑设计** — 在不规则建筑群中规划最大矩形广场

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 85. 最大矩形 | Hard | 单调栈+逐行转化 | 将二维矩阵每一行转化为直方图,复用本题解法 |
| LeetCode 42. 接雨水 | Hard | 单调栈/双指针 | 与本题类似,找左右边界,但计算方式不同 |
| LeetCode 739. 每日温度 | Medium | 单调栈 | 单调栈基础题,找"下一个更大元素" |
| LeetCode 496. 下一个更大元素 I | Easy | 单调栈 | 单调栈入门,理解栈的单调性维护 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想 5 分钟!

**题目**:给定一个 0-1 矩阵,找出其中最大的矩形(全 1 区域)。

例如:
```
matrix = [
  [1,0,1,0,0],
  [1,0,1,1,1],
  [1,1,1,1,1],
  [1,0,0,1,0]
]
最大矩形面积 = 6(从第 2 行到第 3 行,列 2-4)
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

将每一行看作直方图的底,高度为"从当前行往上连续 1 的个数"。对每一行应用本课的单调栈解法。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def maximal_rectangle(matrix: List[List[str]]) -> int:
    """
    思路:逐行转化为柱状图最大矩形
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for row in matrix:
        # 更新当前行的直方图高度
        for j in range(cols):
            if row[j] == '1':
                heights[j] += 1
            else:
                heights[j] = 0

        # 对当前直方图应用单调栈求最大矩形
        max_area = max(max_area, largest_rectangle_area(heights))

    return max_area
```

**核心思路**:
1. 维护一个 heights 数组,记录每列从当前行往上连续 1 的个数
2. 遍历每一行,更新 heights,然后调用本课的 `largest_rectangle_area` 函数
3. 时间复杂度 O(m×n),m 是行数,n 是列数

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
