> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第37课:最大矩形

> **模块**:栈与队列 | **难度**:Hard ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/maximal-rectangle/
> **前置知识**:第36课 柱状图最大矩形(LC 84)、单调栈
> **预计学习时间**:40分钟

---

## 🎯 题目描述

给定一个只包含 `'0'` 和 `'1'` 的二维矩阵,找出其中全为 `'1'` 的最大矩形面积。

**示例:**
```
输入:matrix = [
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出:6
解释:最大矩形如下图所示,面积为 6
```

**约束条件:**
- `rows == matrix.length`
- `cols == matrix[0].length`
- `1 <= rows, cols <= 200`
- `matrix[i][j]` 为 `'0'` 或 `'1'`

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单行单列 | `[["1"]]` | 1 | 最小矩形 |
| 全零矩阵 | `[["0","0"],["0","0"]]` | 0 | 无有效矩形 |
| 全一矩阵 | `[["1","1"],["1","1"]]` | 4 | 整体就是矩形 |
| 单行 | `[["1","1","0","1"]]` | 2 | 退化为一维问题 |
| L形矩阵 | `[["1","1"],["1","0"]]` | 2 | 非规则形状 |
| 大规模 | 200×200 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在俄罗斯方块游戏中,每一行都是一层已经堆叠好的方块。
>
> 🐌 **笨办法**:枚举所有可能的矩形左上角和右下角,逐一验证是否全为'1'。这就像盲目尝试每一种可能的方块组合,效率极低(O(n²m²))。
>
> 🚀 **聪明办法**:把每一行当作"地面",向上累计连续'1'的高度,把二维问题转化为第84题"柱状图最大矩形"!每一行都运行一次第84题的算法,取所有行中的最大值。

### 关键洞察
**核心突破点:将二维矩阵问题降维到一维柱状图问题,逐行计算高度数组,复用第84题的单调栈解法。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二维字符矩阵 `matrix`,元素为 `'0'` 或 `'1'`
- **输出**:最大矩形面积(整数)
- **限制**:必须全为'1'的矩形区域,不能包含'0'

### Step 2:先想笨办法(暴力法)
枚举所有可能的矩形:固定左上角 (r1, c1) 和右下角 (r2, c2),检查区域内是否全为'1'。
- 时间复杂度:O(rows² × cols² × rows × cols) = O(n³m³)
- 瓶颈在哪:枚举矩形 O(n²m²),验证每个矩形 O(nm),太慢了!

### Step 3:瓶颈分析 → 优化方向
暴力法的问题在于"没有利用问题的结构特性"。观察:
- 核心问题:矩形可以看作"一排连续的柱子",每根柱子的高度由连续'1'决定
- 优化思路:能否把二维问题转化为多个一维问题?→ **逐行处理,将每行视为"地面",向上累计高度**

### Step 4:选择武器
- 选用:**逐行转化 + 单调栈(第84题算法)**
- 理由:
  1. 对每一行,计算从该行向上的连续'1'高度,得到一个高度数组
  2. 每个高度数组都是一个"柱状图",直接套用第84题的解法(单调栈 O(cols))
  3. 遍历所有行 O(rows),总复杂度 O(rows × cols)

> 🔑 **模式识别提示**:当题目要求"二维矩阵中的最大矩形",优先考虑"逐行转化为柱状图"模式(第84题的推广)

---

## 🔑 解法一:动态规划累计高度 + 暴力枚举宽度(朴素优化)

### 思路
对每一行,维护一个高度数组 `heights`,表示从当前行向上连续'1'的个数。然后对每一行的高度数组,用双重循环枚举所有可能的矩形宽度,计算面积。

### 图解过程

```
输入矩阵:
  ["1","0","1","0","0"]
  ["1","0","1","1","1"]
  ["1","1","1","1","1"]
  ["1","0","0","1","0"]

Step 1:处理第0行(row=0)
heights = [1, 0, 1, 0, 0]
  ↓ (第0行只有自己,高度都是1或0)

Step 2:处理第1行(row=1)
原 heights = [1, 0, 1, 0, 0]
新一行     = [1, 0, 1, 1, 1]
更新后:
heights[0] = 1+1 = 2 (列0连续两个'1')
heights[1] = 0    (遇到'0'清零)
heights[2] = 1+1 = 2
heights[3] = 0+1 = 1
heights[4] = 0+1 = 1
→ heights = [2, 0, 2, 1, 1]

Step 3:处理第2行(row=2)
原 heights = [2, 0, 2, 1, 1]
新一行     = [1, 1, 1, 1, 1]
更新后:
heights = [3, 1, 3, 2, 2]
  ↑     ↑
  |     +----- 列1从'0'变'1',重新计数为1
  +----------- 列0连续三个'1'

对 heights=[3,1,3,2,2] 枚举矩形:
  - 从索引0到4,最小高度为1,宽度5,面积=1×5=5
  - 从索引2到4,最小高度为2,宽度3,面积=2×3=6 ← 最大!

Step 4:处理第3行(row=3)
heights = [4, 0, 0, 3, 0]
  最大面积不超过之前的6
```

### Python代码

```python
from typing import List


def maximalRectangle(matrix: List[List[str]]) -> int:
    """
    解法一:逐行累计高度 + 暴力枚举宽度
    思路:对每行维护向上连续1的高度,然后双重循环枚举矩形
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols  # 高度数组
    max_area = 0

    for row in range(rows):
        # 更新高度数组
        for col in range(cols):
            if matrix[row][col] == '1':
                heights[col] += 1
            else:
                heights[col] = 0  # 遇到'0'清零

        # 对当前高度数组,暴力枚举所有矩形
        for i in range(cols):
            if heights[i] == 0:
                continue
            min_height = heights[i]
            for j in range(i, cols):
                if heights[j] == 0:
                    break
                min_height = min(min_height, heights[j])
                width = j - i + 1
                area = min_height * width
                max_area = max(max_area, area)

    return max_area


# ✅ 测试
print(maximalRectangle([
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]))  # 期望输出:6

print(maximalRectangle([["0"]]))  # 期望输出:0
print(maximalRectangle([["1"]]))  # 期望输出:1
print(maximalRectangle([["1","1"],["1","1"]]))  # 期望输出:4
```

### 复杂度分析
- **时间复杂度**:O(rows × cols²) — 外层遍历rows行,每行更新heights O(cols),然后双重循环枚举矩形 O(cols²)
  - 具体地说:如果输入规模为 100×100,大约需要 100×100²=1,000,000 次操作
- **空间复杂度**:O(cols) — heights 数组

### 优缺点
- ✅ 思路清晰,容易理解和实现
- ✅ 相比暴力法已经有显著优化(从 O(n³m³) 降到 O(nm²))
- ❌ 对每行的高度数组仍在暴力枚举,当 cols 很大时效率低

---

## 🏆 解法二:逐行转化 + 单调栈(最优解)

### 优化思路
解法一的瓶颈在于对每个高度数组用 O(cols²) 枚举矩形。实际上,第84题"柱状图最大矩形"已经给出了 **O(cols) 的单调栈解法**!直接套用即可。

> 💡 **关键想法**:每一行的高度数组 `heights` 就是一个柱状图,单调栈能在 O(cols) 时间内求出最大矩形面积。

### 图解过程

```
同样的输入矩阵,但这次用单调栈处理每个 heights:

Step 1:row=0, heights=[1,0,1,0,0]
单调栈处理:
  栈维护递增序列,遇到更小元素时弹出计算面积
  → 最大面积 = 1

Step 2:row=1, heights=[2,0,2,1,1]
单调栈处理:
  [0] heights[0]=2 入栈
  [0,1] heights[1]=0 触发弹出,计算索引0的矩形:高2×宽1=2
  [1] heights[1]=0 入栈
  [1,2] heights[2]=2 入栈
  [1,2,3] heights[3]=1 触发弹出索引2:高2×宽1=2
  ...
  → 最大面积 = 2

Step 3:row=2, heights=[3,1,3,2,2]
单调栈处理:
  [0] 入栈 3
  [1] 入栈 1(弹出0,计算 3×1=3)
  [1,2] 入栈 3
  [1,2,3] 入栈 2(弹出2,计算 3×1=3)
  [1,3,4] 入栈 2
  最后清空栈:
    - 索引4: 高2,宽度到栈顶索引3后,宽=4-3=1,面积2×1=2
    - 索引3: 高2,宽度到栈顶索引1后,宽=4-1=3,面积2×3=6 ← 最大!
  → 最大面积 = 6

Step 4:row=3, heights=[4,0,0,3,0]
  → 最大面积 = 4

全局最大面积 = 6
```

**单调栈核心逻辑**(第84题):
```
维护递增栈,遇到更小元素时:
  1. 弹出栈顶idx,此时heights[idx]是矩形高度
  2. 宽度 = 当前索引 - 新栈顶索引 - 1
  3. 面积 = 高度 × 宽度
```

### Python代码

```python
def maximalRectangle_optimal(matrix: List[List[str]]) -> int:
    """
    🏆 解法二:逐行转化 + 单调栈(最优解)
    思路:对每行的高度数组,用单调栈 O(cols) 求柱状图最大矩形
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for row in range(rows):
        # 更新高度数组
        for col in range(cols):
            if matrix[row][col] == '1':
                heights[col] += 1
            else:
                heights[col] = 0

        # 用单调栈计算当前柱状图的最大矩形(第84题算法)
        max_area = max(max_area, largestRectangleArea(heights))

    return max_area


def largestRectangleArea(heights: List[int]) -> int:
    """
    第84题:柱状图最大矩形的单调栈解法
    """
    stack = []  # 单调递增栈,存储下标
    max_area = 0
    heights.append(0)  # 哨兵,确保最后所有柱子都被处理

    for i in range(len(heights)):
        # 当前柱子比栈顶矮,弹出栈顶计算面积
        while stack and heights[i] < heights[stack[-1]]:
            h_index = stack.pop()
            h = heights[h_index]
            # 宽度 = 当前索引 - 新栈顶索引 - 1
            w = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, h * w)
        stack.append(i)

    heights.pop()  # 恢复原数组
    return max_area


# ✅ 测试
print(maximalRectangle_optimal([
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]))  # 期望输出:6

print(maximalRectangle_optimal([["0"]]))  # 期望输出:0
print(maximalRectangle_optimal([["1"]]))  # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(rows × cols) — 遍历 rows 行,每行更新 heights O(cols) + 单调栈处理 O(cols)
  - 具体地说:如果输入规模为 200×200,只需要 200×200 = 40,000 次操作,是解法一的 1000 倍提升!
  - **这已经是理论最优**:必须至少扫描一遍所有元素 O(nm),单调栈没有冗余计算
- **空间复杂度**:O(cols) — heights 数组 + 单调栈

---

## ⚡ 解法三:动态规划(DP记录左右边界,可选)

### 优化思路
不使用栈,用 DP 预处理每个位置的"向左最远能延伸到哪"和"向右最远能延伸到哪",直接计算矩形面积。

> 💡 **关键想法**:对每个位置 (row, col),维护三个数组:
> - `heights[col]`:向上连续'1'的高度
> - `left[col]`:当前高度下,向左最远的边界(闭区间)
> - `right[col]`:当前高度下,向右最远的边界(闭区间)
> - 面积 = `heights[col] × (right[col] - left[col] + 1)`

### Python代码

```python
def maximalRectangle_dp(matrix: List[List[str]]) -> int:
    """
    解法三:动态规划记录左右边界
    思路:维护heights、left、right三个数组,直接计算面积
    """
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    left = [0] * cols  # 左边界(闭区间)
    right = [cols - 1] * cols  # 右边界(闭区间)
    max_area = 0

    for row in range(rows):
        # 更新 heights 和 left(从左到右)
        cur_left = 0  # 当前行'1'的左边界
        for col in range(cols):
            if matrix[row][col] == '1':
                heights[col] += 1
                left[col] = max(left[col], cur_left)  # 取历史和当前的最右边界
            else:
                heights[col] = 0
                left[col] = 0  # 重置
                cur_left = col + 1  # 下一个'1'的左边界至少从这里开始

        # 更新 right(从右到左)
        cur_right = cols - 1
        for col in range(cols - 1, -1, -1):
            if matrix[row][col] == '1':
                right[col] = min(right[col], cur_right)
            else:
                right[col] = cols - 1
                cur_right = col - 1

        # 计算当前行的最大面积
        for col in range(cols):
            if heights[col] > 0:
                width = right[col] - left[col] + 1
                area = heights[col] * width
                max_area = max(max_area, area)

    return max_area


# ✅ 测试
print(maximalRectangle_dp([
    ["1","0","1","0","0"],
    ["1","0","1","1","1"],
    ["1","1","1","1","1"],
    ["1","0","0","1","0"]
]))  # 期望输出:6
```

### 复杂度分析
- **时间复杂度**:O(rows × cols) — 每行三次 O(cols) 遍历(更新heights+left、更新right、计算面积)
- **空间复杂度**:O(cols) — heights、left、right 三个数组

---

## 🐍 Pythonic 写法

利用 Python 的列表推导式简化代码:

```python
def maximalRectangle_pythonic(matrix: List[List[str]]) -> int:
    """Pythonic写法:简化逐行处理逻辑"""
    if not matrix:
        return 0

    cols = len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for row in matrix:
        # 一行内更新heights
        heights = [h + 1 if cell == '1' else 0 for h, cell in zip(heights, row)]
        # 调用单调栈
        max_area = max(max_area, largestRectangleArea(heights[:]))

    return max_area
```

**解释**:用 `zip(heights, row)` 同时迭代高度和当前行,列表推导式一行完成高度更新。

> ⚠️ **面试建议**:先写清晰版本(解法二)展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程和算法理解**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:DP高度+暴力枚举 | 🏆 解法二:DP高度+单调栈(最优) | 解法三:DP三数组 |
|------|---------------------|---------------------------|---------------|
| 时间复杂度 | O(rows×cols²) | **O(rows×cols)** ← 时间最优 | **O(rows×cols)** |
| 空间复杂度 | O(cols) | O(cols) | O(cols) |
| 代码难度 | 简单 | 中等(需理解第84题) | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 教学理解用 | **面试首选,通用性强** | 避免栈的场景 |

**为什么解法二是最优解**:
- 时间复杂度 O(rows×cols) 已经是理论最优(必须扫描所有元素)
- 单调栈是处理"柱状图最大矩形"的标准最优解,复用经典算法
- 代码简洁清晰,容易在面试中正确实现
- 解法三虽然时间复杂度相同,但需要维护三个数组,逻辑更复杂,不如单调栈直观

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题,找出二维矩阵中全为'1'的最大矩形面积。

**你**:(审题30秒)好的,这道题要求在二维矩阵中找最大的全'1'矩形。让我先想一下...

我的第一个想法是暴力枚举所有可能的矩形,但时间复杂度会是 O(n²m²nm),太慢了。

我注意到这道题和第84题"柱状图最大矩形"有关联。核心思路是:**将二维问题降维到一维**。

对每一行,我可以计算从该行向上的连续'1'高度,得到一个高度数组 `heights`。这个高度数组就是一个柱状图!然后用第84题的单调栈解法,在 O(cols) 时间内求出最大矩形。

遍历所有行,总时间复杂度是 O(rows×cols),这是最优的。

**面试官**:很好,请写一下代码。

**你**:(边写边说关键步骤)
```python
def maximalRectangle(matrix):
    if not matrix or not matrix[0]:
        return 0

    rows, cols = len(matrix), len(matrix[0])
    heights = [0] * cols
    max_area = 0

    for row in range(rows):
        # 更新高度数组:遇到'1'累加,遇到'0'清零
        for col in range(cols):
            if matrix[row][col] == '1':
                heights[col] += 1
            else:
                heights[col] = 0

        # 调用第84题的单调栈算法
        max_area = max(max_area, largestRectangleArea(heights))

    return max_area
```

关键在于理解"逐行处理,每行都是一个柱状图"的转化思路。

**面试官**:测试一下?

**你**:用示例走一遍:第0行 heights=[1,0,1,0,0],第1行 heights=[2,0,2,1,1],第2行 heights=[3,1,3,2,2]。对第2行的柱状图,单调栈会找到高度2、宽度3的矩形,面积为6,这是最大值。

再测一个边界:全零矩阵 `[["0"]]`,heights 始终为 [0],返回 0。正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "单调栈的时间复杂度真的是O(n)吗?" | **是的,虽然有嵌套while循环,但每个元素最多入栈和出栈各一次,总操作次数是 2n,均摊 O(n)。** |
| "能不能不用栈?" | **可以用解法三的DP三数组(heights、left、right),时间复杂度相同 O(nm),但代码更复杂,不如单调栈直观。** |
| "如果矩阵非常大,内存不够怎么办?" | **heights 数组空间是 O(cols),已经很小。如果 rows 非常大,可以流式处理,每次只读入一行,不需要存储整个矩阵。** |
| "这道题和第84题的关系?" | **第85题是第84题的推广:84题是一维柱状图,85题把二维矩阵的每一行都转化为一维柱状图,然后分别求解,取最大值。** |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:zip 同步迭代多个列表
heights = [1, 2, 3]
row = ['1', '0', '1']
for h, cell in zip(heights, row):
    new_h = h + 1 if cell == '1' else 0
    print(new_h)  # 2, 0, 4

# 技巧2:列表推导式 + 条件表达式
heights = [h + 1 if c == '1' else 0 for h, c in zip(heights, row)]

# 技巧3:append(0)添加哨兵简化边界处理
heights.append(0)  # 单调栈末尾哨兵
# ... 处理
heights.pop()  # 恢复原数组
```

### 💡 底层原理(选读)

> **为什么单调栈能高效求柱状图最大矩形?**
>
> 单调栈维护一个"递增序列",关键在于:
> 1. **入栈**:当前柱子比栈顶高,说明可以继续扩展,直接入栈
> 2. **出栈**:当前柱子比栈顶矮,说明栈顶柱子"向右不能再扩展了",此时栈顶柱子的"左边界"是它下面的栈顶,"右边界"是当前位置
> 3. **面积计算**:高度=出栈柱子的高度,宽度=右边界-左边界-1
>
> 每个柱子最多入栈出栈各一次,所以总操作次数 ≤ 2n,均摊 O(n)。
>
> **为什么要在末尾加哨兵 0?**
> 确保栈内所有元素都被弹出处理,避免遗漏递增序列末尾的柱子。

### 算法模式卡片 📐
- **模式名称**:二维矩阵降维 + 逐行复用一维算法
- **适用条件**:
  - 二维矩阵中求某种"连续区域"的最值
  - 可以逐行(或逐列)处理,每行转化为一维问题
- **识别关键词**:"最大矩形"、"全为1的矩形"、"二维矩阵"
- **模板代码**:
```python
def solve_2d_matrix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    state = [0] * cols  # 状态数组(如heights)
    result = 0

    for row in range(rows):
        # 更新状态数组
        for col in range(cols):
            if condition:
                state[col] = update_rule(state[col], matrix[row][col])
            else:
                state[col] = 0

        # 对当前状态数组,调用一维问题的最优算法
        result = max(result, solve_1d_problem(state))

    return result
```

### 易错点 ⚠️
1. **忘记在遇到'0'时清零 heights**
   - **错误**:`if matrix[row][col] == '1': heights[col] += 1`(没有 else 分支)
   - **后果**:遇到'0'后 heights 仍保留旧值,导致错误计算不连续的矩形
   - **正确做法**:必须写 `else: heights[col] = 0`

2. **单调栈忘记添加哨兵**
   - **错误**:处理完所有柱子后,栈内还有递增序列没被处理
   - **后果**:遗漏末尾的矩形,如 `heights=[1,2,3,4,5]` 会算不到最大值
   - **正确做法**:`heights.append(0)` 触发最后的清栈

3. **宽度计算错误**
   - **错误**:`width = i - stack[-1]`(没有 -1)
   - **后果**:宽度多算了1,面积偏大
   - **正确做法**:`width = i if not stack else i - stack[-1] - 1`(注意边界)

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:图像处理 — 最大全白矩形检测**
  - 在二值图像中(黑白像素),检测最大的全白矩形区域,用于图像分割、文档扫描中的文本框定位
  - 直接套用本题算法,白色='1',黑色='0'

- **场景2:数据中心机架布局优化**
  - 给定机房的二维平面图(1=可用位置,0=障碍物),找出能放置最大矩形服务器阵列的位置
  - 用本题算法快速找出最优布局方案

- **场景3:游戏地图中的区域规划**
  - 在沙盒游戏中,玩家圈地建造,系统需要验证圈出的区域是否为最大矩形(如《我的世界》的领地系统)
  - 实时计算玩家选中区域的最大矩形面积

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 84. 柱状图中最大的矩形 | Hard | 单调栈 | **本题的核心前置题,必须先掌握!** |
| LeetCode 221. 最大正方形 | Medium | 动态规划 | 类似思路,但要求正方形(边长相等) |
| LeetCode 1277. 统计全为1的正方形子矩阵 | Medium | 动态规划 | 统计数量而非面积,DP状态定义不同 |
| LeetCode 1727. 重新排列后的最大子矩阵 | Medium | 前缀和+本题思路 | 允许重排列,先排序再套用本题算法 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个二维矩阵,元素为非负整数。求矩阵中"元素和不超过K"的最大矩形面积。(不再限制全为1)

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

**提示**:仍然可以逐行处理,但 heights 不再是简单的累加,而是"前缀和"。对每个高度数组,枚举所有可能的矩形,用二维前缀和快速计算区域和,判断是否 ≤ K。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def maxSumSubmatrix(matrix: List[List[int]], k: int) -> int:
    """
    扩展问题:元素和不超过K的最大矩形
    思路:枚举上下边界,对每对边界用一维"最大和不超过K的子数组"算法
    """
    rows, cols = len(matrix), len(matrix[0])
    max_sum = float('-inf')

    # 枚举上边界
    for up in range(rows):
        col_sum = [0] * cols
        # 枚举下边界
        for down in range(up, rows):
            # 累加当前行到 col_sum
            for col in range(cols):
                col_sum[col] += matrix[down][col]

            # 对 col_sum 这个一维数组,求"最大和不超过K的子数组"
            # 用有序集合 + 前缀和(类似第560题)
            from sortedcontainers import SortedList
            sorted_sums = SortedList([0])
            cur_sum = 0
            for num in col_sum:
                cur_sum += num
                # 找 cur_sum - x <= k, 即 x >= cur_sum - k
                idx = sorted_sums.bisect_left(cur_sum - k)
                if idx < len(sorted_sums):
                    max_sum = max(max_sum, cur_sum - sorted_sums[idx])
                sorted_sums.add(cur_sum)

    return max_sum
```

**核心思路**:
1. 枚举所有可能的上下边界组合 O(rows²)
2. 对每对边界,将中间的行压缩为一维数组 `col_sum`
3. 在一维数组上用"前缀和 + 有序集合"求最大和 ≤ K 的子数组 O(cols log cols)
4. 总时间复杂度 O(rows² × cols log cols)

**区别**:不能再用单调栈(单调栈只适用于"柱状图最大矩形",不适用于"和的约束"),需要用前缀和 + 二分查找。

</details>

---

## 📚 课程总结

本课学习了"最大矩形"问题的三种解法:

1. **解法一(DP高度+暴力)**:逐行累计高度,双重循环枚举矩形,O(nm²),适合理解思路
2. **🏆 解法二(DP高度+单调栈,最优)**:将二维问题降维到一维,复用第84题的单调栈解法,**O(nm) 时间 + O(m) 空间,理论最优**
3. **解法三(DP三数组)**:用 heights、left、right 三个数组记录边界,直接计算面积,O(nm),但代码更复杂

**核心要点**:
- **降维思想**:二维矩阵 → 逐行处理 → 每行转化为一维柱状图
- **复用经典算法**:第84题的单调栈是处理柱状图的标准解法,直接套用
- **前置依赖**:本题强依赖第84题,必须先掌握单调栈的原理

**面试建议**:
1. 先口述暴力法 O(n²m²),表明你能想到基本解法
2. 立即提出"逐行转化为柱状图"的优化思路
3. 重点讲解如何复用第84题的单调栈算法
4. 强调时间复杂度 O(nm) 已达理论最优(必须扫描所有元素)
5. 手动测试边界用例(全零、全一、单行单列),展示细致程度

**学习心得**:
- 遇到"二维矩阵中的最大矩形"问题,优先考虑"逐行降维"模式
- 单调栈是处理"柱状图"类问题的万能钥匙,必须熟练掌握
- 算法题之间的关联性很强,85题本质是84题的推广,学会"迁移应用"是关键

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
