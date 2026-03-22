# 📖 第65课:买卖股票的最佳时机

> **模块**:贪心算法 | **难度**:Easy ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/
> **前置知识**:数组遍历、变量更新
> **预计学习时间**:15分钟

---

## 🎯 题目描述

给定一个数组 prices,其中 prices[i] 表示股票在第 i 天的价格。你只能选择某一天买入这只股票,并在未来的某一天卖出。计算你所能获取的最大利润。如果不能获得任何利润,返回 0。

**注意**:你只能完成一笔交易(即买入和卖出各一次)。

**示例:**
```
输入:prices = [7,1,5,3,6,4]
输出:5
解释:在第 2 天(价格 = 1)买入,在第 5 天(价格 = 6)卖出,最大利润 = 6 - 1 = 5。
```

**约束条件:**
- 1 <= prices.length <= 10^5
- 0 <= prices[i] <= 10^4
- 必须先买入后卖出(不能在买入前卖出)

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | prices=[1] | 0 | 只有一天无法交易 |
| 单调递减 | prices=[7,6,4,3,1] | 0 | 无利可图 |
| 单调递增 | prices=[1,2,3,4,5] | 4 | 首尾最优 |
| 大规模 | n=10^5 | - | 必须O(n)算法 |
| 相同价格 | prices=[5,5,5,5] | 0 | 无差价 |

---

## 💡 思路引导

### 生活化比喻
想象你是一个商人,拿到了未来一周的水果价格表。你只能带一车水果,决定什么时候买入、什么时候卖出来赚取最大利润。

> 🐌 **笨办法**:穷举所有可能的买入-卖出组合,对比每一天买入、后续每一天卖出的利润。这就像你把价格表上所有可能的配对都算了一遍,复杂度 O(n²),当价格表有 10 万天时,就要计算 100 亿次!
>
> 🚀 **聪明办法**:边看价格表边思考——"如果今天卖出,我应该在之前哪天买入最划算?"答案显而易见:在今天之前的最低价买入!所以只需要一遍扫描,一边维护"历史最低价",一边计算"今天卖出的最大利润"。

### 关键洞察
**核心思想:贪心地维护历史最低价,实时更新最大利润**

这是贪心算法的入门经典题——每一步都做局部最优决策(维护最小值),最终得到全局最优解(最大利润)。

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组 prices,长度 1 到 10^5
- **输出**:一个整数,表示最大利润(可以为 0)
- **限制**:必须先买入后卖出,只能交易一次

### Step 2:先想笨办法(暴力法)
最直接的想法:枚举所有可能的 (买入日, 卖出日) 组合,计算每个组合的利润,取最大值。
- 时间复杂度:O(n²) — 双层循环
- 瓶颈在哪:对于每个卖出日,都要向前扫描所有可能的买入日,重复计算了很多次"找最小值"的操作

### Step 3:瓶颈分析 → 优化方向
分析暴力法中"重复计算"的环节:
- 核心问题:每次考虑"今天卖出"时,都要重新扫描之前所有天找最低价
- 优化思路:能不能一边遍历一边维护"截止到今天的历史最低价"?这样就不需要每次回头查找了

### Step 4:选择武器
- 选用:**贪心算法 + 一次遍历**
- 理由:每次只需要知道"历史最低价"即可判断当前的最大利润,不需要回溯,符合贪心的无后效性

> 🔑 **模式识别提示**:当题目要求"一次遍历中维护某个历史最值",优先考虑"贪心 + 单变量维护"模式

---

## 🔑 解法一:暴力双循环(直觉法)

### 思路
枚举所有 i < j 的组合,计算 prices[j] - prices[i],取最大值。

### 图解过程

```
prices = [7, 1, 5, 3, 6, 4]

暴力法枚举所有组合:
买入日 i=0 (价格7): 卖出日j=1(1)利润-6, j=2(5)利润-2, j=3(3)利润-4, j=4(6)利润-1, j=5(4)利润-3
买入日 i=1 (价格1): 卖出日j=2(5)利润4, j=3(3)利润2, j=4(6)利润5 ← 最大, j=5(4)利润3
买入日 i=2 (价格5): 卖出日j=3(3)利润-2, j=4(6)利润1, j=5(4)利润-1
买入日 i=3 (价格3): 卖出日j=4(6)利润3, j=5(4)利润1
买入日 i=4 (价格6): 卖出日j=5(4)利润-2

最大利润 = 5
```

### Python代码

```python
from typing import List


def maxProfit_brute(prices: List[int]) -> int:
    """
    解法一:暴力双循环
    思路:枚举所有买入-卖出组合,找最大利润
    """
    max_profit = 0
    n = len(prices)

    # 外层循环枚举买入日
    for i in range(n):
        # 内层循环枚举卖出日(必须在买入日之后)
        for j in range(i + 1, n):
            profit = prices[j] - prices[i]
            max_profit = max(max_profit, profit)

    return max_profit


# ✅ 测试
print(maxProfit_brute([7,1,5,3,6,4]))  # 期望输出:5
print(maxProfit_brute([7,6,4,3,1]))    # 期望输出:0
print(maxProfit_brute([1,2]))          # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(n²) — 双层循环,每个元素都要和后面所有元素配对
  - 具体地说:如果输入规模 n=1000,大约需要 1000×999/2 ≈ 50万次操作
- **空间复杂度**:O(1) — 只用了几个变量

### 优缺点
- ✅ 思路直观,容易理解
- ✅ 代码简洁
- ❌ 时间复杂度O(n²)无法通过大数据量测试(n=10^5会超时)
- ❌ 存在大量重复计算

---

## 🏆 解法二:一次遍历维护最小值(贪心算法 - 最优解)

### 优化思路
从解法一的痛点出发:每次考虑"今天卖出"时,都要重新找之前的最低价。能否一边遍历一边维护历史最低价呢?

> 💡 **关键想法**:贪心策略——对于每一天,只需要知道"之前的最低价"即可计算"今天卖出的最大利润"。维护两个变量:min_price(历史最低价)和 max_profit(最大利润),一次遍历即可。

**为什么这是贪心算法?**
- **局部最优**:每次都选择"历史最低价"买入
- **全局最优**:由于每天都考虑了最优买入价,最终得到的就是全局最大利润
- **无后效性**:当前的决策(今天卖出的利润)只依赖历史最低价,不影响未来

### 图解过程

```
prices = [7, 1, 5, 3, 6, 4]
初始化: min_price = 7, max_profit = 0

第1天(价格7):
  min_price = 7 (历史最低)
  今天卖出利润 = 7 - 7 = 0
  max_profit = max(0, 0) = 0
  状态: min_price=7, max_profit=0

第2天(价格1):
  min_price = min(7, 1) = 1 ← 更新历史最低价
  今天卖出利润 = 1 - 1 = 0
  max_profit = max(0, 0) = 0
  状态: min_price=1, max_profit=0

第3天(价格5):
  min_price = min(1, 5) = 1 (保持)
  今天卖出利润 = 5 - 1 = 4 ← 如果在第2天买入,今天卖出
  max_profit = max(0, 4) = 4
  状态: min_price=1, max_profit=4

第4天(价格3):
  min_price = min(1, 3) = 1 (保持)
  今天卖出利润 = 3 - 1 = 2
  max_profit = max(4, 2) = 4 (保持)
  状态: min_price=1, max_profit=4

第5天(价格6):
  min_price = min(1, 6) = 1 (保持)
  今天卖出利润 = 6 - 1 = 5 ← 最优方案:第2天买,第5天卖
  max_profit = max(4, 5) = 5 ← 更新最大利润
  状态: min_price=1, max_profit=5

第6天(价格4):
  min_price = min(1, 4) = 1 (保持)
  今天卖出利润 = 4 - 1 = 3
  max_profit = max(5, 3) = 5 (保持)
  最终答案: 5

可视化时间线:
  第1天    第2天    第3天    第4天    第5天    第6天
   7        1        5        3        6        4
   |        |        |                 |
   |        |________|_________________|  利润 = 6-1 = 5
   |        ↑                          ↑
   |   历史最低价                   最佳卖出点
   |
   初始价格(不是最优买入点)
```

**再用边界案例演示一次(单调递减)**:
```
prices = [7, 6, 4, 3, 1]

第1天(7): min_price=7, max_profit=0
第2天(6): min_price=6, 今天卖利润=6-6=0, max_profit=0
第3天(4): min_price=4, 今天卖利润=4-4=0, max_profit=0
第4天(3): min_price=3, 今天卖利润=3-3=0, max_profit=0
第5天(1): min_price=1, 今天卖利润=1-1=0, max_profit=0

最终答案: 0 (无利可图)
```

### Python代码

```python
from typing import List


def maxProfit(prices: List[int]) -> int:
    """
    解法二:一次遍历维护最小值(贪心算法)
    思路:维护历史最低价,实时计算今天卖出的最大利润
    """
    if not prices:
        return 0

    # 初始化历史最低价为第一天的价格
    min_price = prices[0]
    # 初始化最大利润为0
    max_profit = 0

    # 从第二天开始遍历
    for price in prices[1:]:
        # 计算今天卖出的利润(前提是在历史最低价买入)
        profit = price - min_price
        # 更新最大利润
        max_profit = max(max_profit, profit)
        # 更新历史最低价
        min_price = min(min_price, price)

    return max_profit


# ✅ 测试
print(maxProfit([7,1,5,3,6,4]))  # 期望输出:5
print(maxProfit([7,6,4,3,1]))    # 期望输出:0
print(maxProfit([1,2,3,4,5]))    # 期望输出:4
print(maxProfit([2,4,1]))        # 期望输出:2
```

### 复杂度分析
- **时间复杂度**:O(n) — 只遍历一次数组,每个元素访问一次
  - 具体地说:如果输入规模 n=100,000,只需要 100,000 次操作,比暴力法快了 50,000 倍!
- **空间复杂度**:O(1) — 只用了 min_price 和 max_profit 两个变量

**为什么O(n)是最优的?**
- 理论下界:至少要遍历一次数组才能知道所有价格,所以 O(n) 是理论最优
- 这个解法已经达到了理论下界,无法再优化

---

## ⚡ 解法三:动态规划思想(可选理解)

### 优化思路
虽然这道题用贪心算法最简洁,但也可以从动态规划角度理解:定义状态 dp[i] 表示"第 i 天卖出能获得的最大利润"。

> 💡 **状态转移**:dp[i] = max(0, prices[i] - min(prices[0:i]))

实际上这和贪心算法是等价的,只是换了一种表述方式。

### Python代码

```python
from typing import List


def maxProfit_dp(prices: List[int]) -> int:
    """
    解法三:动态规划思想
    思路:dp[i] 表示第 i 天卖出的最大利润
    """
    if not prices:
        return 0

    n = len(prices)
    # dp[i] 表示第 i 天卖出能获得的最大利润
    dp = [0] * n
    min_price = prices[0]

    for i in range(1, n):
        # 第 i 天卖出的利润 = 今天价格 - 历史最低价
        dp[i] = max(0, prices[i] - min_price)
        # 更新历史最低价
        min_price = min(min_price, prices[i])

    # 返回所有天中的最大利润
    return max(dp)


# ✅ 测试
print(maxProfit_dp([7,1,5,3,6,4]))  # 期望输出:5
print(maxProfit_dp([7,6,4,3,1]))    # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n) — 一次遍历 + 一次 max()
- **空间复杂度**:O(n) — 需要 dp 数组

### 优缺点
- ✅ 从 DP 角度理解问题,为股票系列进阶题打基础
- ❌ 空间复杂度比贪心法高,但可以优化为 O(1)

---

## 🐍 Pythonic 写法

利用 Python 的语法糖,可以写得更简洁:

```python
def maxProfit_pythonic(prices: List[int]) -> int:
    """Pythonic 一行流写法"""
    min_price = float('inf')
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        max_profit = max(max_profit, price - min_price)
    return max_profit
```

**更极致的函数式写法**(不推荐面试用):
```python
from itertools import accumulate

def maxProfit_functional(prices: List[int]) -> int:
    """函数式编程风格(仅供学习)"""
    if not prices:
        return 0
    # accumulate 实现滚动最小值
    min_prices = accumulate(prices, min)
    profits = [p - mp for p, mp in zip(prices, min_prices)]
    return max(profits)
```

> ⚠️ **面试建议**:优先写清晰版本(解法二)展示思路,再提 Pythonic 写法展示语言功底。面试官更看重你的**思考过程和贪心思想的理解**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:暴力双循环 | 🏆 解法二:贪心维护最小值(最优) | 解法三:动态规划 |
|------|-----------------|---------------------------|--------------|
| 时间复杂度 | O(n²) | **O(n)** ← 时间最优 | O(n) |
| 空间复杂度 | O(1) | **O(1)** ← 空间最优 | O(n) 可优化为 O(1) |
| 代码难度 | 简单 | 简单 | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 只适合小数据 | **面试首选,通用性强** | 为股票系列进阶题铺垫 |

**为什么解法二是最优解?**
1. **时间复杂度O(n)已经是理论最优**:必须至少看一遍所有价格,无法更快
2. **空间复杂度O(1)已经是最优**:只用了两个变量,无法更省
3. **代码简洁易懂**:逻辑清晰,不易出错
4. **贪心思想典型**:完美体现"局部最优→全局最优"的贪心精髓

**面试建议**:
1. 先用30秒口述暴力法思路(O(n²)),表明你能想到基本解法
2. 立即优化到🏆最优解(O(n)贪心法),展示优化能力
3. **重点讲解贪心策略**:"一边遍历一边维护历史最低价,实时计算最大利润"
4. 强调为什么这是最优:时间空间都已达理论下限,无法再优化
5. 手动测试边界用例(单调递减、单调递增),展示对解法的深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道股票买卖的问题。

**你**:(审题30秒)好的,这道题要求在数组中找一对 (买入日, 卖出日),使得利润最大,且买入必须在卖出之前。让我先想一下...

我的第一个想法是暴力枚举所有 i < j 的组合,计算 prices[j] - prices[i],时间复杂度是 O(n²)。但这显然不够优,因为存在大量重复计算——每次考虑"今天卖出"时,都要重新找之前的最低价。

更好的方法是用**贪心算法**:一边遍历一边维护"历史最低价",实时计算"今天卖出的最大利润"。这样只需要一次遍历,时间复杂度优化到 O(n),空间复杂度 O(1)。

**面试官**:很好,为什么这个贪心策略是正确的?

**你**:因为对于任意一天,如果我们决定在这天卖出,那么最优的买入日一定是"这天之前的最低价那天"。所以维护历史最低价就是局部最优决策,而遍历所有天数就能找到全局最优解。这符合贪心算法的"局部最优→全局最优"特性。

**面试官**:请写一下代码。

**你**:(边写边说)我用两个变量,min_price 记录历史最低价,max_profit 记录最大利润。初始化 min_price 为第一天价格,max_profit 为 0。然后从第二天开始遍历,每天做两件事:一是计算今天卖出的利润并更新 max_profit,二是更新 min_price。

```python
def maxProfit(prices):
    min_price = prices[0]
    max_profit = 0
    for price in prices[1:]:
        profit = price - min_price
        max_profit = max(max_profit, profit)
        min_price = min(min_price, price)
    return max_profit
```

**面试官**:测试一下?

**你**:用示例 [7,1,5,3,6,4] 走一遍:
- 第1天价格7,min_price=7,max_profit=0
- 第2天价格1,min_price更新为1,profit=1-1=0,max_profit=0
- 第3天价格5,profit=5-1=4,max_profit更新为4
- 第4天价格3,profit=3-1=2,max_profit保持4
- 第5天价格6,profit=6-1=5,max_profit更新为5 ← 最优方案
- 第6天价格4,profit=4-1=3,max_profit保持5
- 返回 5 ✓

再测一个边界情况 [7,6,4,3,1](单调递减):每天的 profit 都是 0 或负数,max_profit 始终为 0,返回 0 ✓

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间O(n)已经是理论最优(至少要遍历一遍),空间O(1)也已最优。无法再优化。 |
| "如果可以交易多次呢?" | 那就变成 LeetCode 122,用贪心策略:只要后一天价格高于前一天就交易,累加所有上涨差价。 |
| "如果最多交易k次呢?" | 那就是 LeetCode 188,需要用动态规划,状态定义为 dp[i][j][0/1] 表示第i天、已交易j次、当前是否持有股票的最大利润。 |
| "数据量特别大怎么办?" | O(n)已经是线性时间,可以流式处理。如果内存不够,可以分块读取,每块维护局部最小值和最大利润,最后合并结果。 |
| "如果有手续费呢?" | 计算利润时减去手续费:profit = price - min_price - fee,其余逻辑不变。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:无穷大初始化 — 用于找最小值
min_price = float('inf')
# 这样第一次比较时任何实际价格都会更新 min_price

# 技巧2:同时更新两个变量
max_profit = max(max_profit, price - min_price)
min_price = min(min_price, price)
# 注意顺序:先用旧的min_price计算profit,再更新min_price

# 技巧3:三元表达式简化
max_profit = profit if profit > max_profit else max_profit
# 等价于: max_profit = max(max_profit, profit)

# 技巧4:enumerate遍历时可以同时获取索引和值
for i, price in enumerate(prices):
    print(f"第{i}天价格为{price}")
```

### 💡 底层原理(选读)

**为什么贪心算法在这道题有效?**

贪心算法成立的两个关键条件:
1. **最优子结构**:问题的最优解包含子问题的最优解。对于股票问题,如果在第 i 天卖出获得最大利润,那么买入日一定是第 0 到 i-1 天中价格最低的那天。
2. **贪心选择性质**:每一步都选择当前看来最优的选择(维护历史最低价),不会影响后续选择,最终能得到全局最优解。

**与动态规划的区别**:
- 贪心算法:每步只做局部最优决策,不回溯,不需要存储子问题结果。时间O(n),空间O(1)。
- 动态规划:需要存储子问题结果(dp数组),通过状态转移方程求解。时间O(n),空间O(n)(可优化为O(1))。

这道题用贪心更简洁,但理解 DP 思路有助于解决股票系列的进阶题(如 LeetCode 123, 188, 309, 714)。

### 算法模式卡片 📐
- **模式名称**:贪心算法 — 维护历史最值
- **适用条件**:需要在遍历过程中实时维护某个历史最值(最大/最小/最远等),并基于该最值做决策
- **识别关键词**:"历史最低/最高"、"截止到当前"、"一次遍历"、"实时更新"
- **模板代码**:
```python
def greedy_maintain_extreme(arr):
    """贪心维护历史最值模板"""
    if not arr:
        return 0

    extreme_value = arr[0]  # 历史最值(最小/最大)
    result = 0              # 要求的结果

    for val in arr[1:]:
        # 基于历史最值计算当前结果
        current_result = calculate(val, extreme_value)
        result = update(result, current_result)

        # 更新历史最值
        extreme_value = update_extreme(extreme_value, val)

    return result
```

**同类型题目**:
- LeetCode 122. 买卖股票的最佳时机 II(多次交易)
- LeetCode 55. 跳跃游戏(维护最远可达位置)
- LeetCode 45. 跳跃游戏 II(贪心跳跃)
- LeetCode 53. 最大子数组和(Kadane算法)

### 易错点 ⚠️
1. **错误:先更新 min_price 再计算 profit**
   ```python
   # ❌ 错误写法
   min_price = min(min_price, price)
   profit = price - min_price  # 这样会导致 profit 为 0(自己和自己比)
   ```
   **正确做法**:先用旧的 min_price 计算 profit,再更新 min_price。

2. **错误:忘记处理单调递减的情况**
   ```python
   # ❌ 错误写法:初始化 max_profit = -1
   max_profit = -1  # 如果所有价格递减,会返回 -1 而非 0
   ```
   **正确做法**:初始化 max_profit = 0,因为不交易利润为 0。

3. **错误:没有考虑只有一天的情况**
   ```python
   # ❌ 错误写法
   for price in prices[1:]:  # 如果 prices = [5],会跳过循环
   ```
   **正确做法**:要么在开头检查长度,要么确保初始化能处理边界情况。

4. **错误:认为必须找到具体的买入日和卖出日**
   - 题目只要求返回最大利润,不需要返回具体日期
   - 如果要返回日期,需要额外记录 buy_day 和 sell_day

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:股票交易系统**
  - 高频交易系统需要实时计算最优买入卖出时机
  - 本题的贪心思想可以扩展为"滑动窗口内的最优交易策略"

- **场景2:电商价格监控**
  - 电商平台的"价格历史曲线"功能,告诉用户"什么时候买最划算"
  - 维护历史最低价并计算当前价格的"性价比"

- **场景3:资源调度优化**
  - 云计算平台的资源购买策略:在价格低谷期购买计算资源,在高峰期使用
  - 贪心地选择最低价时段采购,最高价时段释放

- **场景4:数据流处理**
  - 流式数据中实时维护"历史最值"是常见需求
  - 例如:监控系统中维护"过去1小时内的最低延迟"用于对比当前性能

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 122. 买卖股票的最佳时机 II | Medium | 贪心(多次交易) | 只要后一天价格高就交易,累加所有上涨差价 |
| LeetCode 55. 跳跃游戏 | Medium | 贪心(维护最远可达) | 维护当前能到达的最远位置 |
| LeetCode 45. 跳跃游戏 II | Medium | 贪心(最少跳跃次数) | 在当前能到达的范围内选择下一跳最远的 |
| LeetCode 53. 最大子数组和 | Medium | 贪心/DP(Kadane) | 维护"以当前元素结尾的最大和" |
| LeetCode 123. 买卖股票的最佳时机 III | Hard | 动态规划(多次交易限制) | 状态机DP,限制交易次数 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定股票价格数组 prices 和一个整数 fee 表示交易手续费(每次交易都要支付)。你可以进行多次交易,但每次卖出时需要支付手续费。计算能获得的最大利润。

示例:prices = [1,3,2,8,4,9], fee = 2,输出:8(买入1卖出8支付2手续费获利5,买入4卖出9支付2手续费获利3,总利润8)

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

这道题是"买卖股票的最佳时机 II"(可多次交易) + 手续费。可以用贪心或动态规划。贪心思路:维护"有效买入价"(考虑手续费后的成本),只有当"卖出价 - 有效买入价 > fee"时才交易。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def maxProfit_with_fee(prices: List[int], fee: int) -> int:
    """
    股票交易含手续费(贪心算法)
    思路:维护有效买入价,只有利润 > 手续费时才卖出
    """
    if not prices:
        return 0

    max_profit = 0
    min_price = prices[0]  # 当前的有效买入价

    for price in prices[1:]:
        if price < min_price:
            # 发现更低价格,更新买入价
            min_price = price
        elif price > min_price + fee:
            # 利润 > 手续费,执行卖出
            max_profit += price - min_price - fee
            # 更新买入价为"卖出价 - 手续费"(允许连续交易优化)
            min_price = price - fee

    return max_profit

# 测试
print(maxProfit_with_fee([1,3,2,8,4,9], 2))  # 输出:8
```

**核心思路**:每次卖出后,将"有效买入价"更新为"卖出价 - fee",这样如果后续价格继续上涨,可以无缝连续交易而不重复支付手续费。这是贪心策略的巧妙应用。

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
