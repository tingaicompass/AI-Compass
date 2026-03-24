> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第87课:股票含冷冻期

> **模块**:动态规划 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-with-cooldown/
> **前置知识**:第71课(爬楼梯)、第73课(打家劫舍)、第65课(买卖股票最佳时机)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定一个整数数组 prices,其中 prices[i] 表示某支股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易(多次买卖一支股票),但有以下限制:

- 卖出股票后,你无法在第二天买入股票(即冷冻期为 1 天)
- 你不能同时参与多笔交易(你必须在再次购买前出售掉之前的股票)

**示例:**
```
输入: prices = [1,2,3,0,2]
输出: 3
解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]
     买入 prices[0] = 1, 卖出 prices[1] = 2, 利润 = 1
     冷冻期 prices[2] = 3 (不能买入)
     买入 prices[3] = 0, 卖出 prices[4] = 2, 利润 = 2
     总利润 = 1 + 2 = 3
```

**约束条件:**
- 1 ≤ prices.length ≤ 5000
- 0 ≤ prices[i] ≤ 1000

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | prices=[1] | 0 | 单天无法交易 |
| 单调递增 | prices=[1,2,3,4,5] | 4 | 一买一卖最优 |
| 单调递减 | prices=[5,4,3,2,1] | 0 | 不交易最优 |
| 含冷冻期影响 | prices=[1,2,3,0,2] | 3 | 需要跳过冷冻期 |
| 大规模 | n=5000 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你是一个股票交易员,但公司有个奇怪的规定:每次卖出股票后,第二天必须休息一天(冷冻期),不能立即买入新股票。
>
> 🐌 **笨办法**:尝试所有可能的买卖组合,每次卖出后记得跳过一天,然后算出最大利润。这样的话,对于5天的股票价格,可能的组合数以指数级增长,计算量巨大。
>
> 🚀 **聪明办法**:每天只需要记录三种状态:
> - **持有股票**:手上有股票,今天不操作或今天买入
> - **不持有且不在冷冻期**:手上没股票,可以随时买入
> - **刚卖出(冷冻期)**:今天刚卖出,明天不能买入
>
> 每天根据前一天的这三种状态,计算今天的最大利润。就像玩状态机游戏,从一个状态跳到另一个状态,最后看哪个状态的分数最高!

### 关键洞察
**核心突破口:用状态机DP建模,每天只有"持有"、"不持有(可交易)"、"冷冻期"三种状态,状态之间按规则转移。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组 prices,表示每天的股票价格
- **输出**:能获得的最大利润(整数)
- **限制**:
  - 卖出后必须冷冻1天才能再买入
  - 不能同时持有多只股票
  - 可以多次买卖

### Step 2:先想笨办法(暴力法)
用回溯枚举所有可能的买卖方案:对每一天,选择"买入"、"卖出"或"什么都不做",然后递归计算后续天数的最大利润。
- 时间复杂度:O(3^n) — 每天3种选择,指数级爆炸
- 瓶颈在哪:大量重复计算,比如"第5天持有股票"这个状态可能被计算上千次

### Step 3:瓶颈分析 → 优化方向
暴力法中,同样的"某一天+某种持有状态"被反复计算。比如:
- "第3天持有股票,花费10元" 这个状态,无论之前怎么操作到达这里,后续的最优策略都是一样的
- 核心问题:如何避免重复计算相同的子问题?
- 优化思路:用动态规划记录每天每种状态的最大利润

### Step 4:选择武器
- 选用:**状态机DP(Dynamic Programming with State Machine)**
- 理由:
  1. 问题具有"最优子结构":今天的最优决策依赖于昨天的状态
  2. 存在"重叠子问题":相同的(天数,持有状态)会被重复访问
  3. 冷冻期限制可以建模为状态转移的约束条件

> 🔑 **模式识别提示**:当题目出现"多阶段决策 + 状态限制"时,优先考虑"状态机DP"

---

## 🔑 解法一:状态机DP — 三状态建模

### 思路
定义三种状态:
- `hold[i]`:第 i 天结束时"持有股票"的最大利润
- `sold[i]`:第 i 天结束时"刚卖出(进入冷冻期)"的最大利润
- `rest[i]`:第 i 天结束时"不持有且不在冷冻期"的最大利润

状态转移:
- `hold[i] = max(hold[i-1], rest[i-1] - prices[i])` — 要么昨天就持有,要么今天从rest状态买入
- `sold[i] = hold[i-1] + prices[i]` — 必须从持有状态卖出
- `rest[i] = max(rest[i-1], sold[i-1])` — 要么昨天就在rest,要么昨天卖出今天进入rest

### 图解过程

```
示例: prices = [1, 2, 3, 0, 2]

初始状态(第0天):
  hold[0] = -1  (买入第0天股票,花费1元)
  sold[0] = 0   (不可能卖出,无意义)
  rest[0] = 0   (什么都不做)

第1天 (price=2):
  hold[1] = max(hold[0], rest[0]-2) = max(-1, 0-2) = -1  (保持持有第0天买的)
  sold[1] = hold[0] + 2 = -1 + 2 = 1                     (卖出,利润1)
  rest[1] = max(rest[0], sold[0]) = max(0, 0) = 0

第2天 (price=3):
  hold[2] = max(hold[1], rest[1]-3) = max(-1, 0-3) = -1
  sold[2] = hold[1] + 3 = -1 + 3 = 2
  rest[2] = max(rest[1], sold[1]) = max(0, 1) = 1

第3天 (price=0):
  hold[3] = max(hold[2], rest[2]-0) = max(-1, 1-0) = 1   (从rest买入,成本0)
  sold[3] = hold[2] + 0 = -1 + 0 = -1
  rest[3] = max(rest[2], sold[2]) = max(1, 2) = 2

第4天 (price=2):
  hold[4] = max(hold[3], rest[3]-2) = max(1, 2-2) = 1
  sold[4] = hold[3] + 2 = 1 + 2 = 3                      (卖出,总利润3)
  rest[4] = max(rest[3], sold[3]) = max(2, -1) = 2

最终答案 = max(sold[4], rest[4]) = max(3, 2) = 3
```

### Python代码

```python
from typing import List


def maxProfit(prices: List[int]) -> int:
    """
    解法一:状态机DP — 三状态建模
    思路:定义持有、卖出、休息三种状态,按规则转移
    """
    if not prices or len(prices) < 2:
        return 0

    n = len(prices)
    # 初始化三个状态数组
    hold = [0] * n  # 持有股票的最大利润
    sold = [0] * n  # 刚卖出(冷冻期)的最大利润
    rest = [0] * n  # 不持有且不在冷冻期的最大利润

    # 第0天初始状态
    hold[0] = -prices[0]  # 买入第0天的股票
    sold[0] = 0           # 第0天不能卖出
    rest[0] = 0           # 第0天什么都不做

    for i in range(1, n):
        # 持有:要么昨天就持有,要么今天从rest买入
        hold[i] = max(hold[i - 1], rest[i - 1] - prices[i])
        # 卖出:必须从持有状态卖出
        sold[i] = hold[i - 1] + prices[i]
        # 休息:要么昨天就在rest,要么昨天卖出
        rest[i] = max(rest[i - 1], sold[i - 1])

    # 最后一天,取卖出或休息中的较大值
    return max(sold[n - 1], rest[n - 1])


# ✅ 测试
print(maxProfit([1, 2, 3, 0, 2]))  # 期望输出:3
print(maxProfit([1]))              # 期望输出:0
print(maxProfit([1, 2, 4]))        # 期望输出:3
```

### 复杂度分析
- **时间复杂度**:O(n) — 遍历一次数组,每天做常数次状态转移
  - 具体地说:如果输入规模 n=5000,大约需要 5000×3=15000 次操作
- **空间复杂度**:O(n) — 需要三个长度为n的数组存储状态

### 优缺点
- ✅ 逻辑清晰,状态定义明确,易于理解和调试
- ✅ 时间O(n)已是最优,必须至少看一遍所有价格
- ❌ 空间O(n)可以优化,因为每天只依赖前一天的状态

---

## 🏆 解法二:状态机DP — 空间优化(最优解)

### 优化思路
观察状态转移方程,发现第 i 天的状态只依赖第 i-1 天,不需要保存所有天的历史状态。用三个变量代替三个数组,滚动更新。

> 💡 **关键想法**:DP数组可以压缩为O(1)空间,因为只需要"上一天"的状态

### 图解过程

```
滚动变量更新示例: prices = [1, 2, 3, 0, 2]

初始:
  hold = -1, sold = 0, rest = 0

第1天: price=2
  new_hold = max(-1, 0-2) = -1
  new_sold = -1+2 = 1
  new_rest = max(0, 0) = 0
  更新: hold=-1, sold=1, rest=0

第2天: price=3
  new_hold = max(-1, 0-3) = -1
  new_sold = -1+3 = 2
  new_rest = max(0, 1) = 1
  更新: hold=-1, sold=2, rest=1

第3天: price=0
  new_hold = max(-1, 1-0) = 1
  new_sold = -1+0 = -1
  new_rest = max(1, 2) = 2
  更新: hold=1, sold=-1, rest=2

第4天: price=2
  new_hold = max(1, 2-2) = 1
  new_sold = 1+2 = 3
  new_rest = max(2, -1) = 2
  更新: hold=1, sold=3, rest=2

答案 = max(3, 2) = 3
```

### Python代码

```python
def maxProfit_optimized(prices: List[int]) -> int:
    """
    解法二:状态机DP — 空间优化(最优解)
    思路:用三个变量代替三个数组,滚动更新
    """
    if not prices or len(prices) < 2:
        return 0

    # 用三个变量代替数组
    hold = -prices[0]  # 持有股票的最大利润
    sold = 0           # 刚卖出的最大利润
    rest = 0           # 休息状态的最大利润

    for i in range(1, len(prices)):
        # 注意:必须先保存旧值,因为更新顺序有依赖
        new_hold = max(hold, rest - prices[i])
        new_sold = hold + prices[i]
        new_rest = max(rest, sold)

        # 更新状态
        hold = new_hold
        sold = new_sold
        rest = new_rest

    # 最后一天,取卖出或休息中的较大值
    return max(sold, rest)


# ✅ 测试
print(maxProfit_optimized([1, 2, 3, 0, 2]))  # 期望输出:3
print(maxProfit_optimized([1]))              # 期望输出:0
print(maxProfit_optimized([5, 4, 3, 2, 1]))  # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n) — 遍历一次数组
- **空间复杂度**:O(1) — 只用3个变量

### 为什么是最优解
- ✅ **时间O(n)已达理论下限**:必须至少看一遍所有价格才能做出决策
- ✅ **空间O(1)已达最优**:不需要额外存储,只需常数个变量
- ✅ **代码简洁**:逻辑清晰,面试中容易手写正确

---

## 🐍 Pythonic 写法

利用 Python 的多重赋值特性,一行更新所有状态:

```python
def maxProfit_pythonic(prices: List[int]) -> int:
    """Pythonic 写法:利用多重赋值一行更新状态"""
    if not prices or len(prices) < 2:
        return 0

    hold, sold, rest = -prices[0], 0, 0

    for price in prices[1:]:
        hold, sold, rest = (
            max(hold, rest - price),  # 新hold
            hold + price,             # 新sold
            max(rest, sold)           # 新rest
        )

    return max(sold, rest)


# ✅ 测试
print(maxProfit_pythonic([1, 2, 3, 0, 2]))  # 期望输出:3
```

这个写法利用了 Python 的特性:
- **多重赋值**:右边的表达式先全部计算完,再统一赋值给左边,避免了临时变量
- **代码更简洁**:从10行压缩到5行,一目了然

> ⚠️ **面试建议**:先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:三状态数组 | 🏆 解法二:空间优化(最优) |
|------|-----------------|----------------------|
| 时间复杂度 | O(n) | **O(n)** ← 理论最优 |
| 空间复杂度 | O(n) | **O(1)** ← 空间最优 |
| 代码难度 | 简单 | 简单 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 学习理解状态机逻辑 | **面试首选,生产环境** |

**为什么是最优解**:
- 时间O(n)已是理论下限(必须遍历所有价格)
- 空间从O(n)优化到O(1),提升巨大
- 代码依然简洁易懂,面试中容易写对

**面试建议**:
1. 先画出状态转移图,说明三种状态及其转移规则
2. 写出🏆最优解(空间优化版),展示对DP优化的理解
3. 强调为什么是最优:时间空间都已达最优,无法再优化
4. 手动模拟一个小示例,证明逻辑正确

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道股票买卖问题,注意有冷冻期限制。

**你**:(审题30秒)好的,这道题要求在有冷冻期的限制下,计算多次买卖股票的最大利润。让我先想一下...

我的第一个想法是用回溯枚举所有买卖方案,但时间复杂度是O(3^n),太慢了。

不过我注意到这个问题有"最优子结构"和"重叠子问题",可以用动态规划。关键是建立状态机模型:每天有三种状态 — 持有股票、刚卖出(冷冻期)、不持有且可交易。

我可以用三个变量滚动更新,时间O(n),空间O(1)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)我定义三个状态变量:hold表示持有股票的最大利润,sold表示刚卖出的最大利润,rest表示休息状态。

初始化:第一天买入,hold=-prices[0],其他为0。

然后遍历每一天,更新三个状态:
- hold要么保持昨天的,要么今天从rest买入
- sold必须从hold卖出
- rest要么保持,要么从sold进入

最后返回sold和rest的较大值。

**面试官**:测试一下?

**你**:用示例 [1,2,3,0,2] 走一遍...

第0天:买入,hold=-1
第1天:卖出,sold=1,利润1
第2天:冷冻期,rest=1
第3天:买入,hold=1(成本0,之前赚了1)
第4天:卖出,sold=3,总利润3

结果正确!再测边界情况 [1],单天无法交易,返回0,也正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能画一下状态转移图吗?" | 画出三个状态节点和箭头,标注转移条件:"hold→sold(卖出)",  "sold→rest(冷冻期)",  "rest→hold(买入)" |
| "为什么最后是max(sold,rest)?" | 因为持有股票(hold)意味着还没卖,利润是负的或较小。最优策略肯定是最后手上没股票,要么刚卖出,要么在休息状态 |
| "如果冷冻期是2天呢?" | 状态数增加,需要sold1(刚卖),sold2(冷冻第2天),rest。转移规则类似,sold1→sold2→rest→hold |
| "能不能不用DP,用贪心?" | 不行。贪心无法处理冷冻期约束,因为当前最优决策(卖出)会影响未来(明天不能买),必须用DP全局考虑 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:多重赋值 — 一行更新多个变量,避免临时变量
a, b, c = max(a, c-x), a+x, max(b, c)  # 右边先全部计算,再统一赋值

# 技巧2:边界处理 — 提前返回简化逻辑
if not prices or len(prices) < 2:
    return 0

# 技巧3:列表切片 — prices[1:] 从第1个元素开始遍历
for price in prices[1:]:
    ...
```

### 💡 底层原理(选读)

> **为什么DP能优化指数级的回溯?**
>
> 回溯法中,"第5天持有股票,成本10元"这个状态可能通过100种不同路径到达,每条路径都会独立计算后续的最优策略,导致大量重复。
>
> DP的核心思想是"无后效性":只要知道当前状态(第几天+持有情况+累计利润),后续的最优策略就是唯一确定的,与之前怎么到达这个状态无关。
>
> 所以我们可以用一个表格(或变量)记录每个状态的最优值,每个状态只计算一次,从而将时间复杂度从O(3^n)降到O(n)。

### 算法模式卡片 📐
- **模式名称**:状态机DP
- **适用条件**:
  - 问题可以分解为多个阶段(天数/步骤)
  - 每个阶段有若干离散状态(持有/不持有/冷冻等)
  - 状态之间有明确的转移规则
  - 求全局最优值(最大利润/最小成本)
- **识别关键词**:"多次交易"、"状态限制"、"冷冻期"、"买卖股票"
- **模板代码**:
```python
# 状态机DP模板
state1, state2, state3 = init_values

for i in range(1, n):
    new_state1 = transition_rule1(state1, state2, state3)
    new_state2 = transition_rule2(state1, state2, state3)
    new_state3 = transition_rule3(state1, state2, state3)

    state1, state2, state3 = new_state1, new_state2, new_state3

return max(state2, state3)  # 根据题意选择最终状态
```

### 易错点 ⚠️
1. **更新顺序错误**:直接 `hold = max(hold, rest - price)` 会导致后续的 `sold = hold + price` 用到了新的hold值,而不是旧值。
   - **正确做法**:先用临时变量保存新值,最后统一更新,或使用Python多重赋值

2. **初始状态设置错误**:第0天的 `hold` 应该是 `-prices[0]`(花钱买入),而不是0。
   - **正确做法**:仔细理解每个状态的含义,hold表示"持有股票的净利润",买入时是负数

3. **最终答案错误**:返回 `max(hold, sold, rest)` 是错的,因为hold表示还持有股票,利润未实现。
   - **正确做法**:返回 `max(sold, rest)`,即最后一天要么刚卖出,要么在休息,手上没股票

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:量化交易策略**:在算法交易中,需要考虑交易成本、滑点、冷静期等因素。状态机DP可以建模这些复杂约束,找到最优买卖时机。

- **场景2:资源调度优化**:云计算中的虚拟机调度,启动VM后需要"预热期",关闭后有"冷却期"。用状态机DP可以优化VM的启停策略,降低成本。

- **场景3:游戏AI决策**:角色扮演游戏中,使用技能后有"冷却时间"。AI可以用状态机DP计算最优技能释放顺序,最大化伤害输出。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 121. 买卖股票的最佳时机 | Easy | 状态机DP入门 | 只能交易一次,状态更简单 |
| LeetCode 122. 买卖股票的最佳时机II | Medium | 状态机DP | 无冷冻期,可以贪心或DP |
| LeetCode 123. 买卖股票的最佳时机III | Hard | 多维状态DP | 最多2次交易,需要4个状态 |
| LeetCode 188. 买卖股票的最佳时机IV | Hard | 多维状态DP | 最多k次交易,状态数2k个 |
| LeetCode 714. 买卖股票的最佳时机含手续费 | Medium | 状态机DP变体 | 每次交易扣手续费,修改转移方程 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如果冷冻期改为2天(即卖出后需要等2天才能再次买入),如何修改状态定义和转移方程?

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

需要区分"刚卖出(冷冻第1天)"和"冷冻第2天"两个状态,然后 sold1→sold2→rest→hold 的状态链条。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def maxProfit_cooldown2(prices: List[int]) -> int:
    """冷冻期为2天的变体"""
    if not prices or len(prices) < 2:
        return 0

    hold = -prices[0]  # 持有股票
    sold1 = 0          # 刚卖出(冷冻第1天)
    sold2 = 0          # 冷冻第2天
    rest = 0           # 可交易状态

    for i in range(1, len(prices)):
        new_hold = max(hold, rest - prices[i])  # 只能从rest买入
        new_sold1 = hold + prices[i]            # 从hold卖出
        new_sold2 = sold1                       # 冷冻第1天→第2天
        new_rest = max(rest, sold2)             # 冷冻第2天→可交易

        hold, sold1, sold2, rest = new_hold, new_sold1, new_sold2, new_rest

    return max(sold1, sold2, rest)


# 测试
print(maxProfit_cooldown2([1, 2, 3, 0, 2]))  # 结果可能不同,需要重新验证
```

**核心思路**:增加一个中间状态 `sold2` 表示"冷冻期的第2天",状态转移链条变为 `hold→sold1→sold2→rest→hold`。最后返回 `max(sold1, sold2, rest)`,确保手上没股票。

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
