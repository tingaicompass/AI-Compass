> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第75课:零钱兑换

> **模块**:动态规划 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/coin-change/
> **前置知识**:第71课(爬楼梯)、第74课(完全平方数)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给你一个整数数组 `coins` 表示不同面额的硬币,以及一个整数 `amount` 表示总金额。计算并返回可以凑成总金额所需的最少硬币个数。如果没有任何一种硬币组合能组成总金额,返回 `-1`。假设每种硬币的数量是无限的。

**示例:**
```
输入:coins = [1, 2, 5], amount = 11
输出:3
解释:11 = 5 + 5 + 1
```

**约束条件:**
- 1 <= coins.length <= 12
- 1 <= coins[i] <= 2³¹ - 1
- 0 <= amount <= 10⁴
- 每种硬币可以使用无限次(完全背包)

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | amount=0 | 0 | 边界处理 |
| 无解情况 | coins=[2], amount=3 | -1 | 奇偶性判断 |
| 单个硬币 | coins=[1], amount=5 | 5 | 基本功能 |
| 贪心失效 | coins=[1,3,4], amount=6 | 2(3+3,而非4+1+1) | 贪心不适用 |
| 大金额 | amount=10000 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在超市收银台需要找零11元,收银员有1元、2元、5元硬币无限多...
>
> 🐌 **笨办法**:用递归尝试所有组合。对于11元,先试"用1个5元",剩6元继续递归;再试"用1个2元",剩9元继续递归...每个金额都要重复计算无数次,比如"6元最少需要几个硬币"可能被计算上百次!
>
> 🚀 **聪明办法**:建一张"找零表",从1元开始往上填。比如3元用1个2元+1个1元=2个硬币,记下来。后面计算8元时直接查表"8-5=3元需要2个",加上这个5元币就是3个,不用重新计算!这就是动态规划——用表格记住之前算过的结果。

### 关键洞察
**完全背包问题的核心:每次选择硬币时,可以重复使用同一面额!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:硬币面额数组 + 目标金额
- **输出**:最少硬币数(无解返回-1)
- **限制**:每种硬币无限使用,必须刚好凑成目标金额

### Step 2:先想笨办法(暴力递归)
对于金额 `amount`,枚举每种硬币:
- 选择coin[i]后,剩余金额为 `amount - coins[i]`,递归求解
- 返回所有选择中的最小值+1

```python
def coin_change(coins, amount):
    if amount == 0: return 0
    if amount < 0: return -1
    res = float('inf')
    for coin in coins:
        sub = coin_change(coins, amount - coin)
        if sub != -1:
            res = min(res, sub + 1)
    return res if res != float('inf') else -1
```

- 时间复杂度:O(amount^n) — n是硬币种类数,每个金额都可能试n次,指数级爆炸
- 瓶颈在哪:**大量重复计算**,比如amount=11时,可能通过5+6、6+5、2+9等多种路径都要计算"6元需要几个硬币"

### Step 3:瓶颈分析 → 优化方向
递归树中"相同金额"被重复计算。比如amount=11时,`coin_change(6)`可能被调用上百次。
- 核心问题:"每个金额的最优解"被重复计算
- 优化思路:用数组 `dp[]` 记录每个金额的最优解,遇到直接查表!

### Step 4:选择武器
- 选用:**动态规划(完全背包)**
- 理由:将大问题(amount=11)拆成子问题(amount=0,1,2...10),每个子问题只算一次并记录,后续直接复用

> 🔑 **模式识别提示**:当题目出现"最少/最多"+"可重复使用资源",优先考虑"完全背包DP"

---

## 🔑 解法一:自底向上DP(完全背包)

### 思路
从金额0开始逐步填表,对于每个金额,尝试用每种硬币,取最小值。

### 图解过程

```
示例:coins = [1, 2, 5], amount = 11

初始化 dp 数组(dp[i]表示凑成金额i的最少硬币数):
dp = [0, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞, ∞]
      0  1  2  3  4  5  6  7  8  9 10 11

遍历每种硬币:

coin=1时,更新所有>=1的金额:
dp[1] = min(∞, dp[0]+1) = 1
dp[2] = min(∞, dp[1]+1) = 2
...
dp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

coin=2时,更新所有>=2的金额:
dp[2] = min(2, dp[0]+1) = 1  (用1个2元代替2个1元)
dp[3] = min(3, dp[1]+1) = 2  (1+2=3,用2个硬币)
dp[4] = min(4, dp[2]+1) = 2  (2+2=4,用2个硬币)
...
dp = [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6]

coin=5时,更新所有>=5的金额:
dp[5] = min(3, dp[0]+1) = 1  (用1个5元)
dp[6] = min(3, dp[1]+1) = 2  (5+1=6,用2个硬币)
dp[10]= min(5, dp[5]+1) = 2  (5+5=10)
dp[11]= min(6, dp[6]+1) = 3  (5+5+1=11) ✅
...
最终 dp = [0, 1, 1, 2, 2, 1, 2, 2, 3, 3, 2, 3]
```

### Python代码

```python
from typing import List


def coin_change(coins: List[int], amount: int) -> int:
    """
    解法一:自底向上DP(完全背包)
    思路:从金额0到amount逐步填表,每个金额尝试所有硬币取最小值
    """
    # 初始化:dp[i]表示凑成金额i的最少硬币数
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0  # 金额0需要0个硬币

    # 遍历每种硬币(外层循环)
    for coin in coins:
        # 更新所有能用这个硬币的金额(内层循环)
        for i in range(coin, amount + 1):
            # 状态转移:当前金额i = 不用当前硬币 vs 用当前硬币后剩余金额+1
            dp[i] = min(dp[i], dp[i - coin] + 1)

    # 返回结果:如果dp[amount]仍为无穷大,说明无解
    return dp[amount] if dp[amount] != float('inf') else -1


# ✅ 测试
print(coin_change([1, 2, 5], 11))  # 期望输出:3 (5+5+1)
print(coin_change([2], 3))         # 期望输出:-1 (无解)
print(coin_change([1], 0))         # 期望输出:0 (金额0)
print(coin_change([1, 3, 4], 6))   # 期望输出:2 (3+3,而非4+1+1)
```

### 复杂度分析
- **时间复杂度**:O(amount × n) — n是硬币种类数,amount是金额。需要填写amount+1个位置,每个位置尝试n种硬币。
  - 具体地说:如果硬币种类n=3,金额amount=10000,大约需要 3 × 10000 = 30000 次操作
- **空间复杂度**:O(amount) — dp数组长度为amount+1

### 优缺点
- ✅ 时间复杂度已达最优,每个子问题只算一次
- ✅ 代码简洁,易于理解和记忆
- ✅ 可扩展到"求有多少种组合"等变体
- ⚠️ 空间O(amount),对于超大金额可能需要优化(但本题amount<=10⁴足够)

---

## 🏆 解法二:记忆化递归(自顶向下DP)(最优解)

### 优化思路
从解法一的迭代改为递归+缓存,思路更直观:对于金额x,尝试每种硬币,递归求解剩余金额,用字典记录已算过的结果。

> 💡 **关键想法**:递归写法更符合人类思维(先分解问题再组合),加上缓存后效率与迭代DP完全相同!

### 图解过程

```
示例:coins = [1, 2, 5], amount = 11

递归树(带缓存):
              coin_change(11)
           /       |        \
       coin=1   coin=2    coin=5
        /          |          \
    dfs(10)     dfs(9)      dfs(6) ← 这些结果会被缓存
      |           |           |
   返回2        返回4        返回2
      |           |           |
   10=5+5      9用4个硬币   6=5+1

最终:min(dfs(10)+1, dfs(9)+1, dfs(6)+1) = min(3, 5, 3) = 3
```

### Python代码

```python
from typing import List
from functools import lru_cache


def coin_change_memo(coins: List[int], amount: int) -> int:
    """
    解法二:记忆化递归(自顶向下DP) 🏆
    思路:递归尝试每种硬币,用@lru_cache自动缓存结果
    """
    @lru_cache(None)
    def dfs(remain: int) -> int:
        """返回凑成金额remain的最少硬币数"""
        # 递归边界
        if remain == 0:
            return 0  # 金额0需要0个硬币
        if remain < 0:
            return float('inf')  # 金额为负,无解

        # 尝试每种硬币,取最小值
        res = float('inf')
        for coin in coins:
            sub = dfs(remain - coin)  # 递归求解剩余金额
            if sub != float('inf'):
                res = min(res, sub + 1)

        return res

    result = dfs(amount)
    return result if result != float('inf') else -1


# ✅ 测试
print(coin_change_memo([1, 2, 5], 11))  # 期望输出:3
print(coin_change_memo([2], 3))         # 期望输出:-1
print(coin_change_memo([1], 0))         # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(amount × n) — 与迭代DP相同,每个金额只计算一次
- **空间复杂度**:O(amount) — 递归栈深度 + 缓存字典

---

## 🐍 Pythonic 写法

利用 Python 的 `@lru_cache` 装饰器和列表推导式:

```python
from functools import lru_cache

def coin_change_pythonic(coins, amount):
    @lru_cache(None)
    def dp(amt):
        if amt == 0: return 0
        if amt < 0: return float('inf')
        return min((dp(amt - c) for c in coins), default=float('inf')) + 1

    res = dp(amount)
    return res if res < float('inf') else -1
```

解释:
- `min(..., default=float('inf'))` 处理coins为空的边界
- 生成器表达式 `(dp(amt - c) for c in coins)` 比列表推导更省内存
- 一行状态转移,代码极简

> ⚠️ **面试建议**:先写清晰的迭代DP展示思路,再提Pythonic写法展示语言功底。面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:迭代DP | 🏆 解法二:记忆化递归(最优) |
|------|--------------|--------------------------|
| 时间复杂度 | O(amount × n) | **O(amount × n)** ← 相同 |
| 空间复杂度 | O(amount) | **O(amount)** ← 相同 |
| 代码难度 | 中等 | **简单** ← 递归更直观 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 空间优化版本 | **通用场景,代码简洁** |

**为什么解法二是最优解**:
- 时间空间复杂度与迭代DP完全相同
- 递归写法更符合人类思维,面试时更容易想到和讲解
- `@lru_cache` 是Python内置工具,代码极简且不易出错
- 可以轻松处理复杂状态转移(如多维DP)

**面试建议**:
1. 先用30秒口述暴力递归(指数级),展示你理解问题本质
2. 立即优化到🏆记忆化递归:"加个缓存就能从指数降到O(amount×n)"
3. **重点讲解状态定义**:`dfs(remain)` 表示凑成金额remain的最少硬币数
4. 强调完全背包特点:每种硬币可以重复使用,所以内层循环正序
5. 手动测试边界用例(amount=0, 无解情况)

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这是一道经典的完全背包问题。要求用最少硬币数凑成目标金额,每种硬币可以重复使用。

我的第一个想法是暴力递归:对于金额 `amount`,尝试每种硬币,递归求解剩余金额,取最小值。但这样时间复杂度是指数级的,因为大量重复计算。

优化方法是用记忆化递归,加一个缓存字典记录已算过的结果,这样每个金额只计算一次,时间复杂度降到 O(amount × n)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
from functools import lru_cache

def coin_change(coins, amount):
    @lru_cache(None)  # 自动缓存递归结果
    def dfs(remain):
        if remain == 0: return 0  # 金额0需要0个硬币
        if remain < 0: return float('inf')  # 无解

        # 尝试每种硬币,取最小值
        res = float('inf')
        for coin in coins:
            res = min(res, dfs(remain - coin) + 1)
        return res

    result = dfs(amount)
    return result if result < float('inf') else -1
```

核心是 `dfs(remain)` 表示凑成金额remain的最少硬币数,递归尝试每种硬币,用 `@lru_cache` 自动缓存结果避免重复计算。

**面试官**:测试一下?

**你**:用示例 `coins=[1,2,5], amount=11` 走一遍...dfs(11)会尝试硬币5,递归到dfs(6),继续尝试硬币5到dfs(1),最后dfs(1)用硬币1返回1,回溯得到 dfs(6)=2, dfs(11)=3。

再测边界 `amount=0` 返回0,`coins=[2], amount=3` 无解返回-1,结果正确!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | "时间已经是O(amount×n)最优,每个金额必须至少看一次。空间可以优化到O(n)用BFS,但代码复杂且实际提升不大" |
| "如果硬币面额非常大呢?" | "可以先对coins排序,在递归时剪枝:如果当前硬币>remain就跳过。但渐进复杂度不变" |
| "能求出具体用了哪些硬币吗?" | "可以!在dp数组旁边维护一个path数组,记录每个金额是由哪个硬币转移来的,最后回溯path即可" |
| "完全背包和0-1背包有什么区别?" | "0-1背包每个物品只能用一次,内层循环倒序;完全背包可以重复使用,内层循环正序。本题是完全背包" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:@lru_cache 自动缓存递归结果
from functools import lru_cache
@lru_cache(None)  # None表示无限缓存大小
def dfs(x):
    pass  # 递归函数会自动缓存

# 技巧2:float('inf') 表示无穷大
dp = [float('inf')] * n  # 初始化为无穷大方便取min
if dp[i] != float('inf'):  # 判断是否有解

# 技巧3:三目运算符简化返回
return result if result < float('inf') else -1
```

### 💡 底层原理(选读)

> **为什么是完全背包而非0-1背包?**
> - 0-1背包:每个物品只能用一次,内层循环**倒序**遍历(避免重复使用)
> - 完全背包:每个物品可以用无限次,内层循环**正序**遍历(允许重复使用)
>
> 本题中每种硬币可以无限次使用,所以是完全背包。
>
> **为什么记忆化递归和迭代DP效率相同?**
> - 记忆化递归:自顶向下,只计算需要的状态(惰性求值)
> - 迭代DP:自底向上,计算所有状态(提前求值)
> - 本题中所有状态都会被访问,所以两者效率相同。但在某些题目中(如斐波那契第n项),记忆化递归可能更快因为不需要计算所有状态。

### 算法模式卡片 📐
- **模式名称**:完全背包DP
- **适用条件**:求最值(最多/最少)+可重复使用资源+刚好达到目标
- **识别关键词**:"最少硬币数"、"每种硬币无限"、"凑成金额"
- **模板代码**:
```python
def complete_knapsack(items, target):
    dp = [float('inf')] * (target + 1)
    dp[0] = 0

    for item in items:  # 外层:每个物品
        for i in range(item, target + 1):  # 内层:正序遍历
            dp[i] = min(dp[i], dp[i - item] + 1)

    return dp[target] if dp[target] < float('inf') else -1
```

### 易错点 ⚠️
1. **初始化错误**:忘记设置 `dp[0] = 0`,导致所有金额都无法转移
   - 错误:`dp = [float('inf')] * (amount + 1)`
   - 正确:`dp[0] = 0; dp = [0] + [float('inf')] * amount`

2. **内层循环顺序错误**:完全背包必须正序,写成倒序会变成0-1背包
   - 错误:`for i in range(amount, coin - 1, -1)`
   - 正确:`for i in range(coin, amount + 1)`

3. **返回值判断错误**:忘记处理无解情况,直接返回 `dp[amount]` 可能是inf
   - 错误:`return dp[amount]`
   - 正确:`return dp[amount] if dp[amount] < float('inf') else -1`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:电商优惠券组合问题 — 用户有多种面额优惠券(可重复使用),如何用最少优惠券凑够满减额度?
- **场景2**:游戏资源兑换系统 — 玩家有不同价值的货币(金币/钻石/积分),如何用最少货币兑换目标物品?
- **场景3**:物流装箱问题 — 有不同规格纸箱(可重复使用),如何用最少纸箱装下所有货物?(变体:求最少容器数)

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 279. 完全平方数 | Medium | 完全背包 | 硬币换成完全平方数,完全相同解法 |
| LeetCode 518. 零钱兑换II | Medium | 完全背包求方案数 | dp[i]改为统计组合数而非最小值 |
| LeetCode 377. 组合总和IV | Medium | 完全背包求排列数 | 内外层循环对调 |
| LeetCode 983. 最低票价 | Medium | 完全背包变体 | 每天可以选择不同时长的票 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:现在硬币不是无限的了,每种硬币只有 `counts[i]` 个,问最少需要几个硬币凑成amount?(多重背包)

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

多重背包可以转化为0-1背包:将每种硬币拆成 `counts[i]` 个独立物品,然后内层循环倒序遍历。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def coin_change_limited(coins, counts, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin, count in zip(coins, counts):
        # 倒序遍历(0-1背包),每种硬币最多用count次
        for i in range(amount, coin - 1, -1):
            for k in range(1, count + 1):
                if i >= k * coin:
                    dp[i] = min(dp[i], dp[i - k * coin] + k)

    return dp[amount] if dp[amount] < float('inf') else -1
```

核心思路:倒序遍历保证每种硬币只用一次,内层k循环枚举使用0~count个该硬币。

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
