> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第88课:目标和

> **模块**:动态规划 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/target-sum/
> **前置知识**:第75课(零钱兑换)、第79课(分割等和子集)、第59课(全排列-回溯基础)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给你一个非负整数数组 nums 和一个整数 target。向数组中的每个整数前添加 '+' 或 '-' 符号,然后串联起所有整数,可以构造一个表达式。返回可以通过上述方法构造的、运算结果等于 target 的不同表达式的数目。

**示例:**
```
输入: nums = [1,1,1,1,1], target = 3
输出: 5
解释: 一共有5种方法让最终目标和为3。
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3
```

**约束条件:**
- 1 ≤ nums.length ≤ 20
- 0 ≤ nums[i] ≤ 1000
- 0 ≤ sum(nums[i]) ≤ 1000
- -1000 ≤ target ≤ 1000

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1], target=1 | 1 | 单个元素,+1=1 |
| 无解情况 | nums=[1], target=2 | 0 | 无法构造 |
| 含零 | nums=[0,0,1], target=1 | 4 | 0可以是+0或-0,组合数翻倍 |
| 负目标 | nums=[1,2], target=-3 | 1 | -1-2=-3 |
| 大规模 | n=20, nums全是1000 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你有5个骰子,每个骰子都是1点。现在你要通过给每个骰子标记"+"或"-",让它们的和等于3。
>
> 🐌 **笨办法**:尝试所有可能的标记方案。5个骰子,每个有2种选择(+或-),总共2^5=32种组合。一个个试,数一数有多少种和为3。
>
> 🚀 **聪明办法**:换个角度思考!
> - 假设标记为"+"的骰子和为 P,标记为"-"的骰子和为 N
> - 那么 P - N = target,且 P + N = sum(所有骰子)
> - 推导出 P = (target + sum) / 2
>
> **问题转化**:从5个骰子中选出一些,使它们的和恰好等于 P。这就变成了经典的"0-1背包"问题!我们只需要计算"有多少种方法凑出和为P",而不是枚举所有2^n种组合。

### 关键洞察
**核心突破口:问题可以转化为0-1背包的"方案数"问题 — 从数组中选出一些数,使其和等于 (target+sum)/2。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:非负整数数组 nums,整数 target
- **输出**:有多少种添加+/-符号的方法,使得表达式结果等于 target
- **限制**:
  - 数组长度最大20,每个元素最大1000
  - 每个数字前必须添加+或-符号

### Step 2:先想笨办法(回溯法)
用回溯枚举所有可能的+/-组合:对每个数字,选择添加+或-,然后递归处理下一个数字,最后统计和为target的方案数。
- 时间复杂度:O(2^n) — 每个数字2种选择,n=20时是100万+种组合
- 瓶颈在哪:大量重复计算,比如"前3个数和为5"这个状态可能被重复访问多次

### Step 3:瓶颈分析 → 优化方向
回溯法中,同样的"前k个数的和"被反复计算。比如:
- 路径1:+1+2-3 → 和=0
- 路径2:-1-2+3 → 和=0
- 这两条路径到达了相同的状态(前3个数和为0),但后续的计算是独立进行的,造成浪费

**数学转化**:
- 设正数和为P,负数和为N(绝对值)
- 则 P - N = target,且 P + N = sum
- 推导:P = (target + sum) / 2

**核心问题**:从数组中选出一些数,使其和等于P,有多少种选法?
**优化思路**:这是0-1背包的"方案数"问题,用DP解决!

### Step 4:选择武器
- 选用:**0-1背包DP(方案数变体)**
- 理由:
  1. 问题转化为"子集和等于目标值的方案数"
  2. 每个数字选或不选,符合0-1背包特征
  3. DP可以避免重复计算,从O(2^n)降到O(n×sum)

> 🔑 **模式识别提示**:当题目出现"每个元素选或不选"+"统计方案数"时,优先考虑"0-1背包DP"

---

## 🔑 解法一:回溯法(直觉解法)

### 思路
用回溯枚举所有2^n种+/-组合,统计和为target的方案数。虽然慢,但逻辑直接,适合理解题意。

### 图解过程

```
示例: nums = [1, 1, 1], target = 1

决策树(深度优先搜索):
                     []
            /                  \
        +1(sum=1)            -1(sum=-1)
        /      \              /      \
    +1(2)    -1(0)        +1(0)    -1(-2)
    /  \      /  \        /  \      /  \
  +1  -1    +1  -1      +1  -1    +1  -1
  (3) (1)✓ (1)✓(-1)    (1)✓(-1)  (1)✓(-3)

找到4条路径和为1:
1. +1 +1 -1 = 1
2. +1 -1 +1 = 1
3. -1 +1 +1 = 1
4. +1 +1 -1 = 1(重复计数,实际是3种)
```

### Python代码

```python
from typing import List


def findTargetSumWays_backtrack(nums: List[int], target: int) -> int:
    """
    解法一:回溯法
    思路:枚举所有+/-组合,统计和为target的方案数
    """
    def backtrack(index: int, current_sum: int) -> int:
        # 递归终止:处理完所有数字
        if index == len(nums):
            return 1 if current_sum == target else 0

        # 选择1:添加+号
        count_plus = backtrack(index + 1, current_sum + nums[index])
        # 选择2:添加-号
        count_minus = backtrack(index + 1, current_sum - nums[index])

        return count_plus + count_minus

    return backtrack(0, 0)


# ✅ 测试
print(findTargetSumWays_backtrack([1, 1, 1, 1, 1], 3))  # 期望输出:5
print(findTargetSumWays_backtrack([1], 1))              # 期望输出:1
print(findTargetSumWays_backtrack([1], 2))              # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(2^n) — 每个数字2种选择,n=20时约100万次递归
  - 具体地说:如果 n=20,大约需要 2^20 ≈ 1,048,576 次递归调用
- **空间复杂度**:O(n) — 递归调用栈深度

### 优缺点
- ✅ 逻辑直接,易于理解
- ✅ 代码简洁,面试中可以快速写出
- ❌ 时间复杂度高,n≥15时会超时
- ❌ 大量重复计算相同的子问题

---

## ⚡ 解法二:回溯+记忆化(优化)

### 优化思路
在回溯基础上,用哈希表记录已计算过的状态(index, current_sum),避免重复计算。

> 💡 **关键想法**:相同的(位置,当前和)状态只需要计算一次,结果可以复用

### Python代码

```python
def findTargetSumWays_memo(nums: List[int], target: int) -> int:
    """
    解法二:回溯+记忆化
    思路:用字典缓存(index, sum)的计算结果
    """
    memo = {}  # 记忆化字典: (index, current_sum) -> 方案数

    def backtrack(index: int, current_sum: int) -> int:
        # 递归终止
        if index == len(nums):
            return 1 if current_sum == target else 0

        # 查缓存
        if (index, current_sum) in memo:
            return memo[(index, current_sum)]

        # 递归计算
        count = (
            backtrack(index + 1, current_sum + nums[index]) +
            backtrack(index + 1, current_sum - nums[index])
        )

        # 存缓存
        memo[(index, current_sum)] = count
        return count

    return backtrack(0, 0)


# ✅ 测试
print(findTargetSumWays_memo([1, 1, 1, 1, 1], 3))  # 期望输出:5
```

### 复杂度分析
- **时间复杂度**:O(n × sum) — 最多有 n×sum 种不同的(index,sum)状态
- **空间复杂度**:O(n × sum) — 记忆化字典和递归栈

---

## 🏆 解法三:动态规划 — 0-1背包(最优解)

### 优化思路
通过数学推导,将问题转化为0-1背包:
- 设正数和为 P,负数和为 N(绝对值)
- P - N = target
- P + N = sum
- 推导出 **P = (target + sum) / 2**

问题转化:**从数组中选出一些数,使其和等于P,有多少种选法?**

这是经典的0-1背包"方案数"问题,可以用DP数组高效求解。

> 💡 **关键想法**:数学转化将O(2^n)的枚举问题降为O(n×P)的DP问题

### 图解过程

```
示例: nums = [1, 1, 1, 1, 1], target = 3

Step 1: 计算目标和 P
  sum = 5, target = 3
  P = (3 + 5) / 2 = 4

Step 2: 问题转化
  "从[1,1,1,1,1]中选数,和为4,有多少种选法?"

Step 3: DP定义
  dp[j] = 和为j的方案数

初始化: dp = [1, 0, 0, 0, 0]  (和为0有1种方法:什么都不选)

处理第1个数(1):
  dp[4] = dp[4] + dp[3] = 0 + 0 = 0
  dp[3] = dp[3] + dp[2] = 0 + 0 = 0
  dp[2] = dp[2] + dp[1] = 0 + 0 = 0
  dp[1] = dp[1] + dp[0] = 0 + 1 = 1
  结果: dp = [1, 1, 0, 0, 0]

处理第2个数(1):
  dp[4] = dp[4] + dp[3] = 0 + 0 = 0
  dp[3] = dp[3] + dp[2] = 0 + 0 = 0
  dp[2] = dp[2] + dp[1] = 0 + 1 = 1
  dp[1] = dp[1] + dp[0] = 1 + 1 = 2
  结果: dp = [1, 2, 1, 0, 0]

处理第3个数(1):
  dp = [1, 3, 3, 1, 0]

处理第4个数(1):
  dp = [1, 4, 6, 4, 1]

处理第5个数(1):
  dp = [1, 5, 10, 10, 5]

答案: dp[4] = 5
```

### Python代码

```python
def findTargetSumWays(nums: List[int], target: int) -> int:
    """
    解法三:动态规划 — 0-1背包(最优解)
    思路:转化为"子集和为P的方案数"问题
    """
    total_sum = sum(nums)

    # 剪枝1:如果target的绝对值大于sum,无解
    if abs(target) > total_sum:
        return 0

    # 剪枝2:如果(target+sum)是奇数,无解(P必须是整数)
    if (target + total_sum) % 2 == 1:
        return 0

    # 计算目标正数和
    P = (target + total_sum) // 2

    # DP定义: dp[j] = 和为j的方案数
    dp = [0] * (P + 1)
    dp[0] = 1  # 和为0的方案数是1(什么都不选)

    # 0-1背包:每个数字选或不选
    for num in nums:
        # 倒序遍历,避免重复使用同一个数字
        for j in range(P, num - 1, -1):
            dp[j] += dp[j - num]

    return dp[P]


# ✅ 测试
print(findTargetSumWays([1, 1, 1, 1, 1], 3))  # 期望输出:5
print(findTargetSumWays([1], 1))              # 期望输出:1
print(findTargetSumWays([1, 0], 1))           # 期望输出:2 (注意0的处理)
print(findTargetSumWays([100], -200))         # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n × P) — n是数组长度,P是目标和,最大值是sum/2
  - 具体地说:如果 n=20,sum=1000,P=500,大约需要 20×500=10,000 次操作
- **空间复杂度**:O(P) — DP数组长度

### 为什么是最优解
- ✅ **时间从O(2^n)降到O(n×P)**:n=20,sum=1000时,从100万降到1万,提升100倍
- ✅ **空间O(P)非常节省**:只需要一维DP数组,比记忆化更优
- ✅ **数学转化巧妙**:将复杂的符号问题转化为简单的子集和问题
- ✅ **代码简洁**:核心逻辑只有10行,易于理解和实现

---

## 🐍 Pythonic 写法

利用 Python 的 sum() 和简洁语法:

```python
def findTargetSumWays_pythonic(nums: List[int], target: int) -> int:
    """Pythonic 写法:一行计算P,简化剪枝"""
    total = sum(nums)
    if abs(target) > total or (target + total) % 2:
        return 0

    P = (target + total) // 2
    dp = [1] + [0] * P

    for num in nums:
        dp = [dp[j] + (dp[j - num] if j >= num else 0) for j in range(P + 1)]

    return dp[P]


# ✅ 测试
print(findTargetSumWays_pythonic([1, 1, 1, 1, 1], 3))  # 期望输出:5
```

这个写法利用了:
- **列表推导式**:一行更新DP数组,代码更简洁
- **三元表达式**:避免索引越界检查

> ⚠️ **面试建议**:Pythonic写法虽然简洁,但可读性略差。面试中建议先写清晰版本,展示思路后再提这个优化。

---

## 📊 解法对比

| 维度 | 解法一:回溯 | 解法二:记忆化 | 🏆 解法三:DP背包(最优) |
|------|-----------|-------------|---------------------|
| 时间复杂度 | O(2^n) | O(n×sum) | **O(n×P)** ← P≤sum/2 |
| 空间复杂度 | O(n) | O(n×sum) | **O(P)** ← 空间最优 |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | n≤15 | n≤20,sum小 | **通用,性能最佳** |

**为什么是最优解**:
- 时间复杂度从指数级O(2^n)降到多项式级O(n×P),提升巨大
- 空间复杂度O(P)远小于记忆化的O(n×sum)
- 数学转化将问题简化,代码更简洁易懂

**面试建议**:
1. 先口述回溯思路,说明暴力法是O(2^n)
2. 立即提出数学转化:P=(target+sum)/2,将问题转化为0-1背包
3. 写出🏆最优解(DP背包),展示对背包问题的深刻理解
4. 强调为什么是最优:时间空间都大幅优化,且逻辑优雅

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道目标和问题。

**你**:(审题30秒)好的,这道题要求给每个数字添加+或-符号,使表达式结果等于target,返回方案数。

我的第一个想法是用回溯枚举所有2^n种符号组合,但时间复杂度太高。

不过我注意到一个数学技巧:设正数和为P,负数和为N,则 P-N=target,P+N=sum,推导出 P=(target+sum)/2。

问题就转化为:**从数组中选数,和为P,有多少种选法?**这是0-1背包的方案数问题,可以用DP解决,时间O(n×P)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)首先处理边界情况:
1. 如果 |target| > sum,无解
2. 如果 (target+sum) 是奇数,P不是整数,无解

然后定义 dp[j] 表示和为j的方案数,初始化 dp[0]=1。

用0-1背包的模板,倒序遍历避免重复使用同一个数,状态转移方程是 `dp[j] += dp[j-num]`。

最后返回 dp[P]。

**面试官**:测试一下?

**你**:用示例 [1,1,1,1,1], target=3 走一遍...

sum=5, P=(3+5)/2=4。初始化 dp=[1,0,0,0,0]。

处理第1个1: dp=[1,1,0,0,0]
处理第2个1: dp=[1,2,1,0,0]
...
最终 dp[4]=5,结果正确!

再测边界情况 [1], target=2,因为 |2|>1,返回0,也正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么P=(target+sum)/2?" | 设正数和P,负数和N(绝对值),则 P-N=target,P+N=sum。两式相加得 2P=target+sum,所以 P=(target+sum)/2 |
| "为什么倒序遍历?" | 0-1背包要求每个数只用一次。正序遍历会导致 dp[j] 被更新后,dp[j+num] 又用了新的 dp[j],相当于重复使用。倒序保证用的是上一轮的旧值 |
| "如果数组中有0怎么办?" | 0可以是+0或-0,对和没影响,但会让方案数翻倍。DP会自动处理:dp[j] += dp[j-0] = dp[j],相当于方案数乘2 |
| "能不能用滚动数组优化?" | 已经是一维DP数组了,空间O(P)已是最优,无需再优化 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:边界检查 — 提前剪枝,避免无效计算
if abs(target) > total or (target + total) % 2:
    return 0

# 技巧2:整数除法 — 使用 // 避免浮点数
P = (target + total_sum) // 2

# 技巧3:倒序遍历 — 0-1背包核心技巧
for j in range(P, num - 1, -1):
    dp[j] += dp[j - num]
```

### 💡 底层原理(选读)

> **为什么倒序遍历是0-1背包的核心?**
>
> 考虑正序遍历 `for j in range(num, P+1)`:
> - 更新 dp[2] 时,用的是新的 dp[1](已被本轮更新)
> - 这相当于同一个数字被使用了多次,变成了"完全背包"
>
> 而倒序遍历 `for j in range(P, num-1, -1)`:
> - 更新 dp[2] 时,用的是旧的 dp[1](上一轮的值)
> - 保证每个数字只使用一次,符合0-1背包定义
>
> **记忆口诀**:0-1背包倒序,完全背包正序!

### 算法模式卡片 📐
- **模式名称**:0-1背包DP(方案数变体)
- **适用条件**:
  - 从数组中选出一些元素(每个选或不选)
  - 使得某个属性(和/积/异或等)等于目标值
  - 求满足条件的选法数量
- **识别关键词**:"选或不选"、"方案数"、"子集和"、"目标值"
- **模板代码**:
```python
# 0-1背包方案数模板
def count_ways(nums: list[int], target: int) -> int:
    dp = [0] * (target + 1)
    dp[0] = 1  # 和为0的方案数是1

    for num in nums:
        for j in range(target, num - 1, -1):  # 倒序!
            dp[j] += dp[j - num]

    return dp[target]
```

### 易错点 ⚠️
1. **忘记处理(target+sum)为奇数的情况**:如果P不是整数,题目无解,需要提前返回0。
   - **正确做法**:检查 `(target + total_sum) % 2 == 1` 时返回0

2. **正序遍历DP数组**:会导致同一个数字被重复使用,变成完全背包,答案错误。
   - **正确做法**:0-1背包必须倒序遍历,`for j in range(P, num-1, -1)`

3. **忘记处理target为负数的情况**:数学推导依然成立,因为绝对值不影响 P=(target+sum)/2。
   - **正确做法**:检查 `abs(target) > total_sum` 时返回0

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:数据分析中的平衡分组**:给定一组数据,如何分成两组使得两组的某个统计量(均值/方差)之差等于目标值?用0-1背包DP可以快速计算所有可行的分组方案。

- **场景2:负载均衡问题**:有n个任务,每个任务有权重。要将任务分配到两台服务器,使得两台服务器的负载差等于某个值,有多少种分配方案?这就是本题的变体。

- **场景3:游戏设计中的装备搭配**:RPG游戏中,玩家有n件装备,每件装备有属性加成(正数)或减益(负数)。要让最终属性值等于目标值,有多少种装备搭配方案?用本题的方法可以快速计算。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 416. 分割等和子集 | Medium | 0-1背包(存在性) | P=sum/2,求是否存在和为P的子集 |
| LeetCode 1049. 最后一块石头的重量II | Medium | 0-1背包(最小化) | 转化为将石头分成两堆,最小化差值 |
| LeetCode 474. 一和零 | Medium | 二维0-1背包 | 两个维度(0的个数和1的个数)的背包 |
| LeetCode 698. 划分为k个相等的子集 | Medium | 回溯+剪枝 | 无法转化为DP,需要用回溯枚举 |
| LeetCode 1982. 从子集的和还原数组 | Hard | 逆向思维 | 给定所有子集和,还原原数组 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如果题目改为"每个数字可以使用任意次(可以重复选)",应该如何修改代码?

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

这变成了"完全背包"问题!核心修改:将倒序遍历改为正序遍历,允许重复使用同一个数字。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def findTargetSumWays_unbounded(nums: List[int], target: int) -> int:
    """变体:完全背包(每个数字可以重复使用)"""
    total_sum = sum(nums)
    if abs(target) > total_sum or (target + total_sum) % 2:
        return 0

    P = (target + total_sum) // 2
    dp = [0] * (P + 1)
    dp[0] = 1

    for num in nums:
        # 正序遍历 — 完全背包允许重复使用
        for j in range(num, P + 1):
            dp[j] += dp[j - num]

    return dp[P]


# 测试
print(findTargetSumWays_unbounded([1, 2], 3))
# 结果会不同,因为可以重复使用: +1+1+1=3, +1+2=3, +2+1=3 等
```

**核心区别**:
- **0-1背包(每个数只用一次)**:倒序遍历 `for j in range(P, num-1, -1)`
- **完全背包(每个数可重复用)**:正序遍历 `for j in range(num, P+1)`

记住这个规律,就能轻松应对所有背包变体!

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
