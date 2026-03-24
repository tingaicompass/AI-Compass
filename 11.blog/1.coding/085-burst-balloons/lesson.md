> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第85课:戳气球

> **模块**:动态规划 | **难度**:Hard ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/burst-balloons/
> **前置知识**:第71-84课(动态规划基础)
> **预计学习时间**:40分钟

---

## 🎯 题目描述

给你一个数组`nums`,代表一排气球,每个气球上有一个数字。你可以戳破气球获得金币。当戳破气球`i`时,你获得`nums[i-1] × nums[i] × nums[i+1]`枚金币(边界视为1)。戳破后,左右两侧气球会相邻。求能获得金币的最大数量。

**示例:**
```
输入:nums = [3,1,5,8]
输出:167
解释:
戳破1: 3 × 1 × 5 = 15
戳破5: 3 × 5 × 8 = 120
戳破3: 1 × 3 × 8 = 24
戳破8: 1 × 8 × 1 = 8
总计:15 + 120 + 24 + 8 = 167
```

**约束条件:**
- 1 ≤ nums.length ≤ 300
- 0 ≤ nums[i] ≤ 100
- 需要找到最优戳破顺序以获得最大金币

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1] | 1 | 单个气球 |
| 两个元素 | nums=[3,1] | 6 (戳1再戳3:1×1×3+1×3×1) | 顺序影响 |
| 含零元素 | nums=[0,1,0] | 1 | 零值处理 |
| 大规模 | n=300 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你要拆除一排老房子,每拆一栋能获得报酬,但报酬取决于两侧邻居房的价值。
>
> 🐌 **笨办法**:尝试所有拆除顺序(6个房子就有720种顺序),用暴力回溯穷举,时间复杂度O(n!)会爆炸。
>
> 🚀 **聪明办法**:换个思路——不考虑"先戳哪个",而是假设"最后戳哪个"。把问题变成"区间内最后戳k号气球,左右两侧是独立子问题",用区间DP自底向上求解。

### 关键洞察

**逆向思考:不是"先戳哪个",而是"最后戳哪个"!这样左右区间互不影响,可以独立求解。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:`nums`数组,表示气球上的数字
- **输出**:整数,能获得的最大金币数
- **限制**:戳破顺序会影响结果,边界外视为数字1

### Step 2:先想笨办法(暴力法)

对于每个位置,尝试先戳它,然后递归处理剩余气球。
- 时间复杂度:O(n!) — n个气球有n!种排列
- 瓶颈在哪:戳破一个气球后,两侧气球变相邻,状态变化复杂,难以用DP表达

### Step 3:瓶颈分析 → 优化方向

暴力法的问题是"先戳"导致两侧气球关系变化,无法定义清晰的子问题。
- 核心问题:戳破顺序导致状态转移复杂
- 优化思路:**逆向思考** — 假设某个气球是区间内"最后戳破"的,这时左右两侧气球固定,子问题独立

### Step 4:选择武器

- 选用:**区间DP**(Interval DP)
- 理由:
  1. 逆向假设"最后戳k号气球",此时`i-1`和`j+1`号气球还在
  2. 定义`dp[i][j]`为戳破区间`(i, j)`内所有气球(不含i和j)能获得的最大金币
  3. 状态转移:`dp[i][j] = max(dp[i][k] + nums[i]*nums[k]*nums[j] + dp[k][j])`,枚举k为最后戳破的气球

> 🔑 **模式识别提示**:当题目涉及"区间操作"且顺序影响结果时,优先考虑"区间DP + 逆向思考"

---

## 🔑 解法一:回溯穷举(直觉法)

### 思路

尝试每个位置作为下一个戳破的气球,递归计算剩余气球的最优解。(此解法仅用于理解问题,实际会超时)

### 图解过程

```
输入:nums = [3,1,5,8]

Step 1:尝试先戳3
  剩余:[1,5,8] → 递归计算

Step 2:尝试先戳1
  剩余:[3,5,8] → 递归计算

...依次尝试所有顺序

问题:状态空间巨大,会TLE
```

### Python代码

```python
from typing import List


def maxCoins_backtrack(nums: List[int]) -> int:
    """
    解法一:回溯穷举
    思路:尝试所有戳破顺序,递归计算最大值
    """
    def backtrack(arr):
        if not arr:
            return 0
        max_coins = 0
        for i in range(len(arr)):
            # 戳破第i个气球
            left = arr[i - 1] if i > 0 else 1
            right = arr[i + 1] if i < len(arr) - 1 else 1
            coins = left * arr[i] * right
            # 递归处理剩余气球
            new_arr = arr[:i] + arr[i+1:]
            max_coins = max(max_coins, coins + backtrack(new_arr))
        return max_coins

    return backtrack(nums)


# ✅ 测试
print(maxCoins_backtrack([3,1,5,8]))  # 期望输出:167
print(maxCoins_backtrack([1,5]))      # 期望输出:10
```

### 复杂度分析

- **时间复杂度**:O(n!) — 每层递归尝试n种选择,深度为n
  - 具体地说:如果n=10,大约需要10! = 3,628,800次操作,不可接受
- **空间复杂度**:O(n²) — 递归栈深度n,每层创建新数组

### 优缺点

- ✅ 思路直观,易于理解
- ❌ 时间复杂度爆炸,n>10就会超时,无法通过OJ

---

## 🏆 解法二:区间DP(最优解)

### 优化思路

关键洞察:**逆向思考** — 不考虑"先戳哪个",而是假设"最后戳哪个"。

定义`dp[i][j]`为戳破开区间`(i, j)`内所有气球能获得的最大金币(不含边界i和j)。
枚举区间内每个位置k作为"最后戳破"的气球,此时左右两侧气球都还在,金币为`nums[i]*nums[k]*nums[j]`。

> 💡 **关键想法**:假设k是区间`(i,j)`内最后戳的,那么它左边`(i,k)`和右边`(k,j)`是独立的子问题!

### 图解过程

```
输入:nums = [3,1,5,8]
添加虚拟边界:nums = [1,3,1,5,8,1]

初始化:dp[i][j] = 0 (所有区间)

区间长度len=3(即包含1个真实气球)的情况:
dp[0][2]: 区间(0,2)即只有气球1(值为3)
  假设k=1最后戳:1*3*1=3
  dp[0][2] = 3

dp[1][3]: 区间(1,3)即只有气球2(值为1)
  假设k=2最后戳:3*1*1=3
  dp[1][3] = 3

...

区间长度len=4(即包含2个真实气球):
dp[0][3]: 区间(0,3)包含气球[3,1]
  k=1最后戳:dp[0][1]+1*3*1+dp[1][3]=0+3+3=6
  k=2最后戳:dp[0][2]+1*1*1+dp[2][3]=3+1+0=4
  dp[0][3] = max(6,4)=6

...

最终:dp[0][5] = 戳破所有气球的最大金币 = 167
```

### Python代码

```python
def maxCoins(nums: List[int]) -> int:
    """
    解法二:区间DP(最优解)
    思路:逆向假设"最后戳哪个",区间DP求最大值
    """
    # 添加虚拟边界,简化边界处理
    nums = [1] + nums + [1]
    n = len(nums)

    # dp[i][j] = 戳破开区间(i,j)内所有气球能获得的最大金币
    dp = [[0] * n for _ in range(n)]

    # 从小区间到大区间枚举
    for length in range(3, n + 1):  # 区间长度至少为3(含两个虚拟边界)
        for i in range(n - length + 1):
            j = i + length - 1
            # 枚举区间(i,j)内最后戳破的气球k
            for k in range(i + 1, j):
                # 最后戳k时,左右两侧气球i和j还在
                coins = dp[i][k] + nums[i] * nums[k] * nums[j] + dp[k][j]
                dp[i][j] = max(dp[i][j], coins)

    return dp[0][n - 1]


# ✅ 测试
print(maxCoins([3,1,5,8]))  # 期望输出:167
print(maxCoins([1,5]))      # 期望输出:10
print(maxCoins([1]))        # 期望输出:1
```

### 复杂度分析

- **时间复杂度**:O(n³) — 三层循环:区间长度O(n),起点O(n),枚举k O(n)
  - 具体地说:n=300时,大约需要300³ = 27,000,000次操作,在1秒内可以完成
- **空间复杂度**:O(n²) — DP表的大小

### 为什么是最优解

- ✅ 时间复杂度O(n³)是区间DP问题的理论最优解
- ✅ 空间O(n²)合理,DP表必须存储所有子区间的结果
- ✅ 代码清晰,符合区间DP的标准模板
- ✅ 通过"逆向思考"巧妙化解了正向思考的状态转移难题

---

## 🐍 Pythonic 写法

利用Python的`itertools`和`lru_cache`可以写出记忆化递归版本:

```python
from functools import lru_cache

def maxCoins_memo(nums: List[int]) -> int:
    """记忆化递归版本:自顶向下的区间DP"""
    nums = [1] + nums + [1]

    @lru_cache(None)
    def dp(i: int, j: int) -> int:
        """返回戳破开区间(i,j)内所有气球的最大金币"""
        if i + 1 == j:  # 区间内没有气球
            return 0
        max_coins = 0
        for k in range(i + 1, j):
            coins = dp(i, k) + nums[i] * nums[k] * nums[j] + dp(k, j)
            max_coins = max(max_coins, coins)
        return max_coins

    return dp(0, len(nums) - 1)
```

这个写法用自顶向下的递归思路,更接近人的思维习惯,`@lru_cache`自动处理重复子问题的缓存。

> ⚠️ **面试建议**:先写清晰的自底向上DP版本(解法二)展示思路,再提记忆化递归展示Python功底。
> 面试官更看重你的**DP建模能力**,而非递归写法。

---

## 📊 解法对比

| 维度 | 解法一:回溯穷举 | 🏆 解法二:区间DP(最优) |
|------|--------------|---------------------|
| 时间复杂度 | O(n!) | **O(n³)** ← 时间最优 |
| 空间复杂度 | O(n²) | **O(n²)** ← 相同 |
| 代码难度 | 简单(但会TLE) | 中等(理解逆向思考) |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 仅用于理解问题 | **面试必会,通用性强** |

**为什么是最优解**:
- 时间复杂度O(n³)是区间DP的理论最优(需枚举所有区间和分割点)
- "逆向思考最后戳哪个"是破解此题的核心技巧
- 代码结构清晰,符合区间DP标准模板

**面试建议**:
1. 先花1分钟分析暴力回溯为什么会超时(O(n!)太大)
2. 重点讲解🏆区间DP的核心思想:"不看先戳谁,而是假设最后戳谁"
3. 强调状态定义:`dp[i][j]`表示开区间`(i,j)`(不含i,j)内的最大金币
4. 展示状态转移:枚举k作为最后戳破的气球,`dp[i][j] = max(dp[i][k] + nums[i]*nums[k]*nums[j] + dp[k][j])`
5. 提醒边界处理:添加虚拟边界`[1,...,1]`简化代码

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找到戳破气球的最优顺序,使得获得的金币最大。我的第一个想法是用回溯穷举所有戳破顺序,但这样时间复杂度是O(n!),n=300时会超时。

我注意到这是一个典型的**区间DP问题**。关键洞察是:**逆向思考** — 不考虑先戳哪个,而是假设某个气球是区间内"最后戳破"的。这样,它左右两侧的气球都还在,形成独立的子问题。

我会定义`dp[i][j]`为戳破开区间`(i,j)`内所有气球(不含i和j)能获得的最大金币。状态转移是枚举区间内每个位置k作为最后戳破的气球。时间复杂度优化到O(n³)。

**面试官**:很好,请写一下代码。

**你**:(边写边说关键步骤)
```python
# 1. 添加虚拟边界[1,...,1],简化边界处理
nums = [1] + nums + [1]

# 2. 初始化DP表
dp = [[0] * n for _ in range(n)]

# 3. 从小区间到大区间枚举
for length in range(3, n + 1):
    for i in range(n - length + 1):
        j = i + length - 1
        # 4. 枚举k作为最后戳破的气球
        for k in range(i + 1, j):
            coins = dp[i][k] + nums[i]*nums[k]*nums[j] + dp[k][j]
            dp[i][j] = max(dp[i][j], coins)
```

**面试官**:测试一下?

**你**:用示例`[3,1,5,8]`走一遍...(手动模拟小区间的DP过程)。再测一个边界情况`[1]`,只有一个气球,输出1。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么要逆向思考?" | 正向考虑"先戳谁"会导致两侧气球变相邻,状态转移复杂;逆向假设"最后戳谁"时,两侧边界固定,子问题独立 |
| "能不能O(n²)优化?" | 不能,必须枚举所有区间(O(n²))和每个区间内的分割点(O(n)),最优就是O(n³) |
| "为什么添加虚拟边界?" | 简化边界处理,避免判断`i-1`和`j+1`是否越界,代码更简洁 |
| "这题和矩阵链乘法有什么关系?" | 都是区间DP,状态转移都是枚举分割点k,模式相同 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:添加虚拟边界简化边界判断
nums = [1] + nums + [1]  # 在首尾添加1

# 技巧2:记忆化递归的装饰器写法
from functools import lru_cache

@lru_cache(None)
def dp(i, j):
    # 递归函数,自动缓存结果
    pass

# 技巧3:三层循环枚举区间
for length in range(3, n + 1):  # 区间长度
    for i in range(n - length + 1):  # 起点
        j = i + length - 1  # 终点
        for k in range(i + 1, j):  # 分割点
            # 状态转移
```

### 💡 底层原理(选读)

> **区间DP的核心思想**:
> 1. **子问题定义**:通常定义为`dp[i][j]`表示区间`[i,j]`或`(i,j)`的最优解
> 2. **状态转移**:枚举区间内的分割点k,将大区间拆成两个小区间
> 3. **枚举顺序**:从小区间到大区间(长度从小到大),保证计算大区间时小区间已求解
> 4. **逆向思考技巧**:当正向顺序导致状态复杂时,尝试"最后做什么"的逆向分析
>
> **本题的巧妙之处**:
> - 正向思考"先戳谁"会导致两侧气球关系变化,无法定义清晰的子问题
> - 逆向思考"最后戳谁"时,左右边界固定,子问题独立且可以合并

### 算法模式卡片 📐

- **模式名称**:区间DP(Interval DP)
- **适用条件**:
  1. 问题涉及对一段连续区间的操作
  2. 大问题可以通过分割成小区间求解
  3. 操作顺序影响结果,需要枚举所有可能
- **识别关键词**:
  - "戳气球"、"合并石子"、"括号匹配"
  - "区间操作"、"最优分割"
  - 题目要求最优化(最大/最小)某个区间操作结果
- **模板代码**:
```python
def interval_dp(nums):
    n = len(nums)
    dp = [[0] * n for _ in range(n)]

    # 从小区间到大区间枚举
    for length in range(1, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            # 枚举分割点k
            for k in range(i, j + 1):
                dp[i][j] = max(dp[i][j],
                              dp[i][k] + dp[k+1][j] + cost(i, k, j))

    return dp[0][n-1]
```

### 易错点 ⚠️

1. **状态定义错误**:
   - ❌ 错误:定义`dp[i][j]`为闭区间`[i,j]`的最大金币
   - ✅ 正确:定义为开区间`(i,j)`(不含边界),边界需要保留作为"两侧气球"
   - 原因:假设k是最后戳的,需要`nums[i]*nums[k]*nums[j]`,边界i和j必须保留

2. **区间枚举顺序错误**:
   - ❌ 错误:从大区间到小区间枚举,或从左到右枚举起点
   - ✅ 正确:必须从小区间到大区间(长度从小到大),保证计算大区间时依赖的小区间已计算
   - 原因:区间DP有依赖关系,`dp[i][j]`依赖于更小的`dp[i][k]`和`dp[k][j]`

3. **忘记添加虚拟边界**:
   - ❌ 错误:直接用原数组,需要大量if判断越界
   - ✅ 正确:在首尾添加虚拟边界`[1]`,简化代码
   - 原因:边界气球视为1,添加虚拟边界后无需特殊判断

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:任务调度优化 — 在工厂生产线上,多个工序有依赖关系,求最优执行顺序使得总成本最小,可以用区间DP建模
- **场景2**:DNA序列对齐 — 生物信息学中,对齐两个DNA序列找到最优匹配方式,涉及插入/删除/替换操作的最优化,与区间DP思想类似
- **场景3**:矩阵链乘法优化 — 数据库查询优化器决定多表连接的顺序,使得总计算量最小,经典的区间DP问题

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 1039. 多边形三角剖分的最低得分 | Medium | 区间DP | 枚举三角形的第三个顶点作为分割点 |
| LeetCode 375. 猜数字大小II | Medium | 区间DP | 枚举猜哪个数字,求最坏情况下的最小成本 |
| LeetCode 1000. 合并石子的最低成本 | Hard | 区间DP + 前缀和 | 枚举合并点k,需要前缀和优化区间和计算 |
| LeetCode 96. 不同的二叉搜索树 | Medium | 区间DP(变体) | 枚举根节点,左右子树是独立子问题 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个字符串,你可以在任意位置插入括号,使得最终的表达式计算结果最大。字符串只包含数字和运算符`+`、`-`、`*`。例如`"2*3-4*5"`可以变为`"2*(3-(4*5))"`,求最大值。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

用区间DP,定义`dp[i][j]`为区间`[i,j]`的最大值和最小值(需要同时维护,因为负负得正)。枚举运算符位置k作为分割点。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def maxValue(s: str) -> int:
    """
    区间DP:同时维护最大值和最小值
    """
    n = len(s)
    # 分离数字和运算符
    nums = []
    ops = []
    num = 0
    for ch in s:
        if ch.isdigit():
            num = num * 10 + int(ch)
        else:
            nums.append(num)
            ops.append(ch)
            num = 0
    nums.append(num)

    m = len(nums)
    # dp_max[i][j] = 区间[i,j]的最大值
    # dp_min[i][j] = 区间[i,j]的最小值
    dp_max = [[float('-inf')] * m for _ in range(m)]
    dp_min = [[float('inf')] * m for _ in range(m)]

    # 初始化:单个数字
    for i in range(m):
        dp_max[i][i] = nums[i]
        dp_min[i][i] = nums[i]

    # 从小区间到大区间枚举
    for length in range(2, m + 1):
        for i in range(m - length + 1):
            j = i + length - 1
            # 枚举分割点k(运算符位置)
            for k in range(i, j):
                op = ops[k]
                # 左右两侧的最大最小值
                left_max, left_min = dp_max[i][k], dp_min[i][k]
                right_max, right_min = dp_max[k+1][j], dp_min[k+1][j]

                # 根据运算符计算可能的值
                if op == '+':
                    vals = [left_max + right_max]
                elif op == '-':
                    vals = [left_max - right_min]
                else:  # *
                    vals = [
                        left_max * right_max,
                        left_max * right_min,
                        left_min * right_max,
                        left_min * right_min
                    ]

                dp_max[i][j] = max(dp_max[i][j], max(vals))
                dp_min[i][j] = min(dp_min[i][j], min(vals))

    return dp_max[0][m-1]
```

核心思路:与戳气球类似,枚举运算符作为分割点,但需要同时维护最大最小值(因为负负得正)。

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
