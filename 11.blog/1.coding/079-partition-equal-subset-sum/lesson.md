> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第79课:分割等和子集

> **模块**:动态规划 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/partition-equal-subset-sum/
> **前置知识**:第71课(爬楼梯)、第73课(打家劫舍)、第75课(零钱兑换)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定一个正整数数组 `nums`,判断是否可以将这个数组分割成两个子集,使得两个子集的元素和相等。

**示例:**
```
输入:nums = [1, 5, 11, 5]
输出:true
解释:数组可以分割为 [1, 5, 5] 和 [11],两个子集和都是 11
```

**约束条件:**
- `1 <= nums.length <= 200`
- `1 <= nums[i] <= 100`
- 每个元素只能使用一次

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1,1] | true | 基本功能 |
| 奇数和 | nums=[1,2,3] | false | 和为奇数无法分割 |
| 单个元素 | nums=[100] | false | 无法分割 |
| 大规模 | nums=200个元素 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你要把一堆硬币分成两份,使得每份的金额相等。
>
> 🐌 **笨办法**:枚举所有可能的分割方式,计算每种分割的和。如果有100个硬币,可能的分割方式有 2^100 种,计算到宇宙毁灭都算不完。
>
> 🚀 **聪明办法**:先算总金额,如果总金额是奇数,直接知道不可能分成两份相等的。如果是偶数,问题就变成:"能不能从这些硬币中挑出一些,凑成总金额的一半?"这就是经典的背包问题!

### 关键洞察
**分割成两个等和子集 = 从数组中选一些元素,使其和等于总和的一半**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:正整数数组 nums,长度 1~200,元素值 1~100
- **输出**:布尔值,能否分割成两个等和子集
- **限制**:每个元素只能使用一次(0-1背包特征)

### Step 2:先想笨办法(暴力法)
枚举所有可能的子集,计算每个子集的和,看是否存在和等于总和一半的子集。
- 时间复杂度:O(2^n) — n 个元素有 2^n 个子集
- 瓶颈在哪:指数级的枚举量,n=200 时完全无法接受

### Step 3:瓶颈分析 → 优化方向
暴力法中存在大量重复计算。例如计算 `[1,2,3]` 能否凑成6,和计算 `[1,2,4]` 能否凑成6,都会重复判断"前两个元素能凑成多少"。
- 核心问题:对每个元素,都要重新判断"能凑成哪些和"
- 优化思路:能不能记住"前 i 个元素能凑成哪些和",避免重复计算?

### Step 4:选择武器
- 选用:**0-1背包动态规划**
- 理由:这是典型的"从 n 个物品中选若干个,使得某种指标达到目标值"问题,与背包问题本质相同

> 🔑 **模式识别提示**:当题目出现"选/不选某些元素,使得和/乘积达到目标值"时,优先考虑"背包DP"

---

## 🔑 解法一:回溯暴力枚举(直觉法)

### 思路
枚举所有可能的子集,对每个元素选择"选"或"不选",当选中的元素和等于目标值时返回 true。

### 图解过程

```
输入:nums = [1, 5, 11, 5], target = 11 (总和22的一半)

决策树:
                    (0, sum=0)
                   /          \
              选1(1,1)       不选1(1,0)
             /      \         /       \
         选5(2,6) 不选5(2,1) 选5(2,5) 不选5(2,0)
         /    \     /    \     /   \      /    \
       ...   ...  ...   ...  ...  ...   ...   ...

当 sum=11 时返回 true
```

### Python代码

```python
from typing import List


def canPartition_backtrack(nums: List[int]) -> bool:
    """
    解法一:回溯暴力枚举
    思路:对每个元素选择"选"或"不选",看能否凑成目标和
    """
    total = sum(nums)
    if total % 2 != 0:  # 总和为奇数,直接返回 false
        return False

    target = total // 2

    def backtrack(index: int, current_sum: int) -> bool:
        # 达到目标
        if current_sum == target:
            return True
        # 超过目标或遍历完所有元素
        if current_sum > target or index >= len(nums):
            return False

        # 选择当前元素
        if backtrack(index + 1, current_sum + nums[index]):
            return True
        # 不选择当前元素
        if backtrack(index + 1, current_sum):
            return True

        return False

    return backtrack(0, 0)


# ✅ 测试
print(canPartition_backtrack([1, 5, 11, 5]))  # 期望输出:True
print(canPartition_backtrack([1, 2, 3, 5]))   # 期望输出:False
print(canPartition_backtrack([1, 1]))         # 期望输出:True
```

### 复杂度分析
- **时间复杂度**:O(2^n) — 每个元素有选/不选两种选择,共 n 个元素
  - 具体地说:如果输入规模 n=20,大约需要 2^20 = 1,048,576 次操作
- **空间复杂度**:O(n) — 递归调用栈深度

### 优缺点
- ✅ 思路直观,易于理解
- ❌ 时间复杂度指数级,n>20 时会超时,无法通过 LeetCode

---

## 🏆 解法二:0-1背包动态规划(最优解)

### 优化思路
回溯法中存在大量重复子问题。例如"前3个元素能否凑成8"这个问题可能被计算多次。我们可以用 DP 数组记录"前 i 个元素能凑成哪些和",避免重复计算。

> 💡 **关键想法**:定义 dp[j] 表示"能否从数组中选若干元素,使其和等于 j"

### 图解过程

```
nums = [1, 5, 11, 5], target = 11

初始化:dp[0] = True (什么都不选,和为0)
       dp[1..11] = False

处理第1个元素(1):
  倒序遍历 j 从 11 到 1:
    dp[1] = dp[1] or dp[0] = True
  结果:dp = [T, T, F, F, F, F, F, F, F, F, F, F]

处理第2个元素(5):
  倒序遍历 j 从 11 到 5:
    dp[6] = dp[6] or dp[1] = True
    dp[5] = dp[5] or dp[0] = True
  结果:dp = [T, T, F, F, F, T, T, F, F, F, F, F]

处理第3个元素(11):
  dp[11] = dp[11] or dp[0] = True ✅
  结果:dp[11] = True,返回 True
```

### Python代码

```python
def canPartition(nums: List[int]) -> bool:
    """
    解法二:0-1背包动态规划(最优解)
    思路:dp[j] 表示能否从数组中选若干元素使其和等于 j
    """
    total = sum(nums)
    if total % 2 != 0:  # 总和为奇数,无法分割
        return False

    target = total // 2
    # dp[j] 表示:能否凑成和为 j
    dp = [False] * (target + 1)
    dp[0] = True  # 什么都不选,和为0

    # 遍历每个元素
    for num in nums:
        # 倒序遍历(避免重复使用同一元素)
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]


# ✅ 测试
print(canPartition([1, 5, 11, 5]))  # 期望输出:True
print(canPartition([1, 2, 3, 5]))   # 期望输出:False
print(canPartition([1, 1]))         # 期望输出:True
```

### 复杂度分析
- **时间复杂度**:O(n × target) — n 是数组长度,target 是总和的一半
  - 具体地说:如果 n=200,元素最大100,则 target 最大10000,约需 200×10000 = 2,000,000 次操作
- **空间复杂度**:O(target) — DP 数组长度

---

## ⚡ 解法三:记忆化搜索(备选)

### 优化思路
在回溯法基础上加入记忆化,记录每个 `(index, current_sum)` 状态的结果,避免重复计算。

### Python代码

```python
def canPartition_memo(nums: List[int]) -> bool:
    """
    解法三:记忆化搜索
    思路:回溯 + 哈希表缓存已计算状态
    """
    total = sum(nums)
    if total % 2 != 0:
        return False

    target = total // 2
    memo = {}  # (index, current_sum) -> bool

    def dfs(index: int, current_sum: int) -> bool:
        if current_sum == target:
            return True
        if current_sum > target or index >= len(nums):
            return False

        if (index, current_sum) in memo:
            return memo[(index, current_sum)]

        # 选或不选当前元素
        result = (dfs(index + 1, current_sum + nums[index]) or
                  dfs(index + 1, current_sum))

        memo[(index, current_sum)] = result
        return result

    return dfs(0, 0)


# ✅ 测试
print(canPartition_memo([1, 5, 11, 5]))  # 期望输出:True
print(canPartition_memo([1, 2, 3, 5]))   # 期望输出:False
```

### 复杂度分析
- **时间复杂度**:O(n × target) — 与DP相同
- **空间复杂度**:O(n × target) — 递归栈 + 哈希表

---

## 🐍 Pythonic 写法

利用 Python 的 set 集合快速实现 DP:

```python
def canPartition_pythonic(nums: List[int]) -> bool:
    """Pythonic 写法:用 set 记录所有可能的和"""
    total = sum(nums)
    if total % 2 != 0:
        return False

    target = total // 2
    possible = {0}  # 初始只有0可达

    for num in nums:
        # 用集合运算生成新的可能和
        possible |= {x + num for x in possible if x + num <= target}
        if target in possible:
            return True

    return target in possible
```

> ⚠️ **面试建议**:先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:回溯枚举 | 解法二:0-1背包DP | 解法三:记忆化搜索 |
|------|--------------|--------------|--------------|
| 时间复杂度 | O(2^n) | **O(n × target)** ← 最优 | O(n × target) |
| 空间复杂度 | O(n) | **O(target)** ← 最优 | O(n × target) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 只适合小规模数据 | **面试首选,最高效** | 便于理解DP推导 |

**为什么解法二是最优解**:
- 时间复杂度 O(n × target) 已经是背包问题的理论最优
- 空间优化到 O(target),一维滚动数组节省内存
- 代码清晰,面试中容易写对

**面试建议**:
1. 先用30秒口述暴力回溯思路(O(2^n)),表明你能想到基本解法
2. 立即优化到🏆最优解(O(n × target) DP),展示优化能力
3. **重点讲解最优解的核心思想**:"问题转化为0-1背包,用一维DP滚动数组优化"
4. 强调为什么倒序遍历:避免重复使用同一元素
5. 手动测试边界用例(如奇数和、单个元素),展示对解法的深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求判断能否将数组分割成两个等和子集。让我先想一下...

首先,如果数组总和是奇数,肯定无法分成两个等和子集,直接返回 false。

如果总和是偶数,问题就转化为:"能否从数组中选若干元素,使其和等于总和的一半?"这是经典的0-1背包问题。

我的第一个想法是回溯枚举所有子集,时间复杂度是 O(2^n)。但 n 最大200,这会超时。

优化方法是用动态规划,定义 dp[j] 表示"能否凑成和为 j",状态转移方程是 dp[j] = dp[j] or dp[j-num]。时间复杂度优化到 O(n × target)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def canPartition(nums):
    total = sum(nums)
    if total % 2 != 0:
        return False

    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        # 关键:倒序遍历,避免重复使用同一元素
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]

    return dp[target]
```

**面试官**:为什么要倒序遍历?

**你**:因为这是0-1背包,每个元素只能使用一次。如果正序遍历,dp[j-num] 可能已经被更新,导致同一元素被使用多次。倒序遍历保证每次更新 dp[j] 时,dp[j-num] 还是上一轮的值。

**面试官**:测试一下?

**你**:用示例 [1,5,11,5] 走一遍:
- 总和22,target=11
- 处理1:dp[1]=true
- 处理5:dp[5]=true, dp[6]=true
- 处理11:dp[11]=true ✅

再测一个边界情况 [1,2,3]:总和6,是偶数,但无法凑成3(1+2=3),结果应该...等等,1+2=3,应该是 true!
让我重新理解题意...哦,我理解错了,[1,2,3] 可以分成 [1,2] 和 [3],两边和都是3,所以是 true。我的算法是对的。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间 O(n×target) 和空间 O(target) 已经是背包问题的最优解,无法进一步优化 |
| "如果数据量非常大呢?" | 可以考虑剪枝优化:排序后优先选大数,提前达到 target;或用位运算压缩 DP 数组 |
| "空间能不能O(1)?" | 无法做到 O(1),因为必须记录所有可能的和。但可以用 bitset 优化常数 |
| "实际工程中怎么用?" | 资源分配问题(如服务器负载均衡)、切割材料问题(如钢管切割) |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:集合推导式 — 快速生成新的可能和
possible |= {x + num for x in possible if x + num <= target}

# 技巧2:倒序 range — 0-1背包关键
for j in range(target, num - 1, -1):  # 从 target 到 num,步长-1
    dp[j] = dp[j] or dp[j - num]
```

### 💡 底层原理(选读)

> **为什么0-1背包要倒序遍历?**
>
> 在一维DP优化中,dp[j] 依赖 dp[j-num],如果正序遍历:
> - dp[1] 被更新后,计算 dp[2] 时用到的 dp[1] 已经是新值
> - 这导致同一元素被使用多次,违反0-1背包"每个元素只用一次"的约束
>
> 倒序遍历确保:
> - 更新 dp[j] 时,所有 dp[k] (k<j) 都还是上一轮的旧值
> - 每个元素对每个状态只影响一次
>
> **完全背包 vs 0-1背包**:
> - 完全背包(元素可重复使用):正序遍历
> - 0-1背包(元素只用一次):倒序遍历

### 算法模式卡片 📐
- **模式名称**:0-1背包动态规划
- **适用条件**:从 n 个物品中选若干个(每个最多选一次),使得某种指标(和/体积)达到目标值
- **识别关键词**:"选若干元素"、"每个元素最多用一次"、"达到目标和/容量"
- **模板代码**:
```python
def knapsack_01(nums, target):
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        for j in range(target, num - 1, -1):
            dp[j] = dp[j] or dp[j - num]
    return dp[target]
```

### 易错点 ⚠️
1. **忘记判断总和奇偶性**:总和为奇数时直接返回 false,节省计算
   - 错误:`target = total // 2` 后直接开始DP
   - 正确:先 `if total % 2 != 0: return False`

2. **正序遍历导致元素重复使用**:0-1背包必须倒序
   - 错误:`for j in range(num, target + 1)`
   - 正确:`for j in range(target, num - 1, -1)`

3. **DP数组长度错误**:应该是 `target + 1`,因为要包含下标 target
   - 错误:`dp = [False] * target`
   - 正确:`dp = [False] * (target + 1)`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:云计算资源分配 — 将虚拟机分配到两个物理服务器,使负载均衡
- **场景2**:物流配送优化 — 将包裹分成两车,使每车重量接近
- **场景3**:竞技分队 — 将选手分成两队,使总实力值接近

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 494. 目标和 | Medium | 0-1背包变体 | 转化为"选正号/负号"的背包问题 |
| LeetCode 1049. 最后一块石头的重量II | Medium | 0-1背包 | 转化为"将石头分成两堆,差值最小" |
| LeetCode 698. 划分为k个相等的子集 | Medium | 回溯+剪枝 | 多个子集的扩展,需要回溯 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定数组 nums 和目标值 k,判断能否将数组分成 k 个非空子集,使每个子集的和都等于目标值。例如 nums=[4,3,2,3,5,2,1], k=4,可以分成 [5],[1,4],[2,3],[2,3],返回 true。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

先计算总和,如果不能被 k 整除则无解。然后用回溯枚举每个子集,每次尝试将元素加入当前子集,如果超过 target 则剪枝。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def canPartitionKSubsets(nums, k):
    total = sum(nums)
    if total % k != 0:
        return False

    target = total // k
    nums.sort(reverse=True)  # 排序优化
    used = [False] * len(nums)

    def backtrack(k, bucket, start):
        if k == 0:
            return True
        if bucket == target:
            return backtrack(k - 1, 0, 0)

        for i in range(start, len(nums)):
            if used[i] or bucket + nums[i] > target:
                continue
            used[i] = True
            if backtrack(k, bucket + nums[i], i + 1):
                return True
            used[i] = False

        return False

    return backtrack(k, 0, 0)
```

核心思路:用回溯枚举每个子集的填充方案,当某个子集凑满 target 时,递归处理下一个子集。排序优化可以提前剪枝。

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
