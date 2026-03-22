# 📖 第15课：长度最小的子数组

> **模块**：滑动窗口 | **难度**：Medium ⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/minimum-size-subarray-sum/
> **前置知识**：第14课 - 无重复字符的最长子串
> **预计学习时间**：25分钟

---

## 🎯 题目描述

给你一个含有 n 个正整数的数组 `nums` 和一个正整数 `target`，请你找出该数组中满足其总和**大于等于** `target` 的**长度最小**的连续子数组，并返回其长度。如果不存在符合条件的子数组，返回 `0`。

**示例 1：**
```
输入：target = 7, nums = [2,3,1,2,4,3]
输出：2
解释：子数组 [4,3] 是该条件下的长度最小的子数组
```

**示例 2：**
```
输入：target = 4, nums = [1,4,4]
输出：1
解释：子数组 [4] 是该条件下的长度最小的子数组
```

**示例 3：**
```
输入：target = 11, nums = [1,1,1,1,1,1,1,1]
输出：0
解释：没有符合条件的子数组
```

**约束条件：**
- `1 <= target <= 10^9`
- `1 <= nums.length <= 10^5`
- `1 <= nums[i] <= 10^4`
- 数组元素**都是正整数**（这是关键约束，保证了窗口单调性）

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | `target=1, nums=[1]` | `1` | 单个元素刚好满足 |
| 单元素大于target | `target=5, nums=[10]` | `1` | 单个元素就足够 |
| 无解情况 | `target=15, nums=[1,2,3,4,5]` | `0` | 所有元素和都不够 |
| 全部元素刚好 | `target=10, nums=[1,2,3,4]` | `4` | 需要全部元素 |
| 连续小数 | `target=7, nums=[2,3,1,2,4,3]` | `2` | 经典用例，[4,3] |
| 所有元素相同 | `target=10, nums=[5,5,5,5]` | `2` | 任意连续2个即可 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在一家自助餐厅打饭，目标是装满一个能装 **target 克食物**的餐盒，每种菜都有重量。你想用**最少的菜品数量**装满餐盒。
>
> 🐌 **笨办法**：你把每一种菜品组合都试一遍——"只拿菜1行不行？""拿菜1+菜2呢？""拿菜1+菜2+菜3呢？"……然后再从菜2开始重复这个过程。每种组合都要重新计算总重量，太慢了！这就是暴力法，时间复杂度 O(n^2)。
>
> 🚀 **聪明办法**：你用一个**可伸缩的夹子**（滑动窗口）来装菜。从左边开始往右边夹，边夹边称重：
> - 当夹子里的菜**还不够重**时，继续往右边加菜（扩大窗口）
> - 当夹子里的菜**已经够重**时，尝试从左边减少菜（缩小窗口），看能不能用更少的菜品达到目标
> - 每次达到目标时记录当前菜品数量，最后返回最小值
>
> 这样你的夹子只需要从左到右扫一遍菜品，每个菜最多被加入/移出各一次，效率极高！

### 关键洞察

**正整数数组 + "连续子数组和 ≥ target" → 窗口内元素和具有单调性：加元素会增大、减元素会减小 → 可以用滑动窗口优化！**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出
- **输入**：正整数数组 `nums`（长度 1~10^5），正整数 `target`
- **输出**：满足"连续子数组和 ≥ target"的**最小长度**（如果无解返回 0）
- **限制**：必须是**连续子数组**，数组元素**全为正整数**（这个很关键！）

### Step 2：先想笨办法（暴力法）
最直接的做法：枚举所有可能的连续子数组，计算它们的和，找出满足条件的最短的。用双层循环，外层枚举起点 i，内层枚举终点 j，每次计算 sum(nums[i:j+1])。
- 时间复杂度：O(n^2) 或 O(n^3)（取决于是否累加求和）
- 瓶颈在哪：**重复计算了大量的子数组和**，比如 [2,3,1] 和 [2,3,1,2] 都要分别算一遍和

### Step 3：瓶颈分析 → 优化方向
暴力法的核心问题：
- 大量重复计算：`sum(nums[i:j])` 和 `sum(nums[i:j+1])` 只差一个元素，却要重算一遍
- 没有利用"正整数"的特性：加元素一定让和变大，减元素一定让和变小

优化思路：能不能维护一个动态的窗口，通过"扩张右边界"和"收缩左边界"来快速找到所有满足条件的子数组？

### Step 4：选择武器
- 选用：**滑动窗口（可变长度窗口）**
- 理由：
  - 数组元素全为正数，保证了窗口和的**单调性**
  - 窗口右扩（加元素）→ 和增大，窗口左缩（减元素）→ 和减小
  - 这种单调性让我们可以用双指针 O(n) 完成遍历

> 🔑 **模式识别提示**：当题目出现"**连续子数组/子串**"+"**最长/最短/计数**"+"**满足某条件**"，优先考虑"**滑动窗口**"

---

## 🔑 解法一:暴力枚举（直觉法）

### 思路
枚举所有可能的连续子数组 [i, j]，计算它们的和，找出满足 sum ≥ target 的最小长度。外层循环枚举起点 i，内层循环枚举终点 j，一边扩展一边累加求和。

### 图解过程

```
示例：target = 7, nums = [2, 3, 1, 2, 4, 3]

从 i=0 开始枚举：
  j=0: sum=2 < 7 ❌
  j=1: sum=2+3=5 < 7 ❌
  j=2: sum=5+1=6 < 7 ❌
  j=3: sum=6+2=8 ≥ 7 ✅ 长度=4 [2,3,1,2]

从 i=1 开始枚举：
  j=1: sum=3 < 7 ❌
  j=2: sum=3+1=4 < 7 ❌
  j=3: sum=4+2=6 < 7 ❌
  j=4: sum=6+4=10 ≥ 7 ✅ 长度=4 [3,1,2,4]

从 i=2 开始枚举：
  j=2: sum=1 < 7 ❌
  j=3: sum=1+2=3 < 7 ❌
  j=4: sum=3+4=7 ≥ 7 ✅ 长度=3 [1,2,4]

从 i=3 开始枚举：
  j=3: sum=2 < 7 ❌
  j=4: sum=2+4=6 < 7 ❌
  j=5: sum=6+3=9 ≥ 7 ✅ 长度=3 [2,4,3]

从 i=4 开始枚举：
  j=4: sum=4 < 7 ❌
  j=5: sum=4+3=7 ≥ 7 ✅ 长度=2 [4,3] ← 最小！

从 i=5 开始枚举：
  j=5: sum=3 < 7 ❌ 无解

最小长度 = 2
```

### Python代码

```python
from typing import List


def min_subarray_len_brute(target: int, nums: List[int]) -> int:
    """
    解法一：暴力枚举所有子数组
    思路：双层循环枚举起点和终点，累加求和
    """
    n = len(nums)
    min_len = float('inf')  # 初始化为无穷大

    for i in range(n):           # 枚举起点
        current_sum = 0
        for j in range(i, n):    # 枚举终点
            current_sum += nums[j]  # 累加到当前终点
            if current_sum >= target:  # 满足条件
                min_len = min(min_len, j - i + 1)
                break  # 找到从i开始的最短子数组，不需要继续扩展j

    return 0 if min_len == float('inf') else min_len


# ✅ 测试
print(min_subarray_len_brute(7, [2, 3, 1, 2, 4, 3]))  # 期望输出：2
print(min_subarray_len_brute(4, [1, 4, 4]))           # 期望输出：1
print(min_subarray_len_brute(11, [1, 1, 1, 1, 1, 1, 1, 1]))  # 期望输出：0
print(min_subarray_len_brute(15, [1, 2, 3, 4, 5]))    # 期望输出：5
```

### 复杂度分析
- **时间复杂度**：O(n^2) — 两层循环，外层 n 次，内层平均 n/2 次
  - 具体地说：如果 n=100000，最坏需要约 50 亿次操作，会超时
- **空间复杂度**：O(1) — 只用了几个变量

### 优缺点
- ✅ 思路直观，容易理解
- ✅ 不需要额外空间
- ❌ 时间复杂度 O(n^2)，数据量大时会**超时**（LeetCode 会 TLE）
- ❌ 有优化空间：每次内层循环都重新累加，实际上可以利用"正数数组"的单调性

---

## ⚡ 解法二：滑动窗口（最优解）

### 优化思路
暴力法的问题在于重复计算。我们用一个**可伸缩的窗口 [left, right]** 来维护当前子数组：
- **右指针 right**：不断向右扩展，把新元素加入窗口
- **左指针 left**：当窗口和 ≥ target 时，尝试从左边收缩窗口，找最小长度
- 每个元素最多被 left 和 right 各访问一次，总时间 O(n)

> 💡 **关键想法**：因为数组全是正数，所以"加元素→和增大，减元素→和减小"。这种单调性保证了：只要当前窗口满足条件，我们就可以立即尝试收缩左边界来找更小的窗口，而不用担心遗漏答案。

### 图解过程

```
示例：target = 7, nums = [2, 3, 1, 2, 4, 3]

初始状态：left=0, right=0, sum=0, min_len=∞

Step 1: right=0, 加入 nums[0]=2
  [2] 3  1  2  4  3
   ↑
  L,R
  sum=2 < 7 → 继续右扩

Step 2: right=1, 加入 nums[1]=3
  [2  3] 1  2  4  3
   ↑  ↑
   L  R
  sum=5 < 7 → 继续右扩

Step 3: right=2, 加入 nums[2]=1
  [2  3  1] 2  4  3
   ↑     ↑
   L     R
  sum=6 < 7 → 继续右扩

Step 4: right=3, 加入 nums[3]=2
  [2  3  1  2] 4  3
   ↑        ↑
   L        R
  sum=8 ≥ 7 ✅ 满足条件！长度=4
  → 尝试左缩：移除 nums[0]=2

Step 5: left=1, 移除 nums[0]=2
   2 [3  1  2] 4  3
      ↑     ↑
      L     R
  sum=6 < 7 → 无法继续缩，右扩

Step 6: right=4, 加入 nums[4]=4
   2 [3  1  2  4] 3
      ↑        ↑
      L        R
  sum=10 ≥ 7 ✅ 长度=4
  → 尝试左缩：移除 nums[1]=3

Step 7: left=2, 移除 nums[1]=3
   2  3 [1  2  4] 3
         ↑     ↑
         L     R
  sum=7 ≥ 7 ✅ 长度=3 (更新 min_len=3)
  → 尝试左缩：移除 nums[2]=1

Step 8: left=3, 移除 nums[2]=1
   2  3  1 [2  4] 3
            ↑  ↑
            L  R
  sum=6 < 7 → 无法继续缩，右扩

Step 9: right=5, 加入 nums[5]=3
   2  3  1 [2  4  3]
            ↑     ↑
            L     R
  sum=9 ≥ 7 ✅ 长度=3
  → 尝试左缩：移除 nums[3]=2

Step 10: left=4, 移除 nums[3]=2
   2  3  1  2 [4  3]
               ↑  ↑
               L  R
  sum=7 ≥ 7 ✅ 长度=2 (更新 min_len=2) ← 最优解！
  → 尝试左缩：移除 nums[4]=4

Step 11: left=5, 移除 nums[4]=4
   2  3  1  2  4 [3]
                  ↑
                 L,R
  sum=3 < 7 → 无法继续缩

right 到达末尾，循环结束
最小长度 = 2
```

### Python代码

```python
from typing import List


def min_subarray_len_sliding_window(target: int, nums: List[int]) -> int:
    """
    解法二：滑动窗口（可变长度）
    思路：右指针扩展窗口，左指针收缩窗口，维护窗口和
    """
    n = len(nums)
    left = 0
    current_sum = 0
    min_len = float('inf')

    for right in range(n):  # 右指针不断右移
        current_sum += nums[right]  # 将右边界元素加入窗口

        # 当窗口和满足条件时，尝试收缩左边界
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)  # 更新最小长度
            current_sum -= nums[left]  # 移除左边界元素
            left += 1  # 左指针右移，收缩窗口

    return 0 if min_len == float('inf') else min_len


# ✅ 测试
print(min_subarray_len_sliding_window(7, [2, 3, 1, 2, 4, 3]))  # 期望输出：2
print(min_subarray_len_sliding_window(4, [1, 4, 4]))           # 期望输出：1
print(min_subarray_len_sliding_window(11, [1, 1, 1, 1, 1, 1, 1, 1]))  # 期望输出：0
print(min_subarray_len_sliding_window(15, [1, 2, 3, 4, 5]))    # 期望输出：5
```

### 复杂度分析
- **时间复杂度**：O(n) — 右指针遍历一遍数组 O(n)，左指针最多也遍历一遍 O(n)，总共 O(2n) = O(n)
  - 具体地说：如果 n=100000，只需要约 20 万次操作，比暴力法快 25000 倍！
- **空间复杂度**：O(1) — 只用了 left、right、current_sum、min_len 几个变量

---

## 🐍 Pythonic 写法

利用 Python 的语法糖简化代码：

```python
from typing import List


def min_subarray_len_pythonic(target: int, nums: List[int]) -> int:
    """
    Pythonic 写法：利用 or 运算符简化无解判断
    """
    left, current_sum, min_len = 0, 0, float('inf')

    for right, num in enumerate(nums):
        current_sum += num
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= nums[left]
            left += 1

    return min_len if min_len != float('inf') else 0


# ✅ 测试
print(min_subarray_len_pythonic(7, [2, 3, 1, 2, 4, 3]))  # 期望输出：2
print(min_subarray_len_pythonic(4, [1, 4, 4]))           # 期望输出：1
```

这个写法用到了：
- **`enumerate()`**：同时获取下标和值，代码更简洁
- **三元表达式**：`min_len if min_len != float('inf') else 0` 简化无解判断

> ⚠️ **面试建议**：先写解法二（滑动窗口）展示清晰的思路和双指针技巧，再提一嘴"也可以用 enumerate 简化"来展示语言功底。面试官更看重你的**思考过程**，而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一：暴力枚举 | 解法二：滑动窗口 | Pythonic写法 |
|------|--------------|--------------|-------------|
| 时间复杂度 | O(n^2) | **O(n)** | O(n) |
| 空间复杂度 | O(1) | **O(1)** | O(1) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐⭐ | ⭐⭐ |
| 适用场景 | 小规模数据或说明思路 | 面试首选，高效且通用 | 快速编码、展示语言功底 |

**面试建议**：先用 30 秒口述暴力法思路和复杂度（展示你能想到基本解法），然后重点讲解滑动窗口优化（展示优化能力）。关键点在于说清楚"为什么可以用滑动窗口"——**正数数组保证了窗口和的单调性**。

---

## 🎤 面试现场

> 模拟面试中的完整对话流程，帮你练习"边想边说"。

**面试官**：给你一个正整数数组和一个目标值，找出数组中满足总和大于等于目标值的最短连续子数组的长度。

**你**：（审题 30 秒）好的，让我确认一下——输入是一个**正整数**数组（这个很关键！）和一个 target，输出是满足"连续子数组和 ≥ target"的最小长度，如果不存在返回 0，对吧？

**面试官**：没错。

**你**：好的。我先想一个最直接的办法：枚举所有连续子数组，计算它们的和，找出满足条件的最短的。用双层循环，时间 O(n^2)，空间 O(1)。不过这个显然不够快。

让我想想怎么优化……这道题的关键特征是：
1. 要求"连续子数组"
2. 数组元素**全是正数**
3. 找"最短长度"

因为是正数数组，所以窗口和具有**单调性**：加元素→和增大，减元素→和减小。这让我可以用**滑动窗口**优化！

具体做法：用两个指针 left 和 right 维护一个动态窗口。right 不断右移扩大窗口，当窗口和 ≥ target 时，尝试移动 left 收缩窗口，找最小长度。这样每个元素最多被访问两次，时间 O(n)，空间 O(1)。

**面试官**：思路清晰，请写代码吧。

**你**：好的。（边写边说）

```python
def minSubArrayLen(self, target, nums):
    left, current_sum, min_len = 0, 0, float('inf')

    for right in range(len(nums)):
        current_sum += nums[right]  # 右扩窗口

        # 满足条件时，尝试收缩左边界
        while current_sum >= target:
            min_len = min(min_len, right - left + 1)
            current_sum -= nums[left]
            left += 1  # 左缩窗口

    return min_len if min_len != float('inf') else 0
```

关键是这个 while 循环——只要当前窗口满足条件，我们就不断收缩左边界，直到不满足为止。因为是正数数组，收缩一定会让和变小，所以不会遗漏答案。

**面试官**：能手动跑一个例子验证一下吗？

**你**：好的，用 `target=7, nums=[2,3,1,2,4,3]`：
- right=0~3: 窗口 [2,3,1,2]，sum=8 ≥ 7 ✅，长度 4
- 收缩 left=1: 窗口 [3,1,2]，sum=6 < 7，停止收缩
- right=4: 窗口 [3,1,2,4]，sum=10 ≥ 7 ✅
- 连续收缩 left=2,3: 窗口最终变为 [2,4]，sum=6 < 7
- right=5: 窗口 [2,4,3]，sum=9 ≥ 7 ✅
- 连续收缩 left=4: 窗口 [4,3]，sum=7 ≥ 7 ✅，长度 2 ← 最优解！
- 继续收缩 left=5: sum=3 < 7，停止

最终返回 min_len=2 ✅

**面试官**：很好。如果数组中有负数或零呢？

**你**：如果有负数或零，窗口和就不再具有单调性了——加元素不一定让和增大，减元素不一定让和减小。这种情况下滑动窗口不适用，可能需要用**前缀和 + 单调队列**或其他方法。这道题限定正整数，正是为了保证单调性，让滑动窗口成为可行解。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗？" | 时间已经是 O(n) 最优（至少要看一遍所有元素），空间 O(1) 也是最优。这就是最优解了 |
| "如果数组中有负数呢？" | 滑动窗口失效，因为窗口和不再单调。可以考虑前缀和 + 二分查找（O(n log n)），或者前缀和 + 单调队列（O(n)），但复杂度分析更复杂 |
| "如果要求和恰好等于 target 呢？" | 变成"和为 K 的子数组"问题，用前缀和 + 哈希表，时间 O(n)。滑动窗口只适用于"≥"或"≤"的单调条件 |
| "实际工程中什么场景会用到？" | 日志分析（找最短时间窗口达到访问量阈值）、网络流量监控（最少数据包达到流量上限）、资源调度（最少任务数达到 CPU 占用目标） |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# enumerate() — 同时拿到下标和值
for i, num in enumerate([10, 20, 30]):
    print(i, num)   # 0 10 / 1 20 / 2 30

# float('inf') — 表示正无穷，用于初始化最小值
min_val = float('inf')
min_val = min(min_val, 5)   # min_val = 5

# while 循环收缩窗口 — 滑动窗口的标准写法
while current_sum >= target:
    # 更新答案
    min_len = min(min_len, right - left + 1)
    # 收缩窗口
    current_sum -= nums[left]
    left += 1

# 三元表达式简化返回值判断
return min_len if min_len != float('inf') else 0
```

### 💡 底层原理（选读）

> **为什么滑动窗口能保证不遗漏答案？**
>
> 关键在于**单调性**。因为数组元素全是正数，所以：
> - **右扩（加元素）→ 和一定增大**：如果当前窗口和 < target，右扩可能找到解
> - **左缩（减元素）→ 和一定减小**：如果当前窗口和 ≥ target，左缩可能找到更优解
>
> 这种单调性保证了：
> 1. 当窗口和 < target 时，继续右扩一定是对的（左缩只会让和更小，不可能满足条件）
> 2. 当窗口和 ≥ target 时，继续左缩直到不满足条件是对的（找到从当前 left 开始的最短窗口）
> 3. 每个 left 位置只会被访问一次，每个 right 位置也只会被访问一次
>
> **如果有负数会怎样？**
> 假设 nums = [3, -2, 5], target = 5：
> - 窗口 [3]：sum=3 < 5 → 右扩
> - 窗口 [3, -2]：sum=1 < 5 → 右扩
> - 窗口 [3, -2, 5]：sum=6 ≥ 5 ✅ 长度 3
> - 左缩：窗口 [-2, 5]：sum=3 < 5 → 停止
>
> 但实际上 [5] 这个长度为 1 的窗口也满足条件！滑动窗口遗漏了，因为从 [3] 扩展到 [3, -2] 时和反而减小了。
>
> **滑动窗口的适用条件总结**：
> - 连续子数组/子串问题
> - 窗口内元素具有**单调性**（正数数组、递增/递减序列等）
> - 求最长/最短/计数

### 算法模式卡片 📐
- **模式名称**：可变长度滑动窗口（Flexible Sliding Window）
- **适用条件**：
  1. 连续子数组/子串问题
  2. 求最长/最短满足某条件的子数组
  3. 窗口内元素具有单调性（加元素→条件"更满足"，减元素→条件"更不满足"）
- **识别关键词**："最长/最短"+"连续子数组"+"和/平均值/包含"+"≥/≤某值"
- **模板代码**：
```python
def sliding_window_min(nums: List[int], target: int) -> int:
    """
    可变长度滑动窗口模板 - 求最短窗口
    """
    left = 0
    window_value = 0  # 窗口内的某个值（和、计数等）
    min_len = float('inf')

    for right in range(len(nums)):
        # 1. 右边界扩展：将 nums[right] 加入窗口
        window_value += nums[right]

        # 2. 当窗口满足条件时，尝试收缩左边界
        while window_value >= target:  # 满足条件
            # 更新答案
            min_len = min(min_len, right - left + 1)
            # 收缩窗口：移除 nums[left]
            window_value -= nums[left]
            left += 1

    return 0 if min_len == float('inf') else min_len


# 对于求"最长窗口"的问题，模板略有不同：
def sliding_window_max(s: str) -> int:
    """
    可变长度滑动窗口模板 - 求最长窗口
    """
    left = 0
    window = {}  # 窗口内的状态（如字符计数）
    max_len = 0

    for right in range(len(s)):
        # 1. 右边界扩展
        c = s[right]
        window[c] = window.get(c, 0) + 1

        # 2. 当窗口不满足条件时，收缩左边界
        while window[c] > 1:  # 不满足条件（如有重复）
            d = s[left]
            window[d] -= 1
            left += 1

        # 3. 更新答案（窗口满足条件时）
        max_len = max(max_len, right - left + 1)

    return max_len
```

### 易错点 ⚠️
1. **忘记检查无解情况**：如果没有任何子数组满足条件，min_len 会保持 float('inf')，要记得返回 0。很多人直接 return min_len 导致错误。解决：最后加上 `return 0 if min_len == float('inf') else min_len`。

2. **while 循环条件写错**：收缩窗口的条件是 `while current_sum >= target`，有的人写成 `if current_sum >= target`，导致只收缩一次就停了，遗漏了更优解。解决：记住是 **while**（尽可能收缩），不是 **if**（收缩一次）。

3. **更新 min_len 的位置错了**：应该在 while 循环**内部**更新 min_len，即每次收缩都检查是否更优。有的人写在 while 循环外面，导致只记录了最后一次收缩的结果。

4. **误以为所有"子数组和"问题都能用滑动窗口**：只有当窗口和具有单调性（如正数数组）时才能用滑动窗口。如果有负数或零，或者求"和恰好等于 K"，要用前缀和 + 哈希表。

---

## 🏗️ 工程实战（选读）

> 这个算法思想在真实项目中的应用，让你知道"学了有什么用"。

- **日志分析 - 最短时间窗口达到访问量阈值**：网站日志中每条记录都是一次访问，我们想找出"最短时间窗口内访问量达到 10000 次"的时间段，用于分析流量高峰。这就是一个"最短连续子数组和 ≥ target"问题，用滑动窗口 O(n) 扫描日志即可。

- **网络流量监控 - 最少数据包达到流量上限**：路由器监控数据包流量，每个数据包有大小（字节数），我们想找出"最少多少个连续数据包的总大小超过 1MB"，用于检测突发流量。滑动窗口可以实时计算，延迟极低。

- **资源调度 - 最少任务数达到 CPU 占用目标**：服务器调度系统中，每个任务消耗一定 CPU 占用率，我们想找出"最少多少个连续任务能让 CPU 占用率超过 80%"，用于优化任务批处理策略。

---

## 🏋️ 举一反三

完成本课后，试试这些同类题目来巩固知识：

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 3. 无重复字符的最长子串 | Medium | 滑动窗口 + 哈希表 | 窗口内不能有重复字符，用 set 或 dict 记录 |
| LeetCode 76. 最小覆盖子串 | Hard | 滑动窗口 + 计数器 | 窗口必须包含目标字符串的所有字符 |
| LeetCode 438. 找到字符串中所有字母异位词 | Medium | 固定长度滑动窗口 | 窗口大小固定，比较窗口内字符计数 |
| LeetCode 713. 乘积小于 K 的子数组 | Medium | 滑动窗口 + 计数 | 乘积 < K，也是单调性问题 |
| LeetCode 862. 和至少为 K 的最短子数组 | Hard | 前缀和 + 单调队列 | 有负数，滑动窗口失效，需要单调队列 |
| LeetCode 904. 水果成篮 | Medium | 滑动窗口 | 窗口内最多两种不同水果，变种"最长" |

---

## 📝 课后小测

试试这道变体题，不要看答案,自己先想 5 分钟！

**题目**：给定一个正整数数组 `nums` 和一个正整数 `k`，找出数组中满足**平均值大于等于 k** 的**最短连续子数组**的长度。如果不存在，返回 0。

示例：`nums = [1, 12, -5, -6, 50, 3], k = 4` → `3`（子数组 [12, -5, -6] 的平均值恰好 ≥ 4... 等等，有负数！）

<details>
<summary>💡 提示 1（实在想不出来再点开）</summary>

"平均值 ≥ k" 可以转化为"总和 ≥ k × 长度"。但是长度是变化的，怎么办？

</details>

<details>
<summary>💡 提示 2（再给你一个线索）</summary>

其实这道题**不能直接用滑动窗口**！因为有负数，窗口和不再单调。但你可以用**二分答案 + 滑动窗口**：二分枚举长度 L，然后用固定长度 L 的滑动窗口检查是否存在平均值 ≥ k 的子数组。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def min_length_avg_k(nums: list[int], k: int) -> int:
    """
    二分答案 + 固定长度滑动窗口
    思路：二分枚举长度 L，检查是否存在长度为 L 的子数组平均值 ≥ k
    """
    def has_avg_geq_k(length: int) -> bool:
        """检查是否存在长度为 length 的子数组平均值 ≥ k"""
        window_sum = sum(nums[:length])
        if window_sum >= k * length:
            return True
        for i in range(length, len(nums)):
            window_sum += nums[i] - nums[i - length]
            if window_sum >= k * length:
                return True
        return False

    # 二分答案：枚举长度
    left, right = 1, len(nums)
    result = 0
    while left <= right:
        mid = (left + right) // 2
        if has_avg_geq_k(mid):
            result = mid
            right = mid - 1  # 尝试更短的长度
        else:
            left = mid + 1

    return result


# 测试
print(min_length_avg_k([1, 2, 3, 4, 5], 3))  # 期望：1（[5]的平均值5≥3）
print(min_length_avg_k([1, 1, 1, 1, 1], 2))  # 期望：0（所有平均值都是1<2）
```

**核心思路**：因为有负数，不能用可变窗口。改用**二分答案**：在 [1, n] 范围内二分长度 L，对每个 L 用固定长度滑动窗口 O(n) 检查是否存在平均值 ≥ k 的子数组。总时间 O(n log n)。

**启示**：并非所有"最短/最长"问题都能用滑动窗口，关键看**单调性**。如果窗口和不单调，考虑二分答案、前缀和、单调队列等其他方法。

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
