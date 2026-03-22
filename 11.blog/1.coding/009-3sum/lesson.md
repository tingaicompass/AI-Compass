# 📖 第9课:三数之和

> **模块**:双指针 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/3sum/
> **前置知识**:第8课(盛最多水的容器-对撞指针基础)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给你一个整数数组 `nums`,请你找出所有满足 `nums[i] + nums[j] + nums[k] = 0` 的三元组 `[nums[i], nums[j], nums[k]]`。

注意:**答案中不能包含重复的三元组**。你可以按任意顺序返回答案。

**示例:**
```
输入:nums = [-1, 0, 1, 2, -1, -4]
输出:[[-1, -1, 2], [-1, 0, 1]]
解释:
  nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0
  nums[1] + nums[3] + nums[4] = 0 + 1 + (-1) = 0
  nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0
  去重后只有 [-1, -1, 2] 和 [-1, 0, 1]
```

```
输入:nums = [0, 1, 1]
输出:[]
解释:唯一可能的三元组和不为 0
```

```
输入:nums = [0, 0, 0]
输出:[[0, 0, 0]]
解释:三个 0 的和为 0,这是唯一的三元组
```

**约束条件:**
- `3 <= nums.length <= 3000` (至少3个元素,最多3000个)
- `-10^5 <= nums[i] <= 10^5` (元素可以为负数或零)

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | `[0, 0, 0]` | `[[0, 0, 0]]` | 全零的特殊情况 |
| 无解 | `[1, 2, 3]` | `[]` | 全正数,和不可能为0 |
| 含重复值 | `[-1, -1, 2, 2]` | `[[-1, -1, 2]]` | 去重逻辑是否正确 |
| 多组解 | `[-4,-1,-1,0,1,2]` | `[[-1,-1,2],[-1,0,1]]` | 多个三元组,去重 |
| 含负数和0 | `[-2, 0, 1, 1, 2]` | `[[-2, 0, 2], [-2, 1, 1]]` | 混合正负零 |
| 大规模 | `n=3000` | — | 暴力O(n³)会超时,需O(n²) |

---

## 💡 思路引导

### 生活化比喻

> 想象你在玩一个**配对游戏**:桌上有一堆卡片,每张卡片上有一个数字(可能是正数、负数或零)。你要找出**所有能凑成和为0的三张卡片组合**。
>
> 🐌 **笨办法**:随机抽三张,看看和是不是0。全试一遍,共 C(n,3) ≈ n³/6 种组合。如果有3000张卡片,要试约45亿次,累死人!而且**重复组合**很难避免(比如 [1,2,-3] 和 [2,1,-3] 其实是同一组)。
>
> 🚀 **聪明办法**:先把卡片**按数字大小排序**!然后:
> 1. **固定第一张卡片**(比如 -4)
> 2. 剩下的卡片用**对撞双指针**找两张,使三张和为0
> 3. 找到一组后,**跳过重复的卡片**(如果连续几张都是 -1,只用第一张)
>
> 这样,每固定一张卡片只需O(n)找剩余两张,总共O(n²)。而且**排序后天然去重**:遇到相同数字直接跳过!

### 关键洞察

**三数之和 = 两数之和的升级版。固定一个数,问题转化为"在剩余数组中找两数之和等于目标值",这正是对撞双指针擅长的!关键是:排序后处理去重。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组 `nums`,可能含重复、负数、零
- **输出**:所有和为0的三元组**列表**,注意是返回**值**而非下标
- **核心公式**:`nums[i] + nums[j] + nums[k] = 0`
- **限制**:不能有重复三元组(如 `[-1,0,1]` 和 `[0,-1,1]` 算重复)

### Step 2:先想笨办法(暴力法)
三层循环枚举所有 `(i, j, k)` 组合:
```python
for i in range(n):
    for j in range(i+1, n):
        for k in range(j+1, n):
            if nums[i] + nums[j] + nums[k] == 0:
                # 加入结果,还要去重...
```
- 时间复杂度:O(n³)
- 瓶颈在哪:
  1. 三层循环太慢,n=3000 时约 4.5×10⁹ 次
  2. **去重很麻烦**:需要用集合或排序后的元组来避免 `[-1,0,1]` 和 `[0,-1,1]` 重复

### Step 3:瓶颈分析 → 优化方向

暴力法的核心问题:
1. **三层循环**的最内层(找第三个数)本质是"查找",能否优化?
2. **去重逻辑**复杂,能否通过某种预处理简化?

优化思路:
- 问题1:**固定第一个数** `nums[i]`,剩下的问题变成"在 `nums[i+1:]` 中找两个数和为 `-nums[i]`" → 这是**两数之和**,可以用对撞双指针 O(n) 解决!
- 问题2:**先排序**,这样相同数字会挨在一起,遇到重复直接跳过,天然去重

### Step 4:选择武器
- 选用:**排序 + 固定一个数 + 对撞双指针**
- 理由:
  1. 排序 O(n log n),不是瓶颈
  2. 外层循环 O(n) × 内层双指针 O(n) = O(n²),比暴力法快 n 倍
  3. 排序后去重变简单:只需检查相邻元素是否相同

> 🔑 **模式识别提示**:当题目涉及"**多数之和**"、"**找所有满足条件的组合**"、"**去重**",考虑"**排序 + 双指针 + 跳过重复元素**"模式

---

## 🔑 解法一:暴力三层循环(直觉法)

### 思路
枚举所有三元组,检查和是否为0,用集合去重。虽然能得到正确答案,但时间复杂度 O(n³),大数据会超时。

### 图解过程

```
示例:nums = [-1, 0, 1, 2, -1, -4]

暴力法:三层循环枚举
i=0 (-1):
  j=1 (0):
    k=2 (1): -1+0+1 = 0 ✅ → [-1, 0, 1]
    k=3 (2): -1+0+2 = 1 ❌
    k=4 (-1): -1+0-1 = -2 ❌
    k=5 (-4): -1+0-4 = -5 ❌
  j=2 (1):
    k=3 (2): -1+1+2 = 2 ❌
    ...
  ...
i=1 (0):
  j=2 (1):
    k=4 (-1): 0+1-1 = 0 ✅ → [0, 1, -1] (和 [-1,0,1] 重复!)
    ...
...

需要用集合存储排序后的元组来去重,复杂!
总计: C(6,3) = 20 次检查
```

### Python代码

```python
from typing import List


def three_sum_brute(nums: List[int]) -> List[List[int]]:
    """
    解法一:暴力三层循环
    思路:枚举所有(i,j,k)三元组,用集合去重
    """
    n = len(nums)
    result_set = set()  # 用集合存储排序后的元组,自动去重

    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if nums[i] + nums[j] + nums[k] == 0:
                    # 排序后加入集合,避免 [-1,0,1] 和 [0,-1,1] 重复
                    triplet = tuple(sorted([nums[i], nums[j], nums[k]]))
                    result_set.add(triplet)

    # 转换为列表返回
    return [list(t) for t in result_set]


# ✅ 测试
print(three_sum_brute([-1, 0, 1, 2, -1, -4]))  # 期望:[[-1,-1,2],[-1,0,1]]
print(three_sum_brute([0, 1, 1]))               # 期望:[]
print(three_sum_brute([0, 0, 0]))               # 期望:[[0,0,0]]
print(three_sum_brute([-2, 0, 1, 1, 2]))        # 期望:[[-2,0,2],[-2,1,1]]
```

### 复杂度分析
- **时间复杂度**:O(n³) — 三层循环,每层最多 n 次
  - 具体地说:如果 n=3000,需要约 4.5×10⁹ 次操作,**会超时**
- **空间复杂度**:O(m) — m 是满足条件的三元组数量(用于存储结果)

### 优缺点
- ✅ 思路直观,容易想到
- ✅ 逻辑简单,不容易出bug
- ❌ 时间复杂度 O(n³),**大数据会超时**
- ❌ 去重逻辑需要额外处理(排序+集合),不够优雅

---

## ⚡ 解法二:排序 + 固定一个数 + 对撞双指针(最优解)

### 优化思路

将三数之和转化为两数之和:
1. **先排序**数组 O(n log n)
2. **固定第一个数** `nums[i]`,目标变成:在 `nums[i+1:]` 中找两数和为 `-nums[i]`
3. 用**对撞双指针**在剩余有序数组中找这两个数 O(n)
4. **去重三要素**:
   - 固定数去重:如果 `nums[i] == nums[i-1]`,跳过(避免重复三元组)
   - 左指针去重:找到一组解后,跳过所有相同的左指针值
   - 右指针去重:找到一组解后,跳过所有相同的右指针值

> 💡 **关键想法**:排序后,相同数字挨在一起,遇到重复直接跳过即可去重。对撞双指针在有序数组上找两数之和非常高效。

### 图解过程

```
示例:nums = [-1, 0, 1, 2, -1, -4]

Step 0: 先排序
  排序后:[-4, -1, -1, 0, 1, 2]
  索引:    0   1   2  3  4  5

Step 1: i=0, nums[i]=-4, 目标:找两数和为 4
  left=1 (-1), right=5 (2)
  -1 + 2 = 1 < 4 → left++

  left=2 (-1), right=5 (2)
  -1 + 2 = 1 < 4 → left++

  left=3 (0), right=5 (2)
  0 + 2 = 2 < 4 → left++

  left=4 (1), right=5 (2)
  1 + 2 = 3 < 4 → left++

  left=5, left >= right, 结束
  → 无解

Step 2: i=1, nums[i]=-1, 目标:找两数和为 1
  left=2 (-1), right=5 (2)
  -1 + 2 = 1 ✅ 找到![-1, -1, 2]

  去重:left++ 跳过重复的 -1
  left=3 (0), right=4 (1) (right已减1)
  0 + 1 = 1 ✅ 找到![-1, 0, 1]

  去重:left=4, right=4, left >= right, 结束

Step 3: i=2, nums[i]=-1
  ⚠️ nums[2] == nums[1] (都是-1), 跳过!避免重复

Step 4: i=3, nums[i]=0, 目标:找两数和为 0
  left=4 (1), right=5 (2)
  1 + 2 = 3 > 0 → right--

  left=4, right=4, left >= right, 结束
  → 无解

Step 5: i=4,5 剩余数组不足2个元素,结束

最终结果:[[-1, -1, 2], [-1, 0, 1]]
```

### Python代码

```python
from typing import List


def three_sum_sort_two_pointers(nums: List[int]) -> List[List[int]]:
    """
    解法二:排序 + 固定一个数 + 对撞双指针
    思路:固定 nums[i],在剩余有序数组中用双指针找两数和为 -nums[i]
    """
    nums.sort()  # 先排序,O(n log n)
    n = len(nums)
    result = []

    for i in range(n - 2):  # 固定第一个数,至少留2个位置给双指针
        # 剪枝:如果最小值都 > 0,后面全是正数,不可能和为0
        if nums[i] > 0:
            break

        # 去重1:跳过重复的固定数(避免重复三元组)
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # 双指针找两数和为 -nums[i]
        left, right = i + 1, n - 1
        target = -nums[i]

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == target:
                # 找到一组解
                result.append([nums[i], nums[left], nums[right]])

                # 去重2:跳过重复的左指针值
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                # 去重3:跳过重复的右指针值
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                # 移动指针继续寻找
                left += 1
                right -= 1

            elif current_sum < target:
                left += 1  # 和太小,左指针右移
            else:
                right -= 1  # 和太大,右指针左移

    return result


# ✅ 测试
print(three_sum_sort_two_pointers([-1, 0, 1, 2, -1, -4]))  # 期望:[[-1,-1,2],[-1,0,1]]
print(three_sum_sort_two_pointers([0, 1, 1]))               # 期望:[]
print(three_sum_sort_two_pointers([0, 0, 0]))               # 期望:[[0,0,0]]
print(three_sum_sort_two_pointers([-2, 0, 1, 1, 2]))        # 期望:[[-2,0,2],[-2,1,1]]
print(three_sum_sort_two_pointers([-4, -1, -1, 0, 1, 2]))   # 期望:[[-1,-1,2],[-1,0,1]]
```

### 复杂度分析
- **时间复杂度**:O(n²) — 排序 O(n log n) + 外层循环 O(n) × 内层双指针 O(n)
  - 具体地说:如果 n=3000,约需 9×10⁶ 次操作,比暴力法快 500 倍!
- **空间复杂度**:O(log n) — 排序的栈空间(如果不算结果列表的话)

---

## 🐍 Pythonic 写法

利用生成器表达式和 `itertools` 可以写得更简洁(仅供学习,面试不推荐):

```python
from itertools import combinations

def three_sum_pythonic(nums: List[int]) -> List[List[int]]:
    """
    Pythonic写法:用itertools.combinations枚举三元组
    ⚠️ 本质还是暴力法,时间复杂度O(n³),会超时!仅供学习
    """
    result_set = {
        tuple(sorted([nums[i], nums[j], nums[k]]))
        for i, j, k in combinations(range(len(nums)), 3)
        if nums[i] + nums[j] + nums[k] == 0
    }
    return [list(t) for t in result_set]
```

或者用更优雅的双指针写法:

```python
def three_sum_elegant(nums: List[int]) -> List[List[int]]:
    """优雅版:简化去重逻辑"""
    nums.sort()
    result = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue  # 跳过重复的固定数
        if nums[i] > 0:
            break  # 剪枝

        # 双指针查找
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0:
                left += 1
            elif total > 0:
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])
                # 一次性跳过所有重复
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1

    return result
```

> ⚠️ **面试建议**:先写清晰版本(解法二),确保逻辑正确。如果面试官问"能否优化",再提出剪枝优化(`nums[i] > 0` 时提前退出)。

---

## 📊 解法对比

| 维度 | 解法一:暴力三层循环 | 解法二:排序+双指针 |
|------|-------------------|------------------|
| 时间复杂度 | O(n³) | O(n²) |
| 空间复杂度 | O(m) | O(log n) |
| 代码难度 | 简单 | 中等(去重逻辑需注意) |
| 面试推荐 | ⭐ | ⭐⭐⭐ |
| 适用场景 | n ≤ 100 的小数据 | 所有情况,尤其 n > 1000 |
| 去重方式 | 集合 + 排序元组 | 跳过相邻重复元素 |

**面试建议**:
1. **先说暴力法**:"最直接的想法是三层循环枚举所有三元组,O(n³),但3000数据会超时"
2. **引出优化**:"我注意到如果固定第一个数,问题就变成两数之和,可以用双指针 O(n) 解决。再加上排序,去重也变简单了"
3. **写代码 + 讲解去重**:特别强调三个去重点:"固定数去重、左指针去重、右指针去重"
4. **测试边界**:`[0,0,0]`, `[-1,-1,2,2]` 验证去重逻辑

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题(给出题目)。

**你**:(审题30秒)好的,这道题是要找所有和为0的三元组,且不能有重复。我的第一个想法是三层循环枚举所有三元组,时间复杂度 O(n³),但数据量可以到3000,暴力法可能会超时。

**面试官**:那怎么优化?

**你**:我注意到这其实是**两数之和的升级版**。如果我固定第一个数 `nums[i]`,问题就变成:"在剩余数组中找两个数,使它们的和等于 `-nums[i]`"。这是经典的两数之和,可以用**对撞双指针** O(n) 解决。外层循环固定 n 次,每次内层双指针 O(n),总体 O(n²)。

**面试官**:不错!那去重怎么处理?

**你**:去重是这道题的难点。我的策略是:**先排序**,这样相同数字会挨在一起。然后在三个位置做去重:
1. **固定数去重**:如果 `nums[i] == nums[i-1]`,跳过,避免重复三元组
2. **左指针去重**:找到一组解后,跳过所有相同的左指针值
3. **右指针去重**:同理,跳过所有相同的右指针值

这样就能保证结果中没有重复三元组。

**面试官**:很好,请写一下代码。

**你**:(边写边说)我先对数组排序,然后用一个循环固定第一个数。注意这里有个**剪枝优化**:如果 `nums[i] > 0`,因为数组已排序,后面都是正数,不可能和为0,可以直接退出。然后对剩余数组用双指针...(写完)

**面试官**:测试一下?

**你**:好的,用 `[-1,0,1,2,-1,-4]`。排序后是 `[-4,-1,-1,0,1,2]`。固定 `i=1` 时 `nums[i]=-1`,双指针从两端开始:左边 `-1`,右边 `2`,和为 `1`,正好等于目标!加入结果 `[-1,-1,2]`。然后去重,跳过重复的 `-1`,继续找到 `[-1,0,1]`。再固定 `i=2`,但 `nums[2] == nums[1]`,跳过避免重复。最终返回 `[[-1,-1,2],[-1,0,1]]`,正确!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么一定要排序?" | "排序有两个好处:1) 使相同数字挨在一起,方便去重;2) 让双指针能根据和的大小移动,保证O(n)时间。不排序的话,去重很复杂,且无法用双指针。" |
| "如果数组本身就是有序的,还需要排序吗?" | "如果已经有序,可以跳过排序步骤,直接双指针。但题目没保证有序,所以必须先排序。" |
| "空间复杂度能更优吗?" | "如果不算结果列表,空间复杂度已经是 O(log n)(排序的栈空间)。如果用堆排序可以做到 O(1),但常数更大,实际更慢。" |
| "如果要找四数之和怎么办?" | "同理!固定两个数,剩下的用双指针找两数之和。时间复杂度变成 O(n³)。LeetCode 18 就是四数之和,可以用类似方法。" |
| "能用哈希表代替双指针吗?" | "可以,但不如双指针优雅。固定 `nums[i]` 后,遍历 `nums[j]`,查哈希表里有没有 `-nums[i]-nums[j]`。但去重更复杂,且空间复杂度变 O(n)。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:原地排序 — 直接修改原数组,不占额外空间
nums.sort()  # 时间 O(n log n), 空间 O(log n) 栈空间

# 技巧2:多条件跳过 — 用 continue 简化逻辑
if i > 0 and nums[i] == nums[i - 1]:
    continue  # 跳过本次循环,继续下一次

# 技巧3:边界检查 — 避免数组越界
while left < right and nums[left] == nums[left + 1]:
    left += 1  # 必须先检查 left < right,否则 left+1 可能越界

# 技巧4:剪枝优化 — 提前退出
if nums[i] > 0:
    break  # 后面都是正数,不可能和为0,直接结束
```

### 💡 底层原理(选读)

> **为什么排序后双指针可以不漏解?**

数学证明:
1. 排序后,数组单调递增
2. 对于固定的 `nums[i]`,我们要找 `nums[left] + nums[right] = -nums[i]`
3. 如果当前和 `< target`,说明 `nums[left]` 太小,必须右移 `left`(因为 `right` 左移只会让和更小)
4. 如果当前和 `> target`,说明 `nums[right]` 太大,必须左移 `right`
5. 每次移动都在缩小搜索空间,且不会漏掉解:
   - 移动 `left` 时,所有 `(left旧, right-k)` 的组合都 `< target`(因为数组递增)
   - 移动 `right` 时,所有 `(left+k, right旧)` 的组合都 `> target`
6. 直到 `left >= right`,搜索空间为空,必然找到所有解

> **Python 的 sort() 是怎么实现的?**
> - Python 使用 **Timsort** 算法,是归并排序和插入排序的混合
> - 时间复杂度:最好 O(n),平均/最坏 O(n log n)
> - 空间复杂度:O(n)(需要辅助数组)
> - **稳定排序**:相同元素的相对顺序不变
> - 对于几乎有序的数组,Timsort 特别快!

### 算法模式卡片 📐

- **模式名称**:排序 + 固定一个数 + 对撞双指针
- **适用条件**:
  1. 问题涉及"多数之和"(三数、四数等)
  2. 需要找**所有满足条件的组合**
  3. 不能有重复结果
- **识别关键词**:"三数之和"、"四数之和"、"所有组合"、"去重"
- **模板代码**:
```python
def k_sum(nums: List[int], k: int, target: int) -> List[List[int]]:
    """
    通用 k 数之和模板
    k=2 时用双指针,k>2 时递归降维
    """
    nums.sort()  # 先排序
    result = []

    def dfs(start: int, k: int, target: int, path: List[int]):
        if k == 2:
            # 双指针找两数之和
            left, right = start, len(nums) - 1
            while left < right:
                current = nums[left] + nums[right]
                if current == target:
                    result.append(path + [nums[left], nums[right]])
                    # 去重
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current < target:
                    left += 1
                else:
                    right -= 1
        else:
            # 固定一个数,递归找 k-1 数之和
            for i in range(start, len(nums) - k + 1):
                if i > start and nums[i] == nums[i - 1]:
                    continue  # 去重
                dfs(i + 1, k - 1, target - nums[i], path + [nums[i]])

    dfs(0, k, target, [])
    return result
```

### 易错点 ⚠️

1. **错误:去重时漏掉边界检查**
   - 常见错误:`while nums[left] == nums[left + 1]: left += 1` (可能越界!)
   - 原因:没有检查 `left < right`,导致 `left + 1` 越界
   - 正确做法:`while left < right and nums[left] == nums[left + 1]: left += 1`

2. **错误:固定数去重条件写错**
   - 常见错误:`if nums[i] == nums[i - 1]: continue` (i=0 时越界!)
   - 原因:没有检查 `i > 0`,第一个元素没有前驱
   - 正确做法:`if i > 0 and nums[i] == nums[i - 1]: continue`

3. **错误:找到一组解后忘记移动指针**
   - 常见错误:找到解后只 `result.append(...)`,没有 `left += 1; right -= 1`
   - 原因:指针卡住,死循环
   - 正确做法:去重后必须同时移动 `left++` 和 `right--`

4. **错误:剪枝条件写错**
   - 常见错误:`if nums[i] >= 0: break` (应该是 `>` 不是 `>=`)
   - 原因:如果 `nums[i] = 0`,可能有 `[0, 0, 0]` 这样的解
   - 正确做法:`if nums[i] > 0: break`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:推荐系统 — 协同过滤**
  在电商推荐中,找"三个用户共同喜欢的商品"可以用类似思路:排序后用多指针扫描,避免重复推荐。

- **场景2:数据去重 — 数据库查询优化**
  在数据分析中,需要找"三个字段组合后唯一"的记录,可以先排序,然后用滑动窗口跳过重复。

- **场景3:金融风控 — 交易异常检测**
  检测"三笔交易金额之和异常"(如洗钱检测),需要高效枚举三元组,双指针比暴力法快得多。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 1. 两数之和 | Easy | 哈希表 | 本题的简化版,先复习一遍 |
| LeetCode 18. 四数之和 | Medium | 排序+双指针+固定两个数 | 固定两个数,剩下的用双指针,O(n³) |
| LeetCode 16. 最接近的三数之和 | Medium | 排序+双指针 | 不要求和为0,而是最接近target |
| LeetCode 259. 较小的三数之和 | Medium | 排序+双指针 | 统计和 < target 的三元组个数 |
| LeetCode 167. 两数之和 II | Easy | 对撞双指针(有序数组) | 巩固双指针基础 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个数组 `nums` 和一个目标值 `target`,找出所有满足 `nums[i] + nums[j] + nums[k] = target` 的三元组(不要求和为0,而是等于 target)。注意不能有重复三元组。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

和本题几乎一样!只需修改一处:固定 `nums[i]` 后,目标变成找两数和为 `target - nums[i]`(而非 `-nums[i]`)。去重逻辑完全相同。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def three_sum_target(nums: List[int], target: int) -> List[List[int]]:
    """
    三数之和等于 target (通用版)
    思路:与三数之和完全相同,只是目标值不同
    """
    nums.sort()
    n = len(nums)
    result = []

    for i in range(n - 2):
        # 去重:跳过重复的固定数
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        # 双指针找两数和为 target - nums[i]
        left, right = i + 1, n - 1
        remaining = target - nums[i]

        while left < right:
            current_sum = nums[left] + nums[right]

            if current_sum == remaining:
                result.append([nums[i], nums[left], nums[right]])
                # 去重
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif current_sum < remaining:
                left += 1
            else:
                right -= 1

    return result


# 测试
print(three_sum_target([-1, 0, 1, 2, -1, -4], 0))   # 和为0,同原题
print(three_sum_target([1, 2, 3, 4, 5], 9))         # 和为9:[1,3,5],[2,3,4]
```

**核心改动**:只有一处 — 目标值从 `-nums[i]` 改为 `target - nums[i]`。这就是模板的威力!

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
