> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第6课:四数相加II

> **模块**:哈希表 | **难度**:Medium ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/4sum-ii/
> **前置知识**:第1课(两数之和)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给你四个整数数组 `nums1`、`nums2`、`nums3` 和 `nums4`,数组长度都是 `n`,请你计算有多少个元组 `(i, j, k, l)` 能满足:

- 0 ≤ i, j, k, l < n
- nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0

**示例:**
```
输入:
nums1 = [1,2]
nums2 = [-2,-1]
nums3 = [-1,2]
nums4 = [0,2]

输出:2

解释:
两个元组如下:
1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1) + 0 = 0
```

**约束条件:**
- n == nums1.length == nums2.length == nums3.length == nums4.length
- 1 ≤ n ≤ 200
- -2²⁸ ≤ nums1[i], nums2[i], nums3[i], nums4[i] ≤ 2²⁸

---

## 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | 4个长度为1的数组,和为0 | 1 | 基本功能 |
| 无解 | nums1=[1], nums2=[1], nums3=[1], nums4=[1] | 0 | 无满足条件的组合 |
| 全零 | nums1=[0,0], nums2=[0,0], nums3=[0,0], nums4=[0,0] | 16 | 所有组合都满足 |
| 大数据 | n=200 | — | 性能边界,O(n²)必须 |
| 负数 | 包含负数的混合数组 | 正确计数 | 负数处理 |

---

## 💡 思路引导

### 生活化比喻

> 想象你是一个拼图游戏玩家,手里有4盒拼图块(4个数组),每盒有n块。你需要从每盒中各选一块,使得4块拼图的"重量和"恰好为0。
>
> 🐌 **笨办法**:你把4盒拼图全摊开,一块一块试,第一盒选一块,第二盒选一块,第三盒选一块,第四盒选一块,检查和是否为0。这样要试 n×n×n×n 次,太慢了!
>
> 🚀 **聪明办法**:你先把前两盒拼图两两组合,算出所有可能的"前半段重量",记在一个小本子(哈希表)上。然后把后两盒拼图两两组合,算出"后半段重量",看看小本子上有没有对应的"负值"能凑成0。这样只需要 n² + n² 次操作,快多了!

### 关键洞察

**分而治之!将4个数组分成两组,用哈希表存储前两组的和,查询后两组的互补值!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:4个长度相同的整数数组,长度为n
- **输出**:整数,表示满足条件的元组数量
- **限制**:四个数组各选一个元素,和为0即可

### Step 2:先想笨办法(暴力法)

最直接的想法:四重循环,枚举所有可能的 (i, j, k, l) 组合,检查和是否为0。

```python
count = 0
for i in range(n):
    for j in range(n):
        for k in range(n):
            for l in range(n):
                if nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0:
                    count += 1
```

- 时间复杂度:O(n⁴)
- 瓶颈在哪:当 n=200 时,需要 200⁴ = 16亿次操作,绝对超时!

### Step 3:瓶颈分析 → 优化方向

观察四数之和的结构:
```
nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0
```

可以变形为:
```
nums1[i] + nums2[j] == -(nums3[k] + nums4[l])
```

核心问题:**这不就是"两数之和"的变体吗?**

优化思路:
1. 先计算所有 `nums1[i] + nums2[j]` 的可能值,存入哈希表,记录每个和出现的次数
2. 再遍历所有 `nums3[k] + nums4[l]`,在哈希表中查找 `-(nums3[k] + nums4[l])` 出现的次数
3. 累加次数即可

### Step 4:选择武器
- 选用:**分组哈希表**
- 理由:
  - 将4个数组分成两组,每组用 O(n²) 处理
  - 总时间 O(n²) + O(n²) = O(n²),从 O(n⁴) 降到 O(n²)!
  - 用哈希表存储频次,O(1) 查询

> 🔑 **模式识别提示**:当题目出现"多个数组求和",考虑"分组"降维 + 哈希表查找

---

## 🔑 解法一:暴力四重循环(朴素)

### 思路

枚举所有可能的四元组,检查和是否为0。

### 图解过程

```
nums1 = [1, 2]
nums2 = [-2, -1]
nums3 = [-1, 2]
nums4 = [0, 2]

暴力枚举所有组合:
i=0, j=0, k=0, l=0: 1 + (-2) + (-1) + 0 = -2 ❌
i=0, j=0, k=0, l=1: 1 + (-2) + (-1) + 2 = 0 ✅ count=1
i=0, j=0, k=1, l=0: 1 + (-2) + 2 + 0 = 1 ❌
i=0, j=0, k=1, l=1: 1 + (-2) + 2 + 2 = 3 ❌
... (共2×2×2×2=16种组合)
i=1, j=1, k=0, l=0: 2 + (-1) + (-1) + 0 = 0 ✅ count=2
...

结果:2
```

### Python代码

```python
from typing import List


def fourSumCount(nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
    """
    解法一:暴力四重循环
    思路:枚举所有可能的四元组
    """
    count = 0
    n = len(nums1)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    if nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0:
                        count += 1

    return count


# ✅ 测试
print(fourSumCount([1, 2], [-2, -1], [-1, 2], [0, 2]))  # 期望输出:2
print(fourSumCount([0], [0], [0], [0]))                 # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(n⁴) — 四重循环
  - 具体地说:如果 n=200,需要 200⁴ = 1,600,000,000 次操作,约16亿次!
- **空间复杂度**:O(1) — 只用了几个变量

### 优缺点
- ✅ 代码简单直接
- ❌ 时间复杂度太高,n>50 就会超时 → 必须优化!

---

## ⚡ 解法二:分组哈希表(优化)

### 优化思路

将4个数组分成两组:
- **第一组**:nums1 + nums2,计算所有可能的和,存入哈希表
- **第二组**:nums3 + nums4,查找哈希表中是否有互补值

关键变形:
```
nums1[i] + nums2[j] + nums3[k] + nums4[l] = 0
↓
nums1[i] + nums2[j] = -(nums3[k] + nums4[l])
```

> 💡 **关键想法**:分组后,问题变成"在哈希表中查找目标值",从O(n⁴)降到O(n²)!

### 图解过程

```
nums1 = [1, 2]
nums2 = [-2, -1]
nums3 = [-1, 2]
nums4 = [0, 2]

Step 1:计算 nums1 + nums2 的所有可能和,存入哈希表
  1 + (-2) = -1 → hashmap = {-1: 1}
  1 + (-1) = 0  → hashmap = {-1: 1, 0: 1}
  2 + (-2) = 0  → hashmap = {-1: 1, 0: 2}
  2 + (-1) = 1  → hashmap = {-1: 1, 0: 2, 1: 1}

Step 2:遍历 nums3 + nums4,查找互补值
  (-1) + 0 = -1 → 查找 -(-1) = 1,hashmap[1] = 1 → count += 1
  (-1) + 2 = 1  → 查找 -(1) = -1,hashmap[-1] = 1 → count += 1
  2 + 0 = 2     → 查找 -(2) = -2,不存在 → count += 0
  2 + 2 = 4     → 查找 -(4) = -4,不存在 → count += 0

结果:count = 2
```

**图示**:
```
第一组合并          第二组查询
nums1  nums2        nums3  nums4      查找
 [1]    [-2]         [-1]   [0]    → -(-1)=1 在表中?
 [1]    [-1]         [-1]   [2]    → -(1)=-1 在表中?
 [2]    [-2]  →哈希表→ [2]    [0]    → -(2)=-2 在表中?
 [2]    [-1]         [2]    [2]    → -(4)=-4 在表中?

 所有组合          存储{和:次数}    查找互补值
   ↓                  ↓               ↓
  n²次              O(n²)空间       n²次查询
```

### Python代码

```python
from typing import List
from collections import defaultdict


def fourSumCount_v2(nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
    """
    解法二:分组哈希表
    思路:将4个数组分成两组,用哈希表存储前两组的和
    """
    # 哈希表:存储 nums1[i] + nums2[j] 的和及其出现次数
    hashmap = defaultdict(int)

    # 第一步:计算 nums1 + nums2 的所有可能和
    for a in nums1:
        for b in nums2:
            hashmap[a + b] += 1

    # 第二步:遍历 nums3 + nums4,查找互补值
    count = 0
    for c in nums3:
        for d in nums4:
            target = -(c + d)  # 我们需要找到和为 -target 的前两组
            if target in hashmap:
                count += hashmap[target]

    return count


# ✅ 测试
print(fourSumCount_v2([1, 2], [-2, -1], [-1, 2], [0, 2]))  # 期望输出:2
print(fourSumCount_v2([0], [0], [0], [0]))                 # 期望输出:1
print(fourSumCount_v2([1], [1], [1], [1]))                 # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n²) — 两个双重循环
  - 第一步:计算 nums1 + nums2,O(n²)
  - 第二步:遍历 nums3 + nums4,O(n²)
  - 总计:O(n²) + O(n²) = O(n²)
  - 如果 n=200,只需 200² = 40,000 次操作,相比16亿次快了4万倍!
- **空间复杂度**:O(n²) — 哈希表最多存储 n² 个不同的和

---

## 🚀 解法三:分组哈希(代码优化版)

### 优化思路

解法二的代码可以进一步简化,使用Counter的特性让代码更Pythonic。

### Python代码

```python
from typing import List
from collections import Counter


def fourSumCount_v3(nums1: List[int], nums2: List[int], nums3: List[int], nums4: List[int]) -> int:
    """
    解法三:分组哈希表(Counter优化版)
    思路:使用Counter简化哈希表操作
    """
    # 计算 nums1 + nums2 的所有可能和
    sum_ab = Counter(a + b for a in nums1 for b in nums2)

    # 遍历 nums3 + nums4,查找互补值
    count = 0
    for c in nums3:
        for d in nums4:
            count += sum_ab[-(c + d)]  # Counter对不存在的键返回0

    return count


# ✅ 测试
print(fourSumCount_v3([1, 2], [-2, -1], [-1, 2], [0, 2]))  # 期望输出:2
print(fourSumCount_v3([0], [0], [0], [0]))                 # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(n²) — 与解法二相同
- **空间复杂度**:O(n²) — 与解法二相同

---

## 🐍 Pythonic 写法

终极简化版:一行搞定核心逻辑!

```python
from collections import Counter

def fourSumCount_pythonic(nums1, nums2, nums3, nums4):
    """一行流:Counter + sum"""
    sum_ab = Counter(a + b for a in nums1 for b in nums2)
    return sum(sum_ab[-(c + d)] for c in nums3 for d in nums4)

# 测试
print(fourSumCount_pythonic([1, 2], [-2, -1], [-1, 2], [0, 2]))  # 2
```

这个写法本质上是解法三,利用了:
- `Counter` 的生成器表达式创建
- `sum()` 累加所有匹配次数
- `Counter[key]` 对不存在的键返回0

> ⚠️ **面试建议**:先写解法二展示分组思想,再提解法三展示Python功底。
> 面试官更看重你的**分组降维思维**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:暴力循环 | 解法二:分组哈希 | 解法三:Counter版 |
|------|--------------|---------------|----------------|
| 时间复杂度 | O(n⁴) | O(n²) | O(n²) |
| 空间复杂度 | O(1) | O(n²) | O(n²) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | n≤20验证 | 通用推荐 | Python专场 |

**面试建议**:先讲暴力法指出O(n⁴)瓶颈,立刻提出分组思想,画图展示如何分成两组,然后写解法二代码。

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求从4个数组中各选一个元素,使得和为0,统计满足条件的组合数。

我的第一个想法是暴力枚举所有四元组,四重循环,时间复杂度是 O(n⁴)。但这样当 n=200 时会有16亿次操作,肯定超时。

我们可以用**分组哈希**优化。观察到等式可以变形:
```
nums1[i] + nums2[j] = -(nums3[k] + nums4[l])
```

这样就把问题分成两部分:
1. 先计算所有 nums1+nums2 的和,用哈希表存储每个和出现的次数,O(n²)
2. 再遍历所有 nums3+nums4 的和,查哈希表里有没有对应的互补值,O(n²)

总时间优化到 O(n²),空间 O(n²)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)我先创建一个哈希表。第一步,双重循环计算 nums1+nums2 的所有可能和,记录频次。第二步,双重循环遍历 nums3+nums4,对每个和 c+d,查找 -(c+d) 在哈希表中出现的次数,累加到结果中。(写下解法二的代码)

**面试官**:测试一下?

**你**:用示例 [1,2], [-2,-1], [-1,2], [0,2] 走一遍。
- 第一步:1+(-2)=-1, 1+(-1)=0, 2+(-2)=0, 2+(-1)=1,哈希表是 {-1:1, 0:2, 1:1}
- 第二步:
  - (-1)+0=-1,查找 1,次数1,count=1
  - (-1)+2=1,查找 -1,次数1,count=2
  - 2+0=2,查找 -2,不存在
  - 2+2=4,查找 -4,不存在
- 结果是2,正确!

**面试官**:如果要求空间O(1)怎么办?

**你**:如果必须O(1)空间,就只能回到暴力法的O(n⁴)时间。或者可以考虑三重循环+二分查找:先对nums4排序,然后三重循环枚举前3个数组,二分查找第4个数组中的目标值,时间是O(n³ log n),空间O(1)。但这比O(n²)哈希法慢很多。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能不能O(n²)时间O(1)空间?" | 不能!要么O(n⁴)时间O(1)空间,要么O(n²)时间O(n²)空间,这是时空权衡 |
| "为什么分成2组而不是3组?" | 分3组无法降维:如A+B存哈希表,C+D也存哈希表,但找不到匹配关系 |
| "如果4个数组长度不同?" | 按长度排序,最短的两个分一组,减少哈希表大小 |
| "扩展到6个数组?" | 分成3组,两两合并:AB、CD、EF,然后两层哈希表嵌套,O(n³) |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:defaultdict(int) — 访问不存在的键返回0
from collections import defaultdict
counter = defaultdict(int)
counter['a'] += 1  # 不需要判断键是否存在

# 技巧2:Counter 对不存在的键也返回0
from collections import Counter
counter = Counter({'a': 1})
print(counter['b'])  # 输出0,不会报错

# 技巧3:生成器表达式创建Counter
sum_ab = Counter(a + b for a in nums1 for b in nums2)

# 技巧4:sum() 累加生成器
total = sum(counter[key] for key in keys)
```

### 💡 底层原理(选读)

> **为什么分组能降维?**
>
> 暴力法需要枚举 n⁴ 种组合,因为4个变量相互独立。
>
> 分组后:
> - 第一组:枚举 n² 种 (i,j) 组合,计算和
> - 第二组:枚举 n² 种 (k,l) 组合,查哈希表
> - 总计:n² + n² = 2n²,而不是 n⁴
>
> 这是**分治思想**的应用:将大问题分解成独立的小问题,分别解决后合并。
>
> **时空权衡**:
> - 用 O(n²) 空间存储中间结果(哈希表)
> - 换来从 O(n⁴) 到 O(n²) 的时间优化
> - 这是算法优化中常见的策略:空间换时间

### 算法模式卡片 📐

- **模式名称**:分组哈希(Group Hashing)
- **适用条件**:
  - 需要从多个数组中各选元素组合
  - 组合满足某种和/差关系
  - 暴力枚举时间复杂度过高
- **识别关键词**:"四数之和"、"k数之和"、"多个数组组合"、"满足等式"
- **核心思想**:将多个数组分成两组,一组存哈希表,一组查询
- **模板代码**:

```python
from collections import defaultdict

def k_sum_group_hash(arrays):
    """k个数组求和的分组哈希模板"""
    # 分成两组:前half和后half
    half = len(arrays) // 2

    # 第一组:计算所有可能的和
    first_group_sums = defaultdict(int)
    # ... 枚举前half个数组的所有组合
    # first_group_sums[sum] += 1

    # 第二组:查找互补值
    count = 0
    # ... 枚举后half个数组的所有组合
    # count += first_group_sums[-sum]

    return count
```

### 易错点 ⚠️

1. **混淆"次数"和"存在性"**
   - 错误:只判断 `if -(c+d) in hashmap`,只能判断是否存在
   - 正确:`count += hashmap[-(c+d)]`,累加所有匹配的次数
   - 例如:{0:2} 表示和为0的组合有2种

2. **分组顺序的影响**
   - 虽然理论上分(1,2)vs(3,4)和分(1,3)vs(2,4)都行
   - 但为了代码清晰,通常按顺序分:前半vs后半

3. **忘记使用defaultdict或Counter**
   - 错误:`hashmap[key] += 1` 对不存在的key会报错
   - 正确:用 `defaultdict(int)` 或 `Counter`,自动初始化为0

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:推荐系统** — 在电商推荐中,计算"买了A和B的用户,同时买了C和D的概率"。4个商品集合的笛卡尔积太大,可以分组:先统计AB组合的用户群,再查询CD组合是否在同一用户群中。

- **场景2:数据分析** — 在多维数据分析中,查找满足多个条件组合的记录数。例如:年龄段A、收入段B、地区C、职业D的交集,分组查询比全表扫描快得多。

- **场景3:密码学** — 在某些哈希碰撞攻击中,需要找到满足特定关系的多个输入。分组预计算可以大幅降低搜索空间。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 1. 两数之和 | Easy | 哈希表查找互补值 | 本题的简化版,2个数组而非4个 |
| LeetCode 15. 三数之和 | Medium | 排序+双指针 | 3个数的情况,不能用哈希(需要去重) |
| LeetCode 18. 四数之和 | Medium | 排序+双指针 | 同样是4个数,但要求不重复的四元组 |
| LeetCode 653. 两数之和IV-BST | Easy | BST+哈希表 | 在二叉搜索树中找两数之和 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定6个长度为n的数组,求有多少组 (i,j,k,l,m,p) 使得:
```
nums1[i] + nums2[j] + nums3[k] + nums4[l] + nums5[m] + nums6[p] = 0
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

还是分组思想!分成3组,每组2个数组:
- 第一组:nums1 + nums2
- 第二组:nums3 + nums4
- 第三组:nums5 + nums6

但这次需要两层哈希表...

</details>

<details>
<summary>✅ 参考答案</summary>

```python
from collections import Counter

def sixSumCount(nums1, nums2, nums3, nums4, nums5, nums6):
    """6个数组的分组哈希"""
    # 第一组:nums1 + nums2
    sum_ab = Counter(a + b for a in nums1 for b in nums2)

    # 第二组:nums3 + nums4
    sum_cd = Counter(c + d for c in nums3 for d in nums4)

    # 第三组:nums5 + nums6,查找前两组的互补值
    count = 0
    for e in nums5:
        for f in nums6:
            target = -(e + f)
            # 枚举第一组和第二组的组合
            for sum1, cnt1 in sum_ab.items():
                if target - sum1 in sum_cd:
                    count += cnt1 * sum_cd[target - sum1]

    return count
```

**复杂度**:O(n³) — 3个双重循环,第三个循环内遍历哈希表(最多n²项)

**核心思路**:
- 分成3组,每组O(n²)合并
- 第三组枚举时,需要查找 sum_ab[x] + sum_cd[y] = target 的所有组合
- 相当于在两个哈希表中做"两数之和"

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
