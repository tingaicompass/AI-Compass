> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第105课:除自身以外数组的乘积

> **模块**:高级技巧 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/product-of-array-except-self/
> **前置知识**:数组遍历
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给你一个整数数组 `nums`,返回数组 `answer`,其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

题目保证数组 `nums` 之中任意元素的全部前缀元素和后缀元素的乘积都在 32 位整数范围内。

**要求:不能使用除法,且在 O(n) 时间复杂度内完成。**

**示例:**
```
输入:nums = [1,2,3,4]
输出:[24,12,8,6]
解释:
  answer[0] = 2*3*4 = 24
  answer[1] = 1*3*4 = 12
  answer[2] = 1*2*4 = 8
  answer[3] = 1*2*3 = 6

输入:nums = [-1,1,0,-3,3]
输出:[0,0,9,0,0]
```

**约束条件:**
- 2 <= nums.length <= 100000
- -30 <= nums[i] <= 30
- 保证乘积在 32 位整数范围内
- **进阶:能否用 O(1) 空间复杂度?(输出数组不计入空间复杂度)**

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1,2] | [2,1] | 两元素数组 |
| 包含0 | nums=[0,1] | [1,0] | 零的特殊处理 |
| 多个0 | nums=[0,0] | [0,0] | 多个零 |
| 负数 | nums=[-1,2,-3] | [(-6),3,(-2)] | 负数乘积 |
| 全1 | nums=[1,1,1] | [1,1,1] | 边界情况 |
| 大规模 | n=100000 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在一个工厂流水线上,需要计算每个位置"除了当前站点"的总产量。
>
> 🐌 **笨办法**:对于每个站点,把其他所有站点的产量全部乘起来。这需要每次都重新遍历整条流水线(O(n²)时间)。
>
> 🚀 **聪明办法**:提前计算好每个站点"左侧的累积产量"和"右侧的累积产量",然后对于每个站点,它的答案就是"左侧累积 × 右侧累积"。这样只需要遍历两次流水线(O(n)时间)!

### 关键洞察

**对于位置 i 的答案 = 它左侧所有元素的乘积 × 它右侧所有元素的乘积**

```
nums:    [a, b, c, d]

answer[1] (位置b的答案):
  = 左侧乘积 × 右侧乘积
  = a × (c*d)
```

---

## 🧠 解题思维链

### Step 1:理解题目 → 锁定输入输出

- **输入**:整数数组 nums,长度 n >= 2
- **输出**:整数数组 answer,其中 answer[i] = nums中除nums[i]外所有元素的乘积
- **限制**:不能用除法,必须 O(n) 时间

### Step 2:先想笨办法(暴力法)

对于每个位置 i,用嵌套循环计算除 nums[i] 外所有元素的乘积:
- 时间复杂度:O(n²)
- 瓶颈在哪:**每个位置都要重新遍历整个数组**

### Step 3:瓶颈分析 → 优化方向

核心问题:如何避免重复计算?

关键观察:
- answer[i] 可以分解为"左侧乘积"和"右侧乘积"
- 如果提前计算好所有位置的左侧乘积和右侧乘积,就可以 O(1) 得到答案

优化思路:
- 能否用**前缀积**和**后缀积**的思想,预先计算好所有位置的"左右两侧"的乘积?

### Step 4:选择武器

- 选用:**前缀积 + 后缀积**
- 理由:类似前缀和的思想,用空间换时间,将 O(n²) 降为 O(n)

> 🔑 **模式识别提示**:当题目要求"除自身外的统计量"时,考虑"左右分治"或"前缀后缀"

---

## 🔑 解法一:暴力双循环(直觉法)

### 思路

对每个位置 i,用内层循环计算除 nums[i] 外所有元素的乘积。

### 图解过程

```
nums = [1,2,3,4]

对于 i=0 (nums[0]=1):
  乘积 = 2*3*4 = 24

对于 i=1 (nums[1]=2):
  乘积 = 1*3*4 = 12

对于 i=2 (nums[2]=3):
  乘积 = 1*2*4 = 8

对于 i=3 (nums[3]=4):
  乘积 = 1*2*3 = 6

answer = [24,12,8,6]
```

### Python代码

```python
from typing import List


def productExceptSelf_bruteforce(nums: List[int]) -> List[int]:
    """
    解法一:暴力双循环
    思路:对每个位置,遍历其他所有位置计算乘积
    """
    n = len(nums)
    answer = []

    for i in range(n):
        product = 1
        for j in range(n):
            if j != i:  # 跳过自己
                product *= nums[j]
        answer.append(product)

    return answer


# ✅ 测试
print(productExceptSelf_bruteforce([1, 2, 3, 4]))  # 期望输出:[24,12,8,6]
print(productExceptSelf_bruteforce([-1, 1, 0, -3, 3]))  # 期望输出:[0,0,9,0,0]
```

### 复杂度分析

- **时间复杂度**:O(n²) — 两层嵌套循环,每次计算一个位置需要遍历 n 个元素
  - 具体地说:如果输入规模 n=1000,大约需要 1000×1000 = 1,000,000 次乘法
- **空间复杂度**:O(1) — 不计输出数组,只用了常量级变量

### 优缺点

- ✅ 思路直观,代码简单
- ❌ **时间复杂度过高,n=100000时会超时**

---

## ⚡ 解法二:前缀积 + 后缀积数组(优化)

### 优化思路

为每个位置 i 预先计算:
- `prefix[i]`:位置 i 左侧所有元素的乘积
- `suffix[i]`:位置 i 右侧所有元素的乘积

然后 `answer[i] = prefix[i] × suffix[i]`

> 💡 **关键想法**:把"除自身外"分解为"左侧"+"右侧",分别预计算

### 图解过程

```
nums = [1, 2, 3, 4]

步骤1:计算前缀积(左侧乘积)
  prefix[0] = 1         (左侧没有元素)
  prefix[1] = 1         (左侧是1)
  prefix[2] = 1*2 = 2   (左侧是1,2)
  prefix[3] = 1*2*3 = 6 (左侧是1,2,3)
  prefix = [1, 1, 2, 6]

步骤2:计算后缀积(右侧乘积)
  suffix[3] = 1         (右侧没有元素)
  suffix[2] = 4         (右侧是4)
  suffix[1] = 4*3 = 12  (右侧是3,4)
  suffix[0] = 4*3*2 = 24(右侧是2,3,4)
  suffix = [24, 12, 4, 1]

步骤3:计算答案
  answer[0] = prefix[0] * suffix[0] = 1 * 24 = 24
  answer[1] = prefix[1] * suffix[1] = 1 * 12 = 12
  answer[2] = prefix[2] * suffix[2] = 2 * 4  = 8
  answer[3] = prefix[3] * suffix[3] = 6 * 1  = 6
  answer = [24, 12, 8, 6] ✓
```

### Python代码

```python
def productExceptSelf_v2(nums: List[int]) -> List[int]:
    """
    解法二:前缀积 + 后缀积数组
    思路:分别计算每个位置的左侧乘积和右侧乘积
    """
    n = len(nums)
    prefix = [1] * n  # 前缀积数组
    suffix = [1] * n  # 后缀积数组

    # 步骤1:计算前缀积(从左到右)
    for i in range(1, n):
        prefix[i] = prefix[i - 1] * nums[i - 1]

    # 步骤2:计算后缀积(从右到左)
    for i in range(n - 2, -1, -1):
        suffix[i] = suffix[i + 1] * nums[i + 1]

    # 步骤3:计算答案
    answer = [prefix[i] * suffix[i] for i in range(n)]

    return answer


# ✅ 测试
print(productExceptSelf_v2([1, 2, 3, 4]))  # 期望输出:[24,12,8,6]
print(productExceptSelf_v2([-1, 1, 0, -3, 3]))  # 期望输出:[0,0,9,0,0]
```

### 复杂度分析

- **时间复杂度**:O(n) — 三次遍历,每次 O(n)
  - 具体地说:如果输入规模 n=10000,大约需要 30000 次操作
- **空间复杂度**:O(n) — 需要两个辅助数组 prefix 和 suffix

---

## 🏆 解法三:前缀积 + 后缀积优化(最优解)

### 优化思路

观察解法二:我们真的需要两个完整的数组吗?

**核心优化**:
1. 先用输出数组 `answer` 存储前缀积
2. 再用一个变量 `suffix_product` 从右往左滚动计算后缀积,边算边更新 `answer`

这样空间复杂度从 O(n) 降为 O(1)!

> 💡 **关键想法**:复用输出数组,用变量滚动替代完整数组

### 图解过程

```
nums = [1, 2, 3, 4]

步骤1:用 answer 存储前缀积
  answer[0] = 1         (左侧没有元素)
  answer[1] = 1         (左侧是1)
  answer[2] = 1*2 = 2   (左侧是1,2)
  answer[3] = 1*2*3 = 6 (左侧是1,2,3)
  answer = [1, 1, 2, 6]

步骤2:从右往左滚动后缀积,边乘边更新 answer
  suffix_product = 1

  i=3: answer[3] = answer[3] * suffix_product = 6 * 1 = 6
       suffix_product = suffix_product * nums[3] = 1 * 4 = 4

  i=2: answer[2] = answer[2] * suffix_product = 2 * 4 = 8
       suffix_product = suffix_product * nums[2] = 4 * 3 = 12

  i=1: answer[1] = answer[1] * suffix_product = 1 * 12 = 12
       suffix_product = suffix_product * nums[1] = 12 * 2 = 24

  i=0: answer[0] = answer[0] * suffix_product = 1 * 24 = 24

  answer = [24, 12, 8, 6] ✓
```

### Python代码

```python
def productExceptSelf(nums: List[int]) -> List[int]:
    """
    解法三:前缀积 + 后缀积优化(最优解)
    思路:复用输出数组存前缀积,用变量滚动后缀积
    """
    n = len(nums)
    answer = [1] * n

    # 步骤1:计算前缀积,存入 answer
    prefix_product = 1
    for i in range(n):
        answer[i] = prefix_product  # 当前位置的左侧乘积
        prefix_product *= nums[i]   # 更新前缀积

    # 步骤2:从右往左滚动后缀积,边乘边更新 answer
    suffix_product = 1
    for i in range(n - 1, -1, -1):
        answer[i] *= suffix_product  # 乘上右侧乘积
        suffix_product *= nums[i]    # 更新后缀积

    return answer


# ✅ 测试
print(productExceptSelf([1, 2, 3, 4]))  # 期望输出:[24,12,8,6]
print(productExceptSelf([-1, 1, 0, -3, 3]))  # 期望输出:[0,0,9,0,0]
print(productExceptSelf([1, 2]))  # 期望输出:[2,1]
```

### 复杂度分析

- **时间复杂度**:O(n) — 两次遍历,每次 O(n)
  - 具体地说:如果输入规模 n=100000,大约需要 200000 次操作
- **空间复杂度**:O(1) — 除了输出数组外,只用了两个变量

**为什么是最优解:**
- 时间复杂度 O(n) 已经是理论最优(至少要遍历一次数组)
- 空间复杂度 O(1) 满足进阶要求(不计输出数组)
- 无需除法,满足题目约束
- 代码简洁清晰,易于理解和实现

---

## 🐍 Pythonic 写法

利用 Python 的列表推导和 `itertools.accumulate`:

```python
from itertools import accumulate
from operator import mul


def productExceptSelf_pythonic(nums: List[int]) -> List[int]:
    """
    Pythonic 写法:使用 accumulate 计算前缀积
    """
    n = len(nums)

    # 前缀积(包含当前元素)
    prefix = list(accumulate(nums, mul, initial=1))

    # 后缀积(包含当前元素)
    suffix = list(accumulate(reversed(nums), mul, initial=1))[::-1]

    # 答案:左侧乘积 * 右侧乘积
    return [prefix[i] * suffix[i + 1] for i in range(n)]


# 测试
print(productExceptSelf_pythonic([1, 2, 3, 4]))  # 输出:[24,12,8,6]
```

**解释:**
- `accumulate(nums, mul, initial=1)` 生成累积乘积序列
- `reversed(nums)` 反转数组计算后缀积
- 一行列表推导完成最终计算

> ⚠️ **面试建议**:先写清晰的循环版本展示思路,再提 Pythonic 写法展示语言功底。面试官更看重你的**优化思路**,而非库函数使用。

---

## 📊 解法对比

| 维度 | 解法一:暴力双循环 | 解法二:前缀后缀数组 | 🏆 解法三:空间优化(最优) |
|------|-----------------|-------------------|----------------------|
| 时间复杂度 | O(n²) | O(n) | **O(n)** ← 时间最优 |
| 空间复杂度 | O(1) | O(n) | **O(1)** ← 空间最优 |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | n很小 | 理解思路 | **面试首选,时间空间双优** |

**为什么解法三是最优:**
- 时间 O(n) 已达最优(必须至少遍历一次数组)
- 空间 O(1) 满足进阶要求,不需额外数组
- 巧妙复用输出数组,体现了对空间的精细控制
- 两次遍历清晰可读,易于调试

**面试建议**:
1. 先用30秒口述暴力法思路(O(n²)),表明你能想到基本解法
2. 立即指出瓶颈:"每个位置都重新计算,有大量重复"
3. **重点讲解🏆最优解**:"用前缀积和后缀积分解问题,复用输出数组节省空间"
4. 强调优化亮点:"从 O(n) 空间优化到 O(1),展示对空间复杂度的深入理解"
5. 手动模拟一个小示例,展示前缀后缀的计算过程

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求计算除自身外所有元素的乘积。我的第一个想法是对每个位置用嵌套循环计算其他元素的乘积,时间复杂度是 O(n²)。

不过这有大量重复计算,我可以用**前缀积和后缀积**的思想优化:对于位置 i,它的答案等于"左侧所有元素的乘积"乘以"右侧所有元素的乘积"。如果预先计算好每个位置的左侧乘积和右侧乘积,就可以 O(1) 得到每个位置的答案,总时间 O(n)。

**面试官**:很好,空间复杂度呢?

**你**:如果用两个数组分别存前缀积和后缀积,空间是 O(n)。但我可以优化:先用输出数组存前缀积,然后用一个变量从右往左滚动计算后缀积,边计算边更新输出数组。这样除了输出数组外,只需要 O(1) 额外空间。

**面试官**:请写一下代码。

**你**:(边写边说)
```python
def productExceptSelf(nums):
    n = len(nums)
    answer = [1] * n

    # 第一遍:计算前缀积
    prefix = 1
    for i in range(n):
        answer[i] = prefix  # 左侧乘积
        prefix *= nums[i]   # 累乘

    # 第二遍:从右往左滚动后缀积
    suffix = 1
    for i in range(n-1, -1, -1):
        answer[i] *= suffix  # 乘上右侧乘积
        suffix *= nums[i]    # 累乘

    return answer
```

**面试官**:测试一下?

**你**:用示例 [1,2,3,4] 走一遍:

第一遍(前缀积):
- i=0: answer[0]=1, prefix变为1
- i=1: answer[1]=1, prefix变为2
- i=2: answer[2]=2, prefix变为6
- i=3: answer[3]=6, prefix变为24
- answer = [1,1,2,6]

第二遍(后缀积):
- i=3: answer[3]=6*1=6, suffix变为4
- i=2: answer[2]=2*4=8, suffix变为12
- i=1: answer[1]=1*12=12, suffix变为24
- i=0: answer[0]=1*24=24, suffix变为24
- answer = [24,12,8,6] ✓

再测一个包含0的情况 [0,1]:
- 第一遍:answer = [1,0]
- 第二遍:answer = [1,0] ✓

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果允许用除法呢?" | "可以先计算所有元素的总乘积 total,然后 answer[i] = total / nums[i]。但需要特殊处理0:如果有多个0,所有答案都是0;如果有一个0,只有那个位置的答案是非零总乘积,其他都是0。时间 O(n),空间 O(1)。" |
| "为什么不用除法更简单?" | "题目明确要求不能用除法,可能是为了避免除以0的边界情况,或者考察对前缀后缀思想的理解。此外,除法的精度问题也可能导致错误。" |
| "能否用递归实现?" | "可以,但递归需要 O(n) 栈空间,不如迭代的 O(1) 空间。递归思路:计算左侧递归乘积和右侧递归乘积,然后合并。" |
| "如果数组很大,怎么优化?" | "当前解法已经是时间空间双重最优。如果内存极其受限,可以考虑流式处理:分块计算,但需要多次遍历。实际工程中,O(n)时间+O(1)空间已足够优秀。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:前缀积计算 — 滚动累乘
def prefix_product(nums):
    result = []
    product = 1
    for num in nums:
        result.append(product)
        product *= num
    return result

# 技巧2:后缀积计算 — 反向滚动
def suffix_product(nums):
    n = len(nums)
    result = [1] * n
    product = 1
    for i in range(n - 1, -1, -1):
        result[i] = product
        product *= nums[i]
    return result

# 技巧3:itertools.accumulate 计算累积
from itertools import accumulate
from operator import mul
prefix = list(accumulate([1,2,3,4], mul, initial=1))
# 结果:[1, 1, 2, 6, 24]

# 技巧4:列表推导 + zip 结合
def combine_prefix_suffix(prefix, suffix):
    return [p * s for p, s in zip(prefix, suffix)]
```

### 💡 底层原理(选读)

> **前缀积(Prefix Product)是前缀和的变体**
>
> **前缀和的定义:**
> - `prefix_sum[i] = nums[0] + nums[1] + ... + nums[i-1]`
> - 用途:快速计算区间和 `[l,r]` = `prefix_sum[r+1] - prefix_sum[l]`
>
> **前缀积的定义:**
> - `prefix_prod[i] = nums[0] * nums[1] * ... * nums[i-1]`
> - 用途:快速计算除自身外的乘积
>
> **为什么前缀积不能像前缀和那样做"区间乘积"?**
> - 加法有逆运算(减法):a+b-b=a
> - 乘法的逆运算是除法:a*b/b=a,但题目禁止除法
> - 所以需要同时维护前缀积和后缀积
>
> **数学本质:**
> ```
> answer[i] = ∏(j≠i) nums[j]
>           = (∏(j<i) nums[j]) × (∏(j>i) nums[j])
>           = prefix[i] × suffix[i]
> ```
>
> **应用场景:**
> - 数据分析:计算"去掉某个异常值后的统计量"
> - 财务计算:计算"除某项外的总成本"
> - 机器学习:Batch Normalization 中的归一化计算

### 算法模式卡片 📐

- **模式名称**:前缀后缀分解 — 除自身外的统计
- **适用条件**:需要计算"除当前位置外"的全局统计量(和、积、最值等)
- **识别关键词**:"除自身外"、"其余元素"、"去掉当前元素"
- **模板代码**:
```python
def except_self_pattern(nums):
    """前缀后缀模板:计算除自身外的统计量"""
    n = len(nums)
    answer = [None] * n

    # 步骤1:前缀统计(左侧)
    prefix = 初始值
    for i in range(n):
        answer[i] = prefix
        prefix = 更新(prefix, nums[i])

    # 步骤2:后缀统计(右侧)
    suffix = 初始值
    for i in range(n - 1, -1, -1):
        answer[i] = 合并(answer[i], suffix)
        suffix = 更新(suffix, nums[i])

    return answer
```

### 易错点 ⚠️

1. **前缀后缀范围错误**
   - 错误:`prefix[i]` 包含 `nums[i]` 本身
   - 原因:题目要求"除自身外",所以前缀积应该是 i 之前的元素
   - 正确:`answer[i] = prefix` 在 `prefix *= nums[i]` 之前

2. **初始值设置错误**
   - 错误:前缀积或后缀积初始化为0
   - 原因:乘法的单位元是1,不是0,用0会导致所有结果都是0
   - 正确:`prefix = 1`, `suffix = 1`

3. **循环方向错误**
   - 错误:后缀积也从左往右遍历
   - 原因:后缀积要从右往左累乘,才能得到"右侧元素的乘积"
   - 正确:`for i in range(n-1, -1, -1)`

4. **边界情况处理**
   - 错误:没有测试包含0的情况
   - 原因:0在乘法中是"吸收元",会影响所有相关位置
   - 正确:测试 `[0,1]`, `[0,0]`, `[1,0,2]` 等

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:数据分析中的归一化**
  - 在机器学习的 Batch Normalization 中,需要计算"除当前样本外的均值和方差"
  - 用前缀和/后缀和可以高效实现 Leave-One-Out 统计

- **场景2:推荐系统的协同过滤**
  - 计算"除当前用户外,其他用户对某商品的平均评分"
  - 前缀后缀思想避免重复计算

- **场景3:时间序列分析**
  - 计算"移除某个时间点后的趋势线"
  - 前缀后缀统计量可以快速更新

- **场景4:财务系统的敏感性分析**
  - 计算"去掉某项成本后的总成本"
  - 前缀后缀乘积可以快速得到结果

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 560. 和为K的子数组 | Medium | 前缀和+哈希 | 用前缀和转化为两数之和问题 |
| LeetCode 724. 寻找数组中心下标 | Easy | 前缀和 | 左侧和 = 右侧和的位置 |
| LeetCode 1031. 两个非重叠子数组的最大和 | Medium | 前缀最大值 | 维护左侧最大和右侧最大 |
| LeetCode 42. 接雨水 | Hard | 前缀最大+后缀最大 | 类似思想:左右两侧的最大高度 |
| LeetCode 152. 乘积最大子数组 | Medium | 前缀积 | 同时维护最大和最小前缀积 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个整数数组 `nums`,返回数组 `answer`,其中 `answer[i]` 等于 `nums` 中所有元素的和,除了 `nums[i]`。要求 O(n) 时间,O(1) 空间。

例如:nums = [1,2,3,4] 返回 [9,8,7,6]

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

先计算总和 total,然后对于每个位置 i,答案就是 total - nums[i]。一次遍历计算总和,第二次遍历计算答案,O(n) 时间 O(1) 空间!

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def sum_except_self(nums: List[int]) -> List[int]:
    """
    计算除自身外的元素和
    思路:总和减去当前元素
    """
    # 步骤1:计算总和
    total = sum(nums)

    # 步骤2:对每个位置,答案 = 总和 - 当前元素
    answer = [total - num for num in nums]

    return answer


# 测试
print(sum_except_self([1, 2, 3, 4]))  # 输出:[9, 8, 7, 6]
print(sum_except_self([5, 5, 5]))     # 输出:[10, 10, 10]
```

**核心思路**:
- 加法有逆运算(减法),所以可以用"总和 - 当前元素"直接得到答案
- 这比乘法简单,因为题目禁止除法,但没有禁止减法

**为什么本题(乘积版本)不能用这个思路?**
- 乘法的逆运算是除法,题目明确禁止使用除法
- 所以只能用前缀积+后缀积的方法

**时间复杂度**:O(n),**空间复杂度**:O(1)

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
