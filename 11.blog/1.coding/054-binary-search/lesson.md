> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第54课:二分查找

> **模块**:二分查找 | **难度**:Easy ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/binary-search/
> **前置知识**:数组基础
> **预计学习时间**:15分钟

---

## 🎯 题目描述

给定一个升序排列的整数数组 `nums` 和一个目标值 `target`,在数组中查找 `target` 并返回它的索引。如果目标值不存在于数组中,返回 `-1`。

你必须编写一个时间复杂度为 O(log n) 的算法。

**示例:**
```
输入:nums = [-1,0,3,5,9,12], target = 9
输出:4
解释:9 存在于 nums 中并且下标为 4
```

**示例 2:**
```
输入:nums = [-1,0,3,5,9,12], target = 2
输出:-1
解释:2 不存在于 nums 中因此返回 -1
```

**约束条件:**
- 1 <= nums.length <= 10^4
- -10^4 < nums[i], target < 10^4
- nums 中的所有整数互不相同
- nums 按升序排列

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单元素数组(找到) | nums=[5], target=5 | 0 | 边界处理 |
| 单元素数组(未找到) | nums=[5], target=-5 | -1 | 边界处理 |
| 目标在首位 | nums=[1,2,3], target=1 | 0 | 左边界 |
| 目标在末位 | nums=[1,2,3], target=3 | 2 | 右边界 |
| 目标不存在(小于最小值) | nums=[1,5,9], target=0 | -1 | 下界检查 |
| 目标不存在(大于最大值) | nums=[1,5,9], target=10 | -1 | 上界检查 |
| 大规模数组 | n=10^4 | — | 性能O(log n) |

---

## 💡 思路引导

### 生活化比喻
> 想象你在图书馆找一本编号为 9527 的书,书架上的书按编号升序排列...
>
> 🐌 **笨办法**:从第一本书开始逐个检查编号,直到找到 9527。如果有 10000 本书,最坏情况下你要翻 10000 次!
>
> 🚀 **聪明办法**:先翻到中间那本书,看编号是 5000。因为 9527 > 5000,所以目标书一定在右半边。然后再翻右半边的中间,发现编号是 7500。现在 9527 > 7500,继续往右半边找...每次都能排除一半的书,只需要翻 14 次就能找到(因为 2^14 > 10000)!

这就是**二分查找**的核心思想:**利用有序性,每次排除一半候选者**。

### 关键洞察
**有序数组 + O(1)随机访问 = 二分查找的完美舞台!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:`nums` 是升序整数数组,`target` 是目标值
- **输出**:返回目标值的索引,不存在返回 -1
- **限制**:必须 O(log n) 时间复杂度(这是最关键的约束!)

### Step 2:先想笨办法(暴力法)
从头到尾线性扫描数组,遇到 `target` 就返回索引。
- 时间复杂度:O(n) — 需要遍历整个数组
- 瓶颈在哪:**完全没有利用"有序"这个条件**,浪费了宝贵的信息

### Step 3:瓶颈分析 → 优化方向
线性查找在无序数组中是最优解,但在有序数组中效率低下。
- 核心问题:每次只能排除一个元素
- 优化思路:能不能每次排除"一半"元素? → 利用有序性 + 分治思想

### Step 4:选择武器
- 选用:**二分查找(Binary Search)**
- 理由:
  - 数组有序 → 可以通过中间元素的大小关系确定目标在哪半边
  - 数组支持 O(1) 随机访问 → 可以快速定位中间元素
  - 每次排除一半 → 时间复杂度降为 O(log n)

> 🔑 **模式识别提示**:当题目出现"有序数组"+"O(log n)要求"时,优先考虑"二分查找"模式

---

## 🔑 解法一:线性查找(直觉法)

### 思路
遍历数组,逐个比较元素是否等于目标值。虽然不是最优解,但展示了最直接的思路。

### 图解过程

```
示例:nums = [-1,0,3,5,9,12], target = 9

Step 1: 检查 nums[0] = -1 ≠ 9,继续
  ↓
 [-1, 0, 3, 5, 9, 12]

Step 2: 检查 nums[1] = 0 ≠ 9,继续
  ↓
 [-1, 0, 3, 5, 9, 12]

Step 3-4: 继续检查 3, 5...

Step 5: 检查 nums[4] = 9 = 9,找到!
  ↓
 [-1, 0, 3, 5, 9, 12]
              ↑
           返回索引 4
```

### Python代码

```python
from typing import List


def search_linear(nums: List[int], target: int) -> int:
    """
    解法一:线性查找
    思路:从头到尾遍历,找到就返回索引
    """
    # 遍历数组
    for i in range(len(nums)):
        if nums[i] == target:  # 找到目标值
            return i

    return -1  # 未找到


# ✅ 测试
print(search_linear([-1, 0, 3, 5, 9, 12], 9))  # 期望输出:4
print(search_linear([-1, 0, 3, 5, 9, 12], 2))  # 期望输出:-1
print(search_linear([5], 5))                   # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n) — 最坏情况下需要遍历整个数组
  - 具体地说:如果输入规模 n=10000,最坏需要 10000 次比较操作
- **空间复杂度**:O(1) — 只用了一个循环变量

### 优缺点
- ✅ 代码简单易懂
- ✅ 适用于无序数组
- ❌ **没有利用有序性**,在大规模数据时效率低下
- ❌ 不满足题目 O(log n) 的要求

---

## 🏆 解法二:标准二分查找(最优解)

### 优化思路
既然数组有序,我们可以每次检查中间元素,根据大小关系排除一半的搜索空间。

> 💡 **关键想法**:利用有序性,通过"三向比较"(大于/等于/小于)将问题规模每次减半,从 O(n) 降为 O(log n)

### 图解过程

```
示例:nums = [-1,0,3,5,9,12], target = 9

初始状态:
  left=0, right=5
  [-1, 0, 3, 5, 9, 12]
   ↑           ↑
  left       right

第 1 轮:
  mid = (0+5)/2 = 2
  nums[mid] = 3 < 9 → 目标在右半边
  [-1, 0, 3, 5, 9, 12]
   ×  ×  × mid  ?  ?
  更新:left = mid + 1 = 3

第 2 轮:
  left=3, right=5, mid=(3+5)/2=4
  nums[mid] = 9 = 9 → 找到!
  [-1, 0, 3, 5, 9, 12]
             × mid ✓
  返回 mid = 4
```

**边界情况示例**:未找到的情况
```
nums = [-1,0,3,5,9,12], target = 2

第 1 轮:mid=2, nums[2]=3 > 2 → 左半边
  left=0, right=1

第 2 轮:mid=0, nums[0]=-1 < 2 → 右半边
  left=1, right=1

第 3 轮:mid=1, nums[1]=0 < 2 → 右半边
  left=2, right=1 → left > right,退出循环

返回 -1(未找到)
```

### Python代码

```python
def search(nums: List[int], target: int) -> int:
    """
    解法二:标准二分查找
    思路:每次取中间元素比较,根据大小关系排除一半
    """
    # 初始化左右指针
    left, right = 0, len(nums) - 1

    # 当搜索区间有效时循环
    while left <= right:
        # 计算中间位置(防溢出写法)
        mid = left + (right - left) // 2

        if nums[mid] == target:  # 找到目标
            return mid
        elif nums[mid] < target:  # 目标在右半边
            left = mid + 1  # 排除左半边(包括mid)
        else:  # nums[mid] > target,目标在左半边
            right = mid - 1  # 排除右半边(包括mid)

    return -1  # 搜索区间为空,未找到


# ✅ 测试
print(search([-1, 0, 3, 5, 9, 12], 9))  # 期望输出:4
print(search([-1, 0, 3, 5, 9, 12], 2))  # 期望输出:-1
print(search([5], 5))                   # 期望输出:0
print(search([5], -5))                  # 期望输出:-1
print(search([1, 2, 3, 4, 5], 1))       # 期望输出:0
print(search([1, 2, 3, 4, 5], 5))       # 期望输出:4
```

### 复杂度分析
- **时间复杂度**:O(log n) — 每次循环排除一半元素
  - 具体地说:如果输入规模 n=10000,只需要 log₂(10000) ≈ 14 次比较
  - **性能提升**:相比线性查找,从 10000 次降为 14 次,快了 700 倍!
- **空间复杂度**:O(1) — 只用了三个指针变量 left、right、mid

**为什么是 O(log n)?**
- 假设数组长度为 n
- 第 1 次查找:搜索空间为 n
- 第 2 次查找:搜索空间为 n/2
- 第 3 次查找:搜索空间为 n/4
- ...
- 第 k 次查找:搜索空间为 n/(2^k)
- 当搜索空间缩减到 1 时:n/(2^k) = 1 → k = log₂(n)

### 优缺点
- ✅ **时间复杂度最优**:O(log n) 是有序数组查找的理论下限
- ✅ 空间复杂度 O(1),原地操作
- ✅ 代码简洁清晰,易于实现
- ⚠️ **必须保证数组有序**,这是前提条件

---

## 🐍 Pythonic 写法

利用 Python 的 `bisect` 模块,标准库已经实现了高效的二分查找:

```python
import bisect


def search_bisect(nums: List[int], target: int) -> int:
    """
    Pythonic 写法:使用 bisect 模块
    bisect_left 返回插入位置(左边界)
    """
    idx = bisect.bisect_left(nums, target)
    # 需要验证该位置的值是否等于 target
    if idx < len(nums) and nums[idx] == target:
        return idx
    return -1


# ✅ 测试
print(search_bisect([-1, 0, 3, 5, 9, 12], 9))  # 期望输出:4
print(search_bisect([-1, 0, 3, 5, 9, 12], 2))  # 期望输出:-1
```

**解释**:
- `bisect.bisect_left(nums, target)` 返回 target 应该插入的**最左位置**
- 如果 target 存在,该位置的值就是 target
- 如果 target 不存在,该位置的值会大于 target

> ⚠️ **面试建议**:先手写标准二分展示思路,再提 bisect 展示对标准库的了解。面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:线性查找 | 🏆 解法二:二分查找(最优) | Pythonic:bisect |
|------|--------------|----------------------|----------------|
| 时间复杂度 | O(n) | **O(log n)** ← 最优 | **O(log n)** |
| 空间复杂度 | O(1) | **O(1)** | **O(1)** |
| 代码难度 | 简单 | 中等 | 简单 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 必会 | ⭐⭐ |
| 适用场景 | 无序数组/数据量小 | **有序数组/数据量大** | 工程实践快速开发 |

**为什么二分查找是最优解**:
- **时间复杂度 O(log n) 已经是理论最优**:在有序数组中,基于比较的查找算法不可能低于 O(log n)
- **空间复杂度 O(1)**:不需要额外空间,原地操作
- **充分利用有序性**:这是数据结构给我们的"免费午餐",不用白不用
- **性能提升巨大**:在 n=1000000 时,线性查找需要 100 万次,二分只需 20 次!

**面试建议**:
1. 先用 30 秒口述线性查找思路(O(n)),证明你能想到基本解法
2. 立即优化到🏆二分查找(O(log n)),展示优化能力
3. **重点讲解二分的核心思想**:"利用有序性,每次排除一半搜索空间"
4. 强调为什么这是最优:时间已达理论下限 O(log n),无法再优化
5. 手动演示边界用例(如单元素数组、目标不存在),展示对边界处理的理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你在一个有序数组中查找目标值,要求时间复杂度 O(log n)。

**你**:(审题30秒)好的,这道题有两个关键信息:1) 数组有序 2) 要求 O(log n)。这明确提示我要用**二分查找**。

让我先说一下思路:最直接的方法是线性扫描,但时间复杂度 O(n),不满足要求。既然数组有序,我可以每次取中间元素比较:如果中间值等于目标就返回;如果中间值小于目标,说明目标在右半边;如果中间值大于目标,说明目标在左半边。这样每次排除一半,时间复杂度就是 O(log n)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def search(nums, target):
    left, right = 0, len(nums) - 1  # 初始化左右边界

    while left <= right:  # 注意是 <= ,确保区间有效
        mid = left + (right - left) // 2  # 防溢出写法

        if nums[mid] == target:
            return mid  # 找到目标
        elif nums[mid] < target:
            left = mid + 1  # 排除左半边
        else:
            right = mid - 1  # 排除右半边

    return -1  # 未找到
```

关键点有三个:
1. **循环条件是 `left <= right`**:等号保证单元素区间也能检查
2. **mid 的计算用 `left + (right - left) // 2`**:防止 left+right 整数溢出(虽然 Python 没这问题,但这是好习惯)
3. **更新边界时排除 mid**:left = mid+1 或 right = mid-1,避免死循环

**面试官**:测试一下?

**你**:用示例 nums=[-1,0,3,5,9,12], target=9 走一遍:
- 第 1 轮:left=0, right=5, mid=2, nums[2]=3 < 9,更新 left=3
- 第 2 轮:left=3, right=5, mid=4, nums[4]=9,找到!返回 4 ✓

再测一个边界情况,nums=[5], target=5:
- 第 1 轮:left=0, right=0, mid=0, nums[0]=5,找到!返回 0 ✓

再测一个不存在的情况,nums=[1,3,5], target=2:
- 第 1 轮:mid=1, nums[1]=3 > 2,更新 right=0
- 第 2 轮:mid=0, nums[0]=1 < 2,更新 left=1
- left > right,退出循环,返回 -1 ✓

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么循环条件是 `<=` 而不是 `<`?" | 因为 left==right 时区间 [left, right] 还包含一个元素,需要检查。如果用 `<`,单元素会漏掉 |
| "如果有重复元素怎么办?" | 标准二分找任意一个即可。如果要找第一个或最后一个,需要用左/右边界二分变体 |
| "能用递归实现吗?" | 可以,但迭代更节省空间。递归需要 O(log n) 栈空间,迭代只需 O(1) |
| "如果数组是降序的呢?" | 只需修改比较逻辑:nums[mid] > target 时往右走,nums[mid] < target 时往左走 |
| "能否在旋转排序数组中二分?" | 可以!先判断哪半边有序,再决定往哪边走。这是 LeetCode 33 题 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧 1:防溢出的 mid 计算
mid = left + (right - left) // 2  # 推荐写法
# 等价于 mid = (left + right) // 2,但在其他语言中防整数溢出

# 技巧 2:bisect 模块快速二分
import bisect
idx = bisect.bisect_left(nums, target)  # 查找插入位置(左边界)
# bisect_right 返回右边界

# 技巧 3:负数索引的妙用
# Python 中 nums[-1] 表示最后一个元素
right = len(nums) - 1  # 等价于 right = -1 的逻辑含义
```

### 💡 底层原理(选读)

> **为什么二分查找这么快?**
>
> 这涉及到**信息论**的概念。每次比较可以得到 1 bit 的信息(大于/等于/小于其实是 1.58 bit),用来排除一半的候选者。对于 n 个元素,需要 log₂(n) bit 的信息才能唯一确定一个元素,所以 log n 是理论下限。
>
> **为什么数组可以二分,链表不行?**
>
> 二分查找的关键是能 **O(1) 访问中间元素**。数组支持随机访问(通过地址偏移),但链表必须从头遍历 O(n) 才能找到中间节点,得不偿失。

### 算法模式卡片 📐
- **模式名称**:二分查找(Binary Search)
- **适用条件**:
  1. 数据结构支持随机访问(如数组)
  2. 数据有序(升序或降序)
  3. 或答案空间具有单调性(如二分答案)
- **识别关键词**:
  - "有序数组"
  - "O(log n) 时间复杂度"
  - "查找目标值"
  - "旋转排序数组"(变体)
- **核心思想**:利用有序性,每次排除一半搜索空间
- **模板代码**:
```python
def binary_search(nums: List[int], target: int) -> int:
    """标准二分查找模板"""
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
```

### 易错点 ⚠️
1. **循环条件写成 `left < right`**
   - **错在哪**:当 left == right 时还有一个元素未检查,会漏掉单元素情况
   - **正确做法**:用 `left <= right`,确保区间闭合

2. **更新边界时忘记 ±1**
   - **错在哪**:写成 `left = mid` 或 `right = mid` 会导致死循环
   - **正确做法**:`left = mid + 1` 和 `right = mid - 1`,排除已检查的 mid
   - **记忆技巧**:mid 已经比较过了,不可能是答案,所以要跳过它

3. **mid 计算整数溢出(其他语言)**
   - **错在哪**:在 C++/Java 中,`(left + right)` 可能超过 int 范围
   - **正确做法**:`mid = left + (right - left) / 2`
   - **Python 注意**:Python 整数无限大,但养成好习惯很重要

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景 1:数据库索引**
  - 数据库的 B+ 树索引本质就是多路二分查找
  - 在百万级数据中,只需 3-4 次磁盘 I/O 就能定位记录

- **场景 2:Git 的 `git bisect` 命令**
  - 用二分法快速定位引入 bug 的 commit
  - 在 1000 个 commit 中,只需测试 10 次就能找到问题版本

- **场景 3:游戏中的 AI 决策**
  - 在排序好的技能列表中快速查找最优技能
  - 例如根据敌人血量二分查找合适的攻击技能

- **场景 4:大数据处理**
  - Hadoop/Spark 在排序分区中使用二分查找数据块
  - 加速 MapReduce 任务中的 shuffle 阶段

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 35. 搜索插入位置 | Easy | 左边界二分 | 返回第一个 >= target 的位置 |
| LeetCode 34. 在排序数组中查找元素的首末位置 | Medium | 左右边界二分 | 分别用左边界和右边界二分 |
| LeetCode 33. 搜索旋转排序数组 | Medium | 二分变体 | 先判断哪半边有序 |
| LeetCode 69. x 的平方根 | Easy | 二分答案 | 在 [0, x] 中二分查找答案 |
| LeetCode 278. 第一个错误的版本 | Easy | 左边界二分 | 找第一个返回 true 的版本 |
| LeetCode 153. 寻找旋转排序数组中的最小值 | Medium | 二分变体 | 利用单调性判断最小值在哪边 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个有序数组和目标值,找出目标值在数组中的**开始位置和结束位置**。如果不存在,返回 [-1, -1]。要求时间复杂度 O(log n)。

示例:
- 输入:nums = [5,7,7,8,8,10], target = 8
- 输出:[3,4]

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

需要用两次二分查找:
1. 第一次找**左边界**(第一个 >= target 的位置)
2. 第二次找**右边界**(最后一个 <= target 的位置)

关键是修改二分查找的终止条件,即使找到 target 也要继续搜索边界。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def searchRange(nums: List[int], target: int) -> List[int]:
    """找目标值的首末位置"""

    def find_left(nums, target):
        """找第一个 >= target 的位置"""
        left, right = 0, len(nums) - 1
        result = -1

        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] >= target:
                if nums[mid] == target:
                    result = mid  # 记录候选位置
                right = mid - 1  # 继续往左找
            else:
                left = mid + 1

        return result

    def find_right(nums, target):
        """找最后一个 <= target 的位置"""
        left, right = 0, len(nums) - 1
        result = -1

        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] <= target:
                if nums[mid] == target:
                    result = mid  # 记录候选位置
                left = mid + 1  # 继续往右找
            else:
                right = mid - 1

        return result

    # 分别查找左右边界
    left_bound = find_left(nums, target)
    right_bound = find_right(nums, target)

    return [left_bound, right_bound]


# 测试
print(searchRange([5, 7, 7, 8, 8, 10], 8))  # 输出:[3, 4]
print(searchRange([5, 7, 7, 8, 8, 10], 6))  # 输出:[-1, -1]
```

**核心思路**:
- **左边界二分**:找到 target 后不立即返回,而是继续在左半边搜索(right = mid - 1)
- **右边界二分**:找到 target 后继续在右半边搜索(left = mid + 1)
- 两次二分都是 O(log n),总时间复杂度仍为 O(log n)

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
