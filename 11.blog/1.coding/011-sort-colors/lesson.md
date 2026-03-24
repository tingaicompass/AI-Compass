> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第11课：颜色分类

> **模块**:双指针 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/sort-colors/
> **前置知识**:[第7课:移动零](../007-move-zeroes/lesson.md)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个包含红色、白色和蓝色三种颜色对象的数组 `nums`，数字 `0`、`1`、`2` 分别代表这三种颜色。你需要**原地**对它们进行排序,使得相同颜色的元素相邻,并按照红色、白色、蓝色的顺序排列。

**示例:**
```
输入:nums = [2,0,2,1,1,0]
输出:[0,0,1,1,2,2]
```

```
输入:nums = [2,0,1]
输出:[0,1,2]
```

**约束条件:**
- `n == nums.length`
- `1 <= n <= 300`
- `nums[i]` 只能是 `0`、`1` 或 `2`
- **必须原地修改**,不使用库函数排序
- **进阶要求**:使用一次遍历完成(单次扫描算法)

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | `[1]` | `[1]` | 单元素处理 |
| 已排序 | `[0,1,2]` | `[0,1,2]` | 无需交换 |
| 逆序 | `[2,2,1,0,0]` | `[0,0,1,2,2]` | 最坏情况 |
| 全部相同 | `[1,1,1,1]` | `[1,1,1,1]` | 边界稳定性 |
| 只有两种颜色 | `[0,0,2,2]` | `[0,0,2,2]` | 缺失中间值 |
| 大规模 | `n=300` | — | 性能边界 O(n) |

---

## 💡 思路引导

### 生活化比喻
> 想象你是一个物流中心的分拣员,传送带上有红、白、蓝三种颜色的包裹混在一起。
>
> 🐌 **笨办法**:记录每种颜色的数量,然后清空传送带,按 0-0-0...1-1-1...2-2-2 的顺序重新摆放。这需要扫描两遍(统计+重建),还要用额外的计数器。
>
> 🚀 **聪明办法**:设置三个区域——"红色区"(左边)、"待处理区"(中间)、"蓝色区"(右边)。你站在待处理区扫描每个包裹:看到红色就扔到左边,看到蓝色就扔到右边,白色就留在中间。**一次扫描完成**,不需要额外空间!

### 关键洞察
**只有三种值,用三个指针分别维护三个区域的边界,一次遍历完成分区!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组 `nums`,元素只能是 0、1、2
- **输出**:原地排序,不返回新数组
- **限制**:必须原地修改(O(1)空间),进阶要求一次遍历

### Step 2:先想笨办法(两遍扫描+计数)
最直接的思路:统计每种颜色的数量,然后重新填充数组。
```python
# 第一遍:统计
count0, count1, count2 = 0, 0, 0
for num in nums:
    if num == 0: count0 += 1
    elif num == 1: count1 += 1
    else: count2 += 1

# 第二遍:重建
idx = 0
for _ in range(count0):
    nums[idx] = 0
    idx += 1
# ...后续填1和2
```
- 时间复杂度:O(n) — 两遍遍历
- 瓶颈在哪:**需要扫描两遍**,不符合进阶要求的"单次扫描"

### Step 3:瓶颈分析 → 优化方向
- 核心问题:计数法需要**先统计再重建**,无法一次完成
- 优化思路:能不能**边扫描边归位**,直接把元素放到最终位置?

**思考 1**:数组只有 3 种值,能否分区?
- 区域 1(左侧):存放所有 0
- 区域 2(中间):存放所有 1
- 区域 3(右侧):存放所有 2

**思考 2**:如何维护三个区域的边界?
- 用三个指针!
  - `left`:指向"下一个 0 应该放的位置"(区域1的右边界)
  - `right`:指向"下一个 2 应该放的位置"(区域3的左边界)
  - `i`:当前扫描的元素

### Step 4:选择武器
- 选用:**三指针分区**(荷兰国旗算法 Dutch National Flag)
- 理由:维护三个区域的边界,遇到 0 就与 left 交换,遇到 2 就与 right 交换,遇到 1 就跳过,确保一次遍历完成

> 🔑 **模式识别提示**:当题目要求"原地分成多个区域"或"只有少量不同值需要排序",优先考虑"多指针分区"

---

## 🔑 解法一:计数排序(两遍扫描)

### 思路
统计每种颜色的数量,然后按顺序重新填充数组。这是最直观的思路,但不符合进阶要求。

### 图解过程

```
输入:nums = [2, 0, 2, 1, 1, 0]

Step 1:统计数量
  扫描数组:
  count0 = 2 (两个0)
  count1 = 2 (两个1)
  count2 = 2 (两个2)

Step 2:重新填充
  填充0: [0, 0, _, _, _, _]
  填充1: [0, 0, 1, 1, _, _]
  填充2: [0, 0, 1, 1, 2, 2]

结果:[0, 0, 1, 1, 2, 2]
```

### Python代码

```python
from typing import List


def sortColors_counting(nums: List[int]) -> None:
    """
    解法一:计数排序
    思路:统计每种颜色数量,然后重新填充
    """
    # 第一遍:统计每种颜色的数量
    count0 = count1 = count2 = 0
    for num in nums:
        if num == 0:
            count0 += 1
        elif num == 1:
            count1 += 1
        else:  # num == 2
            count2 += 1

    # 第二遍:按顺序重新填充数组
    idx = 0
    # 填充所有0
    for _ in range(count0):
        nums[idx] = 0
        idx += 1
    # 填充所有1
    for _ in range(count1):
        nums[idx] = 1
        idx += 1
    # 填充所有2
    for _ in range(count2):
        nums[idx] = 2
        idx += 1


# ✅ 测试
test1 = [2, 0, 2, 1, 1, 0]
sortColors_counting(test1)
print(test1)  # 期望输出:[0, 0, 1, 1, 2, 2]

test2 = [2, 0, 1]
sortColors_counting(test2)
print(test2)  # 期望输出:[0, 1, 2]

test3 = [1]
sortColors_counting(test3)
print(test3)  # 期望输出:[1]
```

### 复杂度分析
- **时间复杂度**:O(n) — 两遍遍历,第一遍统计 O(n),第二遍填充 O(n)
  - 具体地说:如果输入规模 n=300,需要 300 + 300 = 600 次操作
- **空间复杂度**:O(1) — 只用了 3 个计数变量

### 优缺点
- ✅ 代码简单易懂,逻辑清晰
- ✅ 稳定的 O(n) 时间
- ❌ **需要扫描两遍**,不符合进阶要求的"一次遍历"
- ❌ 无法应对更复杂的分区问题

---

## ⚡ 解法二:三指针分区(荷兰国旗算法)

### 优化思路
从解法一的两遍扫描出发,思考能否**边扫描边归位**?

关键在于维护三个区域:
- `[0, left)`:已确定的 0
- `[left, i)`:已确定的 1
- `(right, n-1]`:已确定的 2
- `[i, right]`:待处理区域

> 💡 **关键想法**:当前元素是 0 就扔到左边,是 2 就扔到右边,是 1 就留在中间!

### 图解过程

```
输入:nums = [2, 0, 2, 1, 1, 0]
初始化:left = 0, i = 0, right = 5

Step 1:i=0, nums[0]=2 (是2,和right交换)
  交换 nums[0] ↔ nums[5]:
  [0, 0, 2, 1, 1, 2]
   ↑           ↑
   i           right
  right-- = 4, i 不动(因为换来的元素还没检查)

Step 2:i=0, nums[0]=0 (是0,和left交换)
  交换 nums[0] ↔ nums[0]:
  [0, 0, 2, 1, 1, 2]
   ↑
   left,i
  left++ = 1, i++ = 1

Step 3:i=1, nums[1]=0 (是0,和left交换)
  交换 nums[1] ↔ nums[1]:
  [0, 0, 2, 1, 1, 2]
      ↑
      left,i
  left++ = 2, i++ = 2

Step 4:i=2, nums[2]=2 (是2,和right交换)
  交换 nums[2] ↔ nums[4]:
  [0, 0, 1, 1, 2, 2]
         ↑     ↑
         i     right
  right-- = 3, i 不动

Step 5:i=2, nums[2]=1 (是1,跳过)
  [0, 0, 1, 1, 2, 2]
         ↑
         i
  i++ = 3

Step 6:i=3, nums[3]=1 (是1,跳过)
  [0, 0, 1, 1, 2, 2]
            ↑
            i
  i++ = 4

Step 7:i=4 > right=3, 结束循环

最终结果:[0, 0, 1, 1, 2, 2]
三个区域:
  0区:[0, 2) → [0, 0]
  1区:[2, 4) → [1, 1]
  2区:(3, 5] → [2, 2]
```

### Python代码

```python
def sortColors(nums: List[int]) -> None:
    """
    解法二:三指针分区(荷兰国旗算法)
    思路:维护三个区域,一次遍历完成
    """
    # 初始化三个指针
    left = 0  # [0, left) 区域存放所有0
    right = len(nums) - 1  # (right, n-1] 区域存放所有2
    i = 0  # 当前扫描位置

    # 当 i 还在待处理区域内
    while i <= right:
        if nums[i] == 0:
            # 遇到0:与left位置交换,left和i都前进
            nums[i], nums[left] = nums[left], nums[i]
            left += 1
            i += 1
        elif nums[i] == 2:
            # 遇到2:与right位置交换,right后退,i不动(因为换来的元素还没检查)
            nums[i], nums[right] = nums[right], nums[i]
            right -= 1
            # 注意:这里 i 不能++,因为换过来的元素还没判断
        else:  # nums[i] == 1
            # 遇到1:已经在正确位置,i前进即可
            i += 1


# ✅ 测试
test1 = [2, 0, 2, 1, 1, 0]
sortColors(test1)
print(test1)  # 期望输出:[0, 0, 1, 1, 2, 2]

test2 = [2, 0, 1]
sortColors(test2)
print(test2)  # 期望输出:[0, 1, 2]

test3 = [0]
sortColors(test3)
print(test3)  # 期望输出:[0]

test4 = [1, 1, 1]
sortColors(test4)
print(test4)  # 期望输出:[1, 1, 1]
```

### 复杂度分析
- **时间复杂度**:O(n) — **只需一次遍历**,每个元素最多被访问一次
  - 具体地说:如果输入规模 n=300,只需要最多 300 次操作
- **空间复杂度**:O(1) — 只用了 3 个指针变量,原地交换

---

## 🐍 Pythonic 写法

利用 Python 的元组解包,让交换更简洁:

```python
def sortColors_pythonic(nums: List[int]) -> None:
    """
    Pythonic写法:使用同时赋值让交换更优雅
    """
    left, right, i = 0, len(nums) - 1, 0

    while i <= right:
        if nums[i] == 0:
            nums[i], nums[left] = nums[left], nums[i]  # Python优雅交换
            left, i = left + 1, i + 1  # 同时更新
        elif nums[i] == 2:
            nums[i], nums[right] = nums[right], nums[i]
            right -= 1
        else:
            i += 1


# 更极简的写法(利用多重赋值)
def sortColors_oneliner(nums: List[int]) -> None:
    l, r, i = 0, len(nums) - 1, 0
    while i <= r:
        nums[i], nums[l if nums[i] == 0 else r if nums[i] == 2 else i], l, r, i = (
            nums[l], nums[i], l + (nums[i] == 0), r - (nums[i] == 2), i + (nums[i] != 2)
        ) if nums[i] != 1 else (nums[i], nums[i], l, r, i + 1)
```

这个单行写法用到了 Python 的条件表达式和元组解包,虽然简洁但可读性差。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。在实际工程中,也应优先保证代码可读性。

---

## 📊 解法对比

| 维度 | 解法一:计数排序 | 解法二:三指针分区 |
|------|--------------|--------------|
| 时间复杂度 | O(n) — 两遍遍历 | O(n) — **一遍遍历** |
| 空间复杂度 | O(1) | O(1) |
| 代码难度 | 简单 | 中等(需要理解三区域维护) |
| 面试推荐 | ⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 简单分类统计 | 原地分区、荷兰国旗问题 |

**面试建议**:先提出解法一展示你能快速想出可行方案,然后优化到解法二展示对指针技巧的掌握。如果面试官追问"能否一次遍历",直接给出三指针解法。

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道颜色分类问题,要求原地排序。

**你**:(审题30秒)好的,这道题要求把包含 0、1、2 的数组原地排序,使得相同数字相邻。让我先想一下...

我的第一个想法是**计数排序**:先统计每种颜色的数量,然后按 0→1→2 的顺序重新填充数组。时间复杂度是 O(n),但需要扫描两遍。

不过题目有进阶要求"一次遍历",我想到可以用**三指针分区**来优化——维护三个区域的边界,左边存 0,中间存 1,右边存 2。遇到 0 就扔到左边,遇到 2 就扔到右边,这样只需一次遍历,时间复杂度还是 O(n) 但常数更小。

**面试官**:很好,请写一下三指针的代码。

**你**:(边写边说)我用 `left` 指向下一个 0 应该放的位置,`right` 指向下一个 2 应该放的位置,`i` 是当前扫描位置。

关键在于:
- 遇到 `nums[i] == 0`,与 `nums[left]` 交换,`left++` 和 `i++`
- 遇到 `nums[i] == 2`,与 `nums[right]` 交换,`right--`,**但 `i` 不动**,因为换过来的元素还没检查
- 遇到 `nums[i] == 1`,已经在正确位置,直接 `i++`

(写出解法二的代码)

**面试官**:测试一下?

**你**:用示例 `[2,0,2,1,1,0]` 走一遍...
- i=0 时 nums[0]=2,与 nums[5]=0 交换,得到 `[0,0,2,1,1,2]`,right 变 4
- i=0 时 nums[0]=0,与 nums[0] 交换(自己),left 和 i 都 +1
- i=1 时 nums[1]=0,同上...
- 最后得到 `[0,0,1,1,2,2]`,正确!

再测一个边界情况 `[1,1,1]`:全是 1,不会进入交换分支,i 从 0 走到 2,结果不变,正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么遇到2时i不能++?" | 因为从 right 换过来的元素还没被检查过,可能是 0、1 或 2,需要在下一轮循环中处理。如果 i++,这个元素就被跳过了。 |
| "能否用递归实现?" | 可以但不推荐。递归需要 O(n) 栈空间,且不符合"原地修改"的空间要求。迭代更高效。 |
| "如果有4种颜色怎么办?" | 三指针无法直接扩展到4种,需要用**快速排序的三路划分**或**计数排序**。若颜色数量 k 很大,用快排更好(O(n log n));若 k 很小,用计数排序 O(n+k)。 |
| "这个算法稳定吗?" | **不稳定**。交换操作会打乱相同元素的相对顺序。例如 `[1a, 0, 1b]` 排序后可能变成 `[0, 1b, 1a]`。如果需要稳定性,要用归并排序。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:优雅的交换 — 使用元组解包,无需临时变量
a, b = b, a  # Python特有,底层是元组打包再解包

# 技巧2:同时更新多个变量
left, right, i = left + 1, right - 1, i + 1  # 右边先计算完再赋值给左边

# 技巧3:条件表达式(三元运算符)
next_pos = left if nums[i] == 0 else right if nums[i] == 2 else i
```

### 💡 底层原理(选读)

> **为什么 Python 交换不需要临时变量?**
>
> 在其他语言中交换需要:
> ```c
> temp = a;
> a = b;
> b = temp;
> ```
>
> Python 的 `a, b = b, a` 底层是这样的:
> 1. 右边 `b, a` 先被打包成元组 `(b的值, a的值)`,存储在栈上
> 2. 再从元组中解包赋值给 `a` 和 `b`
> 3. 本质上还是用了"临时存储"(元组),但语法更简洁
>
> **三指针为什么一次遍历就够?**
>
> 关键在于**不变量维护**(Invariant):
> - 循环的每一步都保证:
>   - `[0, left)` 区域全是 0
>   - `(right, n-1]` 区域全是 2
>   - `[left, i)` 区域全是 1
>   - `[i, right]` 是待处理区域
> - 当 `i > right` 时,待处理区域为空,排序完成
>
> 这种"区域划分 + 不变量"思想在快速排序、荷兰国旗问题中都有应用。

### 算法模式卡片 📐
- **模式名称**:三指针分区(荷兰国旗问题)
- **适用条件**:
  - 数组只有少量(通常 2-4 种)不同值
  - 要求原地排序或分区
  - 需要一次遍历完成
- **识别关键词**:"原地"、"只有 k 种值"、"分成 k 个区域"、"荷兰国旗"
- **模板代码**:
```python
def three_way_partition(nums: List[int]) -> None:
    """三指针分区模板"""
    left, right, i = 0, len(nums) - 1, 0

    while i <= right:
        if nums[i] < pivot:  # 小于基准,放左边
            nums[i], nums[left] = nums[left], nums[i]
            left += 1
            i += 1
        elif nums[i] > pivot:  # 大于基准,放右边
            nums[i], nums[right] = nums[right], nums[i]
            right -= 1
            # i 不动,因为换来的元素还没检查
        else:  # 等于基准,留中间
            i += 1
```

### 易错点 ⚠️
1. **遇到 2 时 i 忘记不动**
   - ❌ 错误:`if nums[i] == 2: swap(nums[i], nums[right]); right--; i++`
   - ⚠️ 为什么错:从 right 换来的元素可能是 0,如果 i++,这个 0 就被跳过了
   - ✅ 正确:`if nums[i] == 2: swap(nums[i], nums[right]); right--`(i 不动)

2. **循环条件写成 `i < right`**
   - ❌ 错误:`while i < right:`
   - ⚠️ 为什么错:当 `i == right` 时,这个位置的元素还没被处理,会漏掉最后一个元素
   - ✅ 正确:`while i <= right:`

3. **交换后忘记更新指针**
   - ❌ 错误:只交换不更新 `left` 或 `right`
   - ⚠️ 为什么错:边界指针不动,下次还会处理同一个元素,死循环或逻辑错误
   - ✅ 正确:交换后立即更新对应指针

4. **初始化 `right` 时写成 `len(nums)`**
   - ❌ 错误:`right = len(nums)`
   - ⚠️ 为什么错:数组索引从 0 到 n-1,right=n 会越界
   - ✅ 正确:`right = len(nums) - 1`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:快速排序的三路划分** — 快排中遇到大量重复元素时,三路划分(小于/等于/大于pivot)比二路划分更高效,避免重复元素被多次递归。Rust 标准库的排序就用了改进的三路快排。

- **场景2:日志分级存储** — 运维系统收集日志时,按级别(ERROR/WARNING/INFO)分类存储,可以用类似思想:扫描日志流,ERROR 写入高优先级队列,INFO 写入低优先级队列,WARNING 留在中间缓冲区,一次扫描完成分流。

- **场景3:垃圾回收的三色标记** — JVM 的垃圾回收器使用"三色标记算法":白色(未访问)、灰色(已访问但子节点未访问)、黑色(已访问且子节点已访问)。虽然不是直接的数组分区,但"维护三个集合边界"的思想一脉相承。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 283. 移动零 | Easy | 快慢指针 | 可以看作"两指针分区"(0 和非0) |
| LeetCode 86. 分隔链表 | Medium | 链表双指针 | 将链表分成"小于x"和"大于等于x"两部分 |
| LeetCode 324. 摆动排序 II | Medium | 三路划分+索引映射 | 先用荷兰国旗找中位数,再虚拟索引穿插排列 |
| LeetCode 215. 数组中第K大元素 | Medium | 快速选择 | 快排的三路划分思想,期望 O(n) |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个字符串数组,将所有长度小于 3 的字符串移到前面,长度等于 3 的保持中间,长度大于 3 的移到后面。要求原地修改,一次遍历。

例如:`["a", "hello", "ab", "sky", "world"]` → `["a", "ab", "sky", "hello", "world"]`

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

把字符串长度映射到 0/1/2,就变成了本题的荷兰国旗问题!

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def sortByLength(strs: List[str]) -> None:
    """按长度分三类:< 3, == 3, > 3"""
    def get_category(s: str) -> int:
        length = len(s)
        if length < 3:
            return 0
        elif length == 3:
            return 1
        else:
            return 2

    left, right, i = 0, len(strs) - 1, 0

    while i <= right:
        category = get_category(strs[i])
        if category == 0:  # 短字符串,放左边
            strs[i], strs[left] = strs[left], strs[i]
            left += 1
            i += 1
        elif category == 2:  # 长字符串,放右边
            strs[i], strs[right] = strs[right], strs[i]
            right -= 1
            # i 不动
        else:  # 中等长度,留中间
            i += 1


# 测试
test = ["a", "hello", "ab", "sky", "world"]
sortByLength(test)
print(test)  # ["a", "ab", "sky", "hello", "world"]
```

核心思路:定义一个分类函数 `get_category`,把字符串长度映射到 0/1/2,然后套用荷兰国旗的三指针模板。这展示了算法模式的**迁移能力**——理解本质后,可以应用到各种变体问题。

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
