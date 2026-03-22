# 📖 第38课:数组中第K大元素

> **模块**:栈与队列 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/kth-largest-element-in-an-array/
> **前置知识**:堆、快速排序
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个未排序的整数数组 nums 和一个整数 k,返回数组中第 k 大的元素。

注意:是排序后的第 k 大,而不是第 k 个不同的元素。

**示例:**
```
输入:nums = [3,2,1,5,6,4], k = 2
输出:5
解释:排序后数组为 [1,2,3,4,5,6],第 2 大元素是 5
```

**约束条件:**
- 1 <= k <= nums.length <= 10^5
- -10^4 <= nums[i] <= 10^4

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1], k=1 | 1 | 单元素数组 |
| k=1 | nums=[3,2,1,5,6,4], k=1 | 6 | 最大值 |
| k=n | nums=[3,2,1,5,6,4], k=6 | 1 | 最小值 |
| 含重复元素 | nums=[3,2,3,1,2,4,5,5,6], k=4 | 4 | 重复不影响排序 |
| 负数混合 | nums=[-1,-2,3,4], k=2 | 3 | 负数处理 |
| 大规模 | n=10^5 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在参加歌唱比赛,有 100 个选手,评委要选出第 10 名的选手。
>
> 🐌 **笨办法**:让所有 100 个选手从高分到低分排成一队,然后数到第 10 个。这需要完整排序所有人,耗时长!
>
> 🚀 **聪明办法1(小顶堆)**:评委手里只记住当前前 10 名选手,每有新选手出现,如果比第 10 名(堆顶)差,直接淘汰;如果比第 10 名好,踢掉原第 10 名,把新选手放进前 10 名。这样只需要记住 10 个人,不需要关心其他 90 个选手的具体排名!
>
> 🚀 **聪明办法2(快速选择)**:类似猜数字游戏,选一个"基准选手",比他强的站左边,比他弱的站右边。如果左边恰好有 9 个人,那基准选手就是第 10 名!如果左边有 15 人,说明第 10 名在左边,只需要在左边继续找;如果左边只有 5 人,说明第 10 名在右边,只需要在右边找第 5 名(10-5-1)。每次排除一半选手!

### 关键洞察
**第 k 大元素不需要完整排序,只需要"部分有序"即可!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组 nums + 整数 k (1 <= k <= n)
- **输出**:排序后数组中第 k 大的元素(返回值本身,不是下标)
- **限制**:数组最大 10^5 规模,不能用 O(n²) 算法

### Step 2:先想笨办法(暴力法)
完整排序数组,然后返回 nums[n-k] 即可。
- 时间复杂度:O(n log n) — 排序的代价
- 瓶颈在哪:"第 k 大"只需要部分有序,完整排序浪费了!

### Step 3:瓶颈分析 → 优化方向
完整排序让前 k 个元素、中间元素、后面元素都有序,但我们只关心"谁是第 k 大",不关心其他元素的具体排名。
- 核心问题:如何避免对所有 n 个元素排序?
- 优化思路1:只维护前 k 大的元素 → **小顶堆**
- 优化思路2:利用快速排序的分区思想,每次排除一半 → **快速选择**

### Step 4:选择武器
- 选用1:**小顶堆**(Python heapq)
  - 理由:堆能在 O(log k) 时间内维护前 k 大元素,总时间 O(n log k),当 k << n 时远优于 O(n log n)
- 选用2:**快速选择**(QuickSelect)
  - 理由:平均 O(n) 时间,最优解!利用快排的分区,但只递归一侧

> 🔑 **模式识别提示**:当题目出现"第 K 大/小元素",优先考虑"堆"或"快速选择"

---

## 🔑 解法一:排序(暴力直觉法)

### 思路
最直接的想法:用 Python 内置排序 sorted(),然后返回倒数第 k 个元素。

### 图解过程

```
示例:nums = [3,2,1,5,6,4], k = 2

Step 1:完整排序
  原数组: [3, 2, 1, 5, 6, 4]
  排序后: [1, 2, 3, 4, 5, 6]
           ↑           ↑  ↑
          下标0        4  5

Step 2:返回倒数第 k 个
  第 1 大 = nums[5] = 6
  第 2 大 = nums[4] = 5  ← 答案
  公式:nums[n-k] = nums[6-2] = nums[4] = 5
```

### Python代码

```python
from typing import List


def findKthLargest_sort(nums: List[int], k: int) -> int:
    """
    解法一:完整排序
    思路:sorted()排序后返回倒数第 k 个元素
    """
    # 排序数组(升序)
    nums_sorted = sorted(nums)

    # 返回倒数第 k 个(第 k 大)
    return nums_sorted[-k]  # 或 nums_sorted[len(nums) - k]


# ✅ 测试
print(findKthLargest_sort([3, 2, 1, 5, 6, 4], 2))  # 期望输出:5
print(findKthLargest_sort([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # 期望输出:4
print(findKthLargest_sort([1], 1))  # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(n log n) — Python 的 Timsort 排序算法
  - 具体地说:如果输入规模 n=10000,大约需要 10000 * log₂(10000) ≈ 10000 * 13.3 ≈ 133000 次比较
- **空间复杂度**:O(n) — sorted() 返回新数组(如果用 nums.sort() 则是 O(log n) 递归栈空间)

### 优缺点
- ✅ 代码极简,一行搞定
- ✅ 利用 Python 高度优化的内置排序
- ❌ 做了"无用功":为了找第 k 大,却把所有元素都排了序
- ❌ 当 k 很小(如 k=1 找最大值)时,O(n log n) 仍然浪费

---

## ⚡ 解法二:小顶堆(空间优化)

### 优化思路
从解法一的痛点出发:我们只需要知道"谁是第 k 大",不需要对所有元素排序。

**关键想法**:维护一个大小为 k 的小顶堆,堆顶就是第 k 大元素!

> 💡 **关键想法**:小顶堆的堆顶是堆中最小的元素。如果堆中维护前 k 大元素,那堆顶就是"前 k 大中最小的",即第 k 大!

### 图解过程

```
示例:nums = [3,2,1,5,6,4], k = 2

初始堆(空):[]

遍历元素:
  元素 3:堆 < k,直接加入 → 堆:[3]
  元素 2:堆 < k,直接加入 → 堆:[2,3] (小顶堆,2 在堆顶)

  元素 1:堆满了(size=k=2),比堆顶 2 小,丢弃 → 堆:[2,3]
  元素 5:比堆顶 2 大,踢掉堆顶,加入 5 → 堆:[3,5]
  元素 6:比堆顶 3 大,踢掉堆顶,加入 6 → 堆:[5,6]
  元素 4:比堆顶 5 小,丢弃 → 堆:[5,6]

最终堆顶:5 ← 这就是第 2 大元素!

可视化:
  堆维护的是前 k 大: [6, 5]
  堆顶(最小的):5 = 第 2 大
```

### Python代码

```python
import heapq
from typing import List


def findKthLargest_heap(nums: List[int], k: int) -> int:
    """
    解法二:小顶堆
    思路:维护大小为 k 的小顶堆,堆顶即为第 k 大元素
    """
    # 初始化堆:前 k 个元素
    heap = nums[:k]
    heapq.heapify(heap)  # O(k) 建堆

    # 遍历剩余元素
    for num in nums[k:]:
        if num > heap[0]:  # 比堆顶大,说明是"更大的元素"
            heapq.heapreplace(heap, num)  # 踢掉堆顶,加入新元素

    # 堆顶就是第 k 大
    return heap[0]


# ✅ 测试
print(findKthLargest_heap([3, 2, 1, 5, 6, 4], 2))  # 期望输出:5
print(findKthLargest_heap([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # 期望输出:4
print(findKthLargest_heap([1], 1))  # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(n log k)
  - 建堆:O(k)
  - 遍历 n-k 个元素,每次堆操作 O(log k)
  - 总计:O(k + (n-k) log k) = O(n log k)
  - **当 k << n 时远优于 O(n log n)**!例如 k=10, n=10^5,则 10^5 * log₂(10) ≈ 10^5 * 3.3 ≈ 330000,比排序的 10^5 * 17 快 5 倍!
- **空间复杂度**:O(k) — 堆的大小

### 优缺点
- ✅ 当 k 很小时,时间明显优于排序
- ✅ 空间只需要 O(k),适合 k << n 的场景
- ❌ 当 k 接近 n 时(如 k=n/2),优势不明显
- ❌ 代码稍复杂,需要理解堆的性质

---

## 🏆 解法三:快速选择(QuickSelect,最优解)

### 优化思路
快速排序的核心是"分区"(partition):选一个基准值,把数组分成"小于基准"和"大于基准"两部分。

**关键洞察**:如果基准值恰好是第 k 大,那左边恰好有 k-1 个更大的元素!如果不是,只需要在一侧递归,**每次排除一半元素**!

> 💡 **核心思想**:类似二分查找,但用快排的分区代替比较,平均 O(n) 时间找到第 k 大!

### 图解过程

```
示例:nums = [3,2,1,5,6,4], k = 2(找第 2 大)

第 1 次分区(选基准=4):
  原数组: [3, 2, 1, 5, 6, 4]
  基准 pivot = 4
  分区后: [3, 2, 1, 4] | [5, 6]
           ← 小于等于 4  ← 大于 4

  右侧有 2 个元素 → 右侧第 1 个就是全局第 2 大
  继续在右侧 [5, 6] 找第 1 大

第 2 次分区(选基准=6):
  数组: [5, 6]
  基准 pivot = 6
  分区后: [5] | [6]

  右侧有 1 个元素 → 就是它! → 返回 6

等等,答案不是应该是 5 吗?

让我重新推导:
  第 k 大 = 排序后倒数第 k 个
  第 2 大 = 排序后倒数第 2 个 = 排序后第 n-k+1 小 = 第 5 小

更简单的方式:转化为"第 k 小"问题
  第 k 大 = 第 (n - k + 1) 小
  第 2 大 = 第 (6 - 2 + 1) = 第 5 小

或者,在降序分区中:
  分区后,基准值是第 m 大,如果 m == k,返回;
  如果 m < k,在右边找第 k-m 大;如果 m > k,在左边找第 k 大

实际示例(降序分区):
  [3,2,1,5,6,4], k=2

  选基准=4,分区(降序):
    大于4: [5,6]
    等于4: [4]
    小于4: [3,2,1]

  右侧有2个 > 4 → 答案在右侧[5,6]中,找第2大

  在[5,6]中选基准=5:
    大于5: [6]
    等于5: [5]
    小于5: []

  右侧有1个 > 5 → 答案在右侧,找第2大,但右侧只有1个
  → 答案不在右侧,当前基准是第2大 → 返回5 ✓
```

### Python代码

```python
import random
from typing import List


def findKthLargest_quickselect(nums: List[int], k: int) -> int:
    """
    解法三:快速选择(QuickSelect)
    思路:快排分区思想,平均O(n)找到第k大
    """
    def partition(left: int, right: int, pivot_index: int) -> int:
        """三路分区:把数组分成 > pivot, == pivot, < pivot"""
        pivot = nums[pivot_index]
        # 把pivot移到最右边
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]

        store_index = left
        # 把所有大于pivot的放到左边
        for i in range(left, right):
            if nums[i] > pivot:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1

        # 把pivot放回分界点
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index

    def select(left: int, right: int, k_smallest: int) -> int:
        """在nums[left:right+1]中找第k_smallest小的元素(这里k从1开始)"""
        if left == right:
            return nums[left]

        # 随机选择基准(避免最坏情况)
        pivot_index = random.randint(left, right)

        # 分区,返回基准的最终位置
        pivot_index = partition(left, right, pivot_index)

        # pivot是第几大?(从0开始)
        rank = pivot_index - left + 1

        if rank == k_smallest:
            return nums[pivot_index]
        elif rank > k_smallest:
            # 第k大在左侧
            return select(left, pivot_index - 1, k_smallest)
        else:
            # 第k大在右侧,在右侧找第(k-rank)大
            return select(pivot_index + 1, right, k_smallest - rank)

    # 调用select找第k大(从左往右第k个最大值)
    return select(0, len(nums) - 1, k)


# ✅ 测试
print(findKthLargest_quickselect([3, 2, 1, 5, 6, 4], 2))  # 期望输出:5
print(findKthLargest_quickselect([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))  # 期望输出:4
print(findKthLargest_quickselect([1], 1))  # 期望输出:1
print(findKthLargest_quickselect([2, 1], 1))  # 期望输出:2
```

### 复杂度分析
- **时间复杂度**:
  - **平均 O(n)** — 每次分区排除一半,总和 n + n/2 + n/4 + ... = 2n = O(n)
  - **最坏 O(n²)** — 每次选到最小/最大值作为基准(通过随机化基本避免)
  - 具体地说:n=10^5 时,平均只需要约 2*10^5 次操作,比排序的 10^5 * 17 快 **8 倍以上**!
- **空间复杂度**:O(log n) — 递归栈(平均情况),最坏 O(n)

### 优缺点
- ✅ **平均 O(n) 时间,理论最优**!
- ✅ 原地修改,空间只需递归栈
- ✅ 对所有 k 值都高效(不像堆只在 k 小时有优势)
- ❌ 代码较复杂,需要理解快排分区
- ❌ 最坏情况 O(n²),需要随机化避免(面试时要提到)

---

## 🐍 Pythonic 写法

利用 Python 标准库的 heapq.nlargest() 一行解决:

```python
import heapq

# 方法一:nlargest直接返回前k大的列表
def findKthLargest_pythonic(nums: List[int], k: int) -> int:
    return heapq.nlargest(k, nums)[-1]

# 方法二:nsmallest找第(n-k+1)小
def findKthLargest_pythonic2(nums: List[int], k: int) -> int:
    return heapq.nsmallest(len(nums) - k + 1, nums)[-1]

# 测试
print(findKthLargest_pythonic([3, 2, 1, 5, 6, 4], 2))  # 5
```

**解释**:
- `heapq.nlargest(k, nums)` 返回 nums 中最大的 k 个元素(降序列表)
- 取 `[-1]` 即为第 k 大
- 底层也是堆实现,时间 O(n log k)

> ⚠️ **面试建议**:先写清晰版本展示思路(解法二或解法三),再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。如果直接写 `heapq.nlargest(k, nums)[-1]`,面试官可能追问"如果不让用库函数怎么实现?"

---

## 📊 解法对比

| 维度 | 解法一:排序 | 解法二:小顶堆 | 🏆 解法三:快速选择(最优) |
|------|-----------|-------------|----------------------|
| 时间复杂度 | O(n log n) | O(n log k) | **O(n) 平均** ← 时间最优 |
| 空间复杂度 | O(n) 或 O(log n) | O(k) | **O(log n)** ← 空间优 |
| 代码难度 | 简单 | 中等 | 较难 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 快速原型/简单场景 | k << n 时优秀 | **所有场景的最优解** |

**为什么快速选择是最优解**:
- 平均 O(n) 时间已经是理论最优(至少要扫描所有元素一次)
- 空间 O(log n) 只需递归栈,比堆的 O(k) 在 k 大时更优
- 不依赖额外数据结构,纯算法思想,展示算法功底

**权衡说明**:
- **小顶堆**:当 k 非常小(如 k=1,2,3)时,实际运行可能比快速选择更快,因为常数更小,且实现简单
- **快速选择**:理论最优,但需要处理随机化避免最坏情况,面试时需说明

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你找出数组中第 k 大的元素。

**你**:(审题30秒)好的,这道题要求在未排序数组中找第 k 大元素。让我先想一下...

我的第一个想法是直接排序,然后返回倒数第 k 个,时间复杂度是 O(n log n)。

不过这样做了很多"无用功",我们其实不需要完整排序。我可以用两种方法优化:

1. **小顶堆**:维护大小为 k 的堆,堆顶就是第 k 大,时间 O(n log k)
2. **快速选择**:类似快排的分区,每次排除一半,平均 O(n) 时间,这是最优解

您希望我实现哪种?

**面试官**:快速选择听起来不错,请写一下代码。

**你**:(边写边说)快速选择的核心是分区。我选一个基准值,把数组分成"大于基准"和"小于基准"两部分。

如果基准恰好是第 k 大,直接返回。否则根据基准的排名,只在一侧递归查找...

(写完代码)这里我用了随机化选择基准,避免最坏情况退化到 O(n²)。

**面试官**:测试一下?

**你**:用示例 `[3,2,1,5,6,4], k=2` 走一遍...
(手动模拟)选基准 4,分区后大于 4 的有 `[5,6]` 两个,说明答案在这两个中,继续递归...最终返回 5。

再测一个边界情况 `[1], k=1` → 直接返回 1 ✓

**面试官**:不错!时间复杂度能再详细分析一下吗?

**你**:平均情况下,每次分区把问题规模减半,总时间是 n + n/2 + n/4 + ... = 2n,所以是 O(n)。

最坏情况下,如果每次都选到最小/最大值作为基准,会退化到 O(n²),所以我加了随机化,让最坏情况的概率极低。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 平均 O(n) 已经是理论最优,因为至少要看一遍所有元素。如果需要多次查询不同的 k,可以考虑一次 O(n) 建堆,每次 O(log n) 查询。 |
| "如果数据量非常大呢?" | 可以考虑外部排序或分布式快速选择(MapReduce)。如果 k 很小,用小顶堆内存占用小,适合流式处理。 |
| "能保证最坏情况也是 O(n) 吗?" | 可以用"中位数的中位数"算法(BFPRT)保证最坏 O(n),但常数很大,实际中随机化快速选择更常用。 |
| "为什么用小顶堆而不是大顶堆?" | 因为要维护"前 k 大",堆顶应该是"最小的",这样新元素如果比堆顶小,说明它不在前 k 大中,可以直接丢弃。如果用大顶堆,无法判断新元素是否应该进入。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:heapq 是最小堆,最大堆需要取负数
import heapq
nums = [3, 2, 1, 5, 6, 4]
# 找最大的2个
heapq.nlargest(2, nums)  # [6, 5]
# 找最小的2个
heapq.nsmallest(2, nums)  # [1, 2]

# 技巧2:heapreplace 比 heappop + heappush 更高效
heap = [1, 2, 3]
heapq.heapreplace(heap, 5)  # 一步完成:弹出堆顶,加入5

# 技巧3:原地快速选择修改原数组,如需保留可先拷贝
nums_copy = nums[:]  # 浅拷贝
```

### 💡 底层原理(选读)

> **Python 的 heapq 底层实现**
>
> Python 的 `heapq` 模块使用**数组**实现二叉堆,利用完全二叉树的性质:
> - 父节点索引 `i`,左子节点 `2*i+1`,右子节点 `2*i+2`
> - 堆顶(最小值)始终在 `heap[0]`
> - `heappush` 和 `heappop` 都是 O(log n)
>
> **为什么快速选择平均 O(n)?**
>
> 数学证明:设 T(n) 为处理 n 个元素的时间
> - 第一次分区:O(n)
> - 递归一侧(平均 n/2):T(n/2)
> - 递推式:T(n) = T(n/2) + O(n)
> - 展开:n + n/2 + n/4 + ... = 2n = O(n)
>
> 这是几何级数求和,和归并排序不同(归并是两侧都递归,所以是 n log n)

### 算法模式卡片 📐
- **模式名称**:Top-K 问题
- **适用条件**:在无序数组中找第 k 大/小元素,或前 k 大/小元素
- **识别关键词**:"第 k 大"、"前 k 个最大"、"第 k 小"
- **核心武器**:小顶堆(O(n log k)) 或 快速选择(O(n))
- **模板代码**:
```python
# 模板1:小顶堆找第k大
import heapq
def findKthLargest(nums, k):
    return heapq.nlargest(k, nums)[-1]

# 模板2:快速选择
def quickSelect(nums, k):
    pivot = random.choice(nums)
    left = [x for x in nums if x > pivot]
    mid = [x for x in nums if x == pivot]
    right = [x for x in nums if x < pivot]

    if k <= len(left):
        return quickSelect(left, k)
    if k <= len(left) + len(mid):
        return mid[0]
    return quickSelect(right, k - len(left) - len(mid))
```

### 易错点 ⚠️
1. **混淆"第 k 大"和"第 k 个位置"**
   - 错误:`return nums[k]` (这是第 k+1 大)
   - 正确:`return nums[n - k]` (排序后)或用堆/快选

2. **小顶堆 vs 大顶堆选择错误**
   - 找第 k 大 → 用大小为 k 的**小顶堆**
   - 找第 k 小 → 用大小为 k 的**大顶堆**
   - 记忆:堆顶是"边界值"(第 k 大是前 k 大中最小的)

3. **快速选择的分区边界错误**
   - 错误:忘记处理等于基准的元素,导致死循环
   - 正确:明确分成 `> pivot`, `== pivot`, `< pivot` 三部分

4. **忘记随机化基准**
   - 错误:固定选第一个元素作为基准,在有序数组上退化到 O(n²)
   - 正确:用 `random.randint()` 随机选基准

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:推荐系统**:从 100 万个商品中找出用户最可能感兴趣的前 10 个(Top-10),用小顶堆只需维护 10 个元素,内存占用极小
- **场景2:实时排行榜**:游戏中实时维护前 100 名玩家,每当有玩家分数更新,用堆 O(log 100) 更新榜单
- **场景3:海量数据第 k 大**:10 亿个数字找第 1000 大,用快速选择平均只需扫描一遍数据,不需要排序
- **场景4:数据库查询优化**:SQL 的 `ORDER BY ... LIMIT k` 底层可能用快速选择优化,避免全排序

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 215. 数组中第K大元素 | Medium | 本题原题 | 本课内容 |
| LeetCode 347. 前K个高频元素 | Medium | 堆 + 哈希表 | 先统计频率,再用堆找前k个 |
| LeetCode 973. 最接近原点的K个点 | Medium | 堆 + 距离计算 | 自定义比较函数 |
| LeetCode 703. 数据流中的第K大元素 | Easy | 设计题 + 堆 | 类成员维护小顶堆 |
| LeetCode 1985. 找出数组中第K大的整数 | Medium | 字符串比较 + 堆 | 数字是字符串形式,需自定义比较 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个数组和两个整数 k1 < k2,返回第 k1 小元素和第 k2 小元素之间所有元素的和(不包括这两个边界)。

例如:`nums = [1, 3, 12, 5, 15, 11], k1 = 3, k2 = 6`
排序后:`[1, 3, 5, 11, 12, 15]`,第3小是5,第6小是15,之间的是 `[11, 12]`,和为 23。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

先用快速选择或堆找到第 k1 小和第 k2 小的值,然后遍历数组累加在这两个值之间的元素。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def sumBetweenKthSmallest(nums, k1, k2):
    """找第k1小和第k2小之间元素的和"""
    # 方法1:排序(简单)
    nums_sorted = sorted(nums)
    val1 = nums_sorted[k1 - 1]  # 第k1小
    val2 = nums_sorted[k2 - 1]  # 第k2小

    # 累加在val1和val2之间的元素
    total = 0
    for num in nums:
        if val1 < num < val2:
            total += num
    return total

# 测试
print(sumBetweenKthSmallest([1, 3, 12, 5, 15, 11], 3, 6))  # 23
```

**优化**:如果要求 O(n) 时间,可以用两次快速选择分别找 val1 和 val2,然后一次遍历累加,总时间仍是 O(n)。

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
