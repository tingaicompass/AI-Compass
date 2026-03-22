# 📖 第31课：合并K个升序链表

> **模块**：链表 | **难度**：Hard ⭐⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/merge-k-sorted-lists/
> **前置知识**：第25课(合并两个有序链表)、第30课(归并排序)
> **预计学习时间**：30分钟

---

## 🎯 题目描述

给你一个链表数组,每个链表都已经按**升序**排列。请你将所有链表合并到一个升序链表中,返回合并后的链表。

**示例：**
```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：合并后的链表为 1->1->2->3->4->4->5->6
```

```
输入：lists = []
输出：[]
```

```
输入：lists = [[]]
输出：[]
```

**约束条件：**
- `k == lists.length` (k 是链表数量)
- `0 <= k <= 10^4`
- `0 <= lists[i].length <= 500`
- `-10^4 <= lists[i][j] <= 10^4`
- 链表总节点数不超过 10^4

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空数组 | lists=[] | null | 边界处理 |
| 包含空链表 | lists=[[],[1]] | [1] | 过滤空链表 |
| 单个链表 | lists=[[1,2,3]] | [1,2,3] | 递归终止条件 |
| 两个链表 | lists=[[1,3],[2,4]] | [1,2,3,4] | 基础合并 |
| 所有值相同 | lists=[[1,1],[1,1]] | [1,1,1,1] | 稳定性 |
| 最大规模 | k=10^4, 总节点10^4 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你要合并 K 个已排序的扑克牌堆...
>
> 🐌 **笨办法 1**：把所有牌混在一起,重新排序。简单粗暴,但浪费了"每堆已经排序"的信息。
>
> 🐌 **笨办法 2**：先合并第1和第2堆,得到新堆;再和第3堆合并,再和第4堆...依次合并。问题是第1堆的牌会被反复比较 K 次,效率低。
>
> 🚀 **聪明办法 1（最小堆）**：每次从 K 堆的**堆顶**（最小牌）中选一张最小的,放入结果。用一个**优先队列**（最小堆）维护 K 堆的当前最小值,每次取堆顶,效率 O(log K)。
>
> 🚀 **聪明办法 2（分治）**：把 K 堆两两配对合并,第一轮 K 堆变 K/2 堆,第二轮变 K/4 堆...就像归并排序的分治思想,最多 log K 轮,每轮处理所有牌,总效率 O(N log K)。

### 关键洞察

**合并 K 个有序链表 = 多路归并问题。关键是如何高效地"每次选出 K 个当前最小值"——用最小堆！或者用分治法减少比较次数。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出
- **输入**：K 个升序链表的数组 `lists`
- **输出**：一个合并后的升序链表
- **限制**：链表总节点数最多 10^4,需要高效算法

### Step 2：先想笨办法（暴力法）

最直接的思路:
1. 遍历所有链表,把所有节点值收集到数组中 — O(N)
2. 对数组排序 — O(N log N)
3. 根据排序后的数组重建链表 — O(N)

（N 是所有链表的总节点数）

- 时间复杂度：O(N log N) ✅ 可以接受
- 空间复杂度：O(N) — 需要数组存储所有值
- 瓶颈在哪：**没有利用"每个链表已经有序"的信息,浪费了宝贵的性质**

### Step 3：瓶颈分析 → 优化方向

分析暴力法的核心问题:
- 核心问题：重新排序浪费了已有的有序性,能否在合并过程中保持有序？
- 回顾第25课：合并 2 个有序链表可以 O(N) 时间 O(1) 空间
- 扩展思路 1：**逐一合并** K 个链表（依次合并第1和第2,再和第3...）
  - 时间复杂度：O(k*N) — 第1个链表要参与 k-1 次合并,被比较 k 次
  - 可以优化吗？

- 扩展思路 2：**每次从 K 个链表头中选最小的** → 用最小堆维护 → O(N log k)
- 扩展思路 3：**分治合并**（类似归并排序）→ 两两合并,log k 轮 → O(N log k)

### Step 4：选择武器
- 选用 1：**最小堆（优先队列）**
- 理由：
  1. 每次需要从 K 个链表头中选最小值 → 最小堆的堆顶就是最小值
  2. 取出堆顶后,把该链表的下一个节点加入堆 → O(log k)
  3. 总共 N 个节点,每个节点入堆出堆一次 → O(N log k)

- 选用 2：**分治法（归并思想）**
- 理由：
  1. 类似归并排序,两两合并链表
  2. 第1轮：K 个链表 → K/2 个链表
  3. 第2轮：K/2 个链表 → K/4 个链表
  4. 最多 log K 轮,每轮处理所有 N 个节点 → O(N log k)

> 🔑 **模式识别提示**：当题目要求**合并多个有序序列**,优先考虑"**最小堆（多路归并）**"或"**分治归并**"

---

## 🔑 解法一：逐一合并（朴素优化）

### 思路

从第一个链表开始,依次合并后面的链表:result = merge(lists[0], lists[1]),然后 result = merge(result, lists[2]),以此类推。

### 图解过程

```
示例: lists = [[1,4,5],[1,3,4],[2,6]]

第1次合并: merge([1,4,5], [1,3,4])
  1 -> 4 -> 5
  1 -> 3 -> 4
  ↓
  result = [1,1,3,4,4,5]

第2次合并: merge([1,1,3,4,4,5], [2,6])
  1 -> 1 -> 3 -> 4 -> 4 -> 5
  2 -> 6
  ↓
  result = [1,1,2,3,4,4,5,6]

返回 [1,1,2,3,4,4,5,6]
```

**时间复杂度分析**:
```
设 K 个链表,平均每个链表长度 n,总节点数 N = k*n

第1次合并: n + n = 2n
第2次合并: 2n + n = 3n
第3次合并: 3n + n = 4n
...
第k-1次合并: (k-1)*n + n = k*n

总时间 = 2n + 3n + ... + k*n = n*(2+3+...+k) = n * k*(k+1)/2 ≈ O(k² * n) = O(k*N)

当 k 很大时,效率较低
```

### Python代码

```python
from typing import List, Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeKLists(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    解法一：逐一合并
    思路：依次合并两个链表
    """
    if not lists:
        return None

    # 从第一个链表开始,逐一合并
    result = lists[0]
    for i in range(1, len(lists)):
        result = merge_two_lists(result, lists[i])

    return result


def merge_two_lists(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    """
    合并两个有序链表（第25课学过）
    """
    dummy = ListNode(0)
    curr = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    curr.next = l1 if l1 else l2
    return dummy.next


# ✅ 测试辅助函数
def create_linked_list(values):
    """根据列表创建链表"""
    dummy = ListNode(0)
    curr = dummy
    for val in values:
        curr.next = ListNode(val)
        curr = curr.next
    return dummy.next


def linked_list_to_list(head):
    """将链表转为列表"""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


# ✅ 测试
lists1 = [
    create_linked_list([1, 4, 5]),
    create_linked_list([1, 3, 4]),
    create_linked_list([2, 6])
]
print(linked_list_to_list(mergeKLists(lists1)))  # 期望输出：[1, 1, 2, 3, 4, 4, 5, 6]

lists2 = []
print(linked_list_to_list(mergeKLists(lists2)))  # 期望输出：[]
```

### 复杂度分析
- **时间复杂度**：O(k*N) — k 是链表数量,N 是总节点数
  - 第 i 次合并处理 i*n 个节点,总共 k-1 次合并
- **空间复杂度**：O(1) — 只使用常数个指针

### 优缺点
- ✅ 代码简单,复用合并两个链表的函数
- ❌ **时间复杂度较高,第一个链表要参与所有合并,被反复比较**

---

## ⚡ 解法二：最小堆（优先队列）

### 优化思路

逐一合并的问题是重复比较。能否**每次直接找出 K 个链表头中的最小值**？

**用最小堆维护 K 个链表的当前最小值**:
1. 初始时,把 K 个链表的头节点放入最小堆
2. 每次取出堆顶（当前最小值）,加入结果链表
3. 如果该节点有 next,把 next 加入堆
4. 重复直到堆为空

> 💡 **关键想法**：最小堆可以 O(log k) 时间找到 k 个元素中的最小值,总共 N 个节点,每个节点入堆出堆一次,总时间 O(N log k)。

### 图解过程

```
示例: lists = [[1,4,5],[1,3,4],[2,6]]

Step 1: 初始化最小堆,放入 3 个链表的头节点
  堆: [1(链表0), 1(链表1), 2(链表2)]
  result: dummy

Step 2: 取出堆顶 1(链表0),加入结果
  堆: [1(链表1), 2(链表2), 4(链表0)]  (加入链表0的下一个节点4)
  result: dummy -> 1

Step 3: 取出堆顶 1(链表1),加入结果
  堆: [2(链表2), 3(链表1), 4(链表0)]  (加入链表1的下一个节点3)
  result: dummy -> 1 -> 1

Step 4: 取出堆顶 2(链表2),加入结果
  堆: [3(链表1), 4(链表0), 6(链表2)]  (加入链表2的下一个节点6)
  result: dummy -> 1 -> 1 -> 2

Step 5: 取出堆顶 3(链表1),加入结果
  堆: [4(链表0), 4(链表1), 6(链表2)]
  result: dummy -> 1 -> 1 -> 2 -> 3

Step 6-8: 依次取出 4, 4, 5, 6
  result: dummy -> 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> 5 -> 6

返回 dummy.next
```

### Python代码

```python
import heapq


def mergeKLists_v2(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    解法二：最小堆（优先队列）
    思路：用堆维护 K 个链表的当前最小值
    """
    # 创建最小堆
    min_heap = []

    # 初始化堆：放入所有非空链表的头节点
    # Python 的 heapq 按元组第一个元素排序,所以放 (node.val, index, node)
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(min_heap, (head.val, i, head))

    dummy = ListNode(0)
    curr = dummy

    # 每次取出堆顶（最小值）
    while min_heap:
        val, i, node = heapq.heappop(min_heap)
        curr.next = node
        curr = curr.next

        # 如果该节点有 next,加入堆
        if node.next:
            heapq.heappush(min_heap, (node.next.val, i, node.next))

    return dummy.next


# ✅ 测试
lists1 = [
    create_linked_list([1, 4, 5]),
    create_linked_list([1, 3, 4]),
    create_linked_list([2, 6])
]
print(linked_list_to_list(mergeKLists_v2(lists1)))  # 期望输出：[1, 1, 2, 3, 4, 4, 5, 6]
```

**注意**：Python 3 中 ListNode 对象不能直接比较,所以我们存储 `(node.val, index, node)` 三元组,堆按 val 排序,如果 val 相同则按 index 排序（保证稳定性）。

### 复杂度分析
- **时间复杂度**：O(N log k)
  - 初始化堆：O(k) 次 push,每次 O(log k),总共 O(k log k)
  - 主循环：N 个节点,每个节点 push 和 pop 各一次,O(N log k)
  - 总时间：O(k log k + N log k) = **O(N log k)**（k << N,所以 k log k 可忽略）
- **空间复杂度**：O(k) — 堆中最多 k 个节点

---

## 🚀 解法三：分治归并

### 优化思路

借鉴归并排序的思想:**两两合并,分治处理**。

**递归分治**:
- 如果链表数组为空,返回 null
- 如果只有 1 个链表,直接返回
- 如果有 K 个链表,分成两半:
  - 左半部分：递归合并 lists[0...k/2-1]
  - 右半部分：递归合并 lists[k/2...k-1]
  - 合并左右两个结果

> 💡 **关键想法**：分治法减少了比较次数。每个节点参与 log k 轮合并,每轮 O(1) 比较,总时间 O(N log k)。

### 图解过程

```
示例: lists = [[1,4,5],[1,3,4],[2,6]]

递归树:
                merge(lists[0..2])
               /                  \
     merge(lists[0..1])         lists[2]
      /            \                |
  lists[0]     lists[1]          [2,6]
     |            |
  [1,4,5]      [1,3,4]

第1层合并: merge([1,4,5], [1,3,4]) → [1,1,3,4,4,5]
第2层合并: merge([1,1,3,4,4,5], [2,6]) → [1,1,2,3,4,4,5,6]

返回 [1,1,2,3,4,4,5,6]
```

**时间复杂度分析**:
```
递归树深度 = log k（每次分割减半）
每层处理所有 N 个节点（合并操作）
总时间 = N * log k
```

### Python代码

```python
def mergeKLists_v3(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    解法三：分治归并
    思路：递归两两合并,类似归并排序
    """
    if not lists:
        return None

    # 递归函数：合并 lists[left...right]
    def merge_range(left: int, right: int) -> Optional[ListNode]:
        # 递归终止条件
        if left == right:
            return lists[left]
        if left > right:
            return None

        # 分治：找中点,分成两半
        mid = (left + right) // 2
        l1 = merge_range(left, mid)
        l2 = merge_range(mid + 1, right)

        # 合并两个有序链表
        return merge_two_lists(l1, l2)

    return merge_range(0, len(lists) - 1)


# ✅ 测试
lists1 = [
    create_linked_list([1, 4, 5]),
    create_linked_list([1, 3, 4]),
    create_linked_list([2, 6])
]
print(linked_list_to_list(mergeKLists_v3(lists1)))  # 期望输出：[1, 1, 2, 3, 4, 4, 5, 6]
```

### 复杂度分析
- **时间复杂度**：O(N log k)
  - 递归树深度：log k
  - 每层合并操作处理 N 个节点
  - 总时间：N * log k
- **空间复杂度**：O(log k) — 递归栈深度

---

## 🐍 Pythonic 写法

最小堆解法可以用 Python 的特性简化：

```python
def mergeKLists_pythonic(lists: List[Optional[ListNode]]) -> Optional[ListNode]:
    """
    Pythonic 写法：使用 heapq + 列表推导式
    """
    # 初始化堆（过滤空链表）
    min_heap = [(head.val, i, head) for i, head in enumerate(lists) if head]
    heapq.heapify(min_heap)  # 原地建堆 O(k)

    dummy = curr = ListNode(0)

    while min_heap:
        val, i, node = heapq.heappop(min_heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(min_heap, (node.next.val, i, node.next))

    return dummy.next
```

这个写法的亮点：
- 用列表推导式 `[... for i, head in enumerate(lists) if head]` 过滤空链表并初始化
- 用 `heapq.heapify()` 原地建堆,比逐个 push 更高效（O(k) vs O(k log k)）
- 用 `dummy = curr = ListNode(0)` 同时初始化两个指针

> ⚠️ **面试建议**：面试时推荐**解法二（最小堆）**或**解法三（分治）**。最小堆更直观,分治法更优雅。如果面试官关注空间,两者都是 O(log k) 或 O(k),相差不大。

---

## 📊 解法对比

| 维度 | 解法一：逐一合并 | 解法二：最小堆 | 解法三：分治归并 |
|------|--------------|--------------|--------------|
| 时间复杂度 | O(k*N) | **O(N log k)** | **O(N log k)** |
| 空间复杂度 | O(1) | O(k) | O(log k) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | K 很小时(<5) | **K 中等,需要高效** | **喜欢分治思想** |

**面试建议**：
1. 可以先说逐一合并的思路,展示你理解了问题
2. 立即提出优化:"逐一合并效率低,能否每次直接找 K 个中的最小值？用最小堆！"
3. 重点讲解**解法二的最小堆**,画图演示堆的动态变化
4. 如果面试官喜欢递归,可以提**解法三的分治法**,类比归并排序

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**：请你合并 K 个升序链表。

**你**：（审题30秒）好的,这道题要求合并 K 个已排序的链表,返回一个合并后的有序链表。

我的第一个想法是**逐一合并**:先合并前两个链表,再和第三个合并,以此类推。但这样第一个链表要参与 K-1 次合并,时间复杂度是 O(k*N),当 K 很大时效率较低。

更好的方法是用**最小堆**:
1. 把 K 个链表的头节点放入最小堆
2. 每次取出堆顶（当前 K 个节点中的最小值）,加入结果
3. 如果该节点有 next,把 next 加入堆
4. 重复直到堆为空

这样每个节点只入堆出堆一次,时间复杂度是 O(N log k),N 是总节点数。

**面试官**：为什么用堆？

**你**：因为我们需要**每次从 K 个候选节点中找最小值**,暴力比较需要 O(k),而最小堆的堆顶就是最小值,取出堆顶是 O(1),插入新节点是 O(log k)。

总共 N 个节点,每个节点 push 和 pop 各一次,总时间是 N * 2 * O(log k) = O(N log k)。

**面试官**：还有其他方法吗？

**你**：还可以用**分治法**,类似归并排序:
- 把 K 个链表两两配对合并,第一轮变成 K/2 个
- 第二轮继续两两合并,变成 K/4 个
- 递归进行,最多 log K 轮

每轮处理所有 N 个节点,总时间也是 O(N log k),空间是递归栈 O(log k)。

这两种方法时间复杂度相同,最小堆更直观,分治法更优雅。

**面试官**：请写一下最小堆的代码。

**你**：（边写边说关键步骤）
1. 用 Python 的 heapq 模块,创建最小堆
2. 初始化时,把所有非空链表的头节点放入堆,存储 (node.val, index, node) 三元组
3. 循环取堆顶,加入结果,如果有 next 就 push 进堆
4. 返回 dummy.next

（写完代码）

**面试官**：测试一下？

**你**：用示例 [[1,4,5],[1,3,4],[2,6]]:
1. 初始堆：[1, 1, 2]（3个头节点）
2. 取出 1(链表0),加入 4 → 堆：[1, 2, 4]
3. 取出 1(链表1),加入 3 → 堆：[2, 3, 4]
4. 依次取出 2, 3, 4, 4, 5, 6

结果：[1, 1, 2, 3, 4, 4, 5, 6]，正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果 K 非常大(10^4),堆会不会太大？" | 堆的大小是 K,即使 K=10^4 也只需要几十 KB 内存,完全可以接受。而且题目限制总节点数 ≤ 10^4,所以 K 很大时每个链表很短,堆操作次数少 |
| "能否 O(1) 空间？" | 逐一合并可以 O(1) 空间,但时间是 O(k*N)。最优的 O(N log k) 解法都需要 O(k) 或 O(log k) 空间,这是必要的权衡 |
| "如果链表不是升序,是降序呢？" | 可以先把每个链表反转（O(N)）,然后用相同算法合并。或者用最大堆代替最小堆 |
| "实际工程中有什么应用？" | 多路归并广泛应用于：1)数据库外部排序（合并多个排序文件）2)日志聚合（合并多台服务器的时间戳有序日志）3)多版本文件合并（如 Git 的 octopus merge） |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1: 使用 heapq 创建最小堆
import heapq
heap = []
heapq.heappush(heap, (priority, data))  # 按 priority 排序
min_val = heapq.heappop(heap)  # 取出最小值

# 技巧2: 原地建堆（比逐个 push 更快）
heap = [(priority, data) for ...]
heapq.heapify(heap)  # O(k) 原地建堆,而不是 O(k log k)

# 技巧3: 处理不可比较对象（如 ListNode）
# 存储三元组 (val, index, node),堆按 val 排序
heapq.heappush(heap, (node.val, i, node))

# 技巧4: 列表推导式过滤 + 初始化
min_heap = [(head.val, i, head) for i, head in enumerate(lists) if head]

# 技巧5: enumerate 同时获取索引和值
for i, head in enumerate(lists):
    print(f"链表{i}: {head}")
```

### 💡 底层原理（选读）

> **Python 的 heapq 模块原理**
>
> 1. **数据结构**：heapq 基于**数组实现的二叉最小堆**
>    - 父节点索引 i,左子节点 2*i+1,右子节点 2*i+2
>    - 保证父节点 ≤ 子节点（最小堆性质）
>
> 2. **关键操作**：
>    - `heappush(heap, item)`: 插入元素,向上调整（sift up）,O(log n)
>    - `heappop(heap)`: 取出堆顶,用最后一个元素替换,向下调整（sift down）,O(log n)
>    - `heapify(list)`: 原地建堆,从最后一个非叶子节点开始向下调整,O(n)
>
> 3. **为什么 heapify 是 O(n) 而不是 O(n log n)？**
>    - 叶子节点不需要调整（占一半）,倒数第二层最多下移1层,倒数第三层最多下移2层...
>    - 总代价：n/4 * 1 + n/8 * 2 + n/16 * 3 + ... = O(n)（数学级数求和）
>
> **多路归并的应用场景**
>
> 1. **外部排序**（External Sort）
>    - 问题：待排序数据大于内存（如 100GB 数据,8GB 内存）
>    - 方案：分块排序写入磁盘,再多路归并（用最小堆）
>    - 实例：Hadoop MapReduce 的 Shuffle 阶段
>
> 2. **日志聚合系统**
>    - 问题：多台服务器产生时间戳有序的日志,需要合并成全局有序
>    - 方案：每台服务器一个日志流,用最小堆维护当前最早的日志
>    - 实例：ELK（Elasticsearch-Logstash-Kibana）的日志处理
>
> 3. **数据库索引合并**
>    - 问题：多个索引扫描结果（都是有序的）,需要合并
>    - 方案：多路归并,返回联合结果集
>    - 实例：MySQL 的 Index Merge 优化

### 算法模式卡片 📐

- **模式名称**：多路归并（K-way Merge）
- **适用条件**：合并 K 个有序序列（数组、链表、文件等）
- **识别关键词**："合并 K 个有序..."、"多个排序流"、"多路归并"
- **核心思路**：用最小堆维护 K 个序列的当前最小值,或用分治法两两合并
- **模板代码**：
```python
import heapq

def merge_k_sorted(arrays):
    """
    多路归并模板：合并 K 个有序数组
    """
    # 初始化最小堆：(值, 数组索引, 元素索引)
    min_heap = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(min_heap, (arr[0], i, 0))

    result = []

    while min_heap:
        val, arr_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)

        # 如果该数组还有下一个元素,加入堆
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, arr_idx, elem_idx + 1))

    return result
```

**变体问题**：
- 合并 K 个有序数组 → 存储 (val, array_idx, elem_idx)
- 合并 K 个有序链表 → 存储 (node.val, list_idx, node)
- 第 K 小元素（多个有序数组）→ 堆中取 K 次

### 易错点 ⚠️

1. **heapq 存储 ListNode 对象导致比较错误**
   - ❌ 错误：`heapq.heappush(heap, node)` — Python 3 中 ListNode 不可比较
   - ✅ 正确：`heapq.heappush(heap, (node.val, i, node))` — 存储三元组,按 val 排序,i 作为 tie-breaker
   - 原因：当 node.val 相同时,heapq 会比较第二个元素（i 是整数,可比较）

2. **初始化堆时忘记过滤空链表**
   - ❌ 错误：`for head in lists: heapq.heappush(heap, (head.val, ...))`
   - ✅ 正确：`for head in lists: if head: heapq.heappush(...)`
   - 原因：lists 中可能包含 None（空链表）,访问 `None.val` 会报 AttributeError

3. **分治法边界条件处理错误**
   - ❌ 错误：`if left >= right: return lists[left]`（当 left > right 时访问越界）
   - ✅ 正确：分别处理 `left == right` 和 `left > right`
   - 原因：空数组时 left=0, right=-1,需要返回 None 而不是 lists[0]

4. **heapify vs 逐个 heappush**
   - ⚠️ 注意：初始化堆时,`heapq.heapify(list)` 是 O(k),逐个 `heappush` 是 O(k log k)
   - 建议：如果一开始就有所有元素,用 heapify 更高效
   ```python
   # 方法1: O(k log k)
   for item in items:
       heapq.heappush(heap, item)

   # 方法2: O(k) 更快
   heap = list(items)
   heapq.heapify(heap)
   ```

---

## 🏗️ 工程实战（选读）

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1：分布式日志聚合**
  ElasticSearch 等日志系统需要合并多台服务器的时间戳有序日志流。每台服务器一个日志流,用最小堆维护当前最早的日志,实时输出全局有序的日志。

- **场景2：数据库外部排序**
  当 MySQL 执行 `ORDER BY` 时,如果数据量超过 `sort_buffer_size`,会分批排序写入临时文件,最后用多路归并（最小堆）合并所有文件,得到全局有序结果。

- **场景3：Kafka 多分区消费**
  Kafka 一个 Topic 有多个 Partition,每个 Partition 内消息有序,但全局无序。消费者需要按时间戳全局有序消费时,可以用最小堆合并多个 Partition 的消息流。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识：

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 21. 合并两个有序链表 | Easy | 多路归并的基础 | 本题的子问题,先掌握合并 2 个 |
| LeetCode 88. 合并两个有序数组 | Easy | 双指针归并 | 数组版本,类似思想 |
| LeetCode 378. 有序矩阵中第K小 | Medium | 多路归并 + 堆 | 每行看作一个有序序列,用堆合并 |
| LeetCode 373. 查找和最小的K对数字 | Medium | 多路归并 | 类似多路归并,但找前 K 对 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟！

**题目**：给定 K 个有序数组,找出其中第 K 小的元素。（扩展：多路归并求第 K 小）

<details>
<summary>💡 提示（实在想不出来再点开）</summary>

用最小堆维护 K 个数组的当前最小值,弹出 K 次堆顶,第 K 次弹出的就是第 K 小元素。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
import heapq

def kth_smallest_in_k_arrays(arrays, k):
    """
    多路归并求第 K 小元素
    """
    # 初始化最小堆
    min_heap = []
    for i, arr in enumerate(arrays):
        if arr:
            heapq.heappush(min_heap, (arr[0], i, 0))

    count = 0
    result = -1

    # 弹出 K 次堆顶
    while min_heap and count < k:
        val, arr_idx, elem_idx = heapq.heappop(min_heap)
        result = val
        count += 1

        # 如果该数组还有下一个元素,加入堆
        if elem_idx + 1 < len(arrays[arr_idx]):
            next_val = arrays[arr_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, arr_idx, elem_idx + 1))

    return result


# 测试
arrays = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
print(kth_smallest_in_k_arrays(arrays, 5))  # 输出: 5
```

**核心思路**：
- 与合并 K 个有序链表几乎一样,只是不需要构建完整结果
- 用最小堆维护 K 个数组的当前最小值
- 弹出 K 次堆顶,第 K 次弹出的值就是第 K 小元素
- 时间复杂度：O(k log K)（K 是数组数量,小写 k 是第几小）
- 空间复杂度：O(K)

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
