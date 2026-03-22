# 📖 第99课:合并K个升序链表

> **模块**:堆与优先队列 | **难度**:Hard ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/merge-k-sorted-lists/
> **前置知识**:第25课(合并两个有序链表)、第97课(前K个高频元素)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给你一个链表数组,每个链表都已经按升序排列。请将所有链表合并到一个升序链表中,返回合并后的链表。

**示例:**
```
输入:lists = [[1,4,5],[1,3,4],[2,6]]
输出:[1,1,2,3,4,4,5,6]
解释:
  链表1: 1 → 4 → 5
  链表2: 1 → 3 → 4
  链表3: 2 → 6
  合并后: 1 → 1 → 2 → 3 → 4 → 4 → 5 → 6
```

**约束条件:**
- k == lists.length
- 0 <= k <= 10⁴
- 0 <= lists[i].length <= 500
- -10⁴ <= lists[i][j] <= 10⁴
- lists[i]按升序排列

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空数组 | lists = [] | None | 空输入处理 |
| 单链表 | lists = [[1,2,3]] | 1→2→3 | 基本功能 |
| 含空链表 | lists = [[],[1]] | 1 | 空链表过滤 |
| 全空链表 | lists = [[],[],[]] | None | 全空处理 |
| 两链表 | lists = [[1,2],[3,4]] | 1→2→3→4 | 退化为合并2个 |
| 大规模 | k=10⁴,每个链表500节点 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你是一个图书管理员,桌上有K堆书,每堆都已经按书名字母排好序。你要把这K堆书合并成一堆,仍保持有序。
>
> 🐌 **笨办法1**:每次从K堆书中找出书名最小的那本,放到新堆里。重复这个过程,每次都要看K堆的顶部,太慢了!
>
> 🐌 **笨办法2**:先把第1、2堆合并,再和第3堆合并,再和第4堆合并...后面的堆要等很久!
>
> 🚀 **聪明办法**:用一个"优先级书架",每堆书的当前最小值都放在书架上,书架会自动按书名排序。每次从书架上拿走最小的那本,然后把这本书所在那堆的下一本放上书架。这就是"最小堆"思想!

### 关键洞察

**每次只需要找K个链表的"当前最小值",用最小堆能在O(log K)时间找到,远快于O(K)遍历!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:K个升序链表的数组,每个链表节点值在[-10⁴, 10⁴]
- **输出**:一个合并后的升序链表
- **限制**:K可能很大(10⁴),需要高效的合并策略

### Step 2:先想笨办法(暴力法)

最直接的想法1:每次从K个链表中找出当前最小值,加入结果链表,重复直到所有链表为空。
- 时间复杂度:假设总节点数N,每次找最小值需要O(K),总共O(N*K)
- 瓶颈在哪:每次都要遍历K个链表头找最小值

另一个想法2:逐个合并链表,先合并lists[0]和lists[1],再和lists[2]合并...
- 时间复杂度:第i次合并的链表长度约为i*avg_len,总复杂度O(K² * avg_len) = O(K*N)
- 瓶颈在哪:前面的链表被重复遍历多次

### Step 3:瓶颈分析 → 优化方向

- 核心问题:如何快速找到K个链表的当前最小值?
- 优化思路:如果能用O(log K)时间找最小值,总复杂度就能降到O(N log K)

### Step 4:选择武器

- 选用:**最小堆(优先队列)**
- 理由:
  - 堆能在O(log K)时间插入和弹出最小值
  - 每次弹出最小值后,只需插入该链表的下一个节点
  - 总复杂度O(N log K),在K很大时远优于O(N*K)

> 🔑 **模式识别提示**:当题目涉及"合并K个有序序列"或"多路归并",优先考虑"最小堆"

---

## 🔑 解法一:暴力比较(直觉法)

### 思路

每次遍历K个链表的头节点,找出值最小的节点,加入结果链表,然后移动该链表的指针。

### 图解过程

```
初始:
  lists[0]: 1 → 4 → 5
  lists[1]: 1 → 3 → 4
  lists[2]: 2 → 6

Step 1: 比较[1, 1, 2],最小值1(来自lists[0])
  结果: 1
  lists[0]: 4 → 5
  lists[1]: 1 → 3 → 4
  lists[2]: 2 → 6

Step 2: 比较[4, 1, 2],最小值1(来自lists[1])
  结果: 1 → 1
  lists[0]: 4 → 5
  lists[1]: 3 → 4
  lists[2]: 2 → 6

Step 3: 比较[4, 3, 2],最小值2(来自lists[2])
  结果: 1 → 1 → 2
  ...依此类推
```

### Python代码

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def mergeKLists(lists: list[ListNode]) -> ListNode:
    """
    解法一:暴力比较
    思路:每次从K个链表头中找最小值
    """
    if not lists:
        return None

    dummy = ListNode(0)
    current = dummy

    while True:
        # 找出当前K个链表头的最小值
        min_idx = -1
        min_val = float('inf')

        for i in range(len(lists)):
            if lists[i] and lists[i].val < min_val:
                min_val = lists[i].val
                min_idx = i

        # 所有链表都空了,退出
        if min_idx == -1:
            break

        # 将最小节点加入结果
        current.next = lists[min_idx]
        current = current.next
        lists[min_idx] = lists[min_idx].next

    return dummy.next


# ✅ 测试
def build_list(arr):
    dummy = ListNode(0)
    cur = dummy
    for val in arr:
        cur.next = ListNode(val)
        cur = cur.next
    return dummy.next

def print_list(head):
    vals = []
    while head:
        vals.append(head.val)
        head = head.next
    return vals

lists = [build_list([1,4,5]), build_list([1,3,4]), build_list([2,6])]
result = mergeKLists(lists)
print(print_list(result))  # 期望输出:[1,1,2,3,4,4,5,6]
```

### 复杂度分析

- **时间复杂度**:O(N * K) — N是总节点数,每次找最小值需要O(K)
  - 如果K=1000,N=5000,大约需要500万次操作
- **空间复杂度**:O(1) — 只用了常数额外空间(不含输出链表)

### 优缺点

- ✅ 代码简单,空间占用少
- ❌ 时间复杂度O(N*K),在K很大时非常慢
- ❌ 每次都要遍历K个链表,重复比较太多

---

## ⚡ 解法二:分治合并(优化)

### 优化思路

借鉴归并排序的思想:每次将K个链表两两配对合并,K个变成K/2个,再继续合并,直到只剩1个。这样每层合并的总节点数是N,总共log K层。

> 💡 **关键想法**:分治法将"合并K个"转化为多次"合并2个",复杂度从O(K²)降到O(K log K)!

### 图解过程

```
初始:
  [L1, L2, L3, L4]

第1轮:两两合并
  L1 ← merge(L1, L2)
  L3 ← merge(L3, L4)
  剩余: [L1, L3]

第2轮:两两合并
  L1 ← merge(L1, L3)
  剩余: [L1]

返回 L1

时间复杂度分析:
- 第1轮:合并K/2对,每对平均2N/K个节点,总O(N)
- 第2轮:合并K/4对,每对平均4N/K个节点,总O(N)
- ...共log K轮,总O(N log K)
```

### Python代码

```python
def mergeKLists_divide(lists: list[ListNode]) -> ListNode:
    """
    解法二:分治合并
    思路:两两合并,类似归并排序
    """
    if not lists:
        return None

    def merge_two(l1: ListNode, l2: ListNode) -> ListNode:
        """合并两个有序链表(第25课的经典题)"""
        dummy = ListNode(0)
        current = dummy

        while l1 and l2:
            if l1.val < l2.val:
                current.next = l1
                l1 = l1.next
            else:
                current.next = l2
                l2 = l2.next
            current = current.next

        current.next = l1 if l1 else l2
        return dummy.next

    # 分治合并
    interval = 1
    while interval < len(lists):
        for i in range(0, len(lists) - interval, interval * 2):
            lists[i] = merge_two(lists[i], lists[i + interval])
        interval *= 2

    return lists[0] if lists else None


# ✅ 测试
lists = [build_list([1,4,5]), build_list([1,3,4]), build_list([2,6])]
result = mergeKLists_divide(lists)
print(print_list(result))  # 期望输出:[1,1,2,3,4,4,5,6]
```

### 复杂度分析

- **时间复杂度**:O(N log K) — N是总节点数,分治有log K层,每层合并O(N)
  - 如果K=1000,N=5000,约5000 * 10 = 5万次操作,比暴力法快100倍
- **空间复杂度**:O(log K) — 递归调用栈(如果用迭代则是O(1))

---

## 🏆 解法三:最小堆(最优解)

### 优化思路

维护一个大小为K的最小堆,堆中存储每个链表的当前头节点。每次弹出堆顶(最小值),然后将该节点的next压入堆。这样每次操作只需O(log K),总复杂度O(N log K)。

> 💡 **关键想法**:最小堆自动维护K个候选值的最小值,比暴力遍历快K/(log K)倍!

### 图解过程

```
初始:
  L1: 1 → 4 → 5
  L2: 1 → 3 → 4
  L3: 2 → 6

堆初始化:[(1,L1), (1,L2), (2,L3)]  (按值排序)

Step 1: 弹出(1,L1),加入结果,压入(4,L1)
  堆:[(1,L2), (2,L3), (4,L1)]
  结果: 1

Step 2: 弹出(1,L2),加入结果,压入(3,L2)
  堆:[(2,L3), (3,L2), (4,L1)]
  结果: 1 → 1

Step 3: 弹出(2,L3),加入结果,压入(6,L3)
  堆:[(3,L2), (4,L1), (6,L3)]
  结果: 1 → 1 → 2

...依此类推,直到堆为空

关键:堆的大小始终 <= K,每次操作O(log K)
```

### Python代码

```python
import heapq


def mergeKLists_heap(lists: list[ListNode]) -> ListNode:
    """
    解法三:最小堆
    思路:维护K个链表头的最小堆
    """
    if not lists:
        return None

    # 初始化堆:存(节点值, 链表索引, 节点)
    # 注意:Python的heapq不能直接比较ListNode,需要加索引
    heap = []
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode(0)
    current = dummy

    while heap:
        # 弹出最小值
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        # 如果该链表还有后续节点,压入堆
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))

    return dummy.next


# ✅ 测试
lists = [build_list([1,4,5]), build_list([1,3,4]), build_list([2,6])]
result = mergeKLists_heap(lists)
print(print_list(result))  # 期望输出:[1,1,2,3,4,4,5,6]

# 边界测试
print(print_list(mergeKLists_heap([])))  # 期望输出:[]
print(print_list(mergeKLists_heap([None, build_list([1])])))  # 期望输出:[1]
```

### 复杂度分析

- **时间复杂度**:O(N log K) — N个节点,每次堆操作O(log K)
  - 与分治法相同的复杂度,但堆方法常数更小,实际更快
- **空间复杂度**:O(K) — 堆中最多存K个节点

---

## 🐍 Pythonic 写法

利用Python 3.10+的dataclass和heapq,可以让代码更简洁:

```python
from dataclasses import dataclass, field
import heapq


@dataclass(order=True)
class HeapNode:
    val: int
    idx: int = field(compare=False)
    node: ListNode = field(compare=False)


def mergeKLists_pythonic(lists: list[ListNode]) -> ListNode:
    heap = [HeapNode(head.val, i, head) for i, head in enumerate(lists) if head]
    heapq.heapify(heap)

    dummy = current = ListNode(0)

    while heap:
        item = heapq.heappop(heap)
        current.next = item.node
        current = current.next

        if item.node.next:
            heapq.heappush(heap, HeapNode(item.node.next.val, item.idx, item.node.next))

    return dummy.next
```

这个写法用dataclass自动实现了比较逻辑,避免了手动元组打包。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**堆应用的理解**,而非语法糖的使用。

---

## 📊 解法对比

| 维度 | 解法一:暴力比较 | 解法二:分治合并 | 🏆 解法三:最小堆(最优) |
|------|--------------|--------------|---------------------|
| 时间复杂度 | O(N*K) | O(N log K) | **O(N log K)** ← 同样最优 |
| 空间复杂度 | O(1) | O(log K) | **O(K)** ← 可接受 |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | K很小(<10) | 通用 | **K个有序流的标准解法** |

**为什么最小堆是最优解**:
- 时间复杂度O(N log K)已达理论最优(必须访问所有N个节点)
- 空间O(K)远小于O(N),在K<<N时非常高效
- 代码简洁清晰,heapq模块开箱即用
- 在K很大时,堆方法的常数因子比分治更小

**面试建议**:
1. 先用30秒口述暴力法思路(O(N*K)),表明你能想到基本解法
2. 立即优化到🏆最小堆(O(N log K)),展示对堆的掌握
3. **重点讲解堆的应用**:"K个有序流,堆维护当前K个最小值候选,每次O(log K)找最小"
4. 强调为什么这是最优:时间已达O(N log K)理论最优,空间O(K)远优于暴力法
5. 手动模拟添加3个链表[1,4,5],[1,3,4],[2,6]的堆变化过程

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你合并K个升序链表。

**你**:(审题30秒)好的,这道题要求合并K个有序链表成一个有序链表。让我先想一下...

我的第一个想法是每次从K个链表头中找最小值,时间复杂度是O(N*K),N是总节点数。这在K很大时会很慢。

我可以优化到O(N log K),用最小堆维护K个链表的当前头节点:
- 初始化时将K个链表头都压入最小堆
- 每次弹出堆顶(最小值),加入结果链表
- 将该节点的next压入堆,保持堆中始终有K个候选值
- 重复直到堆为空

**面试官**:很好,请写一下代码。

**你**:(边写边说关键步骤)
```python
# 关键点:
# 1. 堆中存(val, idx, node)三元组,因为heapq不能直接比较ListNode
# 2. idx用于区分值相同的节点,保证堆的稳定性
# 3. 每次弹出后立即压入next,保持堆大小 <= K
```

**面试官**:测试一下?

**你**:用示例[[1,4,5],[1,3,4],[2,6]]走一遍:
1. 初始堆:[(1,0,L1), (1,1,L2), (2,2,L3)]
2. 弹出(1,0,L1),压入(4,0,L1.next) → 结果:1
3. 弹出(1,1,L2),压入(3,1,L2.next) → 结果:1→1
4. 弹出(2,2,L3),压入(6,2,L3.next) → 结果:1→1→2
5. ...依此类推 → 最终:1→1→2→3→4→4→5→6

结果正确!再测边界情况,空数组[] → None,也正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么不用分治法?" | "分治法时间复杂度也是O(N log K),但在K很大时,堆方法的常数因子更小,且代码更简洁" |
| "如果链表数量K非常大(百万级)?" | "可以考虑分批处理:每次取1000个链表用堆合并,得到1000个结果链表,再递归合并这1000个" |
| "能否支持链表动态添加?" | "可以将新链表的头节点直接压入堆,堆会自动调整。时间复杂度仍是O(log K)" |
| "为什么堆中要存idx?" | "因为heapq在比较元组时,如果第一个元素相同会比较第二个。如果第二个是ListNode对象,Python无法比较,会报错。idx保证了稳定性" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:heapq处理自定义对象 — 用元组包装
import heapq
heap = []
heapq.heappush(heap, (priority, unique_id, obj))  # unique_id避免obj比较

# 技巧2:链表转数组辅助调试 — 快速验证结果
def to_list(head):
    return [node.val for node in iter(lambda: head if (head := head.next if head else None) else None, None)]

# 技巧3:批量初始化堆 — 用heapify比逐个push快
nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)  # O(n),比n次push的O(n log n)快
```

### 💡 底层原理(选读)

> **为什么堆的插入/弹出是O(log K)而不是O(K)?**
>
> 堆是完全二叉树,K个元素的堆高度是log K。
> - 插入:先放到末尾(数组最后),然后"上浮"到正确位置,最多上浮log K次
> - 弹出:取出堆顶后,用末尾元素替代,然后"下沉"到正确位置,最多下沉log K次
>
> Python的heapq用列表实现堆,利用下标关系:
> - 父节点i的左子节点:2*i+1,右子节点:2*i+2
> - 子节点i的父节点:(i-1)//2

### 算法模式卡片 📐

- **模式名称**:多路归并(K路归并)
- **适用条件**:合并K个有序序列(链表/数组/数据流)
- **识别关键词**:"合并K个"+"有序"、"多路归并"
- **模板代码**:
```python
import heapq

def merge_k_sorted(lists):
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (值, 链表索引, 元素索引)

    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result
```

### 易错点 ⚠️

1. **忘记处理空链表**:初始化堆时要跳过None链表。正确做法:`if head: heapq.heappush(heap, ...)`
2. **heapq比较ListNode报错**:直接push ListNode对象会因无法比较而报错。正确做法:用元组`(val, idx, node)`
3. **堆中重复压入节点**:弹出节点后忘记检查next是否为空,会导致None入堆。正确做法:`if node.next: heapq.heappush(...)`
4. **索引越界**:在分治法中,`lists[i + interval]`可能越界。正确做法:`for i in range(0, len(lists) - interval, interval * 2)`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:分布式系统中合并多个数据分片的排序结果(如Hadoop MapReduce的Reduce阶段)
- **场景2**:日志系统中合并多个服务器的时间戳有序日志文件
- **场景3**:数据库中合并多个索引扫描的结果集(Multi-Index Merge)
- **场景4**:搜索引擎中合并多个倒排索引的查询结果(按相关性排序)

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 21. 合并两个有序链表 | Easy | 链表归并 | 本题的基础版,递归或迭代 |
| LeetCode 148. 排序链表 | Medium | 链表归并排序 | 分治+合并两个链表 |
| LeetCode 378. 有序矩阵中第K小的元素 | Medium | 堆/二分 | 二维有序,用堆合并K行 |
| LeetCode 373. 查找和最小的K对数字 | Medium | 堆 | 类似K路归并,但合并的是数对 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如果K个链表的总节点数N已知,且K很大(>1000),如何进一步优化空间?

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

可以不一次性把K个链表头都放入堆,而是先放前100个,弹出一个就从后续链表中补充一个。这样堆大小始终保持在100,空间从O(K)降到O(100)=O(1)。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
import heapq


def mergeKLists_space_optimized(lists: list[ListNode], max_heap_size: int = 100) -> ListNode:
    """
    空间优化版:堆大小限制在max_heap_size
    思路:先放前max_heap_size个链表,后续按需补充
    """
    if not lists:
        return None

    heap = []
    next_list_idx = 0

    # 初始化:先放前max_heap_size个非空链表
    for i in range(min(max_heap_size, len(lists))):
        if lists[i]:
            heapq.heappush(heap, (lists[i].val, i, lists[i]))
            next_list_idx = i + 1

    dummy = current = ListNode(0)

    while heap:
        val, idx, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        # 如果该链表还有后续节点,压入堆
        if node.next:
            heapq.heappush(heap, (node.next.val, idx, node.next))
        # 如果该链表已空,且还有未处理的链表,补充一个新链表
        elif next_list_idx < len(lists):
            while next_list_idx < len(lists) and not lists[next_list_idx]:
                next_list_idx += 1
            if next_list_idx < len(lists):
                head = lists[next_list_idx]
                heapq.heappush(heap, (head.val, next_list_idx, head))
                next_list_idx += 1

    return dummy.next
```

核心思路:堆大小从O(K)降到O(min(K, max_heap_size)),在K>>max_heap_size时显著节省空间。时间复杂度仍是O(N log min(K, max_heap_size))。

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
