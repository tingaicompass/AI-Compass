# 📖 第107课:LRU缓存(最后一课🎓)

> **模块**:高级技巧 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/lru-cache/
> **前置知识**:哈希表、双向链表
> **预计学习时间**:35分钟
> **特别说明**:这是本系列的**最后一课**,集大成之作!

---

## 🎊 最后一课寄语

恭喜你来到第107课,这是我们算法学习旅程的终点站!

LRU缓存不是一道简单的题目,它综合考察:
- 数据结构设计能力(哈希表+双向链表)
- 时间复杂度优化思维(如何做到 O(1) 操作)
- 工程实践意识(缓存淘汰策略在真实系统中的应用)

这道题是 **Google、Facebook、Amazon 等大厂面试的高频题**,也是检验你是否真正掌握数据结构的试金石。

准备好了吗?让我们一起攻克这个经典难题! 💪

---

## 🎯 题目描述

请你设计并实现一个满足 **LRU(Least Recently Used,最近最少使用)** 缓存约束的数据结构。

实现 `LRUCache` 类:
- `LRUCache(int capacity)` 以正整数作为容量 capacity 初始化 LRU 缓存
- `int get(int key)` 如果关键字 key 存在于缓存中,则返回对应的值;否则返回 -1
- `void put(int key, int value)` 如果关键字已经存在,则变更其数据值;如果关键字不存在,则插入该组「关键字-值」。当缓存容量达到上限时,它应该在写入新数据之前**删除最久未使用**的数据值,从而为新的数据值留出空间。

**要求**: `get` 和 `put` 操作都必须以 **O(1)** 的平均时间复杂度运行。

**示例:**
```
输入:
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

输出:
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释:
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1);  // 缓存是 {1=1}
lRUCache.put(2, 2);  // 缓存是 {1=1, 2=2}
lRUCache.get(1);     // 返回 1,缓存变为 {2=2, 1=1} (1最近使用)
lRUCache.put(3, 3);  // 缓存满,删除key=2,缓存是 {1=1, 3=3}
lRUCache.get(2);     // 返回 -1 (未找到)
lRUCache.put(4, 4);  // 删除key=1,缓存是 {3=3, 4=4}
lRUCache.get(1);     // 返回 -1 (未找到)
lRUCache.get(3);     // 返回 3
lRUCache.get(4);     // 返回 4
```

**约束条件:**
- 1 <= capacity <= 3000
- 0 <= key <= 10000
- 0 <= value <= 10⁵
- 最多调用 2 × 10⁵ 次 get 和 put

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小容量 | capacity=1, put(1,1), put(2,2), get(1) | -1 | 容量为1时立即淘汰 |
| 更新已存在 | put(1,1), put(1,2), get(1) | 2 | 更新不改变顺序(有的实现需要) |
| 空缓存get | get(1) | -1 | 缓存为空时查询 |
| 满容量淘汰 | capacity=2, put(1,1), put(2,2), put(3,3), get(1) | -1 | 淘汰最久未用 |

---

## 💡 思路引导

### 生活化比喻
> 想象你的书桌只能放 3 本书(容量限制),你是个懒人,总是把最近看的书放在最上面。
>
> 🐌 **笨办法**:
> - 每次找书时遍历整个书堆,看看书在不在 → O(n)
> - 找到后,把书抽出来再放到最上面 → O(n)
> - 加新书时,如果满了,从底部抽出最下面的那本(最久没看的) → O(n)
>
> 🚀 **聪明办法**:
> - 用一个**电子目录**(哈希表)记录每本书的位置 → 找书 O(1)
> - 书堆用**双向链表**排列,最近看的在头部,最久的在尾部 → 移动书 O(1)
> - 查看或添加书时,立即移到链表头部;满了就删除链表尾部 → O(1)

**关键洞察:**
要做到 O(1) 的 get 和 put,必须同时满足:
1. **O(1) 查找** → 哈希表
2. **O(1) 删除和移动** → 双向链表

### 关键洞察
**LRU缓存 = 哈希表(快速查找) + 双向链表(维护访问顺序)**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:capacity(容量), 多次 get(key) 和 put(key, value)
- **输出**:get 返回 value 或 -1, put 无返回值
- **限制**:get 和 put 都必须 O(1) 时间复杂度

### Step 2:先想笨办法(暴力法)
用列表存储 (key, value, timestamp) 三元组:
- `get(key)`: 遍历列表找 key → O(n)
- `put(key, value)`: 遍历找到 key 更新,或在末尾添加 → O(n)
- 淘汰时遍历找最小 timestamp → O(n)
- 瓶颈:每次操作都是 O(n),无法满足 O(1) 要求

### Step 3:瓶颈分析 → 优化方向
为了 O(1),需要解决两个问题:
1. **如何 O(1) 查找** → 哈希表 dict[key] = value
2. **如何 O(1) 找到并删除最久未用的** → 需要维护访问顺序

单纯哈希表无法维护顺序,单纯链表无法 O(1) 查找。

核心问题:**如何同时做到 O(1) 查找和 O(1) 维护顺序?**

优化思路:**组合使用哈希表和双向链表!**

### Step 4:选择武器
- 选用:**哈希表 + 双向链表**
- 理由:
  - 哈希表:dict[key] = 链表节点,O(1) 定位节点
  - 双向链表:维护访问顺序,头部=最近使用,尾部=最久未用
  - 访问时:将节点移到头部 O(1)
  - 淘汰时:删除尾部节点 O(1)

> 🔑 **模式识别提示**:当题目要求"O(1) 查找 + O(1) 插入/删除 + 维护顺序",优先考虑"哈希表+双向链表"组合

---

## 🔑 解法一:Python内置 OrderedDict(快速实现)

### 思路
Python 的 `collections.OrderedDict` 是有序字典,天然维护插入顺序,并提供 `move_to_end()` 方法。

### 图解过程

```
OrderedDict 内部实现:哈希表 + 双向链表

capacity = 2

1. put(1, 1):
   OrderedDict: {1: 1}
   链表: 1(最新)

2. put(2, 2):
   OrderedDict: {1: 1, 2: 2}
   链表: 1 ↔ 2(最新)

3. get(1):
   找到1,移到末尾(最新)
   OrderedDict: {2: 2, 1: 1}
   链表: 2 ↔ 1(最新)
   返回 1

4. put(3, 3):
   容量满,删除首个(最久未用)
   删除 2
   OrderedDict: {1: 1, 3: 3}
   链表: 1 ↔ 3(最新)
```

### Python代码

```python
from collections import OrderedDict


class LRUCache:
    """
    解法一:使用 OrderedDict
    思路:利用 Python 内置的有序字典,自动维护顺序
    """

    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 移到末尾(标记为最近使用)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # 已存在,移到末尾
            self.cache.move_to_end(key)
        self.cache[key] = value
        # 超出容量,删除最久未用(首个)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # FIFO:删除第一个


# ✅ 测试
lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
print(lru.get(1))  # 期望输出:1
lru.put(3, 3)
print(lru.get(2))  # 期望输出:-1
lru.put(4, 4)
print(lru.get(1))  # 期望输出:-1
print(lru.get(3))  # 期望输出:3
print(lru.get(4))  # 期望输出:4
```

### 复杂度分析
- **时间复杂度**:O(1) — `move_to_end()` 和 `popitem()` 都是 O(1)
- **空间复杂度**:O(capacity) — 最多存储 capacity 个键值对

### 优缺点
- ✅ 代码极简,仅10行
- ✅ 充分利用 Python 标准库
- ❌ 面试中需要解释 OrderedDict 底层原理
- ❌ 不是所有语言都有类似数据结构

---

## 🏆 解法二:手动实现(哈希表+双向链表,最优解)

### 优化思路
自己实现双向链表+哈希表的组合,展示对数据结构的深入理解。

> 💡 **关键想法**:
> - 哈希表存储 {key: 链表节点},O(1) 定位
> - 双向链表维护访问顺序,头部=最新,尾部=最旧
> - 访问/插入时移到头部,淘汰时删除尾部

### 图解过程

```
数据结构设计:
┌─────────────────────────────────────┐
│ 哈希表 dict[key] = Node            │
├─────────────────────────────────────┤
│ key=1 → Node(1,1)                  │
│ key=3 → Node(3,3)                  │
└─────────────────────────────────────┘
         ↓              ↓
双向链表: Dummy_Head ↔ Node(3,3) ↔ Node(1,1) ↔ Dummy_Tail
                      (最旧)        (最新)

操作流程示例(capacity=2):

初始: Head ↔ Tail

1. put(1,1):
   Head ↔ Node(1,1) ↔ Tail
   dict: {1: Node(1,1)}

2. put(2,2):
   Head ↔ Node(2,2) ↔ Node(1,1) ↔ Tail
   dict: {1: Node(1,1), 2: Node(2,2)}

3. get(1):
   找到 Node(1,1),移到头部
   Head ↔ Node(1,1) ↔ Node(2,2) ↔ Tail
   返回 1

4. put(3,3):
   容量满,删除尾部前节点 Node(2,2)
   插入 Node(3,3) 到头部
   Head ↔ Node(3,3) ↔ Node(1,1) ↔ Tail
   dict: {1: Node(1,1), 3: Node(3,3)}
```

### Python代码

```python
class DLinkedNode:
    """双向链表节点"""
    def __init__(self, key: int = 0, value: int = 0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCacheManual:
    """
    解法二:手动实现哈希表+双向链表(最优解)
    思路:哈希表O(1)查找,双向链表O(1)维护顺序
    """

    def __init__(self, capacity: int):
        self.cache = {}  # {key: DLinkedNode}
        self.capacity = capacity
        # 虚拟头尾节点,简化边界处理
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        # 移到头部(标记为最近使用)
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # 已存在,更新值并移到头部
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            # 新节点,添加到头部
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_to_head(node)
            # 超出容量,删除尾部
            if len(self.cache) > self.capacity:
                tail_node = self._remove_tail()
                del self.cache[tail_node.key]

    def _add_to_head(self, node: DLinkedNode) -> None:
        """在头部添加节点"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DLinkedNode) -> None:
        """删除节点"""
        node.prev.next = node.next
        node.next.prev = node.prev

    def _move_to_head(self, node: DLinkedNode) -> None:
        """移动节点到头部"""
        self._remove_node(node)
        self._add_to_head(node)

    def _remove_tail(self) -> DLinkedNode:
        """删除并返回尾部节点"""
        node = self.tail.prev
        self._remove_node(node)
        return node


# ✅ 测试
lru2 = LRUCacheManual(2)
lru2.put(1, 1)
lru2.put(2, 2)
print(lru2.get(1))  # 期望输出:1
lru2.put(3, 3)
print(lru2.get(2))  # 期望输出:-1
lru2.put(4, 4)
print(lru2.get(1))  # 期望输出:-1
print(lru2.get(3))  # 期望输出:3
print(lru2.get(4))  # 期望输出:4

# 边界测试
lru3 = LRUCacheManual(1)
lru3.put(2, 1)
print(lru3.get(2))  # 期望输出:1
lru3.put(3, 2)
print(lru3.get(2))  # 期望输出:-1
print(lru3.get(3))  # 期望输出:2
```

### 复杂度分析
- **时间复杂度**:O(1) — 所有操作(查找、插入、删除、移动)都是 O(1)
  - 哈希表查找: O(1)
  - 双向链表删除/插入: O(1) (已知节点位置)
- **空间复杂度**:O(capacity) — 哈希表+链表各存 capacity 个元素

---

## 🐍 Pythonic 写法

利用 Python 3.7+ 的 dict 保持插入顺序特性:

```python
class LRUCachePythonic:
    """利用Python3.7+ dict有序特性的简化实现"""
    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 删除后重新插入,移到末尾
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # 删除第一个(最久未用)
            first_key = next(iter(self.cache))
            self.cache.pop(first_key)


# ✅ 测试
lru4 = LRUCachePythonic(2)
lru4.put(1, 1)
lru4.put(2, 2)
print(lru4.get(1))  # 期望输出:1
lru4.put(3, 3)
print(lru4.get(2))  # 期望输出:-1
```

**说明:**
- Python 3.7+ 的 dict 保持插入顺序
- `pop(key)` 后重新插入 = 移到末尾
- `next(iter(self.cache))` 获取第一个 key

> ⚠️ **面试建议**:
> - **首选解法二(手动实现)**,展示对数据结构的深入理解
> - 如果时间充裕,可以补充解法一(OrderedDict)展示对标准库的熟悉
> - Pythonic 写法可以作为补充,但要说明依赖 Python 3.7+ 特性

---

## 📊 解法对比

| 维度 | 解法一:OrderedDict | 🏆 解法二:手动实现(最优) |
|------|-------------------|----------------------|
| 时间复杂度 | O(1) | **O(1)** |
| 空间复杂度 | O(capacity) | **O(capacity)** |
| 代码难度 | 简单(10行) | 中等(50行) |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 快速原型/刷题 | **面试展示数据结构功底** |

**为什么解法二是最优解**:
- 时间O(1)已经是理论最优(缓存必须支持常数时间访问)
- 空间O(capacity)已经是最优(至少要存储所有缓存项)
- **核心优势**:完全展示了"哈希表+双向链表"的经典组合设计
- 面试官最想看到的是你对底层数据结构的理解,而非调用库函数

**面试建议**:
1. 先用1分钟分析为什么需要 O(1):查找快、淘汰快
2. 分析单一数据结构的局限:哈希表无序、链表查找慢
3. 提出组合方案:哈希表定位+双向链表维护顺序
4. **重点讲解手动实现(解法二)**,边画图边写代码
5. 强调虚拟头尾节点的作用:简化边界处理,避免空指针
6. 手动测试边界用例(capacity=1, 连续put同一key)

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你实现一个 LRU 缓存,要求 get 和 put 都是 O(1)。

**你**:(审题30秒)好的,这道题要求实现最近最少使用缓存,核心挑战是 O(1) 的时间复杂度。
让我先分析一下:
- 要 O(1) 查找,需要哈希表
- 要 O(1) 找到并删除最久未用,需要维护访问顺序
- 哈希表无法维护顺序,所以需要配合双向链表
- 哈希表存 {key: 链表节点},链表按访问时间排序,头部最新、尾部最旧

我会用哈希表+双向链表的组合来实现。

**面试官**:很好,请写一下代码。

**你**:(边写边说)首先定义链表节点,包含 key、value、prev、next。
然后 LRUCache 类维护一个哈希表和两个虚拟头尾节点。
- get 操作:在哈希表中查找,找到后将节点移到头部,返回 value
- put 操作:如果 key 存在就更新并移到头部,否则创建新节点加到头部;超出容量就删除尾部节点

关键辅助函数:
- `_add_to_head`: 在头部添加节点
- `_remove_node`: 删除节点
- `_move_to_head`: 移动节点到头部(先删后加)
- `_remove_tail`: 删除并返回尾部节点

**面试官**:为什么要用虚拟头尾节点?

**你**:虚拟节点(dummy nodes)可以避免处理空链表的边界情况。
不用虚拟节点的话,添加第一个节点、删除最后一个节点都需要特殊判断。
有了 dummy head 和 dummy tail,所有节点都在中间,操作统一。

**面试官**:测试一下?

**你**:用示例走一遍。capacity=2,put(1,1)、put(2,2),链表是 Head↔2↔1↔Tail。
get(1)把1移到头部,变成 Head↔1↔2↔Tail,返回1。
put(3,3)时容量满,删除尾部的2,加入3到头部,变成 Head↔3↔1↔Tail。
get(2)返回-1,因为2已被淘汰。结果正确!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | "时间O(1)和空间O(capacity)都已最优。可以用 OrderedDict 简化实现,但手动实现更能展示对数据结构的理解。" |
| "如果是LFU缓存呢?" | "LFU(Least Frequently Used)淘汰访问次数最少的。需要额外维护频率计数,用哈希表存{频率: 双向链表}。时间仍可做到O(1)。" |
| "如果需要线程安全?" | "加锁保护共享状态,在 get 和 put 入口加 threading.Lock。或用读写锁优化并发性能。" |
| "实际工程中怎么用?" | "Redis、Memcached 等缓存系统都用 LRU。操作系统的页面置换算法也有 LRU。浏览器缓存、CDN 都有类似机制。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:哈希表存对象引用 — 实现 O(1) 定位
cache = {key: node_object}  # 存储对象引用,非值

# 技巧2:虚拟头尾节点(Dummy Nodes) — 简化边界处理
head = DLinkedNode()
tail = DLinkedNode()
head.next = tail
tail.prev = head
# 真实节点永远在 head 和 tail 之间,无需判空

# 技巧3:OrderedDict 的 move_to_end()
from collections import OrderedDict
od = OrderedDict()
od['a'] = 1
od.move_to_end('a')  # 移到末尾,O(1)操作

# 技巧4:Python 3.7+ dict 保持插入顺序
d = {}
d['a'] = 1
d['b'] = 2
list(d.keys())  # ['a', 'b'] 有序!
```

### 💡 底层原理(选读)

> **为什么双向链表能 O(1) 删除?**
>
> 单向链表删除节点需要知道前驱节点,需要 O(n) 遍历查找。
> 双向链表每个节点都有 prev 指针,已知节点位置时可以直接:
> ```python
> node.prev.next = node.next
> node.next.prev = node.prev
> ```
> 不需要遍历,O(1) 完成删除。
>
> **为什么哈希表能 O(1) 查找?**
>
> 哈希表通过哈希函数将 key 映射到数组下标,直接访问,理论O(1)。
> Python 的 dict 基于哈希表实现,处理了冲突(开放寻址法)。
>
> **OrderedDict 底层实现**
>
> OrderedDict 内部就是哈希表+双向链表的组合!
> 每个键值对对应一个链表节点,哈希表存 {key: 节点},链表维护插入顺序。
> 所以 `move_to_end()` 本质就是我们手动实现的 `_move_to_head()`。

### 算法模式卡片 📐
- **模式名称**:哈希表 + 双向链表组合
- **适用条件**:需要 O(1) 查找 + O(1) 插入/删除 + 维护顺序
- **识别关键词**:"LRU缓存"、"O(1)操作"、"淘汰策略"、"访问顺序"
- **核心思想**:用哈希表定位,用双向链表维护顺序,两者配合实现高效操作
- **模板代码**:
```python
class Node:
    def __init__(self, key=0, value=0):
        self.key, self.value = key, value
        self.prev = self.next = None

class DataStructure:
    def __init__(self):
        self.cache = {}  # {key: Node}
        self.head = self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _add_to_head(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
```

### 易错点 ⚠️
1. **忘记更新哈希表**:
   - 错误:删除链表节点后忘记 `del cache[key]`
   - 后果:哈希表中残留无效引用,内存泄漏
   - 正确:删除节点必须同步删除哈希表项

2. **边界处理遗漏**:
   - 错误:不用虚拟节点时,未判断链表是否为空
   - 后果:空指针异常 `NoneType has no attribute 'next'`
   - 正确:使用虚拟头尾节点,或添加空判断

3. **移动节点时顺序错误**:
   - 错误:先断开 prev/next 指针,再赋新值
   - 后果:链表断裂,节点丢失
   - 正确:先保存必要的引用,或确保操作顺序正确

4. **capacity=1 的特殊情况**:
   - 错误:未考虑容量为1时每次 put 都淘汰
   - 正确:测试 capacity=1 的边界用例

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:缓存系统 — Redis、Memcached
  - Redis 的 `maxmemory-policy` 可配置为 `allkeys-lru`
  - 内存满时自动淘汰最久未访问的 key

- **场景2**:操作系统 — 页面置换算法
  - 虚拟内存管理中的 LRU 页面置换
  - 物理内存有限时,换出最久未访问的页面到磁盘

- **场景3**:数据库 — 查询结果缓存
  - MySQL 的 Query Cache(已废弃,但思想仍在)
  - ORM 框架的查询缓存(Django、SQLAlchemy)

- **场景4**:Web开发 — HTTP缓存、CDN
  - 浏览器缓存淘汰策略
  - CDN 边缘节点的内容缓存

- **场景5**:移动开发 — 图片缓存
  - Glide、Picasso 等图片加载库
  - 内存缓存+磁盘缓存的两级 LRU

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 460. LFU缓存 | Hard | 哈希表+双向链表 | 维护频率计数,{频率: 双向链表} |
| LeetCode 432. 全O(1)数据结构 | Hard | 哈希表+双向链表 | 类似LFU,维护计数和键集合 |
| LeetCode 380. O(1)插入删除getRandom | Medium | 哈希表+数组 | 哈希表存下标,数组存值 |
| LeetCode 1472. 设计浏览器历史记录 | Medium | 双向链表/数组 | 维护访问历史,支持前进后退 |
| LeetCode 707. 设计链表 | Medium | 双向链表 | 手动实现链表的基础操作 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如何实现一个支持过期时间的 LRU 缓存?即每个 key 有 TTL(Time To Live),过期后自动失效。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

在节点中增加 `expire_time` 字段,get 时检查是否过期。可以用最小堆维护过期时间,定期清理。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
import time
import heapq


class ExpiringLRUCache:
    """支持过期时间的LRU缓存"""

    def __init__(self, capacity: int):
        self.cache = {}  # {key: (value, expire_time)}
        self.capacity = capacity
        self.access_order = OrderedDict()  # 维护访问顺序
        self.expiry_heap = []  # 最小堆:(expire_time, key)

    def get(self, key: int) -> int:
        self._clean_expired()
        if key not in self.cache:
            return -1
        value, expire_time = self.cache[key]
        if time.time() > expire_time:
            self._remove(key)
            return -1
        # 更新访问顺序
        self.access_order.move_to_end(key)
        return value

    def put(self, key: int, value: int, ttl: int = 60) -> None:
        """ttl: 过期时间(秒)"""
        self._clean_expired()
        expire_time = time.time() + ttl
        if key in self.cache:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = None
        self.cache[key] = (value, expire_time)
        heapq.heappush(self.expiry_heap, (expire_time, key))

        if len(self.cache) > self.capacity:
            oldest_key = next(iter(self.access_order))
            self._remove(oldest_key)

    def _remove(self, key: int) -> None:
        """删除指定key"""
        if key in self.cache:
            del self.cache[key]
            del self.access_order[key]

    def _clean_expired(self) -> None:
        """清理过期项"""
        current_time = time.time()
        while self.expiry_heap and self.expiry_heap[0][0] < current_time:
            _, key = heapq.heappop(self.expiry_heap)
            if key in self.cache:
                value, expire_time = self.cache[key]
                if current_time > expire_time:
                    self._remove(key)


# 测试
cache = ExpiringLRUCache(2)
cache.put(1, 100, ttl=2)  # 2秒后过期
cache.put(2, 200, ttl=5)
print(cache.get(1))  # 100
time.sleep(3)
print(cache.get(1))  # -1 (已过期)
print(cache.get(2))  # 200 (未过期)
```

**核心思路**:
- 在原有 LRU 基础上,节点增加 `expire_time` 字段
- get/put 时先调用 `_clean_expired()` 清理过期项
- 用最小堆维护过期时间,高效找到最早过期的项
- 权衡:定期清理 vs 访问时清理

</details>

---

## 🎊 毕业寄语

恭喜你完成了全部107课的学习! 🎉🎓

从第1课的"两数之和",到第107课的"LRU缓存",你已经系统学习了:
- 15个核心算法模块
- 107道精选LeetCode题目
- 数十种经典算法模式

**你现在掌握的核心能力**:
- ✅ 从暴力法到最优解的优化思维
- ✅ 时间空间复杂度的权衡分析
- ✅ 数据结构的灵活组合运用
- ✅ 面试现场的思考和表达技巧

**接下来该做什么?**
1. **巩固复习**:回顾模块速查卡片,强化记忆
2. **实战演练**:在 LeetCode 上刷同类题目,举一反三
3. **模拟面试**:找朋友或用平台进行 Mock Interview
4. **持续学习**:算法学习永无止境,保持好奇心

**最后的建议**:
> 算法面试的本质不是背题,而是展示你的**思考过程**和**优化能力**。
> 面试官想看到的是:你如何分析问题、如何权衡利弊、如何逐步优化。
>
> 记住这个公式:**清晰的思路 + 扎实的基础 + 充分的练习 = 面试成功**

祝你在算法学习的道路上越走越远,在面试中脱颖而出! 💪

—— LeetCode Python算法大师课程组

---

**特别致谢**:感谢你选择本课程,希望这107课的陪伴对你有所帮助。欢迎分享学习心得,也欢迎提出改进建议。

**联系方式**:[在此可以添加社区链接、反馈邮箱等]

**版本信息**:v1.0 | 最后更新:2026年2月

---

🎓 **你已完成全部107课!请为自己鼓掌!** 👏👏👏

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
