> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第29课：删除链表的倒数第 N 个节点

> **模块**：链表 | **难度**：Medium ⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/remove-nth-node-from-end-of-list/
> **前置知识**：第24课(反转链表)、第26课(环形链表)
> **预计学习时间**：20分钟

---

## 🎯 题目描述

给你一个链表的头节点 `head`,删除链表的倒数第 `n` 个节点,并返回链表的头节点。

**示例：**
```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
解释：删除倒数第2个节点(值为4的节点)后,链表变为 1->2->3->5
```

```
输入：head = [1], n = 1
输出：[]
解释：删除唯一的节点后,链表为空
```

**约束条件：**
- 链表节点数范围是 `[1, 30]`
- `1 <= n <= 链表长度`
- 要求**只遍历一次链表**

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 删除唯一节点 | head=[1], n=1 | [] | 删除后链表为空 |
| 删除头节点 | head=[1,2], n=2 | [2] | 倒数第n个是头节点 |
| 删除尾节点 | head=[1,2,3], n=1 | [1,2] | 倒数第1个是尾节点 |
| 删除中间节点 | head=[1,2,3,4,5], n=2 | [1,2,3,5] | 一般情况 |
| 最长链表 | n=30, 30个节点 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你是一个火车站的调度员,需要摘掉一节车厢...
>
> 🐌 **笨办法**：先数一遍整列火车有多少节车厢(比如100节),然后从头开始走到第(100-n)节,摘掉下一节车厢。这需要走两趟:第一趟数数,第二趟定位。
>
> 🚀 **聪明办法**：派两个人,让"快跑者"先跑到第 n 节车厢,然后两人一起跑,当"快跑者"到达火车尾部时,"慢跑者"恰好在要摘掉的车厢前一节！只需要走一趟。

### 关键洞察

**倒数第 n 个 = 正数第 (length - n + 1) 个,但我们不想先遍历求长度,关键是让两个指针保持固定距离 n!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出
- **输入**：链表头节点 `head` 和整数 `n`
- **输出**：删除倒数第 n 个节点后的链表头节点
- **限制**：题目要求**只遍历一次**,不能先求链表长度再定位

### Step 2：先想笨办法（暴力法）

最直接的思路:
1. 第一遍遍历链表,统计长度 `length`
2. 计算正数位置 `pos = length - n`
3. 第二遍遍历到第 `pos-1` 个节点,删除下一个节点

- 时间复杂度：O(L) 其中 L 是链表长度(实际遍历了两遍)
- 瓶颈在哪：**两次遍历**,能不能一次搞定？

### Step 3：瓶颈分析 → 优化方向

分析暴力法的核心问题:
- 核心问题：删除倒数第 n 个节点,需要定位到倒数第 (n+1) 个节点,但不知道链表长度,所以需要遍历两次
- 优化思路：**能不能用双指针保持固定距离,一次遍历就定位到目标位置？**

### Step 4：选择武器
- 选用：**快慢双指针（间距固定）**
- 理由：让快指针先走 n 步,然后快慢指针同时移动,当快指针到达末尾时,慢指针恰好在倒数第 (n+1) 个位置,可以直接删除下一个节点

> 🔑 **模式识别提示**：当题目出现"倒数第 k 个"、"链表中点"等**相对位置**问题,优先考虑"**快慢指针(固定间距)**"

---

## 🔑 解法一：两次遍历法（朴素）

### 思路

先遍历一次统计链表长度,然后计算正数位置,再遍历到目标位置删除节点。

### 图解过程

```
示例: head = [1,2,3,4,5], n = 2

Step 1: 第一次遍历,统计长度 length = 5
  1 -> 2 -> 3 -> 4 -> 5 -> null
  遍历结束: length = 5

Step 2: 计算正数位置 pos = length - n = 5 - 2 = 3
  要删除第4个节点,需要定位到第3个节点

Step 3: 第二次遍历,走到第3个节点
  1 -> 2 -> 3 -> 4 -> 5
            ^
          curr (第3个节点)

Step 4: 删除 curr.next
  1 -> 2 -> 3 -----> 5
               (跳过4)

结果: [1,2,3,5]
```

**边界情况演示：删除头节点**
```
head = [1,2], n = 2

Step 1: length = 2
Step 2: pos = 2 - 2 = 0 (要删除第1个节点,即头节点)
Step 3: 使用虚拟头节点 dummy
  dummy -> 1 -> 2
  ^
  curr

Step 4: curr.next = curr.next.next
  dummy -----> 2

返回 dummy.next = 2
```

### Python代码

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    解法一：两次遍历法
    思路：先统计长度,再定位删除
    """
    # 第一次遍历：统计链表长度
    length = 0
    curr = head
    while curr:
        length += 1
        curr = curr.next

    # 创建虚拟头节点,处理删除头节点的情况
    dummy = ListNode(0, head)
    curr = dummy

    # 第二次遍历：走到倒数第(n+1)个节点
    pos = length - n  # 正数第几个节点的前一个
    for _ in range(pos):
        curr = curr.next

    # 删除 curr.next 节点
    curr.next = curr.next.next

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
head1 = create_linked_list([1, 2, 3, 4, 5])
print(linked_list_to_list(removeNthFromEnd(head1, 2)))  # 期望输出：[1, 2, 3, 5]

head2 = create_linked_list([1])
print(linked_list_to_list(removeNthFromEnd(head2, 1)))  # 期望输出：[]

head3 = create_linked_list([1, 2])
print(linked_list_to_list(removeNthFromEnd(head3, 2)))  # 期望输出：[2]
```

### 复杂度分析
- **时间复杂度**：O(L) — L 是链表长度,虽然只是 O(L),但实际遍历了两次链表
  - 具体地说：如果链表有 100 个节点,需要先遍历 100 次统计长度,再遍历最多 100 次定位,总共约 200 次操作
- **空间复杂度**：O(1) — 只使用了常数个额外变量

### 优缺点
- ✅ 思路清晰,易于理解
- ✅ 处理了删除头节点的边界情况(虚拟头节点)
- ❌ **需要两次遍历,不符合题目"一次遍历"的优化要求**

---

## ⚡ 解法二：快慢指针（一次遍历）

### 优化思路

解法一的问题是需要两次遍历,能否一次遍历就定位？

**关键想法：让两个指针保持固定距离 n**
- 快指针先走 n 步
- 然后快慢指针同时移动
- 当快指针到达末尾(null)时,慢指针恰好在倒数第 (n+1) 个节点

> 💡 **关键想法**：倒数第 n 个 = 从尾部往前数第 n 个。如果快指针在末尾,慢指针在倒数第 (n+1) 个,那么它们的距离正好是 n。

### 图解过程

```
示例: head = [1,2,3,4,5], n = 2

Step 1: 创建虚拟头节点,快慢指针都指向 dummy
  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
  slow
  fast

Step 2: 快指针先走 n 步 (这里 n=2)
  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
  slow          fast
  (fast 走了 2 步)

Step 3: 快慢指针同时移动,直到 fast.next == null
  第1次移动:
  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
           slow     fast

  第2次移动:
  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                slow          fast

  第3次移动:
  dummy -> 1 -> 2 -> 3 -> 4 -> 5 -> null
                     slow               fast.next

  此时 fast.next == null,停止移动
  slow 指向节点3,slow.next 是要删除的节点4

Step 4: 删除 slow.next
  dummy -> 1 -> 2 -> 3 -----> 5
                      (跳过4)

返回 dummy.next = 节点1
```

**为什么 fast 先走 n 步？**
```
链表: 1 -> 2 -> 3 -> 4 -> 5 -> null
倒数: 5    4    3    2    1    (倒数第0个是null)

要删除倒数第2个(节点4),需要定位到倒数第3个(节点3)

如果 fast 先走 2 步:
初始: slow=dummy, fast=dummy
fast 走2步后: slow=dummy(倒数第6个), fast=2(倒数第4个), 距离=2

同时移动直到 fast.next=null:
此时: fast=5(倒数第1个), slow=3(倒数第3个), 距离仍=2

完美定位到倒数第(n+1)个节点!
```

### Python代码

```python
def removeNthFromEnd_v2(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    解法二：快慢指针（一次遍历）
    思路：快指针先走 n 步,然后快慢同时移动,保持距离 n
    """
    # 创建虚拟头节点,处理删除头节点的情况
    dummy = ListNode(0, head)
    slow = fast = dummy

    # 快指针先走 n 步
    for _ in range(n):
        fast = fast.next

    # 快慢指针同时移动,直到 fast 到达最后一个节点
    while fast.next:
        slow = slow.next
        fast = fast.next

    # 此时 slow 在倒数第(n+1)个节点,删除 slow.next
    slow.next = slow.next.next

    return dummy.next


# ✅ 测试
head1 = create_linked_list([1, 2, 3, 4, 5])
print(linked_list_to_list(removeNthFromEnd_v2(head1, 2)))  # 期望输出：[1, 2, 3, 5]

head2 = create_linked_list([1])
print(linked_list_to_list(removeNthFromEnd_v2(head2, 1)))  # 期望输出：[]

head3 = create_linked_list([1, 2])
print(linked_list_to_list(removeNthFromEnd_v2(head3, 2)))  # 期望输出：[2]
```

### 复杂度分析
- **时间复杂度**：O(L) — L 是链表长度,**只遍历一次链表**
  - 具体地说：如果链表有 100 个节点,快指针走 n 步,然后快慢指针一起走 (100-n) 步,总共约 100 次移动操作
- **空间复杂度**：O(1) — 只使用了两个指针

---

## 🐍 Pythonic 写法

快慢指针解法已经很简洁,但可以用更 Pythonic 的方式初始化：

```python
def removeNthFromEnd_pythonic(head: Optional[ListNode], n: int) -> Optional[ListNode]:
    """
    Pythonic 写法：使用链式初始化
    """
    dummy = slow = fast = ListNode(0, head)

    # 快指针先走 n 步(使用 next 属性链式访问)
    for _ in range(n):
        fast = fast.next

    # 同时移动
    while fast.next:
        slow, fast = slow.next, fast.next

    # 删除节点
    slow.next = slow.next.next
    return dummy.next
```

这个写法的亮点：
- 使用 `dummy = slow = fast = ListNode(0, head)` 一行初始化三个变量
- 使用 `slow, fast = slow.next, fast.next` 同时更新两个指针

> ⚠️ **面试建议**：先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一：两次遍历 | 解法二：快慢指针 |
|------|--------------|--------------|
| 时间复杂度 | O(L) | O(L) |
| 遍历次数 | 2次 | 1次 |
| 空间复杂度 | O(1) | O(1) |
| 代码难度 | 简单 | 中等 |
| 面试推荐 | ⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 理解思路时使用 | **面试首选,符合一次遍历要求** |

**面试建议**：
1. 可以先说两次遍历的思路,展示你理解了问题
2. 然后立即提出优化："能否一次遍历？用快慢指针保持固定距离"
3. 重点讲解**为什么快指针先走 n 步**,这是核心洞察

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**：请你解决一下这道题。

**你**：（审题30秒）好的,这道题要求删除链表的倒数第 n 个节点。让我先想一下...

我的第一个想法是两次遍历:先遍历一次统计长度,计算出倒数第 n 个对应的正数位置,然后再遍历一次删除节点。时间复杂度是 O(L)。

不过题目要求**只遍历一次**,所以我们可以用**快慢指针**来优化:让快指针先走 n 步,然后快慢指针同时移动,当快指针到达链表末尾时,慢指针恰好在倒数第 (n+1) 个节点,可以直接删除下一个节点。核心思路是**保持两个指针的固定距离 n**。

**面试官**：很好,请写一下代码。

**你**：（边写边说）
1. 首先创建虚拟头节点 dummy,这样可以统一处理删除头节点的情况
2. 初始化快慢指针都指向 dummy
3. 快指针先走 n 步,拉开距离
4. 然后快慢指针同时移动,直到 fast.next 为 null
5. 此时 slow.next 就是要删除的节点,执行删除操作
6. 返回 dummy.next

**面试官**：为什么用虚拟头节点？

**你**：因为如果要删除的是头节点(倒数第 length 个),直接操作 head 会比较麻烦,需要特殊判断。用虚拟头节点后,所有节点都可以统一处理,dummy.next 就是新的头节点。

**面试官**：测试一下边界情况。

**你**：
1. 测试删除唯一节点: head=[1], n=1 → 返回 [] (空链表)
2. 测试删除头节点: head=[1,2], n=2 → 返回 [2]
3. 测试删除尾节点: head=[1,2,3], n=1 → 返回 [1,2]

都正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果 n 大于链表长度怎么办？" | 题目保证 n 有效(1 ≤ n ≤ 链表长度),实际工程中可以在快指针先走时加 null 检查,如果走不到 n 步就到 null,说明 n 无效,可以抛异常或返回原链表 |
| "能不能不用虚拟头节点？" | 可以,但需要特判:如果要删除的是头节点,直接返回 head.next；否则正常处理。虚拟头节点让代码更简洁统一 |
| "这个方法对环形链表还适用吗？" | 不适用。这道题假设链表无环,如果有环,fast 指针会一直循环,永远不会到 null。需要先用 Floyd 判环算法检测是否有环 |
| "实际工程中怎么用？" | 很多编辑器的 Undo 功能用链表存储操作历史,删除倒数第 k 个操作就可以用这个技巧。或者 LRU 缓存的双向链表中,快速定位并删除某个节点 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1: 虚拟头节点（Dummy Head）— 统一处理删除头节点的情况
dummy = ListNode(0, head)  # 0 是占位值,head 是 next 指针
# ... 操作链表 ...
return dummy.next  # 返回真正的头节点

# 技巧2: 多指针同时初始化
dummy = slow = fast = ListNode(0, head)
# 等价于:
# dummy = ListNode(0, head)
# slow = dummy
# fast = dummy

# 技巧3: 同时移动多个指针
while fast.next:
    slow, fast = slow.next, fast.next  # Python 的元组解包,同时更新

# 技巧4: 安全的链表遍历条件
while fast and fast.next:  # 确保 fast 和 fast.next 都不为 null
    fast = fast.next.next
```

### 💡 底层原理（选读）

> **为什么链表删除节点不需要释放内存？**
>
> 在 C/C++ 中,删除链表节点需要手动 `free()` 或 `delete` 释放内存。但在 Python 中:
> 1. Python 有**自动垃圾回收**（Garbage Collection）机制
> 2. 当一个对象（如 ListNode）没有任何引用指向它时,GC 会自动回收内存
> 3. 执行 `slow.next = slow.next.next` 后,原来的 `slow.next` 节点失去引用,会被 GC 自动回收
>
> **虚拟头节点的本质**
>
> 虚拟头节点（Dummy Head）是一种常见的链表技巧:
> - 作用：将"删除头节点"变成"删除普通节点",统一处理逻辑
> - 代价：额外 O(1) 空间,但简化代码,减少 bug
> - 适用：所有涉及删除/插入的链表问题（如合并链表、删除重复节点等）

### 算法模式卡片 📐

- **模式名称**：快慢指针（固定间距）
- **适用条件**：链表中需要定位**相对位置**的问题（倒数第 k 个、链表中点等）
- **识别关键词**："倒数第 k 个"、"中点"、"1/3 位置"等相对位置描述
- **核心思路**：让两个指针保持固定距离,同时移动,利用**相对位置不变**的性质定位
- **模板代码**：
```python
def nth_from_end(head, n):
    """找倒数第 n 个节点（不删除）"""
    dummy = ListNode(0, head)
    slow = fast = dummy

    # 快指针先走 n 步
    for _ in range(n):
        fast = fast.next

    # 同时移动,保持距离 n
    while fast.next:
        slow = slow.next
        fast = fast.next

    return slow.next  # slow.next 是倒数第 n 个节点
```

**变体问题：**
- 找倒数第 k 个节点 → 快指针先走 k 步
- 找链表中点 → 快指针每次走 2 步,慢指针每次走 1 步
- 删除倒数第 k 个 → 定位到倒数第 (k+1) 个,删除 next

### 易错点 ⚠️

1. **快指针先走的步数搞错**
   - ❌ 错误：快指针走 `n+1` 步或 `n-1` 步
   - ✅ 正确：快指针走 **恰好 n 步**
   - 原因：要让 fast 和 slow 的距离等于 n,当 fast 到达末尾时,slow 才在倒数第 (n+1) 个

2. **循环条件写错**
   - ❌ 错误：`while fast:` 会导致 slow 指向倒数第 n 个节点（而不是倒数第 n+1 个）
   - ✅ 正确：`while fast.next:` 让 slow 停在倒数第 (n+1) 个节点,才能删除 slow.next
   - 验证：当 fast.next == null 时,fast 在最后一个节点,slow 在倒数第 (n+1) 个

3. **忘记处理删除头节点的情况**
   - ❌ 错误：直接用 head 作为起点,无法删除头节点
   - ✅ 正确：使用虚拟头节点 dummy,统一处理所有情况
   - 示例：head=[1,2], n=2 时,要删除头节点 1,需要 dummy -> 1 -> 2,删除 dummy.next

4. **fast 指针初始化位置错误**
   - ❌ 错误：`fast = head` 然后走 n-1 步（容易混淆）
   - ✅ 正确：`fast = dummy` 然后走 n 步（统一逻辑,更清晰）

---

## 🏗️ 工程实战（选读）

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1：编辑器 Undo 功能**
  在文本编辑器中,Undo 操作历史可以用链表存储。当用户执行"撤销最近的第 n 次操作"时,就需要删除倒数第 n 个节点。使用快慢指针可以高效定位并删除。

- **场景2：音视频播放列表**
  音乐/视频播放器的播放历史用链表维护。"删除最近播放的倒数第 3 首歌"就可以用这个算法,O(n) 时间定位并删除,不需要数组的 O(n) 移动操作。

- **场景3：网络请求队列**
  浏览器的请求队列（如 HTTP/2 多路复用）用链表管理。当需要取消"倒数第 k 个待发送请求"时,可以用快慢指针快速定位并移除。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识：

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 876. 链表的中间结点 | Easy | 快慢指针（不同速度） | 快指针每次走 2 步,慢指针每次走 1 步 |
| LeetCode 234. 回文链表 | Easy | 快慢指针 + 反转链表 | 先找中点,再反转后半部分比较 |
| LeetCode 61. 旋转链表 | Medium | 快慢指针 + 成环 | 找倒数第 k 个节点,断开并重连 |
| LeetCode 剑指Offer 22. 链表中倒数第k个节点 | Easy | 快慢指针 | 完全一样的思路,只是返回节点不删除 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟！

**题目**：给定一个链表,返回倒数第 k 个节点的值（不删除）。如果 k 大于链表长度,返回 -1。

<details>
<summary>💡 提示（实在想不出来再点开）</summary>

用快慢指针,快指针先走 k 步。如果快指针提前到 null,说明 k 大于链表长度。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def kthFromEnd(head: Optional[ListNode], k: int) -> int:
    """返回倒数第 k 个节点的值"""
    slow = fast = head

    # 快指针先走 k 步,如果提前到 null,说明 k 过大
    for _ in range(k):
        if not fast:
            return -1  # k 大于链表长度
        fast = fast.next

    # 同时移动,直到 fast 到达末尾
    while fast:
        slow = slow.next
        fast = fast.next

    return slow.val  # slow 是倒数第 k 个节点
```

**核心思路**：
- 与删除倒数第 n 个节点类似,只是不需要虚拟头节点（因为不删除）
- 循环条件是 `while fast:` 而不是 `while fast.next:`,让 slow 停在倒数第 k 个节点上
- 增加边界检查:如果 k 大于链表长度,快指针会提前到 null,返回 -1

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
