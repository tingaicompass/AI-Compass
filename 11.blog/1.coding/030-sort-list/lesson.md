# 📖 第30课：排序链表

> **模块**：链表 | **难度**：Medium ⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/sort-list/
> **前置知识**：第25课(合并两个有序链表)、第29课(快慢指针找中点)
> **预计学习时间**：30分钟

---

## 🎯 题目描述

给你链表的头节点 `head`,请将其按**升序**排列并返回排序后的链表。

**进阶要求**：你能否在 `O(n log n)` 时间复杂度和 `O(1)` 空间复杂度下完成？

**示例：**
```
输入：head = [4,2,1,3]
输出：[1,2,3,4]
```

```
输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
```

**约束条件：**
- 链表节点数范围是 `[0, 50000]`
- `-100000 <= Node.val <= 100000`

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空链表 | head=null | null | 边界处理 |
| 单节点 | head=[1] | [1] | 递归终止条件 |
| 已排序 | head=[1,2,3,4] | [1,2,3,4] | 最优情况 |
| 逆序 | head=[4,3,2,1] | [1,2,3,4] | 最坏情况 |
| 有重复 | head=[3,1,2,3,1] | [1,1,2,3,3] | 稳定性 |
| 大规模 | 50000个节点 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你要把一副打乱的扑克牌排序...
>
> 🐌 **笨办法**：把所有牌从桌上拿到手里(转成数组),在手里排好序(Array.sort),再一张张放回桌上(转回链表)。这样做简单,但需要额外的空手空间(额外 O(n) 空间)。
>
> 🚀 **聪明办法**：用**归并排序**的思想——把牌分成两堆,分别排序,然后合并。就像整理两叠已排序的文件,从两堆的第一张开始比较,每次取较小的那张放入结果堆。递归地分治,直到每堆只有一张牌(天然有序),然后逐层合并。
>
> 这样只需要在桌面上移动牌,不用额外空间！

### 关键洞察

**链表排序首选归并排序,因为链表不支持随机访问,无法高效使用快速排序!归并排序的关键操作——找中点、合并两个有序链表——都可以 O(1) 空间完成。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出
- **输入**：无序链表的头节点 `head`
- **输出**：排序后的链表头节点
- **限制**：进阶要求 O(n log n) 时间 + O(1) 空间

### Step 2：先想笨办法（暴力法）

最直接的思路:
1. 遍历链表,把所有值存到数组中 — O(n)
2. 对数组排序 — O(n log n)
3. 遍历数组,重新构建链表 — O(n)

- 时间复杂度：O(n log n) ✅
- 空间复杂度：O(n) ❌ (不符合进阶要求的 O(1) 空间)
- 瓶颈在哪：**需要额外 O(n) 数组空间存储所有节点值**

### Step 3：瓶颈分析 → 优化方向

分析暴力法的核心问题:
- 核心问题：数组排序需要额外空间,能否直接在链表上原地排序？
- 排序算法选择：
  - 快速排序：需要随机访问,链表不支持 → ❌
  - 堆排序：需要数组结构建堆 → ❌
  - **归并排序**：只需顺序访问 + 合并操作,完美适配链表 → ✅

- 优化思路：**使用归并排序,链表天然支持分割和合并操作,可以做到 O(1) 空间**

### Step 4：选择武器
- 选用：**归并排序（自顶向下递归）**
- 理由：
  1. 归并排序的分治过程只需要找中点（快慢指针 O(1) 空间）
  2. 合并两个有序链表可以 O(1) 空间完成（第25课学过）
  3. 时间复杂度 O(n log n),递归栈深度 O(log n) 空间（还不是 O(1)）

- **终极优化**：归并排序的**自底向上迭代版本** → O(1) 空间

> 🔑 **模式识别提示**：当题目要求**排序链表**,且需要 O(n log n) 时间,优先考虑"**归并排序**"

---

## 🔑 解法一：转数组排序（暴力）

### 思路

把链表转成数组,用 Python 内置排序,再转回链表。简单直接,但空间复杂度 O(n)。

### 图解过程

```
示例: head = [4,2,1,3]

Step 1: 遍历链表,转成数组
  4 -> 2 -> 1 -> 3 -> null
  ↓
  arr = [4, 2, 1, 3]

Step 2: 排序数组
  arr.sort() → [1, 2, 3, 4]

Step 3: 根据数组重建链表
  1 -> 2 -> 3 -> 4 -> null

返回新链表的头节点
```

### Python代码

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def sortList(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    解法一：转数组排序
    思路：链表 → 数组 → 排序 → 链表
    """
    if not head:
        return None

    # Step 1: 链表转数组
    values = []
    curr = head
    while curr:
        values.append(curr.val)
        curr = curr.next

    # Step 2: 排序数组
    values.sort()

    # Step 3: 重建链表
    dummy = ListNode(0)
    curr = dummy
    for val in values:
        curr.next = ListNode(val)
        curr = curr.next

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
head1 = create_linked_list([4, 2, 1, 3])
print(linked_list_to_list(sortList(head1)))  # 期望输出：[1, 2, 3, 4]

head2 = create_linked_list([-1, 5, 3, 4, 0])
print(linked_list_to_list(sortList(head2)))  # 期望输出：[-1, 0, 3, 4, 5]

head3 = create_linked_list([])
print(linked_list_to_list(sortList(head3)))  # 期望输出：[]
```

### 复杂度分析
- **时间复杂度**：O(n log n) — Python 的 Timsort 排序算法
  - 具体地说：如果链表有 1000 个节点,Timsort 大约需要 1000 * log₂(1000) ≈ 10000 次比较
- **空间复杂度**：O(n) — 需要数组存储所有节点值

### 优缺点
- ✅ 代码简单,易于实现
- ✅ 利用 Python 内置高效排序
- ❌ **额外 O(n) 空间,不符合进阶要求**

---

## ⚡ 解法二：归并排序（自顶向下递归）

### 优化思路

归并排序的核心思想:**分治法**
1. **分**：找到链表中点,分成两个子链表
2. **治**：递归地对两个子链表排序
3. **合**：合并两个有序链表

关键操作:
- **找中点**：快慢指针（第29课学过）
- **合并有序链表**：第25课学过的技巧

> 💡 **关键想法**：递归终止条件是链表只有 0 或 1 个节点,此时天然有序,直接返回。

### 图解过程

```
示例: head = [4,2,1,3]

第1层分治: 分成两半
  [4, 2, 1, 3]
   ↓ (找中点)
  [4, 2] 和 [1, 3]

第2层分治: 继续分
  [4, 2] → [4] 和 [2]
  [1, 3] → [1] 和 [3]

递归终止: 单节点天然有序
  [4], [2], [1], [3]

第1次合并: 两两合并
  merge([4], [2]) → [2, 4]
  merge([1], [3]) → [1, 3]

第2次合并: 最终合并
  merge([2, 4], [1, 3]) → [1, 2, 3, 4]

返回 [1, 2, 3, 4]
```

**详细演示：merge 过程**
```
合并 [2, 4] 和 [1, 3]:

初始:
  L1: 2 -> 4 -> null
  L2: 1 -> 3 -> null
  result: dummy

Step 1: 比较 2 和 1, 取 1
  dummy -> 1
  L2 前进到 3

Step 2: 比较 2 和 3, 取 2
  dummy -> 1 -> 2
  L1 前进到 4

Step 3: 比较 4 和 3, 取 3
  dummy -> 1 -> 2 -> 3
  L2 到达末尾

Step 4: L1 剩余节点直接接上
  dummy -> 1 -> 2 -> 3 -> 4

返回 dummy.next
```

### Python代码

```python
def sortList_v2(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    解法二：归并排序（自顶向下递归）
    思路：分治法——找中点、递归排序、合并
    """
    # 递归终止条件：空链表或单节点
    if not head or not head.next:
        return head

    # Step 1: 找中点,分割链表
    mid = get_middle(head)
    right_head = mid.next
    mid.next = None  # 断开链表

    # Step 2: 递归排序两个子链表
    left = sortList_v2(head)
    right = sortList_v2(right_head)

    # Step 3: 合并两个有序链表
    return merge_two_lists(left, right)


def get_middle(head: ListNode) -> ListNode:
    """
    找链表中点（快慢指针）
    当链表有偶数个节点时,返回前半部分的最后一个节点
    """
    slow = head
    fast = head.next  # fast 从 head.next 开始,确保 slow 停在前半部分

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


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

    # 接上剩余节点
    curr.next = l1 if l1 else l2

    return dummy.next


# ✅ 测试
head1 = create_linked_list([4, 2, 1, 3])
print(linked_list_to_list(sortList_v2(head1)))  # 期望输出：[1, 2, 3, 4]

head2 = create_linked_list([-1, 5, 3, 4, 0])
print(linked_list_to_list(sortList_v2(head2)))  # 期望输出：[-1, 0, 3, 4, 5]
```

### 复杂度分析
- **时间复杂度**：O(n log n)
  - 递归树有 log n 层（每次分割减半）
  - 每层的合并操作总共处理 n 个节点
  - 总时间 = n * log n
- **空间复杂度**：O(log n) — **递归栈深度**（还不是 O(1)）

---

## 🚀 解法三：归并排序（自底向上迭代）

### 优化思路

解法二使用递归,递归栈占用 O(log n) 空间。能否**不用递归,改用迭代**？

**自底向上归并排序**:
1. 第1轮：每次合并长度为 1 的子链表 → 得到长度为 2 的有序段
2. 第2轮：每次合并长度为 2 的子链表 → 得到长度为 4 的有序段
3. 第k轮：每次合并长度为 2^(k-1) 的子链表 → 得到长度为 2^k 的有序段
4. 重复直到子链表长度 ≥ 链表总长度

> 💡 **关键想法**：自底向上避免递归,用循环控制合并长度,每轮合并长度翻倍,最多 log n 轮。

### 图解过程

```
示例: head = [4, 2, 1, 3]

初始链表:
  4 -> 2 -> 1 -> 3

第1轮 (step=1): 合并长度为1的子链表
  合并(4, 2) → [2, 4]
  合并(1, 3) → [1, 3]
  结果: 2 -> 4 -> 1 -> 3

第2轮 (step=2): 合并长度为2的子链表
  合并([2,4], [1,3]) → [1, 2, 3, 4]
  结果: 1 -> 2 -> 3 -> 4

第3轮 (step=4): step >= 链表长度,结束
返回 [1, 2, 3, 4]
```

### Python代码

```python
def sortList_v3(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    解法三：归并排序（自底向上迭代）
    思路：避免递归,用循环控制合并长度,实现 O(1) 空间
    """
    if not head or not head.next:
        return head

    # 计算链表长度
    length = 0
    curr = head
    while curr:
        length += 1
        curr = curr.next

    dummy = ListNode(0, head)

    # 外层循环：合并长度从 1 开始,每次翻倍
    step = 1
    while step < length:
        curr = dummy.next  # 当前轮的起始节点
        tail = dummy  # 用于连接已排序部分

        # 内层循环：遍历链表,每次合并两个长度为 step 的子链表
        while curr:
            left = curr
            right = split(left, step)  # 分割出右半部分
            curr = split(right, step)  # 下一对的起始位置

            # 合并 left 和 right,接到 tail 后面
            tail = merge_and_connect(left, right, tail)

        step *= 2  # 合并长度翻倍

    return dummy.next


def split(head: Optional[ListNode], step: int) -> Optional[ListNode]:
    """
    从 head 开始走 step 步,然后断开,返回后半部分的头节点
    """
    for _ in range(step - 1):
        if not head:
            break
        head = head.next

    if not head:
        return None

    # 断开链表
    next_head = head.next
    head.next = None
    return next_head


def merge_and_connect(l1: Optional[ListNode], l2: Optional[ListNode], tail: ListNode) -> ListNode:
    """
    合并 l1 和 l2,接到 tail 后面,返回合并后的尾节点
    """
    curr = tail

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    # 接上剩余节点
    curr.next = l1 if l1 else l2

    # 移动到尾节点
    while curr.next:
        curr = curr.next

    return curr


# ✅ 测试
head1 = create_linked_list([4, 2, 1, 3])
print(linked_list_to_list(sortList_v3(head1)))  # 期望输出：[1, 2, 3, 4]

head2 = create_linked_list([-1, 5, 3, 4, 0])
print(linked_list_to_list(sortList_v3(head2)))  # 期望输出：[-1, 0, 3, 4, 5]
```

### 复杂度分析
- **时间复杂度**：O(n log n)
  - 外层循环 log n 次（step 从 1 翻倍到 n）
  - 每轮内层循环处理 n 个节点
- **空间复杂度**：O(1) — **没有递归,只使用常数个指针** ✅ 满足进阶要求

---

## 🐍 Pythonic 写法

归并排序的递归版本已经很简洁,但可以用 Python 的特性简化辅助函数：

```python
def sortList_pythonic(head: Optional[ListNode]) -> Optional[ListNode]:
    """
    Pythonic 写法：利用 Python 的多重赋值
    """
    if not head or not head.next:
        return head

    # 找中点并分割（一气呵成）
    slow, fast = head, head.next
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next

    mid, slow.next = slow.next, None  # 断开链表,同时保存右半部分

    # 递归排序并合并
    return merge_two_lists(
        sortList_pythonic(head),
        sortList_pythonic(mid)
    )
```

这个写法的亮点：
- 用 `mid, slow.next = slow.next, None` 同时保存右半部分和断开链表
- 函数式风格:直接返回 `merge_two_lists(递归结果1, 递归结果2)`

> ⚠️ **面试建议**：自底向上迭代版本才是满足 O(1) 空间的完美解法,但代码较复杂。面试时可以先说递归版本思路,再提优化："如果要求 O(1) 空间,可以改成自底向上迭代"。

---

## 📊 解法对比

| 维度 | 解法一：转数组 | 解法二：归并递归 | 解法三：归并迭代 |
|------|--------------|--------------|--------------|
| 时间复杂度 | O(n log n) | O(n log n) | O(n log n) |
| 空间复杂度 | O(n) | O(log n) | **O(1)** |
| 代码难度 | 简单 | 中等 | 较难 |
| 面试推荐 | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 快速实现原型 | **面试常规解法** | **满足进阶要求** |

**面试建议**：
1. 先说解法一的思路,表明你理解了问题
2. 立即提出优化:"转数组需要额外空间,能否直接在链表上排序？用归并排序！"
3. 重点讲解**解法二的递归版本**,画出递归树,演示合并过程
4. 如果面试官追问 O(1) 空间,再提**解法三的自底向上**,并说明"避免递归栈"的思路

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**：请你对链表进行排序,要求 O(n log n) 时间。

**你**：（审题30秒）好的,这道题要求对链表排序,并且时间复杂度是 O(n log n)。

我的第一个想法是把链表转成数组,用 Python 内置的 sort（Timsort,O(n log n)）,然后再转回链表。但这需要 O(n) 额外空间。

更好的方法是用**归并排序**,因为归并排序天然适合链表:
1. 分割链表不需要随机访问,用快慢指针找中点即可
2. 合并两个有序链表可以 O(1) 空间完成

核心思路是**分治法**:
- 找中点,分成两个子链表
- 递归排序两个子链表
- 合并两个有序链表

**面试官**：很好,递归版本的空间复杂度是多少？

**你**：递归版本的空间复杂度是 O(log n),因为递归栈的深度是 log n（每次分割减半）。

如果题目要求 O(1) 空间,可以改用**自底向上的迭代版本**:
- 第1轮合并长度为 1 的子链表
- 第2轮合并长度为 2 的子链表
- 每轮合并长度翻倍,最多 log n 轮

这样避免了递归栈,实现 O(1) 空间。

**面试官**：请写一下递归版本的代码。

**你**：（边写边说）
1. 递归终止条件:空链表或单节点直接返回
2. 用快慢指针找中点,注意 fast 从 head.next 开始,确保 slow 停在前半部分
3. 断开链表,递归排序两个子链表
4. 调用 merge_two_lists 合并结果

（写完代码）

**面试官**：测试一下？

**你**：用示例 [4,2,1,3]:
1. 第1次分割 → [4,2] 和 [1,3]
2. 第2次分割 → [4], [2], [1], [3]（单节点,递归终止）
3. 第1次合并 → [2,4] 和 [1,3]
4. 第2次合并 → [1,2,3,4]

结果正确。再测边界情况:空链表返回 null,单节点返回自身。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么不用快速排序？" | 快速排序需要随机访问（选 pivot 后要分区）,链表不支持 O(1) 随机访问,会退化到 O(n²)。归并排序只需顺序访问,完美适配链表 |
| "归并排序是稳定排序吗？" | 是。稳定排序指相等元素的相对顺序不变。归并排序的合并过程中,当 l1.val == l2.val 时选 l1,保证了稳定性 |
| "如果链表非常大,内存有限怎么办？" | 归并排序的自底向上版本是 O(1) 空间,已经是最优。如果链表大到内存放不下,需要外部排序（分块、外排归并） |
| "能不能用堆排序？" | 理论上可以,但需要用数组建堆,空间 O(n),且链表没有随机访问优势。归并排序更适合链表 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1: 快慢指针找中点（偶数节点时返回前半部分最后一个）
slow, fast = head, head.next
while fast and fast.next:
    slow, fast = slow.next, fast.next.next
# slow 现在是中点

# 技巧2: 同时赋值断开链表
mid, slow.next = slow.next, None  # 保存右半部分,同时断开

# 技巧3: 递归函数直接返回调用结果（函数式风格）
return merge_two_lists(
    sortList(head),
    sortList(mid)
)

# 技巧4: 三元表达式简化条件选择
curr.next = l1 if l1 else l2
```

### 💡 底层原理（选读）

> **为什么归并排序适合链表,快速排序不适合？**
>
> 1. **归并排序的关键操作**:
>    - 分割:链表找中点用快慢指针 O(n/2)
>    - 合并:顺序遍历两个链表 O(n)
>    - **都不需要随机访问**,链表天然支持
>
> 2. **快速排序的关键操作**:
>    - 选 pivot:可以用链表头 O(1)
>    - **分区（partition）**:需要把小于 pivot 的放左边,大于的放右边
>    - 链表的分区要么用额外空间（两个新链表）,要么需要频繁的指针操作,效率低
>    - 数组的分区可以双指针交换,O(1) 空间 O(n) 时间
>
> 3. **时间复杂度对比**:
>    - 归并排序:链表和数组都是 O(n log n),稳定
>    - 快速排序:数组平均 O(n log n),链表容易退化到 O(n²)（无法随机选 pivot 避免最坏情况）
>
> **Python 的 Timsort 是什么？**
>
> Python 内置的 `list.sort()` 使用 Timsort 算法:
> - 结合了归并排序和插入排序的优点
> - 对部分有序数据性能极佳（现实数据往往部分有序）
> - 稳定排序,时间 O(n log n),空间 O(n)
> - 由 Tim Peters 在 2002 年为 Python 设计,现在也被 Java、Android 等采用

### 算法模式卡片 📐

- **模式名称**：归并排序（链表版）
- **适用条件**：链表排序,要求 O(n log n) 时间
- **识别关键词**："链表排序"、"O(n log n)"、"稳定排序"
- **核心思路**：分治法——递归分割链表,合并有序子链表
- **模板代码**：
```python
def mergeSort(head):
    """归并排序链表模板"""
    # 递归终止条件
    if not head or not head.next:
        return head

    # 找中点并分割
    slow, fast = head, head.next
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
    mid = slow.next
    slow.next = None

    # 递归排序
    left = mergeSort(head)
    right = mergeSort(mid)

    # 合并
    return merge(left, right)

def merge(l1, l2):
    """合并两个有序链表"""
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
```

### 易错点 ⚠️

1. **找中点时 fast 指针初始化错误**
   - ❌ 错误：`slow = fast = head` 会导致 slow 停在后半部分的第一个节点
   - ✅ 正确：`slow = head, fast = head.next` 确保 slow 停在前半部分的最后一个节点
   - 原因：如果 fast 从 head 开始,偶数节点时 slow 会偏右,导致递归无法终止（右半部分长度不减少）

2. **忘记断开链表**
   - ❌ 错误：找到中点后直接递归,不断开 `slow.next`
   - ✅ 正确：`slow.next = None` 断开链表,否则左半部分还连着右半部分,导致无限循环
   - 示例：[1,2,3] 不断开会导致左半 [1,2,3],右半 [3],无法终止

3. **合并函数写成原地修改**
   - ❌ 错误：直接修改 l1 和 l2 的指针,可能导致原链表结构混乱
   - ✅ 正确：创建虚拟头节点 dummy,构建新的有序链表,只移动 curr 指针

4. **递归终止条件不全**
   - ❌ 错误：只判断 `if not head:`,遗漏单节点情况
   - ✅ 正确：`if not head or not head.next:`,单节点也要直接返回（已经有序）

---

## 🏗️ 工程实战（选读）

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1：数据库外部排序**
  当数据库需要排序的数据量超过内存时,使用**外部归并排序**:先把数据分块排序写入磁盘,再多路归并。归并排序的稳定性和顺序访问特性使其成为外排的首选。

- **场景2：Git 版本控制**
  Git 在合并分支时,底层使用归并排序合并两个有序的 commit 历史,保证时间线的稳定性（相同时间戳的 commit 顺序不变）。

- **场景3：日志文件合并**
  多个服务器产生的日志文件（按时间戳有序）,需要合并成全局有序的日志流,使用**多路归并**（类似合并 K 个有序链表）,效率高且稳定。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识：

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 21. 合并两个有序链表 | Easy | 归并排序的合并步骤 | 本题的子问题,先掌握合并再学排序 |
| LeetCode 23. 合并K个升序链表 | Hard | 多路归并 + 堆 | 归并排序的扩展,用最小堆优化 |
| LeetCode 147. 对链表进行插入排序 | Medium | 插入排序 | 小数据量时插入排序更简单 |
| LeetCode 剑指Offer 51. 数组中的逆序对 | Hard | 归并排序计数 | 归并排序的副产品:统计逆序对 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟！

**题目**：给定一个链表和一个值 x,将链表分成两部分,使得所有小于 x 的节点在大于等于 x 的节点之前,保持原有的相对顺序。（LeetCode 86. 分隔链表）

<details>
<summary>💡 提示（实在想不出来再点开）</summary>

用两个虚拟头节点,分别构建"小于 x"和"大于等于 x"的链表,最后拼接。类似归并排序的分区思想。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def partition(head: Optional[ListNode], x: int) -> Optional[ListNode]:
    """
    分隔链表:小于 x 的在前,大于等于 x 的在后
    """
    # 创建两个虚拟头节点
    less_dummy = ListNode(0)
    greater_dummy = ListNode(0)
    less = less_dummy
    greater = greater_dummy

    # 遍历链表,分别接到两个链表
    while head:
        if head.val < x:
            less.next = head
            less = less.next
        else:
            greater.next = head
            greater = greater.next
        head = head.next

    # 拼接两个链表
    greater.next = None  # 断开 greater 链表的尾部（避免成环）
    less.next = greater_dummy.next

    return less_dummy.next
```

**核心思路**：
- 类似归并排序的分区思想,但更简单（不需要递归）
- 用两个虚拟头节点分别存储小于和大于等于 x 的节点
- 最后拼接两个链表,注意要断开 greater 的尾部避免成环
- 时间 O(n),空间 O(1),稳定排序

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
