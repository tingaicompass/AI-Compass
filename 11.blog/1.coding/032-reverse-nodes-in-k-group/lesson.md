> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第32课：K个一组翻转链表

> **模块**：链表 | **难度**：Hard ⭐
> **LeetCode 链接**：https://leetcode.cn/problems/reverse-nodes-in-k-group/
> **前置知识**：第24课(反转链表)、第29课(快慢指针)
> **预计学习时间**：35分钟

---

## 🎯 题目描述

给你链表的头节点 `head`,每 `k` 个节点一组进行翻转,请你返回修改后的链表。

`k` 是一个正整数,它的值小于或等于链表的长度。如果节点总数不是 `k` 的整数倍,那么请将**最后剩余的节点保持原有顺序**。

**要求**：只能使用 O(1) 额外空间,**不允许修改节点的值**。

**示例：**
```
输入：head = [1,2,3,4,5], k = 2
输出：[2,1,4,3,5]
解释：
  第1组 [1,2] 翻转 → [2,1]
  第2组 [3,4] 翻转 → [4,3]
  第3组 [5] 不足 k 个,保持原顺序
  结果: [2,1,4,3,5]
```

```
输入：head = [1,2,3,4,5], k = 3
输出：[3,2,1,4,5]
解释：
  第1组 [1,2,3] 翻转 → [3,2,1]
  第2组 [4,5] 不足 k 个,保持原顺序
  结果: [3,2,1,4,5]
```

**约束条件：**
- 链表节点数范围是 `[1, 5000]`
- `1 <= k <= 链表长度`
- 不允许修改节点的值,只能改变指针

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| k=1 | head=[1,2,3], k=1 | [1,2,3] | 不需要翻转 |
| k等于链表长度 | head=[1,2,3], k=3 | [3,2,1] | 整体翻转 |
| k大于链表长度 | head=[1,2], k=3 | [1,2] | 保持原顺序 |
| 恰好整除 | head=[1,2,3,4], k=2 | [2,1,4,3] | 无剩余节点 |
| 有剩余 | head=[1,2,3,4,5], k=2 | [2,1,4,3,5] | 剩余节点保持原顺序 |

---

## 💡 思路引导

### 生活化比喻

> 想象你要整理一副扑克牌,每 K 张一组翻转顺序...
>
> 🐌 **笨办法**：把整副牌的值抄到纸上,按规则重新排序,再一张张放回牌堆。这违反了"不能修改节点值"的要求,而且需要额外空间。
>
> 🚀 **聪明办法**：原地操作
> 1. **识别一组**：数 K 张牌,用橡皮筋圈起来
> 2. **翻转这一组**：只翻转橡皮筋内的 K 张牌（原地反转）
> 3. **连接前后**：把翻转后的这组与前一组和后一组重新连接
> 4. **重复**：继续处理下一组,直到剩余不足 K 张
> 5. **保留剩余**：不足 K 张的牌保持原顺序
>
> 关键是"**局部翻转 + 全局连接**"！

### 关键洞察

**这道题的核心是：1）如何识别每 K 个一组；2）如何反转一组（第24课学过）；3）如何连接翻转后的各组。关键难点在于**指针操作的细节**和**边界条件处理**。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出
- **输入**：链表头节点 `head`,整数 `k`
- **输出**：每 k 个节点翻转后的链表
- **限制**：O(1) 空间,不能修改节点值,只能改指针

### Step 2：先想笨办法（违规解法）

如果允许修改节点值:
1. 遍历链表,收集所有值到数组 — O(n)
2. 对数组每 k 个一组进行翻转 — O(n)
3. 根据数组重建链表 — O(n)

- 时间复杂度：O(n) ✅
- 空间复杂度：O(n) ❌ 且修改了节点值 ❌
- 不符合题目要求

### Step 3：瓶颈分析 → 优化方向

必须**原地操作**,只改指针:
- 核心操作：**反转链表的一部分**（第24课学过反转整个链表）
- 关键步骤：
  1. **识别一组**：从当前位置往后数 k 个节点
  2. **反转这一组**：用迭代法反转这 k 个节点
  3. **连接前后**：把翻转后的这一组与前面和后面的部分连接起来
  4. **移动到下一组**：继续处理剩余部分

### Step 4：选择武器
- 选用：**迭代 + 分组反转链表**
- 理由：
  1. 反转链表的迭代法是 O(1) 空间（第24课学过）
  2. 用虚拟头节点简化连接操作
  3. 用指针标记每组的起始和结束位置

> 🔑 **模式识别提示**：当题目要求**分组处理链表**,优先考虑"**虚拟头节点 + 分组迭代 + 局部操作**"

---

## 🔑 解法一：递归法（优雅但非最优）

### 思路

用递归思想:
1. 先翻转前 k 个节点
2. 递归处理剩余部分
3. 连接翻转后的前 k 个与递归结果

虽然代码优雅,但递归深度是 O(n/k),不符合 O(1) 空间要求。我们先看递归,再优化成迭代。

### 图解过程

```
示例: head = [1,2,3,4,5], k = 2

Step 1: 检查前 k=2 个节点是否存在
  1 -> 2 -> 3 -> 4 -> 5
  有2个节点,可以翻转

Step 2: 翻转前 k=2 个节点 [1,2] → [2,1]
  原链表: 1 -> 2 -> 3 -> 4 -> 5
              ^
            断开

  翻转 [1,2]:
    2 -> 1 -> null

  剩余: 3 -> 4 -> 5

Step 3: 递归处理剩余部分 reverseKGroup([3,4,5], 2)
  返回: 4 -> 3 -> 5

Step 4: 连接翻转后的头部 [2,1] 与递归结果 [4,3,5]
  2 -> 1 -> 4 -> 3 -> 5
       ^
     原来的 head,现在是尾部

返回新头节点 2
```

### Python代码

```python
from typing import Optional


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    解法一：递归法
    思路：翻转前 k 个,递归处理剩余,连接结果
    """
    # 检查是否有 k 个节点
    curr = head
    for i in range(k):
        if not curr:
            return head  # 不足 k 个,保持原顺序
        curr = curr.next

    # 翻转前 k 个节点
    new_head, tail = reverse_first_k(head, k)

    # 递归处理剩余部分,连接到 tail 后面
    tail.next = reverseKGroup(curr, k)

    return new_head


def reverse_first_k(head: ListNode, k: int) -> tuple:
    """
    翻转链表的前 k 个节点
    返回 (新头节点, 原头节点/现在的尾节点)
    """
    prev = None
    curr = head
    for _ in range(k):
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node

    # prev 是新头节点, head 是原头节点(现在是尾节点)
    return prev, head


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
print(linked_list_to_list(reverseKGroup(head1, 2)))  # 期望输出：[2, 1, 4, 3, 5]

head2 = create_linked_list([1, 2, 3, 4, 5])
print(linked_list_to_list(reverseKGroup(head2, 3)))  # 期望输出：[3, 2, 1, 4, 5]

head3 = create_linked_list([1, 2])
print(linked_list_to_list(reverseKGroup(head3, 3)))  # 期望输出：[1, 2]
```

### 复杂度分析
- **时间复杂度**：O(n) — 每个节点被访问常数次（检查 + 翻转）
- **空间复杂度**：O(n/k) — **递归栈深度**,不符合 O(1) 要求

### 优缺点
- ✅ 代码优雅,逻辑清晰
- ❌ **递归栈占用 O(n/k) 空间,不符合进阶要求**

---

## ⚡ 解法二：迭代法（O(1) 空间）

### 优化思路

把递归改成迭代,用循环处理每一组:

**核心流程**:
1. 使用虚拟头节点 `dummy`,初始化 `prev = dummy`（前一组的尾节点）
2. 循环处理每一组:
   a. 检查当前是否有 k 个节点,如果不足就结束
   b. 记录这一组的起始节点 `start` 和结束节点 `end`
   c. 断开这一组与后面的连接,翻转这 k 个节点
   d. 连接前一组(`prev`) → 翻转后的头节点,翻转后的尾节点 → 下一组
   e. 更新 `prev` 为当前组的尾节点（原 start）
3. 返回 `dummy.next`

> 💡 **关键想法**：用 `prev` 指针维护"前一组的尾节点",方便连接翻转后的结果。

### 图解过程

```
示例: head = [1,2,3,4,5], k = 2

初始化:
  dummy -> 1 -> 2 -> 3 -> 4 -> 5
  prev

第1组 [1,2]:
  Step 1: 定位 start=1, end=2
    dummy -> 1 -> 2 -> 3 -> 4 -> 5
    prev   start  end

  Step 2: 断开 end.next, 记录 next=3
    dummy -> 1 -> 2   next: 3 -> 4 -> 5
    prev   start  end

  Step 3: 翻转 [1,2] → [2,1]
    dummy    2 -> 1
    prev

  Step 4: 连接 prev -> 新头(2), 新尾(1) -> next(3)
    dummy -> 2 -> 1 -> 3 -> 4 -> 5
    prev

  Step 5: 更新 prev = 1 (原 start,现在是尾)
    dummy -> 2 -> 1 -> 3 -> 4 -> 5
                  prev

第2组 [3,4]:
  Step 1: 定位 start=3, end=4
    dummy -> 2 -> 1 -> 3 -> 4 -> 5
                  prev start  end

  Step 2: 断开, next=5
  Step 3: 翻转 [3,4] → [4,3]
  Step 4: 连接
    dummy -> 2 -> 1 -> 4 -> 3 -> 5

  Step 5: 更新 prev = 3
    dummy -> 2 -> 1 -> 4 -> 3 -> 5
                            prev

第3组 [5]:
  只有 1 个节点,不足 k=2 个,停止

返回 dummy.next = 2
结果: [2,1,4,3,5]
```

### Python代码

```python
def reverseKGroup_v2(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    解法二：迭代法（O(1) 空间）
    思路：循环处理每一组,原地翻转并连接
    """
    dummy = ListNode(0, head)
    prev = dummy  # prev 是前一组的尾节点

    while True:
        # 检查是否有 k 个节点
        kth = get_kth_node(prev, k)
        if not kth:
            break  # 不足 k 个,结束

        # 记录下一组的起始位置
        next_group_start = kth.next

        # 翻转当前组 [prev.next ... kth]
        prev_node, curr = kth.next, prev.next
        while curr != next_group_start:
            next_node = curr.next
            curr.next = prev_node
            prev_node = curr
            curr = next_node

        # 连接前一组和翻转后的当前组
        temp = prev.next  # 保存原 start（翻转后的尾节点）
        prev.next = kth  # 前一组尾 -> 翻转后的头
        prev = temp  # 更新 prev 为当前组的尾（原 start）

    return dummy.next


def get_kth_node(start: ListNode, k: int) -> Optional[ListNode]:
    """
    从 start 开始,找第 k 个节点（start 算第 0 个）
    如果不足 k 个,返回 None
    """
    curr = start
    for _ in range(k):
        if not curr:
            return None
        curr = curr.next
    return curr


# ✅ 测试
head1 = create_linked_list([1, 2, 3, 4, 5])
print(linked_list_to_list(reverseKGroup_v2(head1, 2)))  # 期望输出：[2, 1, 4, 3, 5]

head2 = create_linked_list([1, 2, 3, 4, 5])
print(linked_list_to_list(reverseKGroup_v2(head2, 3)))  # 期望输出：[3, 2, 1, 4, 5]
```

### 复杂度分析
- **时间复杂度**：O(n) — 每个节点被访问常数次
- **空间复杂度**：O(1) — **只使用常数个指针** ✅ 满足进阶要求

---

## 🐍 Pythonic 写法

迭代法已经很简洁,可以用辅助函数进一步优化可读性：

```python
def reverseKGroup_pythonic(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    Pythonic 写法：提取反转函数,提高可读性
    """
    def reverse_between(start: ListNode, end: ListNode):
        """翻转 start 到 end 之间的节点（不包括 end）"""
        prev, curr = end, start
        while curr != end:
            curr.next, prev, curr = prev, curr, curr.next
        return prev  # 返回新头节点

    dummy = prev = ListNode(0, head)

    while True:
        kth = get_kth_node(prev, k)
        if not kth:
            break
        next_group = kth.next
        # 翻转当前组
        new_head = reverse_between(prev.next, next_group)
        # 连接
        temp = prev.next
        prev.next = new_head
        prev = temp

    return dummy.next
```

这个写法的亮点：
- 提取 `reverse_between` 函数,复用性更高
- 用 `curr.next, prev, curr = prev, curr, curr.next` 同时更新三个指针（Pythonic）

> ⚠️ **面试建议**：面试时推荐**解法二的迭代法**,满足 O(1) 空间要求。重点讲清楚"如何连接前一组和当前组"。

---

## 📊 解法对比

| 维度 | 解法一：递归法 | 解法二：迭代法 |
|------|--------------|--------------|
| 时间复杂度 | O(n) | O(n) |
| 空间复杂度 | O(n/k) 递归栈 | **O(1)** |
| 代码难度 | 中等 | 较难 |
| 面试推荐 | ⭐⭐ | ⭐⭐⭐ |
| 适用场景 | 理解思路 | **满足进阶要求** |

**面试建议**：
1. 可以先说递归法的思路,展示你理解了分治思想
2. 立即指出:"递归有 O(n/k) 栈空间,题目要求 O(1),需要改成迭代"
3. 重点讲解**迭代法**,画图演示指针连接过程
4. 强调关键点:"用 prev 维护前一组的尾节点,翻转后连接 prev -> 新头,新尾 -> 下一组"

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**：请你实现 K 个一组翻转链表,要求 O(1) 空间。

**你**：（审题30秒）好的,这道题要求每 k 个节点一组进行翻转,不足 k 个的保持原顺序,并且要求 O(1) 空间,不能修改节点值。

我的第一个想法是**递归**:
- 先检查是否有 k 个节点
- 翻转前 k 个节点
- 递归处理剩余部分
- 连接翻转后的结果

但递归有 O(n/k) 栈空间,不符合要求。

所以我用**迭代法**:
1. 创建虚拟头节点 dummy,用 prev 维护前一组的尾节点
2. 循环处理每一组:
   - 检查是否有 k 个节点,没有就结束
   - 翻转这 k 个节点
   - 连接前一组和当前组
   - 更新 prev 为当前组的尾节点
3. 返回 dummy.next

核心难点是**指针连接**:prev -> 翻转后的头,翻转后的尾 -> 下一组。

**面试官**：如何翻转一组节点？

**你**：翻转一组节点用迭代法（第24课学过）:
- 用三个指针 prev, curr, next
- 逐个反转指针方向: curr.next = prev
- 移动指针: prev = curr, curr = next

关键是在翻转前记录这一组的结束位置 `kth.next`,翻转后用它连接下一组。

**面试官**：画图演示一下。

**你**：（画出第一组的翻转过程）
```
初始: dummy -> 1 -> 2 -> 3 -> 4 -> 5, k=2
翻转 [1,2]: dummy -> 2 -> 1 -> 3 -> 4 -> 5
连接: prev(dummy).next = 2(新头), 1(新尾).next = 3(下一组)
```

**面试官**：测试一下边界情况。

**你**：
1. k=1: 不需要翻转,直接返回原链表
2. k 等于链表长度: 整体翻转
3. k 大于链表长度: 保持原顺序
4. 有剩余节点: 最后一组不足 k 个,保持原顺序

都正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果要求剩余节点也翻转呢？" | 去掉"检查是否有 k 个节点"的判断,改成"剩余多少个翻转多少个"。循环条件改为 `while prev.next:` |
| "如果 k 非常大,会有问题吗？" | 不会。如果 k 大于链表长度,第一次检查就返回原链表,时间 O(k),最多 O(n)。空间始终 O(1) |
| "能否用递归但优化空间？" | 理论上不行。递归天然有栈空间开销,除非编译器做尾递归优化（Python 不支持）。要 O(1) 只能用迭代 |
| "实际工程中有什么应用？" | 数据分页显示（每 k 条反转顺序）、视频播放列表分组重排、消息队列批量处理等 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1: 同时更新多个指针（链表反转核心）
curr.next, prev, curr = prev, curr, curr.next
# 等价于:
# temp = curr.next
# curr.next = prev
# prev = curr
# curr = temp

# 技巧2: 虚拟头节点简化边界处理
dummy = ListNode(0, head)
prev = dummy  # prev 初始指向 dummy,方便连接第一组

# 技巧3: 辅助函数检查是否有 k 个节点
def get_kth_node(start, k):
    curr = start
    for _ in range(k):
        if not curr:
            return None
        curr = curr.next
    return curr

# 技巧4: 保存临时节点防止丢失引用
temp = prev.next  # 保存原 start（翻转后的尾）
prev.next = kth  # 更新连接
prev = temp  # 用保存的 temp 更新 prev
```

### 💡 底层原理（选读）

> **为什么链表翻转只能 O(1) 空间,数组翻转可能需要 O(n)?**
>
> 1. **链表翻转**:
>    - 只改指针方向,不移动节点本身
>    - 三个指针 prev, curr, next 原地操作 → O(1) 空间
>    - 例: `1->2->3` 改成 `1<-2<-3`,节点物理位置不变,只改 next 指针
>
> 2. **数组翻转**:
>    - 可以原地交换元素: `arr[i], arr[n-1-i] = arr[n-1-i], arr[i]` → O(1) 空间
>    - 但如果需要保留原数组,要复制一份 → O(n) 空间
>
> 3. **本题的关键**:
>    - 每 k 个翻转,需要保持非翻转部分的连接
>    - 链表通过改指针轻松连接,数组需要移动元素
>
> **Python 的元组解包（Tuple Unpacking）原理**
>
> `a, b, c = x, y, z` 的执行过程:
> 1. 右边 `x, y, z` 先计算,创建临时元组 `(x, y, z)`
> 2. 再依次赋值给左边 `a, b, c`
> 3. 所以 `curr.next, prev, curr = prev, curr, curr.next` 是安全的:
>    - 右边先读取 `prev, curr, curr.next` 的**旧值**
>    - 再同时赋值给左边,不会互相干扰

### 算法模式卡片 📐

- **模式名称**：链表分组处理 + 局部反转
- **适用条件**：链表需要分组进行某种操作（反转、交换、删除等）
- **识别关键词**："每 k 个"、"分组"、"一组"
- **核心思路**：虚拟头节点 + prev 指针维护前一组尾部 + 循环处理每一组 + 连接操作
- **模板代码**：
```python
def process_k_groups(head, k):
    """链表分组处理模板"""
    dummy = ListNode(0, head)
    prev = dummy  # 前一组的尾节点

    while True:
        # 检查是否有 k 个节点
        kth = get_kth_node(prev, k)
        if not kth:
            break

        # 记录下一组起始位置
        next_start = kth.next

        # 对当前组 [prev.next ... kth] 进行操作
        # ... 具体操作 ...

        # 连接前一组和当前组
        # prev.next = 处理后的头
        # 处理后的尾.next = next_start

        # 更新 prev 为当前组的尾
        # prev = ...

    return dummy.next
```

### 易错点 ⚠️

1. **翻转后忘记连接下一组**
   - ❌ 错误：翻转后直接 `prev = kth`,导致翻转后的尾节点的 next 指向错误
   - ✅ 正确：翻转后,原 start（现在的尾）的 next 应该指向 `next_start`
   - 调试方法：画图标记每个指针的位置

2. **prev 指针更新错误**
   - ❌ 错误：`prev = kth` 会导致 prev 指向翻转后的头,而不是尾
   - ✅ 正确：`prev = 原 start`（翻转后变成尾）
   - 技巧：翻转前用临时变量保存 `temp = prev.next`

3. **检查节点数量时边界错误**
   - ❌ 错误：`get_kth_node(prev, k)` 从 prev 算起,导致多算一个
   - ✅ 正确：`for _ in range(k): curr = curr.next` 恰好走 k 步
   - 验证：k=2 时,从 dummy 走 2 步应该到第 2 个节点

4. **翻转循环条件写错**
   - ❌ 错误：`while curr:` 会翻转到链表末尾,而不是翻转 k 个
   - ✅ 正确：`while curr != next_start:` 翻转到记录的下一组起始位置
   - 关键：翻转前记录 `next_start = kth.next`,作为循环终止条件

---

## 🏗️ 工程实战（选读）

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1：消息队列批量处理**
  消息中间件（如 Kafka）消费消息时,每 k 条消息批量处理,可以提高吞吐量。处理顺序可能需要局部调整（如按时间戳重排每批）,类似 k 组翻转的思想。

- **场景2：UI 列表分页反转**
  移动端 App 的聊天记录或朋友圈,每页 20 条,显示时需要反转顺序（最新的在上面）。用分组反转思想,每页局部翻转后拼接。

- **场景3：视频播放列表重排**
  音乐/视频播放器的播放列表,用户要求"每 5 首歌随机打乱",但每 5 首内部保持某种顺序。用分组处理 + 局部操作实现。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识：

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 24. 两两交换链表中的节点 | Medium | k=2 的特殊情况 | 本题的简化版,k 固定为 2 |
| LeetCode 206. 反转链表 | Easy | 链表反转基础 | 本题的子问题,先掌握基础反转 |
| LeetCode 92. 反转链表 II | Medium | 反转指定区间 | 反转 [m, n] 区间,类似思想 |
| LeetCode 143. 重排链表 | Medium | 分组处理 | 将链表分成两半,交错合并 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟！

**题目**：给定链表,每 k 个节点交换位置（不是反转）。例如 k=2 时,[1,2,3,4] 变成 [2,1,4,3]。k=3 时,[1,2,3,4,5,6] 变成 [3,2,1,6,5,4]。

<details>
<summary>💡 提示（实在想不出来再点开）</summary>

"交换位置"和"反转"是一样的！每 k 个节点反转就是交换了它们的位置。用本课的方法即可。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def swapKNodes(head: Optional[ListNode], k: int) -> Optional[ListNode]:
    """
    每 k 个节点交换位置（等价于反转）
    """
    # 与 reverseKGroup 完全相同的代码
    return reverseKGroup_v2(head, k)
```

**核心思路**：
- "交换 k 个节点的位置"等价于"反转 k 个节点"
- 例如 k=2: [1,2] 交换位置 = 反转 = [2,1]
- 例如 k=3: [1,2,3] 交换位置 = 反转 = [3,2,1]
- 直接复用本课的代码即可

如果题目是"两两交换"（k=2）,还可以优化:
```python
def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:
    """两两交换节点（k=2 的优化版本）"""
    dummy = ListNode(0, head)
    prev = dummy

    while prev.next and prev.next.next:
        first = prev.next
        second = first.next

        # 交换 first 和 second
        first.next = second.next
        second.next = first
        prev.next = second

        # 移动 prev
        prev = first

    return dummy.next
```

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
