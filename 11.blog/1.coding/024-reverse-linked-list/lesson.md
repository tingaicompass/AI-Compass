# 📖 第24课:反转链表

> **模块**:链表 | **难度**:Easy ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/reverse-linked-list/
> **前置知识**:无(链表基础题)
> **预计学习时间**:20分钟

---

## 🎯 题目描述

给你单链表的头节点 `head`,请你反转链表,并返回反转后的链表。

**示例 1:**
```
输入:head = [1,2,3,4,5]
输出:[5,4,3,2,1]

可视化:
  1 -> 2 -> 3 -> 4 -> 5 -> null
反转后:
  5 -> 4 -> 3 -> 2 -> 1 -> null
```

**示例 2:**
```
输入:head = [1,2]
输出:[2,1]
```

**示例 3:**
```
输入:head = []
输出:[]
```

**约束条件:**
- 链表中节点的数目范围是 `[0, 5000]`
- `-5000 <= Node.val <= 5000`

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空链表 | `[]` | `[]` | 空指针处理 |
| 单节点 | `[1]` | `[1]` | 无需反转 |
| 两节点 | `[1,2]` | `[2,1]` | 基本反转 |
| 多节点 | `[1,2,3,4,5]` | `[5,4,3,2,1]` | 通用情况 |
| 最大规模 | `n=5000` | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你有一串珠子,每颗珠子上有一根线连接到下一颗珠子。现在你要把这串珠子的方向反过来。
>
> 🐌 **笨办法**:你用一个空盒子,从第一颗珠子开始,一颗一颗地摘下来,然后**插到盒子最前面**。第一颗珠子最后会在最后面,最后一颗会在最前面。这样需要一个额外的盒子(空间)。
>
> 🚀 **聪明办法**:你用三只手(三个指针):
> - **左手**抓着前一颗珠子(prev)
> - **中间手**抓着当前珠子(curr)
> - **右手**提前记住下一颗珠子的位置(next)
>
> 然后把当前珠子的线从"指向下一颗"改成"指向上一颗"。每次处理完一颗,三只手都向右移动一格。**关键是你不需要额外的盒子,直接在原地把线重新连接!**

### 关键洞察

**用三个指针(prev, curr, next)遍历链表,不断修改 `curr.next` 的指向,从指向下一个节点改为指向前一个节点。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:单链表的头节点 `head`
- **输出**:反转后的链表的新头节点
- **限制**:原地反转(不创建新节点)

### Step 2:先想笨办法(遍历+头插法)

用一个新的虚拟头节点,遍历原链表,把每个节点插入到新链表的头部(头插法)。
- 时间复杂度:O(n)
- 瓶颈在哪:**虽然简单,但需要理解头插法逻辑**

### Step 3:瓶颈分析 → 优化方向

头插法已经很好了,但还有更直观的方法:直接修改每个节点的 `next` 指针。
- 核心问题:"如何在遍历的同时,不丢失后续节点的引用?"
- 优化思路:"用一个临时变量保存下一个节点,然后修改当前节点的指针"

### Step 4:选择武器

- **方案1**:迭代(三指针) - 最直观,面试首选
- **方案2**:递归 - 优雅但理解难度稍高
- **方案3**:头插法 - 需要虚拟头节点

> 🔑 **模式识别提示**:当题目要求"反转链表"或"修改链表结构"时,优先考虑"多指针遍历"

---

## 🔑 解法一:迭代(三指针法,推荐)

### 思路

用三个指针遍历链表:
- `prev`:前一个节点(初始为 `None`)
- `curr`:当前节点(初始为 `head`)
- `next`:下一个节点(临时保存,防止丢失)

每次迭代:
1. 保存 `next = curr.next`(防止丢失后续节点)
2. 修改 `curr.next = prev`(反转指针)
3. 移动指针:`prev = curr`, `curr = next`

### 图解过程

```
原链表: 1 -> 2 -> 3 -> 4 -> 5 -> null

初始化:
  prev = null
  curr = 1
  next = null

第1步: 反转1的指针
  prev    curr   next
  null  <- 1      2 -> 3 -> 4 -> 5 -> null
           ↑
         反转

  操作:
  next = curr.next  (next = 2)
  curr.next = prev  (1.next = null)
  prev = curr       (prev = 1)
  curr = next       (curr = 2)

  结果: null <- 1    2 -> 3 -> 4 -> 5 -> null

第2步: 反转2的指针
         prev   curr  next
  null <- 1   <- 2     3 -> 4 -> 5 -> null

  操作:
  next = curr.next  (next = 3)
  curr.next = prev  (2.next = 1)
  prev = curr       (prev = 2)
  curr = next       (curr = 3)

  结果: null <- 1 <- 2    3 -> 4 -> 5 -> null

第3步: 反转3的指针
              prev  curr  next
  null <- 1 <- 2 <- 3     4 -> 5 -> null

... 继续 ...

最终:
  null <- 1 <- 2 <- 3 <- 4 <- 5
                              ↑
                             prev (新头节点)
          curr = null (循环结束)

返回 prev
```

### Python代码

```python
# 定义链表节点
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def reverse_list(head: ListNode) -> ListNode:
    """
    解法一:迭代(三指针法)
    思路:遍历链表,不断反转 curr.next 指针
    """
    prev = None
    curr = head

    while curr:
        # 1. 保存下一个节点
        next_node = curr.next

        # 2. 反转当前节点的指针
        curr.next = prev

        # 3. 移动指针
        prev = curr
        curr = next_node

    return prev  # prev 是新的头节点


# ✅ 测试辅助函数
def create_linked_list(values):
    """数组转链表"""
    if not values:
        return None
    head = ListNode(values[0])
    curr = head
    for val in values[1:]:
        curr.next = ListNode(val)
        curr = curr.next
    return head


def print_linked_list(head):
    """链表转数组打印"""
    result = []
    while head:
        result.append(head.val)
        head = head.next
    return result


# ✅ 测试
head1 = create_linked_list([1, 2, 3, 4, 5])
print(print_linked_list(reverse_list(head1)))  # 期望输出: [5, 4, 3, 2, 1]

head2 = create_linked_list([1, 2])
print(print_linked_list(reverse_list(head2)))  # 期望输出: [2, 1]

head3 = create_linked_list([])
print(print_linked_list(reverse_list(head3)))  # 期望输出: []
```

### 复杂度分析

- **时间复杂度**:O(n) - 遍历链表一次,每个节点只访问一次
  - 具体地说:如果链表长度 n=1000,大约需要 1000 次操作
- **空间复杂度**:O(1) - 只用了三个指针变量

### 优缺点

- ✅ 思路直观,易于理解和实现
- ✅ 空间复杂度O(1),原地反转
- ✅ 面试中最常用的解法
- ❌ 需要小心处理指针,容易写错

---

## ⚡ 解法二:递归法(优雅但难理解)

### 优化思路

递归的核心思想:
1. 递归到链表末尾
2. 从后往前依次反转每个节点的指针

> 💡 **关键想法**:假设链表 `1 -> 2 -> 3 -> null`,递归反转 `2 -> 3 -> null` 得到 `3 -> 2 -> null`,然后把 `1` 接到 `2` 的后面。

### 图解过程

```
原链表: 1 -> 2 -> 3 -> null

递归过程:
  reverse_list(1)
    ↓ 递归调用
  reverse_list(2)
    ↓ 递归调用
  reverse_list(3)
    ↓ 递归调用
  reverse_list(null)  → 返回 null

回溯过程:
  第3层: reverse_list(3)
    3.next = null → 基准情况,返回 3
    返回: 3 -> null

  第2层: reverse_list(2)
    已知: reverse_list(3) 返回 3
    操作: 2.next.next = 2  (即 3.next = 2)
          2.next = null
    返回: 3 -> 2 -> null

  第1层: reverse_list(1)
    已知: reverse_list(2) 返回 3
    操作: 1.next.next = 1  (即 2.next = 1)
          1.next = null
    返回: 3 -> 2 -> 1 -> null

最终结果: 3 -> 2 -> 1 -> null
```

### Python代码

```python
def reverse_list_recursive(head: ListNode) -> ListNode:
    """
    解法二:递归法
    思路:递归到末尾,回溯时反转指针
    """
    # 基准情况:空链表或单节点
    if not head or not head.next:
        return head

    # 递归反转后续链表,得到新头节点
    new_head = reverse_list_recursive(head.next)

    # 反转当前节点的指针
    # head.next.next = head 等价于把当前节点接到下一个节点的后面
    head.next.next = head
    head.next = None  # 断开原来的连接

    return new_head  # 返回新头节点(一直是最后一个节点)


# ✅ 测试
head1 = create_linked_list([1, 2, 3, 4, 5])
print(print_linked_list(reverse_list_recursive(head1)))  # 期望输出: [5, 4, 3, 2, 1]

head2 = create_linked_list([1, 2])
print(print_linked_list(reverse_list_recursive(head2)))  # 期望输出: [2, 1]
```

### 复杂度分析

- **时间复杂度**:O(n) - 递归调用n次
- **空间复杂度**:O(n) - 递归调用栈深度为n

---

## 🚀 解法三:头插法(虚拟头节点)

### 优化思路

用一个虚拟头节点 `dummy`,遍历原链表,把每个节点摘下来插到 `dummy` 的后面(头插)。

### 图解过程

```
原链表: 1 -> 2 -> 3 -> null

初始化: dummy -> null

第1步: 摘下1,插到dummy后面
  dummy -> 1 -> null
  剩余: 2 -> 3 -> null

第2步: 摘下2,插到dummy后面(头插)
  dummy -> 2 -> 1 -> null
  剩余: 3 -> null

第3步: 摘下3,插到dummy后面(头插)
  dummy -> 3 -> 2 -> 1 -> null
  剩余: null

返回 dummy.next
```

### Python代码

```python
def reverse_list_head_insert(head: ListNode) -> ListNode:
    """
    解法三:头插法
    思路:用虚拟头节点,依次摘下原链表节点并头插
    """
    dummy = ListNode(0)  # 虚拟头节点
    curr = head

    while curr:
        # 1. 保存下一个节点
        next_node = curr.next

        # 2. 头插:把curr插到dummy后面
        curr.next = dummy.next
        dummy.next = curr

        # 3. 移动到下一个节点
        curr = next_node

    return dummy.next


# ✅ 测试
head1 = create_linked_list([1, 2, 3, 4, 5])
print(print_linked_list(reverse_list_head_insert(head1)))  # 期望输出: [5, 4, 3, 2, 1]
```

### 复杂度分析

- **时间复杂度**:O(n)
- **空间复杂度**:O(1)

---

## 🐍 Pythonic 写法

Python中链表操作通常需要定义节点类,没有特别简洁的"一行写法"。但可以用更Pythonic的风格:

```python
def reverse_list_pythonic(head: ListNode) -> ListNode:
    """Pythonic 风格:元组解包"""
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    return prev
```

**解释**:利用元组解包同时赋值,`curr.next, prev, curr = prev, curr, curr.next` 等价于:
- 临时保存右边的值:`(prev, curr, curr.next)`
- 依次赋值给左边:`curr.next = prev`, `prev = curr`, `curr = curr.next`

> ⚠️ **面试建议**:先写清晰版本(解法一),再提这种Pythonic写法展示语言特性。
> 面试官更看重你的**思路清晰度**,不要为了简洁而牺牲可读性。

---

## 📊 解法对比

| 维度 | 解法一:迭代 | 解法二:递归 | 解法三:头插法 |
|------|-----------|-----------|-------------|
| 时间复杂度 | O(n) | O(n) | O(n) |
| 空间复杂度 | O(1) ⭐ | O(n) | O(1) ⭐ |
| 代码难度 | 简单 | 中等 | 简单 |
| 面试推荐 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 适用场景 | 通用首选 | 展示递归思维 | 熟悉头插技巧 |

**面试建议**:
1. **首选解法一(迭代)**:思路最清晰,易于解释,面试官最容易理解
2. **如果时间充裕**:可以提出"我还能用递归实现",展示多种思路
3. **避免直接用解法三**:头插法虽然也是O(1)空间,但逻辑稍复杂,不如解法一直观

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你反转一个单链表。

**你**:(审题10秒)好的,我理解了。比如链表 `1 -> 2 -> 3 -> null`,反转后变成 `3 -> 2 -> 1 -> null`。

我的思路是用**三个指针迭代遍历**:
- `prev` 指向前一个节点,初始为 `None`
- `curr` 指向当前节点,初始为 `head`
- `next` 临时保存下一个节点,防止丢失引用

每次迭代:
1. 保存 `next = curr.next`
2. 反转指针:`curr.next = prev`
3. 移动指针:`prev = curr`, `curr = next`

循环结束时,`prev` 就是新的头节点。

时间复杂度 O(n),空间复杂度 O(1)。

**面试官**:很好,请写代码。

**你**:(边写边说)我先定义三个指针...遍历链表,每次先保存next防止丢失,然后反转当前节点的指针,最后移动所有指针...

(写完代码)

**面试官**:测试一下?

**你**:用示例 `[1,2,3]` 走一遍:
- 初始:`prev=null, curr=1`
- 第1轮:保存next=2,反转1.next=null,移动prev=1,curr=2
- 第2轮:保存next=3,反转2.next=1,移动prev=2,curr=3
- 第3轮:保存next=null,反转3.next=2,移动prev=3,curr=null
- 循环结束,返回prev=3

结果正确:`3 -> 2 -> 1 -> null`

**面试官**:空链表呢?

**你**:空链表时,`curr=None`,while循环不执行,直接返回 `prev=None`,正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能用递归实现吗?" | 可以!递归到链表末尾,回溯时依次反转指针。代码更简洁,但空间复杂度O(n)(递归栈)。 |
| "如何反转链表的一部分(如第m到n个节点)?" | 先找到第m个节点的前驱,然后对m到n这段用三指针反转,最后重新连接前后部分(LeetCode 92) |
| "如果链表有环怎么办?" | 题目假设是单链表,但如果有环,迭代法会死循环。需要先用快慢指针判环(LeetCode 141) |
| "能否用栈实现?" | 可以,但需要O(n)空间。遍历链表把节点入栈,然后出栈重新连接。不如三指针法空间优 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:元组解包 — 同时赋值多个变量
a, b, c = 1, 2, 3
a, b = b, a  # 交换变量,无需临时变量

# 技巧2:链表节点定义
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 技巧3:虚拟头节点技巧
dummy = ListNode(0)
dummy.next = head  # 统一处理头节点

# 技巧4:链表遍历模板
curr = head
while curr:
    # 处理curr
    curr = curr.next

# 技巧5:Pythonic的三指针
prev, curr = None, head
while curr:
    curr.next, prev, curr = prev, curr, curr.next
```

### 💡 底层原理(选读)

> **为什么需要保存next?**
>
> 链表的每个节点只有一个 `next` 指针指向下一个节点。当你执行 `curr.next = prev` 时,原来的 `curr.next` 就被覆盖了,如果不提前保存,你就找不到下一个节点了!这就像烧断了桥,你就过不去了。
>
> **递归为什么消耗O(n)空间?**
>
> 每次递归调用都会在调用栈上保存当前函数的局部变量和返回地址。链表长度为n,就会有n层递归调用,调用栈深度为n,所以空间复杂度是O(n)。在Python中,默认递归深度限制是1000,超过会报 `RecursionError`。
>
> **迭代和递归的选择?**
>
> - 迭代:空间O(1),速度快,但代码可能稍长
> - 递归:代码简洁优雅,但空间O(n),且有栈溢出风险
> - 面试中:迭代更安全,递归可以作为备选方案展示思维广度

### 算法模式卡片 📐

- **模式名称**:链表指针操作(反转/重排)
- **适用条件**:需要修改链表结构(反转、重新连接)
- **识别关键词**:"反转链表"、"重排链表"、"调整链表顺序"
- **模板代码**:
```python
def reverse_linked_list_template(head: ListNode) -> ListNode:
    """链表反转的通用模板(三指针迭代)"""
    prev = None
    curr = head

    while curr:
        next_node = curr.next  # 1. 保存下一个节点
        curr.next = prev       # 2. 反转指针
        prev = curr            # 3. 移动prev
        curr = next_node       # 4. 移动curr

    return prev  # prev是新头节点
```

### 易错点 ⚠️

1. **忘记保存next导致链表断裂**
   - ❌ 错误:直接 `curr.next = prev`,丢失后续节点引用
   - ✅ 正确:先 `next = curr.next` 保存,再修改 `curr.next`

2. **返回错误的头节点**
   - ❌ 错误:返回 `curr`(循环结束时curr是None)
   - ✅ 正确:返回 `prev`(prev指向反转后的第一个节点)

3. **空链表/单节点未特殊处理**
   - ❌ 错误:直接操作可能导致空指针错误
   - ✅ 正确:迭代法自然处理(while循环不执行),递归法需要基准条件

4. **递归中忘记断开原连接**
   - ❌ 错误:只写 `head.next.next = head`,形成环
   - ✅ 正确:还要写 `head.next = None` 断开

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:浏览器的后退功能**
  - 浏览器的历史记录可以用链表存储,后退时相当于反向遍历(反转链表的思想)

- **场景2:编辑器的撤销栈**
  - 某些编辑器用链表存储操作历史,撤销操作时需要反向应用(类似反转)

- **场景3:区块链**
  - 区块链的每个区块链接形成链表,某些验证算法需要从最新区块回溯到创世区块(反向遍历)

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 92. 反转链表II | Medium | 三指针+区间处理 | 反转链表的第m到n个节点,需要记录前驱和后继 |
| LeetCode 25. K个一组翻转链表 | Hard | 分段反转 | 每k个节点一组反转,本题的进阶版 |
| LeetCode 234. 回文链表 | Easy | 快慢指针+反转 | 判断链表是否回文,需要找中点+反转后半段 |
| LeetCode 143. 重排链表 | Medium | 找中点+反转+合并 | L0→L1→L2...→Ln 重排为 L0→Ln→L1→Ln-1... |
| LeetCode 24. 两两交换链表节点 | Medium | 指针操作 | 每两个节点交换一次,类似部分反转 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定链表头节点和两个整数 `left` 和 `right`,反转从位置 `left` 到 `right` 的链表节点,返回反转后的链表。(LeetCode 92简化版)

例如:
```
输入:head = [1,2,3,4,5], left = 2, right = 4
输出:[1,4,3,2,5]
解释:反转第2到第4个节点
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

分三步:
1. 找到 `left` 的前一个节点 `pre`
2. 对从 `left` 到 `right` 的子链表用三指针反转
3. 重新连接:`pre.next` 接到反转后的头,反转后的尾接到 `right.next`

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def reverse_between(head: ListNode, left: int, right: int) -> ListNode:
    """
    反转链表的第left到right个节点
    """
    if not head or left == right:
        return head

    # 虚拟头节点,简化边界处理
    dummy = ListNode(0)
    dummy.next = head
    pre = dummy

    # 1. 移动到left的前一个节点
    for _ in range(left - 1):
        pre = pre.next

    # 2. 反转从left到right的子链表
    curr = pre.next
    for _ in range(right - left):
        # 头插法:把curr.next摘下来,插到pre后面
        temp = curr.next
        curr.next = temp.next
        temp.next = pre.next
        pre.next = temp

    return dummy.next


# 测试
head = create_linked_list([1, 2, 3, 4, 5])
result = reverse_between(head, 2, 4)
print(print_linked_list(result))  # 输出: [1, 4, 3, 2, 5]
```

核心思路:
1. 用虚拟头节点 `dummy` 简化处理(避免left=1的边界情况)
2. 移动到 `left` 的前一个节点 `pre`
3. 用头插法反转 `left` 到 `right` 的部分(反转 `right-left` 次)

时间复杂度 O(n),空间复杂度 O(1)。

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
