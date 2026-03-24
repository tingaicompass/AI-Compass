> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第25课:合并两个有序链表

> **模块**:链表 | **难度**:Easy ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/merge-two-sorted-lists/
> **前置知识**:第24课(反转链表)
> **预计学习时间**:20分钟

---

## 🎯 题目描述

将两个升序链表合并为一个新的**升序**链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

**示例 1:**
```
输入:l1 = [1,2,4], l2 = [1,3,4]
输出:[1,1,2,3,4,4]

可视化:
  l1: 1 -> 2 -> 4 -> null
  l2: 1 -> 3 -> 4 -> null
合并: 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> null
```

**示例 2:**
```
输入:l1 = [], l2 = []
输出:[]
```

**示例 3:**
```
输入:l1 = [], l2 = [0]
输出:[0]
```

**约束条件:**
- 两个链表的节点数目范围是 `[0, 50]`
- `-100 <= Node.val <= 100`
- `l1` 和 `l2` 均按**非递减顺序**排列

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 两空链表 | `[], []` | `[]` | 空指针处理 |
| 一空一非空 | `[], [1]` | `[1]` | 单链表直接返回 |
| 等长相同 | `[1,2], [1,2]` | `[1,1,2,2]` | 相等元素处理 |
| 不等长 | `[1], [2,3,4]` | `[1,2,3,4]` | 剩余节点拼接 |
| 无交集 | `[1,2], [5,6]` | `[1,2,5,6]` | 完全分离 |
| 最大规模 | `n=50, m=50` | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在整理两摞已经按高度排好序的书,现在要把它们合并成一摞,同时保持从矮到高的顺序。
>
> 🐌 **笨办法**:把两摞书全部拆开扔在地上,然后重新排序。这样太慢了,而且丢失了原本已有的顺序信息!
>
> 🚀 **聪明办法**:你用两只手分别指向两摞书的最上面(最矮的书),每次对比两只手指向的书:
> - 哪本更矮,就把哪本拿出来放到新的一摞
> - 然后那只手移动到下一本书
> - 重复直到一摞拿完,最后把另一摞剩余的书直接摞上去
>
> **关键洞察:利用"已经有序"的特性,只需要双指针逐个对比,不需要重新排序!**

### 关键洞察

**类似归并排序的"归并"过程:用双指针分别遍历两个有序链表,每次选择较小的节点接到结果链表,最后拼接剩余部分。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:两个升序链表的头节点 `l1` 和 `l2`
- **输出**:合并后的升序链表的头节点
- **限制**:原地操作,不创建新节点(只修改指针)

### Step 2:先想笨办法(遍历+重新排序)

把两个链表的所有节点值提取到数组,排序后重新构建链表。
- 时间复杂度:O((m+n)log(m+n)) - 排序的代价
- 瓶颈在哪:**丢失了原有的有序性,浪费了资源**

### Step 3:瓶颈分析 → 优化方向

两个链表已经有序,应该利用这个特性!
- 核心问题:"如何利用已有的顺序,避免重新排序?"
- 优化思路:"双指针对比,每次选较小的节点,类似归并排序的归并过程"

### Step 4:选择武器

- **方案1**:迭代(虚拟头节点 + 双指针) - 最直观,面试首选
- **方案2**:递归 - 代码简洁,优雅

> 🔑 **模式识别提示**:当题目涉及"合并有序序列"时,优先考虑"归并"思想

---

## 🔑 解法一:迭代(虚拟头节点,推荐)

### 思路

1. 创建虚拟头节点 `dummy`,简化边界处理
2. 用指针 `curr` 指向当前构建到的位置
3. 用两个指针 `p1` 和 `p2` 分别遍历 `l1` 和 `l2`
4. 每次对比 `p1.val` 和 `p2.val`,选小的接到 `curr.next`
5. 移动对应的指针
6. 循环结束后,把剩余的链表直接拼接上

### 图解过程

```
示例: l1 = [1,2,4], l2 = [1,3,4]

初始化:
  dummy -> null
  curr = dummy
  p1 = 1 -> 2 -> 4 -> null
  p2 = 1 -> 3 -> 4 -> null

第1步: 对比 p1.val=1, p2.val=1 (相等,选p1)
  dummy -> 1
  curr = 1
  p1 = 2 -> 4 -> null
  p2 = 1 -> 3 -> 4 -> null

第2步: 对比 p1.val=2, p2.val=1 (p2更小)
  dummy -> 1 -> 1
  curr = 1 (第二个)
  p1 = 2 -> 4 -> null
  p2 = 3 -> 4 -> null

第3步: 对比 p1.val=2, p2.val=3 (p1更小)
  dummy -> 1 -> 1 -> 2
  curr = 2
  p1 = 4 -> null
  p2 = 3 -> 4 -> null

第4步: 对比 p1.val=4, p2.val=3 (p2更小)
  dummy -> 1 -> 1 -> 2 -> 3
  curr = 3
  p1 = 4 -> null
  p2 = 4 -> null

第5步: 对比 p1.val=4, p2.val=4 (相等,选p1)
  dummy -> 1 -> 1 -> 2 -> 3 -> 4
  curr = 4
  p1 = null
  p2 = 4 -> null

第6步: p1已空,直接拼接p2剩余部分
  dummy -> 1 -> 1 -> 2 -> 3 -> 4 -> 4 -> null

返回 dummy.next
```

### Python代码

```python
# 定义链表节点(与第24课相同)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def merge_two_lists(l1: ListNode, l2: ListNode) -> ListNode:
    """
    解法一:迭代(虚拟头节点 + 双指针)
    思路:双指针归并,每次选较小的节点
    """
    # 虚拟头节点,简化边界处理
    dummy = ListNode(0)
    curr = dummy

    # 双指针遍历
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    # 拼接剩余部分(至多一个链表有剩余)
    curr.next = l1 if l1 else l2

    return dummy.next


# ✅ 测试辅助函数(与第24课相同)
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
l1 = create_linked_list([1, 2, 4])
l2 = create_linked_list([1, 3, 4])
result = merge_two_lists(l1, l2)
print(print_linked_list(result))  # 期望输出: [1, 1, 2, 3, 4, 4]

l1 = create_linked_list([])
l2 = create_linked_list([0])
result = merge_two_lists(l1, l2)
print(print_linked_list(result))  # 期望输出: [0]
```

### 复杂度分析

- **时间复杂度**:O(m + n) - 每个节点只访问一次,m和n分别是两个链表的长度
  - 具体地说:如果 l1 有50个节点,l2 有50个节点,大约需要 100 次操作
- **空间复杂度**:O(1) - 只用了常数个指针变量(不算输出链表)

### 优缺点

- ✅ 思路清晰,易于理解
- ✅ 虚拟头节点避免了头节点的特殊处理
- ✅ 原地操作,空间O(1)
- ✅ 面试中最推荐的解法

---

## ⚡ 解法二:递归法(优雅简洁)

### 优化思路

递归的核心思想:
- 如果 `l1.val <= l2.val`,则 `l1.next = merge(l1.next, l2)`
- 否则,`l2.next = merge(l1, l2.next)`
- 基准情况:如果某个链表为空,返回另一个链表

> 💡 **关键想法**:合并两个链表 = 选择较小的头节点 + 递归合并剩余部分

### 图解过程

```
示例: l1 = [1,2,4], l2 = [1,3,4]

递归树:
  merge([1,2,4], [1,3,4])
    ↓ 1 <= 1, 选l1
  1 -> merge([2,4], [1,3,4])
         ↓ 2 > 1, 选l2
       1 -> merge([2,4], [3,4])
              ↓ 2 < 3, 选l1
            2 -> merge([4], [3,4])
                   ↓ 4 > 3, 选l2
                 3 -> merge([4], [4])
                        ↓ 4 == 4, 选l1
                      4 -> merge(null, [4])
                             ↓ l1为空,返回l2
                           4 -> null

回溯组装:
  1 -> 1 -> 2 -> 3 -> 4 -> 4 -> null
```

### Python代码

```python
def merge_two_lists_recursive(l1: ListNode, l2: ListNode) -> ListNode:
    """
    解法二:递归法
    思路:选择较小的头节点,递归合并剩余部分
    """
    # 基准情况:某个链表为空
    if not l1:
        return l2
    if not l2:
        return l1

    # 递归情况:选择较小的节点,递归合并
    if l1.val <= l2.val:
        l1.next = merge_two_lists_recursive(l1.next, l2)
        return l1
    else:
        l2.next = merge_two_lists_recursive(l1, l2.next)
        return l2


# ✅ 测试
l1 = create_linked_list([1, 2, 4])
l2 = create_linked_list([1, 3, 4])
result = merge_two_lists_recursive(l1, l2)
print(print_linked_list(result))  # 期望输出: [1, 1, 2, 3, 4, 4]
```

### 复杂度分析

- **时间复杂度**:O(m + n) - 递归调用 m+n 次
- **空间复杂度**:O(m + n) - 递归调用栈深度

---

## 🐍 Pythonic 写法

递归版本已经很简洁了,可以进一步压缩:

```python
def merge_two_lists_pythonic(l1: ListNode, l2: ListNode) -> ListNode:
    """Pythonic 递归一行版本(可读性稍差)"""
    if not l1 or not l2:
        return l1 or l2
    if l1.val <= l2.val:
        l1.next = merge_two_lists_pythonic(l1.next, l2)
        return l1
    else:
        l2.next = merge_two_lists_pythonic(l1, l2.next)
        return l2
```

或者更极致的三元表达式版本:

```python
def merge_two_lists_oneliner(l1: ListNode, l2: ListNode) -> ListNode:
    """极致简洁(不推荐,难以调试)"""
    if not l1 or not l2:
        return l1 or l2
    small, large = (l1, l2) if l1.val <= l2.val else (l2, l1)
    small.next = merge_two_lists_oneliner(small.next, large)
    return small
```

> ⚠️ **面试建议**:先写迭代版本展示清晰思路,再提递归版本展示多种解法。
> Pythonic的极简版本可读性差,面试时不推荐直接写。

---

## 📊 解法对比

| 维度 | 解法一:迭代 | 解法二:递归 |
|------|-----------|-----------|
| 时间复杂度 | O(m+n) | O(m+n) |
| 空间复杂度 | O(1) ⭐ | O(m+n) |
| 代码难度 | 简单 | 简单 |
| 面试推荐 | ⭐⭐⭐ | ⭐⭐ |
| 适用场景 | 通用首选,空间最优 | 代码简洁,展示递归思维 |

**面试建议**:
1. **首选解法一(迭代)**:空间O(1),思路清晰,易于调试
2. **如果时间充裕**:可以补充"我还能用递归实现",展示多种思路
3. **强调虚拟头节点技巧**:这是链表题的常用技巧,避免头节点的特殊判断

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你合并两个升序链表。

**你**:(审题10秒)好的,我理解了。比如 `[1,2,4]` 和 `[1,3,4]` 合并后是 `[1,1,2,3,4,4]`。

我的思路是用**双指针归并**:
- 创建一个虚拟头节点 `dummy`,简化边界处理
- 用两个指针分别遍历两个链表
- 每次对比两个指针指向的节点值,选较小的接到结果链表
- 循环结束后,把剩余的链表(如果有)直接拼接上

这类似归并排序的归并过程,时间复杂度 O(m+n),空间复杂度 O(1)。

**面试官**:很好,请写代码。

**你**:(边写边说)我先创建虚拟头节点...然后双指针遍历,每次选较小的...最后拼接剩余部分...

(写完代码)

**面试官**:测试一下?

**你**:用示例 `[1,2,4]` 和 `[1,3,4]` 走一遍:
- 对比1和1,选第一个1,移动l1
- 对比2和1,选1,移动l2
- 对比2和3,选2,移动l1
- 对比4和3,选3,移动l2
- 对比4和4,选第一个4,移动l1
- l1为空,拼接l2剩余的4
- 最终得到 `[1,1,2,3,4,4]`

结果正确!

**面试官**:如果一个链表为空呢?

**你**:如果 l1 为空,while循环不执行,直接拼接 l2,返回 `dummy.next` 就是 l2 本身。同理 l2 为空也正确处理。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能用递归实现吗?" | 可以!递归思路是:选择较小的头节点,递归合并剩余部分。代码更简洁,但空间复杂度O(m+n)(递归栈)。 |
| "虚拟头节点有什么好处?" | 避免了头节点的特殊判断。如果不用虚拟头节点,需要先单独处理第一个节点,代码会更复杂。 |
| "如果有k个有序链表呢?" | 这是LeetCode 23,可以用最小堆(优先队列)或者分治法。最小堆:每次取k个链表头中最小的,O(N log k);分治:两两合并,O(N log k)。 |
| "能否原地合并?" | 这道题本身就是原地操作,只修改指针,没有创建新节点。空间O(1)(不算输出)。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:虚拟头节点技巧 — 避免头节点特殊处理
dummy = ListNode(0)
curr = dummy
# ... 构建链表 ...
return dummy.next  # 返回真正的头节点

# 技巧2:条件表达式选择
curr.next = l1 if l1 else l2  # 等价于三元表达式

# 技巧3:逻辑或运算的短路特性
return l1 or l2  # 如果l1非空返回l1,否则返回l2

# 技巧4:递归的简洁性
def merge(l1, l2):
    if not l1: return l2
    if not l2: return l1
    if l1.val <= l2.val:
        l1.next = merge(l1.next, l2)
        return l1
    else:
        l2.next = merge(l1, l2.next)
        return l2
```

### 💡 底层原理(选读)

> **为什么虚拟头节点有用?**
>
> 链表操作中,头节点的处理往往是特殊的:
> - 如果不用虚拟头节点,你需要先判断哪个链表的头节点更小,单独处理第一个节点
> - 用虚拟头节点后,所有节点的处理逻辑统一,不需要特殊判断
>
> **归并的本质?**
>
> 归并排序的"归并"步骤,就是合并两个有序数组/链表。核心思想是"利用已有顺序",双指针线性扫描,不需要重新排序。时间复杂度O(n),是最优的(因为至少要看一遍所有元素)。
>
> **递归 vs 迭代的选择?**
>
> - 递归:代码简洁,逻辑清晰,但有栈溢出风险,空间O(n)
> - 迭代:代码稍长,但空间O(1),更稳定
> - 链表长度有限(≤50)时,递归可以接受;长度未知时,迭代更安全

### 算法模式卡片 📐

- **模式名称**:归并(合并有序序列)
- **适用条件**:合并两个或多个已排序的序列
- **识别关键词**:"合并有序"、"归并"、"两个排序"
- **模板代码**:
```python
def merge_two_sorted_template(l1: ListNode, l2: ListNode) -> ListNode:
    """归并两个有序链表的通用模板"""
    dummy = ListNode(0)  # 虚拟头节点
    curr = dummy

    # 双指针归并
    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    # 拼接剩余部分
    curr.next = l1 if l1 else l2

    return dummy.next
```

### 易错点 ⚠️

1. **忘记拼接剩余部分**
   - ❌ 错误:循环结束后直接返回,丢失一个链表的剩余节点
   - ✅ 正确:`curr.next = l1 if l1 else l2`

2. **头节点处理错误**
   - ❌ 错误:不用虚拟头节点,需要单独判断第一个节点,容易出错
   - ✅ 正确:用虚拟头节点 `dummy`,统一处理所有节点

3. **相等时的选择**
   - ❌ 错误:相等时随意选,可能破坏稳定性
   - ✅ 正确:通常选 `l1`(用 `<=` 而非 `<`),保证归并的稳定性

4. **递归未处理空链表**
   - ❌ 错误:直接访问 `l1.val` 可能空指针
   - ✅ 正确:先判断 `if not l1: return l2`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:数据库的归并排序**
  - 多个已排序的数据块需要合并时(如外排序),用归并算法高效合并

- **场景2:分布式系统的数据聚合**
  - 多个服务器返回的有序结果需要合并(如搜索引擎的结果聚合)

- **场景3:版本控制系统**
  - Git的三路合并(three-way merge)底层也用到归并思想,合并两个分支的修改

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 23. 合并K个升序链表 | Hard | 堆/分治 | 本题的进阶版,可以用最小堆或两两归并 |
| LeetCode 88. 合并两个有序数组 | Easy | 双指针归并 | 数组版本,注意从后往前归并避免覆盖(第13课) |
| LeetCode 148. 排序链表 | Medium | 归并排序 | 对链表进行归并排序,O(n log n) |
| LeetCode 2. 两数相加 | Medium | 链表遍历 | 类似归并,但是按位相加,处理进位 |
| LeetCode 1669. 合并两个链表 | Medium | 链表拼接 | 将链表1的一部分替换为链表2 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定k个升序链表,将它们合并为一个升序链表并返回。(LeetCode 23)

例如:
```
输入:lists = [[1,4,5],[1,3,4],[2,6]]
输出:[1,1,2,3,4,4,5,6]
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

方法1(最小堆):
- 用最小堆维护k个链表的当前头节点
- 每次弹出堆顶(最小节点),接到结果链表
- 把该节点的next入堆,继续

方法2(分治):
- 两两归并:先合并相邻的两个链表,得到k/2个链表
- 递归合并,直到只剩一个链表
- 时间复杂度 O(N log k),N是总节点数

</details>

<details>
<summary>✅ 参考答案</summary>

**方法1:最小堆(优先队列)**

```python
import heapq

def merge_k_lists_heap(lists):
    """
    用最小堆合并k个有序链表
    """
    # 最小堆,存储 (节点值, 链表索引, 节点)
    heap = []

    # 初始化:把每个链表的头节点入堆
    for i, head in enumerate(lists):
        if head:
            heapq.heappush(heap, (head.val, i, head))

    dummy = ListNode(0)
    curr = dummy

    while heap:
        # 弹出最小节点
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next

        # 把该节点的next入堆
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

**方法2:分治法(两两归并)**

```python
def merge_k_lists_divide(lists):
    """
    用分治法合并k个有序链表
    """
    if not lists:
        return None
    if len(lists) == 1:
        return lists[0]

    # 分治:两两归并
    mid = len(lists) // 2
    left = merge_k_lists_divide(lists[:mid])
    right = merge_k_lists_divide(lists[mid:])

    # 合并两个链表(复用本课的函数)
    return merge_two_lists(left, right)
```

**复杂度对比**:
- 堆方法:时间 O(N log k),空间 O(k)
- 分治方法:时间 O(N log k),空间 O(log k) (递归栈)

两种方法时间复杂度相同,空间上分治稍优。

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
