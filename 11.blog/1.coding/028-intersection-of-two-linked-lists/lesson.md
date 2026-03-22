# 📖 第28课:相交链表

> **模块**:链表 | **难度**:Easy ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/intersection-of-two-linked-lists/
> **前置知识**:第24课(反转链表) - 双指针思想
> **预计学习时间**:20分钟

---

## 🎯 题目描述

给你两个单链表的头节点 `headA` 和 `headB`,请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点,返回 `null`。

**注意**:相交的定义是两个链表从某个节点开始,后续所有节点都是同一个节点(引用相同),不仅仅是值相同。

**示例 1:**
```
输入:intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5], skipA = 2, skipB = 3
输出:节点8(引用)
解释:
  listA: 4 -> 1 ↘
                 8 -> 4 -> 5 -> null
  listB: 5 -> 6 -> 1 ↗

两个链表在节点8相交
```

**示例 2:**
```
输入:intersectVal = 0, listA = [1,2,3], listB = [4,5], skipA = 3, skipB = 2
输出:null
解释:两个链表不相交
```

**约束条件:**
- `listA` 中节点数目为 `m`
- `listB` 中节点数目为 `n`
- `1 <= m, n <= 3 * 10⁴`
- `-10⁵ <= Node.val <= 10⁵`

**进阶**:你能否设计一个时间复杂度 `O(m + n)`、空间复杂度 `O(1)` 的解决方案?

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 不相交 | `[1,2], [3,4]` | `null` | 基本判断 |
| 完全相同 | `[1,2], [1,2]` | `节点1` | 从头相交 |
| 等长相交 | `[1,8,9], [2,8,9]` | `节点8` | 长度相同 |
| 不等长相交 | `[4,1,8,9], [5,6,1,8,9]` | `节点8` | 长度不同 |
| 单节点相交 | `[1], [2,1]` | `节点1` | 极端情况 |
| 最大规模 | `m=30000, n=30000` | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象两条不同的小路,可能在某个路口汇合,汇合后就变成同一条路。现在你要找到这个汇合点。
>
> 🐌 **笨办法**:你在第一条路上每走一步,就用粉笔在地上做个记号。然后沿着第二条路走,看看哪里有粉笔记号,第一个有记号的地方就是汇合点。这需要很多粉笔(空间)。
>
> 🚀 **聪明办法**:你派两个人,A从第一条路出发,B从第二条路出发:
> - A走完第一条路后,转到第二条路继续走
> - B走完第二条路后,转到第一条路继续走
> - **神奇的是,如果两条路有汇合点,两个人一定会在汇合点相遇!**
>
> **数学原理**:A走了(路A长度 + 路B长度 - 共同部分),B也走了(路B长度 + 路A长度 - 共同部分),距离相等,所以会在汇合点相遇!

### 关键洞察

**双指针等距法:pA走完A再走B,pB走完B再走A。如果有交点,两者会在交点相遇;如果无交点,两者会同时到达null。**

---

## 🧠 解题思维链

### Step 1:理解题目 → 锁定输入输出

- **输入**:两个链表的头节点 `headA` 和 `headB`
- **输出**:相交的起始节点,或 `null`
- **关键**:相交是指节点引用相同,不是值相同

### Step 2:先想笨办法(哈希表)

遍历链表A,把所有节点存入集合。然后遍历链表B,第一个在集合中的节点就是交点。
- 时间O(m+n),空间O(m)

### Step 3:瓶颈分析 → 优化方向

能否优化到O(1)空间?
- 核心问题:"不用额外空间,如何找到交点?"
- 优化思路:"双指针等距法!让两个指针走相同的距离"

### Step 4:选择武器

- **方案1**:哈希表 - O(m)空间
- **方案2**:双指针等距法 - O(1)空间,面试最优

---

## 🔑 解法一:双指针等距法(推荐)

### 思路

用两个指针 `pA` 和 `pB`:
- `pA` 从 `headA` 开始,走完A后转到B继续走
- `pB` 从 `headB` 开始,走完B后转到A继续走

**关键定理**:
- 如果有交点,两者会在交点相遇
- 如果无交点,两者会同时到达null

**数学证明**:
```
假设:
- A链表独有部分长度 = a
- B链表独有部分长度 = b
- 相交部分长度 = c

有交点情况:
  pA走: a + c + b步到达交点
  pB走: b + c + a步到达交点
  a+c+b = b+c+a,距离相等,会在交点相遇!

无交点情况(c=0):
  pA走: a + b步到达null
  pB走: b + a步到达null
  距离相等,同时到达null!
```

### 图解过程

```
示例: A = [4,1,8,4,5], B = [5,6,1,8,4,5]

链表结构:
  A: 4 -> 1 ↘
             8 -> 4 -> 5 -> null
  B: 5 -> 6 -> 1 ↗

  a=2, b=3, c=3

pA的路径: 4 -> 1 -> 8 -> 4 -> 5 -> null -> 5 -> 6 -> 1 -> 8(相遇)
pB的路径: 5 -> 6 -> 1 -> 8(相遇)

计数:
  pA走了: a + c + b = 2 + 3 + 3 = 8步到8
  pB走了: b + c = 3 + 3 = 6步到8

等等,不对?让我重新计算:
  pA走了: a + c = 2 + 3 = 5步到null,然后走b = 3步,共8步到达8
  pB走了: b + c = 3 + 3 = 6步到8

还是不对...正确的是:
  pA走完A(a+c=5步),转到B(走b=3步),总共8步到达交点8
  pB走完B(b+c=6步),转到A(走a=2步),总共8步到达交点8
```

### Python代码

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def get_intersection_node(headA: ListNode, headB: ListNode) -> ListNode:
    """
    解法一:双指针等距法
    思路:pA走完A再走B,pB走完B再走A,会在交点相遇
    """
    if not headA or not headB:
        return None

    pA, pB = headA, headB

    # 循环直到相遇(交点)或同时为null(无交点)
    while pA != pB:
        # pA走完A后转到B,走完B后到null
        pA = pA.next if pA else headB
        # pB走完B后转到A,走完A后到null
        pB = pB.next if pB else headA

    return pA  # 相交则返回交点,不相交则返回null


# ✅ 测试辅助函数
def create_intersected_lists(listA_vals, listB_vals, skipA, skipB):
    """创建两个相交的链表"""
    # 创建链表A的独有部分
    if skipA == 0:
        headA = None
        tailA = None
    else:
        headA = ListNode(listA_vals[0])
        curr = headA
        for i in range(1, skipA):
            curr.next = ListNode(listA_vals[i])
            curr = curr.next
        tailA = curr

    # 创建链表B的独有部分
    if skipB == 0:
        headB = None
        tailB = None
    else:
        headB = ListNode(listB_vals[0])
        curr = headB
        for i in range(1, skipB):
            curr.next = ListNode(listB_vals[i])
            curr = curr.next
        tailB = curr

    # 创建相交部分
    if skipA < len(listA_vals):
        intersect = ListNode(listA_vals[skipA])
        curr = intersect
        for i in range(skipA + 1, len(listA_vals)):
            curr.next = ListNode(listA_vals[i])
            curr = curr.next

        # 连接
        if tailA:
            tailA.next = intersect
        else:
            headA = intersect

        if tailB:
            tailB.next = intersect
        else:
            headB = intersect

    return headA, headB


# ✅ 测试
listA, listB = create_intersected_lists([4,1,8,4,5], [5,6,1,8,4,5], 2, 3)
result = get_intersection_node(listA, listB)
print(result.val if result else None)  # 期望输出: 8
```

### 复杂度分析

- **时间复杂度**:O(m + n) - 每个指针最多走m+n步
- **空间复杂度**:O(1) - 只用了两个指针

### 优缺点

- ✅ 空间O(1),最优解
- ✅ 代码简洁优雅
- ✅ 数学推导巧妙
- ❌ 理解需要一点数学直觉

---

## ⚡ 解法二:哈希表(简单直接)

### Python代码

```python
def get_intersection_node_hashset(headA: ListNode, headB: ListNode) -> ListNode:
    """
    解法二:哈希表
    思路:遍历A存入集合,遍历B查找第一个在集合中的节点
    """
    visited = set()
    curr = headA

    # 遍历A,存入集合
    while curr:
        visited.add(curr)
        curr = curr.next

    # 遍历B,查找第一个在集合中的节点
    curr = headB
    while curr:
        if curr in visited:
            return curr
        curr = curr.next

    return None
```

### 复杂度分析

- **时间复杂度**:O(m + n)
- **空间复杂度**:O(m) - 集合存储链表A的节点

---

## 🚀 解法三:长度差法

### 思路

1. 先遍历两个链表,计算长度 `lenA` 和 `lenB`
2. 让长链表的指针先走 `|lenA - lenB|` 步
3. 然后两个指针同时前进,第一个相同的节点就是交点

### Python代码

```python
def get_intersection_node_length(headA: ListNode, headB: ListNode) -> ListNode:
    """
    解法三:长度差法
    思路:长链表先走差值步,然后同步前进
    """
    # 计算长度
    def get_length(head):
        length = 0
        while head:
            length += 1
            head = head.next
        return length

    lenA = get_length(headA)
    lenB = get_length(headB)

    # 让长链表先走差值步
    pA, pB = headA, headB
    if lenA > lenB:
        for _ in range(lenA - lenB):
            pA = pA.next
    else:
        for _ in range(lenB - lenA):
            pB = pB.next

    # 同步前进,找交点
    while pA and pB:
        if pA == pB:
            return pA
        pA = pA.next
        pB = pB.next

    return None
```

### 复杂度分析

- **时间复杂度**:O(m + n)
- **空间复杂度**:O(1)

---

## 📊 解法对比

| 维度 | 解法一:等距法 | 解法二:哈希表 | 解法三:长度差 |
|------|-------------|-------------|-------------|
| 时间复杂度 | O(m+n) | O(m+n) | O(m+n) |
| 空间复杂度 | O(1) ⭐ | O(m) | O(1) ⭐ |
| 代码难度 | 简单 | 简单 | 中等 |
| 面试推荐 | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| 优点 | 最简洁优雅 | 思路直接 | 逻辑清晰 |

**面试建议**:首选解法一(等距法),代码最简洁,数学推导优雅,是面试官期待的最优解!

---

## 🎤 面试现场

**面试官**:请找出两个链表的相交节点。

**你**:好的,我的思路是用**双指针等距法**:

两个指针pA和pB分别从headA和headB出发。pA走完A后转到B,pB走完B后转到A。如果有交点,两者会在交点相遇;如果无交点,两者会同时到达null。

数学原理:设A独有部分长度a,B独有部分长度b,相交部分长度c。pA走a+c+b步,pB走b+c+a步,距离相等,会在交点相遇。

时间O(m+n),空间O(1)。

**面试官**:很好,请写代码。

**你**:(边写边说)两个指针从两个head开始...while循环直到相等...pA走到末尾后转到headB,pB走到末尾后转到headA...

**面试官**:为什么无交点时会同时到null?

**你**:无交点时相交部分长度c=0,pA走a+b步到null,pB走b+a步到null,距离相等,同时到达null。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果链表有环呢?" | 题目假设无环。如果有环,需要先判环(第26课),处理更复杂 |
| "能否只遍历一次?" | 解法一本质是遍历两次(A+B和B+A),但已经是最优。长度差法也需要先遍历计算长度 |
| "如果要求返回交点后的所有节点?" | 找到交点后,继续遍历返回列表即可 |
| "两个链表完全相同怎么办?" | 会在headA/headB处相遇,正确返回headA |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:三元表达式切换链表
pA = pA.next if pA else headB

# 技巧2:while循环条件
while pA != pB:  # 不相等继续走,相等(交点或null)退出
    pass

# 技巧3:集合判断节点引用
if node in visited:  # 判断引用相等,不是值相等
    pass
```

### 算法模式卡片 📐

- **模式名称**:双指针等距法
- **适用条件**:两个链表/序列找相交点、同步点
- **识别关键词**:"相交链表"、"找交点"、"汇合点"
- **模板代码**:
```python
def find_intersection_template(headA, headB):
    """双指针等距法通用模板"""
    pA, pB = headA, headB
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    return pA  # 交点或null
```

### 易错点 ⚠️

1. **判断值相等而非引用相等**
   - ❌ 错误:`if pA.val == pB.val`
   - ✅ 正确:`if pA == pB`(判断节点引用)

2. **转换链表的条件写错**
   - ❌ 错误:`pA = headB if not pA else pA.next`(顺序反了)
   - ✅ 正确:`pA = pA.next if pA else headB`

3. **无交点时死循环**
   - ❌ 错误:`while pA and pB:`(无交点时死循环)
   - ✅ 正确:`while pA != pB:`(无交点时同时到null退出)

---

## 🏋️ 举一反三

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 141. 环形链表 | Easy | 快慢指针 | 第26课,判断链表是否有环 |
| LeetCode 142. 环形链表II | Medium | Floyd判环 | 第27课,找环入口,数学推导 |
| LeetCode 19. 删除倒数第N个节点 | Medium | 快慢指针 | 快指针先走N步,然后同步前进 |
| LeetCode 876. 链表的中间节点 | Easy | 快慢指针 | fast走2步,slow走1步,fast到末尾时slow在中间 |

---

## 📝 课后小测

**题目**:给定链表头节点和整数n,删除倒数第n个节点。(LeetCode 19)

<details>
<summary>💡 提示</summary>

用快慢指针:
1. fast先走n步
2. fast和slow同时走,fast到末尾时slow在倒数第n+1个节点
3. 删除slow.next

注意:用虚拟头节点处理删除head的情况

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def remove_nth_from_end(head: ListNode, n: int) -> ListNode:
    dummy = ListNode(0, head)
    fast = slow = dummy

    # fast先走n+1步
    for _ in range(n + 1):
        fast = fast.next

    # 同步前进
    while fast:
        fast = fast.next
        slow = slow.next

    # 删除slow.next
    slow.next = slow.next.next

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
