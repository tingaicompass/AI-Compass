# 📖 第27课:环形链表II

> **模块**:链表 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/linked-list-cycle-ii/
> **前置知识**:第26课(环形链表) - 必须先掌握!
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个链表的头节点 `head`,返回链表开始入环的第一个节点。如果链表无环,则返回 `null`。

**不允许修改**链表。

**示例 1:**
```
输入:head = [3,2,0,-4], pos = 1
输出:返回索引为1的节点(值为2)
解释:链表中有一个环,尾节点连接到第二个节点(索引1)

可视化:
  3 -> 2 -> 0 -> -4
       ↑__________↓
       环入口
```

**示例 2:**
```
输入:head = [1,2], pos = 0
输出:返回索引为0的节点(值为1)
解释:链表中有一个环,尾节点连接到第一个节点

  1 -> 2
  ↑____↓
  环入口
```

**示例 3:**
```
输入:head = [1], pos = -1
输出:返回 null
解释:链表中没有环
```

**约束条件:**
- 链表中节点的数目范围在 `[0, 10⁴]`
- `-10⁵ <= Node.val <= 10⁵`
- `pos` 为 `-1` 或者链表中的一个有效索引

**进阶**:你是否可以使用 `O(1)` 空间解决此题?

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 无环 | `[1,2,3], pos=-1` | `null` | 基本判断 |
| 自环 | `[1], pos=0` | `节点1` | 环入口就是head |
| 环在开头 | `[1,2], pos=0` | `节点1` | 入口在head |
| 环在中间 | `[1,2,3,4], pos=1` | `节点2` | 常见情况 |
| 环在末尾 | `[1,2,3], pos=2` | `节点3` | 入口在尾部 |
| 最大规模 | `n=10000` | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 接续第26课的跑道比喻:现在你不仅要判断是环形跑道,还要找到"跑道的起点"在哪里。
>
> 🐌 **笨办法**:你用纸笔记录每个同学经过的位置,第一个重复经过的位置就是环的起点。这需要额外的纸笔(空间)。
>
> 🚀 **聪明办法**:利用第26课的快慢指针相遇后,你让快同学停下,慢同学回到起点。然后两个同学**都以相同速度(每次1步)同时出发**。神奇的是,**他们第二次相遇的地点,正好就是环的起点!**
>
> **这背后有精妙的数学原理!**

### 关键洞察

**数学定理:设head到环入口距离为a,环入口到首次相遇点距离为b,环长为C。快慢指针相遇后,把一个指针移回head,两个指针每次都走1步,再次相遇点就是环入口。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:链表头节点 `head`
- **输出**:环入口节点(如果有环),或 `null`(无环)
- **限制**:不能修改链表,要求 O(1) 空间

### Step 2:先想笨办法(哈希表)

遍历链表,用集合记录访问过的节点,第一个重复的节点就是环入口。
- 时间复杂度 O(n),空间复杂度 O(n)

### Step 3:瓶颈分析 → 优化方向

能否利用快慢指针,在 O(1) 空间内找到环入口?
- 核心问题:"快慢指针相遇后,如何找到环入口?"
- 优化思路:"数学推导!相遇后的距离关系"

### Step 4:选择武器

- **方案1**:哈希表 - O(n)空间
- **方案2**:Floyd判环 + 数学推导 - O(1)空间,面试最优

> 🔑 **模式识别提示**:这道题是第26课的进阶版,必须掌握数学证明

---

## 🔑 解法一:Floyd判环 + 数学推导(推荐)

### 思路

分两个阶段:

**阶段1:快慢指针判环**(第26课的内容)
- slow每次走1步,fast每次走2步
- 如果有环,两者会在环内相遇

**阶段2:找环入口**(本课重点)
- 相遇后,把一个指针移回 `head`
- 两个指针**都以相同速度(每次1步)**前进
- 再次相遇的节点就是环入口

### 数学证明(重要!)

```
假设:
- head到环入口距离 = a
- 环入口到首次相遇点距离 = b
- 环长 = C

阶段1:首次相遇时
  slow走了: a + b
  fast走了: a + b + kC (k为fast在环内多走的圈数,k≥1)

因为fast速度是slow的2倍:
  2(a + b) = a + b + kC
  2a + 2b = a + b + kC
  a + b = kC
  a = kC - b  ← 关键等式!

阶段2:再次相遇
  ptr1从head走a步到达环入口
  ptr2从相遇点走kC-b步 = 绕k圈后退b步 = 也到达环入口

所以两者会在环入口相遇!
```

**图解证明**:

```
示例: head到环入口a=2, 环入口到相遇点b=3, 环长C=5

链表结构:
  0 -> 1 -> 2 -> 3 -> 4 -> 5 -> 6
       ↑    环入口     ↑    ↑
      head            相遇  |
                      点 ←--┘

阶段1:首次相遇
  slow: 走了 a+b = 2+3 = 5步
  fast: 走了 2(a+b) = 10步 = a+b+C = 2+3+5

阶段2:找入口
  ptr1从head走: 0 -> 1 -> 2 (a=2步到环入口)
  ptr2从相遇点走: 5 -> 6 -> 2 (kC-b = 5-3 = 2步到环入口)

相遇于节点2(环入口)!
```

### Python代码

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def detect_cycle(head: ListNode) -> ListNode:
    """
    解法一:Floyd判环 + 数学推导
    思路:快慢指针相遇后,一个回head,两者同速再次相遇点就是环入口
    """
    # 阶段1:快慢指针判环
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            # 阶段2:找环入口
            # 把一个指针移回head,两个都走1步
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr  # 返回环入口节点

    return None  # 无环


# ✅ 测试辅助函数(与第26课相同)
def create_cycle_list(values, pos):
    """创建带环的链表"""
    if not values:
        return None

    nodes = [ListNode(val) for val in values]
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]

    if pos >= 0:
        nodes[-1].next = nodes[pos]

    return nodes[0]


# ✅ 测试
head1 = create_cycle_list([3, 2, 0, -4], 1)
entry1 = detect_cycle(head1)
print(entry1.val if entry1 else None)  # 期望输出: 2

head2 = create_cycle_list([1, 2], 0)
entry2 = detect_cycle(head2)
print(entry2.val if entry2 else None)  # 期望输出: 1

head3 = create_cycle_list([1], -1)
entry3 = detect_cycle(head3)
print(entry3.val if entry3 else None)  # 期望输出: None
```

### 复杂度分析

- **时间复杂度**:O(n)
  - 阶段1:快慢指针相遇,最多O(n)
  - 阶段2:找入口,最多O(n)
  - 总计O(n)
- **空间复杂度**:O(1) - 只用了常数个指针

### 优缺点

- ✅ 空间 O(1),最优解
- ✅ 数学推导优雅,面试亮点
- ✅ 不修改链表结构
- ❌ 数学证明需要理解透彻

---

## ⚡ 解法二:哈希表(简单直接)

### 优化思路

遍历链表,用集合记录访问过的节点,第一个重复的节点就是环入口。

### Python代码

```python
def detect_cycle_hashset(head: ListNode) -> ListNode:
    """
    解法二:哈希表
    思路:第一个重复访问的节点就是环入口
    """
    visited = set()
    curr = head

    while curr:
        if curr in visited:
            return curr  # 第一个重复节点 = 环入口
        visited.add(curr)
        curr = curr.next

    return None  # 无环
```

### 复杂度分析

- **时间复杂度**:O(n)
- **空间复杂度**:O(n) - 集合存储节点

---

## 📊 解法对比

| 维度 | 解法一:Floyd + 数学 | 解法二:哈希表 |
|------|-------------------|-------------|
| 时间复杂度 | O(n) | O(n) |
| 空间复杂度 | O(1) ⭐ | O(n) |
| 代码难度 | 中等(需要理解数学) | 简单 |
| 面试推荐 | ⭐⭐⭐ | ⭐⭐ |
| 适用场景 | 追求空间最优 | 快速实现 |

**面试建议**:
1. **首选解法一**:展示对Floyd算法的深入理解,这是面试官期待看到的
2. **可以先提解法二**:说"最简单的是哈希表",然后主动说"我们可以用Floyd算法优化到O(1)空间"
3. **一定要能解释数学推导**:这是区分普通候选人和优秀候选人的关键!

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请找出链表中环的入口节点,如果无环返回null。

**你**:(审题10秒)好的,这是第26课判环问题的进阶版。

我的思路是用**Floyd判环算法的扩展版本**:
1. 阶段1:快慢指针判环,找到相遇点
2. 阶段2:把一个指针移回head,两个指针都走1步,再次相遇点就是环入口

这背后有数学证明:设head到环入口距离为a,环入口到相遇点距离为b,环长为C。快慢指针相遇时,slow走了a+b,fast走了a+b+kC。因为fast速度是slow的2倍,所以 2(a+b) = a+b+kC,推导出 a = kC - b。这意味着从head走a步 = 从相遇点走kC-b步,都会到达环入口。

时间O(n),空间O(1)。

**面试官**:很好,请写代码。

**你**:(边写边说)我先用快慢指针判环...如果相遇,进入阶段2...把一个指针移回head,两个都走1步...相遇点就是环入口...

(写完代码)

**面试官**:为什么从head走a步等于从相遇点走kC-b步?

**你**:因为kC-b意思是"绕k圈再退b步"。在环形结构中,绕整数圈等于没动,所以kC-b等价于-b,也就是从相遇点往回退b步。而相遇点距离环入口恰好是b步,所以退b步正好到环入口。从head走a步也到环入口,所以两者会在环入口相遇。

**面试官**:如果环很长,会超时吗?

**你**:不会。阶段1最多O(n),阶段2最多O(n),总计O(n),非常高效。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如何计算环的长度?" | 找到环入口后,从入口出发绕一圈回到入口,计数即可。时间O(C),C为环长 |
| "如果链表有多个环呢?" | 单链表不可能有多个环(每个节点只有一个next),这是数据结构的性质 |
| "能否用递归实现?" | 可以,但递归深度O(n),空间复杂度O(n),不如迭代的O(1)空间 |
| "k一定≥1吗?" | 是的。快指针比慢指针快,当慢指针刚进入环时,快指针已经在环内了,至少会多走一圈,所以k≥1 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:Floyd判环找入口模板
def find_cycle_entry(head):
    # 阶段1:判环
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # 阶段2:找入口
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr
    return None

# 技巧2:节点相等性判断
if node1 == node2:  # 判断是否同一个节点(引用相等)
    pass

if node1.val == node2.val:  # 判断节点值是否相等
    pass

# 技巧3:集合判断节点
visited = set()
if curr in visited:  # O(1)时间复杂度
    return curr
```

### 💡 底层原理(选读)

> **数学证明的直观理解?**
>
> 想象一条跑道:
> - 直道部分长度 = a(head到环入口)
> - 环形部分周长 = C
> - 快慢指针在环上距离入口b的地方相遇
>
> **关键洞察**:
> - slow走的路程 = a + b(刚好进入环并走了b步)
> - fast走的路程 = a + b + kC(进入环后多绕了k圈,然后走b步)
> - fast速度是slow的2倍:2(a+b) = a+b+kC
> - 化简:a = kC - b
>
> **为什么再次相遇在入口?**
> - ptr从head走a步 → 到达入口
> - slow从相遇点走a步 = 走kC-b步 = 绕k圈后退b步 → 也到达入口
>
> **为什么k≥1?**
> - 当slow刚进入环(走了a步)时,fast已经走了2a步
> - fast在环内的位置 = (2a - a) % C = a % C
> - slow和fast都在环内,fast追slow至少要走一圈,所以k≥1

### 算法模式卡片 📐

- **模式名称**:Floyd判环 + 找环入口
- **适用条件**:判断链表是否有环,找环入口,计算环长
- **识别关键词**:"环入口"、"环的起点"、"第一个入环节点"
- **模板代码**:
```python
def detect_cycle_template(head: ListNode) -> ListNode:
    """Floyd找环入口的通用模板"""
    # 阶段1:快慢指针判环
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # 阶段2:找环入口
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr  # 环入口节点
    return None  # 无环
```

### 易错点 ⚠️

1. **阶段2中slow和fast都走1步**
   - ❌ 错误:阶段2中fast还是走2步 → 无法相遇
   - ✅ 正确:阶段2中两个指针**都走1步**

2. **忘记处理无环情况**
   - ❌ 错误:阶段1不判断,直接进入阶段2 → 空指针错误
   - ✅ 正确:阶段1 while循环结束后返回 None

3. **数学推导理解错误**
   - ❌ 错误:认为相遇点就是环入口
   - ✅ 正确:相遇点不一定是入口,需要阶段2重新走

4. **ptr初始化错误**
   - ❌ 错误:`ptr = head.next` → 会跳过head是入口的情况
   - ✅ 正确:`ptr = head`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:内存泄漏检测**
  - 程序中对象引用形成环时,垃圾回收器无法回收,导致内存泄漏。用Floyd算法检测引用图中的环

- **场景2:分布式系统死锁定位**
  - 多个服务之间的调用依赖形成环时,会造成死锁。找到环的入口可以定位死锁源头

- **场景3:区块链分叉检测**
  - 区块链分叉时,需要找到分叉点(环入口),确定哪条链是主链

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 287. 寻找重复数 | Medium | Floyd判环变体 | 把数组看作链表,nums[i]指向nums[nums[i]],找环入口即找重复数 |
| LeetCode 160. 相交链表 | Easy | 双指针 | 两个链表相交,交点类似"环入口",用类似思想 |
| LeetCode 202. 快乐数 | Easy | Floyd判环 | 数字变换过程成环则非快乐数,找环入口判断是否为1 |
| LeetCode 457. 环形数组循环 | Medium | Floyd判环 | 数组中的前进/后退指针,判断是否有环 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个包含 `n + 1` 个整数的数组 `nums`,其中每个整数在 `[1, n]` 范围内。证明至少存在一个重复的整数,找出这个重复的数。要求:不修改数组,O(1)空间。(LeetCode 287)

例如:`nums = [1,3,4,2,2]` → 返回 `2`

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

**关键洞察**:把数组看作链表!
- 下标i → 节点i
- 值nums[i] → next指针(指向节点nums[i])

因为有n+1个数,值在[1,n]范围,必有重复(鸽巢原理)。重复的数会导致多个节点指向同一个节点,形成环!环的入口就是重复的数。

用Floyd判环算法找环入口即可。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def find_duplicate(nums: list[int]) -> int:
    """
    Floyd判环找重复数
    思路:把数组看作链表,nums[i]指向nums[nums[i]]
    """
    # 阶段1:快慢指针判环
    slow = fast = 0  # 从下标0开始(虚拟头节点)

    while True:
        slow = nums[slow]           # 走1步
        fast = nums[nums[fast]]     # 走2步

        if slow == fast:
            break  # 相遇

    # 阶段2:找环入口(重复的数)
    ptr = 0
    while ptr != slow:
        ptr = nums[ptr]
        slow = nums[slow]

    return ptr  # 环入口 = 重复的数


# 测试
print(find_duplicate([1, 3, 4, 2, 2]))  # 输出: 2
print(find_duplicate([3, 1, 3, 4, 2]))  # 输出: 3
```

**核心思路**:
1. 数组变链表:`nums = [1,3,4,2,2]`
   - 0 → 1 → 3 → 2 → 4 → 2(形成环,入口是2)
2. Floyd判环找入口 → 重复的数就是2

**为什么环入口是重复数?**
- 重复数被多个位置的值指向
- 这导致多条路径汇聚到重复数
- 重复数就是环的入口

时间O(n),空间O(1),不修改数组,完美满足要求!

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
