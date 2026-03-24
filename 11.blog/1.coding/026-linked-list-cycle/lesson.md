> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第26课:环形链表

> **模块**:链表 | **难度**:Easy ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/linked-list-cycle/
> **前置知识**:第24课(反转链表) - 快慢指针思想
> **预计学习时间**:15分钟

---

## 🎯 题目描述

给你一个链表的头节点 `head`,判断链表中是否有环。

如果链表中存在环,则返回 `true`;否则返回 `false`。

链表中有环的定义:链表中某个节点的 `next` 指针指向链表中之前出现过的节点,形成一个环。

**示例 1:**
```
输入:head = [3,2,0,-4], pos = 1
输出:true
解释:链表中有一个环,尾节点连接到索引为1的节点(值为2)

可视化:
  3 -> 2 -> 0 -> -4
       ↑__________↓
       (形成环)
```

**示例 2:**
```
输入:head = [1,2], pos = 0
输出:true
解释:链表中有一个环,尾节点连接到索引为0的节点

  1 -> 2
  ↑____↓
```

**示例 3:**
```
输入:head = [1], pos = -1
输出:false
解释:链表中没有环
```

**约束条件:**
- 链表中节点的数目范围是 `[0, 10⁴]`
- `-10⁵ <= Node.val <= 10⁵`
- `pos` 为 `-1` 或者链表中的一个有效索引

**进阶**:你能用 O(1) 空间复杂度解决此题吗?

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空链表 | `[]` | `false` | 空指针处理 |
| 单节点无环 | `[1], pos=-1` | `false` | 基本情况 |
| 单节点有环 | `[1], pos=0` | `true` | 自环 |
| 两节点有环 | `[1,2], pos=0` | `true` | 小环 |
| 多节点无环 | `[1,2,3], pos=-1` | `false` | 正常链表 |
| 多节点有环 | `[1,2,3], pos=1` | `true` | 环在中间 |
| 最大规模 | `n=10000` | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在操场的环形跑道上,有两个同学在跑步:一个跑得慢(慢指针),一个跑得快(快指针,速度是慢的2倍)。
>
> 🐌 **笨办法**:你用纸笔记录每个同学经过的位置。如果快同学经过了之前记录过的位置,说明是环形跑道(有环);如果快同学跑到了终点(null),说明不是环形跑道。这样需要额外的纸笔(空间)来记录。
>
> 🚀 **聪明办法**:你让两个同学同时出发,快同学每次跑2步,慢同学每次跑1步。**如果是环形跑道,快同学一定会追上慢同学(快慢指针相遇);如果不是环形,快同学会先到达终点(null)。** 这个方法不需要额外记录,只需要观察两个同学!
>
> **这就是著名的 Floyd判环算法(龟兔赛跑算法)!**

### 关键洞察

**用快慢双指针:慢指针每次走1步,快指针每次走2步。如果有环,快指针一定会追上慢指针;如果无环,快指针会先到达null。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:链表的头节点 `head`
- **输出**:布尔值,是否有环
- **限制**:能否用 O(1) 空间?

### Step 2:先想笨办法(哈希表记录)

用一个集合(set)记录访问过的节点,遍历链表:
- 如果当前节点已在集合中,说明有环
- 如果遍历到 `null`,说明无环

时间复杂度 O(n),空间复杂度 O(n)

### Step 3:瓶颈分析 → 优化方向

哈希表需要额外 O(n) 空间,能否优化到 O(1)?
- 核心问题:"不用额外空间,如何检测环?"
- 优化思路:"快慢指针!如果有环,快的一定能追上慢的"

### Step 4:选择武器

- **方案1**:哈希表 - 简单直接,O(n)空间
- **方案2**:快慢指针(Floyd判环) - O(1)空间,面试最优

> 🔑 **模式识别提示**:当题目要求"判断链表是否有环"时,首选"Floyd快慢指针"

---

## 🔑 解法一:快慢指针(Floyd判环算法,推荐)

### 思路

用两个指针 `slow` 和 `fast`:
- `slow` 每次走1步
- `fast` 每次走2步

**关键定理**:
- 如果有环,fast 一定会在环内追上 slow(相遇)
- 如果无环,fast 会先到达 `null`

### 图解过程

```
示例: 3 -> 2 -> 0 -> -4
            ↑__________↓

初始化:
  slow = 3
  fast = 3

第1步: slow走1步, fast走2步
  slow = 2
  fast = 0

第2步: slow走1步, fast走2步
  slow = 0
  fast = 2 (fast在环内循环了)

第3步: slow走1步, fast走2步
  slow = -4
  fast = -4  ✓ 相遇!

返回 true
```

**为什么一定会相遇?**

```
数学证明:
假设环的长度为 C,slow 进入环时 fast 在环中的位置距离 slow 为 D

每一步:
  - slow 前进 1
  - fast 前进 2
  - 相对速度:fast 每步比 slow 多走 1

所以 fast 每步接近 slow 一个单位,最多 D 步后就会相遇

无论环多大,fast 一定能在环内追上 slow!
```

### Python代码

```python
# 定义链表节点(与之前课程相同)
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def has_cycle(head: ListNode) -> bool:
    """
    解法一:快慢指针(Floyd判环算法)
    思路:slow每次1步,fast每次2步,有环则相遇
    """
    if not head or not head.next:
        return False

    slow = head
    fast = head

    # 快指针每次走2步,慢指针每次走1步
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        # 如果相遇,说明有环
        if slow == fast:
            return True

    # 快指针到达末尾,说明无环
    return False


# ✅ 测试辅助函数
def create_cycle_list(values, pos):
    """创建带环的链表,pos是环的入口索引(-1表示无环)"""
    if not values:
        return None

    # 创建节点
    nodes = [ListNode(val) for val in values]

    # 连接节点
    for i in range(len(nodes) - 1):
        nodes[i].next = nodes[i + 1]

    # 创建环
    if pos >= 0:
        nodes[-1].next = nodes[pos]

    return nodes[0]


# ✅ 测试
head1 = create_cycle_list([3, 2, 0, -4], 1)
print(has_cycle(head1))  # 期望输出: True

head2 = create_cycle_list([1, 2], 0)
print(has_cycle(head2))  # 期望输出: True

head3 = create_cycle_list([1], -1)
print(has_cycle(head3))  # 期望输出: False
```

### 复杂度分析

- **时间复杂度**:O(n) - 最坏情况下遍历所有节点
  - 无环:fast 走到末尾,最多 n/2 次循环
  - 有环:slow 和 fast 在环内相遇,最多 n 次循环(slow 走一圈)
- **空间复杂度**:O(1) - 只用了两个指针变量

### 优缺点

- ✅ 空间 O(1),最优解
- ✅ 经典算法,面试必会
- ✅ 思路巧妙,易于解释
- ❌ 理解稍有难度(需要理解相对速度)

---

## ⚡ 解法二:哈希表记录(简单直接)

### 优化思路

用集合记录访问过的节点,如果遇到重复节点则有环。

### Python代码

```python
def has_cycle_hashset(head: ListNode) -> bool:
    """
    解法二:哈希表
    思路:记录访问过的节点,遇到重复则有环
    """
    visited = set()
    curr = head

    while curr:
        if curr in visited:
            return True
        visited.add(curr)
        curr = curr.next

    return False
```

### 复杂度分析

- **时间复杂度**:O(n) - 遍历链表
- **空间复杂度**:O(n) - 集合存储n个节点

---

## 🚀 解法三:修改链表(破坏性方法,不推荐)

### 优化思路

遍历链表,把访问过的节点的 `next` 指向一个特殊值(如自己)。如果遇到已修改的节点,说明有环。

**注意:这种方法会破坏原链表结构,实际中不推荐!**

### Python代码

```python
def has_cycle_modify(head: ListNode) -> bool:
    """
    解法三:修改节点指针(破坏性,不推荐)
    思路:把访问过的节点next指向自己,遇到则有环
    """
    curr = head

    while curr:
        # 如果next指向自己,说明已访问过(有环)
        if curr.next == curr:
            return True

        # 保存下一个节点
        next_node = curr.next

        # 修改当前节点的next指向自己
        curr.next = curr

        # 移动到下一个节点
        curr = next_node

    return False
```

**⚠️ 警告**:这种方法会破坏链表,面试中不推荐使用!

---

## 🐍 Pythonic 写法

快慢指针的简洁版本:

```python
def has_cycle_pythonic(head: ListNode) -> bool:
    """Pythonic 快慢指针"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow == fast:
            return True
    return False
```

> ⚠️ **面试建议**:直接写解法一即可,简洁且易懂。Pythonic版本差异不大。

---

## 📊 解法对比

| 维度 | 解法一:快慢指针 | 解法二:哈希表 | 解法三:修改链表 |
|------|---------------|-------------|---------------|
| 时间复杂度 | O(n) | O(n) | O(n) |
| 空间复杂度 | O(1) ⭐ | O(n) | O(1) |
| 面试推荐 | ⭐⭐⭐ | ⭐⭐ | ❌ |
| 优点 | 空间最优,经典算法 | 思路简单 | 空间O(1) |
| 缺点 | 理解稍难 | 空间开销大 | 破坏原链表 |

**面试建议**:
1. **首选解法一(快慢指针)**:这是面试官期待的最优解,体现算法功底
2. **可以先提解法二**:展示思路,然后说"用快慢指针可以优化到O(1)空间"
3. **避免解法三**:破坏性方法,工程中不可接受

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请判断一个链表是否有环。

**你**:(审题10秒)好的,我理解了。链表有环是指某个节点的next指针指向之前出现过的节点,形成循环。

我有两种思路:
1. **哈希表**:记录访问过的节点,如果遇到重复节点说明有环。时间O(n),空间O(n)。
2. **快慢指针(Floyd判环算法)**:用两个指针,slow每次走1步,fast每次走2步。如果有环,fast一定会在环内追上slow(相遇);如果无环,fast会先到达null。时间O(n),空间O(1)。

我推荐用第二种,空间更优。

**面试官**:很好,请写快慢指针的代码。

**你**:(边写边说)我用两个指针slow和fast,都从head开始。while循环条件是 `fast and fast.next`,因为fast每次走2步,要确保fast.next存在...每次slow走1步,fast走2步...如果slow==fast说明相遇,返回true...循环结束说明无环,返回false...

(写完代码)

**面试官**:为什么fast一定能追上slow?

**你**:这是因为相对速度。假设slow进入环时,fast在环中距离slow为D。每一步,fast比slow多走1步,也就是每步接近slow一个单位。所以最多D步后,fast就会追上slow。无论环多大,fast一定能在环内追上slow。

**面试官**:如果fast每次走3步呢?

**你**:也可以判环,但不一定在第一圈就相遇,可能需要多绕几圈。fast走2步是最优的:既能保证相遇,又能尽快相遇。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如何找到环的入口节点?" | 这是LeetCode 142。相遇后,把一个指针移回head,两个指针每次都走1步,再次相遇点就是环入口(数学证明:相遇点到入口的距离 = head到入口的距离) |
| "如何计算环的长度?" | 相遇后,固定一个指针,另一个继续走,计数直到再次相遇,计数值就是环长 |
| "能否用递归实现?" | 可以,但递归深度为O(n),空间复杂度O(n),不如快慢指针的O(1)空间 |
| "如果链表很长,会超时吗?" | 不会。时间复杂度O(n),即使链表有10000个节点,fast最多走2n步,非常快 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:快慢指针模板
slow = fast = head
while fast and fast.next:
    slow = slow.next
    fast = fast.next.next
    if slow == fast:
        return True
return False

# 技巧2:同时赋值(元组解包)
slow, fast = slow.next, fast.next.next

# 技巧3:while条件的短路特性
while fast and fast.next:  # fast为None时不会检查fast.next
    pass

# 技巧4:集合判断节点
visited = set()
if node in visited:  # O(1)时间复杂度
    pass
```

### 💡 底层原理(选读)

> **Floyd判环算法的数学原理?**
>
> 假设链表总长n,环的起点距离head为a,环的长度为C。
>
> slow和fast相遇时:
> - slow走了 s 步
> - fast走了 2s 步(因为速度是slow的2倍)
> - fast比slow多走了一圈或多圈环:2s - s = kC (k为圈数)
> - 所以 s = kC
>
> 这就是为什么fast一定能追上slow:fast在环内的速度优势,会让它不断接近slow,直到相遇。
>
> **为什么fast走2步最优?**
>
> - 走1步:退化为普通遍历,无法判环
> - 走2步:相对速度为1,最快相遇
> - 走3步或更多:相对速度更大,但可能"跳过"slow,需要多绕几圈
>
> **空间复杂度为什么是O(1)?**
>
> 只用了slow和fast两个指针变量,无论链表多长,都只需要这两个变量,所以是O(1)常数空间。

### 算法模式卡片 📐

- **模式名称**:Floyd判环算法(快慢指针)
- **适用条件**:判断链表/序列是否有环,找环的入口,计算环长
- **识别关键词**:"判断是否有环"、"环形链表"、"循环检测"
- **模板代码**:
```python
def detect_cycle_template(head: ListNode) -> bool:
    """Floyd判环算法通用模板"""
    slow = fast = head

    while fast and fast.next:
        slow = slow.next        # 慢指针走1步
        fast = fast.next.next   # 快指针走2步

        if slow == fast:        # 相遇则有环
            return True

    return False  # fast到达末尾,无环
```

### 易错点 ⚠️

1. **while条件写错导致空指针**
   - ❌ 错误:`while fast:` → fast.next.next可能空指针
   - ✅ 正确:`while fast and fast.next:`

2. **初始化错误**
   - ❌ 错误:`slow = head, fast = head.next` → 相遇条件复杂
   - ✅ 正确:`slow = fast = head` → 简化逻辑

3. **相遇判断位置错误**
   - ❌ 错误:在移动指针前判断 → 初始就判断会误判
   - ✅ 正确:在移动指针后判断

4. **空链表未处理**
   - ❌ 错误:直接 `slow = head.next` → head为None时报错
   - ✅ 正确:先判断 `if not head or not head.next: return False`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:操作系统死锁检测**
  - 进程等待资源时可能形成环形等待(死锁),用Floyd算法检测资源依赖图中的环

- **场景2:网络路由环路检测**
  - 网络数据包在路由器间传输,如果配置错误可能形成环路,导致数据包无限循环。用TTL(Time To Live)和环路检测算法避免

- **场景3:垃圾回收(GC)**
  - Java/Python的垃圾回收器需要检测对象引用图中的环,判断哪些对象可以回收

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 142. 环形链表II | Medium | Floyd判环+数学 | 找环的入口节点,相遇后一个指针回到head |
| LeetCode 287. 寻找重复数 | Medium | Floyd判环变体 | 把数组看作链表,值作为next指针,判环找重复 |
| LeetCode 202. 快乐数 | Easy | Floyd判环 | 数字变换过程可能成环,用快慢指针检测 |
| LeetCode 876. 链表的中间节点 | Easy | 快慢指针 | fast走2步到末尾时,slow正好在中间 |
| LeetCode 160. 相交链表 | Easy | 双指针 | 两个指针分别遍历两个链表,会在交点相遇 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个链表,返回链表开始入环的第一个节点。如果链表无环,返回 `null`。(LeetCode 142)

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

1. 先用快慢指针判断是否有环,记录相遇点
2. 相遇后,把一个指针移回head
3. 两个指针每次都走1步,再次相遇的节点就是环入口

**数学证明**:设head到环入口距离为a,环入口到相遇点距离为b,环长为C。
- slow走了 a+b
- fast走了 a+b+kC (k为圈数)
- 因为fast速度是slow的2倍:2(a+b) = a+b+kC → a = kC - b
- 所以从head走a步 = 从相遇点走kC-b步(即绕k圈后退b步,正好到环入口)

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def detect_cycle(head: ListNode) -> ListNode:
    """
    返回环的入口节点,无环返回None
    """
    # 阶段1:快慢指针判环
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            # 阶段2:找环入口
            # 把一个指针移回head,两个都走1步,相遇点就是入口
            ptr = head
            while ptr != slow:
                ptr = ptr.next
                slow = slow.next
            return ptr  # 返回环入口节点

    return None  # 无环


# 测试
head = create_cycle_list([3, 2, 0, -4], 1)
entry = detect_cycle(head)
print(entry.val if entry else None)  # 输出: 2 (环入口节点的值)
```

核心思路:
1. 快慢指针相遇后,说明有环
2. 一个指针回到head,两个指针每次都走1步
3. 再次相遇点就是环入口(数学证明见提示)

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

---

> 如果这篇内容对你有帮助，推荐收藏 AI Compass：https://github.com/tingaicompass/AI-Compass
> 更多系统化题解、编程基础和 AI 学习资料都在这里，后续复习和拓展会更省时间。
