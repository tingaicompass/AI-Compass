> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第42课:对称二叉树

> **模块**:二叉树 | **难度**:Easy ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/symmetric-tree/
> **前置知识**:第40课(二叉树最大深度)、第41课(翻转二叉树)
> **预计学习时间**:20分钟

---

## 🎯 题目描述

给定一个二叉树的根节点,判断该树是否轴对称。也就是说,树的左子树和右子树是否互为镜像。

**示例:**
```
对称树:        非对称树:
    1              1
   / \            / \
  2   2          2   2
 / \ / \          \   \
3  4 4  3          3   3

输入:root = [1,2,2,3,4,4,3]
输出:true
解释:左子树[2,3,4]和右子树[2,4,3]镜像对称

输入:root = [1,2,2,null,3,null,3]
输出:false
解释:左子树和右子树结构不对称
```

**约束条件:**
- 树中节点数量范围是 [1, 1000]
- -100 <= Node.val <= 100

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单节点 | root=[1] | true | 基本功能 |
| 两节点对称 | root=[1,2,2] | true | 简单对称 |
| 两节点不对称值 | root=[1,2,3] | false | 值不同 |
| 结构不对称 | root=[1,2,2,3,null,null,3] | false | 结构不同 |
| 完全对称树 | root=[1,2,2,3,4,4,3] | true | 完整对称 |
| 多层不对称 | root=[1,2,2,null,3,3] | false | 深层判断 |

---

## 💡 思路引导

### 生活化比喻
> 想象你站在一面镜子前,要判断自己是否完全对称(比如检查面部对称性)。
>
> 🐌 **笨办法**:用相机拍下左半边脸,水平翻转,然后逐像素比对右半边是否一样。这需要额外空间存储翻转后的图像。
>
> 🚀 **聪明办法**:用两个手指,一个从左眼开始,一个从右眼开始,同时向外移动,检查对应位置是否镜像相同。左手往下移一格,右手也往下移一格;左手往左移,右手就往右移。这就是**双指针同步遍历**的思想,无需额外空间。

### 关键洞察
**判断对称 = 递归判断左子树的左孩子是否等于右子树的右孩子,且左子树的右孩子等于右子树的左孩子**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树根节点 TreeNode
- **输出**:布尔值,表示是否对称
- **限制**:节点数≤1000,递归深度安全

### Step 2:先想笨办法(翻转后比较)
先翻转右子树,然后判断左子树和翻转后的右子树是否完全相同。
- 时间复杂度:O(n) — 翻转O(n)+比较O(n)
- 瓶颈在哪:需要先翻转,然后再遍历比较,两次遍历

### Step 3:瓶颈分析 → 优化方向
其实不需要真的翻转!可以边遍历边镜像比较。
- 核心问题:"如何同步遍历两棵子树并镜像比对?"
- 优化思路:定义一个辅助函数,接收两个节点,递归判断它们是否镜像

### Step 4:选择武器
- 选用:**递归双指针(镜像遍历)**
- 理由:用两个指针同步遍历左右子树,一个往左一个往右,天然镜像

> 🔑 **模式识别提示**:当题目涉及"对称"、"镜像",优先考虑"双指针镜像递归"

---

## 🏆 解法一:递归双指针(最优解)

### 思路
定义辅助函数`isMirror(left, right)`:
1. 两节点都为空 → 对称
2. 一个空一个非空 → 不对称
3. 值不同 → 不对称
4. 递归判断:`left.left`与`right.right`对称,且`left.right`与`right.left`对称

这是**双指针同步递归**:一个往外,一个往内,镜像前进。

### 图解过程

```
示例1:对称树
      1
     / \
    2   2
   / \ / \
  3  4 4  3

递归调用过程:
isMirror(root.left=2, root.right=2)
├─ 值相同 ✓
├─ isMirror(2的左=3, 2的右=3)
│  └─ 值相同,都是叶子 ✓
└─ isMirror(2的右=4, 2的左=4)
   └─ 值相同,都是叶子 ✓

返回 true ✅


示例2:不对称树
      1
     / \
    2   2
     \   \
      3   3

isMirror(root.left=2, root.right=2)
├─ 值相同 ✓
├─ isMirror(2的左=null, 2的右=3)
│  └─ 一个空一个非空 ✗
返回 false ✅
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def isSymmetric(root: Optional[TreeNode]) -> bool:
    """
    解法一:递归双指针(最优解)
    思路:定义镜像判断函数,同步遍历左右子树
    """
    def isMirror(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
        # 两个都为空,镜像
        if not left and not right:
            return True
        # 一个空一个非空,不镜像
        if not left or not right:
            return False
        # 值不同,不镜像
        if left.val != right.val:
            return False

        # 递归判断:左的左对右的右,左的右对右的左
        return (isMirror(left.left, right.right) and
                isMirror(left.right, right.left))

    # 空树是对称的
    if not root:
        return True

    # 判断左右子树是否镜像
    return isMirror(root.left, root.right)


# ✅ 测试
# 对称树:    1
#          / \
#         2   2
#        / \ / \
#       3  4 4  3
root1 = TreeNode(1)
root1.left = TreeNode(2, TreeNode(3), TreeNode(4))
root1.right = TreeNode(2, TreeNode(4), TreeNode(3))
print(isSymmetric(root1))  # 期望:True

# 不对称树:  1
#          / \
#         2   2
#          \   \
#           3   3
root2 = TreeNode(1)
root2.left = TreeNode(2, None, TreeNode(3))
root2.right = TreeNode(2, None, TreeNode(3))
print(isSymmetric(root2))  # 期望:False

# 边界测试
print(isSymmetric(TreeNode(1)))  # 期望:True(单节点)
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点访问恰好一次
  - 具体地说:1000个节点需要约500对镜像比较
- **空间复杂度**:O(h) — 递归栈深度等于树高h
  - 平衡树:O(log n),如1000个节点约10层
  - 最坏链状树:O(n),但对称树一般较平衡

### 优缺点
- ✅ 代码简洁,逻辑清晰
- ✅ 时间O(n)最优,一次遍历
- ✅ 空间O(h)最优,不需额外拷贝

---

## ⚡ 解法二:BFS层序遍历(迭代法)

### 优化思路
递归虽优雅,但有些场景要求迭代。改用队列,每次取出一对镜像节点比较,然后按镜像顺序加入下一层节点。

> 💡 **关键想法**:用队列存储成对的镜像节点,每次出队一对进行比较

### 图解过程

```
示例:  1
      / \
     2   2
    / \ / \
   3  4 4  3

BFS过程:
初始: queue=[(2, 2)]  # 左右子树根

Step1: 出队(2,2)
       值相同 ✓
       入队镜像对:(3,3), (4,4)
       queue=[(3,3), (4,4)]

Step2: 出队(3,3)
       值相同,都是叶子 ✓
       queue=[(4,4)]

Step3: 出队(4,4)
       值相同,都是叶子 ✓
       queue=[]

返回 true ✅
```

### Python代码

```python
from collections import deque


def isSymmetric_bfs(root: Optional[TreeNode]) -> bool:
    """
    解法二:BFS迭代
    思路:队列存储镜像节点对,逐对比较
    """
    if not root:
        return True

    # 队列存储成对的节点
    queue = deque([(root.left, root.right)])

    while queue:
        left, right = queue.popleft()

        # 都为空,继续
        if not left and not right:
            continue
        # 一个空一个非空,不对称
        if not left or not right:
            return False
        # 值不同,不对称
        if left.val != right.val:
            return False

        # 按镜像顺序加入下一层
        queue.append((left.left, right.right))  # 外侧对
        queue.append((left.right, right.left))  # 内侧对

    return True


# ✅ 测试
root1 = TreeNode(1)
root1.left = TreeNode(2, TreeNode(3), TreeNode(4))
root1.right = TreeNode(2, TreeNode(4), TreeNode(3))
print(isSymmetric_bfs(root1))  # 期望:True

root2 = TreeNode(1)
root2.left = TreeNode(2, None, TreeNode(3))
root2.right = TreeNode(2, None, TreeNode(3))
print(isSymmetric_bfs(root2))  # 期望:False
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点入队出队各一次
- **空间复杂度**:O(w) — w为树的最大宽度,完全树最坏O(n)

---

## 🐍 Pythonic 写法

利用递归简化:

```python
# 精简版(推荐)
def isSymmetric_compact(root: Optional[TreeNode]) -> bool:
    def mirror(t1, t2):
        if not t1 and not t2: return True
        if not t1 or not t2: return False
        return (t1.val == t2.val and
                mirror(t1.left, t2.right) and
                mirror(t1.right, t2.left))
    return mirror(root, root) if root else True

# 一行版(不推荐)
isSymmetric_oneline = lambda r: (lambda m: m(r,r) if r else True)(
    lambda l,r: not l and not r or l and r and l.val==r.val and ...)
```

> ⚠️ **面试建议**:先写清晰版本展示思路,通过后可以说"可以简化",但不要影响可读性。

---

## 📊 解法对比

| 维度 | 🏆 解法一:递归双指针 | 解法二:BFS迭代 |
|------|-------------------|--------------|
| 时间复杂度 | **O(n)** ← 最优 | O(n) |
| 空间复杂度 | **O(h) 约O(log n)** | O(w) 约O(n) |
| 代码难度 | **简单** | 中等 |
| 面试推荐 | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | **通用,最直观** | 避免递归要求 |
| 代码行数 | 约15行 | 约20行 |

**为什么解法一是最优解**:
- 时间O(n)已是理论最优(必须访问所有节点对)
- 空间O(log n)优于BFS的O(n)
- 双指针镜像递归思想清晰,面试中易讲解
- 代码简洁,易于理解和实现

**面试建议**:
1. 直接说出🏆最优解思路:"定义镜像判断函数,左的左对右的右,左的右对右的左"
2. 写代码时强调递归三个出口:"都空、一空、值不同"
3. 手动模拟一个对称树和非对称树的判断过程
4. 主动测试边界:"单节点返回true,空树返回true"
5. 如被问"能否迭代?",给出解法二BFS方案

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你判断一棵二叉树是否对称。

**你**:(审题15秒)好的,这道题要求判断树是否轴对称,也就是左右子树互为镜像。

我的思路是用**递归双指针**。定义一个辅助函数`isMirror(left, right)`,同时遍历左右子树,判断它们是否镜像。关键是镜像的定义:左的左对右的右,左的右对右的左。时间复杂度O(n),空间复杂度O(h)。

**面试官**:很好,请写代码。

**你**:(边写边说)
```python
def isSymmetric(root):
    def isMirror(left, right):
        # 三个递归出口
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        # 递归判断镜像
        return (isMirror(left.left, right.right) and
                isMirror(left.right, right.left))

    if not root:
        return True
    return isMirror(root.left, root.right)
```

**面试官**:为什么要用两个参数的辅助函数?

**你**:因为对称判断需要**同时遍历两棵子树**,而单个递归函数只能处理一个节点。辅助函数接收两个节点,可以同步递归,一个往左一个往右,实现镜像比对。这是"双指针同步递归"的思想。

**面试官**:测试一下?

**你**:用示例树[1,2,2,3,4,4,3]...
- `isMirror(左2, 右2)`:值相同
  - `isMirror(3, 3)`:都是叶子,返回true
  - `isMirror(4, 4)`:都是叶子,返回true
  - 都返回true,所以根节点对称 ✅

再测不对称的[1,2,2,null,3,null,3]:
- `isMirror(左2, 右2)`:值相同
  - `isMirror(左2的左null, 右2的右3)`:一空一非空,返回false ✗

边界:单节点树,左右都是null,返回true ✅

**面试官**:不错!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能否不用辅助函数?" | "可以用栈或队列迭代,但辅助函数最清晰,分离了'判断对称'和'判断镜像'两个逻辑" |
| "如果允许翻转节点呢?" | "那就变成'翻转等价'问题,需要额外判断不翻转的情况,复杂度变O(n²)" |
| "空间能O(1)吗?" | "不能,必须用栈(显式或递归)来同步遍历两个子树,无法纯迭代实现" |
| "这题和翻转二叉树什么关系?" | "可以先翻转右子树,再判断左右是否相同,但那样需要两次遍历,不如双指针一次完成" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:多条件短路判断 — and会短路,遇False立即停止
if not left and not right:  # 都空才继续
if not left or not right:   # 任意一个空就返回

# 技巧2:嵌套函数访问外层变量 — 闭包特性
def outer(root):
    def inner(node):
        return node.val  # 可访问root
    return inner(root)

# 技巧3:递归返回布尔值的优雅写法
return (cond1 and cond2 and cond3)  # 多条件同时满足
```

### 💡 底层原理(选读)

> **为什么对称判断需要"双指针"?**
>
> 对称性的本质是**同步比对两个镜像位置**。单指针只能遍历一个子树,无法知道镜像位置的节点是什么。
>
> 类比:判断字符串是否回文,为什么用双指针?
> ```python
> def is_palindrome(s):
>     left, right = 0, len(s) - 1
>     while left < right:
>         if s[left] != s[right]:
>             return False
>         left += 1
>         right -= 1
>     return True
> ```
> 树的对称判断是这个思想的递归版本:左指针往外走,右指针往内走,镜像前进。

> **递归vs迭代:谁的空间更优?**
>
> - 递归:栈深度=树高h,对称树通常较平衡,约O(log n)
> - 迭代(BFS):队列宽度=树的最大宽度w,完全树约O(n/2)
> - 所以对于对称树,递归通常空间更优!
>
> 但如果树极度偏斜(不对称),递归可能O(n),迭代可能O(1),这时迭代更优。

### 算法模式卡片 📐
- **模式名称**:双指针镜像递归
- **适用条件**:需要同步比对树的镜像位置
- **识别关键词**:"对称"、"镜像"、"回文"
- **模板代码**:
```python
def check_symmetric(root):
    def mirror(left, right):
        # 递归出口:都空/一空/值不同
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val != right.val:
            return False
        # 镜像递归:外对外,内对内
        return (mirror(left.left, right.right) and
                mirror(left.right, right.left))
    return mirror(root.left, root.right)
```

### 易错点 ⚠️
1. **递归顺序错误**
   - ❌ 错误:`mirror(left.left, right.left)` 不是镜像,是同侧
   - ✅ 正确:`mirror(left.left, right.right)` 外侧对外侧

2. **忘记处理空节点**
   - ❌ 错误:只判断`if left.val != right.val`,在None上调用.val会报错
   - ✅ 正确:先判断是否为空,再比较值

3. **单节点返回值错误**
   - ❌ 错误:认为单节点不对称返回false
   - ✅ 正确:单节点左右都是null,镜像,返回true

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:图形对称性检测
  ```python
  # Photoshop的"检查对称性"工具
  # 用于设计Logo时验证左右对称
  # 底层就是像素块树的镜像递归
  ```

- **场景2**:分子对称性分析
  ```python
  # 化学软件判断分子是否有对称轴
  # 决定分子的旋光性(光学活性)
  # 原子树的镜像判断
  ```

- **场景3**:代码语法树对称检测
  ```python
  # 某些编程语言的语法糖检测
  # 如Lisp的括号对称性验证
  # (a (b c) (c b)) 对称
  ```

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 100. 相同的树 | Easy | 双指针递归 | 不需要镜像,直接左对左右对右 |
| LeetCode 572. 另一棵树的子树 | Easy | 递归+遍历 | 先遍历找根,再判断子树是否相同 |
| LeetCode 951. 翻转等价二叉树 | Medium | 镜像递归 | 允许翻转,需判断两种情况 |
| LeetCode 1490. 克隆N叉树 | Medium | 递归复制 | 类似思想:递归克隆子树 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一棵二叉树,判断它是否**垂直对称**(即以根节点为轴,上下对称,而非左右)。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

垂直对称等价于:按层遍历,每一层是回文数组。可以用BFS层序遍历,每层检查是否回文。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
from collections import deque

def isVerticalSymmetric(root: Optional[TreeNode]) -> bool:
    """
    判断树是否垂直对称(上下对称)
    思路:BFS层序遍历,每层检查是否回文
    """
    if not root:
        return True

    queue = deque([root])

    while queue:
        level_vals = []
        level_size = len(queue)

        for _ in range(level_size):
            node = queue.popleft()
            if node:
                level_vals.append(node.val)
                queue.append(node.left)
                queue.append(node.right)
            else:
                level_vals.append(None)  # 占位符

        # 检查当前层是否回文
        if level_vals != level_vals[::-1]:
            return False

    return True
```

**核心思路**:
- BFS遍历每一层,包括None占位符
- 每层用双指针检查回文
- 时间O(n),空间O(w)

**注意**:这题在LeetCode上没有,是自创变体,用于练习BFS+回文判断。

</details>

---

## 💬 知识拓展

**对称性的数学本质**:
- 轴对称(本题):沿中轴翻转后与自己重合
- 中心对称:绕中心旋转180°后重合
- 旋转对称:旋转某个角度后重合

在树的语境中:
- 轴对称:左右子树镜像
- 中心对称:不存在(树没有旋转180°的概念)
- 旋转对称:N叉树可能有(如完全三叉树旋转120°)

这道题训练的是**递归思维+镜像遍历**,是理解树的对称性的最佳入门题!

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
