> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第49课:二叉树的最近公共祖先

> **模块**:二叉树 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/
> **前置知识**:第39课(二叉树中序遍历)、第40课(二叉树最大深度)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定一个二叉树的根节点和两个指定节点p和q,找到这两个节点的最近公共祖先(LCA, Lowest Common Ancestor)。最近公共祖先定义为:两个节点p和q的公共祖先中,离这两个节点最近的那一个。注意一个节点可以是它自己的祖先。

**示例:**
```
输入:root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
         3
       /   \
      5     1
     / \   / \
    6   2 0   8
       / \
      7   4
输出:3
解释:节点5和节点1的最近公共祖先是节点3
```

```
输入:root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
输出:5
解释:节点5和节点4的最近公共祖先是节点5(节点可以是自己的祖先)
```

**约束条件:**
- 树中节点数量范围为[2, 10^5]
- -10^9 <= Node.val <= 10^9
- 所有节点值唯一
- **p和q一定存在于树中**

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| p和q在不同子树 | p=5,q=1 | 3 | 基本功能 |
| p是q的祖先 | p=5,q=4 | 5 | 祖先定义 |
| q是p的祖先 | p=4,q=5 | 5 | 对称情况 |
| p和q是兄弟节点 | p=6,q=2 | 5 | 同层节点 |
| 根节点是LCA | p=5,q=1 | 3 | 根节点情况 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在查家谱,要找两个人的"最近共同祖先"(比如你和你表哥的最近共同祖先可能是你们的爷爷)。
>
> 🐌 **笨办法**:分别从两个人往上追溯到根(记录路径),然后比较两条路径,找到最后一个相同的祖先。就像两个人分别往上报自己的家谱,然后对比找共同点。需要额外纸笔记录路径(O(n)空间)。
>
> 🚀 **聪明办法**:从家谱树的根开始往下走,如果在某个节点发现"两个人分别在我的左右两边",那我就是最近公共祖先!如果两人都在左边,那答案在左子树继续找;都在右边就去右子树找。就像一个裁判站在树顶,不断向下判断直到找到分岔点,不需要记录路径!

### 关键洞察
**核心是"后序遍历",从子树返回信息汇总到父节点判断:如果左右子树分别找到p和q,当前节点就是LCA**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树根节点root,两个目标节点p和q
- **输出**:返回p和q的最近公共祖先节点
- **限制**:p和q一定存在;节点值唯一;一个节点可以是自己的祖先

### Step 2:先想笨办法(暴力法)
最直接的思路:
1. 从root到p记录路径path_p,从root到q记录路径path_q
2. 遍历两条路径,找最后一个相同节点
- 时间复杂度:O(n) — 需要遍历两次找路径
- 空间复杂度:O(h) — 需要存储路径,h为树高
- 瓶颈在哪:需要额外存储路径信息

### Step 3:瓶颈分析 → 优化方向
分析暴力法的问题:
- 核心问题:为什么需要先存路径?因为要对比找共同祖先
- 优化思路:能否在遍历过程中直接判断?关键洞察:**如果当前节点的左子树包含p,右子树包含q(或反过来),当前节点就是LCA!**

### Step 4:选择武器
- 选用:**后序遍历(左→右→根)**
- 理由:后序遍历保证先处理子树,子树返回"是否找到p或q"的信息,当前节点根据左右子树返回值判断自己是否是LCA;只需一次遍历,不需要存路径

> 🔑 **模式识别提示**:当题目需要"自底向上汇总子树信息",优先考虑**后序遍历**模式

---

## 🔑 解法一:存储父节点 + 路径回溯(直觉法)

### 思路
先遍历树记录每个节点的父节点,然后从p开始往上回溯记录所有祖先,再从q往上回溯,第一个在p的祖先集合中出现的节点就是LCA。

### 图解过程

```
树结构:
         3
       /   \
      5     1
     / \   / \
    6   2 0   8
       / \
      7   4

查找p=5, q=1的LCA:

Step 1: 记录父节点关系
  parent = {5:3, 1:3, 6:5, 2:5, 0:1, 8:1, 7:2, 4:2}

Step 2: 从p=5往上回溯,记录祖先
  5 → 3 → None
  ancestors = {5, 3}

Step 3: 从q=1往上回溯,找第一个在ancestors中的
  1 → 检查1,不在ancestors中
  1的父节点 → 3 → 检查3,在ancestors中! ✓

返回: 3
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def lowestCommonAncestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    解法一:存储父节点 + 路径回溯
    思路:记录父节点关系,从p和q往上回溯找第一个共同祖先
    """
    # Step 1: 遍历树,记录每个节点的父节点
    parent = {root: None}

    def record_parent(node):
        if not node:
            return
        if node.left:
            parent[node.left] = node
            record_parent(node.left)
        if node.right:
            parent[node.right] = node
            record_parent(node.right)

    record_parent(root)

    # Step 2: 从p往上回溯,记录所有祖先
    ancestors = set()
    while p:
        ancestors.add(p)
        p = parent[p]

    # Step 3: 从q往上回溯,找第一个在祖先集合中的节点
    while q not in ancestors:
        q = parent[q]

    return q


# ✅ 测试
def build_tree():
    root = TreeNode(3)
    root.left = TreeNode(5)
    root.right = TreeNode(1)
    root.left.left = TreeNode(6)
    root.left.right = TreeNode(2)
    root.right.left = TreeNode(0)
    root.right.right = TreeNode(8)
    root.left.right.left = TreeNode(7)
    root.left.right.right = TreeNode(4)
    return root

root = build_tree()
p, q = root.left, root.right  # 5, 1
print(lowestCommonAncestor(root, p, q).val)  # 期望输出:3

p, q = root.left, root.left.right.right  # 5, 4
print(lowestCommonAncestor(root, p, q).val)  # 期望输出:5
```

### 复杂度分析
- **时间复杂度**:O(n) — 遍历一次树记录父节点O(n),回溯最多O(h)
  - 具体地说:如果树有1000个节点,最坏需要约1000次遍历 + 树高次回溯
- **空间复杂度**:O(n) — 需要哈希表存储所有节点的父节点关系

### 优缺点
- ✅ 思路直观,易于理解
- ✅ 通用性强,适用于需要多次查询LCA的场景(预处理一次)
- ❌ 需要O(n)额外空间存储父节点关系

---

## 🏆 解法二:后序遍历递归(最优解)

### 优化思路
利用后序遍历"自底向上"的特性,递归函数返回值表示"当前子树是否包含p或q"。当某个节点发现左右子树分别找到p和q时,该节点就是LCA。

> 💡 **关键想法**:后序遍历保证子树先返回结果,当前节点根据左右返回值做三种判断:1)左右都找到→我是LCA; 2)只有一边找到→返回那一边的结果; 3)都没找到→返回null

### 图解过程

```
树结构:
         3
       /   \
      5     1
     / \   / \
    6   2 0   8
       / \
      7   4

查找p=5, q=1的LCA:

递归返回值含义:
- 返回None: 子树中不包含p和q
- 返回某节点: 子树中找到p或q,或已找到LCA

Step 1: 递归到叶子节点7
  左右子树都返回None
  7 != p且7 != q → 返回None

Step 2: 递归到叶子节点4
  左右子树都返回None
  4 != p且4 != q → 返回None

Step 3: 递归到节点2
  左子树返回None, 右子树返回None
  2 != p且2 != q → 返回None

Step 4: 递归到节点6
  6 != p且6 != q → 返回None

Step 5: 递归到节点5 ⭐
  左子树(6)返回None, 右子树(2)返回None
  5 == p → 返回5

Step 6: 递归到节点0
  0 != p且0 != q → 返回None

Step 7: 递归到节点8
  8 != p且8 != q → 返回None

Step 8: 递归到节点1 ⭐
  左子树(0)返回None, 右子树(8)返回None
  1 == q → 返回1

Step 9: 递归到根节点3 ⭐⭐
  左子树返回5 (找到p)
  右子树返回1 (找到q)
  左右都不为None → 当前节点3是LCA! 返回3
```

### Python代码

```python
def lowestCommonAncestor_v2(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    🏆 解法二:后序遍历递归(最优解)
    思路:后序遍历,左右子树返回是否找到p/q,当前节点判断是否是LCA
    """
    # 递归终止条件
    if not root:
        return None
    if root == p or root == q:
        return root  # 找到目标节点之一,返回它

    # 后序遍历:先递归左右子树
    left = lowestCommonAncestor_v2(root.left, p, q)
    right = lowestCommonAncestor_v2(root.right, p, q)

    # 根据左右子树返回值判断
    if left and right:
        # 左右子树都找到了 → 当前节点是LCA
        return root

    # 只有一边找到 → 返回找到的那一边(可能是p/q本身,或已找到的LCA)
    return left if left else right


# ✅ 测试
root = build_tree()
p, q = root.left, root.right  # 5, 1
print(lowestCommonAncestor_v2(root, p, q).val)  # 期望输出:3

p, q = root.left, root.left.right.right  # 5, 4
print(lowestCommonAncestor_v2(root, p, q).val)  # 期望输出:5

p, q = root.left.left, root.left.right  # 6, 2
print(lowestCommonAncestor_v2(root, p, q).val)  # 期望输出:5
```

### 复杂度分析
- **时间复杂度**:O(n) — 最坏情况遍历所有节点一次
- **空间复杂度**:O(h) — 递归栈深度,h为树高,平衡树O(log n),最坏O(n)

---

## ⚡ 解法三:迭代 + 父节点记录(空间优化变体)

### 优化思路
解法一的改进版:不存储所有节点的父节点,只在遍历过程中用栈同时记录路径,找到p和q后立即比较路径。

### Python代码

```python
def lowestCommonAncestor_v3(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """
    解法三:迭代 + 路径记录
    思路:用栈遍历,记录到p和q的路径,找最后一个公共节点
    """
    # 辅助函数:找从root到target的路径
    def find_path(root, target):
        path = []
        stack = [(root, [root])]

        while stack:
            node, current_path = stack.pop()
            if node == target:
                return current_path
            if node.right:
                stack.append((node.right, current_path + [node.right]))
            if node.left:
                stack.append((node.left, current_path + [node.left]))
        return []

    # 找到两条路径
    path_p = find_path(root, p)
    path_q = find_path(root, q)

    # 找最后一个公共节点
    lca = root
    for i in range(min(len(path_p), len(path_q))):
        if path_p[i] == path_q[i]:
            lca = path_p[i]
        else:
            break

    return lca


# ✅ 测试
root = build_tree()
p, q = root.left, root.right
print(lowestCommonAncestor_v3(root, p, q).val)  # 期望输出:3
```

### 复杂度分析
- **时间复杂度**:O(n) — 最坏遍历两次树
- **空间复杂度**:O(h) — 栈空间和路径存储

---

## 🐍 Pythonic 写法

利用Python的逻辑短路和三元表达式简化解法二:

```python
def lowestCommonAncestor_pythonic(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """Pythonic写法:一行三元表达式搞定判断逻辑"""
    if not root or root == p or root == q:
        return root

    left = lowestCommonAncestor_pythonic(root.left, p, q)
    right = lowestCommonAncestor_pythonic(root.right, p, q)

    # 一行搞定三种情况判断
    return root if (left and right) else (left or right)
```

这个写法利用`or`的短路特性:`left or right`在left为None时返回right,否则返回left。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:父节点记录 | 🏆 解法二:后序遍历(最优) | 解法三:迭代路径 |
|------|----------------|---------------------|--------------|
| 时间复杂度 | O(n) | **O(n)** ← 时间相同 | O(n) |
| 空间复杂度 | O(n) | **O(h)** ← 空间最优 | O(h) |
| 代码难度 | 中等 | 简单 | 较难 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 | ⭐ |
| 适用场景 | 多次查询LCA | **单次查询,代码简洁** | 需要显式路径 |

**为什么解法二是最优解**:
- 时间复杂度O(n)已经是理论最优(最坏要访问所有节点)
- 空间复杂度O(h)优于O(n),利用递归栈而非额外哈希表
- 代码极简,只需10行核心逻辑,面试中容易写对
- 后序遍历思想优雅,体现对树结构的深入理解

**面试建议**:
1. 先用30秒口述解法一思路(记录父节点回溯),表明你能想到基本解法
2. 立即优化到🏆解法二(后序遍历),展示对递归和树遍历的掌握
3. **重点讲解最优解的核心思想**:"后序遍历自底向上,左右子树返回是否找到p/q,当前节点判断:左右都找到→我是LCA;只有一边找到→返回那一边"
4. 强调为什么这是最优:只需一次遍历,空间仅O(h)递归栈,代码简洁优雅
5. 手动模拟示例,特别强调"p是q祖先"的情况(找到p后直接返回p,不继续往下)

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你找到二叉树中两个节点的最近公共祖先。

**你**:(审题30秒)好的,最近公共祖先LCA是指两个节点的公共祖先中最靠近它们的那一个。让我先想一下...
我的第一个想法是记录每个节点的父节点,然后从p和q分别往上回溯找第一个公共祖先,时间O(n)但空间也是O(n)。
不过我们可以用后序遍历优化到O(h)空间,核心思路是:递归左右子树,如果左子树找到p,右子树找到q(或反过来),当前节点就是LCA;如果只有一边找到,返回那一边的结果。

**面试官**:很好,请写一下后序遍历的代码。

**你**:(边写边说关键步骤)我用递归实现,终止条件是节点为空或找到p/q就返回。然后后序遍历左右子树。关键判断:如果left和right都不为空,说明p和q分别在左右子树,当前节点是LCA;如果只有一边非空,返回那一边(可能是p/q本身,或子树中已找到的LCA)。

**面试官**:测试一下?

**你**:用示例p=5,q=1走一遍...(手动模拟)递归到节点3时,左子树返回5,右子树返回1,left和right都非空,所以3是LCA,返回3。再测p=5,q=4的情况...(模拟)递归到5时,5==p直接返回5,向上传递,最终5就是LCA。边界情况都正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果p或q不存在于树中怎么办?" | "当前解法假设p和q一定存在。如果可能不存在,需要先遍历一次确认存在性,或修改返回值携带'是否找到'的标志位。" |
| "如果是二叉搜索树呢?" | "BST可以利用有序性优化:从根开始,如果p和q都小于当前节点,LCA在左子树;都大于则在右子树;一大一小则当前节点就是LCA。时间O(h),更高效。" |
| "如果需要多次查询不同的p和q?" | "可以预处理:用解法一记录所有父节点关系,或用倍增算法(Binary Lifting)预处理每个节点的2^k级祖先,单次查询优化到O(log n)。" |
| "能否用迭代实现?" | "可以,但较复杂。需要用栈模拟后序遍历,同时维护节点状态(左右子树是否已访问)。递归版本更简洁,面试中推荐递归。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 1. 逻辑短路 — 简化if-else
result = left if left else right
# 等价于:
# if left:
#     result = left
# else:
#     result = right

# 2. 链式比较 — 判断范围
if p.val < root.val < q.val:
    # p和q分别在root两侧

# 3. 集合操作 — 快速判断存在
ancestors = set()
while p:
    ancestors.add(p)
    p = parent[p]
# O(1)判断
if q in ancestors:
    return q
```

### 💡 底层原理(选读)

> **为什么后序遍历适合求LCA?**
> 后序遍历的特点是"左→右→根",即先处理子树再处理当前节点。在求LCA时,我们需要知道"p和q分别在哪个子树",这个信息只有子树处理完才能得到,所以必须用后序遍历。前序或中序遍历都无法在处理当前节点时获得子树的完整信息。
>
> **递归返回值的语义设计**:
> - 返回None:子树中不包含p和q
> - 返回非None节点:可能是三种情况之一:1)找到p或q本身; 2)子树中已找到LCA; 3)子树中只找到p或q之一
> 这种"多义"返回值设计是递归的精髓:通过巧妙的终止条件和合并逻辑,一个返回值承载多种语义。
>
> **BST的LCA为什么更简单?**
> 二叉搜索树的有序性保证:如果p.val < root.val < q.val,则p和q必定分别在root的左右子树,root就是LCA。这个判断O(1)完成,无需遍历子树,时间复杂度降为O(h)。

### 算法模式卡片 📐
- **模式名称**:后序遍历 + 信息汇总
- **适用条件**:
  - 需要根据子树信息做决策
  - 自底向上传递信息(如求LCA、树的直径、验证BST)
  - 子问题的解需要合并得到父问题的解
- **识别关键词**:"最近"、"自底向上"、"子树信息"、"汇总"
- **模板代码**:
```python
def postorder_with_info(root):
    """后序遍历汇总子树信息模板"""
    if not root:
        return None  # 或其他默认值

    # 后序:先递归左右子树
    left_info = postorder_with_info(root.left)
    right_info = postorder_with_info(root.right)

    # 根据子树信息计算当前节点的信息
    current_info = combine(left_info, right_info, root)

    return current_info
```

### 易错点 ⚠️
1. **忘记处理"节点是自己祖先"的情况** — 当root==p时,即使q在p的子树中,答案也应该是p,而非继续往下找。
   - **正确做法**:在递归开始就判断`if root == p or root == q: return root`,提前终止

2. **左右子树返回值判断错误** — 常见错误是写成`if left or right: return root`,这会导致只找到一个节点就误判为LCA。
   - **正确做法**:必须是`if left and right: return root`,两边都找到才是LCA

3. **BST的LCA误用普通二叉树解法** — 在BST中可以利用大小关系更高效,如果用后序遍历就浪费了有序性。
   - **正确做法**:BST单独判断:`if p.val < root.val and q.val < root.val: 往左找; elif ...: 往右找; else: return root`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:版本控制系统(Git) — Git中找两个分支的最近公共提交(merge base),用于三方合并。提交历史形成DAG(有向无环图),LCA算法找分叉点。

- **场景2**:组织架构管理系统 — 企业管理软件中查找两个员工的"最近共同上级",组织架构是树形结构,LCA快速定位共同汇报对象。

- **场景3**:社交网络关系图 — 找两个用户的"最近共同好友",社交关系可抽象为图/树,LCA帮助推荐共同认识的人。

- **场景4**:区块链分叉处理 — 区块链出现分叉时,找两条链的最近公共区块,决定从哪里开始同步。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 235. 二叉搜索树的最近公共祖先 | Medium | LCA、BST性质 | 利用BST有序性,O(h)时间不需遍历所有节点 |
| LeetCode 1676. 二叉树的最近公共祖先IV | Medium | LCA、多节点 | 找多个节点的LCA,扩展解法二的判断逻辑 |
| LeetCode 1644. 二叉树的最近公共祖先II | Medium | LCA、节点可能不存在 | p或q可能不存在,需修改返回值携带存在性标志 |
| LeetCode 865. 具有所有最深节点的最小子树 | Medium | LCA、树的深度 | 找最深叶子节点的LCA,结合深度计算 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定二叉搜索树(BST)和两个节点p、q,找它们的最近公共祖先。要求利用BST的有序性优化到O(h)时间。
```
输入:root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
      6
    /   \
   2     8
  / \   / \
 0   4 7   9
    / \
   3   5
输出:6
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

利用BST性质:如果p和q的值都小于当前节点,LCA在左子树;都大于则在右子树;一大一小则当前节点就是LCA。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def lowestCommonAncestor_BST(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
    """BST的LCA:利用有序性优化"""
    # 保证p.val <= q.val,简化判断
    if p.val > q.val:
        p, q = q, p

    while root:
        if q.val < root.val:
            # p和q都在左子树
            root = root.left
        elif p.val > root.val:
            # p和q都在右子树
            root = root.right
        else:
            # p <= root <= q,找到LCA
            return root

    return None  # 理论上不会到这里(题目保证p、q存在)


# ✅ 测试
# root = [6,2,8,0,4,7,9,null,null,3,5]
# p=2, q=8 → 输出6
# p=2, q=4 → 输出2
```

**核心思路**:BST的有序性保证我们可以通过值比较直接判断走向,无需遍历整棵树。时间O(h),空间O(1)(迭代版本)。这是BST专属优化!

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
