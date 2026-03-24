> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第48课:二叉树展开为链表

> **模块**:二叉树 | **难度**:Medium ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/
> **前置知识**:第39课(二叉树中序遍历)、第24课(反转链表)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个二叉树的根节点,将其原地展开为一个"单链表",展开后的单链表使用同样的TreeNode类,其中right子指针指向下一个节点,而left子指针始终为null。展开后的顺序应该与二叉树的前序遍历顺序相同。

**示例:**
```
输入:root = [1,2,5,3,4,null,6]
       1
      / \
     2   5
    / \   \
   3   4   6

输出:[1,null,2,null,3,null,4,null,5,null,6]
    1
     \
      2
       \
        3
         \
          4
           \
            5
             \
              6
```

**约束条件:**
- 树中节点数量范围为[0, 2000]
- -100 <= Node.val <= 100
- **必须原地展开,不能创建新节点**

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空树 | root=None | None | 空指针处理 |
| 单节点 | root=[1] | [1] | 基本功能 |
| 只有左子树 | root=[1,2,null,3] | [1,null,2,null,3] | 左子树处理 |
| 只有右子树 | root=[1,null,2,null,3] | [1,null,2,null,3] | 右子树处理 |
| 完全二叉树 | 示例输入 | 示例输出 | 完整逻辑 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在整理一串珍珠项链,原本珍珠是层次分明地挂在一棵树状支架上,现在要把它们拆下来重新串成一条直线。
>
> 🐌 **笨办法**:先用纸笔记录下所有珍珠的顺序(前序遍历),然后拆掉整个支架,再按记录的顺序一个个重新串起来。这需要额外的纸笔(O(n)空间)。
>
> 🚀 **聪明办法**:从最深处的珍珠开始,每次处理一小串,先把左边的珍珠串接到主线右侧,再把原来的右侧珍珠接到左侧珍珠的末尾。就像拉拉链一样,从下往上逐层"拉直",不需要额外工具!

### 关键洞察
**核心是"左子树插入到根节点与右子树之间",递归处理后左右子树都已展开成链表**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树根节点TreeNode
- **输出**:无返回值,原地修改树结构
- **限制**:必须原地操作,不能创建新节点;展开顺序为前序遍历顺序

### Step 2:先想笨办法(暴力法)
最直接的思路:
1. 用列表存储前序遍历结果
2. 遍历列表,逐个修改节点的left和right指针
- 时间复杂度:O(n)
- 空间复杂度:O(n) — 需要额外列表存储节点
- 瓶颈在哪:需要O(n)额外空间存储遍历结果

### Step 3:瓶颈分析 → 优化方向
分析暴力法的问题:
- 核心问题:为什么需要先存储?因为修改指针会破坏原有结构,导致无法继续遍历
- 优化思路:能否边遍历边修改,或者从后往前处理避免破坏结构?

### Step 4:选择武器
- 选用:**后序遍历 + 前驱节点记录**
- 理由:后序遍历保证子树先处理完,处理当前节点时子树已经展开成链表,可以安全连接;用一个全局变量记录上一个访问的节点,从后往前构建链表

> 🔑 **模式识别提示**:当题目要求"原地修改树结构"且涉及"遍历顺序",考虑**后序遍历 + 全局变量**模式

---

## 🔑 解法一:前序遍历 + 列表存储(直觉法)

### 思路
先用前序遍历将所有节点按顺序存入列表,然后遍历列表重建链表结构。

### 图解过程

```
原始树:
       1
      / \
     2   5
    / \   \
   3   4   6

Step 1: 前序遍历收集节点
  遍历顺序: 1 → 2 → 3 → 4 → 5 → 6
  nodes = [1, 2, 3, 4, 5, 6]

Step 2: 逐个连接节点
  1.left = None, 1.right = 2
  2.left = None, 2.right = 3
  3.left = None, 3.right = 4
  4.left = None, 4.right = 5
  5.left = None, 5.right = 6
  6.left = None, 6.right = None

最终结果:
  1 → 2 → 3 → 4 → 5 → 6
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def flatten(root: Optional[TreeNode]) -> None:
    """
    解法一:前序遍历 + 列表存储
    思路:先收集所有节点,再重建链表
    """
    if not root:
        return

    # Step 1: 前序遍历收集所有节点
    nodes = []

    def preorder(node):
        if not node:
            return
        nodes.append(node)  # 根
        preorder(node.left)   # 左
        preorder(node.right)  # 右

    preorder(root)

    # Step 2: 重建链表结构
    for i in range(len(nodes) - 1):
        nodes[i].left = None           # 清空左指针
        nodes[i].right = nodes[i + 1]  # 右指针指向下一个节点
    nodes[-1].left = None
    nodes[-1].right = None


# ✅ 测试
def build_tree():
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(5)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(4)
    root.right.right = TreeNode(6)
    return root

def print_flattened(root):
    result = []
    while root:
        result.append(root.val)
        root = root.right
    return result

root = build_tree()
flatten(root)
print(print_flattened(root))  # 期望输出:[1, 2, 3, 4, 5, 6]
```

### 复杂度分析
- **时间复杂度**:O(n) — 遍历一次n个节点 + 重建一次链表
  - 具体地说:如果树有1000个节点,需要约2000次操作(1000次遍历 + 1000次连接)
- **空间复杂度**:O(n) — 需要列表存储所有节点

### 优缺点
- ✅ 思路清晰,易于理解和实现
- ✅ 不会破坏树结构导致遍历中断
- ❌ 需要O(n)额外空间,不满足空间最优

---

## ⚡ 解法二:后序遍历 + 前驱节点(空间优化)

### 优化思路
关键洞察:**如果从右往左构建链表,每次只需要知道"上一个节点"是谁**。采用"右→左→根"的反向前序遍历,用一个全局变量prev记录前驱节点,当前节点的right指向prev即可。

> 💡 **关键想法**:后序遍历保证子树先处理完,处理当前节点时左右子树已展开,可以安全连接

### 图解过程

```
原始树:
       1
      / \
     2   5
    / \   \
   3   4   6

采用"右→左→根"顺序(反向前序):6 → 5 → 4 → 3 → 2 → 1

Step 1: 处理节点6 (最右下)
  prev = None
  6.right = None
  prev = 6

Step 2: 处理节点5
  prev = 6
  5.left = None, 5.right = 6
  prev = 5
  链表: 5 → 6

Step 3: 处理节点4
  prev = 5
  4.left = None, 4.right = 5
  prev = 4
  链表: 4 → 5 → 6

Step 4: 处理节点3
  prev = 4
  3.left = None, 3.right = 4
  prev = 3
  链表: 3 → 4 → 5 → 6

Step 5: 处理节点2
  prev = 3
  2.left = None, 2.right = 3
  prev = 2
  链表: 2 → 3 → 4 → 5 → 6

Step 6: 处理节点1 (根)
  prev = 2
  1.left = None, 1.right = 2
  链表: 1 → 2 → 3 → 4 → 5 → 6 ✓
```

### Python代码

```python
def flatten_v2(root: Optional[TreeNode]) -> None:
    """
    解法二:后序遍历 + 前驱节点
    思路:反向前序遍历(右→左→根),用prev记录前驱
    """
    prev = [None]  # 用列表包装使其可在内部函数修改

    def reverse_preorder(node):
        if not node:
            return

        # 先递归处理右子树
        reverse_preorder(node.right)
        # 再递归处理左子树
        reverse_preorder(node.left)

        # 最后处理根节点
        node.left = None           # 清空左指针
        node.right = prev[0]       # 右指针指向前驱
        prev[0] = node             # 更新前驱为当前节点

    reverse_preorder(root)


# ✅ 测试
root = build_tree()
flatten_v2(root)
print(print_flattened(root))  # 期望输出:[1, 2, 3, 4, 5, 6]
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点访问一次
- **空间复杂度**:O(h) — 递归栈深度,h为树高,最坏O(n),平衡树O(log n)

---

## 🏆 解法三:Morris遍历(最优解 - O(1)空间)

### 优化思路
Morris遍历利用树中大量的null指针来存储遍历信息,无需递归栈。核心思想:**对每个节点,找到其左子树的最右节点,将右子树接到这个最右节点后面,然后将左子树移到右边**。

> 💡 **关键想法**:左子树的最右节点是前序遍历中当前节点左子树的最后一个节点,将原右子树接在此处即可

### 图解过程

```
原始树:
       1
      / \
     2   5
    / \   \
   3   4   6

Step 1: 处理节点1
  找到左子树(2)的最右节点: 4
  将右子树(5)接到4的右边:
       1
      /
     2
    / \
   3   4
        \
         5
          \
           6
  将左子树移到右边,左指针置空:
    1
     \
      2
     / \
    3   4
         \
          5
           \
            6
  curr移动到2

Step 2: 处理节点2
  找到左子树(3)的最右节点: 3
  将右子树(4→5→6)接到3的右边:
    1
     \
      2
     /
    3
     \
      4
       \
        5
         \
          6
  将左子树移到右边:
    1
     \
      2
       \
        3
         \
          4
           \
            5
             \
              6
  curr移动到3

Step 3: 处理节点3
  无左子树,curr移动到4

Step 4: 处理节点4
  无左子树,curr移动到5

Step 5: 处理节点5
  无左子树,curr移动到6

Step 6: 处理节点6
  无左子树,遍历结束

最终结果: 1 → 2 → 3 → 4 → 5 → 6
```

### Python代码

```python
def flatten_v3(root: Optional[TreeNode]) -> None:
    """
    🏆 解法三:Morris遍历(最优解)
    思路:利用左子树最右节点连接右子树,O(1)空间
    """
    curr = root

    while curr:
        if curr.left:
            # 找到左子树的最右节点
            rightmost = curr.left
            while rightmost.right:
                rightmost = rightmost.right

            # 将当前节点的右子树接到左子树最右节点的右边
            rightmost.right = curr.right

            # 将左子树移到右边
            curr.right = curr.left
            curr.left = None

        # 移动到下一个节点
        curr = curr.right


# ✅ 测试
root = build_tree()
flatten_v3(root)
print(print_flattened(root))  # 期望输出:[1, 2, 3, 4, 5, 6]

# 边界测试
print(print_flattened(None))  # 期望输出:[]
single = TreeNode(1)
flatten_v3(single)
print(print_flattened(single))  # 期望输出:[1]
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点最多访问两次(一次展开,一次移动到下一个)
- **空间复杂度**:O(1) — 只用常数个变量,无递归栈

---

## 🐍 Pythonic 写法

利用Python的非局部变量和闭包简化解法二:

```python
def flatten_pythonic(root: Optional[TreeNode]) -> None:
    """Pythonic写法:使用nonlocal简化前驱节点传递"""
    prev = None

    def reverse_preorder(node):
        nonlocal prev  # 声明使用外部变量
        if not node:
            return
        reverse_preorder(node.right)
        reverse_preorder(node.left)
        node.left = None
        node.right = prev
        prev = node

    reverse_preorder(root)
```

这个写法用`nonlocal`关键字替代列表包装,代码更简洁。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:前序遍历+列表 | 解法二:后序遍历+前驱 | 🏆 解法三:Morris遍历(最优) |
|------|-------------------|-------------------|------------------------|
| 时间复杂度 | O(n) | O(n) | **O(n)** ← 时间相同 |
| 空间复杂度 | O(n) | O(h),最坏O(n) | **O(1)** ← 空间最优 |
| 代码难度 | 简单 | 中等 | 较难 |
| 面试推荐 | ⭐ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 理解基础思路 | 空间受限但可用递归 | **严格O(1)空间要求** |

**为什么解法三是最优解**:
- 时间复杂度O(n)已经是理论最优(至少要访问所有节点一次)
- 空间复杂度O(1)达到极致优化,无任何额外存储
- Morris遍历是树遍历中空间最优的经典技巧,面试加分项

**面试建议**:
1. 先用1分钟口述解法一思路(前序遍历+列表),表明你能想到基本解法
2. 立即优化到🏆解法三(Morris遍历),展示对高级技巧的掌握
3. **重点讲解最优解的核心思想**:"找到左子树最右节点,将右子树接在其后,然后左子树移到右边"
4. 强调为什么这是最优:O(1)空间且不破坏遍历过程,利用了树中null指针
5. 手动模拟示例,展示对Morris遍历的深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你将二叉树原地展开为链表,展开顺序为前序遍历。

**你**:(审题30秒)好的,这道题要求原地展开,展开后左指针全为null,右指针连成前序遍历顺序。让我先想一下...
我的第一个想法是先前序遍历收集所有节点到列表,再重建链表,时间O(n)但空间也是O(n)。
不过我们可以用Morris遍历优化到O(1)空间,核心思路是:对每个有左子树的节点,找到左子树的最右节点,将当前节点的右子树接到那里,然后把左子树移到右边。

**面试官**:很好,请写一下Morris遍历的代码。

**你**:(边写边说关键步骤)我用一个while循环遍历树,对每个有左子树的节点,先找到左子树的最右节点rightmost,然后执行三步:1)将curr.right接到rightmost.right; 2)将curr.left移到curr.right; 3)清空curr.left。最后移动到下一个节点。

**面试官**:测试一下?

**你**:用示例[1,2,5,3,4,null,6]走一遍...(手动模拟)处理节点1时,左子树2的最右节点是4,将5接到4.right,然后2移到1.right...最终得到1→2→3→4→5→6,结果正确。再测边界情况空树和单节点...(验证)都正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么叫Morris遍历?" | "Morris遍历是1979年J.H.Morris提出的利用线索二叉树思想实现O(1)空间遍历的算法,它巧妙利用了树中大量的null指针来存储遍历信息,避免递归栈和额外数组。" |
| "能用迭代的前序遍历实现吗?" | "可以,用栈模拟前序遍历,但空间复杂度是O(h)。Morris遍历的优势在于彻底消除了栈空间。" |
| "如果要展开成后序遍历顺序呢?" | "可以用类似思路,但需要调整连接逻辑,或者用'左→右→根'的反向后序遍历配合前驱节点。" |
| "时间复杂度真的是O(n)吗?看起来有嵌套循环" | "是的,虽然有while嵌套,但每条边最多被访问两次(一次向下找最右节点,一次移动到下一个),总操作次数是O(n)。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 1. nonlocal关键字 — 在嵌套函数中修改外部变量
def outer():
    count = 0
    def inner():
        nonlocal count  # 声明使用外部变量
        count += 1
    inner()
    return count

# 2. 列表包装传递可变状态 — 替代nonlocal
def outer_v2():
    count = [0]  # 用列表包装
    def inner():
        count[0] += 1  # 修改列表内容
    inner()
    return count[0]

# 3. while循环遍历链表 — 树展开后遍历
def traverse_flattened(root):
    while root:
        print(root.val)
        root = root.right
```

### 💡 底层原理(选读)

> **Morris遍历的本质是什么?**
> Morris遍历利用了"线索二叉树"(Threaded Binary Tree)的思想。在普通二叉树中,有大量null指针被浪费。Morris遍历临时利用这些null指针来存储"返回路径",实现无栈遍历。
>
> **为什么能做到O(1)空间?**
> 递归和栈本质上都需要O(h)空间来记录"回溯路径"。Morris遍历通过修改树的指针结构,将"回溯路径"编码在树本身中,遍历完成后指针已重新排列成目标形态,巧妙避免了额外存储。
>
> **Morris遍历的局限性**:虽然空间最优,但会临时破坏树结构(遍历过程中修改指针),不适合并发环境或需要保留原结构的场景。

### 算法模式卡片 📐
- **模式名称**:Morris遍历 / 原地树结构修改
- **适用条件**:
  - 需要遍历二叉树但严格限制O(1)空间
  - 允许临时修改树结构(遍历过程中)
  - 要求原地重组树节点
- **识别关键词**:"原地"、"O(1)空间"、"展开"、"遍历"
- **模板代码**:
```python
# Morris遍历通用模板
def morris_traversal(root):
    curr = root
    while curr:
        if curr.left:
            # 找前驱(左子树最右节点)
            pred = curr.left
            while pred.right and pred.right != curr:
                pred = pred.right

            if not pred.right:
                # 建立线索
                pred.right = curr
                curr = curr.left
            else:
                # 恢复结构(或执行操作)
                pred.right = None
                curr = curr.right
        else:
            curr = curr.right
```

### 易错点 ⚠️
1. **忘记清空左指针** — 展开后必须将所有节点的left置为None,否则不是单链表结构。
   - **正确做法**:每次移动左子树到右边后,立即执行`curr.left = None`

2. **Morris遍历中找最右节点的循环条件错误** — 应该是`while rightmost.right`,而非`while rightmost.right and rightmost.right != curr`(后者用于标准Morris遍历恢复结构)。
   - **原因**:本题中我们直接修改结构不恢复,所以只需找到最右的null节点即可

3. **解法二中prev初始化错误** — 如果直接用`prev = None`,内部函数无法修改外部变量,必须用`prev = [None]`列表包装或`nonlocal`声明。
   - **正确做法**:使用`nonlocal prev`或`prev = [None]`包装

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:文件系统树形目录展平 — 操作系统中将树形目录结构展平为顺序访问链表,用于快速遍历文件(如`find`命令的内部实现)。

- **场景2**:编译器语法树线性化 — 编译器将抽象语法树(AST)展平为中间代码的线性指令序列,Morris遍历可在内存受限环境下完成。

- **场景3**:数据库B+树叶子节点链表 — B+树的叶子节点通过指针连成链表,支持范围查询,构建过程类似树展开为链表。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 897. 递增顺序搜索树 | Easy | 树展平、中序遍历 | 将BST展平为只有右子树的递增链表 |
| LeetCode 426. 将二叉搜索树转化为排序的双向链表 | Medium | 树展平、中序遍历 | 类似本题但要双向链表,需修改left指针 |
| LeetCode 99. 恢复二叉搜索树 | Medium | Morris遍历、BST性质 | 用Morris遍历O(1)空间找到BST中两个错位节点 |
| LeetCode 173. 二叉搜索树迭代器 | Medium | 受控遍历、栈 | 类似思路但需要支持next()按需遍历 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定二叉树,要求将其展开为单链表,但展开顺序为**后序遍历**而非前序遍历。例如:
```
输入: [1,2,3]
   1
  / \
 2   3
输出: 2 → 3 → 1
```
请用O(1)空间实现。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

类似Morris遍历,但需要"右→左→根"的反向后序遍历,用prev记录前驱节点,从后往前构建链表。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def flatten_postorder(root: Optional[TreeNode]) -> None:
    """后序遍历展开:右→左→根反向构建"""
    prev = None

    def reverse_postorder(node):
        nonlocal prev
        if not node:
            return

        # 先根
        reverse_postorder(node.left)
        reverse_postorder(node.right)

        # 后处理(反向构建链表)
        node.left = None
        node.right = prev
        prev = node

    reverse_postorder(root)
```

**核心思路**:后序遍历的逆序就是"根→右→左",用反向遍历配合prev即可从后往前构建链表。时间O(n),空间O(h)。如果要O(1)空间,需要设计更复杂的Morris后序遍历变体。

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
