# 📖 第39课:二叉树中序遍历

> **模块**:二叉树 | **难度**:Easy ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/binary-tree-inorder-traversal/
> **前置知识**:无(二叉树模块第一题)
> **预计学习时间**:20分钟

---

## 🎯 题目描述

给定一个二叉树的根节点 `root`,返回它的 **中序遍历** 结果。

**中序遍历定义**:对于每个节点,访问顺序为 **左子树 → 根节点 → 右子树**。

**示例:**
```
输入:root = [1,null,2,3]
      1
       \
        2
       /
      3
输出:[1,3,2]
解释:中序遍历顺序 1(根) → 3(左) → 2(右)
```

**示例2:**
```
输入:root = [1,2,3,4,5]
       1
      / \
     2   3
    / \
   4   5
输出:[4,2,5,1,3]
```

**约束条件:**
- 树中节点数目在范围 `[0, 100]` 内
- `-100 ≤ Node.val ≤ 100`

**进阶**:递归算法很简单,你能用迭代算法完成吗?

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空树 | root = None | [] | 基本边界 |
| 单节点 | root = [1] | [1] | 最小有效树 |
| 只有左子树 | root = [1,2,null,3] | [3,2,1] | 单侧链 |
| 只有右子树 | root = [1,null,2,3] | [1,3,2] | 单侧链 |
| 完全二叉树 | root = [1,2,3,4,5,6,7] | [4,2,5,1,6,3,7] | 对称结构 |
| 最大规模 | 100个节点 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在图书馆整理书架上的书,要按特定顺序记录所有书名。
>
> 🐌 **笨办法**:把所有书都取下来摆在桌上,然后一本本记录 → 需要额外空间,效率低。
>
> 🚀 **聪明办法**:利用"递归规则"直接在书架上操作:
> 1. 对于每个格子,先记录左边小格子里的书
> 2. 再记录这个格子自己的书
> 3. 最后记录右边小格子里的书
>
> 这就是**中序遍历的递归思想** — 每个节点都按"左中右"规则处理,自动形成有序序列。

### 关键洞察
**二叉树的递归定义天然适合递归算法:每个节点的处理方式和整棵树相同**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树的根节点 `root`(可能为空)
- **输出**:中序遍历的结果列表 `[val1, val2, ...]`
- **遍历顺序**:左子树 → 根节点 → 右子树(递归定义)
- **限制**:节点值范围 [-100, 100],数量 [0, 100]

### Step 2:识别模式
这是典型的**树的遍历**问题,有三种经典顺序:
- **前序遍历**:根 → 左 → 右(根在前)
- **中序遍历**:左 → 根 → 右(根在中) ← 本题
- **后序遍历**:左 → 右 → 根(根在后)

### Step 3:选择武器
**方案1**:递归 — 直接按定义实现,代码简洁
**方案2**:迭代 + 栈 — 手动模拟递归调用栈
**方案3**:Morris遍历 — O(1)空间的进阶算法

### Step 4:确定最优解
递归解法是**最优解**:
- 时间O(n)已达理论最优(必须访问所有节点)
- 空间O(h),h为树高,这是递归必需的
- 代码极简,面试首选

---

## 🏆 解法一:递归(最优解)

### 💡 核心思想
按照中序遍历的定义,递归处理每个节点:
1. 递归遍历左子树
2. 访问根节点(添加到结果)
3. 递归遍历右子树

**递归三要素**:
1. **递归函数定义**:将 node 的中序遍历结果添加到 result
2. **递归终止条件**:node 为空时返回
3. **递归调用**:先递归左子树,再访问根,最后递归右子树

### 📊 图解演示
```
示例:root = [1,2,3,4,5]
       1
      / \
     2   3
    / \
   4   5

递归执行流程(调用栈):

Step 1: inorder(1)
  ├─ inorder(2)  ← 先递归左子树
  │   ├─ inorder(4)
  │   │   ├─ inorder(None) → 返回
  │   │   ├─ 访问 4 → result = [4]
  │   │   └─ inorder(None) → 返回
  │   ├─ 访问 2 → result = [4,2]
  │   └─ inorder(5)
  │       ├─ inorder(None) → 返回
  │       ├─ 访问 5 → result = [4,2,5]
  │       └─ inorder(None) → 返回
  ├─ 访问 1 → result = [4,2,5,1]
  └─ inorder(3)  ← 最后递归右子树
      ├─ inorder(None) → 返回
      ├─ 访问 3 → result = [4,2,5,1,3]
      └─ inorder(None) → 返回

最终结果:[4,2,5,1,3]
```

### 📝 代码实现
```python
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        """递归实现中序遍历"""
        result = []

        def inorder(node):
            """辅助递归函数"""
            if not node:
                return

            inorder(node.left)   # 1. 递归左子树
            result.append(node.val)  # 2. 访问根节点
            inorder(node.right)  # 3. 递归右子树

        inorder(root)
        return result


# 完整测试用例
def build_tree(values):
    """从列表构建二叉树(层序遍历)"""
    if not values:
        return None

    root = TreeNode(values[0])
    queue = [root]
    i = 1

    while queue and i < len(values):
        node = queue.pop(0)

        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1

        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1

    return root

def test_inorder():
    sol = Solution()

    # 测试1:示例1
    root1 = build_tree([1, None, 2, 3])
    assert sol.inorderTraversal(root1) == [1, 3, 2], "示例1失败"

    # 测试2:示例2
    root2 = build_tree([1, 2, 3, 4, 5])
    assert sol.inorderTraversal(root2) == [4, 2, 5, 1, 3], "示例2失败"

    # 测试3:空树
    assert sol.inorderTraversal(None) == [], "空树失败"

    # 测试4:单节点
    root4 = TreeNode(1)
    assert sol.inorderTraversal(root4) == [1], "单节点失败"

    print("✅ 所有测试通过!")

test_inorder()
```

### 📊 复杂度分析
- **时间复杂度**:O(n)
  - 每个节点恰好被访问一次
  - n = 100 时,约100次操作

- **空间复杂度**:O(h),h为树高
  - 递归调用栈的深度等于树高
  - 最好情况(完全平衡树):O(log n)
  - 最坏情况(退化成链):O(n)
  - 平均情况:O(log n)

### ✅ 为什么是最优解
1. **时间最优**:O(n)是遍历所有节点的理论下限
2. **空间必要**:递归调用栈是遍历树所必需的(除非用Morris遍历)
3. **代码简洁**:仅5行核心代码,面试中最容易写对
4. **可读性强**:直接按定义实现,易于理解和解释

---

## ⚡ 解法二:迭代 + 栈

### 💡 核心思想
用显式的栈来模拟递归调用栈:
1. 一直往左走,沿途节点入栈
2. 栈顶出栈,访问该节点
3. 转向右子树,重复步骤1

**关键**:栈中存储的是"还未访问,但左子树已处理完"的节点。

### 📊 图解演示
```
示例:root = [1,2,3,4,5]
       1
      / \
     2   3
    / \
   4   5

迭代执行流程:

初始: curr = 1, stack = [], result = []

Step 1: 一直往左走,入栈
curr = 1 → stack = [1]
curr = 2 → stack = [1,2]
curr = 4 → stack = [1,2,4]
curr = None → 停止

Step 2: 栈顶出栈,访问
pop 4 → result = [4], curr = None(4的右子树)

Step 3: curr为空,继续出栈
pop 2 → result = [4,2], curr = 5(2的右子树)

Step 4: 处理curr=5
curr = 5 → stack = [1,5]
curr = None → 停止
pop 5 → result = [4,2,5], curr = None

Step 5: 继续出栈
pop 1 → result = [4,2,5,1], curr = 3(1的右子树)

Step 6: 处理curr=3
curr = 3 → stack = [3]
curr = None → 停止
pop 3 → result = [4,2,5,1,3], curr = None

栈空且curr为空 → 结束
最终结果:[4,2,5,1,3]
```

### 📝 代码实现
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        """迭代实现中序遍历"""
        result = []
        stack = []
        curr = root

        while curr or stack:
            # 1. 一直往左走,沿途节点入栈
            while curr:
                stack.append(curr)
                curr = curr.left

            # 2. 栈顶出栈,访问该节点
            node = stack.pop()
            result.append(node.val)

            # 3. 转向右子树
            curr = node.right

        return result
```

### 📊 复杂度分析
- **时间复杂度**:O(n) — 每个节点入栈出栈各一次
- **空间复杂度**:O(h) — 栈的最大深度等于树高

### ✅ 优点
- 不使用递归,避免栈溢出(对超深的树)
- 空间复杂度与递归相同
- 展示了对递归的深刻理解

### ⚠️ 缺点
- 代码复杂度比递归高
- 面试时容易写错边界条件

---

## 🐍 Pythonic 写法

### 技巧1:递归生成器(yield)
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        """使用生成器的递归实现"""
        def inorder(node):
            if node:
                yield from inorder(node.left)
                yield node.val
                yield from inorder(node.right)

        return list(inorder(root))
```
**优势**:`yield from` 更Pythonic,代码更简洁。

### 技巧2:一行递归(不推荐面试用)
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> list[int]:
        return (self.inorderTraversal(root.left) +
                [root.val] +
                self.inorderTraversal(root.right)) if root else []
```
**警告**:每次递归创建新列表,空间复杂度实际是O(n²),性能差。

---

## 📊 解法对比

| 维度 | 🏆 解法一:递归(最优) | 解法二:迭代+栈 |
|------|-------------------|-------------|
| 时间复杂度 | **O(n)** ← 最优 | O(n) |
| 空间复杂度 | **O(h)** ← 最优 | O(h) |
| 代码复杂度 | **极简(5行)** | 中等(10行) |
| 面试推荐 | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | **通用,面试标准答案** | 递归深度受限时 |

### 🏆 为什么递归是最优解
1. **时间空间均已最优**:O(n)和O(h)都是理论下限
2. **代码最简洁**:5行核心代码,面试中3分钟可写完
3. **可扩展性强**:改一个字符就能变成前序/后序遍历
4. **易于理解**:直接按定义实现,面试官一看就懂

### 💡 面试建议
1. **起手式**:直接说"我用递归实现,最简洁" ← 10秒
2. **🏆 重点讲解**:边写代码边解释三要素(定义、终止、调用) ← 2分钟
3. **写代码**:递归函数 + 测试用例 ← 3分钟
4. **追问准备**:"能用迭代实现吗?" → 展示解法二
5. **深度讨论**:提到Morris遍历(O(1)空间),展示技术深度

---

## 🎤 面试现场模拟

**面试官**:"实现二叉树的中序遍历。"

**你**:"明白。中序遍历就是'左-根-右'的顺序访问每个节点。我用递归实现,最简洁。" ← 展示思路

**你** *(开始写代码)*:
```python
def inorderTraversal(self, root):
    result = []

    def inorder(node):
        if not node:  # 递归终止
            return
        inorder(node.left)  # 左
        result.append(node.val)  # 根
        inorder(node.right)  # 右

    inorder(root)
    return result
```
← 边写边解释

**面试官**:"时间空间复杂度?"

**你**:"时间O(n),因为每个节点恰好访问一次。空间O(h),递归调用栈的深度等于树高,平均O(log n),最坏O(n)。" ← 详细分析

**面试官**:"能不用递归吗?"

**你**:"可以!用栈模拟递归调用栈。思路是一直往左走入栈,栈顶出栈访问,然后转向右子树。" ← 展示迭代解法

**面试官**:"空间复杂度能优化到O(1)吗?"

**你**:"能!Morris遍历利用叶子节点的空指针建立临时连接,实现O(1)空间。但代码复杂,工程中不推荐。" ← 展示深度

---

## ❓ 高频追问

| 追问 | 标准回答 |
|------|---------|
| 递归和迭代哪个更好? | 递归代码简洁易懂,是面试首选。迭代适合递归深度受限的场景,如超深的链式树。 |
| 为什么不用全局变量存result? | 函数应该无副作用。用内部变量result,函数可重入,便于测试和并发调用。 |
| 前序/后序遍历怎么改? | 前序:根左右 → 先访问根,再递归左右。后序:左右根 → 递归左右后访问根。只需调整3行代码顺序。 |
| Morris遍历是什么? | 利用叶子节点的空指针建立临时连接,遍历完后恢复。实现O(1)空间,但代码复杂,面试很少要求。 |
| 如果节点有父指针呢? | 可以不用栈,直接通过父指针回溯。但大多数题目节点没有父指针。 |
| 递归会栈溢出吗? | Python默认递归深度约1000。题目限制100节点,不会溢出。工程中可用sys.setrecursionlimit调整。 |

---

## 🐍 Python 技巧卡片

### 1. 生成器表达式
```python
# 使用 yield from 简化递归
def inorder(node):
    if node:
        yield from inorder(node.left)
        yield node.val
        yield from inorder(node.right)

result = list(inorder(root))
```

### 2. 列表推导嵌套(不推荐)
```python
# 虽然简洁,但性能差
def inorder(root):
    return (inorder(root.left) + [root.val] + inorder(root.right)) if root else []
```

### 3. 递归深度限制
```python
import sys
sys.setrecursionlimit(10000)  # 调整递归深度限制
```

### 4. 栈操作技巧
```python
# 检查栈是否为空
while stack:  # Pythonic写法
    node = stack.pop()
```

---

## 🔬 底层原理

### 递归调用栈的本质

递归本质是利用**函数调用栈**自动管理状态:

```
递归调用 inorder(1) 的调用栈:

栈底 | inorder(1) - 等待左子树返回
     | inorder(2) - 等待左子树返回
栈顶 | inorder(4) - 正在执行

当 inorder(4) 返回时:
- 栈顶弹出
- 回到 inorder(2),继续执行"访问根"
```

Python 函数调用栈存储:
- 局部变量
- 返回地址
- 参数值

### 迭代的栈模拟

迭代解法用**显式栈**代替隐式调用栈:

```python
# 递归隐式存储"下一步要访问根"
inorder(node.left)
result.append(node.val)  # 左子树返回后执行

# 迭代显式存储"待访问的节点"
stack.append(node)  # 记住这个节点
curr = node.left    # 先去左子树
node = stack.pop()  # 左子树处理完,回来访问
```

---

## 📋 算法模式卡片

**模式名称**:树的递归遍历

**适用场景**:
- 需要按特定顺序访问树的所有节点
- 前序/中序/后序遍历
- 树的路径问题、深度问题

**核心思想**:
利用树的递归定义,每个节点的处理方式和整棵树相同。

**通用模板**:
```python
def traverse(root):
    result = []

    def helper(node):
        if not node:
            return

        # 前序遍历:先访问根
        # result.append(node.val)

        helper(node.left)   # 递归左子树

        # 中序遍历:中间访问根
        result.append(node.val)

        helper(node.right)  # 递归右子树

        # 后序遍历:最后访问根
        # result.append(node.val)

    helper(root)
    return result
```

**变体题目**:
- LC 94:中序遍历(本题)
- LC 144:前序遍历(根左右)
- LC 145:后序遍历(左右根)
- LC 102:层序遍历(用队列BFS)

---

## ⚠️ 易错点

### 1. 忘记递归终止条件
```python
# ❌ 错误
def inorder(node):
    inorder(node.left)  # 如果node为空会报错!
    result.append(node.val)

# ✅ 正确
def inorder(node):
    if not node:  # 必须先判断
        return
    inorder(node.left)
```

### 2. 访问根的位置错误
```python
# ❌ 错误:这是前序遍历
def inorder(node):
    if not node:
        return
    result.append(node.val)  # 根在前
    inorder(node.left)
    inorder(node.right)

# ✅ 正确:中序遍历
def inorder(node):
    if not node:
        return
    inorder(node.left)
    result.append(node.val)  # 根在中间
    inorder(node.right)
```

### 3. 迭代解法循环条件错误
```python
# ❌ 错误
while curr:  # 只检查curr,栈为空时会提前退出
    ...

# ✅ 正确
while curr or stack:  # 两个条件都要检查
    ...
```

### 4. 全局变量导致重复
```python
# ❌ 错误
result = []  # 类变量,多次调用会累积
def inorderTraversal(self, root):
    self.result.append(...)
    return self.result

# ✅ 正确
def inorderTraversal(self, root):
    result = []  # 局部变量,每次调用独立
    def inorder(node):
        ...
    return result
```

---

## 🏗️ 工程实战(选读)

### 场景1:表达式树求值
**需求**:计算算术表达式树的值。

```python
# 中序遍历构建表达式字符串
def build_expression(root):
    """
    示例:
        +
       / \
      2   3
    输出:"2 + 3"
    """
    if not root:
        return ""

    if not root.left and not root.right:
        return str(root.val)  # 叶子节点直接返回值

    left_expr = build_expression(root.left)
    right_expr = build_expression(root.right)

    return f"({left_expr} {root.val} {right_expr})"
```

### 场景2:二叉搜索树验证
**需求**:验证是否为二叉搜索树(BST)。

```python
def is_valid_bst(root):
    """
    BST性质:中序遍历结果是严格递增的
    """
    result = []

    def inorder(node):
        if not node:
            return
        inorder(node.left)
        result.append(node.val)
        inorder(node.right)

    inorder(root)

    # 检查是否严格递增
    return all(result[i] < result[i+1] for i in range(len(result)-1))
```

### 场景3:序列化与反序列化
**需求**:将二叉树序列化为字符串。

```python
def serialize_inorder(root):
    """
    注意:单独用中序遍历无法唯一确定树
    需要配合前序或后序遍历
    """
    result = []

    def inorder(node):
        if not node:
            result.append('null')
            return
        inorder(node.left)
        result.append(str(node.val))
        inorder(node.right)

    inorder(root)
    return ','.join(result)
```

---

## 🏋️ 举一反三

### 相关题目

| 题目 | 难度 | 关键区别 |
|------|------|---------|
| **LC 144** - 二叉树前序遍历 | Easy | 访问顺序改为:根-左-右 |
| **LC 145** - 二叉树后序遍历 | Easy | 访问顺序改为:左-右-根 |
| **LC 102** - 二叉树层序遍历 | Medium | 用队列BFS,按层遍历 |
| **LC 98** - 验证二叉搜索树 | Medium | 利用中序遍历结果递增的性质 |
| **LC 230** - 二叉搜索树第K小元素 | Medium | 中序遍历BST得到有序序列,取第K个 |
| **LC 105** - 从前序与中序遍历序列构造二叉树 | Medium | 利用遍历序列的性质重建树 |

### 练习建议
1. **必做** LC 94(本题) + LC 144 + LC 145 — 掌握三种遍历
2. **进阶** LC 102 — 学习层序遍历(BFS)
3. **应用** LC 98 + LC 230 — 遍历的实际应用

---

## 📝 课后小测

<details>
<summary>💡 点击查看提示</summary>

**题目**:给定一个二叉树,返回它的 **之字形层序遍历** (第1层从左到右,第2层从右到左,第3层又从左到右...)

```
输入:root = [3,9,20,null,null,15,7]
    3
   / \
  9  20
    /  \
   15   7

输出:[[3], [20,9], [15,7]]
```

**提示**:
- 和中序遍历有什么区别?
- 需要用什么数据结构?
- 如何控制方向?

</details>

<details>
<summary>✅ 点击查看答案</summary>

**答案**:需要层序遍历(BFS) + 层级标记。

```python
def zigzagLevelOrder(root):
    if not root:
        return []

    result = []
    queue = [root]
    left_to_right = True  # 方向标记

    while queue:
        level_size = len(queue)
        level = []

        for _ in range(level_size):
            node = queue.pop(0)
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        # 根据方向决定是否反转
        if not left_to_right:
            level.reverse()

        result.append(level)
        left_to_right = not left_to_right  # 切换方向

    return result
```

**复杂度**:O(n) 时间,O(n) 空间(队列)

**核心区别**:
- 中序遍历:DFS,按"左-根-右"深度优先
- 层序遍历:BFS,按层从上到下广度优先
- 之字形遍历:BFS + 方向标记

</details>

---

**恭喜你完成第39课!** 🎉

你已经掌握了:
- ✅ 二叉树中序遍历的递归和迭代两种实现
- ✅ 树的递归思想和递归三要素
- ✅ 如何用栈模拟递归调用栈
- ✅ 前序/中序/后序遍历的区别
- ✅ 遍历在BST验证等场景的应用

**下一课预告**:第40课 - 二叉树最大深度(递归基础的进阶应用) 🌳

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
