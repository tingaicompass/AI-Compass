# 📖 第47课:从前序与中序遍历序列构造二叉树

> **模块**:二叉树 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
> **前置知识**:第39课(二叉树中序遍历)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定两个整数数组`preorder`和`inorder`,其中`preorder`是二叉树的**前序遍历**结果,`inorder`是同一棵树的**中序遍历**结果。请构造并返回这棵二叉树。

**遍历顺序回顾**:
- **前序遍历**:根 → 左 → 右
- **中序遍历**:左 → 根 → 右

**示例:**
```
输入: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
输出: [3,9,20,null,null,15,7]

构造的树:
    3
   / \
  9  20
    /  \
   15   7

解释:
前序[3,9,20,15,7]: 先访问根3,再访问左子树9,最后访问右子树20,15,7
中序[9,3,15,20,7]: 先访问左子树9,再访问根3,最后访问右子树15,20,7
```

**约束条件:**
- 1 ≤ preorder.length ≤ 3000
- inorder.length == preorder.length
- -3000 ≤ preorder[i], inorder[i] ≤ 3000
- preorder和inorder均**无重复元素**
- inorder保证是preorder的中序遍历

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单节点 | preorder=[1], inorder=[1] | root=1 | 基本功能 |
| 左斜树 | preorder=[1,2,3], inorder=[3,2,1] | 1->2->3(左链) | 递归边界 |
| 右斜树 | preorder=[1,2,3], inorder=[1,2,3] | 1->2->3(右链) | 递归边界 |
| 完全树 | preorder=[1,2,4,5,3,6,7], inorder=[4,2,5,1,6,3,7] | 完全二叉树 | 复杂结构 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在玩一个拼图游戏,手里有两份线索:
>
> 📋 **线索1(前序遍历)**:告诉你"先访问哪个节点" — 第一个元素一定是根节点!
>
> 📋 **线索2(中序遍历)**:告诉你"哪些在左边,哪些在右边" — 找到根节点后,它左边的都是左子树,右边的都是右子树!
>
> 🐌 **笨办法**:每次从前序找根,然后在中序数组里线性扫描找位置,时间O(n²)。
>
> 🚀 **聪明办法**:提前用哈希表记录中序数组每个值的下标,找根位置只需O(1)!然后递归构造左右子树,时间优化到O(n)。

### 关键洞察
**前序遍历的第一个元素永远是根!找到根在中序遍历中的位置,就能划分左右子树范围,然后递归构造。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:两个数组,preorder(前序遍历)和inorder(中序遍历),元素无重复
- **输出**:构造出的二叉树的根节点
- **限制**:必须根据遍历结果唯一确定树的结构

### Step 2:先想笨办法(暴力递归)
1. 前序第一个元素是根节点
2. 在中序数组中找到根节点的位置(线性扫描O(n))
3. 根据位置划分左右子树的中序和前序数组
4. 递归构造左右子树

- 时间复杂度:O(n²) — 每层递归都要O(n)扫描找根位置,共n层
- 瓶颈在哪:**重复扫描中序数组**查找根节点位置

### Step 3:瓶颈分析 → 优化方向
- 核心问题:每次都要在中序数组中线性查找根节点位置
- 优化思路:能否提前预处理,把"值→下标"的映射存起来?用哈希表实现O(1)查找!

### Step 4:选择武器
- 选用:**递归分治 + 哈希表优化**
- 理由:
  - 递归分治符合"构造子树"的自然思路
  - 哈希表将查找从O(n)优化到O(1),总时间从O(n²)降到O(n)

> 🔑 **模式识别提示**:当题目出现"根据遍历结果构造树"、"前序/中序/后序",优先考虑"递归分治 + 哈希表定位"

---

## 🔑 解法一:朴素递归(未优化)

### 思路
直接根据前序和中序的性质递归构造:
1. 前序第一个元素是根
2. 在中序中找到根,划分左右子树
3. 递归构造左右子树

### 图解过程

```
示例: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]

Step 1: 构造根节点
  preorder[0] = 3 → 根节点是3
  在inorder中找3的位置: index=1

  中序数组划分:
    左子树inorder: [9] (index左边)
    右子树inorder: [15,20,7] (index右边)

  前序数组划分:
    根: [3]
    左子树preorder: [9] (长度与左子树inorder相同)
    右子树preorder: [20,15,7] (剩余部分)

Step 2: 递归构造左子树
  preorder=[9], inorder=[9]
  根节点=9,无左右子树

Step 3: 递归构造右子树
  preorder=[20,15,7], inorder=[15,20,7]
  根节点=20,在inorder中位置=1
  左子树inorder=[15], 右子树inorder=[7]
  左子树preorder=[15], 右子树preorder=[7]

  继续递归...

最终构造出:
    3
   / \
  9  20
    /  \
   15   7
```

### Python代码

```python
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def buildTree(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    解法一:朴素递归(未优化)
    思路:根据前序找根,在中序中划分左右子树
    """
    if not preorder:  # 空数组返回None
        return None

    # Step 1: 前序第一个元素是根
    root_val = preorder[0]
    root = TreeNode(root_val)

    # Step 2: 在中序中找根的位置(线性扫描)
    mid = inorder.index(root_val)  # O(n)时间

    # Step 3: 划分左右子树的中序数组
    left_inorder = inorder[:mid]
    right_inorder = inorder[mid+1:]

    # Step 4: 划分左右子树的前序数组(根据左子树大小)
    left_size = len(left_inorder)
    left_preorder = preorder[1:1+left_size]
    right_preorder = preorder[1+left_size:]

    # Step 5: 递归构造左右子树
    root.left = buildTree(left_preorder, left_inorder)
    root.right = buildTree(right_preorder, right_inorder)

    return root


# ✅ 测试
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
root = buildTree(preorder, inorder)
print(f"根节点:{root.val}, 左孩子:{root.left.val}, 右孩子:{root.right.val}")
# 期望输出:根节点:3, 左孩子:9, 右孩子:20
```

### 复杂度分析
- **时间复杂度**:O(n²) — 每次递归调用inorder.index()需要O(n),共n层递归
  - 具体地说:n=3000时,最坏情况约需要 3000² = 900万 次操作
- **空间复杂度**:O(n) — 递归栈深度O(n),切片创建新数组也是O(n)

### 优缺点
- ✅ 逻辑清晰,易于理解
- ✅ 直接使用数组切片,代码简洁
- ❌ 时间复杂度O(n²),大规模数据会超时
- ❌ 数组切片产生大量临时数组,空间浪费

---

## 🏆 解法二:哈希表优化 + 索引传递(最优解)

### 优化思路
两个关键优化:
1. **哈希表预处理**:提前构建inorder的`值→下标`映射,查找根位置从O(n)降到O(1)
2. **索引传递替代切片**:不创建新数组,只传递左右边界索引,避免空间浪费

> 💡 **关键想法**:不需要真的切分数组,只需要知道"当前处理的是哪个范围"!

### 图解过程

```
示例: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]

预处理: inorder_map = {9:0, 3:1, 15:2, 20:3, 7:4}

递归过程(用索引标记范围):

Step 1: build(preStart=0, inStart=0, inEnd=4)
  根 = preorder[0] = 3
  在map中查到3的位置: mid=1 (O(1)时间!)
  左子树大小 = mid - inStart = 1 - 0 = 1

  递归左子树: build(preStart=1, inStart=0, inEnd=0)
  递归右子树: build(preStart=2, inStart=2, inEnd=4)

Step 2: 左子树 build(preStart=1, inStart=0, inEnd=0)
  根 = preorder[1] = 9
  mid = 0
  左子树大小 = 0 (无左子树)
  右子树大小 = 0 (无右子树)
  返回节点9

Step 3: 右子树 build(preStart=2, inStart=2, inEnd=4)
  根 = preorder[2] = 20
  mid = 3
  左子树大小 = 1
  递归构造...

最终构造完成!
```

### Python代码

```python
def buildTree_v2(preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
    """
    解法二:哈希表优化 + 索引传递(最优解)
    思路:预处理哈希表,用索引替代切片
    """
    # Step 1: 预处理哈希表,记录inorder中每个值的下标
    inorder_map = {val: idx for idx, val in enumerate(inorder)}

    def build(pre_start, in_start, in_end):
        """
        构造子树
        pre_start: 当前子树在preorder中的根节点位置
        in_start, in_end: 当前子树在inorder中的范围[in_start, in_end]
        """
        # 递归终止条件
        if in_start > in_end:
            return None

        # Step 2: 创建根节点
        root_val = preorder[pre_start]
        root = TreeNode(root_val)

        # Step 3: 在哈希表中O(1)查找根在inorder中的位置
        mid = inorder_map[root_val]

        # Step 4: 计算左子树大小
        left_size = mid - in_start

        # Step 5: 递归构造左右子树
        # 左子树: preorder从pre_start+1开始,长度为left_size
        #        inorder从in_start到mid-1
        root.left = build(pre_start + 1, in_start, mid - 1)

        # 右子树: preorder从pre_start+1+left_size开始
        #        inorder从mid+1到in_end
        root.right = build(pre_start + 1 + left_size, mid + 1, in_end)

        return root

    # 初始调用:前序从0开始,中序范围[0, len-1]
    return build(0, 0, len(inorder) - 1)


# ✅ 测试
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
root = buildTree_v2(preorder, inorder)
print(f"根节点:{root.val}, 左孩子:{root.left.val}, 右孩子:{root.right.val}")
# 期望输出:根节点:3, 左孩子:9, 右孩子:20

# 边界测试:单节点
root2 = buildTree_v2([1], [1])
print(f"单节点:{root2.val}")  # 期望输出:1

# 左斜树
root3 = buildTree_v2([1,2,3], [3,2,1])
print(f"左斜树根:{root3.val}, 左孩子:{root3.left.val}")  # 期望输出:1, 2
```

### 复杂度分析
- **时间复杂度**:O(n) — 预处理哈希表O(n),每个节点访问一次O(n),总共O(n)
  - 具体地说:n=3000时,只需约 3000×2 = 6000 次操作,比O(n²)快1500倍!
- **空间复杂度**:O(n) — 哈希表O(n),递归栈O(h),总共O(n)

### 为什么这是最优解
- ✅ **时间最优**:O(n)已经是理论下限(至少要访问每个节点一次)
- ✅ **空间最优**:O(n)无法避免(哈希表和递归栈必需)
- ✅ **代码优雅**:索引传递避免了数组切片,内存友好
- ✅ **实战推荐**:面试中这是标准解法,展示了优化思维

---

## 🐍 Pythonic 写法

利用`enumerate()`和字典推导式简化哈希表构建:

```python
# 一行构建哈希表
inorder_map = {val: idx for idx, val in enumerate(inorder)}

# 或者用dict()
inorder_map = dict(enumerate(inorder))  # 错误!这样key是下标,val是值

# 正确写法必须是 {值: 下标}
```

**解构赋值简化测试**:
```python
# 如果需要验证树结构,可以用层序遍历打印
from collections import deque

def level_order(root):
    if not root:
        return []
    result, queue = [], deque([root])
    while queue:
        node = queue.popleft()
        result.append(node.val if node else None)
        if node:
            queue.append(node.left)
            queue.append(node.right)
    return result

print(level_order(root))  # [3, 9, 20, None, None, 15, 7]
```

> ⚠️ **面试建议**:哈希表优化是核心考点,必须掌握!先写暴力版展示思路,再优化展示功底。

---

## 📊 解法对比

| 维度 | 解法一:朴素递归 | 🏆 解法二:哈希表优化(最优) |
|------|--------------|---------------------------|
| 时间复杂度 | O(n²) | **O(n)** ← 最优 |
| 空间复杂度 | O(n) | **O(n)** ← 相同 |
| 代码难度 | 简单(直接切片) | **中等** ← 需理解索引传递 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 必须掌握 |
| 适用场景 | n<100的小数据 | **所有场景,尤其n>1000** |

**为什么解法二是最优解**:
- 时间复杂度从O(n²)优化到O(n),在n=3000时性能提升**1500倍**!
- 面试官期待看到"发现瓶颈→用哈希表优化"的思维过程
- 索引传递避免切片,展示了对内存的关注

**面试建议**:
1. 先花1分钟画图理解前序和中序的关系:"前序第一个是根,中序根的位置划分左右"
2. 口述暴力法:"可以用index()找根,但每次O(n),总共O(n²)"
3. 立即提出优化:"我可以预处理哈希表,把查找优化到O(1),总时间O(n)"
4. 重点讲解🏆最优解:边写边说索引的计算逻辑
5. **强调数学关系**:左子树大小 = mid - in_start,前序中右子树起点 = pre_start + 1 + left_size

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:给定一棵二叉树的前序和中序遍历结果,请你重建这棵树。

**你**:(审题30秒,画图)好的,我先理解一下。前序是"根-左-右",中序是"左-根-右"。所以前序的第一个元素一定是根,然后我在中序中找到根的位置,就能知道哪些是左子树,哪些是右子树了。

**面试官**:没错,说说你的思路。

**你**:我的思路是递归分治。每次从前序取第一个元素作为根,然后在中序中找它的位置。假设位置是mid,那么中序的[0, mid-1]是左子树,[mid+1, end]是右子树。前序的划分也类似,根据左子树大小来切分。递归构造左右子树即可。

不过直接这样做有个问题:每次在中序中找根位置需要O(n)时间,总共O(n²)。我可以预处理一个哈希表,存储中序数组的"值→下标"映射,这样查找只需O(1),总时间优化到O(n)。

**面试官**:很好,请实现哈希表优化的版本。

**你**:(边写边说)首先构建哈希表。然后写递归函数,参数是前序起点和中序范围。找到根在中序的位置后,计算左子树大小,递归构造左右子树。左子树的前序起点是pre_start+1,中序范围是[in_start, mid-1];右子树的前序起点是pre_start+1+左子树大小,中序范围是[mid+1, in_end]。

**面试官**:测试一下[3,9,20,15,7]和[9,3,15,20,7]。

**你**:(手动模拟)前序第一个是3,在中序中位置是1。左子树中序是[9],前序是[9];右子树中序是[15,20,7],前序是[20,15,7]。递归构造左子树得到节点9,递归右子树,根是20,继续划分...最终得到正确的树。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果给的是中序和后序呢?" | "后序是'左-右-根',根在最后。逻辑类似,从后序取最后一个元素作为根,在中序中划分左右,递归构造。同样用哈希表优化。" |
| "如果只给前序和后序能重建吗?" | "不能唯一确定!比如前序[1,2],后序[2,1],可以是1->2(左孩子)也可以是1->2(右孩子)。必须有中序才能区分。" |
| "数组有重复元素怎么办?" | "题目保证无重复,如果有重复,哈希表映射会失效,需要额外信息(如父节点指针)才能区分。" |
| "能用迭代实现吗?" | "理论上可以用栈模拟递归,但代码复杂且不直观,实际中不推荐。递归版已经很高效。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:字典推导式构建哈希表 — 简洁高效
inorder_map = {val: idx for idx, val in enumerate(inorder)}

# 技巧2:嵌套函数访问外层变量 — 避免传递大量参数
def outer():
    data = [1, 2, 3]
    def inner(idx):
        return data[idx]  # 直接访问外层data
    return inner(1)

# 技巧3:递归函数的参数设计 — 传索引比传切片高效
# ❌ 低效:build(preorder[1:], inorder[:mid])  # 创建新数组
# ✅ 高效:build(pre_start+1, in_start, mid-1)  # 只传数字
```

### 💡 底层原理(选读)

> **为什么前序+中序能唯一确定树?**
>
> - **前序**告诉你"根是谁"(第一个元素)
> - **中序**告诉你"左右子树的分界"(根左边是左子树,右边是右子树)
> - 两者结合,递归地就能确定每个子树的根和边界,唯一重建整棵树
>
> **为什么前序+后序不能唯一确定?**
>
> - 前序:根-左-右,后序:左-右-根
> - 都能找到根,但无法区分"第二个元素是左孩子还是右孩子"
> - 比如前序[1,2],后序[2,1]:可能是1->2(左)或1->2(右),有歧义!
>
> **哈希表为什么能O(1)查找?**
>
> Python的dict基于哈希表实现,通过哈希函数将key映射到数组下标,平均查找时间O(1)。哈希冲突时用链表或开放寻址解决。

### 算法模式卡片 📐
- **模式名称**:递归分治 + 哈希表优化
- **适用条件**:需要根据遍历结果重建树,且有重复查找操作
- **识别关键词**:"前序中序构造树"、"中序后序构造树"、"重建二叉树"
- **模板代码**:
```python
def build_tree(traversal1, traversal2):
    # Step 1: 预处理哈希表(如果需要查找)
    index_map = {val: idx for idx, val in enumerate(traversal2)}

    def build(start1, start2, end2):
        if start2 > end2:
            return None

        # Step 2: 找根(通常在traversal1的起点或终点)
        root_val = traversal1[start1]
        root = TreeNode(root_val)

        # Step 3: 在traversal2中定位根
        mid = index_map[root_val]

        # Step 4: 计算子树大小
        left_size = mid - start2

        # Step 5: 递归构造左右子树
        root.left = build(start1+1, start2, mid-1)
        root.right = build(start1+1+left_size, mid+1, end2)

        return root

    return build(0, 0, len(traversal2)-1)
```

### 易错点 ⚠️
1. **索引计算错误**:最容易出错的是右子树前序起点的计算。记住公式:`pre_start + 1 + left_size`,其中left_size = mid - in_start。画图标注索引关系能避免错误。
2. **递归边界**:当`in_start > in_end`时返回None,不要写成`in_start == in_end`(相等时还有一个节点)。
3. **哈希表key错误**:必须是`{值: 下标}`,不要反过来写成`{下标: 值}`。

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:序列化与反序列化** — 存储树结构时,常保存前序和中序(或层序和中序),加载时用此算法重建。比如数据库的B+树索引持久化。
- **场景2:语法树解析** — 编译器从token序列构建抽象语法树(AST),可以看作"从遍历序列重建树"的变体。
- **场景3:版本控制系统** — Git的commit树可以看作二叉树,从log记录重建commit历史树时用类似思想。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 106. 从中序与后序遍历序列构造二叉树 | Medium | 递归分治 | 后序最后一个是根,逻辑类似 |
| LeetCode 889. 根据前序和后序遍历构造二叉树 | Medium | 递归分治 | 不能唯一确定,返回任意一种即可 |
| LeetCode 297. 二叉树的序列化与反序列化 | Hard | BFS/DFS | 另一种重建树的方式 |
| LeetCode 108. 将有序数组转换为二叉搜索树 | Easy | 递归分治 | 类似思想:中间元素是根 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定中序遍历`inorder = [9,3,15,20,7]`和后序遍历`postorder = [9,15,7,20,3]`,请构造这棵二叉树。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

后序是"左-右-根",最后一个元素是根。在中序中找根位置,划分左右子树,递归构造。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def buildTreeFromInPost(inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
    """
    中序 + 后序构造树
    思路:后序最后一个是根,在中序中划分左右
    """
    inorder_map = {val: idx for idx, val in enumerate(inorder)}

    def build(in_start, in_end, post_start, post_end):
        if in_start > in_end:
            return None

        # 后序最后一个是根
        root_val = postorder[post_end]
        root = TreeNode(root_val)

        # 在中序中定位根
        mid = inorder_map[root_val]
        left_size = mid - in_start

        # 递归构造左右子树
        # 左子树:后序[post_start, post_start+left_size-1]
        root.left = build(in_start, mid-1, post_start, post_start+left_size-1)
        # 右子树:后序[post_start+left_size, post_end-1]
        root.right = build(mid+1, in_end, post_start+left_size, post_end-1)

        return root

    return build(0, len(inorder)-1, 0, len(postorder)-1)
```

关键区别:后序的根在**最后**,右子树在后序中紧挨着根。时间O(n),空间O(n)。

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
