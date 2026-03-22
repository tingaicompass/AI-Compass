# 📖 第53课:二叉搜索树中第K小

> **模块**:二叉树 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/kth-smallest-element-in-a-bst/
> **前置知识**:第39课(二叉树中序遍历)、第46课(验证BST)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一棵二叉搜索树(BST)的根节点root和一个整数k,返回BST中第k小的元素(从1开始计数)。

**二叉搜索树的性质**:
- 左子树的所有节点值 < 根节点值
- 右子树的所有节点值 > 根节点值
- 左右子树也都是二叉搜索树

**示例:**
```
输入:root = [3,1,4,null,2], k = 1

    3
   / \
  1   4
   \
    2

输出:1
解释:第1小的元素是1
```

```
输入:root = [5,3,6,2,4,null,null,1], k = 3

       5
      / \
     3   6
    / \
   2   4
  /
 1

输出:3
解释:按从小到大排序:[1,2,3,4,5,6],第3小是3
```

**约束条件:**
- 树中节点数为n,其中 1 <= k <= n <= 10^4
- 0 <= Node.val <= 10^4

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单节点 | root=[1], k=1 | 1 | 基本功能 |
| k=1最小 | root=[3,1,4], k=1 | 1 | 找最小值 |
| k=n最大 | root=[3,1,4], k=3 | 4 | 找最大值 |
| 完全左偏树 | root=[5,4,null,3,null,2,null,1], k=2 | 2 | 链式结构 |
| 完全右偏树 | root=[1,null,2,null,3], k=2 | 2 | 链式结构 |

---

## 💡 思路引导

### 生活化比喻
> 想象图书馆的书架按编号从小到大排列(类似BST的有序性)。
>
> 🐌 **笨办法**:把所有书都取下来放到一个箱子里,排序后找第K本。这样需要搬所有书(O(n)空间),还要排序(O(n log n)时间),太累了!
>
> 🚀 **聪明办法**:书架已经按顺序排好了!你只需要从最左边(最小的书)开始,依次向右走,数到第K本就停下。不需要取下所有书,也不需要排序,只要按顺序访问K本书就够了!
>
> 这就是**中序遍历**的威力:BST的中序遍历天然就是从小到大的有序序列!

### 关键洞察
**BST的中序遍历结果是递增序列!所以第K小的元素就是中序遍历的第K个节点。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:BST根节点root + 整数k(1 <= k <= n)
- **输出**:第k小的元素值(整数)
- **关键约束**:输入是BST,不是普通二叉树!这意味着有序性可利用

### Step 2:先想笨办法(暴力法)
遍历整棵树收集所有节点值,排序后返回第k个。
- 时间复杂度:O(n log n) — 遍历O(n) + 排序O(n log n)
- 空间复杂度:O(n) — 需要存储所有节点
- 瓶颈在哪:**忽略了BST的有序性,白白浪费了排序时间**

### Step 3:瓶颈分析 → 优化方向
暴力法的问题:**BST本身就有序,为什么还要排序?**

回忆BST的性质:左 < 根 < 右。这正是中序遍历的顺序!
- 中序遍历:左子树 → 根 → 右子树
- 对BST来说:小值 → 中值 → 大值

**核心问题**:能否直接得到有序序列?
**优化思路**:用中序遍历!遍历过程中数到第K个就停止,不需要遍历所有节点

### Step 4:选择武器
- 选用:**中序遍历**(递归或迭代)
- 理由:BST的中序遍历天然有序,只需遍历K个节点就能找到答案

> 🔑 **模式识别提示**:当题目出现"BST"+"第K小/大"时,优先考虑"中序遍历"模式

---

## 🔑 解法一:中序遍历收集全部节点(直觉法)

### 思路
先中序遍历整棵树,将所有节点值存入数组,然后返回数组的第k个元素(索引k-1)。

### 图解过程

```
示例:root = [5,3,6,2,4,null,null,1], k = 3

       5
      / \
     3   6
    / \
   2   4
  /
 1

中序遍历过程(左→根→右):
Step 1: 访问最左节点1 → result = [1]
Step 2: 回到节点2 → result = [1, 2]
Step 3: 访问节点3的左子树完毕,访问节点3 → result = [1, 2, 3]
Step 4: 访问节点4 → result = [1, 2, 3, 4]
Step 5: 回到根节点5 → result = [1, 2, 3, 4, 5]
Step 6: 访问节点6 → result = [1, 2, 3, 4, 5, 6]

返回 result[k-1] = result[2] = 3 ✓
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def kthSmallest(root: Optional[TreeNode], k: int) -> int:
    """
    解法一:中序遍历收集全部节点
    思路:利用BST中序遍历有序的特性,遍历全部节点后返回第k个
    """
    result = []

    def inorder(node: Optional[TreeNode]):
        """中序遍历:左→根→右"""
        if not node:
            return
        inorder(node.left)       # 遍历左子树
        result.append(node.val)  # 访问根节点
        inorder(node.right)      # 遍历右子树

    inorder(root)
    return result[k - 1]  # 返回第k个元素(索引k-1)


# ✅ 测试
def build_tree1():
    root = TreeNode(3)
    root.left = TreeNode(1)
    root.right = TreeNode(4)
    root.left.right = TreeNode(2)
    return root

def build_tree2():
    root = TreeNode(5)
    root.left = TreeNode(3)
    root.right = TreeNode(6)
    root.left.left = TreeNode(2)
    root.left.right = TreeNode(4)
    root.left.left.left = TreeNode(1)
    return root

print(kthSmallest(build_tree1(), 1))  # 期望输出:1
print(kthSmallest(build_tree2(), 3))  # 期望输出:3
print(kthSmallest(TreeNode(1), 1))    # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(n) — 需要遍历所有n个节点
  - 具体地说:如果树有1000个节点,即使k=1,也要访问全部1000个节点
- **空间复杂度**:O(n) — 需要存储所有节点值 + 递归栈O(h)

### 优缺点
- ✅ 代码简单,易于理解
- ✅ 可以一次性求出多个"第K小"(如果有多次查询)
- ❌ 必须遍历所有节点,即使k很小(如k=1)
- ❌ 空间占用大,存储了不需要的节点

---

## 🏆 解法二:中序遍历提前终止(最优解)

### 优化思路
核心改进:**不需要遍历所有节点!** 在中序遍历过程中维护一个计数器,访问到第K个节点就立即返回,不再继续遍历。

> 💡 **关键想法**:中序遍历访问的第K个节点就是第K小的元素,后面的节点都比它大,不需要再看

### 图解过程

```
示例:root = [5,3,6,2,4,null,null,1], k = 3

       5
      / \
     3   6
    / \
   2   4
  /
 1

中序遍历过程(提前终止):
Step 1: 访问节点1 → count=1 (第1小)
Step 2: 访问节点2 → count=2 (第2小)
Step 3: 访问节点3 → count=3 (第3小,找到答案!) ✓
        立即返回3,不再访问节点4、5、6

优化效果:只访问了3个节点,而不是全部6个节点!
```

### Python代码

```python
def kthSmallest_optimal(root: Optional[TreeNode], k: int) -> int:
    """
    解法二:中序遍历提前终止(最优解)
    思路:维护计数器,访问到第k个节点立即返回
    """
    count = 0
    result = None

    def inorder(node: Optional[TreeNode]) -> bool:
        """中序遍历,找到第k个节点后返回True表示终止"""
        nonlocal count, result
        if not node:
            return False

        # 遍历左子树
        if inorder(node.left):
            return True  # 左子树已找到,提前终止

        # 访问当前节点
        count += 1
        if count == k:
            result = node.val
            return True  # 找到第k个,提前终止

        # 遍历右子树
        return inorder(node.right)

    inorder(root)
    return result


# ✅ 测试
print(kthSmallest_optimal(build_tree1(), 1))  # 期望输出:1
print(kthSmallest_optimal(build_tree2(), 3))  # 期望输出:3
print(kthSmallest_optimal(TreeNode(1), 1))    # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(h + k) — h为树高(到达最左节点) + k个节点访问
  - 最好情况:O(k) — 完全右偏树,第k个节点在浅层
  - 最坏情况:O(n) — 完全左偏树且k=n,需要遍历所有节点
  - 平均情况:O(log n + k) — 平衡BST,h=log n
  - 具体地说:如果k=1,只需访问约log n个节点,远少于n!
- **空间复杂度**:O(h) — 递归栈深度

---

## ⚡ 解法三:迭代中序遍历(空间优化)

### 优化思路
用迭代代替递归,显式使用栈模拟递归过程,避免递归栈开销。对于超深的树(如链式BST),迭代版更安全。

### Python代码

```python
def kthSmallest_iterative(root: Optional[TreeNode], k: int) -> int:
    """
    解法三:迭代中序遍历
    思路:用栈模拟递归,更直观地控制遍历过程
    """
    stack = []
    current = root
    count = 0

    while current or stack:
        # 一直向左走,将所有左子树节点入栈
        while current:
            stack.append(current)
            current = current.left

        # 弹出栈顶节点(当前最小节点)
        current = stack.pop()
        count += 1

        # 找到第k小的元素
        if count == k:
            return current.val

        # 转向右子树
        current = current.right

    return -1  # 不会到这里(题目保证k有效)


# ✅ 测试
print(kthSmallest_iterative(build_tree1(), 1))  # 期望输出:1
print(kthSmallest_iterative(build_tree2(), 3))  # 期望输出:3
```

### 图解迭代过程

```
示例:root = [5,3,6,2,4], k = 3

       5
      / \
     3   6
    / \
   2   4

迭代中序遍历过程:
Step 1: current=5,向左走到底
        stack = [5, 3, 2], current=null

Step 2: 弹出2,count=1,第1小
        stack = [5, 3], current=2
        2无右子树,current=null

Step 3: 弹出3,count=2,第2小
        stack = [5], current=3
        转向3的右子树,current=4

Step 4: 4向左走到底(4无左子树)
        stack = [5, 4], current=null

Step 5: 弹出4,count=3,第3小 ✓
        返回4
```

### 复杂度分析
- **时间复杂度**:O(h + k) — 与递归版相同
- **空间复杂度**:O(h) — 栈最多存储h个节点(h为树高)

---

## 🐍 Pythonic 写法

利用生成器简化中序遍历:

```python
def kthSmallest_pythonic(root: Optional[TreeNode], k: int) -> int:
    """Pythonic写法:用生成器实现惰性求值"""
    def inorder(node):
        """生成器:按中序遍历yield节点值"""
        if node:
            yield from inorder(node.left)   # 左子树
            yield node.val                  # 根节点
            yield from inorder(node.right)  # 右子树

    # 使用itertools跳过前k-1个,返回第k个
    from itertools import islice
    return next(islice(inorder(root), k - 1, k))


# 更简洁的版本(直接用enumerate)
def kthSmallest_pythonic_v2(root: Optional[TreeNode], k: int) -> int:
    """用enumerate计数"""
    def inorder(node):
        if node:
            yield from inorder(node.left)
            yield node.val
            yield from inorder(node.right)

    for i, val in enumerate(inorder(root), 1):
        if i == k:
            return val
```

这个写法用到了:
- `yield from`:递归地yield生成器的所有元素
- `islice`:切片生成器,实现惰性求值(只生成需要的k个元素)
- 生成器天然支持"提前终止",不会生成不需要的元素

> ⚠️ **面试建议**:先写清晰的递归或迭代版本,再提生成器写法展示Python功底。
> 但要能解释为什么生成器更优雅:惰性求值、代码简洁、自动提前终止。

---

## 📊 解法对比

| 维度 | 解法一:收集全部节点 | 🏆 解法二:提前终止(最优) | 解法三:迭代版 |
|------|------------------|----------------------|-------------|
| 时间复杂度 | O(n) | **O(h + k)** ← 时间最优 | O(h + k) |
| 空间复杂度 | O(n) | O(h) | **O(h)** ← 空间最优 |
| 代码难度 | 简单 | 中等 | 中等 |
| 提前终止 | ✗ | ✓ | ✓ |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 需要多次查询不同k | **单次查询,k较小** | 避免递归栈溢出 |

**为什么解法二是最优解**:
- 时间复杂度O(h + k)优于O(n),当k较小时优势明显(k=1只需O(log n))
- 空间复杂度O(h)优于O(n),不需要存储所有节点
- 提前终止机制:找到第k个节点后立即返回,不浪费时间
- 在平衡BST中,h=log n,整体复杂度O(log n + k)非常高效

**权衡说明**:
- 如果只查询一次,选择🏆解法二(提前终止版)
- 如果树很深担心栈溢出,选择解法三(迭代版)
- 如果需要多次查询不同k值,可以一次性收集全部节点(解法一),后续查询O(1)

**面试建议**:
1. 先用10秒提到暴力法:遍历所有节点排序,O(n log n)
2. 立即优化到🏆最优解:利用BST中序遍历有序特性,O(h + k)
3. **重点讲解核心洞察**:"BST的中序遍历天然有序,第k个访问的节点就是第k小"
4. 说明提前终止机制:找到答案后立即返回,不遍历后续节点
5. 如果面试官问"能否不用递归",展示迭代版(解法三)
6. 强调平衡BST的优势:h=log n时复杂度为O(log n + k),比O(n)快很多

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题20秒)好的,这道题要求找BST中第k小的元素。让我先想一下...

第一个想法是遍历所有节点,排序后返回第k个,时间O(n log n)。但我注意到这是BST而不是普通二叉树,可以利用BST的有序性!

BST有个重要性质:**中序遍历的结果是递增序列**。所以我们可以中序遍历树,访问的第k个节点就是第k小的元素。时间复杂度O(h + k),h是树高,k是我们要找的位置。在平衡树中h=log n,比排序快很多!

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def kthSmallest(root, k):
    count = 0
    result = None

    def inorder(node):
        nonlocal count, result
        if not node:
            return False

        # 先遍历左子树
        if inorder(node.left):
            return True  # 左子树已找到,提前返回

        # 访问当前节点
        count += 1
        if count == k:
            result = node.val
            return True  # 找到第k个,提前返回

        # 再遍历右子树
        return inorder(node.right)

    inorder(root)
    return result
```

关键点是:
1. 用计数器count记录访问了多少个节点
2. 中序遍历顺序:左→根→右,保证从小到大
3. 找到第k个节点后返回True,提前终止后续遍历

**面试官**:测试一下?

**你**:用示例`root=[5,3,6,2,4,null,null,1], k=3`走一遍:
- 中序遍历访问顺序:1(count=1) → 2(count=2) → 3(count=3,找到!)
- 返回3,不再访问节点4、5、6,提前终止

边界情况:k=1返回最小值,k=n返回最大值,单节点树返回该节点,都正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果要频繁查询不同的k呢?" | "如果查询次数远多于n,可以考虑在BST节点中额外存储'左子树节点数',这样可以O(log n)直接定位第k小节点,不需要每次都遍历。或者一次性中序遍历存储所有节点值,后续查询O(1)。" |
| "能用迭代而不是递归吗?" | "可以!用栈模拟中序遍历。"(展示解法三代码)"迭代版的好处是避免深度递归导致的栈溢出,适合链式BST。" |
| "时间复杂度最好是多少?" | "最好O(k),当树右偏且第k个节点在浅层时。平衡BST的平均情况是O(log n + k),因为需要先走到最左节点(O(log n)),然后访问k个节点。" |
| "空间能到O(1)吗?" | "递归版至少O(h)因为递归栈。迭代版也需要O(h)的栈空间。如果要严格O(1)空间,可以用Morris遍历(线索二叉树),但代码复杂度高,面试中通常不要求。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:nonlocal关键字 — 在嵌套函数中修改外层变量
def outer():
    count = 0
    def inner():
        nonlocal count  # 声明count是外层变量
        count += 1
    inner()
    return count

# 技巧2:生成器的yield from — 递归生成器
def inorder(node):
    if node:
        yield from inorder(node.left)   # 递归yield所有左子树节点
        yield node.val                  # yield当前节点
        yield from inorder(node.right)  # 递归yield所有右子树节点

# 技巧3:enumerate从1开始计数
for i, val in enumerate(items, 1):  # 从1开始而不是0
    print(f"第{i}个元素是{val}")
```

### 💡 底层原理(选读)

> **为什么BST的中序遍历是有序的?**
>
> 这源于BST的定义和中序遍历的顺序:
> - BST定义:左子树 < 根 < 右子树(所有节点递归满足)
> - 中序遍历:左子树 → 根 → 右子树
>
> 数学归纳法证明:
> 1. **基础情况**:单节点树,中序遍历就是该节点,天然有序
> 2. **归纳假设**:假设左右子树中序遍历都是有序的
> 3. **归纳步骤**:中序遍历先访问左子树(都比根小),再访问根,最后访问右子树(都比根大),所以整体有序
>
> **提前终止的原理**:
> 中序遍历访问的第k个节点就是第k小,因为:
> - 前k-1个节点都比它小(BST性质 + 中序遍历顺序)
> - 后面的节点都比它大(BST性质)
> 所以找到第k个节点后,不需要再访问后续节点
>
> **时间复杂度分析**:
> - 最好O(k):树右偏,最小的k个节点都在浅层
> - 最坏O(n):树左偏且k=n,需要遍历所有节点
> - 平均O(log n + k):平衡BST,先走到最左节点O(log n),再访问k个节点O(k)

### 算法模式卡片 📐
- **模式名称**:BST中序遍历
- **适用条件**:二叉搜索树(BST)相关问题,需要利用有序性
- **识别关键词**:"BST"、"第K小/大"、"有序"、"范围查找"
- **模板代码**:
```python
# 递归中序遍历(提前终止版)
def inorder_kth(root, k):
    count = 0
    result = None

    def inorder(node):
        nonlocal count, result
        if not node or result is not None:
            return

        inorder(node.left)  # 左
        count += 1
        if count == k:
            result = node.val
            return
        inorder(node.right)  # 右

    inorder(root)
    return result

# 迭代中序遍历
def inorder_iterative(root):
    stack, current = [], root
    while current or stack:
        while current:
            stack.append(current)
            current = current.left
        current = stack.pop()
        yield current.val  # 处理节点
        current = current.right
```

### 易错点 ⚠️
1. **忘记BST的中序遍历是有序的**:把BST当成普通二叉树,用排序解决,浪费了BST的有序性。
   - **错误**:遍历所有节点,排序后返回第k个
   - **正确**:直接中序遍历,访问第k个节点

2. **中序遍历顺序错误**:写成"根→左→右"或"右→根→左",导致结果不是有序的。
   - **错误**:`inorder(node.right) → visit(node) → inorder(node.left)`
   - **正确**:`inorder(node.left) → visit(node) → inorder(node.right)` (左→根→右)

3. **忘记提前终止**:找到第k个节点后继续遍历,浪费时间。
   - **错误**:遍历所有节点后返回第k个
   - **正确**:找到第k个节点立即返回True,终止后续遍历

4. **迭代版栈操作错误**:弹出节点后忘记转向右子树,或重复入栈。
   - **错误**:弹出节点后没有`current = current.right`
   - **正确**:弹出节点处理后,必须转向右子树继续遍历

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:数据库索引** — 数据库的B+树索引(BST的变体)用中序遍历实现范围查询。比如"查找年龄在25-30岁的用户",就是中序遍历B+树找到第一个满足条件的节点,然后按序返回直到超出范围。

- **场景2:排行榜系统** — 游戏排行榜用BST存储玩家分数,要查"第100名的分数"就是找第100小的节点。中序遍历可以高效返回Top-K排名,复杂度O(log n + k)。

- **场景3:中位数查询** — 在动态数据流中维护中位数,可以用两个堆(大顶+小顶)或一棵平衡BST。用BST时,中位数就是第n/2小的元素,中序遍历到一半即可。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 94. 二叉树中序遍历 | Easy | 中序遍历基础 | 本题的前置知识,掌握递归和迭代两种写法 |
| Leetcode 98. 验证BST | Medium | BST性质+中序遍历 | 利用中序遍历结果必须严格递增 |
| LeetCode 285. BST中序后继 | Medium | 中序遍历+查找 | 找比给定节点大的最小节点(中序遍历的下一个) |
| LeetCode 530. BST最小绝对差 | Easy | 中序遍历+相邻节点 | 中序遍历相邻节点的差值最小 |
| LeetCode 700. BST中的搜索 | Easy | BST查找 | 利用BST性质,O(log n)查找,不需要遍历 |
| LeetCode 450. 删除BST中的节点 | Medium | BST删除操作 | 删除后要保持BST性质,涉及中序前驱/后继 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一棵BST和两个值low、high,返回BST中所有节点值在[low, high]范围内的节点值之和。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

利用BST的性质和中序遍历:
- 如果当前节点值 < low,不需要访问左子树(都比low小)
- 如果当前节点值 > high,不需要访问右子树(都比high大)
- 如果节点值在范围内,加入总和并继续遍历左右子树

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def rangeSumBST(root: Optional[TreeNode], low: int, high: int) -> int:
    """BST范围求和:利用BST性质剪枝"""
    if not root:
        return 0

    # 如果当前节点值小于low,只需要访问右子树
    if root.val < low:
        return rangeSumBST(root.right, low, high)

    # 如果当前节点值大于high,只需要访问左子树
    if root.val > high:
        return rangeSumBST(root.left, low, high)

    # 当前节点在范围内,加上左右子树的和
    return (root.val +
            rangeSumBST(root.left, low, high) +
            rangeSumBST(root.right, low, high))


# 迭代中序遍历版本
def rangeSumBST_iterative(root: Optional[TreeNode], low: int, high: int) -> int:
    """迭代中序遍历版本"""
    stack = []
    current = root
    total = 0

    while current or stack:
        # 向左走到底,但遇到小于low的节点可以剪枝
        while current:
            if current.val < low:
                current = current.right  # 剪枝:左子树都小于low
                break
            stack.append(current)
            current = current.left

        if not current and stack:
            current = stack.pop()

            # 如果节点值大于high,后续节点都大于high,提前终止
            if current.val > high:
                break

            # 累加范围内的节点值
            if low <= current.val <= high:
                total += current.val

            current = current.right

    return total
```

**核心思路**:利用BST性质进行剪枝,避免访问不在范围内的子树。时间复杂度O(h + k),k为范围内节点数。这比暴力遍历所有节点O(n)更高效!

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
