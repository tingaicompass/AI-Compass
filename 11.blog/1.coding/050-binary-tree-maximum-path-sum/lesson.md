# 📖 第50课:最大路径和

> **模块**:二叉树 | **难度**:Hard ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/binary-tree-maximum-path-sum/
> **前置知识**:第39课(二叉树中序遍历)、第40课(二叉树最大深度)、第43课(二叉树的直径)
> **预计学习时间**:35分钟

---

## 🎯 题目描述

给定一个二叉树,找出其中任意一条路径的最大和。路径被定义为从树中任意节点出发,沿着父子连接到达任意节点的序列。同一节点在路径中最多出现一次,路径至少包含一个节点。

**示例:**
```
输入:root = [1,2,3]
     1
    / \
   2   3
输出:6
解释:路径 2->1->3 的和为 2+1+3=6,是最大的
```

**示例2:**
```
输入:root = [-10,9,20,null,null,15,7]
      -10
      /  \
     9   20
        /  \
       15   7
输出:42
解释:路径 15->20->7 的和为 15+20+7=42
```

**约束条件:**
- 树中节点数范围在 [1, 3*10^4]
- -1000 <= Node.val <= 1000
- 节点值可能为负数,这是关键约束

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单节点 | root=[5] | 5 | 基本功能 |
| 全负数 | root=[-3,-2,-1] | -1 | 负数处理 |
| 只有左子树 | root=[1,2,null] | 3 | 单侧路径 |
| 路径穿过根 | root=[1,2,3] | 6 | 完整路径 |
| 最优不过根 | root=[-10,9,20,null,null,15,7] | 42 | 局部最优 |

---

## 💡 思路引导

### 生活化比喻
> 想象你是一个登山者,站在山脉中的某个山峰上,想找到"海拔总和最高"的登山路线。
>
> 🐌 **笨办法**:枚举所有可能的路径(从任意节点到任意节点),计算每条路径的和,取最大值。这就像要走遍所有可能的登山路线才能找到最优的,效率极低。
>
> 🚀 **聪明办法**:站在每个山峰(节点)上,计算"以我为转折点的最高路径"是多少(左臂+我+右臂),同时记录下来,然后只向上汇报"我单侧最高的高度"(因为父节点只能选择一条路径)。这样遍历一次就能找到全局最优解。

### 关键洞察
**每个节点可以作为路径的"转折点",最优路径要么穿过当前节点(左+根+右),要么在左右子树中的某个节点处。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树的根节点 root
- **输出**:任意路径的最大和(整数)
- **限制**:
  - 路径可以从任意节点开始和结束
  - 同一节点在路径中只能出现一次
  - 节点值可能为负数

### Step 2:先想笨办法(暴力法)
枚举所有可能的路径:从每个节点出发,DFS搜索所有可达路径并计算和,记录最大值。
- 时间复杂度:O(n²) 到 O(n³)
- 瓶颈在哪:对每个节点都要重复计算子树的路径,大量重复计算

### Step 3:瓶颈分析 → 优化方向
暴力法的问题是:
- 重复计算:每个节点的子树路径被重复计算多次
- 枚举起点终点:实际上不需要枚举所有起点终点组合

优化思路:
- **一次DFS遍历**:每个节点只访问一次
- **后序遍历**:先处理子树,再处理当前节点
- **维护全局最大值**:每个节点处更新全局答案

### Step 4:选择武器
- 选用:**后序DFS + 全局变量**
- 理由:
  1. 后序遍历自底向上,先知道子树信息
  2. 每个节点可以做两件事:
     - 更新全局答案(考虑以当前节点为转折点的路径:左+根+右)
     - 向父节点返回单侧最大值(max(左, 右) + 根)

> 🔑 **模式识别提示**:当题目需要"全局最优+递归汇报"时,考虑"后序DFS+全局变量"模式

---

## 🔑 解法一:递归DFS + 路径枚举(暴力法)

### 思路
从每个节点出发,枚举所有可能的路径,计算路径和,记录最大值。虽然这个方法不是最优解,但帮助理解问题。

### 图解过程

```
示例:[-10,9,20,null,null,15,7]

       -10
       /  \
      9   20
         /  \
        15   7

暴力法思路:枚举所有路径
- 单节点路径:[-10], [9], [20], [15], [7] → 最大 20
- 两节点路径:[9,-10], [-10,20], [20,15], [20,7] → 最大 20
- 三节点路径:[9,-10,20], [-10,20,15], [-10,20,7], [15,20,7] → 最大 42 ✓

问题:需要枚举大量路径,效率低下
```

### Python代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxPathSum_brute(root: TreeNode) -> int:
    """
    解法一:暴力枚举路径(不推荐,仅供理解)
    思路:从每个节点出发尝试所有可能的路径
    """
    max_sum = float('-inf')

    def dfs_all_paths(node):
        nonlocal max_sum
        if not node:
            return

        # 以当前节点为起点,计算所有可能的向下路径
        def path_from_node(node, current_sum):
            nonlocal max_sum
            if not node:
                return
            current_sum += node.val
            max_sum = max(max_sum, current_sum)
            path_from_node(node.left, current_sum)
            path_from_node(node.right, current_sum)

        path_from_node(node, 0)
        dfs_all_paths(node.left)
        dfs_all_paths(node.right)

    dfs_all_paths(root)
    return max_sum


# ✅ 测试
root1 = TreeNode(1, TreeNode(2), TreeNode(3))
print(maxPathSum_brute(root1))  # 期望输出:6

root2 = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
print(maxPathSum_brute(root2))  # 期望输出:42
```

### 复杂度分析
- **时间复杂度**:O(n²) — 对每个节点(n个)都要DFS遍历其子树(最坏O(n))
  - 具体地说:如果树有1000个节点,最坏情况需要约 1000*1000 = 100万次操作
- **空间复杂度**:O(h) — 递归栈深度,h为树高

### 优缺点
- ✅ 思路直观,容易理解
- ❌ 效率低下,大量重复计算
- ❌ 实际上没有考虑"穿过某节点的路径",只考虑了从某节点向下的路径

---

## 🏆 解法二:后序DFS + 全局最大值(最优解)

### 优化思路
关键洞察:
1. **每个节点只需要向父节点汇报一个值**:从该节点出发向下的单侧最大路径和
2. **每个节点内部要做一件事**:计算以该节点为转折点的最大路径(左+根+右),更新全局答案
3. **负数剪枝**:如果某侧贡献为负,不如不要(取0)

> 💡 **关键想法**:用后序遍历,每个节点返回"我能给父节点提供的最大贡献",同时更新全局最优解

### 图解过程

```
示例:[-10,9,20,null,null,15,7]

步骤1:后序遍历到叶子节点
       -10
       /  \
      9   20
         /  \
       [15] [7]  ← 先处理叶子

节点15:
  - 左右都是None,返回值 = 0
  - 经过15的最大路径 = 0 + 15 + 0 = 15
  - 向上汇报:15
  - 全局最大值更新为 15

节点7:
  - 经过7的最大路径 = 0 + 7 + 0 = 7
  - 向上汇报:7
  - 全局最大值仍为 15

步骤2:处理节点20
       -10
       /  \
      9   [20]  ← 处理20
         /  \
        15   7

节点20:
  - 左子树贡献:15
  - 右子树贡献:7
  - 经过20的最大路径 = 15 + 20 + 7 = 42 ✓
  - 向上汇报:max(15, 7) + 20 = 35
  - 全局最大值更新为 42

步骤3:处理节点9
       -10
       /  \
     [9]  20  ← 处理9
         /  \
        15   7

节点9:
  - 左右都是None
  - 经过9的最大路径 = 9
  - 向上汇报:9
  - 全局最大值仍为 42

步骤4:处理根节点-10
      [-10]  ← 处理根
       /  \
      9   20
         /  \
        15   7

节点-10:
  - 左子树贡献:9
  - 右子树贡献:35
  - 经过-10的最大路径 = 9 + (-10) + 35 = 34
  - 全局最大值仍为 42

最终答案:42
```

### Python代码

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxPathSum(root: TreeNode) -> int:
    """
    解法二:后序DFS + 全局最大值(最优解)
    思路:后序遍历,每个节点返回单侧最大值,内部更新全局答案
    """
    max_sum = float('-inf')  # 全局最大路径和

    def dfs(node):
        """
        返回:从当前节点出发向下的单侧最大路径和
        副作用:更新全局max_sum(考虑以当前节点为转折点的路径)
        """
        nonlocal max_sum

        if not node:
            return 0

        # 后序遍历:先处理左右子树
        left_gain = max(dfs(node.left), 0)   # 左侧贡献,负数则舍弃
        right_gain = max(dfs(node.right), 0)  # 右侧贡献,负数则舍弃

        # 计算以当前节点为转折点的路径和(左臂+根+右臂)
        current_path_sum = node.val + left_gain + right_gain

        # 更新全局最大值
        max_sum = max(max_sum, current_path_sum)

        # 向父节点返回单侧最大值(只能选左或右其中一条)
        return node.val + max(left_gain, right_gain)

    dfs(root)
    return max_sum


# ✅ 测试
root1 = TreeNode(1, TreeNode(2), TreeNode(3))
print(maxPathSum(root1))  # 期望输出:6

root2 = TreeNode(-10, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
print(maxPathSum(root2))  # 期望输出:42

root3 = TreeNode(-3)
print(maxPathSum(root3))  # 期望输出:-3(单节点)

root4 = TreeNode(5, TreeNode(-2), TreeNode(3))
print(maxPathSum(root4))  # 期望输出:8 (5+3,舍弃-2)
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点访问恰好一次
  - 具体地说:如果树有1000个节点,只需要1000次操作,相比暴力法提升1000倍
- **空间复杂度**:O(h) — 递归栈深度,h为树高
  - 平衡树:O(log n)
  - 最坏链状树:O(n)

### 优缺点
- ✅ 时间最优:O(n)已经是理论最优(至少要访问每个节点一次)
- ✅ 逻辑清晰:后序遍历 + 全局变量,模式化解决
- ✅ 处理负数:通过max(gain, 0)巧妙剪枝
- ✅ 面试首选:代码简洁,容易讲清楚

---

## 🐍 Pythonic 写法

Python的nonlocal关键字让全局变量处理更优雅:

```python
def maxPathSum_pythonic(root: TreeNode) -> int:
    """使用nonlocal更新全局变量,代码更简洁"""
    max_sum = float('-inf')

    def dfs(node):
        nonlocal max_sum
        if not node:
            return 0

        # 一行处理左右子树,负数直接取0
        L, R = max(dfs(node.left), 0), max(dfs(node.right), 0)

        # 更新全局最大值并返回单侧最大值
        max_sum = max(max_sum, node.val + L + R)
        return node.val + max(L, R)

    dfs(root)
    return max_sum
```

**解释**:
- 用元组赋值 `L, R = ...` 让代码更紧凑
- 直接在返回语句中计算,减少临时变量

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:暴力枚举 | 🏆 解法二:后序DFS(最优) |
|------|--------------|---------------------|
| 时间复杂度 | O(n²) | **O(n)** ← 时间最优 |
| 空间复杂度 | O(h) | **O(h)** ← 空间最优 |
| 代码难度 | 中等(逻辑复杂) | 中等(需理解后序+全局变量) |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 仅用于理解问题 | **面试首选,工业标准** |

**为什么是最优解**:
- 时间O(n)已经是理论最优(必须访问每个节点才能确定答案)
- 空间O(h)是递归的固有开销,无法进一步优化
- 逻辑清晰:后序遍历的经典应用,符合树形DP模式

**面试建议**:
1. 先用30秒口述思路:"后序遍历,每个节点做两件事:更新全局答案,向上返回单侧最大值"
2. 强调关键点:
   - **负数剪枝**:用max(gain, 0)舍弃负贡献
   - **两个返回**:更新全局(左+根+右),返回单侧(根+max(左,右))
3. 手动trace示例,特别是负数节点的处理
4. 讨论时间复杂度:O(n)是最优,无法再优化

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请找出二叉树中任意路径的最大和。

**你**:(审题30秒)好的,这道题有几个关键点:
1. 路径可以从任意节点开始和结束
2. 节点值可能为负数
3. 需要考虑所有可能的路径

我的第一反应是暴力枚举所有路径,但这会达到O(n²)的复杂度。
更好的方法是用**后序DFS**:每个节点做两件事:
1. 内部计算"以我为转折点的路径和"(左+根+右),更新全局答案
2. 向父节点返回"我这条单侧最大能提供多少"(根+max(左,右))

时间复杂度O(n),每个节点访问一次。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def maxPathSum(root):
    max_sum = float('-inf')  # 全局最大值

    def dfs(node):
        nonlocal max_sum
        if not node:
            return 0

        # 后序遍历:先算左右子树的贡献
        left = max(dfs(node.left), 0)   # 负数就不要了
        right = max(dfs(node.right), 0)

        # 以当前节点为转折点的路径和
        max_sum = max(max_sum, node.val + left + right)

        # 向上只能返回一条路径
        return node.val + max(left, right)

    dfs(root)
    return max_sum
```

关键是这一行:`left = max(dfs(node.left), 0)`,如果子树贡献是负数,我们就舍弃它,不如不走那条路径。

**面试官**:测试一下?

**你**:用示例[-10,9,20,null,null,15,7]走一遍:
- 叶子15:返回15,全局更新为15
- 叶子7:返回7
- 节点20:左15+右7+根20=42,全局更新为42 ✓;向上返回35
- 叶子9:返回9
- 根-10:左9+右35+根-10=34,全局仍为42

再测一个边界用例,单节点[-3]:直接返回-3,正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么用后序遍历?" | "因为需要先知道左右子树的信息,才能计算当前节点的最优解,这是自底向上的,所以用后序" |
| "能不能不用全局变量?" | "可以,可以在递归函数中返回两个值:(单侧最大值, 全局最大值),用元组返回,但代码会稍微复杂一点" |
| "如果所有节点都是负数?" | "也能正确处理,最终会返回最大的负数(也就是绝对值最小的负数)" |
| "空间能优化吗?" | "递归栈是O(h)固有开销,除非改写成Morris遍历(极其复杂),但不推荐,空间O(h)是可接受的" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:nonlocal更新外层变量
def outer():
    count = 0
    def inner():
        nonlocal count  # 声明要修改外层变量
        count += 1
    inner()
    return count

# 技巧2:max剪枝负数贡献
left_gain = max(dfs(node.left), 0)  # 负数直接变0,舍弃负贡献

# 技巧3:float('-inf')作为初始最小值
max_sum = float('-inf')  # 这样即使所有节点都是负数也能正确处理
```

### 💡 底层原理(选读)

**为什么后序遍历适合这道题?**

树的遍历顺序:
- **前序**:根→左→右,适合"从根到叶"的信息传递(如路径记录)
- **中序**:左→根→右,适合BST的有序遍历
- **后序**:左→右→根,适合"从叶到根"的信息汇总(如子树信息、树形DP)

本题需要:
1. 先知道左右子树能提供的最大贡献(需要先处理子树)
2. 再在当前节点做决策(左+根+右 vs 只选一侧)

这正是后序遍历的强项:自底向上汇总信息。

**全局变量 vs 返回值?**

两种写法对比:
```python
# 方式1:全局变量(推荐,代码简洁)
max_sum = float('-inf')
def dfs(node):
    nonlocal max_sum
    # ...更新max_sum
    return single_path

# 方式2:返回元组(不推荐,代码复杂)
def dfs(node):
    # ...计算
    return (single_path, global_max)
```

面试中推荐方式1,因为逻辑更清晰,代码更简洁。

### 算法模式卡片 📐
- **模式名称**:后序DFS + 全局最优(树形DP)
- **适用条件**:
  - 需要从子树汇总信息到父节点
  - 全局答案可能在任意节点处产生
  - 需要向父节点返回某个值(单侧最优),同时更新全局答案
- **识别关键词**:
  - "二叉树中任意路径"
  - "最大/最小路径和"
  - "需要考虑穿过某节点的路径"
- **模板代码**:
```python
def tree_dp_with_global(root):
    global_ans = initial_value

    def dfs(node):
        nonlocal global_ans
        if not node:
            return base_value

        # 后序:先处理子树
        left = dfs(node.left)
        right = dfs(node.right)

        # 更新全局答案(考虑以当前节点为关键点的答案)
        global_ans = update_function(global_ans, left, right, node.val)

        # 向父节点返回值(只能选一条路径)
        return compute_return_value(left, right, node.val)

    dfs(root)
    return global_ans
```

### 易错点 ⚠️
1. **忘记处理负数**
   - 错误:`return node.val + dfs(node.left) + dfs(node.right)`
   - 问题:负数会减少路径和,应该舍弃
   - 正确:`return node.val + max(dfs(node.left), 0) + max(dfs(node.right), 0)`

2. **混淆"更新全局"和"返回给父节点"**
   - 错误:直接返回 `left + node.val + right` 给父节点
   - 问题:父节点连接后会形成"三叉路",违反路径定义(每个节点最多一次)
   - 正确:更新全局时用 `left+root+right`,返回给父节点只用 `node.val+max(left,right)`

3. **初始值设置错误**
   - 错误:`max_sum = 0`
   - 问题:如果所有节点都是负数,答案应该是"最大的负数",而不是0
   - 正确:`max_sum = float('-inf')`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:社交网络影响力分析**
  - 问题:在社交网络树中,找到"影响力最大的社交链"(用户A→B→C,影响力累加)
  - 应用:用类似的后序DFS,计算每条社交链的总影响力,找到最有价值的传播路径

- **场景2:企业组织架构优化**
  - 问题:在公司组织树中,找到"产出价值最大的项目团队链"
  - 应用:团队成员可能有负产出(成本),用max(0, gain)剪枝,找到最优团队组合

- **场景3:游戏技能树**
  - 问题:在技能树中,找到"属性加成最大的技能路径"
  - 应用:某些技能有负属性(debuff),用类似逻辑找到最优加点路径

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 543. 二叉树的直径 | Easy | 后序DFS+全局最大 | 把"路径和"改成"路径长度",其他逻辑完全相同 |
| LeetCode 687. 最长同值路径 | Medium | 后序DFS+全局最大 | 只在值相同时才累加,否则重置为0 |
| LeetCode 437. 路径总和III | Medium | DFS+前缀和 | 不需要连续路径,用前缀和技巧 |
| LeetCode 129. 求根到叶子节点数字之和 | Medium | DFS路径累加 | 前序遍历,向下传递累加值 |
| LeetCode 988. 从叶到根的最小字符串 | Medium | DFS+字符串路径 | 后序遍历,字符串比较 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定二叉树,找出从根到叶子节点的所有路径中,路径和最大的那条路径的路径和。注意:路径必须从根开始,到叶子结束。

示例:
```
输入:root = [1,2,3]
     1
    / \
   2   3
输出:4
解释:路径1->3的和为4,比1->2的和3大
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

从根到叶子的路径,用前序DFS向下传递累加和,到叶子时更新全局最大值。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def maxPathSumRootToLeaf(root: TreeNode) -> int:
    """从根到叶子的最大路径和"""
    max_sum = float('-inf')

    def dfs(node, current_sum):
        nonlocal max_sum
        if not node:
            return

        current_sum += node.val

        # 如果是叶子节点,更新全局最大值
        if not node.left and not node.right:
            max_sum = max(max_sum, current_sum)
            return

        # 前序遍历:向下传递累加和
        dfs(node.left, current_sum)
        dfs(node.right, current_sum)

    dfs(root, 0)
    return max_sum


# 测试
root = TreeNode(1, TreeNode(2), TreeNode(3))
print(maxPathSumRootToLeaf(root))  # 输出:4
```

**核心思路**:
- 与本题的区别:本题的路径可以在任意节点开始/结束,课后题必须从根到叶子
- 解法差异:用**前序遍历**向下传递累加和,而不是后序向上汇总
- 判断叶子:`if not node.left and not node.right`,只在叶子处更新答案

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
