# 📖 第52课:路径总和III

> **模块**:二叉树 | **难度**:Medium ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/path-sum-iii/
> **前置知识**:第39课(二叉树中序遍历)、第40课(二叉树最大深度)、第4课(和为K的子数组)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定一棵二叉树和一个目标和targetSum,找出树中路径和等于targetSum的路径数量。

**注意**:路径不需要从根节点开始,也不需要在叶子节点结束,但必须是向下的(即只能从父节点到子节点)。

**示例:**
```
输入:root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8

        10
       /  \
      5   -3
     / \    \
    3   2   11
   / \   \
  3  -2   1

输出:3
解释:和等于8的路径有:
  1. 5 -> 3
  2. 5 -> 2 -> 1
  3. -3 -> 11
```

**约束条件:**
- 树中节点数在范围 [0, 1000] 内
- -10^9 <= Node.val <= 10^9
- -1000 <= targetSum <= 1000

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空树 | root=null, target=0 | 0 | 空树处理 |
| 单节点匹配 | root=[8], target=8 | 1 | 基本功能 |
| 负数路径 | root=[1,-2,3], target=1 | 2 | 负数处理 |
| 长路径 | root=[1,1,1,1,1], target=3 | 3 | 多个路径 |
| 无匹配 | root=[1,2,3], target=10 | 0 | 无解情况 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在公司的组织架构图中找"工资总和等于100万"的团队组合。
>
> 🐌 **笨办法**:对每个员工,暴力枚举所有以他为起点的向下路径,逐一检查是否满足条件。如果有1000个员工,每个员工可能有10条路径,就要检查10000次!
>
> 🚀 **聪明办法**:你在走访每个员工时,随身带一个"从CEO到当前员工的历史工资记录本"。当你到达某个员工时,翻翻这个记录本:"如果从某个祖先到我的工资和等于100万,那就找到一个组合!"这样只需遍历一次所有员工,每次查本子都是瞬间完成。

### 关键洞察
**这道题是"和为K的子数组"(第4课)在树上的变体!用前缀和+哈希表,把子数组问题转化为树路径问题。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树根节点root + 目标和targetSum(整数,可为负)
- **输出**:满足条件的路径数量(整数)
- **限制**:路径必须向下(父→子),但不必从根节点开始或到叶子节点结束

### Step 2:先想笨办法(暴力法)
对每个节点作为路径起点,DFS遍历所有向下的路径,累加路径和判断是否等于target。
- 时间复杂度:O(n²) — 外层遍历n个节点,内层每个节点平均访问O(n)次
- 瓶颈在哪:**对每个节点都要重新遍历一遍向下路径,大量重复计算**

### Step 3:瓶颈分析 → 优化方向
暴力法中,我们多次计算"某段路径的和"。比如路径A→B→C→D,我们会分别计算:
- A→B, A→B→C, A→B→C→D
- B→C, B→C→D
- C→D

**核心问题**:如何避免重复计算路径和?
**优化思路**:用前缀和!记录"从根到当前节点的累积和",那么任意一段路径[A,B]的和 = prefix_sum[B] - prefix_sum[A的父节点]

### Step 4:选择武器
- 选用:**前缀和 + 哈希表**
- 理由:用哈希表记录"从根到当前路径上出现过的前缀和及其次数",在O(1)时间内查找"是否存在某个前缀和prefix_sum,使得当前和 - prefix_sum = targetSum"

> 🔑 **模式识别提示**:当题目出现"连续路径和"、"子数组和"时,优先考虑"前缀和+哈希表"模式

---

## 🔑 解法一:暴力DFS(直觉法)

### 思路
对每个节点作为起点,DFS向下遍历所有可能路径,累加路径和判断是否等于targetSum。

### 图解过程

```
示例: root = [10,5,-3,3,2,null,11], targetSum = 8

        10
       /  \
      5   -3
     / \    \
    3   2   11

Step 1: 以节点10为起点搜索
  路径10: sum=10 ≠ 8 ✗
  路径10→5: sum=15 ≠ 8 ✗
  路径10→5→3: sum=18 ≠ 8 ✗
  路径10→5→2: sum=17 ≠ 8 ✗
  路径10→-3: sum=7 ≠ 8 ✗
  路径10→-3→11: sum=18 ≠ 8 ✗

Step 2: 以节点5为起点搜索
  路径5: sum=5 ≠ 8 ✗
  路径5→3: sum=8 = 8 ✓ (找到1条)
  路径5→2: sum=7 ≠ 8 ✗

Step 3: 以节点-3为起点搜索
  路径-3: sum=-3 ≠ 8 ✗
  路径-3→11: sum=8 = 8 ✓ (找到1条)

...继续遍历其他节点

总共找到3条路径
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def pathSum(root: Optional[TreeNode], targetSum: int) -> int:
    """
    解法一:暴力DFS
    思路:对每个节点作为起点,DFS向下遍历所有路径
    """
    def count_paths_from_node(node: Optional[TreeNode], target: int) -> int:
        """从当前节点向下搜索所有路径"""
        if not node:
            return 0

        # 当前节点是否满足条件
        count = 1 if node.val == target else 0

        # 递归搜索左右子树(目标变为target - node.val)
        count += count_paths_from_node(node.left, target - node.val)
        count += count_paths_from_node(node.right, target - node.val)

        return count

    if not root:
        return 0

    # 以当前节点为起点的路径数
    result = count_paths_from_node(root, targetSum)

    # 递归搜索左右子树(让子树节点也作为起点)
    result += pathSum(root.left, targetSum)
    result += pathSum(root.right, targetSum)

    return result


# ✅ 测试
def build_tree():
    root = TreeNode(10)
    root.left = TreeNode(5)
    root.right = TreeNode(-3)
    root.left.left = TreeNode(3)
    root.left.right = TreeNode(2)
    root.right.right = TreeNode(11)
    root.left.left.left = TreeNode(3)
    root.left.left.right = TreeNode(-2)
    root.left.right.right = TreeNode(1)
    return root

print(pathSum(build_tree(), 8))  # 期望输出:3
print(pathSum(TreeNode(1), 1))   # 期望输出:1
print(pathSum(None, 0))          # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n²) — 外层遍历n个节点,内层每个节点平均向下访问O(n)个节点
  - 具体地说:如果树有1000个节点,最坏情况下需要约100万次操作
- **空间复杂度**:O(h) — 递归栈深度,h为树的高度(平衡树O(log n),链式树O(n))

### 优缺点
- ✅ 思路简单,易于理解和实现
- ✅ 不需要额外的数据结构
- ❌ 时间复杂度高,有大量重复计算
- ❌ 对于大型树会超时

---

## 🏆 解法二:前缀和+哈希表(最优解)

### 优化思路
核心思想:**路径[A,B]的和 = 前缀和[B] - 前缀和[A的父节点]**

类比"和为K的子数组"(第4课):
- 子数组问题:找`prefix_sum[j] - prefix_sum[i] = k`的配对
- 树路径问题:找`current_sum - prefix_sum = targetSum`的配对

用哈希表记录"从根到当前路径上出现过的前缀和及其次数",在O(1)时间内查找满足条件的路径。

> 💡 **关键想法**:在DFS过程中维护一个哈希表{前缀和: 出现次数},每到一个节点就问:"存在多少个历史前缀和prefix,使得current_sum - prefix = targetSum?"

### 图解过程

```
示例: root = [10,5,-3,3,2,null,11], targetSum = 8

初始:prefix_count = {0: 1}  # 前缀和为0出现1次(空路径)

        10 (current_sum=10)
        ├─ 查找: 10-8=2 在prefix_count中吗? 否
        ├─ 更新: prefix_count = {0:1, 10:1}
        │
        ├─ 左子树: 5 (current_sum=15)
        │   ├─ 查找: 15-8=7 在prefix_count中吗? 否
        │   ├─ 更新: prefix_count = {0:1, 10:1, 15:1}
        │   │
        │   ├─ 左子树: 3 (current_sum=18)
        │   │   ├─ 查找: 18-8=10 在prefix_count中吗? 是!次数=1 ✓
        │   │   │   (找到路径:5→3,因为prefix_sum=10是节点10处)
        │   │   ├─ 更新: prefix_count = {0:1, 10:1, 15:1, 18:1}
        │   │   └─ 回溯: prefix_count = {0:1, 10:1, 15:1} (删除18)
        │   │
        │   └─ 右子树: 2 (current_sum=17)
        │       ├─ 查找: 17-8=9 在prefix_count中吗? 否
        │       └─ ...
        │
        └─ 右子树: -3 (current_sum=7)
            ├─ 查找: 7-8=-1 在prefix_count中吗? 否
            ├─ 更新: prefix_count = {0:1, 10:1, 7:1}
            │
            └─ 右子树: 11 (current_sum=18)
                ├─ 查找: 18-8=10 在prefix_count中吗? 是!次数=1 ✓
                │   (找到路径:-3→11)
                └─ ...

关键点:前缀和哈希表让我们O(1)找到"某个祖先到当前节点的路径和是否等于targetSum"
```

### Python代码

```python
def pathSum_optimal(root: Optional[TreeNode], targetSum: int) -> int:
    """
    解法二:前缀和+哈希表(最优解)
    思路:用哈希表记录从根到当前路径的前缀和,O(1)查找满足条件的路径
    """
    def dfs(node: Optional[TreeNode], current_sum: int, prefix_count: dict) -> int:
        if not node:
            return 0

        # 更新当前前缀和
        current_sum += node.val

        # 查找:存在多少个前缀和prefix,使得current_sum - prefix = targetSum
        # 即:prefix = current_sum - targetSum
        count = prefix_count.get(current_sum - targetSum, 0)

        # 将当前前缀和加入哈希表
        prefix_count[current_sum] = prefix_count.get(current_sum, 0) + 1

        # 递归左右子树
        count += dfs(node.left, current_sum, prefix_count)
        count += dfs(node.right, current_sum, prefix_count)

        # 回溯:移除当前前缀和(避免影响其他分支)
        prefix_count[current_sum] -= 1

        return count

    # 初始化:前缀和为0出现1次(表示空路径)
    return dfs(root, 0, {0: 1})


# ✅ 测试
print(pathSum_optimal(build_tree(), 8))  # 期望输出:3
print(pathSum_optimal(TreeNode(1), 1))   # 期望输出:1
print(pathSum_optimal(None, 0))          # 期望输出:0

# 测试负数情况
root_negative = TreeNode(1)
root_negative.left = TreeNode(-2)
root_negative.right = TreeNode(3)
print(pathSum_optimal(root_negative, 1))  # 期望输出:2 (路径:1 和 -2→3)
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点访问一次,哈希表操作O(1)
  - 具体地说:如果树有1000个节点,只需约1000次操作,比暴力法快1000倍!
- **空间复杂度**:O(h) — 递归栈O(h) + 哈希表最多存储O(h)个前缀和(h为树高)

---

## 🐍 Pythonic 写法

利用 defaultdict 简化哈希表操作:

```python
from collections import defaultdict

def pathSum_pythonic(root: Optional[TreeNode], targetSum: int) -> int:
    """Pythonic写法:用defaultdict简化代码"""
    def dfs(node, current_sum):
        if not node:
            return 0

        current_sum += node.val
        count = prefix_count[current_sum - targetSum]

        prefix_count[current_sum] += 1
        count += dfs(node.left, current_sum) + dfs(node.right, current_sum)
        prefix_count[current_sum] -= 1  # 回溯

        return count

    prefix_count = defaultdict(int)
    prefix_count[0] = 1
    return dfs(root, 0)
```

这个写法用到了:
- `defaultdict(int)`:自动初始化不存在的键为0,省去`.get(key, 0)`
- 链式递归:`dfs(left) + dfs(right)`更简洁

> ⚠️ **面试建议**:先写清晰版本(解法二)展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:暴力DFS | 🏆 解法二:前缀和+哈希表(最优) |
|------|--------------|---------------------------|
| 时间复杂度 | O(n²) | **O(n)** ← 时间最优 |
| 空间复杂度 | O(h) | **O(h)** ← 相同 |
| 代码难度 | 简单 | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 小规模树或理解概念 | **面试首选,通用性强** |

**为什么是最优解**:
- 时间复杂度O(n)已经是理论最优(至少要遍历一遍所有节点)
- 用哈希表以O(h)的空间代价避免了O(n)的重复计算,性能提升n倍
- 这是"前缀和+哈希表"经典模式在树上的应用,展示对算法模式的深入理解

**面试建议**:
1. 先用30秒口述暴力法思路(O(n²)),表明你能想到基本解法
2. 立即优化到🏆最优解(O(n)前缀和+哈希表),展示优化能力
3. **重点讲解最优解的核心思想**:"用哈希表记录历史前缀和,O(1)查找满足条件的路径"
4. 类比第4课"和为K的子数组",说明这是同一个模式在树上的应用
5. 强调回溯的重要性:避免不同分支互相影响
6. 手动测试边界用例(空树、负数),展示对解法的深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找二叉树中路径和等于targetSum的路径数量,路径必须向下但不必从根开始。让我先想一下...

我的第一个想法是暴力法:对每个节点作为起点,DFS向下遍历所有路径,时间复杂度是O(n²)。

不过我注意到这道题和"和为K的子数组"(LeetCode 560)非常相似!我们可以用前缀和+哈希表优化到O(n)。核心思路是:在DFS过程中维护一个哈希表,记录从根到当前路径上出现过的前缀和及其次数。当我们到达某个节点时,查找"是否存在某个前缀和prefix,使得current_sum - prefix = targetSum"。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def pathSum(root, targetSum):
    def dfs(node, current_sum, prefix_count):
        if not node:
            return 0

        # 更新当前前缀和
        current_sum += node.val

        # 查找满足条件的路径数
        count = prefix_count.get(current_sum - targetSum, 0)

        # 将当前前缀和加入哈希表
        prefix_count[current_sum] = prefix_count.get(current_sum, 0) + 1

        # 递归左右子树
        count += dfs(node.left, current_sum, prefix_count)
        count += dfs(node.right, current_sum, prefix_count)

        # 回溯:移除当前前缀和
        prefix_count[current_sum] -= 1

        return count

    return dfs(root, 0, {0: 1})
```

关键点是:
1. 初始化`{0: 1}`,表示前缀和为0出现1次(空路径)
2. 查找`current_sum - targetSum`在哈希表中的次数
3. 回溯时要移除当前前缀和,避免影响其他分支

**面试官**:测试一下?

**你**:用示例`root=[10,5,-3,3,2,null,11], targetSum=8`走一遍:
- 节点10:current_sum=10,查找10-8=2,不存在,count=0
- 节点5:current_sum=15,查找15-8=7,不存在,count=0
- 节点3:current_sum=18,查找18-8=10,存在1次!count=1(找到路径5→3)
- ...
最终找到3条路径。

再测一个边界:空树返回0,单节点[8]目标8返回1,都正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么需要回溯?" | "因为DFS遍历树时,不同分支不应该共享前缀和信息。比如左子树的前缀和不应该影响右子树的判断,所以访问完左子树后要把前缀和移除,再访问右子树。" |
| "如果树非常大怎么办?" | "O(n)时间已经是最优,无法进一步优化时间。空间方面,哈希表最多存储O(h)个前缀和,h为树高,已经很高效。如果内存受限可以考虑迭代DFS减少递归栈开销。" |
| "能处理负数吗?" | "能!前缀和+哈希表的方法不受节点值正负影响。负数反而会让某些路径和增大或减小,但查找逻辑不变。" |
| "这个模式还能用在哪?" | "这是'前缀和+哈希表'经典模式,适用于所有'连续子序列/路径和等于K'的问题。比如LeetCode 560(和为K的子数组)、437(本题)、二维矩阵路径和等。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:字典的get方法 — 避免KeyError
count = prefix_count.get(key, 0)  # key不存在时返回0
# 等价于:
count = prefix_count[key] if key in prefix_count else 0

# 技巧2:defaultdict自动初始化
from collections import defaultdict
prefix_count = defaultdict(int)  # 访问不存在的key自动返回0
count = prefix_count[key]  # 不会KeyError

# 技巧3:前缀和计算
prefix_sum[i] = prefix_sum[i-1] + nums[i]
# 区间[i,j]的和 = prefix_sum[j] - prefix_sum[i-1]
```

### 💡 底层原理(选读)

> **为什么前缀和+哈希表这么强大?**
>
> 前缀和的本质是"累积信息",让我们O(1)查询任意区间和。哈希表的本质是"空间换时间",用O(n)空间换O(1)查找。
>
> 两者结合:**前缀和把问题转化为"找两个前缀和的差",哈希表让"找差值"从O(n)变为O(1)**。
>
> 在树上应用时,关键是理解"路径和 = 当前前缀和 - 某个祖先的前缀和",这样就把树的路径问题转化为了数组的子序列问题。
>
> **回溯的作用**:树的DFS遍历不同分支时,哈希表中的前缀和应该"只属于当前分支的祖先",所以访问完一个节点后要移除其前缀和,避免影响其他分支。这是树DFS中典型的"状态恢复"技巧。

### 算法模式卡片 📐
- **模式名称**:前缀和+哈希表(树路径版)
- **适用条件**:树中路径和相关问题,路径必须连续向下
- **识别关键词**:"路径和等于K"、"连续路径"、"子数组和"
- **模板代码**:
```python
def tree_path_sum(root, target):
    def dfs(node, current_sum, prefix_count):
        if not node:
            return 0

        current_sum += node.val
        count = prefix_count.get(current_sum - target, 0)

        prefix_count[current_sum] = prefix_count.get(current_sum, 0) + 1
        count += dfs(node.left, current_sum, prefix_count)
        count += dfs(node.right, current_sum, prefix_count)
        prefix_count[current_sum] -= 1  # 回溯

        return count

    return dfs(root, 0, {0: 1})
```

### 易错点 ⚠️
1. **忘记初始化`{0: 1}`**:前缀和为0表示"从根节点到当前节点的完整路径",必须初始化为1次。否则会漏掉根节点到某节点正好等于target的路径。
   - **错误**:`prefix_count = {}`
   - **正确**:`prefix_count = {0: 1}`

2. **忘记回溯**:DFS访问完一个节点后不移除其前缀和,会导致不同分支互相干扰。
   - **错误**:递归后不执行`prefix_count[current_sum] -= 1`
   - **正确**:递归后必须回溯,恢复哈希表状态

3. **混淆"路径和"与"前缀和"**:路径[A,B]的和 = prefix_sum[B] - prefix_sum[A的父节点],不是prefix_sum[B] - prefix_sum[A]。
   - **错误**:`count = prefix_count.get(current_sum, 0)`
   - **正确**:`count = prefix_count.get(current_sum - targetSum, 0)`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:文件系统路径分析** — 在文件目录树中查找"大小总和等于N MB"的子目录路径,用于磁盘清理工具。前缀和表示从根目录到当前目录的累积大小,哈希表快速找到满足条件的路径。

- **场景2:公司组织架构分析** — 在员工层级树中查找"工资总和等于预算"的团队组合,用于成本控制系统。前缀和表示从CEO到当前员工的累积工资,哈希表快速找到符合预算的团队。

- **场景3:网络流量监控** — 在网络拓扑树中查找"流量总和等于阈值"的路径,用于流量分析和异常检测。前缀和表示从根路由器到当前节点的累积流量,哈希表快速定位流量异常路径。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 560. 和为K的子数组 | Medium | 前缀和+哈希表(数组版) | 本题的"数组版本",先掌握这个再学本题更容易 |
| LeetCode 112. 路径总和 | Easy | DFS基础 | 判断是否存在根到叶子路径和等于target,是本题的简化版 |
| LeetCode 113. 路径总和II | Medium | DFS+回溯 | 找出所有根到叶子路径和等于target的路径,需要记录路径 |
| LeetCode 124. 二叉树中的最大路径和 | Hard | DFS+全局变量 | 路径可以不向下,需要考虑"拐弯"的情况 |
| LeetCode 666. 路径总和IV | Medium | 哈希表+DFS | 树用数组表示,需要先建树再DFS |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一棵二叉树和目标和targetSum,找出树中路径和等于targetSum的**最长路径长度**(路径上节点个数)。路径不需要从根节点开始,也不需要在叶子节点结束,但必须向下。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

在前缀和哈希表中不仅存"前缀和:出现次数",而是存"前缀和:(出现次数,最小深度)"。当找到满足条件的路径时,用当前深度减去历史最小深度,得到路径长度。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def longest_path_sum(root: Optional[TreeNode], targetSum: int) -> int:
    """找路径和等于targetSum的最长路径长度"""
    max_length = 0

    def dfs(node, current_sum, depth, prefix_map):
        nonlocal max_length
        if not node:
            return

        current_sum += node.val

        # 查找:是否存在前缀和prefix,使得current_sum - prefix = targetSum
        if current_sum - targetSum in prefix_map:
            # 路径长度 = 当前深度 - 历史最小深度
            length = depth - prefix_map[current_sum - targetSum]
            max_length = max(max_length, length)

        # 记录当前前缀和及其最小深度
        if current_sum not in prefix_map:
            prefix_map[current_sum] = depth

        # 递归左右子树
        dfs(node.left, current_sum, depth + 1, prefix_map)
        dfs(node.right, current_sum, depth + 1, prefix_map)

        # 回溯:移除当前前缀和(如果深度匹配)
        if prefix_map.get(current_sum) == depth:
            del prefix_map[current_sum]

    dfs(root, 0, 1, {0: 0})  # {0: 0}表示前缀和0在深度0
    return max_length
```

**核心思路**:用哈希表存储`{前缀和: 该前缀和首次出现的深度}`,这样路径长度 = 当前深度 - 历史深度。注意只保留每个前缀和的最小深度,这样才能得到最长路径。

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
