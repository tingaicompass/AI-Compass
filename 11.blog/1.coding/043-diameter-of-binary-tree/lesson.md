> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第43课:二叉树的直径

> **模块**:二叉树 | **难度**:Easy ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/diameter-of-binary-tree/
> **前置知识**:第39课(二叉树中序遍历)、第40课(二叉树最大深度)
> **预计学习时间**:20分钟

---

## 🎯 题目描述

给你一棵二叉树的根节点,返回该树的**直径**。二叉树的直径是指树中任意两个节点之间**最长路径的长度**。这条路径可能经过也可能不经过根节点。两节点之间的路径长度由它们之间的**边数**表示。

**示例:**
```
输入:root = [1,2,3,4,5]
      1
     / \
    2   3
   / \
  4   5
输出:3
解释:最长路径是 [4,2,1,3] 或者 [5,2,1,3],长度为3条边
```

**约束条件:**
- 树中节点数量范围:[1, 10⁴]
- -100 ≤ Node.val ≤ 100

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单节点 | root = [1] | 0 | 基本功能(无边) |
| 链式树 | root = [1,2,null,3] | 2 | 退化成链表 |
| 完全二叉树 | root = [1,2,3,4,5,6,7] | 4 | 直径经过根节点 |
| 偏斜树 | 左子树深度3,右子树深度1 | 4 | 直径不经过某侧 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在测量一棵真实的大树,要找出树冠最宽的地方(两端树叶之间的最远距离)。
>
> 🐌 **笨办法**:爬到每个树枝上,从这个位置向下探索所有可能的路径,记录最远距离。这样要重复爬很多次树,太累了!
>
> 🚀 **聪明办法**:从树根开始,每测量一个树枝时,顺便记录"经过这个树枝的最宽跨度 = 左边深度 + 右边深度"。一次爬树就能找到答案!

### 关键洞察
**直径 = 某个节点的左子树深度 + 右子树深度的最大值**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树根节点 root(TreeNode 类型)
- **输出**:整数,表示树的直径(边数,不是节点数)
- **限制**:直径可能不经过根节点,需要考察所有节点

### Step 2:先想笨办法(暴力法)
对每个节点,计算"左子树最大深度 + 右子树最大深度",然后取全局最大值。但如果分别计算每个节点的左右深度,会重复遍历很多次。
- 时间复杂度:O(n²)
- 瓶颈在哪:对每个节点都要重新计算左右子树深度,重复计算

### Step 3:瓶颈分析 → 优化方向
- 核心问题:计算深度时已经遍历了整棵树,但没有利用这个过程顺便计算直径
- 优化思路:在一次DFS求深度的过程中,顺便更新全局最大直径

### Step 4:选择武器
- 选用:**后序遍历DFS + 全局变量**
- 理由:后序遍历时先算出左右子树深度,再处理当前节点,正好可以边算深度边更新直径

> 🔑 **模式识别提示**:当题目要求"路径和"、"路径长度"、"经过某节点的最值"时,优先考虑"DFS + 全局变量"模式

---

## 🔑 解法一:分别计算深度(朴素法)

### 思路
对每个节点,调用辅助函数计算左右子树深度,更新最大直径。但这样每个节点的深度都要重新计算。

### 图解过程

```
示例:root = [1,2,3,4,5]

      1
     / \
    2   3
   / \
  4   5

Step 1:处理节点1
  计算左子树(2)深度 → 需要遍历4,5,2 → 深度=2
  计算右子树(3)深度 → 需要遍历3 → 深度=1
  直径候选 = 2+1 = 3

Step 2:处理节点2
  计算左子树(4)深度 → 遍历4 → 深度=1
  计算右子树(5)深度 → 遍历5 → 深度=1
  直径候选 = 1+1 = 2

Step 3:处理节点3,4,5...
  每次都要重新遍历子树计算深度

❌ 问题:大量重复计算
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def diameterOfBinaryTree_v1(root: Optional[TreeNode]) -> int:
    """
    解法一:分别计算深度
    思路:对每个节点单独计算左右深度,存在重复计算
    """
    def get_depth(node):
        """辅助函数:计算节点深度"""
        if not node:
            return 0
        return 1 + max(get_depth(node.left), get_depth(node.right))

    def dfs(node):
        """遍历每个节点,计算经过该节点的直径"""
        if not node:
            return 0

        # 计算当前节点的左右子树深度
        left_depth = get_depth(node.left)
        right_depth = get_depth(node.right)
        current_diameter = left_depth + right_depth

        # 递归计算子树的直径
        left_diameter = dfs(node.left)
        right_diameter = dfs(node.right)

        # 返回三者最大值
        return max(current_diameter, left_diameter, right_diameter)

    return dfs(root)


# ✅ 测试
root1 = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
print(diameterOfBinaryTree_v1(root1))  # 期望输出:3

root2 = TreeNode(1, TreeNode(2), None)
print(diameterOfBinaryTree_v1(root2))  # 期望输出:1
```

### 复杂度分析
- **时间复杂度**:O(n²) — 对每个节点(n个)都要遍历其子树计算深度
  - 具体地说:如果是链式树(n=1000),最坏情况需要 1000×500 ≈ 500000 次操作
- **空间复杂度**:O(h) — 递归栈深度,h为树高度

### 优缺点
- ✅ 思路直观,容易理解
- ❌ 存在大量重复计算,时间复杂度高

---

## 🏆 解法二:一次DFS求解(最优解)

### 优化思路
在计算深度的DFS过程中,顺便记录每个节点的"左深度+右深度",用全局变量保存最大值。**一次遍历完成所有计算**。

> 💡 **关键想法**:求深度本身就是DFS,在这个过程中顺便计算直径,避免重复遍历

### 图解过程

```
示例:root = [1,2,3,4,5]

      1
     / \
    2   3
   / \
  4   5

DFS后序遍历(左→右→根):

Step 1:访问节点4(叶子)
  深度 = 0, 返回 1

Step 2:访问节点5(叶子)
  深度 = 0, 返回 1

Step 3:访问节点2
  左深度 = 1(来自节点4)
  右深度 = 1(来自节点5)
  直径候选 = 1+1 = 2, 更新 max_diameter = 2
  返回深度 = 1 + max(1,1) = 2

Step 4:访问节点3(叶子)
  深度 = 0, 返回 1

Step 5:访问节点1(根)
  左深度 = 2(来自节点2)
  右深度 = 1(来自节点3)
  直径候选 = 2+1 = 3, 更新 max_diameter = 3 ✓
  返回深度 = 1 + max(2,1) = 3

最终答案:max_diameter = 3
```

### Python代码

```python
def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:
    """
    🏆 解法二:一次DFS求解(最优解)
    思路:后序遍历计算深度时,顺便更新全局最大直径
    """
    max_diameter = 0  # 全局变量记录最大直径

    def dfs(node):
        """返回当前节点的深度,同时更新全局直径"""
        nonlocal max_diameter

        if not node:
            return 0

        # 递归计算左右子树深度
        left_depth = dfs(node.left)
        right_depth = dfs(node.right)

        # 更新全局最大直径(经过当前节点的路径长度)
        max_diameter = max(max_diameter, left_depth + right_depth)

        # 返回当前节点的深度(给父节点用)
        return 1 + max(left_depth, right_depth)

    dfs(root)
    return max_diameter


# ✅ 测试
root1 = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
print(diameterOfBinaryTree(root1))  # 期望输出:3

root2 = TreeNode(1, TreeNode(2), None)
print(diameterOfBinaryTree(root2))  # 期望输出:1

root3 = TreeNode(1)
print(diameterOfBinaryTree(root3))  # 期望输出:0(单节点)
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点只访问一次
  - 具体地说:如果树有1000个节点,只需要恰好1000次访问
- **空间复杂度**:O(h) — 递归栈深度,最坏h=n(链式树),平均h=log n

---

## 🐍 Pythonic 写法

利用Python的闭包特性和多返回值,可以更简洁地表达:

```python
# 方法:利用列表作为可变对象传递全局状态
def diameterOfBinaryTree_pythonic(root: Optional[TreeNode]) -> int:
    def dfs(node):
        if not node:
            return 0
        left, right = dfs(node.left), dfs(node.right)
        result[0] = max(result[0], left + right)  # 列表可变,闭包中可修改
        return 1 + max(left, right)

    result = [0]  # 用列表绕过 nonlocal
    dfs(root)
    return result[0]
```

这个写法用列表 `[0]` 来避免使用 `nonlocal` 关键字,利用了列表是可变对象的特性。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:分别计算深度 | 🏆 解法二:一次DFS(最优) |
|------|-----------------|---------------------|
| 时间复杂度 | O(n²) | **O(n)** ← 时间最优 |
| 空间复杂度 | O(h) | **O(h)** ← 相同 |
| 代码难度 | 中等 | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 仅适合小规模树 | **通用,面试标准解法** |

**为什么是最优解**:
- 时间复杂度O(n)已经是理论最优(必须至少访问每个节点一次)
- 一次遍历即可完成,没有任何冗余计算
- 代码简洁优雅,后序遍历是处理树路径问题的经典模式

**面试建议**:
1. 先花20秒说明暴力思路:"每个节点分别算深度,但会重复计算"
2. 立即优化到🏆最优解:"在求深度的DFS中顺便更新直径,一次遍历搞定"
3. **重点讲解最优解的核心思想**:"后序遍历天然适合路径问题,因为先知道子树信息再处理当前节点"
4. 强调为什么这是最优:每个节点只访问一次,无法更快
5. 手动模拟测试用例,证明理解深刻

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找二叉树的直径,即任意两节点间最长路径的边数。让我先想一下...

我的第一个想法是对每个节点分别计算左右子树深度,然后求最大和,但这样会重复计算很多次子树深度,时间复杂度是 O(n²)。

不过我们可以用后序遍历优化到 O(n):在一次DFS求深度的过程中,顺便记录每个节点的"左深度+右深度",用全局变量保存最大值。核心思路是**后序遍历先处理子树再处理当前节点,天然适合路径问题**。

**面试官**:很好,请写一下代码。

**你**:(边写边说关键步骤)
```python
# 用全局变量记录最大直径
max_diameter = 0

# DFS后序遍历
def dfs(node):
    if not node: return 0

    # 先递归计算左右深度
    left = dfs(node.left)
    right = dfs(node.right)

    # 更新全局最大直径
    max_diameter = max(max_diameter, left + right)

    # 返回当前深度给父节点
    return 1 + max(left, right)
```

**面试官**:测试一下?

**你**:用示例 [1,2,3,4,5] 走一遍:叶子节点4、5返回深度1,节点2计算得左深度1、右深度1,更新直径为2,返回深度2;节点3返回深度1;最后根节点1得到左深度2、右深度1,更新直径为3 ✓。再测一个边界情况:单节点树返回0 ✓,结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | "时间O(n)已经是最优,因为必须访问每个节点。空间受限于递归栈O(h),如果改用迭代+栈,空间复杂度不变但代码复杂度增加,不值得。" |
| "为什么用后序遍历?" | "后序遍历是'左→右→根',先知道子树信息(左右深度)再处理当前节点,正好符合'直径=左深度+右深度'的计算顺序。" |
| "如果要返回路径本身呢?" | "需要在更新最大直径时,额外记录对应的节点,然后从该节点分别向左右子树回溯路径。空间复杂度变为O(n)。" |
| "这个模式能用到哪些题?" | "所有'树上路径'问题都可以用这个模式,如路径总和、最大路径和(LeetCode 124)、二叉树最大深度等。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:nonlocal 修改外层变量 — 闭包中修改非全局变量
def outer():
    count = 0
    def inner():
        nonlocal count  # 声明使用外层的 count
        count += 1
    inner()
    return count

# 技巧2:用列表绕过 nonlocal — 可变对象特性
def outer_v2():
    result = [0]  # 列表是可变对象
    def inner():
        result[0] += 1  # 不需要 nonlocal
    inner()
    return result[0]

# 技巧3:多返回值解包 — 简化代码
left_depth, right_depth = dfs(node.left), dfs(node.right)
```

### 💡 底层原理(选读)

> **为什么后序遍历适合路径问题?**
>
> 后序遍历的执行顺序是"左子树→右子树→根节点",这意味着处理当前节点时,已经拥有了子树的所有信息(深度、路径和等)。这种"自底向上"的信息流动,天然适合:
> - 路径问题(需要知道子树路径才能计算经过当前节点的路径)
> - 子树统计(如节点数、深度)
> - 验证性质(如BST验证)
>
> **对比前序遍历**:前序是"根→左→右",适合自顶向下传递信息(如路径前缀、上界下界)。

### 算法模式卡片 📐
- **模式名称**:DFS后序遍历 + 全局变量
- **适用条件**:求解树上路径相关的最值问题(路径和、路径长度、路径数量)
- **识别关键词**:"路径"、"直径"、"经过某节点的最大/最小"、"任意两节点"
- **模板代码**:
```python
def tree_path_problem(root):
    global_result = 初始值  # 全局变量记录答案

    def dfs(node):
        nonlocal global_result
        if not node:
            return 边界值

        # 后序:先递归处理子树
        left_info = dfs(node.left)
        right_info = dfs(node.right)

        # 利用子树信息更新全局结果
        global_result = update(global_result, left_info, right_info)

        # 返回当前节点信息给父节点
        return compute(left_info, right_info)

    dfs(root)
    return global_result
```

### 易错点 ⚠️
1. **混淆"深度"和"直径"**:深度是节点数,直径是边数。深度=边数+1。
   - ❌ 错误:`return left + right`(返回的是节点数-1)
   - ✅ 正确:直径用 `left_depth + right_depth`(边数),深度用 `1 + max(left, right)`(节点数)

2. **忘记处理空节点**:递归边界必须返回0(深度为0)
   - ❌ 错误:`if not node: return None` → 会导致 `max(None, 1)` 报错
   - ✅ 正确:`if not node: return 0`

3. **误以为直径一定经过根节点**:直径可能只在某个子树内部
   - 💡 解决:必须遍历所有节点,在每个节点处都尝试更新全局最大值

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:社交网络分析 — 计算组织架构中的"最长汇报链"(CEO到最底层员工的层级数)
- **场景2**:版本控制系统 — Git中计算两个commit之间的最短路径(共同祖先到两个节点的距离)
- **场景3**:文件系统 — 计算目录树的最大宽度,用于UI渲染或存储优化

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 124. 二叉树中的最大路径和 | Hard | DFS+全局变量 | 同样的模式,但要处理负数和选择性跳过节点 |
| LeetCode 687. 最长同值路径 | Medium | DFS+全局变量 | 增加值相等的约束条件 |
| LeetCode 104. 二叉树的最大深度 | Easy | DFS后序遍历 | 本题的简化版,只需返回深度 |
| LeetCode 563. 二叉树的坡度 | Easy | DFS+子树和 | 相似模式,计算子树和而非深度 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一棵二叉树,返回**经过根节点的最长路径长度**。(注意:必须经过根节点)

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

只需要计算根节点的左右子树深度之和,不需要遍历所有节点。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def diameter_through_root(root: Optional[TreeNode]) -> int:
    """只计算经过根节点的直径"""
    def get_depth(node):
        if not node:
            return 0
        return 1 + max(get_depth(node.left), get_depth(node.right))

    if not root:
        return 0

    # 只计算根节点的左右深度之和
    return get_depth(root.left) + get_depth(root.right)


# 测试
root = TreeNode(1, TreeNode(2, TreeNode(4), TreeNode(5)), TreeNode(3))
print(diameter_through_root(root))  # 输出:3
```

核心思路:与原题区别在于,这里只需要计算一次根节点的左右深度,不需要遍历所有节点作为候选。时间复杂度O(n),但常数更小。

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
