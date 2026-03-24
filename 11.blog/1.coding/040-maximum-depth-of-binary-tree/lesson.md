> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第40课:二叉树最大深度

> **模块**:二叉树 | **难度**:Easy ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/maximum-depth-of-binary-tree/
> **前置知识**:无
> **预计学习时间**:15分钟

---

## 🎯 题目描述

给定一个二叉树的根节点,返回该树的最大深度。二叉树的深度是指从根节点到最远叶子节点的最长路径上的节点数。

**示例:**
```
    3
   / \
  9  20
    /  \
   15   7

输入:root = [3,9,20,null,null,15,7]
输出:3
解释:最长路径是 3 → 20 → 15(或7),共3个节点
```

**约束条件:**
- 树中节点数量范围是 [0, 10^4]
- -100 <= Node.val <= 100

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空树 | root=None | 0 | 递归出口 |
| 单节点 | root=[1] | 1 | 基本功能 |
| 只有左子树 | root=[1,2,3] | 3 | 偏斜树 |
| 只有右子树 | root=[1,null,2,null,3] | 3 | 偏斜树 |
| 完全二叉树 | root=[1,2,3,4,5,6,7] | 3 | 平衡情况 |
| 大规模 | n=10000 | — | 栈溢出风险 |

---

## 💡 思路引导

### 生活化比喻
> 想象你是一个公司的HR,要统计公司的层级结构有多少层。
>
> 🐌 **笨办法**:从CEO开始,逐个访问每个员工,记录他们的层级,最后找最大值。这样需要遍历所有员工。
>
> 🚀 **聪明办法**:直接问CEO:"你的左边团队有几层?右边团队有几层?" CEO再分别问他的直接下属,一层层递归下去。最后CEO把左右两边的较大值+1(加上自己这一层)就是答案。这就是**递归分治思想**。

### 关键洞察
**树的深度 = max(左子树深度, 右子树深度) + 1**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树根节点 TreeNode
- **输出**:整数,表示最大深度
- **限制**:节点数最多10^4,需考虑递归栈深度

### Step 2:先想笨办法(层序遍历)
用队列进行BFS层序遍历,每遍历完一层就让深度+1,最后返回层数。
- 时间复杂度:O(n) — 需要访问所有节点
- 瓶颈在哪:需要额外的队列空间,且代码较繁琐

### Step 3:瓶颈分析 → 优化方向
层序遍历虽然直观,但需要维护队列。能否更简洁?
- 核心问题:"如何不显式维护队列也能知道深度?"
- 优化思路:利用递归的调用栈天然表示层级关系

### Step 4:选择武器
- 选用:**深度优先搜索(DFS)递归**
- 理由:树的深度问题具有明显的递归子结构,且递归代码极简

> 🔑 **模式识别提示**:当题目涉及"树的深度/高度/路径",优先考虑"DFS递归"

---

## 🏆 解法一:DFS递归(最优解)

### 思路
利用递归定义:
1. 空树深度为0(递归出口)
2. 非空树深度 = max(左子树深度, 右子树深度) + 1

这是**后序遍历**的思想:先计算左右子树结果,再处理当前节点。

### 图解过程

```
示例1:
    3
   / \
  9  20
    /  \
   15   7

递归调用过程:
maxDepth(3)
├─ maxDepth(9)
│  ├─ maxDepth(null) → 0
│  └─ maxDepth(null) → 0
│  返回: max(0, 0) + 1 = 1
│
└─ maxDepth(20)
   ├─ maxDepth(15)
   │  ├─ maxDepth(null) → 0
   │  └─ maxDepth(null) → 0
   │  返回: max(0, 0) + 1 = 1
   │
   └─ maxDepth(7)
      ├─ maxDepth(null) → 0
      └─ maxDepth(null) → 0
      返回: max(0, 0) + 1 = 1

   返回: max(1, 1) + 1 = 2

最终返回: max(1, 2) + 1 = 3 ✅


示例2(边界):只有左子树
    1
   /
  2
 /
3

maxDepth(1)
├─ maxDepth(2)
│  ├─ maxDepth(3)
│  │  ├─ maxDepth(null) → 0
│  │  └─ maxDepth(null) → 0
│  │  返回: 1
│  └─ maxDepth(null) → 0
│  返回: max(1, 0) + 1 = 2
│
└─ maxDepth(null) → 0

返回: max(2, 0) + 1 = 3 ✅
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def maxDepth(root: Optional[TreeNode]) -> int:
    """
    解法一:DFS递归(最优解)
    思路:深度 = max(左子树深度, 右子树深度) + 1
    """
    # 递归出口:空节点深度为0
    if not root:
        return 0

    # 递归计算左右子树深度
    left_depth = maxDepth(root.left)
    right_depth = maxDepth(root.right)

    # 当前树深度 = 较深子树 + 1(根节点)
    return max(left_depth, right_depth) + 1


# ✅ 测试
# 构建示例树:    3
#              / \
#             9  20
#               /  \
#              15   7
root1 = TreeNode(3)
root1.left = TreeNode(9)
root1.right = TreeNode(20, TreeNode(15), TreeNode(7))
print(maxDepth(root1))  # 期望输出:3

# 边界测试
print(maxDepth(None))  # 期望输出:0
print(maxDepth(TreeNode(1)))  # 期望输出:1

# 偏斜树
root2 = TreeNode(1, TreeNode(2, TreeNode(3)))
print(maxDepth(root2))  # 期望输出:3
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点访问恰好一次
  - 具体地说:如果树有1000个节点,需要1000次函数调用
- **空间复杂度**:O(h) — 递归栈深度等于树高h
  - 最好情况(平衡树):O(log n),如1000个节点约10层
  - 最坏情况(链状树):O(n),如1000个节点1000层(可能栈溢出)

### 优缺点
- ✅ 代码极简,仅3行核心逻辑
- ✅ 时间最优,无冗余操作
- ⚠️ 极端偏斜树可能栈溢出(n=10^4时Python默认栈深度约1000)

---

## ⚡ 解法二:BFS层序遍历(迭代法)

### 优化思路
递归虽简洁,但可能栈溢出。改用显式队列的迭代法,空间由递归栈转为队列,但栈溢出风险更可控。

> 💡 **关键想法**:用队列按层遍历,每处理完一层就让深度+1

### 图解过程

```
示例:    3
        / \
       9  20
         /  \
        15   7

层序遍历过程:
第1层: queue=[3]           → depth=1
       出队3, 入队9,20

第2层: queue=[9,20]        → depth=2
       出队9,20, 入队15,7

第3层: queue=[15,7]        → depth=3
       出队15,7, 队列空

返回 depth=3 ✅
```

### Python代码

```python
from collections import deque


def maxDepth_bfs(root: Optional[TreeNode]) -> int:
    """
    解法二:BFS层序遍历
    思路:用队列按层遍历,统计层数
    """
    if not root:
        return 0

    queue = deque([root])
    depth = 0

    while queue:
        # 当前层的节点数
        level_size = len(queue)

        # 处理当前层的所有节点
        for _ in range(level_size):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        # 处理完一层,深度+1
        depth += 1

    return depth


# ✅ 测试
root1 = TreeNode(3)
root1.left = TreeNode(9)
root1.right = TreeNode(20, TreeNode(15), TreeNode(7))
print(maxDepth_bfs(root1))  # 期望输出:3
print(maxDepth_bfs(None))   # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点入队出队各一次
- **空间复杂度**:O(w) — w为树的最大宽度
  - 完全二叉树最坏情况:最底层约n/2个节点,即O(n)

---

## 🐍 Pythonic 写法

利用三元表达式和递归的简洁性:

```python
# 一行版本(不推荐,可读性差)
maxDepth_oneline = lambda root: 0 if not root else max(maxDepth_oneline(root.left), maxDepth_oneline(root.right)) + 1

# 推荐的简洁版
def maxDepth_compact(root: Optional[TreeNode]) -> int:
    return 0 if not root else 1 + max(maxDepth_compact(root.left), maxDepth_compact(root.right))
```

> ⚠️ **面试建议**:先写清晰版本(解法一的分步写法)展示思路,通过后再提"可以简化为一行"展示语言功底。面试官更看重你的**递归理解**,而非代码行数。

---

## 📊 解法对比

| 维度 | 🏆 解法一:DFS递归 | 解法二:BFS迭代 |
|------|-----------------|--------------|
| 时间复杂度 | **O(n)** ← 最优 | O(n) |
| 空间复杂度 | O(h) 平衡树O(log n) | O(w) 完全树O(n) |
| 代码难度 | **简单** | 中等 |
| 面试推荐 | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | **通用,代码最简** | 避免栈溢出 |
| 栈溢出风险 | 偏斜树可能 | 无 |

**为什么解法一是最优解**:
- 时间O(n)已是理论最优(必须访问所有节点才能知道最大深度)
- 平衡树下空间O(log n)优于BFS的O(n)
- 代码仅3行,面试中最易写对
- Python递归深度限制约1000,题目限制n≤10^4,实际偏斜树极少见

**面试建议**:
1. 直接说出🏆最优解思路:"用递归,深度=max(左,右)+1"
2. 写代码时同步讲解递归三要素:出口、递推、返回值
3. 手动模拟一个3层树的递归过程,展示理解深度
4. 主动提及边界:"空树返回0,单节点返回1"
5. 如被问"能否不用递归?",给出解法二BFS方案

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题:求二叉树最大深度。

**你**:(审题10秒)好的,这道题要求从根节点到最远叶子的最长路径长度。让我先想一下...

我的第一反应是用**递归**。因为树的深度有明显的递归定义:空树深度为0,非空树深度等于左右子树较大深度+1。时间复杂度是O(n),因为要访问所有节点。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def maxDepth(root):
    # 先处理递归出口:空节点深度为0
    if not root:
        return 0
    # 递归计算左右子树深度
    left = maxDepth(root.left)
    right = maxDepth(root.right)
    # 当前深度 = 较深子树 + 根节点
    return max(left, right) + 1
```

**面试官**:测试一下?

**你**:用示例[3,9,20,null,null,15,7]走一遍...
- 先递归到节点9,左右都是null,返回0,所以9的深度是max(0,0)+1=1
- 再递归到节点20,它的左孩子15深度1,右孩子7深度1,所以20的深度是max(1,1)+1=2
- 最后根节点3,左孩子9深度1,右孩子20深度2,返回max(1,2)+1=3 ✅

再测一个边界:空树应该返回0...代码第2行处理了,正确 ✅

**面试官**:不错!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有其他方法吗?" | "可以用BFS层序遍历,每遍历完一层深度+1,时间仍是O(n)但空间可能更大,代码也更长,所以递归是首选" |
| "如果树非常深会栈溢出吗?" | "确实,Python默认栈深度约1000。但题目限制n≤10^4,且极度偏斜的树很少见。如果真担心可以用BFS迭代法,或用sys.setrecursionlimit调整" |
| "能一行写完吗?" | "可以:`return 0 if not root else 1 + max(maxDepth(root.left), maxDepth(root.right))`,但面试时建议分步写清楚思路" |
| "时间能优化到O(log n)吗?" | "不能,必须访问所有节点才能确定最大深度,O(n)已是理论下界" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:Optional类型标注 — 表示可能为None
from typing import Optional
def func(root: Optional[TreeNode]) -> int:
    pass

# 技巧2:递归的三元表达式简化
return 0 if not root else 1 + max(f(root.left), f(root.right))

# 技巧3:collections.deque — 双端队列,O(1)首尾操作
from collections import deque
queue = deque([root])
queue.append(node)      # 尾部添加
queue.popleft()         # 头部弹出
```

### 💡 底层原理(选读)

> **递归为什么消耗栈空间?**
>
> 每次函数调用都会在调用栈上分配一个"栈帧",保存局部变量和返回地址。递归maxDepth时,调用链是:
> ```
> maxDepth(根) → maxDepth(左孩子) → maxDepth(左孙子) → ...
> ```
> 在最深叶子返回之前,所有中间栈帧都在内存中。树高h就需要h个栈帧,所以空间O(h)。
>
> Python默认栈深度限制:`sys.getrecursionlimit()` 约1000,可通过`sys.setrecursionlimit(10000)`调整,但不推荐设太大(可能内存溢出)。

> **BFS为什么用deque而非list?**
>
> list的`pop(0)`删除首元素需要O(n)时间(所有后续元素前移),而deque的`popleft()`是O(1)。BFS需要频繁队首出队,用deque避免性能退化。

### 算法模式卡片 📐
- **模式名称**:树的DFS递归(后序遍历)
- **适用条件**:需要先得到子树结果再计算当前节点
- **识别关键词**:"树的深度"、"树的高度"、"路径和"、"子树信息汇总"
- **模板代码**:
```python
def dfs(root):
    # 1. 递归出口
    if not root:
        return base_value

    # 2. 递归处理左右子树
    left_result = dfs(root.left)
    right_result = dfs(root.right)

    # 3. 合并结果(后序位置)
    current_result = process(left_result, right_result, root.val)

    return current_result
```

### 易错点 ⚠️
1. **忘记处理空节点**
   - ❌ 错误:`return 1 + max(maxDepth(root.left), maxDepth(root.right))` 直接调用会在None上调用.left报错
   - ✅ 正确:先判断`if not root: return 0`

2. **递归顺序混淆**
   - ❌ 错误:以为是前序遍历,先处理root再递归
   - ✅ 正确:这是后序遍历,先递归得到左右结果,再处理当前节点

3. **边界返回值错误**
   - ❌ 错误:空树返回1(认为至少有根)
   - ✅ 正确:空树深度为0,单节点树深度为1

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:文件系统目录深度统计
  ```python
  # Unix的find命令`find . -type d -printf '%d\n' | sort -n | tail -1`
  # 就是在计算目录树最大深度,用于检测过深的嵌套(可能是软链接循环)
  ```

- **场景2**:组织架构层级分析
  ```python
  # HR系统计算汇报链最长层级,用于识别组织扁平化程度
  # 超过6层可能管理效率低下
  ```

- **场景3**:JSON/XML解析深度限制
  ```python
  # 防止恶意深层嵌套导致栈溢出攻击
  # 很多JSON解析器会限制最大深度(如64层)
  import json
  json.loads(data, parse_constant=lambda x: depth_check(x))
  ```

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 111. 二叉树最小深度 | Easy | DFS递归 | 注意单子树情况需特殊处理 |
| LeetCode 110. 平衡二叉树 | Easy | DFS+深度 | 在计算深度的同时判断平衡 |
| LeetCode 543. 二叉树的直径 | Easy | DFS+全局变量 | 最长路径可能不过根,需全局记录 |
| LeetCode 559. N叉树的最大深度 | Easy | DFS递归 | 推广到多叉树,用max遍历所有子节点 |
| LeetCode 102. 二叉树层序遍历 | Medium | BFS | 本题解法二的扩展,返回每层节点值 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定二叉树,返回最小深度(根节点到最近叶子节点的最短路径节点数)。注意:叶子节点是指没有子节点的节点。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

关键区别:最大深度用max,最小深度用min,但要注意**单子树情况**!
- 如果只有左子树,不能返回min(left, 0)+1=1(错误!右边不是叶子)
- 应该返回left+1(继续往左找叶子)

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def minDepth(root: Optional[TreeNode]) -> int:
    """
    最小深度 — 到最近叶子节点的路径
    关键:单子树情况要特殊处理
    """
    if not root:
        return 0

    # 叶子节点
    if not root.left and not root.right:
        return 1

    # 只有左子树,必须往左找(右边不是叶子)
    if not root.right:
        return minDepth(root.left) + 1

    # 只有右子树,必须往右找
    if not root.left:
        return minDepth(root.right) + 1

    # 两个子树都有,取较小值
    return min(minDepth(root.left), minDepth(root.right)) + 1
```

**核心思路**:
- 与最大深度的唯一区别:必须到达**叶子节点**才算一条完整路径
- 单子树情况不能简单用min(因为None那边不是叶子),要强制走非空子树
- 可以用BFS更简洁:第一个遇到的叶子节点就是最小深度

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
