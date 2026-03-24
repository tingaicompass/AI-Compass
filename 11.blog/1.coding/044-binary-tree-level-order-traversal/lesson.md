> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第44课:层序遍历

> **模块**:二叉树 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/binary-tree-level-order-traversal/
> **前置知识**:第39课(二叉树中序遍历)、队列基础
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给你二叉树的根节点 `root`,返回其节点值的**层序遍历**结果。即逐层地,从左到右访问所有节点,返回一个二维数组,每个子数组代表一层的所有节点值。

**示例:**
```
输入:root = [3,9,20,null,null,15,7]
      3
     / \
    9  20
      /  \
     15   7
输出:[[3], [9,20], [15,7]]
解释:第1层[3],第2层[9,20],第3层[15,7]
```

**约束条件:**
- 树中节点数量范围:[0, 2000]
- -1000 ≤ Node.val ≤ 1000

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空树 | root = None | [] | 空输入处理 |
| 单节点 | root = [1] | [[1]] | 基本功能 |
| 完全二叉树 | root = [1,2,3,4,5,6,7] | [[1],[2,3],[4,5,6,7]] | 标准情况 |
| 链式树 | root = [1,2,null,3] | [[1],[2],[3]] | 每层只有一个节点 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在拍摄一张集体照,要求**按楼层排列**:
>
> 🐌 **笨办法**:用递归DFS,记录每个人的楼层号,最后再按楼层分组排序。但这样需要额外记录每个节点的层级,还要后期整理。
>
> 🚀 **聪明办法**:用队列BFS,像排队一样:先让第1层的人全部入队并拍照,然后他们离开,第2层的人全部入队拍照...每层拍一张照,自然就是按层分组的结果!

### 关键洞察
**层序遍历 = BFS广度优先搜索,用队列控制每层节点的访问顺序**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树根节点 root(TreeNode 类型或 None)
- **输出**:二维列表 `List[List[int]]`,外层列表每个元素是一层的所有节点值
- **限制**:必须按层分组,每层内部从左到右

### Step 2:先想笨办法(DFS + 层级记录)
用DFS递归遍历,同时传递一个 `level` 参数表示当前层级,把节点值追加到 `result[level]` 中。
- 时间复杂度:O(n)
- 瓶颈在哪:需要额外传递层级参数,且要初始化足够多的子列表,不够直观

### Step 3:瓶颈分析 → 优化方向
- 核心问题:DFS是"深度优先",访问顺序是"一路到底",不符合"一层层"的自然顺序
- 优化思路:改用BFS"广度优先",用队列天然地按层访问节点

### Step 4:选择武器
- 选用:**BFS + 队列(deque)**
- 理由:队列FIFO特性正好实现"先访问的节点,其子节点也先访问",符合层序遍历逻辑

> 🔑 **模式识别提示**:当题目要求"按层"、"逐层"、"最短路径"、"最少步数"时,优先考虑"BFS + 队列"模式

---

## 🔑 解法一:DFS递归 + 层级参数(非最优)

### 思路
用递归DFS遍历,传递当前层级 `level`,把节点值追加到 `result[level]` 对应的列表中。

### 图解过程

```
示例:root = [3,9,20,null,null,15,7]

      3       level=0
     / \
    9  20     level=1
      /  \
     15   7   level=2

DFS递归(前序遍历顺序):

Step 1:访问节点3(level=0)
  result = [[3]]

Step 2:访问节点9(level=1)
  result = [[3], [9]]

Step 3:访问节点20(level=1)
  result = [[3], [9, 20]]

Step 4:访问节点15(level=2)
  result = [[3], [9, 20], [15]]

Step 5:访问节点7(level=2)
  result = [[3], [9, 20], [15, 7]]

✓ 最终结果按层分组
```

### Python代码

```python
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def levelOrder_dfs(root: Optional[TreeNode]) -> List[List[int]]:
    """
    解法一:DFS递归 + 层级参数
    思路:递归时传递层级level,将节点追加到对应层的列表
    """
    result = []

    def dfs(node, level):
        if not node:
            return

        # 如果当前层还没有列表,创建一个
        if level == len(result):
            result.append([])

        # 将当前节点值追加到对应层
        result[level].append(node.val)

        # 递归处理左右子树,层级+1
        dfs(node.left, level + 1)
        dfs(node.right, level + 1)

    dfs(root, 0)
    return result


# ✅ 测试
root1 = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
print(levelOrder_dfs(root1))  # 期望输出:[[3], [9, 20], [15, 7]]

root2 = TreeNode(1)
print(levelOrder_dfs(root2))  # 期望输出:[[1]]

root3 = None
print(levelOrder_dfs(root3))  # 期望输出:[]
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点访问一次
  - 具体地说:如果树有100个节点,需要恰好100次访问
- **空间复杂度**:O(h) — 递归栈深度,h为树高度(最坏h=n,平均h=log n)

### 优缺点
- ✅ 代码简洁,利用递归自动管理状态
- ❌ 不够直观,DFS顺序与层序概念不匹配
- ❌ 递归深度受限于树高度,极端情况可能栈溢出

---

## 🏆 解法二:BFS队列(最优解)

### 优化思路
使用队列实现BFS广度优先搜索:每次处理队列中当前层的所有节点,并将它们的子节点加入队列(成为下一层)。用一个循环控制"逐层处理"。

> 💡 **关键想法**:队列天然支持"先进先出",正好对应"上层先访问,下层后访问"

### 图解过程

```
示例:root = [3,9,20,null,null,15,7]

      3
     / \
    9  20
      /  \
     15   7

BFS层序遍历:

初始化:queue = [3], result = []

--- 第1层 ---
queue = [3] (当前层大小size=1)
弹出3 → 记录[3] → 加入子节点9,20
queue = [9, 20]
result = [[3]]

--- 第2层 ---
queue = [9, 20] (当前层大小size=2)
弹出9 → 记录9 → 无子节点
弹出20 → 记录20 → 加入子节点15,7
queue = [15, 7]
result = [[3], [9, 20]]

--- 第3层 ---
queue = [15, 7] (当前层大小size=2)
弹出15 → 记录15 → 无子节点
弹出7 → 记录7 → 无子节点
queue = []
result = [[3], [9, 20], [15, 7]] ✓

队列为空,结束
```

### Python代码

```python
from collections import deque


def levelOrder(root: Optional[TreeNode]) -> List[List[int]]:
    """
    🏆 解法二:BFS队列(最优解)
    思路:用队列逐层处理节点,每层单独记录
    """
    if not root:
        return []

    result = []
    queue = deque([root])  # 初始化队列,根节点入队

    while queue:
        level_size = len(queue)  # 当前层的节点数
        current_level = []       # 存储当前层的节点值

        # 处理当前层的所有节点
        for _ in range(level_size):
            node = queue.popleft()  # 弹出队首节点
            current_level.append(node.val)

            # 将子节点加入队列(成为下一层)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)  # 当前层处理完毕,加入结果

    return result


# ✅ 测试
root1 = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
print(levelOrder(root1))  # 期望输出:[[3], [9, 20], [15, 7]]

root2 = TreeNode(1)
print(levelOrder(root2))  # 期望输出:[[1]]

root3 = None
print(levelOrder(root3))  # 期望输出:[]

# 边界测试:链式树
root4 = TreeNode(1, TreeNode(2, TreeNode(3)))
print(levelOrder(root4))  # 期望输出:[[1], [2], [3]]
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点入队、出队各一次
  - 具体地说:如果树有100个节点,需要100次入队 + 100次出队 = 200次操作
- **空间复杂度**:O(w) — w为树的最大宽度(队列最多存储一层的所有节点)
  - 完全二叉树最坏情况:w ≈ n/2,如最底层有50个节点

---

## 🐍 Pythonic 写法

利用Python的列表推导式和队列迭代,可以更简洁:

```python
# 方法:简化版BFS,保持核心逻辑不变
def levelOrder_pythonic(root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []

    result, queue = [], deque([root])

    while queue:
        # 一行代码处理当前层并加入下层节点
        result.append([node.val for node in queue])
        queue = deque([child for node in queue for child in (node.left, node.right) if child])

    return result
```

这个写法用列表推导式一次性处理当前层并构建下一层队列,代码更紧凑但可读性略降。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:DFS递归 | 🏆 解法二:BFS队列(最优) |
|------|--------------|---------------------|
| 时间复杂度 | O(n) | **O(n)** ← 相同 |
| 空间复杂度 | O(h) 递归栈 | **O(w)** ← 队列宽度 |
| 代码难度 | 中等 | 简单 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 适合递归思维 | **层序遍历标准解法,通用性强** |

**为什么是最优解**:
- BFS天然符合"层序"语义,代码逻辑直观易懂
- 队列空间复杂度O(w)通常优于递归栈O(h),尤其对于平衡树
- 面试中提到"层序遍历"必然考察BFS,这是教科书级别的经典解法
- 可扩展性强:容易改造为自底向上、锯齿形遍历等变体

**面试建议**:
1. 开口就说:"层序遍历用BFS + 队列是标准解法"
2. 画图演示队列的变化过程,强调"当前层大小 `size = len(queue)` 是关键"
3. **重点讲解最优解的核心思想**:"外层while控制是否还有层,内层for处理当前层的所有节点"
4. 提及空间优化:队列最多存储一层节点,完全二叉树约为n/2
5. 手动模拟测试用例,展示对BFS的深刻理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你实现二叉树的层序遍历。

**你**:(审题10秒)好的,层序遍历要求按层输出节点值,每层是一个子数组。

我的思路是用**BFS广度优先搜索 + 队列**:
1. 初始化队列,根节点入队
2. 每次循环处理当前层的所有节点(关键:记录当前队列大小 `size`)
3. 弹出 `size` 个节点,记录它们的值,并将子节点加入队列
4. 重复直到队列为空

时间复杂度 O(n),空间复杂度 O(w)(w为树的最大宽度)。

**面试官**:很好,请写一下代码。

**你**:(边写边说关键步骤)
```python
from collections import deque

def levelOrder(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        size = len(queue)  # 关键:记录当前层大小
        level = []

        for _ in range(size):  # 只处理当前层的节点
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)

        result.append(level)

    return result
```

**面试官**:测试一下?

**你**:用示例 [3,9,20,null,null,15,7] 走一遍:
- 初始队列[3],弹出3得到第1层[3],加入9、20
- 队列变为[9,20],弹出2个得到第2层[9,20],加入15、7
- 队列变为[15,7],弹出2个得到第3层[15,7]
- 队列为空,结束 ✓

再测试边界:空树返回[],单节点返回[[1]] ✓,结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能不能自底向上输出?" | "可以,最后对 `result` 做一次反转:`return result[::-1]`,或者用栈代替列表存储每层结果。" |
| "如果要锯齿形遍历呢?" | "用一个标志位 `left_to_right`,奇数层正常append,偶数层用 `level.insert(0, val)` 或最后反转该层。" |
| "空间能优化到O(1)吗?" | "不能,BFS必须用队列存储当前层节点,除非题目不要求按层分组,可以边遍历边输出。" |
| "DFS能实现吗?" | "可以,用递归传递层级参数(解法一),但不如BFS直观,面试中BFS是首选。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:deque 双端队列 — O(1) 两端操作
from collections import deque
q = deque([1, 2, 3])
q.append(4)      # 右端入队:[1,2,3,4]
q.popleft()      # 左端出队:1,剩[2,3,4]
q.appendleft(0)  # 左端入队:[0,2,3,4]

# 技巧2:列表推导式生成队列 — 简洁构建下一层
next_level = [child for node in queue for child in (node.left, node.right) if child]

# 技巧3:enumerate遍历层级 — 同时获取索引和值
for level_idx, level_nodes in enumerate(result):
    print(f"第{level_idx}层:{level_nodes}")
```

### 💡 底层原理(选读)

> **为什么BFS用队列,DFS用栈?**
>
> - **BFS(广度优先)**:目标是"先访问近的节点,再访问远的节点",队列的FIFO特性保证先入队的节点(上层)先被处理。
> - **DFS(深度优先)**:目标是"先走到底,再回溯",栈的LIFO特性(或递归调用栈)保证后访问的节点(深层)先处理完。
>
> **队列 vs 列表**:
> - Python的 `list` 作为队列时,`pop(0)` 是O(n)操作(需要移动所有元素)
> - `collections.deque` 是双端队列,`popleft()` 和 `append()` 都是O(1),专为BFS优化

### 算法模式卡片 📐
- **模式名称**:BFS层序遍历
- **适用条件**:需要按层处理、最短路径、最少步数、树/图的逐层展开
- **识别关键词**:"层序"、"按层"、"逐层"、"最短"、"最少步数"、"广度"
- **模板代码**:
```python
from collections import deque

def bfs_level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)  # 关键:记录当前层大小
        current_level = []

        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)  # 处理节点

            # 加入下一层节点
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)

    return result
```

### 易错点 ⚠️
1. **忘记记录当前层大小**:直接 `while queue` 循环会混淆层级
   - ❌ 错误:`while queue: node = queue.popleft()` → 无法区分哪些节点属于同一层
   - ✅ 正确:`size = len(queue); for _ in range(size)` → 保证每层独立处理

2. **用list代替deque**:性能问题
   - ❌ 错误:`queue = []; val = queue.pop(0)` → O(n)时间,大数据超时
   - ✅ 正确:`from collections import deque; queue.popleft()` → O(1)时间

3. **子节点加入时机错误**:在内层循环外加入子节点会导致层级混乱
   - 💡 解决:必须在处理当前节点时立即加入其子节点(内层for循环内)

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:前端组件树渲染 — React/Vue按层渲染虚拟DOM,优先显示上层组件
- **场景2**:网络爬虫 — 广度优先爬取网页,先爬完同级链接再深入下一级
- **场景3**:游戏AI寻路 — BFS找最短路径(如吃豆人游戏中寻找最近的豆子)
- **场景4**:社交网络 — 查找"N度好友"(1度是直接好友,2度是好友的好友)

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 107. 二叉树的层序遍历II | Medium | BFS+反转 | 结果反转即可:`return result[::-1]` |
| LeetCode 103. 二叉树的锯齿形层序遍历 | Medium | BFS+方向标志 | 奇偶层交替反转 |
| LeetCode 199. 二叉树的右视图 | Medium | BFS取每层最后 | 每层只取最后一个元素 |
| LeetCode 513. 找树左下角的值 | Medium | BFS找最后一层第一个 | 记录最后一层的第一个节点 |
| LeetCode 111. 二叉树的最小深度 | Easy | BFS最短路径 | 第一次遇到叶子节点就是最短路径 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一棵二叉树,返回每一层节点值的**平均值**,结果是一个浮点数列表。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

在BFS层序遍历的基础上,对每层的值求和后除以节点数即可。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
from collections import deque

def averageOfLevels(root: Optional[TreeNode]) -> List[float]:
    """返回每层节点值的平均值"""
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level_sum = 0  # 当前层的和

        for _ in range(level_size):
            node = queue.popleft()
            level_sum += node.val  # 累加节点值

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        # 计算当前层平均值
        result.append(level_sum / level_size)

    return result


# 测试
root = TreeNode(3, TreeNode(9), TreeNode(20, TreeNode(15), TreeNode(7)))
print(averageOfLevels(root))  # 输出:[3.0, 14.5, 11.0]
```

核心思路:在层序遍历框架上,增加 `level_sum` 累加每层节点值,最后除以 `level_size` 得到平均值。时间复杂度O(n),空间复杂度O(w)。

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
