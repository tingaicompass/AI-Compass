> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第64课:单词搜索

> **模块**:回溯算法 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/word-search/
> **前置知识**:第59课(全排列)、第63课(括号生成)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定一个 m×n 的二维字符网格 board 和一个字符串单词 word,判断单词是否存在于网格中。

单词必须按照字母顺序,通过相邻的单元格内的字母构成,其中"相邻"单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母不允许被重复使用。

**示例:**
```
输入:
board = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]
word = "ABCCED"

输出:true
解释:路径为 A→B→C→C→E→D

输入:word = "SEE"
输出:true
解释:路径为 S→E→E

输入:word = "ABCB"
输出:false
解释:B 不能被重复使用
```

**约束条件:**
- `m == board.length`
- `n == board[i].length`
- `1 <= m, n <= 6`
- `1 <= word.length <= 15`
- board 和 word 仅由大小写英文字母组成

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单字母 | word="A", board=[['A']] | true | 基本功能 |
| 找不到 | word="Z", board=[['A','B']] | false | 不存在的字母 |
| 需要回溯 | word="ABCB", board=[['A','B','C']] | false | 不能重复使用单元格 |
| 曲折路径 | word="SEE", 如示例 | true | 需要转弯 |
| 最大规模 | 6×6 网格, word长度15 | - | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在一个迷宫里寻宝,宝藏的位置用一串密码表示(word):
>
> 🗺️ **规则**:
> - 你从迷宫的某个格子出发,只能往上下左右四个方向走
> - 每走到一个格子,必须捡起那里的字母,拼成密码的下一个字母
> - 走过的格子不能再走(否则会触发机关!)
> - 如果能拼出完整密码,就找到宝藏
>
> 🐌 **笨办法**:尝试所有可能的路径(指数级),逐一检查是否能拼出 word
>
> 🚀 **聪明办法**:**DFS + 回溯 + 剪枝**:
> - 从每个格子出发,尝试匹配 word 的第一个字母
> - 如果匹配,标记为"已访问",继续往四个方向搜索下一个字母
> - 如果某个方向不匹配,立即剪枝(不继续搜索)
> - 回溯时,恢复"已访问"标记,让其他路径可以使用这个格子

### 关键洞察
**核心技巧:**用原地修改(将访问过的格子临时改为特殊字符如 `'#'`)来标记路径,避免使用额外的 visited 数组,回溯时再恢复原字符。

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二维字符数组 board (m×n),字符串 word
- **输出**:布尔值,表示是否存在路径
- **限制**:只能上下左右移动,不能重复使用同一个格子

### Step 2:先想笨办法(暴力法)
从每个格子出发,用 DFS 尝试所有可能的路径,检查是否能拼出 word。
- 时间复杂度:O(m * n * 4^L) — 从 m*n 个起点出发,每个格子有 4 个方向,路径长度 L
- 瓶颈在哪:大量重复搜索,没有剪枝

### Step 3:瓶颈分析 → 优化方向
笨办法的问题是"不匹配也继续搜索"。能不能**一旦发现不匹配就立即停止**?
- 核心问题:如何避免重复访问同一个格子
- 优化思路:用"标记+回溯"来管理访问状态,用"字符不匹配"来剪枝

### Step 4:选择武器
- 选用:**DFS 回溯 + 原地标记 + 剪枝**
- 理由:
  - DFS 天然适合"路径搜索"
  - 回溯可以在探索失败后恢复状态
  - 原地标记节省空间
  - 提前剪枝避免无效搜索

> 🔑 **模式识别提示**:当题目涉及"二维网格+路径搜索+不能重复访问",优先考虑"DFS回溯+visited标记"

---

## 🔑 解法一:DFS回溯 + visited数组(清晰版)

### 思路
用一个额外的 visited 二维数组标记访问状态,DFS 时检查四个方向,回溯时恢复 visited 状态。

### 图解过程

```
以示例为例,搜索 word = "ABCCED":

board = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Step 1:遍历找到起点 'A'(0,0)
visited = [
  [T, F, F, F],
  [F, F, F, F],
  [F, F, F, F]
]

Step 2:从 'A' 搜索下一个字符 'B'
  - 上:越界 ✗
  - 下:'S' ≠ 'B' ✗
  - 左:越界 ✗
  - 右:'B' = 'B' ✓ → 继续
visited = [
  [T, T, F, F],
  [F, F, F, F],
  [F, F, F, F]
]

Step 3:从 'B' 搜索 'C'
  - 上:越界 ✗
  - 下:'F' ≠ 'C' ✗
  - 左:'A' 已访问 ✗
  - 右:'C' = 'C' ✓ → 继续

...继续搜索,路径为:
(0,0)A → (0,1)B → (0,2)C → (1,2)C → (2,2)E → (2,1)D ✓

最终找到完整路径,返回 true!
```

### Python代码

```python
from typing import List


def exist_with_visited(board: List[List[str]], word: str) -> bool:
    """
    解法一:DFS回溯 + visited数组
    思路:用额外数组标记访问状态
    """
    if not board or not board[0]:
        return False

    m, n = len(board), len(board[0])
    visited = [[False] * n for _ in range(m)]

    def dfs(r: int, c: int, index: int) -> bool:
        """
        从 (r, c) 开始搜索 word[index:]
        """
        # 终止条件:匹配完整个 word
        if index == len(word):
            return True

        # 边界检查
        if r < 0 or r >= m or c < 0 or c >= n:
            return False
        # 已访问或字符不匹配
        if visited[r][c] or board[r][c] != word[index]:
            return False

        # 标记为已访问
        visited[r][c] = True

        # 尝试四个方向
        found = (dfs(r + 1, c, index + 1) or  # 下
                 dfs(r - 1, c, index + 1) or  # 上
                 dfs(r, c + 1, index + 1) or  # 右
                 dfs(r, c - 1, index + 1))    # 左

        # 回溯:恢复状态
        visited[r][c] = False

        return found

    # 从每个格子尝试作为起点
    for i in range(m):
        for j in range(n):
            if board[i][j] == word[0]:  # 优化:只从匹配首字母的格子开始
                if dfs(i, j, 0):
                    return True

    return False


# ✅ 测试
board1 = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
print(exist_with_visited(board1, "ABCCED"))  # 期望输出:True
print(exist_with_visited(board1, "SEE"))     # 期望输出:True
print(exist_with_visited(board1, "ABCB"))    # 期望输出:False
```

### 复杂度分析
- **时间复杂度**:O(m * n * 4^L) — 最坏情况下从每个格子出发搜索,每次有 4 个方向,路径长度 L
  - 实际情况:剪枝使得平均复杂度远低于理论上界(大多数路径很快就被剪枝了)
- **空间复杂度**:O(m * n + L) — visited 数组 O(m*n) + 递归栈 O(L)

### 优缺点
- ✅ 思路清晰,visited 数组直观
- ❌ 额外空间占用 O(m*n)

---

## 🏆 解法二:DFS回溯 + 原地标记(最优解)

### 优化思路
不用额外的 visited 数组,而是**原地修改 board**:
- 访问某个格子时,临时将其改为特殊字符(如 `'#'` 或 `'\0'`)
- 回溯时,恢复原字符

这样空间复杂度降为 O(L)(只有递归栈),且代码更简洁!

> 💡 **关键想法**:因为 board 和 word 只包含大小写字母,我们可以用 `'#'` 或任何非字母字符作为"已访问"标记,不会与原字符冲突。

### 图解过程

```
相同的搜索过程,但标记方式不同:

初始 board:
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Step 1:访问 (0,0) 'A' → 临时改为 '#'
[
  ['#','B','C','E'],  ← 标记已访问
  ['S','F','C','S'],
  ['A','D','E','E']
]

Step 2:访问 (0,1) 'B' → 临时改为 '#'
[
  ['#','#','C','E'],  ← 继续标记
  ['S','F','C','S'],
  ['A','D','E','E']
]

...搜索完成后,回溯恢复:
[
  ['#','#','C','E'],
  ['S','F','C','S'],
  ['A','D','#','E']  ← 假设搜索到这里失败
]
↓ 回溯恢复
[
  ['A','B','C','E'],  ← 恢复原状
  ['S','F','C','S'],
  ['A','D','E','E']
]
```

### Python代码

```python
def exist(board: List[List[str]], word: str) -> bool:
    """
    🏆 解法二:DFS回溯 + 原地标记(最优解)
    思路:用原地修改代替 visited 数组,节省空间
    """
    if not board or not board[0]:
        return False

    m, n = len(board), len(board[0])

    def dfs(r: int, c: int, index: int) -> bool:
        """
        从 (r, c) 开始搜索 word[index:]
        """
        # 终止条件:匹配完整个 word
        if index == len(word):
            return True

        # 边界检查和字符匹配
        if (r < 0 or r >= m or c < 0 or c >= n or
            board[r][c] != word[index]):
            return False

        # 原地标记:保存原字符,然后改为 '#'
        temp = board[r][c]
        board[r][c] = '#'  # 标记为已访问

        # 尝试四个方向(上下左右)
        found = (dfs(r + 1, c, index + 1) or  # 下
                 dfs(r - 1, c, index + 1) or  # 上
                 dfs(r, c + 1, index + 1) or  # 右
                 dfs(r, c - 1, index + 1))    # 左

        # 回溯:恢复原字符
        board[r][c] = temp

        return found

    # 从每个格子尝试作为起点
    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):  # 直接搜索,dfs内部会检查首字母
                return True

    return False


# ✅ 测试
board1 = [
    ['A', 'B', 'C', 'E'],
    ['S', 'F', 'C', 'S'],
    ['A', 'D', 'E', 'E']
]
print(exist(board1, "ABCCED"))  # 期望输出:True
print(exist(board1, "SEE"))     # 期望输出:True
print(exist(board1, "ABCB"))    # 期望输出:False

# 边界测试
board2 = [['A']]
print(exist(board2, "A"))  # 期望输出:True
board3 = [['A', 'B'], ['C', 'D']]
print(exist(board3, "ABDC"))  # 期望输出:True
```

### 复杂度分析
- **时间复杂度**:O(m * n * 4^L) — 与解法一相同,但实际运行更快(减少了 visited 访问开销)
  - 具体地说:对于 3×3 网格,word 长度 5,最坏情况约 9 * 4^5 = 9216 次操作
  - 剪枝效果:平均情况下,大部分搜索在前几步就被剪枝,实际远少于理论上界
- **空间复杂度**:O(L) — 只有递归栈,不需要额外 visited 数组

### 为什么是最优解
- ✅ 空间复杂度从 O(m*n) 降为 O(L),在大网格上节省明显
- ✅ 原地标记避免了额外的数组访问,实际运行更快
- ✅ 代码更简洁,只需一个 temp 变量

---

## 🐍 Pythonic 写法

利用 Python 的元组和解包简化方向遍历:

```python
def exist_pythonic(board: List[List[str]], word: str) -> bool:
    """Pythonic 版本:用元组表示四个方向"""
    if not board or not board[0]:
        return False

    m, n = len(board), len(board[0])
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # 下上右左

    def dfs(r: int, c: int, index: int) -> bool:
        if index == len(word):
            return True
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != word[index]:
            return False

        temp, board[r][c] = board[r][c], '#'  # Pythonic 交换

        # 用 any() 简化 or 链
        found = any(dfs(r + dr, c + dc, index + 1) for dr, dc in directions)

        board[r][c] = temp
        return found

    # 用 any() 简化双重循环
    return any(dfs(i, j, 0) for i in range(m) for j in range(n))
```

> ⚠️ **面试建议**:先写清晰版本,再提 Pythonic 优化。面试官更看重你的**DFS回溯思路**和**剪枝策略**。

---

## 📊 解法对比

| 维度 | 解法一:visited数组 | 🏆 解法二:原地标记(最优) |
|------|----------------|---------------------|
| 时间复杂度 | O(m*n*4^L) | **O(m*n*4^L)** ← 实际更快 |
| 空间复杂度 | O(m*n + L) | **O(L)** ← 更优 |
| 代码难度 | 简单 | 简单 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 适合不能修改原数组的场景 | **通用,空间受限场景首选** |

**面试建议**:
1. 先口述思路:"这是一个网格搜索问题,用 DFS 回溯"(30秒)
2. 讲解核心技巧:
   - **四方向搜索**:上下左右四个方向
   - **剪枝条件**:越界、字符不匹配、已访问
   - **回溯恢复**:搜索失败后恢复状态
3. 写代码时强调🏆原地标记的优势:"用 `'#'` 标记已访问,节省 O(m*n) 空间"
4. 手动演示小例子,如 2×2 网格搜索 "AB"
5. 分析复杂度:虽然理论是 O(4^L),但剪枝使平均情况远优于此

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请判断二维网格中是否存在某个单词。

**你**:(审题30秒)好的,这道题是在二维网格中搜索路径,需要满足:
1. 路径上的字符按顺序组成 word
2. 只能上下左右移动
3. 同一个格子不能重复使用

这是一个典型的**网格 DFS 回溯**问题。思路是:
1. 从每个格子作为起点尝试
2. 用 DFS 递归搜索四个方向
3. 用"原地标记"记录访问状态(将当前格子改为 `'#'`)
4. 如果找到完整路径返回 true,否则回溯并恢复状态

时间复杂度 O(m*n*4^L),空间复杂度 O(L)(只有递归栈)。

**面试官**:很好,请写代码。

**你**:(边写边说)
```python
def exist(board, word):
    m, n = len(board), len(board[0])

    def dfs(r, c, index):
        # 终止:匹配完整个 word
        if index == len(word):
            return True

        # 剪枝:越界或不匹配
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != word[index]:
            return False

        # 标记已访问
        temp = board[r][c]
        board[r][c] = '#'

        # 搜索四个方向
        found = (dfs(r+1, c, index+1) or dfs(r-1, c, index+1) or
                 dfs(r, c+1, index+1) or dfs(r, c-1, index+1))

        # 回溯恢复
        board[r][c] = temp
        return found

    # 从每个格子尝试
    for i in range(m):
        for j in range(n):
            if dfs(i, j, 0):
                return True
    return False
```

**面试官**:测试一下?

**你**:用示例测试:
```
board = [['A','B','C','E'],
         ['S','F','C','S'],
         ['A','D','E','E']]
word = "ABCCED"
```
- 从 (0,0) 'A' 开始,匹配 word[0]
- 搜索四个方向,右边是 'B',匹配 word[1]
- 继续搜索...,最终路径: A→B→C→C(向下)→E→D
- 返回 true ✓

再测试 word="ABCB":
- 路径 A→B→C 后,需要 'B',但左边的 'B' 已被标记为 '#'
- 无法继续,返回 false ✓

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么用 `'#'` 标记而非布尔值?" | **原因**:题目保证 board 只包含大小写字母,`'#'` 不会与原字符冲突。这样可以原地修改,不需要额外空间。如果不能修改原数组,才用 visited 数组。 |
| "如果有多个单词要搜索呢?" | **优化方向**:可以用 Trie(前缀树)优化。把所有单词插入 Trie,然后一次 DFS 就能同时搜索多个单词,复杂度从 O(k*m*n*4^L) 降为 O(m*n*4^L),其中 k 是单词数。这就是 LeetCode 212"单词搜索 II"。 |
| "能不能用 BFS?" | **可以,但不推荐**:BFS 需要队列存储状态(位置+已访问集合),空间复杂度会很高。DFS 回溯更自然,且空间只需 O(L)。 |
| "如果网格很大,如何优化?" | **剪枝优化**:1)提前统计 board 中每个字符的出现次数,如果 word 中某个字符的次数超过 board,直接返回 false。2)如果 word[0] 在 board 中出现次数少,从 word[0] 开始搜;否则从 word[-1] 反向搜。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:原地标记 — 节省空间的经典技巧
temp, board[r][c] = board[r][c], '#'  # 保存+标记
# ... 搜索 ...
board[r][c] = temp  # 恢复

# 技巧2:方向数组 — 简化四方向遍历
directions = [(1,0), (-1,0), (0,1), (0,-1)]
for dr, dc in directions:
    dfs(r + dr, c + dc, index + 1)

# 技巧3:any() 短路求值 — 找到一个就停止
found = any(dfs(r+dr, c+dc, index+1) for dr, dc in directions)
# 等价于 or 链,但更 Pythonic
```

### 💡 底层原理(选读)

> **DFS vs BFS 在网格搜索中的选择**:
>
> 1. **DFS 适用场景**:
>    - 需要找到"一条"路径(本题)
>    - 需要回溯状态(如撤销标记)
>    - 路径可能很长(递归深度可接受)
>
> 2. **BFS 适用场景**:
>    - 需要找"最短"路径(如最少步数)
>    - 多源同时扩散(如腐烂的橘子)
>    - 不需要回溯状态
>
> 3. **为什么本题用 DFS?**
>    - 本题只需判断"存在性",不需要最短路径
>    - DFS 可以用原地标记+回溯,空间 O(L)
>    - BFS 需要存储状态(位置+visited 集合),空间 O(m*n*L)

### 算法模式卡片 📐
- **模式名称**:网格 DFS 回溯(Grid DFS Backtracking)
- **适用条件**:二维网格中的路径搜索,不能重复访问
- **识别关键词**:"二维网格"+"路径"+"不能重复"+"存在性"
- **模板代码**:
```python
def grid_dfs_backtrack(grid, target):
    m, n = len(grid), len(grid[0])

    def dfs(r, c, state):
        # 终止条件
        if is_target_reached(state):
            return True

        # 边界和剪枝
        if r < 0 or r >= m or c < 0 or c >= n or not is_valid(grid[r][c], state):
            return False

        # 标记
        temp, grid[r][c] = grid[r][c], VISITED_MARK

        # 四方向搜索
        found = (dfs(r+1, c, next_state) or dfs(r-1, c, next_state) or
                 dfs(r, c+1, next_state) or dfs(r, c-1, next_state))

        # 回溯
        grid[r][c] = temp
        return found

    # 从每个格子尝试
    for i in range(m):
        for j in range(n):
            if dfs(i, j, initial_state):
                return True
    return False
```

### 易错点 ⚠️
1. **忘记回溯恢复**:只标记 `board[r][c] = '#'`,不恢复原值 → 导致后续搜索失败
   - 正确做法:必须在回溯时恢复 `board[r][c] = temp`

2. **边界条件不全**:只检查 `board[r][c] != word[index]`,不检查越界 → 数组越界错误
   - 正确做法:先检查边界,再检查字符匹配

3. **递归终止条件错误**:写成 `if index == len(word) - 1` → 会漏掉最后一个字符的匹配
   - 正确做法:`if index == len(word)` 表示已经匹配完所有字符

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:拼字游戏**:Scrabble、Wordle 等游戏判断用户输入的单词是否能在棋盘上拼出
- **场景2:迷宫寻路**:游戏中的 NPC 寻找从起点到终点的路径(DFS 找存在性,BFS 找最短路)
- **场景3:图像处理**:泛洪填充(flood fill)算法,如 Photoshop 的"魔棒"工具,用 DFS 找连通区域
- **场景4:电路板布线**:检查电路板上两个焊点之间是否存在连通路径

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 200. 岛屿数量 | Medium | 网格 DFS | 经典入门题,统计连通块数量 |
| LeetCode 212. 单词搜索 II | Hard | Trie + DFS | 本题的进阶版,搜索多个单词,需要用前缀树优化 |
| LeetCode 130. 被围绕的区域 | Medium | DFS/BFS | 从边界开始 DFS,标记不被围绕的区域 |
| LeetCode 417. 太平洋大西洋水流问题 | Medium | 逆向 DFS | 从两个边界分别 DFS,找交集 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定 board 和 word,返回**所有可能的路径**(用坐标列表表示)。例如:
```
board = [['A','B'],['C','D']]
word = "AB"
输出:[[(0,0), (0,1)]]  # 只有一条路径
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

在 DFS 中增加一个参数 `path` 记录当前路径,终止时将 path 加入结果列表。注意要用 `path[:]` 复制路径,否则会被后续修改影响。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def findAllPaths(board: List[List[str]], word: str) -> List[List[tuple]]:
    """
    返回所有可能的路径(坐标列表)
    """
    m, n = len(board), len(board[0])
    result = []

    def dfs(r: int, c: int, index: int, path: List[tuple]):
        # 终止:找到完整路径
        if index == len(word):
            result.append(path[:])  # 必须复制!
            return

        # 剪枝
        if r < 0 or r >= m or c < 0 or c >= n or board[r][c] != word[index]:
            return

        # 标记+记录路径
        temp = board[r][c]
        board[r][c] = '#'
        path.append((r, c))

        # 四方向搜索
        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            dfs(r + dr, c + dc, index + 1, path)

        # 回溯:恢复标记和路径
        board[r][c] = temp
        path.pop()

    # 从每个格子尝试
    for i in range(m):
        for j in range(n):
            dfs(i, j, 0, [])

    return result


# 测试
board = [['A','B','C'],
         ['D','E','F']]
paths = findAllPaths(board, "ABEF")
print(paths)  # 输出:所有可能的路径坐标
```

**关键修改**:
1. 增加 `path` 参数记录路径
2. 终止时用 `path[:]` 复制当前路径(不能直接 append path,因为后续会被修改)
3. 回溯时要 `path.pop()` 恢复路径状态

**扩展**:这个变体在某些场景很有用,比如棋类游戏中需要展示所有可能的走法。

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
