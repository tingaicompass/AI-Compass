# 📖 第103课:单词搜索II

> **模块**:前缀树 | **难度**:Hard ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/word-search-ii/
> **前置知识**:第64课(单词搜索),第101课(实现Trie),第102课(添加与搜索单词)
> **预计学习时间**:40分钟

---

## 🎯 题目描述

给定一个二维字符网格`board`和一个单词列表`words`,找出所有同时在二维网格和单词列表中出现的单词。

单词必须按照字母顺序,通过相邻的单元格内的字母构成。相邻单元格是水平或垂直方向相邻,同一单元格内的字母在一个单词中不能被重复使用。

**示例:**
```
输入:
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]

输出: ["eat","oath"]

解释:
"oath": o(0,0) → a(0,1) → t(1,1) → h(2,1)
"eat":  e(1,0) → a(1,1) → t(1,2)
```

**约束条件:**
- m == board.length
- n == board[i].length
- 1 ≤ m, n ≤ 12
- board[i][j] 是小写英文字母
- 1 ≤ words.length ≤ 3 * 10^4
- 1 ≤ words[i].length ≤ 10
- words[i] 由小写英文字母组成
- words 中的所有字符串互不相同

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单字母单词 | words=["a"], board=[["a"]] | ["a"] | 最小输入 |
| 不存在的单词 | words=["xyz"] | [] | 负向测试 |
| 重复使用字母 | "aa"但board只有一个'a' | [] | 路径不能重复 |
| 多个单词共享前缀 | ["app","apple","application"] | 正确返回所有找到的 | Trie剪枝优势 |
| 最大规模 | 12x12网格,30000个单词 | 正确处理不超时 | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在一个巨大的字母迷宫中寻找宝藏。你有一张藏宝图(单词列表),每个宝藏都是一个单词,需要按照特定路径在迷宫中找到它。
>
> 🐌 **笨办法**:对每个单词,从网格的每个位置开始DFS搜索。如果有10000个单词,12x12的网格,就要进行10000 * 144次搜索,每次都要重新探索路径,大量重复劳动。时间复杂度O(k * m * n * 4^L),k是单词数,L是单词长度。
>
> 🚀 **聪明办法**:先把所有单词构建成一棵"藏宝图树"(Trie前缀树)。从网格的每个位置开始一次DFS,同时对照Trie树前进。如果当前路径的前缀不在Trie中,立即剪枝;如果路径匹配到Trie中的单词结尾,就找到一个宝藏。这样只需遍历网格一次,复杂度降为O(m * n * 4^L),且实际有大量剪枝。

### 关键洞察
**Trie + 网格DFS回溯:用Trie剪枝,避免探索不可能形成任何单词的路径。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二维字符网格`board`和单词列表`words`
- **输出**:所有在网格中能找到的单词(去重)
- **限制**:单词路径不能重复使用同一单元格

### Step 2:先想笨办法(暴力法)
对每个单词,从网格的每个位置尝试DFS搜索:
- 外层循环:遍历每个单词(k个)
- 中层循环:遍历网格每个位置(m*n个)
- 内层:DFS回溯搜索该单词(最多4^L个分支)
- 时间复杂度:O(k * m * n * 4^L),k=30000,m=n=12,L=10时,约10^13次操作,必然超时
- 瓶颈在哪:大量重复的前缀搜索,如"app"、"apple"、"application"都要重复搜索"app"前缀

### Step 3:瓶颈分析 → 优化方向
暴力法的核心问题:
- **重复前缀搜索**:多个单词如果有公共前缀,会重复探索相同路径
- **无效路径无法提前剪枝**:即使当前路径"xyz"不是任何单词的前缀,仍要继续探索

优化思路:
- **能不能一次DFS同时搜索所有单词?** → 用Trie将单词组织成树结构
- **能不能提前剪枝无效路径?** → Trie中不存在的前缀立即回溯

### Step 4:选择武器
- 选用:**Trie前缀树 + 网格DFS回溯**
- 理由:
  1. Trie构建:O(sum(len(word)))一次性构建,复杂度分摊到所有单词
  2. DFS时对照Trie前进,当前路径不在Trie中立即剪枝
  3. 找到单词后立即从Trie中删除,避免重复结果
  4. 时间复杂度优化到O(m * n * 4^L),且有大量剪枝,实际远小于此

> 🔑 **模式识别提示**:当题目出现"在网格中搜索多个单词"、"大量单词有公共前缀",优先考虑"Trie + 网格DFS"

---

## 🔑 解法一:逐个单词DFS(暴力法)

### 思路
对每个单词,从网格的每个位置尝试DFS搜索。这是最直接的方法,但大量重复搜索导致超时。

### 图解过程

```
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath", "eat"]

搜索"oath":
尝试从(0,0)开始: o → a → ... (探索所有路径)
尝试从(0,1)开始: a → ... (不匹配)
...
最终从(0,0) → (0,1) → (1,1) → (2,1) 找到

搜索"eat":
重新从所有位置开始搜索...
最终从(1,0) → (1,1) → (1,2) 找到

问题:两个单词的搜索是独立的,无法共享前缀
```

### Python代码

```python
from typing import List


class Solution:
    """
    解法一:逐个单词DFS
    思路:对每个单词在网格中独立搜索
    """
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not words:
            return []

        m, n = len(board), len(board[0])
        result = []

        def dfs(r: int, c: int, word: str, index: int) -> bool:
            """DFS搜索单词word,当前处理第index个字符"""
            # 递归终止:找到完整单词
            if index == len(word):
                return True

            # 边界检查
            if r < 0 or r >= m or c < 0 or c >= n:
                return False
            if board[r][c] != word[index]:
                return False

            # 标记当前位置已访问
            temp = board[r][c]
            board[r][c] = '#'

            # 向四个方向搜索
            found = (
                dfs(r + 1, c, word, index + 1) or
                dfs(r - 1, c, word, index + 1) or
                dfs(r, c + 1, word, index + 1) or
                dfs(r, c - 1, word, index + 1)
            )

            # 回溯:恢复现场
            board[r][c] = temp
            return found

        # 对每个单词独立搜索
        for word in words:
            # 从网格的每个位置尝试
            for i in range(m):
                for j in range(n):
                    if dfs(i, j, word, 0):
                        result.append(word)
                        break  # 找到后跳出内层循环
                else:
                    continue
                break  # 跳出外层循环

        return result


# ✅ 测试
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]
sol = Solution()
print(sol.findWords(board, words))  # 期望输出:["oath","eat"]
```

### 复杂度分析
- **时间复杂度**:O(k * m * n * 4^L)
  - k是单词数量(最多30000)
  - m*n是网格大小(最多144)
  - 4^L是DFS分支(最坏每层4个方向,深度L=10)
  - 具体地说:30000 * 144 * 4^10 ≈ 4.6 * 10^11,必然超时
- **空间复杂度**:O(L) — 递归栈深度

### 优缺点
- ✅ 实现直观,易于理解
- ❌ 大量重复搜索,无法共享公共前缀
- ❌ 无法提前剪枝,性能极差

---

## 🏆 解法二:Trie + 网格DFS回溯(最优解)

### 优化思路
解法一的痛点是对每个单词独立搜索。我们可以:
1. **预处理**:将所有单词构建成Trie树
2. **一次DFS搜索所有单词**:从网格每个位置开始DFS,同时对照Trie前进
3. **剪枝**:当前路径不在Trie中,立即回溯
4. **去重**:找到单词后立即从Trie中删除,避免重复添加

> 💡 **关键想法**:
> - Trie将多个单词的公共前缀合并,一次探索可以同时匹配多个单词
> - DFS时携带当前Trie节点,字符匹配失败或节点为空时立即剪枝
> - 找到单词后删除Trie节点,既去重又减少后续搜索空间

### 图解过程

```
Step 1: 构建Trie
words = ["oath", "eat"]

Trie结构:
      root
     /    \
    o      e
    |      |
    a      a
    |      |
    t      t*  (* 表示单词结尾)
    |
    h*

Step 2: 网格DFS
从(0,0)开始,字符'o':
  检查Trie: root → 'o' 存在,继续
  移动到(0,1),字符'a':
    检查Trie: o → 'a' 存在,继续
    移动到(1,1),字符't':
      检查Trie: a → 't' 存在,继续
      移动到(2,1),字符'h':
        检查Trie: t → 'h' 存在,且是单词结尾
        → 找到"oath",加入结果,从Trie删除

从(1,0)开始,字符'e':
  检查Trie: root → 'e' 存在,继续
  移动到(1,1),字符'a':
    检查Trie: e → 'a' 存在,继续
    移动到(1,2),字符't':
      检查Trie: a → 't' 存在,且是单词结尾
      → 找到"eat",加入结果

剪枝示例:
从(0,0)开始,字符'o':
  移动到(1,0),字符'e':
    检查Trie: o → 'e' 不存在
    → 立即剪枝,回溯
```

### Python代码

```python
from typing import List


class TrieNode:
    """Trie树节点"""
    def __init__(self):
        self.children = {}
        self.word = None  # 存储完整单词(而非is_end标记)


class Solution:
    """
    解法二:Trie + 网格DFS回溯
    思路:用Trie组织单词,DFS时对照Trie剪枝
    """
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        if not board or not words:
            return []

        # Step 1: 构建Trie树
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word  # 叶子节点存储完整单词

        # Step 2: 网格DFS搜索
        m, n = len(board), len(board[0])
        result = []

        def dfs(r: int, c: int, node: TrieNode):
            """
            DFS回溯
            r, c: 当前网格位置
            node: 当前Trie节点
            """
            # 边界检查
            if r < 0 or r >= m or c < 0 or c >= n:
                return
            char = board[r][c]
            if char == '#' or char not in node.children:
                return  # 已访问或路径不在Trie中,剪枝

            # 移动到下一个Trie节点
            next_node = node.children[char]

            # 找到单词
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None  # 去重:删除避免重复添加

            # 标记当前位置已访问
            board[r][c] = '#'

            # 向四个方向DFS
            dfs(r + 1, c, next_node)
            dfs(r - 1, c, next_node)
            dfs(r, c + 1, next_node)
            dfs(r, c - 1, next_node)

            # 回溯:恢复现场
            board[r][c] = char

            # 优化:删除空节点(可选,减少后续搜索空间)
            if not next_node.children:
                del node.children[char]

        # 从网格每个位置开始DFS
        for i in range(m):
            for j in range(n):
                dfs(i, j, root)

        return result


# ✅ 测试
board = [
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]
words = ["oath","pea","eat","rain"]
sol = Solution()
print(sol.findWords(board, words))  # 期望输出:["oath","eat"]

# 边界测试
board2 = [['a','b'],['c','d']]
words2 = ["abcd"]
print(sol.findWords(board2, words2))  # 期望输出:[]

board3 = [['a']]
words3 = ["a"]
print(sol.findWords(board3, words3))  # 期望输出:["a"]
```

### 复杂度分析
- **时间复杂度**:
  - 构建Trie: O(sum(len(word))) — 遍历所有单词的所有字符
  - DFS搜索: O(m * n * 4^L) — 从每个位置开始,最多探索4^L个分支
  - 但实际有大量剪枝:
    - Trie剪枝:路径不在Trie中立即停止
    - 删除节点:找到单词后删除,减少后续搜索
    - 实际复杂度远小于理论上限,通常接近O(m * n * L)
- **空间复杂度**:
  - Trie: O(sum(len(word))) — 最坏所有单词无公共前缀
  - 递归栈: O(L) — 最大深度为单词长度

### 关键优化点
1. **Trie存储完整单词**:`node.word = word`而非布尔标记,便于直接添加结果
2. **找到后立即删除**:`next_node.word = None`避免重复添加同一单词
3. **动态删除空节点**:回溯时删除无子节点的节点,减少后续搜索空间
4. **原地修改标记访问**:用`'#'`标记已访问,无需额外visited集合

---

## ⚡ 解法三:Trie + 按长度优化(进阶优化)

### 优化思路
在解法二基础上,先按单词长度排序,优先搜索短单词。因为短单词更容易找到,早期找到可以提前剪枝Trie的更多分支。

### Python代码

```python
class Solution:
    """解法三:Trie + 按长度排序优化"""
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # 按长度排序,优先搜索短单词
        words.sort(key=len)

        # 其余逻辑同解法二
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word

        m, n = len(board), len(board[0])
        result = []

        def dfs(r, c, node):
            if r < 0 or r >= m or c < 0 or c >= n:
                return
            char = board[r][c]
            if char == '#' or char not in node.children:
                return

            next_node = node.children[char]
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None

            board[r][c] = '#'
            dfs(r + 1, c, next_node)
            dfs(r - 1, c, next_node)
            dfs(r, c + 1, next_node)
            dfs(r, c - 1, next_node)
            board[r][c] = char

            if not next_node.children:
                del node.children[char]

        for i in range(m):
            for j in range(n):
                dfs(i, j, root)

        return result
```

### 复杂度分析
- **时间复杂度**:多了排序O(k log k),但DFS中剪枝更多,总体可能更快
- **空间复杂度**:同解法二

---

## 🐍 Pythonic 写法

利用字典嵌套简化Trie节点:

```python
class Solution:
    """Pythonic写法:字典嵌套实现Trie"""
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # 构建Trie(字典嵌套)
        trie = {}
        for word in words:
            node = trie
            for char in word:
                node = node.setdefault(char, {})
            node['#'] = word  # '#'标记单词结尾,值为完整单词

        m, n = len(board), len(board[0])
        result = []

        def dfs(r, c, node):
            if not (0 <= r < m and 0 <= c < n):
                return
            char = board[r][c]
            if char not in node or char == '#':
                return

            next_node = node[char]
            if '#' in next_node:
                result.append(next_node['#'])
                del next_node['#']

            board[r][c] = '#'
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                dfs(r + dr, c + dc, next_node)
            board[r][c] = char

            if not next_node:
                del node[char]

        for i in range(m):
            for j in range(n):
                dfs(i, j, trie)

        return result
```

**解释**:
- `setdefault(char, {})`:简化Trie构建
- `for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]`:四方向遍历更简洁
- `if not (0 <= r < m and 0 <= c < n)`:边界检查更Pythonic

> ⚠️ **面试建议**:先写解法二的清晰版本(用TrieNode类),逻辑清晰易调试,再提Pythonic写法。

---

## 📊 解法对比

| 维度 | 解法一:逐个DFS | 🏆 解法二:Trie+DFS(最优) | 解法三:按长度优化 |
|------|--------------|----------------------|----------------|
| 时间复杂度 | O(k*m*n*4^L) | **O(m*n*4^L)** ← 时间最优 | O(k log k + m*n*4^L) |
| 空间复杂度 | O(L) | O(sum(len(word)) + L) | O(sum(len(word)) + L) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 仅2-3个单词 | **通用,尤其大量单词** | 大量单词且长度差异大 |

**为什么是最优解**:
- 时间:Trie将k个单词合并,DFS只需O(m*n)次,而非O(k*m*n)次
- 剪枝:路径不在Trie中立即停止,大幅减少无效搜索
- 去重:找到单词立即删除,避免重复结果
- 扩展性:支持动态添加单词、前缀查询等操作

**面试建议**:
1. 先用30秒口述暴力法(逐个单词DFS),表明理解问题
2. 立即优化到🏆最优解(Trie + DFS),展示高级数据结构运用
3. **重点讲解核心思想**:"Trie合并公共前缀,DFS时对照Trie剪枝"
4. 手动模拟一个例子,如搜索"oath"和"eat"的过程,展示Trie如何剪枝
5. 强调关键优化:找到单词后删除节点,回溯时删除空节点

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:给定一个字符网格和单词列表,找出所有在网格中出现的单词。

**你**:(审题30秒)好的,这是一道网格搜索问题。我先想一下...

我的第一个想法是对每个单词,从网格的每个位置开始DFS搜索,时间复杂度O(k * m * n * 4^L)。但这个方法有大量重复搜索,比如"app"和"apple"都要重复搜索"app"前缀。

更优的方法是用**Trie前缀树**预处理所有单词,然后从网格每个位置开始一次DFS,同时对照Trie前进。如果当前路径不在Trie中,立即剪枝。这样复杂度优化到O(m * n * 4^L),且有大量剪枝,实际远小于此。

**面试官**:很好,请写一下代码。

**你**:(边写边说)首先定义Trie节点,包含子节点字典和完整单词...构建Trie,将所有单词插入...然后网格DFS,携带当前Trie节点,字符不匹配或节点为空时剪枝...找到单词立即从Trie删除避免重复...

**面试官**:测试一下?

**你**:用示例输入走一遍...构建Trie后,从(0,0)的'o'开始,对照Trie走o→a→t→h路径,找到"oath"...从(1,0)的'e'开始,走e→a→t路径,找到"eat"...其他位置的DFS因为Trie中不存在对应路径而快速剪枝。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果单词列表非常长,如何优化?" | "可以按长度分组,优先搜索短单词,找到后删除节点可剪枝更多分支;或者多线程并行搜索不同区域" |
| "如果网格非常大呢?" | "可以分块处理,将网格划分成多个子区域,每个区域独立DFS;或者用启发式搜索优先从字符频率高的位置开始" |
| "找到单词后为什么要删除节点?" | "一是去重,避免同一单词被重复添加;二是剪枝,删除后该分支不会再被探索,减少后续搜索空间" |
| "能否不修改board原数组?" | "可以用额外的visited集合标记访问状态,但会增加空间复杂度到O(m*n);原地修改是空间最优解" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:四方向遍历简写 — 用列表推导式简化方向移动
for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    dfs(r + dr, c + dc, next_node)

# 技巧2:Trie存储完整单词 — 节点直接存word而非布尔值
node.word = word  # 而非 node.is_end = True

# 技巧3:动态删除空节点 — 回溯时清理Trie
if not next_node.children:
    del node.children[char]

# 技巧4:边界检查简写 — 链式比较更Pythonic
if not (0 <= r < m and 0 <= c < n):
    return
```

### 💡 底层原理(选读)

> **为什么Trie + DFS比逐个单词DFS快这么多?**
>
> 关键在于**剪枝的时机和粒度**:
> - 逐个单词DFS:只有完整探索一个单词的所有路径后,才能判断是否存在。无法提前剪枝。
> - Trie + DFS:每走一步都检查Trie,路径前缀不存在立即回溯。剪枝粒度是单个字符,而非完整单词。
>
> **举例**:搜索"apple"和"application"
> - 暴力法:先完整搜索"apple"(即使前3个字符"app"不存在),再完整搜索"application"
> - Trie法:搜索时发现'a'不在Trie中,立即停止,两个单词的搜索都被一次剪枝终止
>
> **Trie剪枝的数学本质**:
> - 假设网格中不存在以'a'开头的路径,暴力法仍要尝试所有以'a'开头的单词,复杂度O(k_a * search_cost)
> - Trie法检查一次'a'不存在,剪掉整个'a'子树,复杂度O(1)
> - 当单词列表有大量公共前缀时,剪枝收益呈指数级增长

### 算法模式卡片 📐
- **模式名称**:Trie + 网格DFS回溯
- **适用条件**:在网格/图中搜索大量字符串,且字符串有公共前缀
- **识别关键词**:"网格"、"搜索多个单词"、"路径不重复"、"大量单词"
- **模板代码**:
```python
# 1. 构建Trie
root = TrieNode()
for word in words:
    node = root
    for char in word:
        if char not in node.children:
            node.children[char] = TrieNode()
        node = node.children[char]
    node.word = word

# 2. 网格DFS
def dfs(r, c, node):
    if 越界 or board[r][c] == '#' or board[r][c] not in node.children:
        return

    next_node = node.children[board[r][c]]
    if next_node.word:  # 找到单词
        result.append(next_node.word)
        next_node.word = None  # 删除避免重复

    # 标记访问
    temp = board[r][c]
    board[r][c] = '#'

    # 四方向DFS
    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
        dfs(r+dr, c+dc, next_node)

    # 回溯
    board[r][c] = temp
    if not next_node.children:
        del node.children[temp]

# 3. 从每个位置启动DFS
for i in range(m):
    for j in range(n):
        dfs(i, j, root)
```

### 易错点 ⚠️
1. **忘记回溯恢复现场**:修改`board[r][c] = '#'`后必须在返回前恢复
   - 错误:DFS后不恢复,导致其他路径无法使用该位置
   - 正确:
   ```python
   temp = board[r][c]
   board[r][c] = '#'
   # ... DFS ...
   board[r][c] = temp  # 必须恢复
   ```

2. **重复添加同一单词**:网格中可能有多条路径形成同一单词
   - 错误:`if next_node.word: result.append(next_node.word)` 不删除
   - 正确:
   ```python
   if next_node.word:
       result.append(next_node.word)
       next_node.word = None  # 立即删除
   ```

3. **边界检查顺序错误**:先访问`board[r][c]`再检查越界会导致数组越界
   - 错误:
   ```python
   char = board[r][c]
   if r < 0 or r >= m:  # 已经越界访问了!
       return
   ```
   - 正确:
   ```python
   if r < 0 or r >= m or c < 0 or c >= n:
       return
   char = board[r][c]  # 确保不越界后再访问
   ```

4. **删除节点时机错误**:不能在DFS过程中删除当前正在使用的节点
   - 正确做法:回溯时检查子节点为空再删除
   ```python
   if not next_node.children:
       del node.children[char]
   ```

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:拼字游戏AI**:Scrabble、Words With Friends等游戏中,AI需要在字母板上快速找出所有可能的单词,用Trie存储词典
- **场景2:文本编辑器查找**:VSCode的多文件查找功能,在大量文件中搜索多个关键词,用Trie组织关键词树
- **场景3:生物信息学序列匹配**:在DNA序列网格中搜索大量已知基因片段,Trie加速匹配过程
- **场景4:地图路径搜索**:在地图网格中搜索多个地名路径,如"北京→上海→广州",用Trie组织地名序列

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 79. 单词搜索 | Medium | 网格DFS回溯 | 本题简化版,只搜索一个单词 |
| LeetCode 211. 添加与搜索单词 | Medium | Trie + DFS | 支持通配符`.`的单词搜索 |
| LeetCode 1255. 得分最高的单词集合 | Hard | 回溯 + 位运算 | 在有限字母中组合单词求最大分数 |
| LeetCode 127. 单词接龙 | Hard | BFS + 字典树 | 单词变换的最短路径,可用Trie优化 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:在原题基础上,单词可以沿对角线移动(八个方向),且同一单元格在一个单词中最多使用2次。如何修改算法?

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

1. 方向数组从4个扩展到8个:加上四个对角线方向
2. 访问标记改为计数:用字典记录每个位置的访问次数,不超过2次
3. 回溯时减少计数而非直接恢复

</details>

<details>
<summary>✅ 参考答案</summary>

```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # 构建Trie
        root = TrieNode()
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.word = word

        m, n = len(board), len(board[0])
        result = []
        visit_count = {}  # 记录每个位置的访问次数

        def dfs(r, c, node):
            if not (0 <= r < m and 0 <= c < n):
                return
            char = board[r][c]
            pos = (r, c)

            # 检查:访问次数不超过2,且字符在Trie中
            if visit_count.get(pos, 0) >= 2 or char not in node.children:
                return

            next_node = node.children[char]
            if next_node.word:
                result.append(next_node.word)
                next_node.word = None

            # 标记访问
            visit_count[pos] = visit_count.get(pos, 0) + 1

            # 八方向DFS(四个正方向 + 四个对角线)
            directions = [
                (0, 1), (0, -1), (1, 0), (-1, 0),  # 上下左右
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # 四个对角线
            ]
            for dr, dc in directions:
                dfs(r + dr, c + dc, next_node)

            # 回溯:减少计数
            visit_count[pos] -= 1
            if visit_count[pos] == 0:
                del visit_count[pos]

            if not next_node.children:
                del node.children[char]

        for i in range(m):
            for j in range(n):
                dfs(i, j, root)

        return result
```

**核心修改**:
1. 方向数组扩展到8个
2. 用`visit_count`字典记录访问次数,允许最多2次
3. 回溯时`visit_count[pos] -= 1`而非直接删除

时间复杂度:O(m * n * 8^L),但实际有Trie剪枝

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
