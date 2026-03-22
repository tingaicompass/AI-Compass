# 📖 第102课:添加与搜索单词

> **模块**:前缀树 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/design-add-and-search-words-data-structure/
> **前置知识**:第101课(实现Trie前缀树)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

设计一个数据结构,支持两种操作:添加单词和搜索单词。搜索功能需要支持通配符 `.` 匹配任意单个字母。

**示例:**
```
输入:
["WordDictionary","addWord","addWord","addWord","search","search","search","search"]
[[],["bad"],["dad"],["mad"],["pad"],["bad"],[".ad"],["b.."]]

输出:
[null,null,null,null,false,true,true,true]

解释:
WordDictionary wordDictionary = new WordDictionary();
wordDictionary.addWord("bad");
wordDictionary.addWord("dad");
wordDictionary.addWord("mad");
wordDictionary.search("pad"); → false
wordDictionary.search("bad"); → true
wordDictionary.search(".ad"); → true  (匹配"bad","dad","mad")
wordDictionary.search("b.."); → true  (匹配"bad")
```

**约束条件:**
- 单词长度范围: 1 ≤ word.length ≤ 25
- 单词和搜索模式仅由小写英文字母和 `.` 组成
- addWord 中的单词不含 `.`
- 最多调用 10^4 次 addWord 和 search

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单字母搜索 | search("a") | 根据是否添加过"a" | 最小输入 |
| 全通配符 | search("...") | 匹配所有长度为3的词 | 通配符处理 |
| 混合模式 | search(".a.") | 匹配中间为a的3字母词 | 模式匹配 |
| 不存在 | search("xyz") | false | 负向测试 |
| 长单词 | 25个字符的单词 | 正确添加和搜索 | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在管理一个图书馆的图书检索系统。
>
> 🐌 **笨办法**:把所有书名存在一个数组里,每次搜索时遍历整个数组,逐个比对每个字符。如果搜索"b.d",要检查每本书的书名,判断第1个字符是否为b,第3个字符是否为d,第2个字符可以是任意字母。时间复杂度O(n*m),其中n是单词数量,m是单词长度。
>
> 🚀 **聪明办法**:用前缀树(Trie)按字母顺序组织书名。搜索"b.d"时,从根节点走到'b'子节点,然后对于'.',同时探索所有26个可能的子节点,最后检查这些分支中哪些有'd'结尾。通过树结构剪枝,大幅减少比较次数。

### 关键洞察
**Trie + DFS回溯:遇到通配符时,用DFS同时探索所有可能的分支路径。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:addWord接收字符串,search接收可能含有`.`的模式字符串
- **输出**:search返回布尔值,表示是否存在匹配的单词
- **限制**:需要支持`.`匹配任意单个字符,这是核心难点

### Step 2:先想笨办法(暴力法)
用数组存储所有单词,搜索时遍历数组,逐个字符比较:
- 如果是普通字母,必须严格匹配
- 如果是`.`,可以匹配任意字母
- 时间复杂度:O(n*m),n是单词数量,m是单词长度
- 瓶颈在哪:每次搜索都要遍历所有单词,没有利用单词间的公共前缀

### Step 3:瓶颈分析 → 优化方向
暴力法的核心问题是:
- **重复比较**:即使很多单词的前缀不同,仍要逐个检查
- **无法剪枝**:明知道没有以某个字母开头的单词,仍要扫描全部

优化思路:
- **能不能用结构化方式组织单词?** → 前缀树可以共享公共前缀
- **能不能遇到`.`时只探索有效分支?** → DFS回溯

### Step 4:选择武器
- 选用:**Trie前缀树 + DFS回溯**
- 理由:
  1. Trie共享前缀,addWord时间O(m),空间高效
  2. 搜索时,普通字符直接沿树走,遇到`.`用DFS探索所有子节点
  3. 树结构天然剪枝,无需检查不存在的分支

> 🔑 **模式识别提示**:当题目出现"字符串前缀操作 + 模糊匹配",优先考虑"Trie + DFS"

---

## 🔑 解法一:数组暴力匹配(朴素法)

### 思路
用列表存储所有单词,搜索时遍历列表,逐字符比对。这是最直接的实现,但效率低下。

### 图解过程

```
添加单词: ["bad", "dad", "mad"]
words = ["bad", "dad", "mad"]

搜索 ".ad":
Step 1: 检查 "bad"
  b vs . → 匹配(. 可以是任意字符)
  a vs a → 匹配
  d vs d → 匹配
  → 找到匹配,返回 True

如果继续搜索 "b.." :
Step 1: 检查 "bad"
  b vs b → 匹配
  a vs . → 匹配
  d vs . → 匹配
  → 找到匹配,返回 True
```

### Python代码

```python
class WordDictionary:
    """
    解法一:数组暴力匹配
    思路:用列表存储所有单词,搜索时逐个比对
    """
    def __init__(self):
        self.words = []

    def addWord(self, word: str) -> None:
        """添加单词到列表"""
        self.words.append(word)

    def search(self, word: str) -> bool:
        """搜索单词,支持通配符 . """
        for stored_word in self.words:
            if len(stored_word) != len(word):
                continue  # 长度不同,直接跳过
            if self._match(stored_word, word):
                return True
        return False

    def _match(self, stored: str, pattern: str) -> bool:
        """辅助函数:检查 stored 是否匹配 pattern"""
        for i in range(len(stored)):
            if pattern[i] != '.' and pattern[i] != stored[i]:
                return False
        return True


# ✅ 测试
wd = WordDictionary()
wd.addWord("bad")
wd.addWord("dad")
wd.addWord("mad")
print(wd.search("pad"))  # 期望输出:False
print(wd.search("bad"))  # 期望输出:True
print(wd.search(".ad"))  # 期望输出:True
print(wd.search("b.."))  # 期望输出:True
```

### 复杂度分析
- **时间复杂度**:
  - addWord: O(1) — 直接追加到列表
  - search: O(n*m) — n是单词数量,m是单词长度,需要遍历所有单词并逐字符比较
  - 具体地说:如果有1000个单词,每个长度10,搜索一次最坏需要10000次字符比较
- **空间复杂度**:O(n*m) — 存储n个单词

### 优缺点
- ✅ 实现简单,易于理解
- ✅ 空间开销相对较小(相比Trie)
- ❌ 搜索效率低,无法利用公共前缀
- ❌ 无法剪枝,即使明知不匹配仍要检查

---

## 🏆 解法二:Trie前缀树 + DFS回溯(最优解)

### 优化思路
解法一的痛点在于每次搜索都要遍历所有单词。Trie前缀树可以共享公共前缀,搜索时只走有效分支。

遇到`.`通配符时,用DFS同时探索当前节点的所有子节点,找到任意一个匹配即可返回True。

> 💡 **关键想法**:
> - 普通字符:沿Trie树唯一路径前进
> - 遇到`.`:DFS探索所有26个可能的子分支
> - 递归终止:到达单词末尾且当前节点标记为单词结尾

### 图解过程

```
构建Trie树:
addWord("bad")
addWord("dad")
addWord("mad")

Trie结构:
        root
       / | \
      b  d  m
      |  |  |
      a  a  a
      |  |  |
      d* d* d*  (* 表示单词结尾)

搜索 ".ad":
Step 1: 从root开始,遇到 '.'
  → DFS探索所有子节点: b, d, m

Step 2: 对每个分支继续搜索 "ad"
  分支1: b → a → d (找到,返回True)
  分支2: d → a → d (找到,返回True)
  分支3: m → a → d (找到,返回True)

搜索 "bat":
Step 1: 从root找 'b' → 找到
Step 2: 从 b 找 'a' → 找到
Step 3: 从 a 找 't' → 不存在,返回False
```

### Python代码

```python
class TrieNode:
    """Trie树节点"""
    def __init__(self):
        self.children = {}  # 子节点字典
        self.is_end = False  # 是否为单词结尾


class WordDictionary:
    """
    解法二:Trie前缀树 + DFS回溯
    思路:用Trie存储单词,搜索时遇到.用DFS探索所有可能分支
    """
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        """添加单词到Trie树"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True  # 标记单词结尾

    def search(self, word: str) -> bool:
        """搜索单词,支持通配符 . """
        return self._dfs(word, 0, self.root)

    def _dfs(self, word: str, index: int, node: TrieNode) -> bool:
        """
        DFS回溯搜索
        word: 搜索模式
        index: 当前处理的字符位置
        node: 当前Trie节点
        """
        # 递归终止:搜索完所有字符
        if index == len(word):
            return node.is_end  # 必须是单词结尾才算匹配

        char = word[index]

        if char == '.':
            # 通配符:尝试所有可能的子节点
            for child in node.children.values():
                if self._dfs(word, index + 1, child):
                    return True  # 找到任意一个匹配即可
            return False  # 所有分支都不匹配
        else:
            # 普通字符:直接沿树前进
            if char not in node.children:
                return False
            return self._dfs(word, index + 1, node.children[char])


# ✅ 测试
wd = WordDictionary()
wd.addWord("bad")
wd.addWord("dad")
wd.addWord("mad")
print(wd.search("pad"))  # 期望输出:False
print(wd.search("bad"))  # 期望输出:True
print(wd.search(".ad"))  # 期望输出:True
print(wd.search("b.."))  # 期望输出:True
print(wd.search("..."))  # 期望输出:True (匹配所有长度为3的词)
print(wd.search("ba."))  # 期望输出:True (匹配"bad")
```

### 复杂度分析
- **时间复杂度**:
  - addWord: O(m) — m是单词长度,沿树插入每个字符
  - search:
    - 最好情况(无`.`):O(m) — 直接沿树查找
    - 最坏情况(全是`.`):O(26^m) — 每个位置探索26个分支,但实际远小于此,因为Trie剪枝了不存在的分支
    - 平均情况:O(m*k) — k是实际存在的分支数,通常远小于26
- **空间复杂度**:O(n*m) — n个单词,每个长度m,存储在Trie中

### 关键优化点
1. **Trie剪枝**:遇到不存在的字符立即返回False,无需继续
2. **DFS短路**:找到任意一个匹配即返回True,无需遍历所有分支
3. **共享前缀**:相同前缀的单词共用节点,节省空间

---

## 🐍 Pythonic 写法

利用字典的`get`方法和递归优化:

```python
class WordDictionary:
    """Pythonic写法:使用字典嵌套代替自定义节点类"""
    def __init__(self):
        self.trie = {}

    def addWord(self, word: str) -> None:
        node = self.trie
        for char in word:
            node = node.setdefault(char, {})
        node['#'] = True  # '#'标记单词结尾

    def search(self, word: str) -> bool:
        def dfs(node, i):
            if i == len(word):
                return '#' in node
            if word[i] == '.':
                return any(dfs(child, i + 1) for child in node.values() if isinstance(child, dict))
            return word[i] in node and dfs(node[word[i]], i + 1)

        return dfs(self.trie, 0)
```

**解释**:
- `setdefault(char, {})`:如果字符不存在则创建空字典,存在则返回,一行代码完成插入逻辑
- `any(...)`:找到任意一个True即短路,比显式循环更简洁
- `isinstance(child, dict)`:过滤掉`'#'`这个标记,只遍历实际的子节点

> ⚠️ **面试建议**:先写解法二的清晰版本(用TrieNode类),展示结构化思维,再提Pythonic写法展示Python功底。

---

## 📊 解法对比

| 维度 | 解法一:数组暴力 | 🏆 解法二:Trie + DFS(最优) |
|------|--------------|--------------------------|
| 时间复杂度(addWord) | O(1) | O(m) |
| 时间复杂度(search) | O(n*m) | **O(m*k)** ← 时间最优 |
| 空间复杂度 | O(n*m) | O(n*m) |
| 代码难度 | 简单 | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 仅适合少量单词 | **通用,尤其是大量单词场景** |

**为什么是最优解**:
- 时间:Trie剪枝大幅减少无效搜索,平均远优于O(n*m)
- 空间:虽然同为O(n*m),但Trie共享前缀,实际占用更少
- 扩展性:支持其他操作如前缀匹配、删除单词等

**面试建议**:
1. 先用30秒口述暴力法(数组遍历),表明理解基本思路
2. 立即优化到🏆最优解(Trie + DFS),展示数据结构运用能力
3. **重点讲解关键点**:"普通字符直接走,遇到`.`DFS探索所有分支"
4. 手动模拟一个例子,如".ad"的搜索过程,展示DFS回溯逻辑
5. 强调为什么Trie是最优:前缀共享 + 剪枝

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请设计一个数据结构,支持添加单词和搜索单词,搜索需要支持通配符`.`匹配任意字符。

**你**:(审题30秒)好的,这道题的核心是实现一个支持模糊匹配的字典。我先想一下...

我的第一个想法是用数组存储所有单词,搜索时遍历数组逐个比对,时间复杂度O(n*m)。但这个方法效率较低,没有利用单词间的公共前缀。

更优的方法是用**Trie前缀树**。添加单词时,按字符构建树结构。搜索时,如果是普通字符就沿树直接走,如果遇到`.`就用DFS同时探索所有可能的子节点。这样可以大幅减少无效搜索,时间复杂度优化到O(m*k),k是实际存在的分支数。

**面试官**:很好,请写一下代码。

**你**:(边写边说)首先定义Trie节点,包含子节点字典和单词结尾标记...然后实现addWord,沿树插入每个字符...search方法用DFS递归,遇到`.`时遍历所有子节点...

**面试官**:测试一下?

**你**:用示例输入走一遍...添加"bad","dad","mad"后,搜索".ad"时,从根节点遇到`.`,DFS探索b/d/m三个分支,都能匹配到"ad"后缀,返回True。再测试边界情况,搜索"bat",走到b→a后发现没有t子节点,返回False。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果通配符不是单字符而是*匹配任意长度呢?" | "需要改用正则表达式DP或更复杂的递归,对每个*尝试匹配0到多个字符,复杂度会上升" |
| "能不能优化空间?" | "可以按长度分组存储,搜索时只在对应长度的Trie中查找,但会增加代码复杂度" |
| "如果需要删除单词怎么办?" | "递归删除时检查子节点数量,如果删除后某节点无子节点且不是其他单词结尾,则删除该节点" |
| "Trie的空间开销会不会太大?" | "确实,最坏情况每个单词都不共享前缀时空间较大,可以用HashMap压缩或改用哈希表按长度分组" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:setdefault简化字典插入 — 一行代码完成"不存在则创建,存在则返回"
node = node.setdefault(char, {})

# 技巧2:any短路求值 — 找到第一个True即停止
return any(dfs(child, i+1) for child in node.values() if isinstance(child, dict))

# 技巧3:递归函数内嵌 — 访问外部变量无需传参
def search(self, word):
    def dfs(node, i):  # 可以直接访问外部的word
        if i == len(word):
            return '#' in node
        ...
    return dfs(self.trie, 0)
```

### 💡 底层原理(选读)

> **为什么Trie能高效处理字符串前缀?**
>
> Trie的核心是**路径压缩**和**公共前缀共享**:
> - 如果100个单词都以"app"开头,Trie只存储一次"a→p→p"这条路径,后续单词从这个节点分叉
> - 搜索时,前缀匹配O(m)时间,而哈希表需要存储完整字符串才能判断
> - DFS回溯利用递归栈,空间复杂度O(m),而非O(n*m)
>
> **`.`通配符为什么用DFS而不是BFS?**
> - DFS可以短路:找到第一个匹配立即返回,无需遍历所有可能
> - BFS需要队列存储所有可能的节点,空间开销更大
> - DFS递归代码更简洁,易于维护

### 算法模式卡片 📐
- **模式名称**:Trie + DFS回溯
- **适用条件**:字符串集合需要支持模糊匹配或前缀查询
- **识别关键词**:"添加单词"、"搜索单词"、"通配符匹配"、"前缀匹配"
- **模板代码**:
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def search_with_wildcard(self, word):
        def dfs(node, i):
            if i == len(word):
                return node.is_end
            if word[i] == '.':
                # 通配符:探索所有子节点
                for child in node.children.values():
                    if dfs(child, i + 1):
                        return True
                return False
            # 普通字符:直接查找
            if word[i] not in node.children:
                return False
            return dfs(node.children[word[i]], i + 1)
        return dfs(self.root, 0)
```

### 易错点 ⚠️
1. **忘记检查单词结尾标记**:到达单词末尾时,必须检查`node.is_end`,否则"ba"会错误匹配"bad"的前缀
   - 错误:`if index == len(word): return True`
   - 正确:`if index == len(word): return node.is_end`

2. **DFS中遗漏普通字符的存在性检查**:直接访问`node.children[char]`会导致KeyError
   - 正确做法:先检查`if char not in node.children: return False`

3. **通配符DFS中返回逻辑错误**:需要找到任意一个匹配即返回True,而非等待所有分支都检查完
   - 错误:
   ```python
   for child in node.children.values():
       result = dfs(child, i+1)
   return result  # 只返回最后一个分支的结果
   ```
   - 正确:
   ```python
   for child in node.children.values():
       if dfs(child, i+1):
           return True  # 立即返回
   return False
   ```

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:搜索引擎自动补全**:用户输入"pyth",Trie快速返回"python","pytorch"等候选词,支持模糊匹配拼写错误
- **场景2:敏感词过滤系统**:将敏感词存入Trie,检测文本时支持通配符匹配变体,如"f**k"匹配多种拼写
- **场景3:路由匹配**:Web框架用Trie存储路由规则,支持"/user/:id"这种动态路由匹配
- **场景4:代码编辑器智能提示**:IDE用Trie存储API函数名,输入前几个字母快速过滤候选列表

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 208. 实现Trie前缀树 | Medium | Trie基础 | 本题的前置题,不含通配符 |
| LeetCode 212. 单词搜索II | Hard | Trie + 网格DFS | Trie + 二维网格回溯的综合应用 |
| LeetCode 745. 前缀和后缀搜索 | Hard | Trie变体 | 需要同时匹配前缀和后缀 |
| LeetCode 676. 实现魔法字典 | Medium | Trie + DFS | 允许最多一个字符不同 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:在原题基础上,新增一个操作`searchPrefix(prefix)`,返回所有以prefix为前缀的单词列表。例如添加了"bad","bat","dad"后,`searchPrefix("ba")`返回`["bad","bat"]`。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

先用原方法找到prefix对应的Trie节点,然后从该节点开始DFS遍历所有子树,收集所有标记为单词结尾的路径。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def searchPrefix(self, prefix: str) -> list[str]:
        """返回所有以prefix为前缀的单词"""
        # 1. 找到prefix对应的节点
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # 没有单词以此为前缀
            node = node.children[char]

        # 2. 从该节点DFS收集所有单词
        result = []
        def dfs(node, path):
            if node.is_end:
                result.append(prefix + path)
            for char, child in node.children.items():
                dfs(child, path + char)

        dfs(node, "")
        return result

# 测试
wd = WordDictionary()
wd.addWord("bad")
wd.addWord("bat")
wd.addWord("dad")
print(wd.searchPrefix("ba"))  # 输出:["bad", "bat"]
print(wd.searchPrefix("d"))   # 输出:["dad"]
```

**核心思路**:先定位到前缀节点,再从该节点DFS遍历所有后代,收集标记为单词结尾的完整路径。时间复杂度O(m+k*l),m是前缀长度,k是匹配单词数量,l是平均单词长度。

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
