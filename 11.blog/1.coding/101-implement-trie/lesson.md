> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第101课:实现Trie前缀树

> **模块**:前缀树 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/implement-trie-prefix-tree/
> **前置知识**:字典、递归、树的基本概念
> **预计学习时间**:25分钟

---

## 🎯 题目描述

实现一个Trie(前缀树/字典树),支持以下操作:
- `insert(word)`:插入字符串word到Trie中
- `search(word)`:如果字符串word在Trie中,返回True;否则返回False
- `startsWith(prefix)`:如果之前插入过字符串以prefix为前缀,返回True;否则返回False

**示例:**
```
输入:
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
输出:
[null, null, true, false, true, null, true]

解释:
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // 返回 True
trie.search("app");     // 返回 False
trie.startsWith("app"); // 返回 True
trie.insert("app");
trie.search("app");     // 返回 True
```

**约束条件:**
- 1 <= word.length, prefix.length <= 2000
- word 和 prefix 仅由小写英文字母组成
- 最多调用3 * 10^4次 insert、search 和 startsWith

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单字符 | insert("a"), search("a") | True | 基本功能 |
| 前缀查询 | insert("apple"), startsWith("app") | True | 前缀匹配 |
| 完整词vs前缀 | insert("apple"), search("app") | False | 区分完整词和前缀 |
| 空树查询 | search("x")在空Trie | False | 空树处理 |
| 大规模 | 插入30000个单词 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在查英文字典。
>
> 🐌 **笨办法**:把所有单词存在列表里,每次查找都遍历整个列表,时间O(n*m)(n个单词,每个长m)。
>
> 🚀 **聪明办法**:像真实字典一样按首字母分类!a开头的单词在一起,b开头的在一起...
>
> 更进一步:在a开头的单词中,再按第二个字母分类(aa、ab、ac...)。形成一棵树:
> ```
>        root
>       /  |  \
>      a   b   c
>     / \
>    p   s
>   / \
>  p   t
> ```
> 这样查找"app"时,只需沿着 root→a→p→p 路径,时间O(m)!

### 关键洞察
**Trie树将共享前缀的单词合并成一条路径,查找时间只取决于单词长度,与已存单词数量无关!**

---

## 🧠 解题思维链

### Step 1:理解题目 → 锁定输入输出
- **输入**:单词(字符串)、前缀(字符串)
- **输出**:布尔值(是否存在)
- **关键区别**:search要求完整单词,startsWith只要求前缀

### Step 2:先想笨办法(用列表存储)
可以用集合存储所有单词,search时直接查集合,startsWith时遍历集合检查前缀。
- 时间复杂度:search O(m),startsWith O(n*m)(n为单词数)
- 瓶颈在哪:startsWith需要遍历所有单词

### Step 3:瓶颈分析 → 优化方向
集合无法高效处理前缀查询。能不能利用"共享前缀"这一特性?
- 核心问题:如何表示和查找前缀?
- 优化思路:用树结构,共享前缀的单词共享路径

### Step 4:选择武器
- 选用:**Trie树(前缀树)**
- 理由:
  - 查找时间O(m),只与单词长度有关
  - 天然支持前缀查询
  - 空间换时间,共享前缀节省空间

> 🔑 **模式识别提示**:当题目涉及"前缀匹配"、"自动补全"、"字典查询",优先考虑"Trie树"

---

## 🔑 解法一:字典嵌套实现(Pythonic)

### 思路
用嵌套字典表示Trie树:每个节点是一个字典,键是字符,值是子节点(也是字典)。用特殊键`'#'`标记单词结束。

### 图解过程

```
插入"apple"和"app":

初始:root = {}

插入"apple":
root → {'a': {'p': {'p': {'l': {'e': {'#': True}}}}}}

可视化:
    root
     |
     a
     |
     p
     |
     p
     |
     l
     |
     e
     |
     # (结束标记)

再插入"app":
root → {'a': {'p': {'p': {'#': True, 'l': {'e': {'#': True}}}}}}

可视化:
    root
     |
     a
     |
     p
     |
     p ← 这里标记#(app结束)
     |
     l
     |
     e
     |
     # (apple结束)

查找"app":
  root → 'a' → 'p' → 'p' → 检查是否有'#' → True

查找"ap":
  root → 'a' → 'p' → 检查是否有'#' → False (只是前缀,不是完整词)

前缀查询"app":
  root → 'a' → 'p' → 'p' → 能走到 → True
```

### Python代码

```python
class Trie:
    """
    解法一:字典嵌套实现
    思路:用嵌套字典表示树,#标记单词结束
    """
    def __init__(self):
        self.root = {}

    def insert(self, word: str) -> None:
        """插入单词"""
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True  # 标记单词结束

    def search(self, word: str) -> bool:
        """查找完整单词"""
        node = self._find_prefix(word)
        return node is not None and '#' in node

    def startsWith(self, prefix: str) -> bool:
        """查找前缀"""
        return self._find_prefix(prefix) is not None

    def _find_prefix(self, prefix: str):
        """辅助方法:找到前缀对应的节点"""
        node = self.root
        for char in prefix:
            if char not in node:
                return None
            node = node[char]
        return node


# ✅ 测试
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))   # True
print(trie.search("app"))     # False
print(trie.startsWith("app")) # True
trie.insert("app")
print(trie.search("app"))     # True
```

### 复杂度分析
- **时间复杂度**:
  - insert:O(m) — m为单词长度,需要遍历每个字符
  - search:O(m) — 最多遍历m个字符
  - startsWith:O(m) — 同上
- **空间复杂度**:O(ALPHABET_SIZE * N * M) — N个单词,平均长度M,字母表大小26
  - 具体地说:如果有1000个单词,平均长度10,最坏情况需要26*1000*10=260K节点
  - 但实际中,共享前缀大幅减少空间

### 优缺点
- ✅ 代码简洁,Pythonic风格
- ✅ 不需要定义额外类
- ❌ 性能略低于节点类实现

---

## 🏆 解法二:节点类实现(最优解)

### 优化思路
用显式的TrieNode类表示节点,每个节点包含:
1. children:字典,存储子节点
2. is_end:布尔值,标记是否是单词结束

这样结构更清晰,性能更好。

> 💡 **关键想法**:显式节点类让代码更易维护和扩展,也更符合OOP设计

### 图解过程

```
节点结构:
TrieNode {
    children: {'a': TrieNode, 'b': TrieNode, ...}
    is_end: False/True
}

插入"app"和"apple":

      root(is_end=False)
       |
      'a' → TrieNode(is_end=False)
       |
      'p' → TrieNode(is_end=False)
       |
      'p' → TrieNode(is_end=True) ← app结束
       |
      'l' → TrieNode(is_end=False)
       |
      'e' → TrieNode(is_end=True) ← apple结束

查找"app":
  root → 'a' → 'p' → 'p' → node.is_end=True → 返回True

查找"appl":
  root → 'a' → 'p' → 'p' → 'l' → node.is_end=False → 返回False

前缀查询"ap":
  root → 'a' → 'p' → 能走到节点 → 返回True
```

### Python代码

```python
class TrieNode:
    """Trie树节点"""
    def __init__(self):
        self.children = {}  # 子节点字典
        self.is_end = False  # 是否是单词结束


class Trie:
    """
    解法二:节点类实现(最优解)
    思路:显式TrieNode类,结构清晰,性能更好
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """插入单词 - O(m)时间"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        """查找完整单词 - O(m)时间"""
        node = self._find_node(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        """查找前缀 - O(m)时间"""
        return self._find_node(prefix) is not None

    def _find_node(self, prefix: str):
        """辅助方法:找到前缀对应的节点"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node


# ✅ 测试
trie = Trie()
trie.insert("apple")
print(trie.search("apple"))   # True
print(trie.search("app"))     # False
print(trie.startsWith("app")) # True
trie.insert("app")
print(trie.search("app"))     # True
print(trie.search("application"))  # False
```

### 复杂度分析
- **时间复杂度**:
  - insert、search、startsWith:O(m) — m为单词/前缀长度
- **空间复杂度**:O(ALPHABET_SIZE * N * M) — 同解法一,但实际更优(共享前缀)

### 为什么是最优解
- ✅ 时间O(m)已经是理论最优(必须遍历所有字符)
- ✅ 结构清晰,易于扩展(如添加删除操作)
- ✅ 面向对象设计,符合工程实践
- ✅ 代码可读性高,面试中容易讲清楚

---

## 🐍 Pythonic 写法

利用 defaultdict 简化:

```python
from collections import defaultdict

class TrieNodeSimple:
    def __init__(self):
        self.children = defaultdict(TrieNodeSimple)
        self.is_end = False

class TriePythonic:
    def __init__(self):
        self.root = TrieNodeSimple()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            node = node.children[char]  # defaultdict自动创建
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

> ⚠️ **面试建议**:先写清晰的节点类版本,再提 defaultdict 优化。
> 面试官更看重你的**数据结构设计能力**,而非Python技巧。

---

## 📊 解法对比

| 维度 | 解法一:字典嵌套 | 🏆 解法二:节点类(最优) |
|------|---------------|---------------------|
| 时间复杂度 | O(m) | **O(m)** ← 时间最优 |
| 空间复杂度 | O(ALPHABET * N * M) | **O(ALPHABET * N * M)** |
| 代码难度 | 简单 | 简单 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 扩展性 | 较差 | **优秀**(易添加删除等操作) |
| 适用场景 | 快速实现 | **工程实践、面试首选** |

**为什么节点类是最优解**:
- 时间O(m)已经是理论最优
- 结构清晰,符合OOP设计原则
- 易于扩展新功能(如删除、统计前缀数量等)

**面试建议**:
1. 先讲解Trie的核心思想:"共享前缀的树结构"
2. 手绘示意图,展示插入和查找过程
3. 写🏆最优解:节点类实现
4. 强调为什么用is_end区分完整词和前缀
5. 测试边界用例:空字符串、单字符、重复插入

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你实现一个Trie前缀树。

**你**:(审题30秒)好的,Trie是一种树形数据结构,专门用于字符串前缀匹配。让我先想一下数据结构...

我会用**TrieNode节点类**来实现,每个节点包含:
1. children字典:存储子节点,键是字符
2. is_end布尔值:标记是否是单词结束

核心思路是:**共享前缀的单词共享路径**。比如"app"和"apple",它们共享前三个字符的路径。

**面试官**:很好,请实现insert、search和startsWith方法。

**你**:(边写边说)
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        # 遍历每个字符,创建路径
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        # 标记单词结束
        node.is_end = True

    def search(self, word):
        node = self._find_node(word)
        # 必须找到节点且标记为单词结束
        return node is not None and node.is_end

    def startsWith(self, prefix):
        # 只需找到节点即可
        return self._find_node(prefix) is not None

    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

**面试官**:search和startsWith有什么区别?

**你**:关键区别在于is_end标记。search要求完整单词,所以需要检查`node.is_end=True`;而startsWith只要求前缀存在,只需能找到对应节点即可。

比如插入"apple"后:
- search("app")返回False(app不是完整单词)
- startsWith("app")返回True(app是apple的前缀)

**面试官**:时间复杂度?

**你**:三个操作都是O(m),m是单词/前缀长度。因为只需遍历一次字符即可。空间复杂度O(ALPHABET_SIZE * N * M),但实际中共享前缀大幅降低空间。

**面试官**:测试一下?

**你**:
```python
trie = Trie()
trie.insert("apple")
# 画图展示树结构: root→a→p→p→l→e(is_end=True)
print(trie.search("apple"))   # True
print(trie.search("app"))     # False(走到p节点,is_end=False)
print(trie.startsWith("app")) # True(能走到p节点)
```

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如何实现删除操作?" | "递归删除:从叶子向根回溯,删除不再需要的节点。需要判断节点是否还有其他子节点或被其他单词使用" |
| "如何统计以某前缀开头的单词数量?" | "在节点中增加count字段,insert时沿路径所有节点count+1,查询时返回前缀节点的count" |
| "空间能优化吗?" | "可以用数组代替字典(children = [None] * 26),空间更紧凑但只支持小写字母" |
| "实际应用场景?" | "搜索引擎自动补全、拼写检查、IP路由表、字符串匹配引擎等" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:字典的get方法 — 避免KeyError
node.children.get('a', None)  # 不存在返回None

# 技巧2:defaultdict自动创建 — 简化代码
from collections import defaultdict
children = defaultdict(TrieNode)
children['a']  # 自动创建TrieNode()

# 技巧3:in操作符检查键 — O(1)时间
if 'a' in node.children:  # 高效检查
```

### 💡 底层原理(选读)

> **为什么Trie适合前缀查询?**
>
> Trie的本质是**状态机**,每个节点代表一个状态(已匹配的前缀),每条边代表一个转移(匹配下一个字符)。前缀查询就是状态转移,时间只与路径长度(单词长度)有关,与已存单词数无关。
>
> **Trie vs 哈希表**:
> - 哈希表:O(1)精确查找,但不支持前缀查询(需遍历所有键)
> - Trie:O(m)前缀查询,且能获取所有匹配单词
> - 工程选择:小数据量用哈希表,大量前缀查询用Trie

### 算法模式卡片 📐
- **模式名称**:Trie树(前缀树)
- **适用条件**:字符串前缀匹配、自动补全、字典查询
- **识别关键词**:"前缀"、"自动补全"、"词典"、"字符串匹配"
- **模板代码**:
```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word):
        node = self._find(word)
        return node and node.is_end

    def _find(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

### 易错点 ⚠️
1. **忘记is_end标记**:导致search和startsWith返回相同结果,无法区分完整词和前缀
2. **重复插入同一单词**:应该幂等,不影响结果。注意不要重复创建节点
3. **空字符串处理**:空字符串是所有单词的前缀,startsWith("")应返回True
4. **节点引用错误**:循环中node = node.children[char],容易写成node = self.root

---

## 🏗️ 工程实战(选读)

> 这个数据结构在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:搜索引擎自动补全。用户输入"pyth",Trie快速找到所有"pyth"开头的搜索词(python、pythagorean...)
- **场景2**:IP路由表。路由器用Trie存储IP前缀,O(32)时间查找最长匹配前缀
- **场景3**:拼写检查。将字典存入Trie,输入单词后快速判断是否拼写正确,或提供相近建议
- **场景4**:字符串匹配引擎。AC自动机(Aho-Corasick)基于Trie实现,用于多模式串匹配

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 211. 添加与搜索单词 | Medium | Trie+DFS | 支持通配符'.',需要用DFS回溯 |
| LeetCode 212. 单词搜索II | Hard | Trie+DFS回溯 | 网格中搜索多个单词,Trie剪枝 |
| LeetCode 648. 单词替换 | Medium | Trie前缀匹配 | 找最短前缀替换单词 |
| LeetCode 1804. 实现Trie(前缀树)II | Medium | Trie扩展 | 增加计数功能(统计前缀数量) |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:在Trie类中增加一个方法`countWordsStartingWith(prefix)`,返回以prefix开头的单词数量。比如插入["apple", "app", "apricot"]后,countWordsStartingWith("app")返回2。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

在TrieNode中增加count字段,insert时沿路径每个节点count+1,查询时返回前缀节点的count。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0  # 新增:经过该节点的单词数

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1  # 沿路径增加计数
        node.is_end = True

    def countWordsStartingWith(self, prefix: str) -> int:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count  # 返回该前缀节点的计数

# 测试
trie = Trie()
trie.insert("apple")
trie.insert("app")
trie.insert("apricot")
print(trie.countWordsStartingWith("app"))  # 2 (apple, app)
print(trie.countWordsStartingWith("apr"))  # 1 (apricot)
```

核心思路:insert时沿路径每个节点count+1,这样每个节点的count就表示"经过该节点的单词数",也就是"以该节点代表的前缀开头的单词数"。

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
