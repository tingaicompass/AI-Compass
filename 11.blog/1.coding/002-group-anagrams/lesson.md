# 📖 第2课：字母异位词分组

> **模块**：哈希表进阶 | **难度**：Medium ⭐⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/group-anagrams/
> **前置知识**：第01课 两数之和（哈希表基础）
> **预计学习时间**：25分钟

---

## 🎯 题目描述

给你一个字符串数组，请你把所有**字母异位词**归为一组。字母异位词是指：两个字符串包含的字母种类和数量完全相同，只是顺序可能不同。

比如 `"eat"` 和 `"tea"` 是异位词，因为它们都由 e、a、t 三个字母组成。你需要把这些"长得像"的字符串放到同一个组里。

**示例：**
```
输入：strs = ["eat","tea","tan","ate","nat","bat"]
输出：[["bat"],["nat","tan"],["ate","eat","tea"]]
解释：
  - "eat", "tea", "ate" 是一组（都由 e、a、t 组成）
  - "tan", "nat" 是一组（都由 t、a、n 组成）
  - "bat" 单独一组
```

**约束条件：**
- `1 <= strs.length <= 10^4`（最多1万个字符串）
- `0 <= strs[i].length <= 100`（每个字符串最长100字符）
- `strs[i]` 只包含小写英文字母

---

## 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | `["a"]` | `[["a"]]` | 单个字符串 |
| 空字符串 | `[""]` | `[[""]]` | 空串处理 |
| 无异位词 | `["abc","def","ghi"]` | `[["abc"],["def"],["ghi"]]` | 每个单独一组 |
| 全是异位词 | `["abc","bca","cab"]` | `[["abc","bca","cab"]]` | 全在一组 |
| 大规模 | n=10000, 每个长度100 | — | 性能考察 O(n) |
| 相同字符串 | `["a","a","a"]` | `[["a","a","a"]]` | 重复字符串 |

---

## 💡 思路引导

### 生活化比喻

> 想象你是图书管理员，收到一堆打乱字母的单词卡片，需要把"拼写相同但顺序不同"的卡片放在一起。
>
> 🐌 **笨办法**：拿每张卡片和其他所有卡片逐个比较，看看字母是否完全一样——这需要 n² 次比较，而且每次比较还要数字母，太慢了！
>
> 🚀 **聪明办法**：先给每张卡片制作一个"身份证"（把字母按字母表顺序排序，比如 "tea" → "aet"），然后按身份证分类——只要身份证相同的，就是异位词！就像图书馆用书号分类一样，一眼就能找到同类。
>
> 💡 这个"身份证"就是**哈希表的 key**，同一个 key 对应的所有字符串就是一组！

### 关键洞察

**核心突破：将字符串转换成"标准化形式"作为 key，相同 key 的归为一组**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出

- **输入**：字符串数组 `strs`，每个字符串只包含小写字母
- **输出**：二维列表，每个子列表包含一组异位词
- **限制**：需要判断"字母种类和数量完全相同"

### Step 2：先想笨办法（暴力法）

最直接的思路：对每个字符串，遍历其他所有字符串，逐个比较字母是否相同。

- 时间复杂度：O(n² × k)，n 是字符串数量，k 是字符串平均长度
- 瓶颈在哪：**n² 次字符串比较**，而且每次比较还要统计字母频率

### Step 3：瓶颈分析 → 优化方向

暴力法的问题：
- 核心问题：**重复比较** —— 每次都要遍历所有字符串看谁是异位词
- 优化思路：能不能 **O(1) 判断两个字符串是否是异位词**？

💡 关键发现：异位词有个特点——**排序后长得一样**！
- `"eat"` 排序后 → `"aet"`
- `"tea"` 排序后 → `"aet"`
- `"tan"` 排序后 → `"ant"`

所以可以用 **排序后的字符串作为哈希表的 key**！

### Step 4：确定最优解 → 哈希表分组

- **方案**：用哈希表，key 是排序后的字符串，value 是原字符串列表
- **流程**：
  1. 遍历每个字符串
  2. 将其排序得到 key
  3. 把原字符串加入 `hash_map[key]`
  4. 返回 `hash_map.values()`

- **时间复杂度**：O(n × k log k)，n 个字符串，每个排序需要 O(k log k)
- **空间复杂度**：O(n × k)，存储所有字符串

---

## 🔑 解法一：排序作为 Key（标准解法）

### 核心思路

把每个字符串**排序**后作为哈希表的 key，原字符串作为 value 添加到对应的列表中。

### 图解演示

```
输入: ["eat","tea","tan","ate","nat","bat"]

Step 1: 遍历 "eat"
  排序后: "aet"
  hash_map = {"aet": ["eat"]}

Step 2: 遍历 "tea"
  排序后: "aet"（和 "eat" 一样！）
  hash_map = {"aet": ["eat", "tea"]}

Step 3: 遍历 "tan"
  排序后: "ant"
  hash_map = {"aet": ["eat", "tea"], "ant": ["tan"]}

Step 4: 遍历 "ate"
  排序后: "aet"
  hash_map = {"aet": ["eat", "tea", "ate"], "ant": ["tan"]}

Step 5: 遍历 "nat"
  排序后: "ant"
  hash_map = {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"]}

Step 6: 遍历 "bat"
  排序后: "abt"
  hash_map = {"aet": ["eat", "tea", "ate"], "ant": ["tan", "nat"], "abt": ["bat"]}

最终输出: [["eat","tea","ate"], ["tan","nat"], ["bat"]]
```

### 完整代码

```python
def groupAnagrams(strs):
    """
    使用排序后的字符串作为哈希表的 key

    :param strs: List[str] - 输入字符串数组
    :return: List[List[str]] - 分组后的异位词
    """
    hash_map = {}

    for s in strs:
        # 将字符串排序作为 key
        key = ''.join(sorted(s))

        # 如果 key 不存在，初始化为空列表
        if key not in hash_map:
            hash_map[key] = []

        # 将原字符串加入对应的组
        hash_map[key].append(s)

    # 返回所有分组（hash_map 的所有 value）
    return list(hash_map.values())


# 测试用例
if __name__ == "__main__":
    # 测试1：标准用例
    strs1 = ["eat","tea","tan","ate","nat","bat"]
    print(groupAnagrams(strs1))
    # 输出: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]

    # 测试2：空字符串
    strs2 = [""]
    print(groupAnagrams(strs2))
    # 输出: [['']]

    # 测试3：单个字符
    strs3 = ["a"]
    print(groupAnagrams(strs3))
    # 输出: [['a']]
```

### 复杂度分析

- **时间复杂度**：**O(n × k log k)**
  - n 是字符串数量（比如 10000 个）
  - k 是字符串平均长度（比如 100）
  - 每个字符串排序需要 O(k log k) ≈ 100 × log(100) ≈ 664
  - 总共 10000 × 664 ≈ 660万 次操作

- **空间复杂度**：**O(n × k)**
  - 哈希表存储所有字符串：10000 × 100 = 100万 字符

---

## ⚡ 解法二：字符计数作为 Key（终极优化）

### 核心思路

不用排序，而是**统计每个字母出现的次数**，用计数结果作为 key。这样可以把时间复杂度从 O(k log k) 降到 O(k)。

比如 `"eat"` 的字符计数可以表示为 `(1,0,0,0,1,0,...,0,1,0,...)`（a出现1次，e出现1次，t出现1次）

### 图解演示

```
输入: ["eat","tea","tan"]

"eat" → 字符计数: a=1, e=1, t=1 → key = "#1#0#0#0#1#0...#1#0..."
"tea" → 字符计数: a=1, e=1, t=1 → key = "#1#0#0#0#1#0...#1#0..." (相同!)
"tan" → 字符计数: a=1, n=1, t=1 → key = "#1#0#0#0#0...#1...#1#0..."

hash_map = {
  "#1#0#0#0#1#0...#1#0...": ["eat", "tea"],
  "#1#0#0#0#0...#1...#1#0...": ["tan"]
}
```

### 完整代码

```python
from collections import defaultdict

def groupAnagrams(strs):
    """
    使用字符计数作为 key（更快）

    :param strs: List[str] - 输入字符串数组
    :return: List[List[str]] - 分组后的异位词
    """
    hash_map = defaultdict(list)

    for s in strs:
        # 统计每个字母出现次数（26个字母）
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1

        # 将计数数组转换为元组作为 key（列表不能做 key）
        key = tuple(count)

        # 直接添加，defaultdict 会自动初始化
        hash_map[key].append(s)

    return list(hash_map.values())


# 测试用例
if __name__ == "__main__":
    strs = ["eat","tea","tan","ate","nat","bat"]
    print(groupAnagrams(strs))
    # 输出: [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

### 复杂度分析

- **时间复杂度**：**O(n × k)**
  - n 是字符串数量（10000）
  - k 是字符串平均长度（100）
  - 每个字符串只需遍历一遍统计字母：10000 × 100 = 100万 次操作
  - **比解法一快了 6 倍！**（不需要排序）

- **空间复杂度**：**O(n × k)**
  - 哈希表存储：100万 字符
  - 每个 key 是固定 26 个整数的元组

---

## 🐍 Pythonic 写法

利用 Python 的 `defaultdict` 和字符串处理技巧，代码可以更简洁：

```python
from collections import defaultdict

def groupAnagrams(strs):
    """一行核心逻辑的 Pythonic 写法"""
    groups = defaultdict(list)

    for s in strs:
        # 直接用 sorted(s) 作为 key，Python 会自动转为元组
        groups[tuple(sorted(s))].append(s)

    return list(groups.values())


# 🔥 终极简化版（一行解决）
def groupAnagrams(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    [groups[tuple(sorted(s))].append(s) for s in strs]
    return list(groups.values())
```

**Python 技巧解析**：
- `defaultdict(list)`：自动初始化，不用判断 key 是否存在
- `tuple(sorted(s))`：排序后的字符列表转元组（可哈希）
- 列表推导式：用 `[...]` 替代 for 循环（但不推荐在生产代码中过度使用）

---

## 📊 解法对比表

| 维度 | 解法一：排序作 Key | 解法二：计数作 Key | Pythonic 写法 |
|-----|------------------|------------------|-------------|
| **时间复杂度** | O(n × k log k) | **O(n × k)** ⭐ | O(n × k log k) |
| **空间复杂度** | O(n × k) | O(n × k) | O(n × k) |
| **代码简洁度** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **运行速度** | 中等（排序有开销） | **最快** ⭐ | 中等 |
| **适用场景** | 通用，面试推荐 | 字母表固定（26字母）| 快速原型 |
| **推荐指数** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**选择建议**：
- **面试白板**：用解法一（排序），清晰易懂，面试官最爱
- **工程优化**：用解法二（计数），性能最优
- **快速刷题**：用 Pythonic 写法，代码最短

---

## 🎤 面试现场模拟

**面试官**：请实现字母异位词分组。

**你**：好的，我先理解一下题目。字母异位词就是字母种类和数量相同但顺序不同的字符串，对吗？

**面试官**：没错。

**你**：那我首先想到的是哈希表。关键是找一个合适的 key 把异位词映射到同一个组。我的想法是，**把字符串排序后作为 key**，因为异位词排序后都一样。

**面试官**：很好！时间复杂度是多少？

**你**：排序每个字符串需要 O(k log k)，一共 n 个字符串，所以是 **O(n × k log k)**。

**面试官**：能优化吗？

**你**：可以！我可以用**字符计数**代替排序。统计每个字母出现次数，用计数数组作为 key，这样只需要 O(k) 时间，总复杂度降到 **O(n × k)**。

**面试官**：非常好！用 Python 怎么实现字符计数作为 key？

**你**：用一个长度为 26 的列表统计字母频率，然后转成 `tuple` 作为 key（因为列表不能做 dict 的 key）。

（开始写代码...）

---

## ❓ 高频追问表

| 追问 | 参考回答 |
|-----|---------|
| 为什么要排序/计数？能直接比较两个字符串吗？ | 直接比较无法判断异位词。排序/计数是"标准化"字符串的方式，让异位词有相同的表示。 |
| 用 sorted(s) 还是 ''.join(sorted(s)) 作为 key？ | **必须用 ''.join(sorted(s))**，因为 sorted() 返回列表（不可哈希），字符串才能做 dict 的 key。 |
| 为什么计数要用 tuple 而不是 list？ | **Python 的 dict key 必须是不可变类型**，list 是可变的，tuple 是不可变的。 |
| defaultdict 和普通 dict 的区别？ | `defaultdict(list)` 会在访问不存在的 key 时自动初始化为空列表，避免判断 `if key not in dict`。 |
| 如果字符串很长（如1万个字符），哪个解法更快？ | 计数法更快！排序 O(k log k) = 10000 × log(10000) ≈ 13万，计数 O(k) = 10000，快13倍。 |
| 能用 Counter 吗？ | 可以：`from collections import Counter; key = tuple(sorted(Counter(s).items()))`，但更慢。 |

---

## 🐍 Python 技巧卡片

### 1. defaultdict 自动初始化

```python
from collections import defaultdict

# 普通 dict 需要判断
hash_map = {}
if key not in hash_map:
    hash_map[key] = []
hash_map[key].append(value)

# defaultdict 自动处理
hash_map = defaultdict(list)
hash_map[key].append(value)  # key 不存在时自动创建空列表
```

### 2. sorted() 返回列表，需转字符串

```python
s = "eat"
sorted(s)          # ['a', 'e', 't'] - 列表，不能做 dict key
''.join(sorted(s)) # "aet" - 字符串，可以做 key ✓
tuple(sorted(s))   # ('a', 'e', 't') - 元组，也可以做 key ✓
```

### 3. ord() 计算字母偏移

```python
ord('a')  # 97
ord('b')  # 98
ord('z')  # 122

# 计算字母索引（a=0, b=1, ..., z=25）
index = ord('e') - ord('a')  # 4
```

### 4. 列表推导式创建计数数组

```python
# 初始化26个0
count = [0] * 26

# 统计字符频率
for char in "eat":
    count[ord(char) - ord('a')] += 1
# count = [1, 0, 0, 0, 1, ..., 1, ...]
#          a  b  c  d  e       t
```

---

## 🔬 底层原理说明

### Python 哈希表的实现

1. **哈希函数**：Python 用 `hash()` 函数计算 key 的哈希值
   ```python
   hash("aet")  # 根据字符串内容计算一个整数
   hash(tuple([1,0,0,...]))  # 元组也可以哈希
   ```

2. **冲突处理**：Python 使用**开放寻址法**（不是链表法）
   - 如果两个 key 的哈希值冲突，会找下一个空槽位
   - 这就是为什么 dict 的查找是平均 O(1) 而不是严格 O(1)

3. **为什么列表不能做 key？**
   - key 必须是**不可变对象**（immutable）
   - 列表可以修改，如果修改后哈希值变了，dict 就乱了
   - 字符串和元组是不可变的，可以做 key

### sorted() 的排序算法

Python 的 `sorted()` 使用 **Timsort** 算法：
- 最坏时间复杂度：O(n log n)
- 最好时间复杂度：O(n)（已排序情况）
- 空间复杂度：O(n)

---

## 📋 算法模式卡片

### 模式名称：哈希表分组（Group by Hash Key）

**适用场景**：
- 需要把相似/等价的元素归为一组
- 有办法为每个元素生成"特征码"（hash key）

**核心步骤**：
1. 为每个元素生成特征码（key）
2. 用哈希表 `{key: [元素列表]}` 存储
3. 返回 `hash_map.values()`

**关键代码模板**：
```python
from collections import defaultdict

def group_by_key(items):
    groups = defaultdict(list)
    for item in items:
        key = compute_key(item)  # 核心：如何生成 key
        groups[key].append(item)
    return list(groups.values())
```

**类似题目**：
- LC 49：字母异位词分组（本题）
- LC 1：两数之和（用值作 key 查找配对）
- LC 560：和为 K 的子数组（用前缀和作 key）

---

## ⚠️ 易错点

### 1. ❌ 用 sorted(s) 直接作为 key

```python
# 错误：sorted() 返回列表，不能做 dict key
hash_map[sorted(s)] = ...  # TypeError: unhashable type: 'list'

# 正确：转为字符串或元组
hash_map[''.join(sorted(s))] = ...  # ✓
hash_map[tuple(sorted(s))] = ...    # ✓
```

### 2. ❌ 忘记处理空字符串

```python
# 空字符串也是有效输入
strs = [""]
# sorted("") = []
# ''.join([]) = ""  ✓ 可以作为 key
```

### 3. ❌ 用 list 作为计数 key

```python
# 错误：list 不可哈希
count = [1, 0, 0, ...]
hash_map[count] = ...  # TypeError

# 正确：转为 tuple
hash_map[tuple(count)] = ...  # ✓
```

### 4. ❌ 直接返回 hash_map（而不是 values）

```python
# 错误：返回的是 dict，不是列表
return hash_map

# 正确：返回所有分组
return list(hash_map.values())  # ✓
```

---

## 🏗️ 工程实战（选读）

### 场景1：搜索引擎中的查询纠错

**问题**：用户输入 "teh" 时，推荐 "the"（常见拼写错误）

**方案**：
- 预处理词库：把所有单词按**字母组成**分组
- 用户输入时：查找同组的高频词推荐

```python
# 词库预处理
word_groups = defaultdict(list)
for word in dictionary:
    key = ''.join(sorted(word))
    word_groups[key].append((word, frequency))

# 查询纠错
def suggest(input_word):
    key = ''.join(sorted(input_word))
    candidates = word_groups[key]
    return sorted(candidates, key=lambda x: x[1], reverse=True)[:3]
```

### 场景2：日志分析中的错误聚合

**问题**：10万条错误日志，需要按错误类型分组统计

**方案**：
- 提取错误关键词（去掉数字、ID等变化部分）
- 用关键词作 key 分组，统计每组数量

```python
import re

def group_errors(logs):
    groups = defaultdict(list)
    for log in logs:
        # 提取错误模式（去掉数字）
        pattern = re.sub(r'\d+', 'N', log)
        groups[pattern].append(log)

    # 按数量排序
    return sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
```

### 场景3：分布式系统中的数据分片

**问题**：1亿用户数据需要分散到10台服务器

**方案**：
- 用用户ID的哈希值模10，决定存储位置
- 异位词分组的思路：用特征值决定分组

```python
def assign_server(user_id, num_servers=10):
    # 用哈希函数决定服务器编号
    return hash(user_id) % num_servers

# 类似异位词分组：相同特征 → 相同位置
```

---

## 🏋️ 举一反三

掌握本题后，可以挑战这些类似题目：

### 相同模式题

| LeetCode | 题目 | 难度 | 核心思路 |
|----------|------|------|---------|
| **242** | 有效的字母异位词 | Easy | 判断两个字符串是否是异位词（本题的子问题） |
| **438** | 找到字母异位词 | Medium | 滑动窗口 + 字符计数（本题的动态版本） |
| **249** | 移位字符串分组 | Medium | 类似思路，但 key 是"移位后的形式" |

### 哈希分组题

| LeetCode | 题目 | 难度 | 核心思路 |
|----------|------|------|---------|
| **1** | 两数之和 | Easy | 用"差值"作 key 分组查找 |
| **347** | 前 K 个高频元素 | Medium | 用频率作 key 分组 |

### 练习建议

1. **先做 LC 242**（有效的字母异位词）：本题的基础版，练习判断逻辑
2. **再做 LC 438**（找到字母异位词）：动态版本，加深理解
3. **挑战 LC 249**（移位字符串分组）：换一种 key 生成方式

---

## 📝 课后小测

<details>
<summary>💡 问题1：为什么用 sorted(s) 比用 Counter(s) 做 key 更简洁？</summary>

**答案**：
- `sorted(s)` 直接返回字符列表，转成字符串或元组即可
- `Counter(s)` 返回字典，需要 `tuple(sorted(Counter(s).items()))` 才能做 key
- 例如：`"eat"` → `sorted()` = `['a','e','t']`，`Counter()` = `{'e':1,'a':1,'t':1}`
</details>

<details>
<summary>💡 问题2：如果要求输出按字典序排序，代码怎么改？</summary>

**提示**：对最终结果排序

**答案**：
```python
result = list(hash_map.values())
return sorted(result)  # 按每组的第一个元素字典序排序
```
</details>

<details>
<summary>💡 问题3：如果字符串包含大小写和数字，怎么处理？</summary>

**提示**：预处理 + 排序

**答案**：
```python
# 统一转小写
key = ''.join(sorted(s.lower()))

# 或者过滤非字母字符
key = ''.join(sorted(c for c in s if c.isalpha()))
```
</details>

<details>
<summary>💡 问题4：解法二的计数数组为什么是 26？能不能用 dict 代替？</summary>

**答案**：
- **26 = 英文字母数量**（a-z），题目限定只有小写字母
- **可以用 dict**：`Counter(s)` 返回 dict，但需要转成 tuple 才能做 key
- **数组更快**：固定大小，索引访问 O(1)，而 dict 需要哈希计算
</details>

---

## 🎓 总结

### 核心要点

1. **异位词的本质**：字母种类和数量相同，顺序不同
2. **关键技巧**：将字符串"标准化"（排序或计数），用标准化结果作哈希 key
3. **两种方法**：
   - 排序法：O(n × k log k)，代码简洁
   - 计数法：O(n × k)，性能最优
4. **Python 技巧**：`defaultdict`、`sorted()`、`tuple()`

### 面试要点

- 先说暴力法思路，再优化
- 明确说出时间复杂度并解释为什么
- 提到"标准化 key"这个核心思想
- 如果面试官追问，可以提计数法优化

### 下一步

- 练习 LC 242（判断异位词）巩固基础
- 尝试 LC 438（滑动窗口版本）提升难度
- 思考：如果是"相似字符串分组"（允许1个字母不同），怎么做？

---

**恭喜完成第2课！** 🎉

你已经掌握了**哈希表分组**这个重要模式，这在很多算法题中都会用到。记住核心思路：**找到合适的 key，让相似的元素自然聚在一起**！

下一课我们将学习**最长连续序列**，继续深入哈希表的高级应用！

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
