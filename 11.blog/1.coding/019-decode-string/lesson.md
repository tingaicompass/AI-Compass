# 📖 第19课：字符串解码

> **模块**：字符串 | **难度**：Medium ⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/decode-string/
> **前置知识**：第33课（有效的括号）、栈基础
> **预计学习时间**：25分钟

---

## 🎯 题目描述

给定一个经过编码的字符串，返回它解码后的字符串。

编码规则为：`k[encoded_string]`，表示其中方括号内的 `encoded_string` 正好重复 `k` 次。注意 `k` 保证为正整数。

你可以认为输入字符串总是有效的；输入字符串中没有额外的空格，且方括号内的字符串不包含数字（除了嵌套的情况）。

此外，你可以认为原始数据不包含数字，所有的数字只表示重复的次数 `k`。

**示例1：**
```
输入：s = "3[a]2[bc]"
输出："aaabcbc"
```

**示例2：**
```
输入：s = "3[a2[c]]"
输出："accaccacc"
解释：内层 2[c] 先解码为 "cc"，然后与 'a' 拼接得到 "acc"，最后重复3次
```

**示例3：**
```
输入：s = "2[abc]3[cd]ef"
输出："abcabccdcdcdef"
```

**约束条件：**
- `1 <= s.length <= 30`
- `s` 由小写英文字母、数字和方括号 `'[]'` 组成
- `s` 保证是一个有效的输入
- `s` 中所有整数的取值范围为 `[1, 300]`

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 无括号 | `s = "abc"` | `"abc"` | 直接返回原字符串 |
| 单层括号 | `s = "3[a]"` | `"aaa"` | 基础重复 |
| 多段括号 | `s = "2[a]3[b]"` | `"aabbb"` | 多个重复段 |
| 嵌套括号 | `s = "2[a2[b]]"` | `"abbabb"` | 嵌套处理 |
| 深层嵌套 | `s = "3[a2[c]]"` | `"accaccacc"` | 多层嵌套 |
| 混合字符 | `s = "2[abc]3[cd]ef"` | `"abcabccdcdcdef"` | 括号外有字符 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在读一份"压缩说明书"，里面写着："做3次[煮2次[放盐]]"...
>
> 🐌 **笨办法**：从外到内逐层手动展开，每次都要找到最外层括号，展开一次，然后重新扫描。就像剥洋葱一样，一层层慢慢剥，效率极低。
>
> 🚀 **聪明办法**：用一个"待办清单"（栈）记录每一层的任务！从左往右读，遇到数字就记下"要重复几次"，遇到 `[` 就开始新任务，遇到 `]` 就完成当前任务并重复指定次数。就像递归函数一样，先处理内层，再向外返回。

### 关键洞察
**括号嵌套结构天然适合用"栈"或"递归"处理——遇到 `[` 入栈，遇到 `]` 出栈并展开。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1：理解题目 → 锁定输入输出
- **输入**：编码字符串 `s`，长度 1~30，包含小写字母、数字、方括号
- **输出**：解码后的字符串
- **核心规则**：`k[string]` 表示 `string` 重复 `k` 次，支持嵌套

### Step 2：先想笨办法（暴力法）
最直接的想法：反复查找最内层的括号（没有嵌套的括号），展开它，直到没有括号为止。
- 每次用正则或循环找到最内层 `k[...]`
- 替换成重复 k 次的字符串
- 重复上述过程直到没有括号
- **瓶颈**：每次展开都要重新扫描整个字符串，复杂度高

### Step 3：瓶颈分析 → 优化方向
暴力法的问题：**每次展开一个括号后，都要从头扫描，重复工作太多**。

括号嵌套的核心特征：**后进先出（LIFO）** — 最先遇到的 `[` 对应最后遇到的 `]`
- 核心问题：能不能一次遍历就搞定？
- 优化思路：用**栈**维护当前状态（数字、字符串），遇到 `]` 就出栈处理

### Step 4：选择武器
- 选用：**栈（Stack）**
- 理由：
  - 遇到 `[` 时，把当前的"重复次数"和"已构建的字符串"压入栈
  - 遇到 `]` 时，弹出栈顶的"次数"和"前缀"，重复当前字符串
  - 一次遍历完成所有嵌套处理

> 🔑 **模式识别提示**：当题目出现"括号匹配"、"嵌套结构"、"最近相关性"，优先考虑"栈"

---

## 🔑 解法一：栈解法（推荐⭐⭐⭐）

### 思路
用两个栈分别存储：
1. **数字栈**：记录每一层的重复次数
2. **字符串栈**：记录每一层进入括号前的字符串

遍历字符串：
- 遇到**数字**：累积成完整数字（可能是多位数）
- 遇到 `[`：将当前数字和当前字符串压入栈，重置为新的一层
- 遇到 `]`：弹出栈顶的数字和前缀，将当前字符串重复指定次数后拼接
- 遇到**字母**：追加到当前字符串

### 图解过程

```
示例：s = "3[a2[c]]"

初始状态：
num_stack = []
str_stack = []
num = 0
result = ""

第1步：遇到 '3'
num = 3

第2步：遇到 '['
num_stack = [3]       # 压入数字3
str_stack = [""]      # 压入空字符串（当前result为空）
num = 0               # 重置数字
result = ""           # 重置result

第3步：遇到 'a'
result = "a"

第4步：遇到 '2'
num = 2

第5步：遇到 '['
num_stack = [3, 2]    # 压入数字2
str_stack = ["", "a"] # 压入"a"（进入新括号前的字符串）
num = 0
result = ""

第6步：遇到 'c'
result = "c"

第7步：遇到 ']' （内层括号结束）
弹出：prev_num = 2, prev_str = "a"
result = prev_str + result * prev_num
       = "a" + "c" * 2
       = "acc"
num_stack = [3]
str_stack = [""]

第8步：遇到 ']' （外层括号结束）
弹出：prev_num = 3, prev_str = ""
result = prev_str + result * prev_num
       = "" + "acc" * 3
       = "accaccacc"
num_stack = []
str_stack = []

最终结果："accaccacc"
```

再看一个示例：`s = "2[abc]3[cd]ef"`
```
遍历过程：
'2' → num=2
'[' → 压入(2, ""), 重置
'a','b','c' → result="abc"
']' → 弹出(2, ""), result="" + "abc"*2 = "abcabc"
'3' → num=3
'[' → 压入(3, "abcabc"), 重置
'c','d' → result="cd"
']' → 弹出(3, "abcabc"), result="abcabc" + "cd"*3 = "abcabccdcdcd"
'e','f' → result="abcabccdcdcdef"
```

### Python代码

```python
def decode_string(s: str) -> str:
    """
    解法一：栈解法
    思路：用两个栈分别存储数字和字符串，处理嵌套结构
    """
    num_stack = []    # 存储重复次数
    str_stack = []    # 存储每一层的前缀字符串
    num = 0           # 当前数字（可能是多位数）
    result = ""       # 当前构建的字符串

    for char in s:
        if char.isdigit():
            # 累积数字（处理多位数，如 "100[a]"）
            num = num * 10 + int(char)

        elif char == '[':
            # 进入新的嵌套层，保存当前状态
            num_stack.append(num)      # 保存重复次数
            str_stack.append(result)   # 保存当前字符串
            num = 0                    # 重置数字
            result = ""                # 重置字符串

        elif char == ']':
            # 当前层结束，弹出并展开
            prev_num = num_stack.pop()    # 这一层的重复次数
            prev_str = str_stack.pop()    # 进入这一层前的字符串
            result = prev_str + result * prev_num  # 拼接

        else:  # 字母
            # 追加到当前字符串
            result += char

    return result


# ✅ 测试
print(decode_string("3[a]2[bc]"))       # 期望输出："aaabcbc"
print(decode_string("3[a2[c]]"))        # 期望输出："accaccacc"
print(decode_string("2[abc]3[cd]ef"))   # 期望输出："abcabccdcdcdef"
print(decode_string("abc"))             # 期望输出："abc"
```

### 复杂度分析
- **时间复杂度**：O(S) — 其中 S 是解码后字符串的长度，每个字符最多被处理一次
  - 具体地说：如果输入是 `"100[a]"`，输出长度100，时间复杂度是 O(100)
- **空间复杂度**：O(n) — 栈的最大深度取决于嵌套层数，最坏情况 O(n)

---

## ⚡ 解法二：递归解法（DFS）

### 优化思路
用递归模拟栈的行为：遇到 `[` 就进入下一层递归，遇到 `]` 就返回当前层的结果。

> 💡 **关键想法**：递归天然适合处理嵌套结构，函数调用栈就是"栈"！

### Python代码

```python
def decode_string_recursive(s: str) -> str:
    """
    解法二：递归解法
    思路：用递归处理嵌套，函数调用栈替代显式栈
    """
    def dfs(index: int) -> tuple[str, int]:
        """
        从index开始递归解码，返回(解码字符串, 下一个位置)
        """
        result = ""
        num = 0

        while index < len(s):
            char = s[index]

            if char.isdigit():
                num = num * 10 + int(char)
                index += 1

            elif char == '[':
                # 递归处理内层
                sub_str, next_index = dfs(index + 1)
                result += sub_str * num
                num = 0
                index = next_index

            elif char == ']':
                # 当前层结束，返回给上一层
                return result, index + 1

            else:  # 字母
                result += char
                index += 1

        return result, index

    decoded, _ = dfs(0)
    return decoded


# ✅ 测试
print(decode_string_recursive("3[a]2[bc]"))       # "aaabcbc"
print(decode_string_recursive("3[a2[c]]"))        # "accaccacc"
print(decode_string_recursive("2[abc]3[cd]ef"))   # "abcabccdcdcdef"
```

### 复杂度分析
- **时间复杂度**：O(S) — 与栈解法相同
- **空间复杂度**：O(n) — 递归调用栈深度，最坏情况为嵌套层数

---

## 🐍 Pythonic 写法

利用 Python 的字符串操作和栈的简洁性：

```python
# 方法一：单栈优化（用元组同时存储数字和字符串）
def decode_string_compact(s: str) -> str:
    """紧凑版栈解法：用一个栈存储(重复次数, 前缀字符串)元组"""
    stack = []
    num = 0
    result = ""

    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '[':
            stack.append((num, result))
            num, result = 0, ""
        elif char == ']':
            prev_num, prev_str = stack.pop()
            result = prev_str + result * prev_num
        else:
            result += char

    return result


# 方法二：利用列表join优化字符串拼接
def decode_string_optimized(s: str) -> str:
    """优化字符串拼接：用列表+join代替频繁的+="""
    stack = []
    num = 0
    result = []

    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '[':
            stack.append((num, result))
            num, result = 0, []
        elif char == ']':
            prev_num, prev_list = stack.pop()
            result = prev_list + result * prev_num
        else:
            result.append(char)

    return ''.join(result)


print(decode_string_compact("3[a2[c]]"))      # "accaccacc"
print(decode_string_optimized("3[a2[c]]"))    # "accaccacc"
```

> ⚠️ **面试建议**：先写清晰版本展示思路（解法一），再提优化版本展示工程能力。
> 解释"为什么用列表+join比直接+=快"（Python字符串不可变，每次+=都会创建新对象）。

---

## 📊 解法对比

| 维度 | 解法一：显式栈 | 解法二：递归 |
|------|--------------|------------|
| 时间复杂度 | O(S) | O(S) |
| 空间复杂度 | O(n) | O(n) |
| 代码难度 | 中等 | 中等 |
| 面试推荐 | ⭐⭐⭐ | ⭐⭐ |
| 适用场景 | **面试首选，逻辑清晰** | 展示递归能力 |
| 优点 | 易理解、易调试 | 代码简洁 |
| 缺点 | 需要两个栈 | 递归栈可能溢出（极端嵌套） |

**面试建议**：
1. 先用**解法一（栈）**讲解思路（最直观，容易画图演示）
2. 如果面试官问"能不能不用栈？"，再提解法二（递归）
3. 强调栈和递归本质相同，都是利用 LIFO 特性处理嵌套

---

## 🎤 面试现场

> 模拟面试中的完整对话流程，帮你练习"边想边说"。

**面试官**：请你解码这个字符串 `"3[a2[c]]"`。

**你**：（审题30秒）好的，这道题是字符串解码问题。规则是 `k[string]` 表示重复 k 次，支持嵌套。我的第一个想法是用**栈**来处理，因为括号嵌套是典型的"后进先出"结构。

让我梳理一下思路：
1. 遍历字符串，遇到数字就累积（可能是多位数）
2. 遇到 `[` 就把当前数字和字符串压入栈，开始新的一层
3. 遇到 `]` 就弹出栈顶，重复当前字符串并拼接
4. 遇到字母就追加到当前结果

时间复杂度 O(S)，S 是解码后的长度。

**面试官**：很好，请写一下代码。

**你**：（边写边说）我用两个栈，一个存数字，一个存字符串...（写出解法一的代码）

**面试官**：测试一下？

**你**：用 `"3[a2[c]]"` 走一遍：
1. 遇到 `3[`，压入 (3, "")
2. 遇到 `a`，result = "a"
3. 遇到 `2[`，压入 (2, "a")
4. 遇到 `c`，result = "c"
5. 遇到第一个 `]`，弹出 (2, "a")，result = "a" + "c"*2 = "acc"
6. 遇到第二个 `]`，弹出 (3, "")，result = "" + "acc"*3 = "accaccacc"

结果正确 ✅

**面试官**：如果输入是 `"100[leetcode]"`，你的代码能处理吗？

**你**：能！我用 `num = num * 10 + int(char)` 累积多位数，所以 "100" 会正确解析为 100。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能不能不用两个栈？" | "可以！用一个栈存储 `(数字, 字符串)` 元组，逻辑相同但代码更简洁。" |
| "能用递归做吗？" | "可以！递归天然适合嵌套结构，遇到 `[` 就递归进入下一层，遇到 `]` 就返回。本质和栈一样。" |
| "如果字符串特别长怎么办？" | "Python字符串拼接用 `+=` 效率低（每次创建新对象），可以用列表 `append` 再 `join`，时间从 O(n²) 降到 O(n)。" |
| "空间能优化吗？" | "栈的空间取决于嵌套层数，最坏情况 O(n)，无法优化到 O(1)。但实际嵌套层数通常很小。" |
| "实际工程中怎么用？" | "配置文件解析（如模板引擎）、压缩格式解码（run-length encoding）、正则表达式引擎等。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1：判断字符类型
char.isdigit()  # 判断是否数字
char.isalpha()  # 判断是否字母

# 技巧2：多位数字累积
num = 0
for char in "123":
    num = num * 10 + int(char)  # num = 123

# 技巧3：字符串重复
"ab" * 3  # "ababab"

# 技巧4：高效字符串拼接
# ❌ 低效：result += char  (每次创建新字符串)
# ✅ 高效：用列表
result = []
result.append(char)
final = ''.join(result)
```

### 💡 底层原理（选读）

> **为什么栈适合处理嵌套结构？**
>
> - **栈的LIFO特性**与括号匹配的"后开先闭"完美契合
> - 遇到 `[`（开始）→ 压栈（保存状态）
> - 遇到 `]`（结束）→ 出栈（恢复并处理）
> - 典型应用：表达式求值、括号匹配、HTML标签解析、函数调用栈
>
> **Python 字符串拼接为什么慢？**
>
> - Python 字符串**不可变**（immutable），每次 `s += char` 都会：
>   1. 创建新字符串对象（长度+1）
>   2. 复制原字符串内容
>   3. 追加新字符
> - n 次拼接总时间：1+2+3+...+n = O(n²)
> - **优化方案**：用列表 `append` (O(1)) + 最后 `join` (O(n)) = 总 O(n)

### 算法模式卡片 📐
- **模式名称**：栈处理嵌套结构
- **适用条件**：括号匹配、嵌套标签、表达式求值、配置解析
- **识别关键词**：题目中出现"括号"、"嵌套"、"匹配"、"最近相关性"
- **模板代码**：
```python
def process_nested(s: str):
    """栈处理嵌套结构通用模板"""
    stack = []
    for char in s:
        if char == '开始符号':
            stack.append(当前状态)
            重置状态()
        elif char == '结束符号':
            prev_state = stack.pop()
            处理当前层(prev_state)
        else:
            累积当前层数据()
    return 最终结果
```

### 易错点 ⚠️
1. **忘记处理多位数** — 数字可能是 "100" 而不是单个字符 "1"
   - ❌ 错误：`num = int(char)` （只能处理单位数）
   - ✅ 正确：`num = num * 10 + int(char)` （累积多位数）

2. **字符串拼接顺序错误** — 弹出栈时要先拼接前缀
   - ❌ 错误：`result = result * prev_num + prev_str` （顺序反了）
   - ✅ 正确：`result = prev_str + result * prev_num` （前缀在前）

3. **栈未清空导致状态污染** — 进入新的 `[` 时要重置 num 和 result
   - ❌ 错误：忘记 `num = 0; result = ""` 导致累积到下一层
   - ✅ 正确：每次遇到 `[` 都重置当前状态

---

## 🏗️ 工程实战（选读）

> 这个算法思想在真实项目中的应用，让你知道"学了有什么用"。

- **场景1：配置文件解析** — 模板引擎（如Jinja2）解析嵌套模板语法 `{% for %}...{% endfor %}`，用栈处理控制流嵌套。

- **场景2：压缩算法** — Run-Length Encoding (RLE) 压缩格式解码，如 `"3A2B1C"` 解码为 `"AAABBC"`，处理嵌套压缩。

- **场景3：前端框架** — React/Vue 的 JSX/模板解析，处理嵌套组件和标签，构建虚拟 DOM 树。

---

## 🏋️ 举一反三

完成本课后，试试这些同类题目来巩固知识：

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 20. 有效的括号 | Easy | 栈匹配 | 本题的简化版，只判断括号是否匹配 |
| LeetCode 726. 原子数量 | Hard | 栈+哈希表 | 化学式解析，类似嵌套结构 |
| LeetCode 385. 迷你语法分析器 | Medium | 栈+递归 | 解析嵌套列表字符串 |
| LeetCode 1096. 花括号展开II | Hard | 栈+集合 | 更复杂的括号展开规则 |
| LeetCode 636. 函数的独占时间 | Medium | 栈模拟调用栈 | 模拟函数调用的嵌套执行 |

---

## 📝 课后小测

试试这道变体题，不要看答案，自己先想5分钟！

**题目**：给定编码字符串 `s`，规则改为 `k<string>`（用尖括号代替方括号），且数字可能在括号内部，如 `"2<a3<b>>"`。请解码。

例如：
- 输入：`s = "2<a3<b>>"`，输出：`"abbbabbb"` (先 `3<b>` -> `"bbb"`，然后 `2<abbb>` -> `"abbbabbb"`)

<details>
<summary>💡 提示（实在想不出来再点开）</summary>

核心思路不变，只需把 `[` 改成 `<`，`]` 改成 `>`。注意数字可能在括号内部，需要区分"重复次数"和"字符串中的数字"。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def decode_string_angle_brackets(s: str) -> str:
    """
    变体题：用尖括号的字符串解码
    思路：与原题相同，只需替换括号符号
    """
    stack = []
    num = 0
    result = ""

    for char in s:
        if char.isdigit():
            num = num * 10 + int(char)
        elif char == '<':  # 改为尖括号
            stack.append((num, result))
            num, result = 0, ""
        elif char == '>':  # 改为尖括号
            prev_num, prev_str = stack.pop()
            result = prev_str + result * prev_num
        else:
            result += char

    return result


# ✅ 测试
print(decode_string_angle_brackets("2<a3<b>>"))  # "abbbabbb"
print(decode_string_angle_brackets("3<a>2<bc>"))  # "aaabcbc"
```

**核心思路**：
1. 算法完全相同，只是括号符号变了
2. 时间复杂度 O(S)，空间复杂度 O(n)
3. 说明栈处理嵌套结构的通用性——只要是"开始-结束"配对，都能用栈

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
