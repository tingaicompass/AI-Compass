# 📖 第23课:反转字符串中的单词

> **模块**:字符串 | **难度**:Medium ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/reverse-words-in-a-string/
> **前置知识**:无(基础题)
> **预计学习时间**:20分钟

---

## 🎯 题目描述

给你一个字符串 `s`,请你反转字符串中**单词**的顺序。

**单词**是由非空格字符组成的字符串。`s` 中使用至少一个空格将字符串中的**单词**分隔开。

返回**单词顺序颠倒**且**单词之间用单个空格**连接的结果字符串。

**注意**:输入字符串 `s` 中可能会存在前导空格、尾随空格或者单词间的多个空格。返回的结果字符串中,单词间应当仅用单个空格分隔,且不包含任何额外的空格。

**示例 1:**
```
输入:s = "the sky is blue"
输出:"blue is sky the"
```

**示例 2:**
```
输入:s = "  hello world  "
输出:"world hello"
解释:反转后的字符串中不能存在前导空格和尾随空格
```

**示例 3:**
```
输入:s = "a good   example"
输出:"example good a"
解释:如果两个单词间有多余的空格,反转后的字符串需要将单词间的空格减少到仅有一个
```

**约束条件:**
- `1 <= s.length <= 10⁴`
- `s` 包含英文大小写字母、数字和空格 `' '`
- `s` 中**至少存在一个**单词

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单个单词 | `"hello"` | `"hello"` | 无需反转 |
| 前导空格 | `"  hello"` | `"hello"` | 去除前导 |
| 尾随空格 | `"hello  "` | `"hello"` | 去除尾随 |
| 多余空格 | `"a  b"` | `"b a"` | 压缩为单个 |
| 全是空格 | `"   "` | `""` | 无单词 |
| 两个单词 | `"a b"` | `"b a"` | 基本情况 |
| 最大长度 | `n=10000` | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在整理一排书,书名按顺序是"红楼梦 西游记 水浒传 三国演义",现在你要把这排书倒过来摆放。
>
> 🐌 **笨办法**:你先把每本书从书架上拿下来,摆在地上,然后再从最后一本开始依次放回书架。但有个问题:书之间原本的间距不一样(有的书之间有两个书挡,有的只有一个),你放回去时要手动调整成统一的一个书挡间距。
>
> 🚀 **聪明办法**:你用Python的"魔法工具" - `split()` 自动把所有书提取出来(忽略间距),然后用 `reverse()` 把顺序颠倒,最后用 `join()` 用统一的间距(一个空格)重新排列。**关键是Python已经帮你处理好了空格问题!**

### 关键洞察

**Python的 `split()` 不带参数时会自动处理所有空格(前导、尾随、多余),返回纯粹的单词列表。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:字符串 `s`,可能有前导/尾随/多余空格
- **输出**:单词顺序颠倒的字符串,单词间只有一个空格
- **限制**:要处理所有类型的空格

### Step 2:先想笨办法(手动分割)

手动遍历字符串,用一个变量累积当前单词,遇到空格就把单词保存到列表,最后反转列表并拼接。
- 时间复杂度:O(n)
- 瓶颈在哪:**手动处理空格很麻烦,要考虑前导、尾随、连续空格等多种情况**

### Step 3:瓶颈分析 → 优化方向

手动分割容易出错,Python的 `split()` 已经帮我们处理好了!
- 核心问题:"多余空格的处理很繁琐"
- 优化思路:"用内置函数 `split()` 一次性解决"

### Step 4:选择武器

- 选用:**Python内置方法 `split()` + `reverse()` + `join()`**
- 理由:`split()` 不带参数时自动处理所有空格,返回单词列表

> 🔑 **模式识别提示**:当题目要求处理空格时,优先考虑Python的 `split()` 和 `join()`

---

## 🔑 解法一:Python内置方法(推荐)

### 思路

利用Python的字符串方法:
1. `split()` 分割成单词列表(自动处理所有空格)
2. `reverse()` 或切片 `[::-1]` 反转列表
3. `' '.join()` 用单个空格连接

### 图解过程

```
示例: s = "  hello world  "

Step 1: split() 分割
  "  hello world  " → ["hello", "world"]
  (自动去除前导、尾随和多余空格)

Step 2: reverse() 反转
  ["hello", "world"] → ["world", "hello"]

Step 3: join(' ') 连接
  ["world", "hello"] → "world hello"

最终答案: "world hello"
```

### Python代码

```python
def reverse_words(s: str) -> str:
    """
    解法一:Python内置方法(推荐)
    思路:split分割 → 反转 → join连接
    """
    # 方法1:split + reverse + join
    words = s.split()  # 自动处理所有空格
    words.reverse()    # 原地反转
    return ' '.join(words)


# ✅ 测试
print(reverse_words("the sky is blue"))      # 期望输出: "blue is sky the"
print(reverse_words("  hello world  "))      # 期望输出: "world hello"
print(reverse_words("a good   example"))     # 期望输出: "example good a"
```

### 复杂度分析

- **时间复杂度**:O(n) - split遍历一次,reverse O(k) (k为单词数),join遍历一次
  - 具体地说:如果输入规模 n=1000,大约需要 2000-3000 次操作
- **空间复杂度**:O(n) - 存储单词列表和结果字符串

### 优缺点

- ✅ 代码极简,只需3行
- ✅ 不易出错,Python自动处理边界情况
- ✅ 可读性强,一眼看懂逻辑
- ❌ 需要O(n)额外空间(面试官可能追问"能否O(1)?")

---

## ⚡ 解法二:一行Pythonic写法

### 优化思路

利用切片反转和链式调用,压缩成一行。

### Python代码

```python
def reverse_words_oneliner(s: str) -> str:
    """Pythonic 一行写法"""
    return ' '.join(s.split()[::-1])


# 或者用 reversed() 函数
def reverse_words_oneliner_v2(s: str) -> str:
    """另一种一行写法"""
    return ' '.join(reversed(s.split()))
```

**解释**:
- `s.split()`:分割成 `['the', 'sky', 'is', 'blue']`
- `[::-1]`:切片反转 → `['blue', 'is', 'sky', 'the']`
- `' '.join(...)`:用空格连接 → `"blue is sky the"`

或者用 `reversed()` 返回迭代器(更节省内存):
- `reversed(s.split())`:返回反向迭代器
- `' '.join(...)`:join可以接受迭代器

### 复杂度分析

- **时间复杂度**:O(n)
- **空间复杂度**:O(n)

---

## 🚀 解法三:双指针手动分割(原地算法思想)

### 优化思路

如果要求O(1)空间(在字符数组上原地操作),可以分两步:
1. 反转整个字符串:`"the sky" → "yks eht"`
2. 反转每个单词:`"yks eht" → "sky the"`

但在Python中字符串不可变,需要转成列表。这种方法更适合C/C++等语言。

### 图解过程

```
示例: s = "the sky"

Step 1: 反转整个字符串
  "the sky" → "yks eht"

Step 2: 反转每个单词
  "yks" → "sky"
  "eht" → "the"
  最终: "sky the"
```

### Python代码

```python
def reverse_words_inplace(s: str) -> str:
    """
    解法三:模拟原地算法(字符数组)
    思路:先反转整个字符串,再反转每个单词
    """
    # 1. 去除多余空格并转为列表
    chars = list(s.strip())  # 去前导尾随空格
    # 压缩多余空格
    clean_chars = []
    for i, c in enumerate(chars):
        if c != ' ' or (i > 0 and chars[i - 1] != ' '):
            clean_chars.append(c)

    # 2. 反转整个字符串
    clean_chars.reverse()

    # 3. 反转每个单词
    def reverse_word(chars, left, right):
        while left < right:
            chars[left], chars[right] = chars[right], chars[left]
            left += 1
            right -= 1

    start = 0
    for i in range(len(clean_chars)):
        if clean_chars[i] == ' ':
            reverse_word(clean_chars, start, i - 1)
            start = i + 1
    reverse_word(clean_chars, start, len(clean_chars) - 1)  # 最后一个单词

    return ''.join(clean_chars)


# ✅ 测试
print(reverse_words_inplace("the sky is blue"))      # 期望输出: "blue is sky the"
print(reverse_words_inplace("  hello world  "))      # 期望输出: "world hello"
```

### 复杂度分析

- **时间复杂度**:O(n) - 遍历3次(去空格、反转整体、反转单词)
- **空间复杂度**:O(n) - Python字符串不可变,必须转列表(在C++等语言中可以做到O(1))

---

## 🐍 Pythonic 写法总结

```python
# 最简洁版本
def reverse_words_best(s: str) -> str:
    return ' '.join(s.split()[::-1])

# 或者用 reversed (节省内存)
def reverse_words_best_v2(s: str) -> str:
    return ' '.join(reversed(s.split()))
```

> ⚠️ **面试建议**:直接写解法一或一行版本即可,因为Python就是为了简洁而生!
> 如果面试官追问"能否O(1)空间",可以回答:"Python字符串不可变,必须用O(n)空间。在C++中可以用解法三原地算法做到O(1)"

---

## 📊 解法对比

| 维度 | 解法一:内置方法 | 解法二:一行 | 解法三:原地算法 |
|------|----------------|------------|----------------|
| 时间复杂度 | O(n) | O(n) | O(n) |
| 空间复杂度 | O(n) | O(n) | O(n)* |
| 代码难度 | 简单 | 极简 | 中等 |
| 面试推荐 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| 适用场景 | Python面试首选 | 展示语言特性 | C++等语言 |

*Python字符串不可变,解法三也需要O(n)空间;在C++中可以真正做到O(1)

**面试建议**:直接写一行版本 `' '.join(s.split()[::-1])`,简洁高效!如果面试官要求解释,可以展开成解法一的3步。

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你反转字符串中单词的顺序,注意处理多余空格。

**你**:(审题10秒)好的,举个例子,"the sky is blue" 要变成 "blue is sky the"。而且要去除前导、尾随和多余空格。

我的思路很简单:
1. 用 `split()` 分割成单词列表,它会自动处理所有空格
2. 用切片 `[::-1]` 反转列表
3. 用 `' '.join()` 连接成字符串

一行代码就能搞定:`' '.join(s.split()[::-1])`

**面试官**:很好,请写一下。

**你**:(写代码)

```python
def reverse_words(s: str) -> str:
    return ' '.join(s.split()[::-1])
```

**面试官**:测试一下?

**你**:用示例 `"  hello world  "` 测试:
- `split()` 得到 `["hello", "world"]`
- `[::-1]` 反转得到 `["world", "hello"]`
- `join()` 得到 `"world hello"`

结果正确!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能否O(1)空间?" | Python字符串不可变,必须O(n)。在C++中可以原地修改字符数组:先反转整个字符串,再反转每个单词,这样可以做到O(1)空间。 |
| "split()的参数是什么?" | 不带参数时,split()会按所有空白字符(空格、tab、换行)分割,并自动去除前导/尾随/多余空格。如果传入参数如 `split(' ')`,则只按单个空格分割,不会去除多余空格。 |
| "reversed和[::-1]有什么区别?" | `[::-1]`是切片,立即创建反转后的新列表,O(n)空间。`reversed()`返回迭代器,延迟计算,节省空间。两者都能用,但 `reversed()` 更节省内存。 |
| "如果单词内部也要反转呢?" | 那就需要先反转单词顺序,再反转每个单词内部:`' '.join(word[::-1] for word in reversed(s.split()))` |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:split() 不带参数 — 自动处理所有空白字符
s = "  a  b  "
s.split()       # ['a', 'b'] (去除所有多余空格)
s.split(' ')    # ['', '', 'a', '', 'b', '', ''] (保留空字符串)

# 技巧2:join() — 用指定分隔符连接列表
words = ['hello', 'world']
' '.join(words)       # "hello world"
', '.join(words)      # "hello, world"

# 技巧3:切片反转 [::-1]
lst = [1, 2, 3]
lst[::-1]       # [3, 2, 1]

# 技巧4:reversed() — 返回迭代器
lst = [1, 2, 3]
list(reversed(lst))   # [3, 2, 1]
' '.join(reversed(['a', 'b']))  # "b a" (join可以接受迭代器)

# 技巧5:列表推导 + 链式调用
' '.join(word[::-1] for word in reversed(s.split()))
```

### 💡 底层原理(选读)

> **split() 为什么这么强大?**
>
> `split()` 不带参数时,底层会调用 C 实现的字符串分割算法,它会:
> 1. 跳过所有前导空白字符
> 2. 遇到非空白字符开始累积单词
> 3. 遇到空白字符时保存单词,然后跳过所有连续空白
> 4. 重复直到字符串末尾
>
> 这个算法的时间复杂度是O(n),而且高度优化,比手动实现快得多。
>
> **reversed() vs [::-1] 的性能差异?**
>
> - `[::-1]`:立即创建新列表,内存开销大,但访问速度快(直接索引)
> - `reversed()`:返回迭代器,内存开销小,但只能遍历一次
>
> 在 `join()` 中使用时,两者性能接近,因为 `join()` 会遍历一次。但如果需要多次访问,`[::-1]` 更快。

### 算法模式卡片 📐

- **模式名称**:字符串分割重组
- **适用条件**:需要按分隔符分割字符串,处理后重新组合
- **识别关键词**:"反转单词"、"重组单词"、"处理空格"
- **模板代码**:
```python
def split_process_join_template(s: str, sep: str = ' ') -> str:
    """字符串分割-处理-重组的通用模板"""
    # 1. 分割
    parts = s.split(sep)  # 不带参数时自动处理所有空白

    # 2. 处理(过滤、转换、反转等)
    processed = [process(part) for part in parts if is_valid(part)]

    # 3. 重组
    return sep.join(processed)
```

### 易错点 ⚠️

1. **split() 带参数 vs 不带参数**
   - ❌ 错误:`s.split(' ')` 会保留空字符串,如 `"a  b".split(' ')` → `['a', '', 'b']`
   - ✅ 正确:`s.split()` 自动去除所有多余空格,`"a  b".split()` → `['a', 'b']`

2. **忘记处理前导/尾随空格**
   - ❌ 错误:手动分割时忘记 `strip()`
   - ✅ 正确:用 `split()` 或者手动 `s.strip().split(' ')`

3. **join() 的参数类型**
   - ❌ 错误:`' '.join("abc")` → `"a b c"` (字符串也是可迭代对象!)
   - ✅ 正确:`' '.join(['a', 'b', 'c'])` → `"a b c"`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:日志解析**
  - 解析日志文件时,需要提取关键字段并重新组合。如把 `"2024-01-01 ERROR main.py:42 Connection failed"` 转换成 `"failed Connection main.py:42 ERROR 2024-01-01"`

- **场景2:搜索引擎优化**
  - 处理用户搜索词,去除多余空格,调整关键词顺序以匹配索引

- **场景3:自然语言处理(NLP)**
  - 文本预处理时,需要清理、标准化文本(去空格、反转、重组等)

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 557. 反转字符串中的单词III | Easy | 字符串处理 | 反转每个单词的字符顺序,但保持单词顺序不变 |
| LeetCode 186. 反转字符串中的单词II | Medium | 原地算法 | 字符数组原地反转,O(1)空间 |
| LeetCode 58. 最后一个单词的长度 | Easy | 字符串遍历 | 找出最后一个单词的长度,注意尾随空格 |
| LeetCode 14. 最长公共前缀 | Easy | 字符串处理 | 找出字符串数组的最长公共前缀 |
| LeetCode 344. 反转字符串 | Easy | 双指针 | 原地反转整个字符串 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个字符串,反转每个单词的字符,但保持单词顺序不变。(LeetCode 557)

例如:`"Let's take LeetCode contest"` → `"s'teL ekat edoCteeL tsetnoc"`

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

与本题相反,这次是保持单词顺序,反转每个单词内部的字符。可以:
1. `split()` 分割成单词
2. 对每个单词用 `[::-1]` 反转
3. `join()` 连接回去

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def reverse_words_iii(s: str) -> str:
    """
    反转每个单词的字符,保持单词顺序
    """
    return ' '.join(word[::-1] for word in s.split())


# 测试
print(reverse_words_iii("Let's take LeetCode contest"))
# 输出: "s'teL ekat edoCteeL tsetnoc"
```

或者一行版本:
```python
def reverse_words_iii_oneliner(s: str) -> str:
    return ' '.join(s.split()[::-1][::-1])  # 不对,这样会反转单词顺序

# 正确的一行:
def reverse_words_iii_correct(s: str) -> str:
    return ' '.join([word[::-1] for word in s.split()])
```

核心思路:遍历每个单词,反转其字符,保持单词顺序。

时间复杂度 O(n),空间复杂度 O(n)。

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
