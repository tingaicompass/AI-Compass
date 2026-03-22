# 07 - 字典dict:哈希表的Python实现

> **学习目标**: 掌握字典的创建、访问和在算法题中的应用

---

## 📖 知识点讲解

### 什么是字典?

字典(dict)是Python中的**键值对**数据结构,对应其他语言中的HashMap/HashTable。

**生活化比喻**:
> 字典就像一本**真正的字典书**:
> - **键(key)** = 要查的单词(比如"apple")
> - **值(value)** = 单词的解释(比如"苹果")
> - 查询速度极快:不管字典有多厚,翻到任何一个单词都只需要**1秒钟**(O(1)时间复杂度)

### 为什么字典这么重要?

在107道LeetCode题中,**80%的题目**都用到字典!

| 应用场景 | 例子 |
|---------|------|
| 查找配对 | 第1课:两数之和 |
| 计数统计 | 第2课:字母异位词分组 |
| 缓存结果 | 避免重复计算 |
| 构建映射 | 字符到索引的映射 |

**核心优势**: 查找、插入、删除都是**O(1)**,比列表的O(n)快得多!

---

## 💻 代码示例

### 示例1:创建字典

```python
# 方法1:花括号 {}
student = {"name": "Alice", "age": 20, "grade": "A"}
print(student)  # {'name': 'Alice', 'age': 20, 'grade': 'A'}

# 方法2:dict()函数
another = dict(name="Bob", age=21)
print(another)  # {'name': 'Bob', 'age': 21}

# 方法3:空字典
empty = {}
# 或
empty = dict()

# 方法4:从列表创建
pairs = [("a", 1), ("b", 2), ("c", 3)]
d = dict(pairs)
print(d)  # {'a': 1, 'b': 2, 'c': 3}

# ⚠️ 注意:键必须是不可变类型(字符串、数字、元组)
valid = {1: "one", "two": 2, (3, 4): "tuple"}  # ✅ 合法
# invalid = {[1, 2]: "list"}  # ❌ 报错:列表不能作为键
```

---

### 示例2:访问和修改

```python
scores = {"Alice": 90, "Bob": 85, "Charlie": 92}

# 访问值(如果键不存在会报错KeyError)
print(scores["Alice"])  # 90
# print(scores["David"])  # ❌ KeyError!

# 安全访问:get方法(键不存在返回None或默认值)
print(scores.get("Alice"))     # 90
print(scores.get("David"))     # None
print(scores.get("David", 0))  # 0 (指定默认值)

# 修改值
scores["Alice"] = 95
print(scores)  # {'Alice': 95, 'Bob': 85, 'Charlie': 92}

# 添加新键值对
scores["David"] = 88
print(scores)  # {'Alice': 95, 'Bob': 85, 'Charlie': 92, 'David': 88}

# 删除键值对
del scores["Bob"]
print(scores)  # {'Alice': 95, 'Charlie': 92, 'David': 88}

# 弹出键值对(返回值)
david_score = scores.pop("David")
print(david_score)  # 88
print(scores)  # {'Alice': 95, 'Charlie': 92}
```

---

### 示例3:检查键是否存在

```python
scores = {"Alice": 90, "Bob": 85}

# 方法1:in关键字(最常用)
if "Alice" in scores:
    print("Alice在字典中")  # ✅ 输出

if "Charlie" not in scores:
    print("Charlie不在字典中")  # ✅ 输出

# 方法2:get方法配合判断
if scores.get("Alice") is not None:
    print("Alice存在")
```

---

### 示例4:遍历字典

```python
scores = {"Alice": 90, "Bob": 85, "Charlie": 92}

# 遍历键
for name in scores:
    print(name)  # Alice, Bob, Charlie

# 遍历键(显式)
for name in scores.keys():
    print(name)

# 遍历值
for score in scores.values():
    print(score)  # 90, 85, 92

# 遍历键值对(最常用!)
for name, score in scores.items():
    print(f"{name}: {score}")
    # Alice: 90
    # Bob: 85
    # Charlie: 92
```

---

### 示例5:字典常用方法

```python
d = {"a": 1, "b": 2, "c": 3}

# 获取所有键
keys = list(d.keys())
print(keys)  # ['a', 'b', 'c']

# 获取所有值
values = list(d.values())
print(values)  # [1, 2, 3]

# 获取所有键值对
items = list(d.items())
print(items)  # [('a', 1), ('b', 2), ('c', 3)]

# 清空字典
d.clear()
print(d)  # {}

# 复制字典
original = {"a": 1, "b": 2}
copy = original.copy()
copy["a"] = 10
print(original)  # {'a': 1, 'b': 2} (原字典不变)
print(copy)  # {'a': 10, 'b': 2}

# setdefault:如果键不存在就设置默认值
d = {"a": 1}
d.setdefault("a", 10)  # a已存在,不改变
print(d)  # {'a': 1}

d.setdefault("b", 20)  # b不存在,设置为20
print(d)  # {'a': 1, 'b': 20}
```

---

### 示例6:字典推导式

```python
# 创建平方数字典
squares = {x: x**2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 筛选字典
scores = {"Alice": 90, "Bob": 75, "Charlie": 92, "David": 65}
passed = {name: score for name, score in scores.items() if score >= 80}
print(passed)  # {'Alice': 90, 'Charlie': 92}

# 键值互换
original = {"a": 1, "b": 2, "c": 3}
swapped = {value: key for key, value in original.items()}
print(swapped)  # {1: 'a', 2: 'b', 3: 'c'}
```

---

## 🎯 在算法题中的应用

### 应用场景1:查找配对 - 两数之和

**第1课:两数之和**
```python
def twoSum(nums, target):
    seen = {}  # ← 字典存储已见过的数

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:  # ← O(1)查找
            return [seen[complement], i]
        seen[num] = i  # ← 存储数字和索引

    return []
```

**关键点**: 字典的`in`操作是O(1),比列表的O(n)快得多!

---

### 应用场景2:计数统计 - 字母异位词分组

**第2课:字母异位词分组**
```python
def groupAnagrams(strs):
    groups = {}  # ← 字典存储分组

    for word in strs:
        # 排序后的字符串作为键
        key = "".join(sorted(word))

        # 如果键不存在,创建空列表
        if key not in groups:
            groups[key] = []

        # 添加单词到对应组
        groups[key].append(word)

    return list(groups.values())

# 或者用get方法
def groupAnagrams(strs):
    groups = {}
    for word in strs:
        key = "".join(sorted(word))
        groups[key] = groups.get(key, []) + [word]  # ← get提供默认值
    return list(groups.values())
```

---

### 应用场景3:前缀和 - 和为K的子数组

**第4课:和为K的子数组**
```python
def subarraySum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # ← 字典存储前缀和出现次数

    for num in nums:
        prefix_sum += num

        # 查找是否存在 prefix_sum - k
        if prefix_sum - k in prefix_count:  # ← O(1)查找
            count += prefix_count[prefix_sum - k]

        # 更新当前前缀和的计数
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count
```

---

### 应用场景4:字符频率 - 滑动窗口

**第5课:找到字符串中所有字母异位词**
```python
def findAnagrams(s, p):
    from collections import Counter

    result = []
    p_count = Counter(p)  # ← 字典统计p的字符频率
    window = {}

    left = 0
    for right in range(len(s)):
        # 右边字符进入窗口
        char = s[right]
        window[char] = window.get(char, 0) + 1

        # 窗口大小达到p的长度
        if right - left + 1 == len(p):
            # 比较两个字典
            if window == p_count:  # ← 字典可以直接比较!
                result.append(left)

            # 左边字符移出窗口
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

    return result
```

---

## 🏋️ 快速练习

### 练习1:统计字符出现次数

给定字符串`"hello"`,统计每个字符出现的次数。

<details>
<summary>点击查看答案</summary>

```python
s = "hello"

# 方法1:手动统计
count = {}
for char in s:
    count[char] = count.get(char, 0) + 1
print(count)  # {'h': 1, 'e': 1, 'l': 2, 'o': 1}

# 方法2:使用setdefault
count = {}
for char in s:
    count.setdefault(char, 0)
    count[char] += 1
print(count)

# 方法3:使用Counter(推荐)
from collections import Counter
count = Counter(s)
print(dict(count))  # {'h': 1, 'e': 1, 'l': 2, 'o': 1}
```

</details>

---

### 练习2:检查两个字符串是否为异位词

异位词:字母相同但顺序不同,如"listen"和"silent"。

```python
def isAnagram(s, t):
    # 在这里实现
    pass

print(isAnagram("listen", "silent"))  # True
print(isAnagram("hello", "world"))    # False
```

<details>
<summary>点击查看答案</summary>

```python
def isAnagram(s, t):
    # 方法1:比较字符频率字典
    if len(s) != len(t):
        return False

    count_s = {}
    count_t = {}

    for char in s:
        count_s[char] = count_s.get(char, 0) + 1

    for char in t:
        count_t[char] = count_t.get(char, 0) + 1

    return count_s == count_t

# 方法2:使用Counter
from collections import Counter
def isAnagram(s, t):
    return Counter(s) == Counter(t)

# 方法3:排序比较(不用字典)
def isAnagram(s, t):
    return sorted(s) == sorted(t)
```

</details>

---

### 练习3:找出数组中第一个只出现一次的元素

```python
nums = [2, 3, 4, 2, 3, 5]
# 答案应该是4
```

<details>
<summary>点击查看答案</summary>

```python
nums = [2, 3, 4, 2, 3, 5]

# 步骤1:统计每个数字出现次数
count = {}
for num in nums:
    count[num] = count.get(num, 0) + 1

# 步骤2:找出第一个出现次数为1的
for num in nums:
    if count[num] == 1:
        print(num)  # 4
        break
```

</details>

---

## 🎓 小结

✅ **创建字典**: `{}`, `dict()`, `{key: value}`
✅ **访问元素**: `d[key]`, `d.get(key, default)`
✅ **检查键**: `key in d`
✅ **遍历**: `for k, v in d.items()`
✅ **常用方法**: `keys()`, `values()`, `items()`, `pop()`, `get()`

**核心优势**: O(1)查找,是算法题的利器!

**最常用的模式**:
```python
# 模式1:计数
count = {}
for x in data:
    count[x] = count.get(x, 0) + 1

# 模式2:查找配对
seen = {}
for i, x in enumerate(data):
    if target - x in seen:
        return [seen[target - x], i]
    seen[x] = i
```

**下一步**: [08-集合set.md](./08-集合set.md)

---

*字典是Python最强大的数据结构之一,必须熟练掌握!* 🚀

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
