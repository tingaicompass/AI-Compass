> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 19 - Pythonic 写法: 地道的Python代码

> **学习目标**: 掌握Pythonic编程风格,写出优雅、高效、可读性强的Python代码

---

## 💡 什么是 Pythonic?

"Pythonic" 是指遵循 Python 社区的编程风格和最佳实践,充分利用 Python 的语言特性,写出简洁、优雅、高效的代码。

**核心理念**: The Zen of Python (在终端输入 `import this` 查看)
- 优美胜于丑陋
- 明了胜于晦涩
- 简洁胜于复杂
- 可读性很重要

---

## 📊 快速参考表

| 场景 | ❌ 不够 Pythonic | ✅ Pythonic | 说明 |
|------|-----------------|------------|------|
| **变量交换** | `temp=a; a=b; b=temp` | `a, b = b, a` | 序列解包 |
| **列表反转** | `for i in range(len(a)-1,-1,-1)` | `a[::-1]` | 切片 |
| **遍历列表** | `for i in range(len(a))` | `for i, v in enumerate(a)` | enumerate |
| **并行遍历** | `for i in range(len(a))` | `for x, y in zip(a, b)` | zip |
| **列表构建** | `for x in a: result.append(f(x))` | `[f(x) for x in a]` | 列表推导式 |
| **条件表达式** | `if x>0: r="正" else: r="负"` | `r = "正" if x>0 else "负"` | 三元表达式 |
| **默认值** | `if v is None: v=[]` | `v = v or []` | or 短路 |
| **字符串拼接** | `for w in words: s+=w` | `"".join(words)` | join |
| **文件读取** | `f=open(); f.read(); f.close()` | `with open() as f: f.read()` | with |
| **检查存在** | `found=False; for x in a: if x==t: found=True` | `t in a` | in |
| **字典取值** | `d["key"]` | `d.get("key", default)` | get |
| **列表去重** | `for x in a: if x not in r: r.append(x)` | `list(set(a))` | set |
| **多条件** | `if x==1 or x==2 or x==3` | `if x in (1,2,3)` | in |
| **范围判断** | `if x>0 and x<10` | `if 0<x<10` | 链式比较 |
| **字典合并** | `for k,v in d2.items(): d1[k]=v` | `d1 \| d2` (3.9+) | 字典运算符 |
| **列表展平** | `for row in m: for x in row: r.append(x)` | `[x for row in m for x in row]` | 推导式 |
| **函数解包** | `f(a[0], a[1], a[2])` | `f(*a)` | * 解包 |
| **任意满足** | `for x in a: if cond(x): found=True; break` | `any(cond(x) for x in a)` | any |
| **全部满足** | `for x in a: if not cond(x): ok=False; break` | `all(cond(x) for x in a)` | all |
| **空值检查** | `if len(a)==0` | `if not a` | 真值测试 |
| **频率统计** | `for x in a: d[x]=d.get(x,0)+1` | `Counter(a)` | Counter |
| **最大K个** | `sorted(a)[-k:]` | `heapq.nlargest(k, a)` | heapq |
| **二分查找** | 手动实现 | `bisect.bisect_left(a, x)` | bisect |
| **格式化** | `"Name: "+name+", Age: "+str(age)` | `f"Name: {name}, Age: {age}"` | f-string |

---

## 💻 Pythonic 代码示例

### 1. 变量交换

```python
# ❌ 不够Pythonic (使用临时变量)
temp = a
a = b
b = temp

# ✅ Pythonic (元组解包)
a, b = b, a
```

### 2. 列表/字符串反转

```python
# ❌ 不够Pythonic
result = []
for i in range(len(arr)-1, -1, -1):
    result.append(arr[i])

# ✅ Pythonic (切片)
result = arr[::-1]
s = s[::-1]  # 字符串反转
```

### 3. 遍历列表

```python
nums = [10, 20, 30]

# ❌ 不够Pythonic (使用索引)
for i in range(len(nums)):
    print(i, nums[i])

# ✅ Pythonic (使用enumerate)
for i, num in enumerate(nums):
    print(i, num)
```

### 4. 同时遍历多个列表

```python
names = ["Alice", "Bob", "Charlie"]
scores = [85, 92, 78]

# ❌ 不够Pythonic
for i in range(len(names)):
    print(names[i], scores[i])

# ✅ Pythonic (使用zip)
for name, score in zip(names, scores):
    print(name, score)
```

### 5. 列表构建

```python
# ❌ 不够Pythonic
squares = []
for i in range(10):
    if i % 2 == 0:
        squares.append(i ** 2)

# ✅ Pythonic (列表推导式)
squares = [i ** 2 for i in range(10) if i % 2 == 0]
```

### 6. 条件表达式

```python
# ❌ 不够Pythonic
if x > 0:
    result = "正数"
else:
    result = "非正数"

# ✅ Pythonic (三元表达式)
result = "正数" if x > 0 else "非正数"
```

### 7. 默认值处理

```python
# ❌ 不够Pythonic
if value is None:
    value = []

# ✅ Pythonic (使用or)
value = value or []

# ✅ 更好 (避免0、False等假值的问题)
value = value if value is not None else []
```

### 8. 字符串拼接

```python
words = ["Python", "is", "awesome"]

# ❌ 不够Pythonic (循环拼接,效率低)
result = ""
for word in words:
    result += word + " "

# ✅ Pythonic (使用join)
result = " ".join(words)
```

### 9. 文件读取

```python
# ❌ 不够Pythonic
f = open("file.txt")
content = f.read()
f.close()

# ✅ Pythonic (使用with,自动关闭)
with open("file.txt") as f:
    content = f.read()
```

### 10. 检查元素是否存在

```python
# ❌ 不够Pythonic
found = False
for item in items:
    if item == target:
        found = True
        break

# ✅ Pythonic (使用in)
found = target in items
```

### 11. 获取字典值

```python
d = {"name": "Alice", "age": 20}

# ❌ 可能抛出KeyError
value = d["city"]

# ✅ Pythonic (使用get,提供默认值)
value = d.get("city", "Unknown")
```

### 12. 列表去重

```python
nums = [1, 2, 2, 3, 1, 4]

# ❌ 不够Pythonic
result = []
for num in nums:
    if num not in result:
        result.append(num)

# ✅ Pythonic (使用set)
result = list(set(nums))

# ✅ 更好 (保持原始顺序)
result = list(dict.fromkeys(nums))
```

### 13. 多条件判断

```python
# ❌ 不够Pythonic
if x == 1 or x == 2 or x == 3:
    print("找到了")

# ✅ Pythonic (使用in)
if x in (1, 2, 3):
    print("找到了")

# ✅ 范围判断
if 0 <= x < 10:
    print("在范围内")
```

### 14. 字典合并

```python
d1 = {"a": 1, "b": 2}
d2 = {"b": 3, "c": 4}

# ✅ Pythonic (Python 3.9+)
merged = d1 | d2

# ✅ Pythonic (Python 3.5+)
merged = {**d1, **d2}
```

### 15. 链式比较

```python
x = 5

# ❌ 不够Pythonic
if x > 0 and x < 10:
    print("在范围内")

# ✅ Pythonic (链式比较)
if 0 < x < 10:
    print("在范围内")
```

### 16. 列表展平

```python
matrix = [[1, 2], [3, 4], [5, 6]]

# ❌ 不够Pythonic
result = []
for row in matrix:
    for num in row:
        result.append(num)

# ✅ Pythonic (列表推导式)
result = [num for row in matrix for num in row]
```

### 17. 函数参数解包

```python
def add(a, b, c):
    return a + b + c

nums = [1, 2, 3]

# ❌ 不够Pythonic
result = add(nums[0], nums[1], nums[2])

# ✅ Pythonic (使用*解包)
result = add(*nums)
```

### 18. 使用any()和all()

```python
nums = [1, 2, 3, 4, 5]

# ❌ 不够Pythonic
has_even = False
for num in nums:
    if num % 2 == 0:
        has_even = True
        break

# ✅ Pythonic (使用any)
has_even = any(num % 2 == 0 for num in nums)

# ✅ 检查是否全部满足条件
all_positive = all(num > 0 for num in nums)
```

### 19. 空值检查

```python
nums = []

# ❌ 不够Pythonic
if len(nums) == 0:
    print("空列表")

# ✅ Pythonic (利用真值测试)
if not nums:
    print("空列表")

# 非空检查
if nums:
    print("非空")
```

### 20. 多个返回值

```python
# ✅ Pythonic (返回元组)
def get_user_info():
    return "Alice", 20, "Beijing"

name, age, city = get_user_info()

# 忽略不需要的值
name, _, city = get_user_info()
```

### 21. 字符串格式化

```python
name = "Alice"
age = 20

# ❌ 不够Pythonic (字符串拼接)
result = "Name: " + name + ", Age: " + str(age)

# ✅ Pythonic (f-string, Python 3.6+, 最推荐)
result = f"Name: {name}, Age: {age}"

# ✅ Pythonic (format 方法)
result = "Name: {}, Age: {}".format(name, age)
result = "Name: {n}, Age: {a}".format(n=name, a=age)

# f-string 的高级用法
price = 123.456
print(f"Price: {price:.2f}")  # Price: 123.46

nums = [1, 2, 3]
print(f"Sum: {sum(nums)}")    # 可以包含表达式
```

### 22. reversed() 倒序遍历

```python
nums = [1, 2, 3, 4, 5]

# ❌ 不够Pythonic
for i in range(len(nums)-1, -1, -1):
    print(nums[i])

# ✅ Pythonic (reversed)
for num in reversed(nums):
    print(num)

# ✅ enumerate + reversed
for i, num in enumerate(reversed(nums)):
    print(i, num)
```

### 23. is vs == 的区别

```python
# == 比较值是否相等
# is 比较是否是同一个对象(内存地址)

# ✅ Pythonic (None, True, False 使用 is)
if x is None:
    print("x is None")

if flag is True:
    print("flag is True")

# ❌ 不推荐
if x == None:
    pass

# ✅ 其他情况使用 ==
if name == "Alice":
    pass
```

### 24. extend vs append

```python
nums = [1, 2, 3]

# append: 添加单个元素(可以是列表)
nums.append([4, 5])
print(nums)  # [1, 2, 3, [4, 5]]

nums = [1, 2, 3]
# ✅ extend: 扩展多个元素
nums.extend([4, 5])
print(nums)  # [1, 2, 3, 4, 5]

# ✅ 更Pythonic (使用+=)
nums += [6, 7]
print(nums)  # [1, 2, 3, 4, 5, 6, 7]
```

### 25. 切片赋值

```python
nums = [1, 2, 3, 4, 5]

# ✅ Pythonic (切片赋值)
nums[1:3] = [20, 30]
print(nums)  # [1, 20, 30, 4, 5]

# 插入多个元素
nums[2:2] = [100, 200]
print(nums)  # [1, 20, 100, 200, 30, 4, 5]

# 删除多个元素
nums[1:3] = []
print(nums)  # [1, 200, 30, 4, 5]
```

### 26. sorted 的高级用法

```python
# 多关键字排序
students = [
    ("Alice", 20, 85),
    ("Bob", 19, 92),
    ("Charlie", 20, 78)
]

# ✅ 按年龄升序,成绩降序
sorted_students = sorted(students, key=lambda x: (x[1], -x[2]))

# ✅ 使用 operator.itemgetter (更快)
from operator import itemgetter
sorted_students = sorted(students, key=itemgetter(1, 2))

# ✅ 倒序
sorted_students = sorted(students, key=itemgetter(2), reverse=True)
```

### 27. min/max 的 key 参数

```python
words = ["apple", "banana", "cherry", "date"]

# ❌ 不够Pythonic
longest = ""
for word in words:
    if len(word) > len(longest):
        longest = word

# ✅ Pythonic (max with key)
longest = max(words, key=len)

# ✅ 找到字典中值最大的键
scores = {"Alice": 85, "Bob": 92, "Charlie": 78}
best_student = max(scores, key=scores.get)
print(best_student)  # Bob
```

### 28. 字典推导式高级用法

```python
# ✅ 过滤和转换
nums = [1, 2, 3, 4, 5]
square_dict = {n: n**2 for n in nums if n % 2 == 0}
print(square_dict)  # {2: 4, 4: 16}

# ✅ 从两个列表构建字典
keys = ["a", "b", "c"]
values = [1, 2, 3]
d = dict(zip(keys, values))
# 或使用推导式
d = {k: v for k, v in zip(keys, values)}
```

### 29. 集合运算

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# ✅ Pythonic (集合运算)
union = a | b          # 并集: {1, 2, 3, 4, 5, 6}
intersection = a & b   # 交集: {3, 4}
difference = a - b     # 差集: {1, 2}
sym_diff = a ^ b       # 对称差: {1, 2, 5, 6}

# 判断子集/超集
is_subset = {1, 2} <= a
is_superset = a >= {1, 2}
```

### 30. 统计和分组

```python
from collections import Counter

# ✅ Counter 统计频率
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
counter = Counter(words)
print(counter)  # Counter({'apple': 3, 'banana': 2, 'cherry': 1})
print(counter.most_common(2))  # [('apple', 3), ('banana', 2)]

# ✅ 字符串字符统计
s = "hello world"
char_count = Counter(s)
print(char_count)  # Counter({'l': 3, 'o': 2, ...})
```

---

## 🎯 在算法题中的应用

### 1. 链表反转 (序列解包)

```python
def reverseList(head):
    prev, curr = None, head
    while curr:
        curr.next, prev, curr = prev, curr, curr.next
    return prev
```

### 2. 两数之和 (enumerate + 字典)

```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
```

### 3. 最大子数组和 (链式赋值)

```python
def maxSubArray(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum
```

### 4. 有效的括号 (栈 + in)

```python
def isValid(s):
    stack = []
    pairs = {'(': ')', '[': ']', '{': '}'}

    for char in s:
        if char in pairs:
            stack.append(char)
        elif not stack or pairs[stack.pop()] != char:
            return False

    return not stack
```

### 5. 字母异位词 (sorted + Counter)

```python
from collections import Counter

def isAnagram(s, t):
    return Counter(s) == Counter(t)
    # 或者
    return sorted(s) == sorted(t)
```

### 6. 合并区间 (lambda排序)

```python
def merge(intervals):
    intervals.sort(key=lambda x: x[0])
    result = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])

    return result
```

### 7. 滑动窗口最大值 (双端队列)

```python
from collections import deque

def maxSlidingWindow(nums, k):
    dq = deque()
    result = []

    for i, num in enumerate(nums):
        # 移除超出窗口的元素
        if dq and dq[0] < i - k + 1:
            dq.popleft()

        # 维护递减队列
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        # 窗口形成后开始记录结果
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

### 8. 分组字母异位词 (defaultdict)

```python
from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())

# 使用
print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]
```

### 9. 前K个高频元素 (Counter + heapq)

```python
from collections import Counter
import heapq

def topKFrequent(nums, k):
    # ✅ Pythonic: Counter统计 + most_common
    return [num for num, _ in Counter(nums).most_common(k)]

# 或使用 heapq.nlargest
def topKFrequent2(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)
```

### 10. 斐波那契数列 (lru_cache 缓存)

```python
from functools import lru_cache

# ✅ Pythonic: 使用缓存避免重复计算
@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

print(fib(100))  # 秒出结果
```

### 11. 子集生成 (itertools.combinations)

```python
from itertools import combinations, chain

def subsets(nums):
    # ✅ Pythonic: 使用 combinations 生成所有子集
    return list(chain.from_iterable(
        combinations(nums, r) for r in range(len(nums) + 1)
    ))

print(subsets([1, 2, 3]))
# [(), (1,), (2,), (3,), (1,2), (1,3), (2,3), (1,2,3)]
```

### 12. 寻找第K大元素 (heapq.nlargest)

```python
import heapq

def findKthLargest(nums, k):
    # ✅ Pythonic: 一行代码
    return heapq.nlargest(k, nums)[-1]
```

### 13. 单词模式匹配 (zip + set)

```python
def wordPattern(pattern, s):
    words = s.split()
    # ✅ Pythonic: 使用 zip 和 set 检查一一对应关系
    return (len(pattern) == len(words) and
            len(set(zip(pattern, words))) == len(set(pattern)) == len(set(words)))

print(wordPattern("abba", "dog cat cat dog"))  # True
print(wordPattern("abba", "dog cat cat fish")) # False
```

### 14. 移除重复元素 (海象运算符)

```python
def removeDuplicates(nums):
    # ✅ Pythonic: 使用海象运算符 (Python 3.8+)
    seen = set()
    return [x for x in nums if x not in seen and not seen.add(x)]

# 注意: set.add() 返回 None (假值), not None 为 True
```

### 15. 区间合并 (sorted + 生成器)

```python
def merge(intervals):
    # ✅ Pythonic: sorted + 生成器
    intervals = sorted(intervals, key=lambda x: x[0])

    def merge_intervals():
        current = intervals[0]
        for interval in intervals[1:]:
            if interval[0] <= current[1]:
                current[1] = max(current[1], interval[1])
            else:
                yield current
                current = interval
        yield current

    return list(merge_intervals())
```

---

## 🔥 高级Pythonic技巧

### 1. 生成器表达式与生成器函数

```python
# 生成器表达式 (按需生成,节省内存)
squares = (x**2 for x in range(1000000))

# ✅ 生成器函数 (yield)
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 使用生成器
for num in fibonacci(10):
    print(num)  # 0, 1, 1, 2, 3, 5, 8, 13, 21, 34

# ✅ 生成器可以节省内存
# 读取大文件
def read_large_file(file_path):
    with open(file_path) as f:
        for line in f:  # 逐行读取,不会一次性加载到内存
            yield line.strip()
```

### 2. 海象运算符 := (Python 3.8+)

```python
# ❌ 不够Pythonic
match = re.search(pattern, text)
if match:
    print(match.group(1))

# ✅ Pythonic (海象运算符)
if match := re.search(pattern, text):
    print(match.group(1))

# ✅ 在列表推导式中使用
# 只计算一次复杂表达式
results = [y for x in data if (y := process(x)) > threshold]

# ✅ while 循环中
while (line := f.readline()):
    process(line)
```

### 3. 字典的 setdefault 和 defaultdict

```python
# ❌ 不够Pythonic
if key not in d:
    d[key] = []
d[key].append(value)

# ✅ Pythonic (setdefault)
d.setdefault(key, []).append(value)

# ✅ 更好 (defaultdict)
from collections import defaultdict
d = defaultdict(list)
d[key].append(value)  # 不需要检查key是否存在

# ✅ defaultdict 的其他用法
# 计数
counter = defaultdict(int)
for item in items:
    counter[item] += 1

# 分组
groups = defaultdict(list)
for item in items:
    groups[item.category].append(item)
```

### 4. functools 常用工具

```python
from functools import lru_cache, reduce, partial

# ✅ lru_cache: 缓存函数结果
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# ✅ reduce: 累积操作
from operator import add
total = reduce(add, [1, 2, 3, 4, 5])  # 15
# 等价于 sum([1, 2, 3, 4, 5])

# ✅ partial: 部分应用函数
def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
cube = partial(power, exp=3)

print(square(5))  # 25
print(cube(3))    # 27
```

### 5. operator 模块 (替代 lambda)

```python
from operator import itemgetter, attrgetter, methodcaller

# ✅ itemgetter (比 lambda 更快)
students = [("Alice", 85), ("Bob", 92), ("Charlie", 78)]

# lambda 方式
sorted_students = sorted(students, key=lambda x: x[1])

# ✅ itemgetter 方式 (更快,更清晰)
sorted_students = sorted(students, key=itemgetter(1))

# ✅ 多个键排序
sorted_students = sorted(students, key=itemgetter(1, 0))

# ✅ attrgetter (获取对象属性)
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

students = [Student("Alice", 85), Student("Bob", 92)]
sorted_students = sorted(students, key=attrgetter('score'))

# ✅ methodcaller (调用对象方法)
upper_list = list(map(methodcaller('upper'), ['hello', 'world']))
# ['HELLO', 'WORLD']
```

### 6. itertools 强大工具

```python
from itertools import (
    chain, combinations, permutations, product,
    groupby, accumulate, islice, cycle, repeat
)

# ✅ chain: 连接多个可迭代对象
list(chain([1, 2], [3, 4], [5]))  # [1, 2, 3, 4, 5]

# ✅ combinations: 组合
list(combinations([1, 2, 3], 2))  # [(1,2), (1,3), (2,3)]

# ✅ permutations: 排列
list(permutations([1, 2, 3], 2))  # [(1,2), (1,3), (2,1), (2,3), (3,1), (3,2)]

# ✅ product: 笛卡尔积
list(product([1, 2], ['a', 'b']))  # [(1,'a'), (1,'b'), (2,'a'), (2,'b')]

# ✅ groupby: 分组
data = [('A', 1), ('A', 2), ('B', 3), ('B', 4)]
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))
# A [('A', 1), ('A', 2)]
# B [('B', 3), ('B', 4)]

# ✅ accumulate: 累积
list(accumulate([1, 2, 3, 4]))  # [1, 3, 6, 10]

# ✅ islice: 切片迭代器
list(islice(range(10), 2, 8, 2))  # [2, 4, 6]

# ✅ cycle: 无限循环
from itertools import cycle
counter = cycle([1, 2, 3])
# 无限生成: 1, 2, 3, 1, 2, 3, ...
```

### 7. *args 和 **kwargs

```python
# ✅ *args: 可变位置参数
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4))  # 10

# ✅ **kwargs: 可变关键字参数
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=20, city="Beijing")

# ✅ 组合使用
def func(a, b, *args, key1=None, **kwargs):
    print(f"a={a}, b={b}")
    print(f"args={args}")
    print(f"key1={key1}")
    print(f"kwargs={kwargs}")

func(1, 2, 3, 4, key1="value1", key2="value2")
# a=1, b=2
# args=(3, 4)
# key1=value1
# kwargs={'key2': 'value2'}

# ✅ 解包字典作为函数参数
params = {"name": "Alice", "age": 20}
print_info(**params)
```

### 8. 上下文管理器 (contextlib)

```python
from contextlib import contextmanager

# ✅ 自定义上下文管理器
@contextmanager
def timer(name):
    import time
    start = time.time()
    print(f"{name} started")
    yield
    end = time.time()
    print(f"{name} finished in {end-start:.2f}s")

# 使用
with timer("Process"):
    # 你的代码
    time.sleep(1)

# ✅ suppress: 忽略特定异常
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("non_existent_file.txt")  # 不会抛出异常
```

### 9. ChainMap: 链式字典

```python
from collections import ChainMap

# ✅ ChainMap: 合并多个字典视图
defaults = {"color": "red", "user": "guest"}
custom = {"user": "admin"}

# 优先使用 custom 中的值
settings = ChainMap(custom, defaults)
print(settings["user"])   # admin
print(settings["color"])  # red

# 修改只影响第一个字典
settings["theme"] = "dark"
print(custom)  # {'user': 'admin', 'theme': 'dark'}
print(defaults)  # {'color': 'red', 'user': 'guest'}
```

### 10. bisect: 二分查找模块

```python
import bisect

# ✅ bisect_left: 找到插入位置
nums = [1, 3, 5, 7, 9]
pos = bisect.bisect_left(nums, 6)
print(pos)  # 3

# ✅ insort: 保持有序插入
bisect.insort(nums, 6)
print(nums)  # [1, 3, 5, 6, 7, 9]

# ✅ 实际应用: 根据分数判定等级
def get_grade(score):
    breakpoints = [60, 70, 80, 90]
    grades = ['F', 'D', 'C', 'B', 'A']
    index = bisect.bisect(breakpoints, score)
    return grades[index]

print(get_grade(85))  # B
```

### 11. 链式函数调用

```python
# ✅ 字符串处理链
result = s.strip().lower().replace(' ', '_')

# ✅ Pandas 风格的链式调用
class QueryBuilder:
    def __init__(self, data):
        self.data = data

    def filter(self, condition):
        self.data = [x for x in self.data if condition(x)]
        return self  # 返回self支持链式调用

    def map(self, func):
        self.data = [func(x) for x in self.data]
        return self

    def collect(self):
        return self.data

# 使用
result = (QueryBuilder([1, 2, 3, 4, 5])
    .filter(lambda x: x > 2)
    .map(lambda x: x * 2)
    .collect())  # [6, 8, 10]
```

---

## 📋 Pythonic 代码检查清单

### 🔍 基础检查 (必须掌握)

在写代码时,问自己这些问题:

✅ 是否使用了**列表推导式**而不是 `for` 循环?
✅ 是否使用了 `enumerate()` 而不是 `range(len())`?
✅ 是否使用了 `zip()` 来并行遍历?
✅ 是否使用了 `in` 来检查成员关系?
✅ 是否使用了**三元表达式**简化 `if-else`?
✅ 是否使用了字典的 `get()` 方法避免 `KeyError`?
✅ 是否使用了 `with` 语句管理资源?
✅ 是否使用了 `any()` 和 `all()` 简化判断?
✅ 是否充分利用了**切片**操作 (如 `[::-1]` 反转)?
✅ 是否使用了**序列解包** (如 `a, b = b, a`)?

### 🚀 进阶检查 (推荐使用)

✅ 是否使用了 **f-string** 进行字符串格式化?
✅ 是否使用了 `reversed()` 进行倒序遍历?
✅ 是否对 `None` 使用 `is` 而不是 `==`?
✅ 是否使用了 `extend()` 或 `+=` 而不是循环 `append()`?
✅ 是否使用了 `sorted()` 的 `key` 参数?
✅ 是否使用了 `min()`/`max()` 的 `key` 参数?
✅ 是否使用了**集合运算** (`|`, `&`, `-`, `^`)?
✅ 是否使用了 `Counter` 进行频率统计?
✅ 是否使用了 `defaultdict` 避免检查键是否存在?
✅ 是否使用了 `join()` 而不是循环拼接字符串?

### 💎 高级检查 (算法优化)

✅ 是否可以使用**生成器**节省内存?
✅ 是否可以使用**海象运算符** `:=` (Python 3.8+)?
✅ 是否可以使用 `@lru_cache` 缓存函数结果?
✅ 是否使用了 `operator` 模块替代 `lambda`?
✅ 是否使用了 `itertools` 的强大工具?
✅ 是否使用了 `heapq.nlargest`/`nsmallest`?
✅ 是否使用了 `bisect` 进行二分操作?
✅ 是否使用了 `*args` 和 `**kwargs`?
✅ 是否可以使用上下文管理器简化代码?
✅ 是否可以使用 `chain`/`groupby` 处理可迭代对象?

---

## ⚠️ 常见反模式 (避免这样写)

### 1. 不必要的列表遍历

```python
# ❌ 反模式
result = []
for i in range(len(nums)):
    result.append(nums[i] * 2)

# ✅ Pythonic
result = [num * 2 for num in nums]
```

### 2. 手动初始化字典键

```python
# ❌ 反模式
word_count = {}
for word in words:
    if word not in word_count:
        word_count[word] = 0
    word_count[word] += 1

# ✅ Pythonic (使用 Counter)
from collections import Counter
word_count = Counter(words)

# ✅ Pythonic (使用 defaultdict)
from collections import defaultdict
word_count = defaultdict(int)
for word in words:
    word_count[word] += 1
```

### 3. 循环中的字符串拼接

```python
# ❌ 反模式 (效率低,每次都创建新字符串)
result = ""
for word in words:
    result += word + " "

# ✅ Pythonic
result = " ".join(words)
```

### 4. 不使用生成器处理大数据

```python
# ❌ 反模式 (一次性加载到内存)
def read_file(filename):
    with open(filename) as f:
        return [line.strip() for line in f]

# ✅ Pythonic (使用生成器)
def read_file(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()
```

### 5. 忽略内置函数

```python
# ❌ 反模式
max_val = nums[0]
for num in nums[1:]:
    if num > max_val:
        max_val = num

# ✅ Pythonic
max_val = max(nums)
```

### 6. 重复计算

```python
# ❌ 反模式 (重复计算斐波那契)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

# ✅ Pythonic (使用缓存)
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)
```

### 7. 不必要的类型转换

```python
# ❌ 反模式
if len(my_list) > 0:
    do_something()

# ✅ Pythonic (利用真值测试)
if my_list:
    do_something()
```

### 8. 手动实现已有功能

```python
# ❌ 反模式 (手动实现排序查找)
def find_kth_largest(nums, k):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] < nums[j]:
                nums[i], nums[j] = nums[j], nums[i]
    return nums[k-1]

# ✅ Pythonic (使用 heapq)
import heapq
def find_kth_largest(nums, k):
    return heapq.nlargest(k, nums)[-1]
```

### 9. 忽略解包特性

```python
# ❌ 反模式
point = (10, 20)
x = point[0]
y = point[1]

# ✅ Pythonic
x, y = point
```

### 10. 不使用上下文管理器

```python
# ❌ 反模式
lock.acquire()
try:
    # 操作
    pass
finally:
    lock.release()

# ✅ Pythonic
with lock:
    # 操作
    pass
```

---

## 🎯 性能对比

### 字符串拼接性能

```python
import timeit

# 循环拼接 (慢)
def concat_loop():
    result = ""
    for i in range(1000):
        result += str(i)
    return result

# join (快 10-100 倍)
def concat_join():
    return "".join(str(i) for i in range(1000))

print(timeit.timeit(concat_loop, number=1000))  # ~0.15s
print(timeit.timeit(concat_join, number=1000))  # ~0.015s
```

### 列表推导式 vs 循环

```python
# 循环 append
def loop_append():
    result = []
    for i in range(10000):
        result.append(i * 2)
    return result

# 列表推导式 (快 30% 左右)
def list_comp():
    return [i * 2 for i in range(10000)]

print(timeit.timeit(loop_append, number=1000))  # ~0.6s
print(timeit.timeit(list_comp, number=1000))    # ~0.4s
```

### set 查找 vs list 查找

```python
big_list = list(range(10000))
big_set = set(range(10000))

# list 查找: O(n)
timeit.timeit(lambda: 9999 in big_list, number=10000)  # ~0.5s

# set 查找: O(1) (快 1000+ 倍)
timeit.timeit(lambda: 9999 in big_set, number=10000)   # ~0.0005s
```

---

## 🎓 小结

### Pythonic 代码的五大原则

✅ **简洁性**: 用更少的代码表达更多的意思
   - 列表推导式 > 循环 append
   - `enumerate/zip` > `range(len())`
   - 内置函数 > 手动实现

✅ **可读性**: 代码要清晰易懂,不要过度聪明
   - f-string > 字符串拼接
   - `if x in [1,2,3]` > `if x==1 or x==2 or x==3`
   - 有意义的变量名 > 单字母变量

✅ **惯用法**: 遵循 Python 社区的最佳实践
   - `with` 管理资源
   - `is None` 不用 `== None`
   - 序列解包 `a, b = b, a`

✅ **效率**: 选择合适的数据结构和算法
   - `set` 查找 > `list` 查找
   - `join()` 拼接 > 循环 `+=`
   - `defaultdict` > 手动初始化

✅ **显式优于隐式**: 代码应该明确表达意图
   - 类型提示 (type hints)
   - 清晰的函数/变量命名
   - 适当的注释说明

### 学习路径建议

1. **基础阶段**: 掌握前 20 个常规用法
2. **进阶阶段**: 学习 collections、itertools 等标准库
3. **高级阶段**: 掌握生成器、装饰器、上下文管理器
4. **实践阶段**: 在 LeetCode 题目中应用这些技巧

### Pythonic 编程格言

> "There should be one-- and preferably only one --obvious way to do it."
>
> "Simple is better than complex."
>
> "Readability counts."
>
> -- The Zen of Python

**核心原则**: 写出让其他 Python 程序员一眼就能看懂的代码!

---

## 🔗 相关章节

- [11-推导式.md](./11-推导式.md) - 列表/字典/集合推导式
- [12-内置函数.md](./12-内置函数.md) - Python内置函数大全
- [16-高级特性.md](./16-高级特性.md) - lambda、切片、解包等
- [13-collections模块.md](./13-collections模块.md) - 高级数据结构

**延伸学习**: PEP 8 (Python代码风格指南)

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
