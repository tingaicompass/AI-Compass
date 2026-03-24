> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第34课:最小栈

> **模块**:栈与队列 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/min-stack/
> **前置知识**:第33课(有效的括号)
> **预计学习时间**:20分钟

---

## 🎯 题目描述

设计一个支持以下操作的栈,并且所有操作的时间复杂度都要求是 **O(1)**:

- `push(x)` — 将元素 x 推入栈中
- `pop()` — 删除栈顶元素
- `top()` — 获取栈顶元素
- `getMin()` — 检索栈中的最小元素

**示例:**
```
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin();   --> 返回 -3
minStack.pop();
minStack.top();      --> 返回 0
minStack.getMin();   --> 返回 -2
```

**约束条件:**
- `-2³¹ ≤ val ≤ 2³¹ - 1`
- pop、top 和 getMin 操作总是在非空栈上调用
- **关键约束**:所有操作必须在 O(1) 时间内完成

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单元素 | push(5), getMin() | 5 | 基本功能 |
| 重复最小值 | push(1), push(1), pop(), getMin() | 1 | 最小值重复处理 |
| 递减序列 | push(3), push(2), push(1) | getMin()=1, pop(), getMin()=2 | 最小值动态更新 |
| 先减后增 | push(0), push(1), push(0) | getMin()=0 | 最小值可能不在栈顶 |
| 负数 | push(-5), push(-10), getMin() | -10 | 负数处理 |
| 大规模 | 3×10⁴ 次操作 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在玩"叠罗汉"游戏,需要随时知道当前最矮的人是谁。
>
> 🐌 **笨办法**:每次查询最矮的人时,让所有人重新比一遍身高 → 需要 O(n) 时间,太慢!
>
> 🚀 **聪明办法**:每来一个新人站上去时,就在他身上贴个小纸条,写着"到目前为止最矮是XXcm"。这样查询时直接看栈顶那张纸条就行,O(1) 秒杀!
>
> 这就是**同步维护最小值信息**的核心思想 — 在数据入栈时就"顺手"记录当前最小值,而不是需要时再去找。

### 关键洞察
**空间换时间:用额外的存储空间(辅助栈或元组)来换取 O(1) 的查询速度**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:一系列 push/pop/top/getMin 操作
- **输出**:对应的返回值
- **核心限制**:所有操作必须 O(1) 时间复杂度
- **难点**:普通栈的 getMin() 需要遍历所有元素找最小值,是 O(n)

### Step 2:先想笨办法(暴力法)
最直接的思路:用一个普通列表当栈,getMin() 时遍历整个栈找最小值。
- 时间复杂度:push/pop/top 都是 O(1),但 **getMin() 是 O(n)**
- 瓶颈在哪:每次 getMin() 都要扫描所有元素,不满足题目要求

### Step 3:找优化突破口
**核心问题**:如何让 getMin() 也变成 O(1)?

**关键发现**:
1. 栈的最小值会随着 push/pop 动态变化
2. 但是,在任意时刻,栈中元素的最小值是**确定的**
3. 如果我们能在每次 push 时**同步记录当前最小值**,查询时就不用重新计算了

**优化方向**:
- **方案1**:额外维护一个"最小值栈",与主栈同步更新
- **方案2**:在主栈的每个元素上"附加"当前最小值信息(元组)

### Step 4:确定最优解法
两种方案的时间复杂度都是 O(1),空间都是 O(n)。
选择**辅助栈法**作为最优解,因为代码更清晰,职责分离明确。

---

## 🔑 解法一:暴力遍历(不推荐)

### 💡 核心思想
用普通列表实现栈,getMin() 时遍历找最小值。

### 📝 代码实现
```python
class MinStack:
    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return min(self.stack)  # O(n) 遍历


# 测试
minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin())  # 输出: -3
minStack.pop()
print(minStack.top())     # 输出: 0
print(minStack.getMin())  # 输出: -2
```

### 📊 复杂度分析
- **时间复杂度**:
  - push/pop/top: O(1)
  - getMin: **O(n)** ← 不满足题目要求
- **空间复杂度**:O(n)

### ⚠️ 为什么不推荐
虽然代码简单,但 getMin() 的 O(n) 复杂度不符合题目要求,**面试直接不通过**。

---

## ⚡ 解法二:元组栈(优化)

### 💡 核心思想
在栈的每个元素上"携带"当前最小值信息,把每个元素存储为 `(value, current_min)` 元组。

### 📊 图解演示
```
操作序列: push(-2) → push(0) → push(-3) → getMin() → pop()

Step 1: push(-2)
栈: [(-2, -2)]
    └──┬──┘
       值  当前最小值

Step 2: push(0)
比较: min(-2, 0) = -2
栈: [(-2, -2), (0, -2)]

Step 3: push(-3)
比较: min(-2, -3) = -3
栈: [(-2, -2), (0, -2), (-3, -3)]

Step 4: getMin()
直接返回栈顶元组的第二个值: -3  ← O(1)

Step 5: pop()
弹出 (-3, -3)
栈: [(-2, -2), (0, -2)]
getMin() → -2  ← 自动更新!
```

### 📝 代码实现
```python
class MinStack:
    def __init__(self):
        self.stack = []  # 存储 (val, current_min) 元组

    def push(self, val: int) -> None:
        if not self.stack:
            # 栈空时,最小值就是自己
            self.stack.append((val, val))
        else:
            # 新的最小值 = min(当前值, 之前的最小值)
            current_min = min(val, self.stack[-1][1])
            self.stack.append((val, current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]  # 返回元组的第一个值

    def getMin(self) -> int:
        return self.stack[-1][1]  # 返回元组的第二个值


# 测试
minStack = MinStack()
minStack.push(-2)
minStack.push(0)
minStack.push(-3)
print(minStack.getMin())  # 输出: -3
minStack.pop()
print(minStack.top())     # 输出: 0
print(minStack.getMin())  # 输出: -2
```

### 📊 复杂度分析
- **时间复杂度**:所有操作都是 O(1) ✅
- **空间复杂度**:O(n) — 每个元素存储两个值

### ✅ 优点
- 所有操作都满足 O(1) 要求
- 代码简洁,只用一个栈
- 逻辑清晰,每个元素"自带"最小值信息

### ⚠️ 缺点
- 空间冗余:即使最小值不变,也要在每个元素上重复存储

---

## 🏆 解法三:辅助栈(最优解)

### 💡 核心思想
维护两个栈:
- **主栈**:存储所有元素
- **最小值栈**:只存储"当前最小值"的历史记录

两个栈**同步更新**,最小值栈的栈顶始终是当前全局最小值。

### 📊 图解演示
```
操作序列: push(-2) → push(0) → push(-3) → getMin() → pop() → getMin()

Step 1: push(-2)
主栈: [-2]
最小栈: [-2]  ← -2 是当前最小值

Step 2: push(0)
0 > -2,最小值不变
主栈: [-2, 0]
最小栈: [-2, -2]  ← 重复压入 -2,保持同步

Step 3: push(-3)
-3 < -2,新最小值!
主栈: [-2, 0, -3]
最小栈: [-2, -2, -3]  ← 压入新最小值

Step 4: getMin()
直接返回最小栈栈顶: -3  ← O(1)

Step 5: pop()
主栈弹出 -3,最小栈也弹出
主栈: [-2, 0]
最小栈: [-2, -2]  ← 自动回退到上一个最小值

Step 6: getMin()
返回最小栈栈顶: -2  ← 最小值自动更新!
```

### 📝 代码实现
```python
class MinStack:
    def __init__(self):
        self.stack = []      # 主栈:存储所有元素
        self.min_stack = []  # 最小栈:同步记录当前最小值

    def push(self, val: int) -> None:
        self.stack.append(val)

        # 更新最小栈
        if not self.min_stack:
            # 最小栈为空,直接压入
            self.min_stack.append(val)
        else:
            # 压入 min(新值, 当前最小值)
            self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()  # 同步弹出

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]  # O(1) 获取最小值


# 完整测试用例
def test_min_stack():
    minStack = MinStack()

    # 测试1:基本操作
    minStack.push(-2)
    minStack.push(0)
    minStack.push(-3)
    assert minStack.getMin() == -3, "应该返回 -3"

    minStack.pop()
    assert minStack.top() == 0, "应该返回 0"
    assert minStack.getMin() == -2, "应该返回 -2"

    # 测试2:重复最小值
    minStack.push(-2)
    assert minStack.getMin() == -2
    minStack.pop()
    assert minStack.getMin() == -2  # 还有一个 -2

    print("✅ 所有测试通过!")

test_min_stack()
```

### 📊 复杂度分析
- **时间复杂度**:
  - push: O(1) — 两个栈各压入一次
  - pop: O(1) — 两个栈各弹出一次
  - top: O(1) — 直接访问主栈栈顶
  - getMin: O(1) — 直接访问最小栈栈顶
- **空间复杂度**:O(n) — 需要额外的最小栈,最坏情况每个元素都存

### ✅ 为什么是最优解
1. **时间最优**:所有操作都是严格 O(1),没有任何遍历
2. **逻辑清晰**:职责分离,主栈存数据,最小栈存元数据
3. **面试友好**:代码简洁,容易在白板上写对
4. **可扩展**:如果要支持 getMax(),只需再加一个 max_stack

---

## 🐍 Pythonic 写法

### 技巧1:初始化时处理哨兵
```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = [float('inf')]  # 哨兵值,简化边界判断

    def push(self, val: int) -> None:
        self.stack.append(val)
        self.min_stack.append(min(val, self.min_stack[-1]))

    def pop(self) -> None:
        self.stack.pop()
        self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```
**优势**:push 时不需要判断 min_stack 是否为空,代码更简洁。

### 技巧2:压缩最小栈(进阶)
```python
class MinStack:
    """只在最小值变化时才压入最小栈,节省空间"""
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.stack.append(val)
        # 只在 val ≤ 当前最小值时才压入最小栈
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self) -> None:
        val = self.stack.pop()
        # 只在弹出的是最小值时才弹出最小栈
        if val == self.min_stack[-1]:
            self.min_stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.min_stack[-1]
```
**优势**:最小值变化不频繁时,空间可能节省 50%+。

---

## 📊 解法对比

| 维度 | 解法一:暴力遍历 | 解法二:元组栈 | 🏆 解法三:辅助栈(最优) |
|------|--------------|-------------|---------------------|
| 时间复杂度 | getMin O(n) ❌ | **O(1)** ✅ | **O(1)** ✅ ← 时间最优 |
| 空间复杂度 | O(n) | O(n) | O(n) |
| 代码复杂度 | 简单 | 中等 | **简单** ← 最易实现 |
| 面试推荐 | ❌ | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 不满足题目要求 | 可用,稍冗余 | **面试标准答案** |

### 🏆 为什么辅助栈是最优解
1. **职责清晰**:主栈负责数据,最小栈负责元数据,符合单一职责原则
2. **面试友好**:代码逻辑直观,容易解释给面试官听
3. **可扩展**:如果要同时支持 getMax(),只需并行维护 max_stack
4. **无冗余**:虽然元组栈也是 O(1),但每个元素都存两份数据,不够优雅

### 💡 面试建议
1. **起手式**:先说暴力法(遍历找最小值),表明你理解问题 ← 30秒
2. **快速优化**:提出"能否在 push 时就记录最小值"的关键洞察 ← 10秒
3. **🏆 重点讲解**:详细说明辅助栈法,画出两个栈的同步过程 ← 3分钟
4. **写代码**:直接写最优解(辅助栈),边写边解释 push/pop 的同步逻辑
5. **测试用例**:手动测试包含重复最小值的用例(如 [1,1,2])
6. **拓展讨论**:提到空间优化(只在最小值变化时压栈),展示深度思考

---

## 🎤 面试现场模拟

**面试官**:"设计一个栈,能在 O(1) 时间获取最小值。"

**你**:"明白。我先想想最直接的做法 ← 展示思考过程

**你**:"最简单的方法是用列表实现栈,getMin 时遍历找最小值。但这样 getMin 是 O(n),不符合要求。" ← 排除暴力法

**你**:"关键洞察是:**能否在 push 时就记录当前最小值,查询时直接用?**" ← 点出核心

**你**:"我想到两种方案:
1. 元组栈:每个元素存 (值, 当前最小值)
2. 辅助栈:额外维护一个最小值栈

两者时间都是 O(1),我选辅助栈,因为职责更清晰。" ← 对比方案

**你** *(开始画图)*:
```
push(-2):  主栈[-2]  最小栈[-2]
push(0):   主栈[-2,0]  最小栈[-2,-2]  ← 0>-2,重复压-2
push(-3):  主栈[-2,0,-3]  最小栈[-2,-2,-3]  ← 新最小值
getMin():  返回最小栈栈顶 = -3
pop():     两个栈同步弹出,最小栈自动回退到-2
```
← 图解关键步骤

**面试官**:"如果有很多重复的最小值,会浪费空间吗?"

**你**:"好问题!可以优化:只在值 ≤ 当前最小值时才压入最小栈,pop 时判断是否需要弹出最小栈。" ← 展示优化思路

**面试官**:"时间复杂度怎么保证?"

**你**:"所有操作都是栈顶操作(append/pop/[-1]),Python 列表的这些操作都是均摊 O(1)。" ← 底层实现认知

---

## ❓ 高频追问

| 追问 | 标准回答 |
|------|---------|
| 为什么不用单个变量记录最小值? | 因为 pop 时无法恢复上一个最小值。比如 push(1), push(2), pop(),最小值应该回退到 1,单变量做不到。 |
| 元组栈和辅助栈哪个更好? | 时间空间都是 O(1)/O(n),但辅助栈职责更清晰,面试更推荐。元组栈在最小值不变时会重复存储。 |
| 能否只在最小值变化时才压栈? | 可以!push 时判断 `val <= min_stack[-1]` 才压入,pop 时判断 `val == min_stack[-1]` 才弹出。节省空间但代码稍复杂。 |
| 空间复杂度能优化到 O(1) 吗? | 不能。O(1) 空间意味着只能用常数个变量,但栈的深度是 n,需要记录 n 个历史最小值。 |
| 如果同时要 getMax() 呢? | 并行维护一个 max_stack,逻辑完全对称。 |
| Python 列表的 append/pop 是 O(1) 吗? | 是均摊 O(1)。底层用动态数组,偶尔需要扩容(O(n)),但摊还下来每次操作是 O(1)。 |

---

## 🐍 Python 技巧卡片

### 1. 列表作为栈
```python
stack = []
stack.append(x)   # 入栈 O(1)
stack.pop()       # 出栈 O(1)
stack[-1]         # 栈顶 O(1)
len(stack) == 0   # 判空
```

### 2. 哨兵值简化边界
```python
min_stack = [float('inf')]  # 初始哨兵
# push 时不需要判断 if not min_stack
min_stack.append(min(val, min_stack[-1]))
```

### 3. 元组解包
```python
stack = [(1, 2), (3, 4)]
val, min_val = stack[-1]  # 直接解包
```

---

## 🔬 底层原理

### Python 列表的栈操作为什么是 O(1)?

Python 的 `list` 底层是**动态数组**(类似 C++ 的 vector):
```
内存布局:
[元素1|元素2|元素3|...|预留空间]
                      ↑
                   栈顶指针
```

1. **append(x)**:
   - 直接在栈顶指针位置写入 x,指针+1
   - 如果没有预留空间,触发扩容(申请 2 倍大小的新数组,拷贝数据)
   - 虽然扩容是 O(n),但发生频率是 1/n,**均摊 O(1)**

2. **pop()**:
   - 直接返回栈顶元素,指针-1
   - 不需要移动其他元素,严格 O(1)

3. **[-1] 访问**:
   - 数组支持随机访问,直接用指针定位,O(1)

### 为什么不用链表实现栈?
虽然链表的 push/pop 也是 O(1),但:
- 每个节点需要额外存储指针(8 字节),空间浪费
- 内存不连续,缓存命中率低
- Python 列表(动态数组)的均摊 O(1) 性能已足够好

---

## 📋 算法模式卡片

**模式名称**:同步辅助栈

**适用场景**:
- 需要在 O(1) 时间获取栈的某种**全局属性**(最小值/最大值/众数)
- 属性会随着栈的变化而动态更新

**核心思想**:
用一个辅助栈**同步记录**每个时刻的全局属性,主栈和辅助栈一起 push/pop。

**模板代码**:
```python
class SpecialStack:
    def __init__(self):
        self.stack = []        # 主栈
        self.property_stack = []  # 属性栈(如最小值栈)

    def push(self, val):
        self.stack.append(val)
        # 更新属性栈(取决于具体属性)
        new_property = self._compute_property(val)
        self.property_stack.append(new_property)

    def pop(self):
        self.stack.pop()
        self.property_stack.pop()  # 同步弹出

    def get_property(self):
        return self.property_stack[-1]  # O(1) 获取
```

**变体题目**:
- LC 155:最小栈(本题)
- 最大栈:维护 max_stack
- 中位数栈:维护两个堆(较难)

---

## ⚠️ 易错点

### 1. 忘记同步弹出最小栈
```python
# ❌ 错误
def pop(self):
    self.stack.pop()
    # 忘记 self.min_stack.pop()
```
**后果**:最小栈和主栈长度不一致,getMin() 返回错误值。

### 2. 最小栈初始化错误
```python
# ❌ 错误
def push(self, val):
    self.min_stack.append(min(val, self.min_stack[-1]))
    # 如果 min_stack 为空,[-1] 会报错
```
**正确做法**:push 前判断 `if not self.min_stack`,或初始化哨兵值。

### 3. 边界判断顺序错误
```python
# ❌ 错误(压缩版最小栈)
def push(self, val):
    self.stack.append(val)
    if val <= self.min_stack[-1]:  # min_stack 可能为空!
        self.min_stack.append(val)

# ✅ 正确
def push(self, val):
    self.stack.append(val)
    if not self.min_stack or val <= self.min_stack[-1]:
        self.min_stack.append(val)
```

### 4. 误用 `<` 而非 `<=`
```python
# ❌ 错误(压缩版)
if val < self.min_stack[-1]:  # 应该用 <=
    self.min_stack.append(val)
```
**反例**:`push(1), push(1), pop()` → 第二个 1 不会被压入最小栈,pop 时会错误弹出最小栈,导致 getMin() 失败。

---

## 🏗️ 工程实战(选读)

### 场景1:浏览器的"撤销"功能
**需求**:支持撤销操作,并实时显示"当前页面的最早访问时间"。

```python
class BrowserHistory:
    def __init__(self):
        self.history = []       # 存储 (url, timestamp)
        self.min_time_stack = []  # 记录当前最早时间

    def visit(self, url, timestamp):
        self.history.append((url, timestamp))
        if not self.min_time_stack:
            self.min_time_stack.append(timestamp)
        else:
            self.min_time_stack.append(
                min(timestamp, self.min_time_stack[-1])
            )

    def back(self):
        self.history.pop()
        self.min_time_stack.pop()

    def get_earliest_time(self):
        return self.min_time_stack[-1]  # O(1)
```

### 场景2:股票交易的"历史最低价"监控
**需求**:实时显示从开盘到当前的最低价。

```python
class StockMonitor:
    def __init__(self):
        self.prices = []       # 价格序列
        self.min_price_stack = []  # 历史最低价

    def record_price(self, price):
        self.prices.append(price)
        if not self.min_price_stack:
            self.min_price_stack.append(price)
        else:
            self.min_price_stack.append(
                min(price, self.min_price_stack[-1])
            )

    def get_historical_low(self):
        return self.min_price_stack[-1]
```

---

## 🏋️ 举一反三

### 相关题目

| 题目 | 难度 | 关键区别 |
|------|------|---------|
| **LC 716** - 最大栈 | Hard | 在最小栈基础上增加 popMax() 操作,需要用两个栈或栈+堆 |
| **LC 895** - 最大频率栈 | Hard | 返回出现频率最高的元素,需要维护频率哈希表+多个栈 |
| **剑指 Offer 30** - 包含 min 函数的栈 | Easy | 和本题完全相同 |

### 练习建议
1. **先做** LC 155(本题),掌握辅助栈的核心思想
2. **再做** LC 716,理解如何支持"删除最大值"(需要从栈中间删除)
3. **挑战** LC 895,综合运用哈希表和多栈

---

## 📝 课后小测

<details>
<summary>💡 点击查看提示</summary>

**题目**:如果栈的操作变成:
- push(x):压入 x
- pop():弹出栈顶
- getSecondMin():获取栈中**第二小**的元素

如何修改辅助栈法?时间复杂度是多少?

**提示**:
- 是否需要两个辅助栈?
- 如何处理最小值和第二小值相等的情况?

</details>

<details>
<summary>✅ 点击查看答案</summary>

**答案**:需要维护两个辅助栈 `min_stack` 和 `second_min_stack`。

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
        self.second_min_stack = []

    def push(self, val):
        self.stack.append(val)

        # 更新最小值栈
        if not self.min_stack:
            self.min_stack.append(val)
            self.second_min_stack.append(float('inf'))
        else:
            current_min = min(val, self.min_stack[-1])
            self.min_stack.append(current_min)

            # 更新第二小值栈
            if val < self.min_stack[-2]:
                # 新值比之前的最小值还小
                second = self.min_stack[-2]
            elif val == self.min_stack[-2]:
                # 新值等于之前的最小值
                second = self.second_min_stack[-1]
            else:
                # 新值 > 最小值
                second = min(val, self.second_min_stack[-1])
            self.second_min_stack.append(second)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()
        self.second_min_stack.pop()

    def getMin(self):
        return self.min_stack[-1]

    def getSecondMin(self):
        return self.second_min_stack[-1]
```

**复杂度**:所有操作仍然是 O(1) 时间,空间 O(n)。

**核心难点**:
1. 当新值成为最小值时,第二小值变成"之前的最小值"
2. 需要同时维护两个辅助栈的同步更新逻辑

</details>

---

**恭喜你完成第 34 课!** 🎉

你已经掌握了:
- ✅ 辅助栈的核心思想:同步维护全局属性
- ✅ 空间换时间的经典应用
- ✅ 面试中如何快速从暴力法优化到最优解
- ✅ Python 列表作为栈的底层原理

**下一课预告**:第 35 课 - 每日温度(单调栈的经典应用) 🌡️

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
