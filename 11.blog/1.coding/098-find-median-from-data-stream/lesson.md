> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第98课:数据流中位数

> **模块**:堆与优先队列 | **难度**:Hard ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/find-median-from-data-stream/
> **前置知识**:第97课(前K个高频元素)、堆的基础操作
> **预计学习时间**:35分钟

---

## 🎯 题目描述

设计一个数据结构,支持以下两个操作:

1. `addNum(num)`:从数据流中添加一个整数到数据结构
2. `findMedian()`:返回当前所有元素的中位数

**示例:**
```
MedianFinder medianFinder = new MedianFinder()
medianFinder.addNum(1)
medianFinder.addNum(2)
medianFinder.findMedian() → 1.5
medianFinder.addNum(3)
medianFinder.findMedian() → 2.0
```

**约束条件:**
- 数据流中元素的数量可能非常大
- 需要在添加元素后快速查询中位数
- 中位数定义:如果元素总数为奇数,返回中间元素;偶数则返回中间两数的平均值

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单个元素 | addNum(5), findMedian() | 5.0 | 基本功能 |
| 两个元素 | addNum(1), addNum(2), findMedian() | 1.5 | 偶数个处理 |
| 奇数个元素 | 连续添加1,2,3,findMedian() | 2.0 | 奇数个处理 |
| 重复元素 | addNum(1), addNum(1) | 1.0 | 重复值处理 |
| 负数 | addNum(-1), addNum(-2) | -1.5 | 负数处理 |
| 大规模 | 添加10⁵个元素 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻

> 想象你是一个体育老师,正在记录学生的百米冲刺成绩。每跑完一个学生,家长就会问:"现在成绩的中位数是多少?"
>
> 🐌 **笨办法**:每次都把所有成绩本子拿出来重新排序,然后找中间的成绩。如果已经有1000个学生跑完了,又来一个学生,你得把1000个成绩重新排一遍,太慢了!
>
> 🚀 **聪明办法**:准备两个本子——"跑得快的一半"和"跑得慢的一半",保证快的一半里最慢的那个成绩刚好能和慢的一半里最快的那个成绩接上。这样每次只需要看这两个本子的最上面那个成绩,瞬间就能找到中位数!这就是"对顶堆"思想。

### 关键洞察

**中位数是"左边一半"和"右边一半"的分界点。我们用两个堆维护这两部分,就能在O(1)时间查询中位数!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出

- **输入**:动态添加的整数流,数量未知
- **输出**:`addNum`无返回值,`findMedian`返回浮点数中位数
- **限制**:需要在动态数据流中快速维护中位数,不能每次都重新排序

### Step 2:先想笨办法(暴力法)

最直接的想法:用一个列表存所有数字,每次调用`findMedian`时排序后取中位数。
- 时间复杂度:每次`findMedian`需要O(n log n)排序
- 瓶颈在哪:频繁调用`findMedian`时,每次都要重新排序整个数组

### Step 3:瓶颈分析 → 优化方向

核心问题:如何在不完全排序的情况下,快速找到"中间"的元素?
- 观察:中位数只关心"左半部分最大值"和"右半部分最小值"
- 优化思路:如果能用O(log n)时间维护这两个值,就能在O(1)时间查询中位数

### Step 4:选择武器

- 选用:**对顶堆(双堆)**
- 理由:
  - 大顶堆维护较小的一半,堆顶是这一半的最大值
  - 小顶堆维护较大的一半,堆顶是这一半的最小值
  - 插入O(log n),查询O(1),完美解决瓶颈

> 🔑 **模式识别提示**:当题目要求"动态维护最值"或"数据流中的第K大/中位数",优先考虑"堆"

---

## 🔑 解法一:排序数组(直觉法)

### 思路

维护一个有序列表,每次添加元素后重新排序,查询中位数时直接取中间位置。

### 图解过程

```
初始: nums = []

addNum(1):
  nums = [1]

addNum(2):
  nums = [1, 2]

findMedian():
  排序后 [1, 2]
  中位数 = (1 + 2) / 2 = 1.5

addNum(3):
  nums = [1, 2, 3]

findMedian():
  排序后 [1, 2, 3]
  中位数 = 2.0
```

### Python代码

```python
class MedianFinder:
    """
    解法一:排序数组
    思路:每次查询中位数时重新排序
    """
    def __init__(self):
        self.nums = []

    def addNum(self, num: int) -> None:
        self.nums.append(num)  # 直接添加

    def findMedian(self) -> float:
        self.nums.sort()  # 每次查询都排序
        n = len(self.nums)
        if n % 2 == 1:
            return float(self.nums[n // 2])
        else:
            return (self.nums[n // 2 - 1] + self.nums[n // 2]) / 2.0


# ✅ 测试
mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  # 期望输出:1.5
mf.addNum(3)
print(mf.findMedian())  # 期望输出:2.0
```

### 复杂度分析

- **时间复杂度**:
  - `addNum`: O(1) — 只是追加到列表末尾
  - `findMedian`: O(n log n) — 每次都要排序
  - 如果输入规模n=10000,调用100次`findMedian`,大约需要10⁶ * 100 = 10⁸次操作
- **空间复杂度**:O(n) — 存储所有元素

### 优缺点

- ✅ 代码简单,易于理解
- ❌ 查询中位数太慢,无法应对频繁查询的场景
- ❌ 重复排序浪费计算资源

---

## 🏆 解法二:对顶堆(最优解)

### 优化思路

核心观察:中位数只需要知道"较小一半的最大值"和"较大一半的最小值",不需要完整排序。我们用两个堆分别维护这两部分:
- **大顶堆(max_heap)**:存较小的一半,堆顶是这部分的最大值
- **小顶堆(min_heap)**:存较大的一半,堆顶是这部分的最小值

保持平衡:让大顶堆的元素个数等于或比小顶堆多1个。

> 💡 **关键想法**:两个堆的堆顶就是中位数的候选值,查询时直接取堆顶即可!

### 图解过程

```
初始状态:
  大顶堆(较小一半): []
  小顶堆(较大一半): []

addNum(1):
  1. 先放入大顶堆: [1]
  2. 平衡后:
     大顶堆: [1]
     小顶堆: []

addNum(2):
  1. 先放入大顶堆: [1, 2] → 弹出最大值2
  2. 将2放入小顶堆: [2]
  3. 平衡后:
     大顶堆: [1]       (1个元素)
     小顶堆: [2]       (1个元素)
  findMedian() = (1 + 2) / 2 = 1.5

addNum(3):
  1. 先放入大顶堆: [1] → 弹出1
  2. 将1放入小顶堆: [1, 2] → 弹出1
  3. 将1放回大顶堆: [1]
  4. 平衡后(小顶堆多了,需要调整):
     大顶堆: [1, 2]    (2个元素,堆顶2)
     小顶堆: [3]       (1个元素)
  findMedian() = 2.0 (大顶堆堆顶)

对顶堆结构:
     [较小一半]  |  [较大一半]
    大顶堆      |   小顶堆
      ▲         |     ▼
    最大值      |   最小值
       ↓        |     ↓
      中位数候选值
```

### Python代码

```python
import heapq


class MedianFinder:
    """
    解法二:对顶堆
    思路:用大顶堆维护较小一半,小顶堆维护较大一半
    """
    def __init__(self):
        # 大顶堆(Python只有小顶堆,用负数模拟大顶堆)
        self.small = []  # 存较小一半,堆顶是最大值
        # 小顶堆
        self.large = []  # 存较大一半,堆顶是最小值

    def addNum(self, num: int) -> None:
        # 策略:始终先放入大顶堆,然后平衡
        # 1. 放入大顶堆(用负数)
        heapq.heappush(self.small, -num)

        # 2. 弹出大顶堆的最大值,放入小顶堆
        #    (保证small中的所有值 <= large中的所有值)
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # 3. 保持平衡:大顶堆元素数 >= 小顶堆元素数
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            # 奇数个元素,大顶堆多1个,返回堆顶
            return float(-self.small[0])
        else:
            # 偶数个元素,返回两个堆顶的平均值
            return (-self.small[0] + self.large[0]) / 2.0


# ✅ 测试
mf = MedianFinder()
mf.addNum(1)
mf.addNum(2)
print(mf.findMedian())  # 期望输出:1.5
mf.addNum(3)
print(mf.findMedian())  # 期望输出:2.0

# 边界测试
mf2 = MedianFinder()
mf2.addNum(-1)
mf2.addNum(-2)
print(mf2.findMedian())  # 期望输出:-1.5

mf3 = MedianFinder()
mf3.addNum(1)
mf3.addNum(1)
print(mf3.findMedian())  # 期望输出:1.0
```

### 复杂度分析

- **时间复杂度**:
  - `addNum`: O(log n) — 堆的插入和调整操作
  - `findMedian`: O(1) — 直接访问堆顶
  - 如果调用100次`addNum`和100次`findMedian`,总复杂度约O(100 log 100) ≈ O(664)次操作,比暴力法的10⁸次快了15万倍!
- **空间复杂度**:O(n) — 两个堆共存储n个元素

---

## 🐍 Pythonic 写法

利用Python的heapq模块和属性封装,可以让代码更清晰:

```python
import heapq


class MedianFinder:
    def __init__(self):
        self.small = []  # 大顶堆(负数模拟)
        self.large = []  # 小顶堆

    def addNum(self, num: int) -> None:
        # 链式操作:先放small,再平衡到large,最后保持small >= large
        heapq.heappush(self.large, -heapq.heappushpop(self.small, -num))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        return -self.small[0] if len(self.small) > len(self.large) \
               else (-self.small[0] + self.large[0]) / 2.0
```

这个写法利用了`heappushpop`的原子操作,减少了一次显式的push-pop组合。

> ⚠️ **面试建议**:先写清晰版本展示思路,再提Pythonic写法展示语言功底。
> 面试官更看重你的**对顶堆平衡策略的理解**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:排序数组 | 🏆 解法二:对顶堆(最优) |
|------|--------------|---------------------|
| 时间复杂度(addNum) | O(1) | **O(log n)** |
| 时间复杂度(findMedian) | O(n log n) | **O(1)** ← 查询极快 |
| 空间复杂度 | O(n) | O(n) |
| 代码难度 | 简单 | 中等(需理解堆平衡) |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 只适合极少查询 | **数据流场景的标准解法** |

**为什么是最优解**:
- 在数据流场景下,`findMedian`调用频率远高于`addNum`,O(1)查询是核心优势
- 虽然`addNum`从O(1)变成O(log n),但这是可接受的代价
- 对于100次操作,对顶堆比排序法快15万倍!

**面试建议**:
1. 先用30秒口述排序法思路(O(n log n)查询),表明你能想到基本解法
2. 立即优化到🏆对顶堆(O(1)查询),展示优化能力
3. **重点讲解对顶堆的平衡策略**:"保证small堆元素数 >= large堆,且small的最大值 <= large的最小值"
4. 强调为什么这是最优:查询时间O(1)已达理论最优,适合数据流高频查询场景
5. 手动模拟添加[1,2,3]的过程,展示对堆平衡的深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你设计一个数据结构,支持从数据流中添加元素和查询中位数。

**你**:(审题30秒)好的,这道题要求在动态数据流中维护中位数。让我先想一下...

我的第一个想法是用一个列表存所有数字,每次查询中位数时排序,时间复杂度是O(n log n)。但这在频繁查询时会很慢。

我可以优化到O(1)查询,用"对顶堆"方案:
- 用大顶堆维护较小的一半,堆顶是这部分的最大值
- 用小顶堆维护较大的一半,堆顶是这部分的最小值
- 保持两个堆的大小平衡,查询中位数时直接看堆顶

这样`addNum`是O(log n),`findMedian`是O(1)。

**面试官**:很好,请写一下代码,特别是如何保持平衡。

**你**:(边写边说关键步骤)
```python
# 核心策略:
# 1. 每次新元素先放入大顶堆
# 2. 立即将大顶堆的最大值弹出放入小顶堆(保证small <= large)
# 3. 如果小顶堆元素更多,弹出最小值放回大顶堆(保持平衡)
```

**面试官**:测试一下?

**你**:用示例[1,2,3]走一遍:
1. addNum(1):small=[1],large=[] → 中位数1.0
2. addNum(2):先放small=[1,2],弹出2放large=[2],平衡后small=[1],large=[2] → 中位数1.5
3. addNum(3):先放small=[1,3],弹出3放large=[2,3],弹出2回small=[1,2] → 中位数2.0

结果正确!再测边界情况,负数[-1,-2] → -1.5,也正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么不用平衡二叉树(AVL/红黑树)?" | "AVL能做到O(log n)插入和查询,但查询中位数仍需O(log n)找第k大。对顶堆的O(1)查询更优,且Python的heapq实现简单" |
| "如果数据量非常大,内存放不下?" | "可以考虑分桶统计:将数据范围分成若干区间,统计每个区间的元素数,中位数一定在某个区间,再在该区间内精确查找" |
| "能否支持删除元素?" | "需要用惰性删除:标记删除但不立即移除,查询时跳过已删除元素。或者用支持删除的平衡树" |
| "为什么大顶堆用负数模拟?" | "Python的heapq只提供小顶堆,将元素取负后,最小堆的堆顶(-x)就是原数据的最大值(x)" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:heappushpop原子操作 — 先push再pop,比分两步快
import heapq
heap = [1, 3, 5]
heapq.heapify(heap)
val = heapq.heappushpop(heap, 2)  # 先push 2,再pop最小值 → 返回1

# 技巧2:用负数模拟大顶堆 — Python只有小顶堆
max_heap = []
heapq.heappush(max_heap, -10)  # 存-10
heapq.heappush(max_heap, -5)   # 存-5
print(-max_heap[0])  # 输出10(最大值)

# 技巧3:堆的初始化 — 从列表快速建堆O(n)
nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)  # 原地转换为小顶堆
```

### 💡 底层原理(选读)

> **堆为什么插入和弹出是O(log n)?**
>
> 堆是一种完全二叉树,满足父节点 >= (或<=)子节点。因为是完全二叉树,高度是log n。
> - 插入:先放到末尾,然后"上浮"到合适位置,最多上浮log n层
> - 弹出:取出堆顶后,用末尾元素替代,然后"下沉"到合适位置,最多下沉log n层
>
> Python的heapq用列表实现,对于下标i:
> - 左子节点:2*i + 1
> - 右子节点:2*i + 2
> - 父节点:(i-1) // 2

### 算法模式卡片 📐

- **模式名称**:对顶堆(双堆)
- **适用条件**:需要在动态数据流中维护中位数、或者维护动态的"第k大/小"元素
- **识别关键词**:"数据流"+"中位数"、"动态维护最值"
- **模板代码**:
```python
class MedianFinder:
    def __init__(self):
        self.small = []  # 大顶堆(负数模拟)
        self.large = []  # 小顶堆

    def addNum(self, num: int) -> None:
        heapq.heappush(self.small, -num)
        heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2.0
```

### 易错点 ⚠️

1. **忘记用负数模拟大顶堆**:Python的heapq是小顶堆,需要存负数来模拟大顶堆。正确做法:`heapq.heappush(max_heap, -num)`
2. **平衡策略错误**:必须保证`len(small) == len(large)`或`len(small) == len(large) + 1`,否则中位数计算会错。正确做法:每次`addNum`后检查并调整。
3. **查询时忘记取负**:大顶堆的堆顶是负数,查询时要转回正数。正确做法:`-self.small[0]`
4. **边界情况漏判**:空堆时不能访问堆顶。正确做法:初始化后立即添加元素,或在`findMedian`中加空判断。

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:实时监控系统中计算CPU使用率中位数,用于异常检测
- **场景2**:游戏排行榜中实时计算玩家分数的中位数,展示中等水平
- **场景3**:广告系统中计算点击率的中位数,用于评估广告质量
- **场景4**:数据库查询优化器,估算查询结果集的中位数大小,选择合适的索引

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 480. 滑动窗口中位数 | Hard | 对顶堆+滑动窗口 | 需要支持删除操作,用惰性删除或multiset |
| LeetCode 4. 两个正序数组的中位数 | Hard | 二分查找 | 另一种O(log n)求中位数的方法 |
| LeetCode 703. 数据流中的第K大元素 | Easy | 小顶堆 | 对顶堆的简化版,只需一个堆 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如果数据流中99%的元素都在[0, 100]范围内,如何优化空间和时间?

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

可以用计数数组记录[0,100]内每个数字的频率,用两个变量记录超出范围的极端值。查询中位数时先看计数数组能否确定,不能再看堆。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
class OptimizedMedianFinder:
    def __init__(self):
        self.count = [0] * 101  # 计数[0,100]
        self.outliers = []      # 存超出范围的值
        self.total = 0

    def addNum(self, num: int) -> None:
        if 0 <= num <= 100:
            self.count[num] += 1
        else:
            self.outliers.append(num)
        self.total += 1

    def findMedian(self) -> float:
        target = self.total // 2
        cumulative = 0

        # 先统计超出范围的较小值
        small_outliers = sorted([x for x in self.outliers if x < 0])
        cumulative += len(small_outliers)

        # 遍历计数数组找中位数
        for num in range(101):
            cumulative += self.count[num]
            if cumulative > target:
                if self.total % 2 == 1:
                    return float(num)
                else:
                    # 找前一个数
                    prev = self._find_prev(num, target)
                    return (prev + num) / 2.0

        # 中位数在大的outliers中
        large_outliers = sorted([x for x in self.outliers if x > 100])
        idx = target - cumulative
        return float(large_outliers[idx])

    def _find_prev(self, num: int, target: int) -> int:
        # 找第target个元素(从0计数)
        cumulative = len([x for x in self.outliers if x < 0])
        for i in range(num + 1):
            if cumulative == target:
                return i
            cumulative += self.count[i]
        return num
```

核心思路:利用数据分布特性,用O(1)空间的计数数组替代大部分堆操作。在最好情况下,`findMedian`从O(log n)优化到O(100)=O(1)。

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
