> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第97课:前K个高频元素

> **模块**:堆与优先队列 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/top-k-frequent-elements/
> **前置知识**:第38课(数组中第K大元素)、哈希表基础
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个整数数组 nums 和一个整数 k,返回出现频率前 k 高的元素。你可以按任意顺序返回答案。

**示例:**
```
输入:nums = [1,1,1,2,2,3], k = 2
输出:[1,2]
解释:元素1出现3次,元素2出现2次,元素3出现1次,所以前2个高频元素是[1,2]
```

```
输入:nums = [1], k = 1
输出:[1]
解释:只有一个元素,就是它
```

**约束条件:**
- 1 <= nums.length <= 10^5
- k 的范围是 [1, 数组中不相同元素的个数]
- 题目保证答案唯一
- 要求时间复杂度优于 O(n log n)

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1], k=1 | [1] | 只有一个元素的情况 |
| 全部相同 | nums=[7,7,7,7], k=1 | [7] | 频率相同的处理 |
| 全不相同 | nums=[1,2,3,4,5], k=2 | 任意2个 | 频率都是1的情况 |
| 负数和零 | nums=[0,-1,-1,2,2,2], k=2 | [2,-1] | 特殊值处理 |
| 大规模 | n=100000, k=100 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在统计一个班级的投票结果,需要找出得票最多的前K名同学。
>
> 🐌 **笨办法**:先统计每个人的票数,然后对所有人按票数排序,取前K个。这就像让所有100个同学按票数从高到低站成一排,但我们只需要前3名,却要对所有人排序,时间复杂度O(n log n),有点浪费。
>
> 🚀 **聪明办法**:用一个只能容纳K个人的"颁奖台"(最小堆)。每次来一个新同学,如果颁奖台未满就直接上台;如果满了,就和台上票数最少的比较,如果新同学票数更多,就把台上那个人踢下去。最后台上剩下的K个人就是答案。时间复杂度优化到O(n log k),当k远小于n时,效率提升巨大!

### 关键洞察
**Top-K问题的核心是"动态维护K个最值",用堆比全排序快得多**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组 nums 和整数 k
- **输出**:出现频率最高的k个元素(返回元素值,不是频率)
- **限制**:要求时间复杂度优于O(n log n)

### Step 2:先想笨办法(暴力法)
最直接的思路:
1. 用哈希表统计每个元素的频率 → O(n)
2. 对所有元素按频率排序 → O(n log n)
3. 取前k个 → O(k)

- 时间复杂度:O(n log n)
- 瓶颈在哪:排序这一步对所有n个元素操作,但我们只需要前k个

### Step 3:瓶颈分析 → 优化方向
问题分析:
- 排序获取了"全部元素的相对顺序",但我们只需要"前k个最大值"
- 这就像找班级前3名,不需要让所有100人排队,只需要维护一个3人的"前三名榜单"

核心问题:如何在O(n log k)时间内找到前k大?
优化思路:用最小堆维护大小为k的"候选集",堆顶是第k大的元素

### Step 4:选择武器
- 选用:**最小堆(heapq模块)** + **Counter计数器**
- 理由:
  - Counter快速统计频率 O(n)
  - 最小堆维护k个最大频率的元素 O(n log k)
  - 堆顶是第k大,比堆顶大的才能进堆

> 🔑 **模式识别提示**:当题目出现"前K大/小"、"最高频的K个",优先考虑"堆"模式

---

## 🔑 解法一:排序法(直觉法)

### 思路
先用Counter统计频率,然后直接对所有元素按频率排序,取前k个。

### 图解过程

```
示例:nums = [1,1,1,2,2,3], k = 2

Step 1:统计频率
  Counter({1: 3, 2: 2, 3: 1})

Step 2:转为列表并排序
  [(1,3), (2,2), (3,1)]  →  按频率降序排序
  [(1,3), (2,2), (3,1)]

Step 3:取前k个元素
  [(1,3), (2,2)]  →  提取元素  →  [1, 2]
```

再看一个边界情况:
```
输入:nums = [4,4,4,5,5,6], k = 2

Step 1:Counter({4: 3, 5: 2, 6: 1})
Step 2:排序后 [(4,3), (5,2), (6,1)]
Step 3:取前2个 → [4, 5]
```

### Python代码

```python
from typing import List
from collections import Counter


def topKFrequent_sort(nums: List[int], k: int) -> List[int]:
    """
    解法一:排序法
    思路:统计频率后全排序,取前k个
    """
    # Step 1:统计每个元素的频率
    count = Counter(nums)  # O(n)

    # Step 2:按频率降序排序(lambda取频率值)
    sorted_items = sorted(count.items(), key=lambda x: x[1], reverse=True)  # O(n log n)

    # Step 3:取前k个元素(只要元素值,不要频率)
    return [item[0] for item in sorted_items[:k]]  # O(k)


# ✅ 测试
print(topKFrequent_sort([1,1,1,2,2,3], 2))  # 期望输出:[1, 2]
print(topKFrequent_sort([1], 1))             # 期望输出:[1]
print(topKFrequent_sort([4,1,1,1,2,2,3], 2))  # 期望输出:[1, 2]
```

### 复杂度分析
- **时间复杂度**:O(n log n) — 瓶颈在排序
  - 统计频率 O(n)
  - 排序所有元素 O(n log n)
  - 具体地说:如果n=100000,k=10,需要约100000×17≈170万次比较
- **空间复杂度**:O(n) — Counter字典和排序后的列表

### 优缺点
- ✅ 代码简单直观,易于理解
- ✅ 对所有k都是同样的复杂度
- ❌ 没有利用"只需要前k个"这一信息,做了多余的排序

---

## 🏆 解法二:最小堆(最优解)

### 优化思路
排序法对所有n个元素排序,但我们只需要前k个。用最小堆维护一个大小为k的"候选池",只保留频率最高的k个元素,避免全排序。

> 💡 **关键想法**:维护大小为k的最小堆,堆顶是第k大元素,任何比堆顶大的新元素都能进堆并踢掉堆顶

### 图解过程

```
示例:nums = [1,1,1,2,2,3,4,4,4,4], k = 2

Step 1:统计频率
  Counter: {4:4, 1:3, 2:2, 3:1}

Step 2:用最小堆维护前k个高频元素(堆大小≤k)

  遍历 (频率, 元素):

  ① 加入 (4, 4):堆 = [(4, 4)]                    堆大小<k,直接入堆
  ② 加入 (3, 1):堆 = [(3, 1), (4, 4)]            堆大小<k,直接入堆
  ③ 遇到 (2, 2):堆满了,堆顶(3,1),2<3,不入堆      堆 = [(3, 1), (4, 4)]
  ④ 遇到 (1, 3):堆顶(3,1),1<3,不入堆              堆 = [(3, 1), (4, 4)]

  注意:Python的heapq是最小堆,堆顶是最小值

Step 3:提取堆中所有元素
  堆 = [(3, 1), (4, 4)]  →  元素 = [1, 4]
```

再看详细过程:
```
nums = [1,1,1,2,2,3], k = 2
Counter: {1:3, 2:2, 3:1}

建堆过程:
  初始堆:[]

  ① 加入 (3, 1):堆 = [(3, 1)]                    堆大小=1 < k=2
  ② 加入 (2, 2):堆 = [(2, 2), (3, 1)]            堆大小=2 = k
  ③ 遇到 (1, 3):堆顶(2,2),1<2,不入堆              堆 = [(2, 2), (3, 1)]

  最终:提取元素 [2, 1]  (顺序无所谓)
```

### Python代码

```python
import heapq
from typing import List
from collections import Counter


def topKFrequent(nums: List[int], k: int) -> List[int]:
    """
    🏆 解法二:最小堆(最优解)
    思路:维护大小为k的最小堆,堆中保留频率最高的k个元素
    """
    # Step 1:统计频率
    count = Counter(nums)  # O(n)

    # Step 2:用最小堆维护前k个高频元素
    # 堆中存储 (频率, 元素),Python heapq自动按第一个元素(频率)建最小堆
    heap = []

    for num, freq in count.items():
        if len(heap) < k:
            # 堆未满,直接入堆
            heapq.heappush(heap, (freq, num))  # O(log k)
        else:
            # 堆满了,如果当前频率大于堆顶(最小频率),则替换
            if freq > heap[0][0]:
                heapq.heapreplace(heap, (freq, num))  # O(log k)

    # Step 3:提取堆中所有元素(只要元素值,不要频率)
    return [item[1] for item in heap]  # O(k)


# ✅ 测试
print(topKFrequent([1,1,1,2,2,3], 2))  # 期望输出:[1, 2]
print(topKFrequent([1], 1))             # 期望输出:[1]
print(topKFrequent([4,1,1,1,2,2,3], 2))  # 期望输出:[1, 2]
```

### 复杂度分析
- **时间复杂度**:O(n log k) — 🏆 这是最优解
  - 统计频率 O(n)
  - 维护堆:最坏情况每个元素都入堆出堆一次,n次堆操作,每次O(log k),总共O(n log k)
  - 提取结果 O(k)
  - **为什么最优**:当k远小于n时(如n=100000,k=10),O(n log k) ≈ O(n×3.3) 远小于 O(n log n) ≈ O(n×17)
- **空间复杂度**:O(n) — Counter字典O(n),堆O(k),总体O(n)

### 为什么是最优解?
1. **时间已达题目要求**:题目要求"优于O(n log n)",我们做到了O(n log k)
2. **无法更优**:至少需要O(n)遍历统计频率,所以理论下限是O(n),而我们的O(n log k)在k<<n时接近O(n)
3. **实际性能优势明显**:当n=100000,k=10时,log k=3.3,log n=17,速度提升约5倍

---

## ⚡ 解法三:桶排序(线性时间)

### 优化思路
如果频率的范围有限(最大为n),可以用"桶"来避免堆操作。用数组下标表示频率,值是该频率对应的元素列表。从高频到低频遍历桶,收集k个元素。

> 💡 **关键想法**:频率范围[1,n],用n个桶代替堆,从后往前取即可

### 图解过程

```
示例:nums = [1,1,1,2,2,3], k = 2

Step 1:统计频率
  Counter: {1:3, 2:2, 3:1}

Step 2:建立频率桶(下标表示频率)
  buckets = [
    [],      # 频率0(不存在)
    [3],     # 频率1:元素3
    [2],     # 频率2:元素2
    [1],     # 频率3:元素1
    [],      # 频率4
    [],      # 频率5
    []       # 频率6
  ]

Step 3:从高频到低频遍历桶,收集k个元素
  从 buckets[3] 开始:收集1  →  result=[1], 还需1个
  到 buckets[2]:收集2  →  result=[1,2], 凑够k=2个

  返回 [1, 2]
```

### Python代码

```python
from typing import List
from collections import Counter


def topKFrequent_bucket(nums: List[int], k: int) -> List[int]:
    """
    解法三:桶排序
    思路:用频率作为桶的下标,从高频桶往低频桶收集元素
    """
    # Step 1:统计频率
    count = Counter(nums)

    # Step 2:建立频率桶(下标=频率,值=该频率的元素列表)
    n = len(nums)
    buckets = [[] for _ in range(n + 1)]  # 频率范围[0, n]

    for num, freq in count.items():
        buckets[freq].append(num)  # O(n)

    # Step 3:从高频到低频遍历桶,收集k个元素
    result = []
    for freq in range(n, 0, -1):  # 从高到低遍历
        if buckets[freq]:
            result.extend(buckets[freq])
            if len(result) >= k:
                return result[:k]  # 只取前k个

    return result


# ✅ 测试
print(topKFrequent_bucket([1,1,1,2,2,3], 2))  # 期望输出:[1, 2]
print(topKFrequent_bucket([1], 1))             # 期望输出:[1]
print(topKFrequent_bucket([4,1,1,1,2,2,3], 2))  # 期望输出:[1, 2]
```

### 复杂度分析
- **时间复杂度**:O(n) — 线性时间
  - 统计频率 O(n)
  - 建桶 O(n)
  - 遍历桶 O(n)
- **空间复杂度**:O(n) — 桶数组

### 优缺点
- ✅ 时间复杂度O(n)最优
- ❌ 空间需要O(n)的桶数组(对比堆只需O(k))
- ❌ 只适用于频率范围有限的场景

---

## 🐍 Pythonic 写法

利用 Python 标准库的简洁写法:

```python
from collections import Counter

def topKFrequent_pythonic(nums: list[int], k: int) -> list[int]:
    """
    Pythonic写法:直接用Counter.most_common()
    """
    # Counter.most_common(k) 内部用堆实现,返回前k个(元素,频率)元组
    return [item[0] for item in Counter(nums).most_common(k)]

# 超级简洁,一行版:
topKFrequent_oneliner = lambda nums, k: [x for x, _ in Counter(nums).most_common(k)]
```

**解释**:
- `Counter.most_common(k)`内部使用堆实现,时间复杂度O(n log k),与解法二相同
- 返回格式是[(元素,频率),...],用列表推导式提取元素
- 这个写法展示了对Python标准库的熟练掌握

> ⚠️ **面试建议**:先写清晰版本(解法二)展示算法思路,再提这个Pythonic写法展示语言功底。
> 面试官更看重你的**算法思维**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:排序法 | 🏆 解法二:最小堆(最优) | 解法三:桶排序 |
|------|------------|---------------------|------------|
| 时间复杂度 | O(n log n) | **O(n log k)** ← 时间最优(k<<n时) | O(n) ← 理论最优 |
| 空间复杂度 | O(n) | **O(n)** | O(n) |
| 代码难度 | 简单 | 中等 | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | k接近n时差异不大 | **通用,k<<n时性能优势明显** | 频率范围已知且有限 |

**为什么解法二是最优解?**
- 当k远小于n时(典型场景),O(n log k)远优于O(n log n)
- 例如n=100000,k=10:解法二比解法一快约5倍
- 满足题目"优于O(n log n)"的要求
- 堆是Top-K问题的标准解法,面试中最受认可

**面试建议**:
1. 先用30秒口述排序法思路(O(n log n)),表明你能想到基本解法
2. 立即优化到🏆最优解(O(n log k)最小堆),展示优化能力
3. **重点讲解最优解的核心思想**:"维护大小为k的最小堆,只保留频率最高的k个元素,避免全排序"
4. 强调为什么这是最优:当k<<n时,log k远小于log n,时间从O(n log n)降到O(n log k)
5. 如果时间充裕,可以提及桶排序O(n)方案,但说明空间换时间的权衡

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找出数组中出现频率最高的k个元素。让我先想一下...
我的第一个想法是先用Counter统计频率,然后对所有元素按频率排序,取前k个,时间复杂度是O(n log n)。
不过题目要求优于O(n log n),我们可以用**最小堆**来优化到O(n log k)。核心思路是维护一个大小为k的最小堆,堆中保留频率最高的k个元素,避免对所有n个元素排序。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
首先用Counter统计频率,然后建一个最小堆。遍历频率字典,如果堆大小小于k就直接入堆;如果堆满了,就和堆顶(当前第k大)比较,如果更大就替换堆顶。最后堆中剩下的就是频率最高的k个元素。

**面试官**:测试一下?

**你**:用示例[1,1,1,2,2,3], k=2走一遍。
- Counter得到{1:3, 2:2, 3:1}
- 遍历:(3,1)入堆 → (2,2)入堆,堆满 → (1,3)因为1<2不入堆
- 最终堆=[（2,2), (3,1)],提取元素[2,1]

再测一个边界情况[1],k=1,直接返回[1]。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | "可以用桶排序做到O(n),但需要O(n)空间建桶。堆方案在k<<n时已经接近O(n),且空间更灵活,是更通用的最优解" |
| "如果k很大,接近n呢?" | "当k接近n时,O(n log k)接近O(n log n),此时排序法和堆方案性能相当。但堆方案仍更优,因为避免了全排序" |
| "能不能不用Counter?" | "可以手动用字典统计,但Counter更简洁。核心是堆的使用,统计方式不影响复杂度" |
| "为什么用最小堆而不是最大堆?" | "最小堆堆顶是第k大元素,作为'门槛':大于门槛才能进堆。最大堆堆顶是最大元素,无法快速判断是否属于前k大" |
| "heapreplace和heappop+heappush有什么区别?" | "heapreplace是原子操作,效率更高。它先返回堆顶,再把新元素入堆并调整,避免了两次堆调整" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:Counter快速统计频率
from collections import Counter
count = Counter([1,1,2,2,2,3])  # Counter({2: 3, 1: 2, 3: 1})
print(count.most_common(2))     # [(2, 3), (1, 2)]

# 技巧2:heapq维护最小堆
import heapq
heap = []
heapq.heappush(heap, (3, 'a'))    # 插入O(log k)
heapq.heappush(heap, (1, 'b'))
print(heap[0])                     # (1, 'b') 堆顶是最小元素

# 技巧3:heapreplace原子操作
heapq.heapreplace(heap, (2, 'c'))  # 弹出堆顶并插入新元素,一次调整

# 技巧4:Counter.items()遍历
for num, freq in count.items():
    print(f"{num}出现{freq}次")
```

### 💡 底层原理(选读)

> **Python的堆是什么?**
> - Python的heapq实现的是**最小堆**,堆顶是最小元素
> - 底层用数组存储,父节点下标i,左子节点2i+1,右子节点2i+2
> - 插入/删除操作通过"上浮"和"下沉"维持堆性质,时间O(log k)
>
> **为什么堆比排序快?**
> - 排序需要确定所有n个元素的相对顺序 → O(n log n)
> - 堆只需维护k个元素的偏序关系(堆顶最小) → O(n log k)
> - 当k=10,n=10000时,log k=3.3,log n=13.3,差4倍
>
> **Counter底层是什么?**
> - Counter继承自dict,是一个特殊的字典
> - key是元素,value是计数
> - most_common()内部用堆实现,时间O(n log k)

### 算法模式卡片 📐
- **模式名称**:Top-K问题 + 最小堆
- **适用条件**:需要在大量数据中找出"前K大/小"元素
- **识别关键词**:题目出现"前K个最大/最小"、"频率最高的K个"、"最常见的K个"
- **核心思想**:维护大小为k的最小堆(求Top-K大)或最大堆(求Top-K小),堆顶作为"门槛"
- **时间复杂度**:O(n log k),当k<<n时远优于排序的O(n log n)
- **模板代码**:
```python
import heapq

def top_k_template(items, k, key_func):
    """
    Top-K通用模板
    items: 待处理数据
    k: 取前k个
    key_func: 排序依据(如频率、值等)
    """
    heap = []
    for item in items:
        score = key_func(item)
        if len(heap) < k:
            heapq.heappush(heap, (score, item))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, item))

    return [item for score, item in heap]
```

### 易错点 ⚠️
1. **堆中存储格式错误**:必须存储`(频率, 元素)`元组,因为heapq按第一个元素(频率)排序。如果只存元素,会按元素值排序而非频率
   ```python
   # ❌ 错误:只存元素
   heapq.heappush(heap, num)

   # ✅ 正确:存(频率,元素)
   heapq.heappush(heap, (freq, num))
   ```

2. **最大堆和最小堆混淆**:求Top-K**大**用**最小堆**,堆顶是第k大元素;求Top-K**小**用**最大堆**(需要对值取负)
   ```python
   # 求Top-K大:最小堆
   heapq.heappush(heap, (freq, num))  # 小的在堆顶

   # 求Top-K小:最大堆(取负模拟)
   heapq.heappush(heap, (-freq, num))  # 大的取负后变小,在堆顶
   ```

3. **堆满后忘记比较就替换**:必须先判断新元素是否大于堆顶,才能决定是否替换
   ```python
   # ❌ 错误:堆满后直接替换
   if len(heap) == k:
       heapq.heapreplace(heap, (freq, num))

   # ✅ 正确:比较后再决定
   if len(heap) < k:
       heapq.heappush(heap, (freq, num))
   elif freq > heap[0][0]:  # 大于堆顶才替换
       heapq.heapreplace(heap, (freq, num))
   ```

4. **返回结果时忘记只提取元素**:堆中存的是`(频率,元素)`,返回时只要元素
   ```python
   # ❌ 错误:返回整个元组
   return heap  # [(2,1), (3,2)]

   # ✅ 正确:只提取元素
   return [item[1] for item in heap]  # [1, 2]
   ```

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:搜索引擎的热搜榜 - 实时统计搜索关键词频率,维护Top-K热词。用最小堆每秒更新一次,避免全排序
- **场景2**:电商平台的热销商品 - 统计每个商品的销量,展示Top-10热销榜。用堆维护,新订单来时增量更新
- **场景3**:日志分析系统 - 从海量日志中找出Top-K高频错误码。数据量大(亿级),用堆比排序节省90%内存和时间
- **场景4**:推荐系统的协同过滤 - 找出与用户最相似的Top-K个用户。计算相似度后用堆筛选,避免对百万用户排序

**工程优化**:
- 数据流场景:用滑动窗口+堆,只统计最近N条记录的Top-K
- 分布式场景:每台机器计算局部Top-K,最后汇总合并
- 内存受限:用Count-Min Sketch近似统计频率,再用堆

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 215. 数组中第K大元素 | Medium | 堆/快速选择 | 本题的简化版,只需找一个值 |
| LeetCode 692. 前K个高频单词 | Medium | 堆+自定义排序 | 相同频率按字典序排序 |
| LeetCode 973. 最接近原点的K个点 | Medium | 堆/快速选择 | 按距离排序,找前K个 |
| LeetCode 703. 数据流中第K大元素 | Easy | 最小堆 | 动态维护第K大,用大小为K的最小堆 |
| LeetCode 295. 数据流中位数 | Hard | 对顶堆 | 用两个堆(大顶+小顶)维护中位数 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定字符串数组words和整数k,返回出现频率最高的前k个字符串。如果两个字符串频率相同,按字典序返回较小的那个。

例如:words = ["i", "love", "leetcode", "i", "love", "coding"], k = 2
输出:["i", "love"]
解释:"i"和"love"都出现2次,其他1次

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

关键在于堆的比较规则:Python的元组比较是逐元素的,可以利用`(-freq, word)`让频率高的排前面,频率相同时字典序小的排前面(注意负号的作用)

</details>

<details>
<summary>✅ 参考答案</summary>

```python
import heapq
from collections import Counter

def topKFrequent_words(words: list[str], k: int) -> list[str]:
    """
    变体:前K个高频单词,频率相同按字典序
    """
    count = Counter(words)

    # 关键:用(-freq, word)作为堆元素
    # 负频率让高频词排前面,字典序自然升序
    heap = []
    for word, freq in count.items():
        heapq.heappush(heap, (-freq, word))

    # 弹出前k个
    return [heapq.heappop(heap)[1] for _ in range(k)]

# 测试
print(topKFrequent_words(["i", "love", "leetcode", "i", "love", "coding"], 2))
# 输出:["i", "love"]
```

**核心思路**:
- 用`(-freq, word)`作为堆元素,利用Python元组的自然排序
- 负频率确保高频词在堆顶(最小堆变最大堆)
- 频率相同时,字典序自然升序(无需额外处理)
- 时间复杂度仍是O(n log n)(因为需要全排序保证字典序),但代码简洁

**注意**:本题因为要求字典序,无法用大小为k的堆优化,必须全排序。真实场景中可以先用堆筛选,再对k个元素排序。

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
