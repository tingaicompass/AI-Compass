> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第12课：删除有序数组中的重复项

> **模块**:双指针 | **难度**:Easy ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/remove-duplicates-from-sorted-array/
> **前置知识**:[第7课:移动零](../007-move-zeroes/lesson.md)
> **预计学习时间**:15分钟

---

## 🎯 题目描述

给定一个**已排序**的整数数组 `nums`,你需要**原地**删除重复出现的元素,使每个元素只出现一次,返回删除后数组的新长度。

**不要使用额外的数组空间**,必须在原地修改输入数组并在使用 O(1) 额外空间的条件下完成。

**示例:**
```
输入:nums = [1,1,2]
输出:2, nums = [1,2,_]
解释:函数应该返回新的长度 2,并且原数组 nums 的前两个元素被修改为 1, 2。不需要考虑数组中超出新长度后面的元素。
```

```
输入:nums = [0,0,1,1,1,2,2,3,3,4]
输出:5, nums = [0,1,2,3,4,_,_,_,_,_]
解释:函数应该返回新的长度 5,并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。
```

**约束条件:**
- `1 <= nums.length <= 3 * 10^4`
- `-100 <= nums[i] <= 100`
- `nums` 已按**升序排列**

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | `[1]` | `1` | 单元素无重复 |
| 无重复 | `[1,2,3,4,5]` | `5` | 已满足条件 |
| 全部重复 | `[2,2,2,2]` | `1` | 极端重复 |
| 连续重复 | `[1,1,2,2,3,3]` | `3` | 多组重复 |
| 大规模 | `n=30000` | — | 性能边界 O(n) |
| 负数 | `[-3,-3,-1,0,0,1]` | `4` | 负数处理 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在整理一排**已排好序**的书籍,相邻的书如果是重复的,你要把重复的抽出来。
>
> 🐌 **笨办法**:创建一个新书架,从左到右扫描,每次遇到新书就放到新书架上,重复的就跳过。但这需要额外的书架空间,不符合"原地"要求。
>
> 🚀 **聪明办法**:不用新书架!你用左手(慢指针)指着"保留区"的最后一本书,右手(快指针)扫描后面的书。遇到和左手不同的新书,就把它移到左手下一个位置,左手前进。**因为数组已排序,重复的书一定相邻**,所以只需比较相邻元素!

### 关键洞察
**数组已排序 → 重复元素必相邻 → 只需比较相邻元素,用快慢指针维护"不重复区域"!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:已排序的整数数组 `nums`
- **输出**:删除重复后的新长度 `k`,且 `nums` 的前 `k` 个元素为不重复元素
- **限制**:必须原地修改(O(1)空间),不能创建新数组

### Step 2:先想笨办法(额外空间)
最直接的思路:遍历数组,用集合 `set` 去重,然后重建数组。
```python
# 需要额外空间 O(n)
seen = set()
result = []
for num in nums:
    if num not in seen:
        seen.add(num)
        result.append(num)
# 再把 result 拷贝回 nums
```
- 时间复杂度:O(n)
- 瓶颈在哪:**需要 O(n) 额外空间**,不符合原地修改要求

### Step 3:瓶颈分析 → 优化方向
- 核心问题:如何在**不使用额外数组**的情况下去重?
- 优化思路:能否直接在原数组上操作,把不重复元素"移到前面"?

**关键洞察**:
- 数组已排序 → **重复元素一定相邻**
- 只需比较 `nums[i]` 和 `nums[i-1]`,不同则保留
- 用**快慢指针**:
  - `slow`:指向"不重复区域"的末尾
  - `fast`:扫描整个数组,找新元素

### Step 4:选择武器
- 选用:**快慢指针**(双指针)
- 理由:慢指针维护结果区域,快指针探索新元素,遇到新元素就写入慢指针位置,O(1)空间完成原地去重

> 🔑 **模式识别提示**:当题目要求"原地操作"+"有序数组去重/移除元素",优先考虑"快慢指针"

---

## 🔑 解法一:暴力法(额外空间)

### 思路
使用额外的数据结构(如集合)去重,然后拷贝回原数组。这不符合题目要求,但可以作为思路起点。

### Python代码

```python
from typing import List


def removeDuplicates_extra_space(nums: List[int]) -> int:
    """
    解法一:使用额外空间
    思路:遍历数组,用有序集合保持不重复元素
    """
    if not nums:
        return 0

    # 利用 dict 保持插入顺序(Python 3.7+)
    seen = {}
    for num in nums:
        if num not in seen:
            seen[num] = True

    # 拷贝回原数组
    unique_nums = list(seen.keys())
    for i in range(len(unique_nums)):
        nums[i] = unique_nums[i]

    return len(unique_nums)


# ✅ 测试
test1 = [1, 1, 2]
k1 = removeDuplicates_extra_space(test1)
print(f"长度: {k1}, 数组: {test1[:k1]}")  # 期望: 长度: 2, 数组: [1, 2]

test2 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
k2 = removeDuplicates_extra_space(test2)
print(f"长度: {k2}, 数组: {test2[:k2]}")  # 期望: 长度: 5, 数组: [0, 1, 2, 3, 4]
```

### 复杂度分析
- **时间复杂度**:O(n) — 遍历数组一次
- **空间复杂度**:O(n) — 需要额外的字典/集合存储不重复元素

### 优缺点
- ✅ 思路清晰,易于理解
- ❌ **使用了额外空间**,不符合题目 O(1) 空间要求
- ❌ 没有利用"数组已排序"的条件

---

## ⚡ 解法二:快慢指针(原地去重)

### 优化思路
既然数组已排序,**重复元素一定相邻**,我们可以:
- 用 `slow` 指针维护"不重复区域" `[0, slow]`
- 用 `fast` 指针扫描整个数组,找新元素
- 当 `nums[fast] != nums[slow]` 时,说明找到新元素,写入 `nums[slow+1]`

> 💡 **关键想法**:数组已排序,只需比较相邻元素,快指针找到新元素就追加到慢指针后面!

### 图解过程

```
输入:nums = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]

初始化:slow = 0, fast = 1
[0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
 ↑  ↑
slow fast

Step 1: nums[fast]=0 == nums[slow]=0, 重复,fast++
[0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
 ↑     ↑
slow  fast

Step 2: nums[fast]=1 != nums[slow]=0, 新元素!
  slow++, nums[slow] = nums[fast], fast++
[0, 1, 1, 1, 1, 2, 2, 3, 3, 4]
    ↑     ↑
  slow   fast

Step 3: nums[fast]=1 == nums[slow]=1, 重复,fast++
[0, 1, 1, 1, 1, 2, 2, 3, 3, 4]
    ↑        ↑
  slow      fast

Step 4: nums[fast]=1 == nums[slow]=1, 重复,fast++
[0, 1, 1, 1, 1, 2, 2, 3, 3, 4]
    ↑           ↑
  slow         fast

Step 5: nums[fast]=2 != nums[slow]=1, 新元素!
  slow++, nums[slow] = nums[fast], fast++
[0, 1, 2, 1, 1, 2, 2, 3, 3, 4]
       ↑           ↑
     slow         fast

Step 6: nums[fast]=2 == nums[slow]=2, 重复,fast++
[0, 1, 2, 1, 1, 2, 2, 3, 3, 4]
       ↑              ↑
     slow            fast

Step 7: nums[fast]=3 != nums[slow]=2, 新元素!
  slow++, nums[slow] = nums[fast], fast++
[0, 1, 2, 3, 1, 2, 2, 3, 3, 4]
          ↑              ↑
        slow            fast

Step 8: nums[fast]=3 == nums[slow]=3, 重复,fast++
[0, 1, 2, 3, 1, 2, 2, 3, 3, 4]
          ↑                 ↑
        slow               fast

Step 9: nums[fast]=4 != nums[slow]=3, 新元素!
  slow++, nums[slow] = nums[fast], fast++
[0, 1, 2, 3, 4, 2, 2, 3, 3, 4]
             ↑                 ↑
           slow               fast(超出范围,结束)

最终结果:slow = 4, 返回 slow + 1 = 5
前5个元素: [0, 1, 2, 3, 4]
```

### Python代码

```python
def removeDuplicates(nums: List[int]) -> int:
    """
    解法二:快慢指针(原地去重)
    思路:慢指针维护不重复区域,快指针扫描找新元素
    """
    # 边界:空数组或单元素
    if not nums or len(nums) == 0:
        return 0

    # 初始化慢指针
    slow = 0  # slow 指向不重复区域的末尾

    # 快指针从第二个元素开始扫描
    for fast in range(1, len(nums)):
        # 如果发现新元素(与slow位置的元素不同)
        if nums[fast] != nums[slow]:
            slow += 1  # 慢指针前进
            nums[slow] = nums[fast]  # 将新元素写入慢指针位置

    # 返回不重复元素的个数(slow是索引,长度要+1)
    return slow + 1


# ✅ 测试
test1 = [1, 1, 2]
k1 = removeDuplicates(test1)
print(f"长度: {k1}, 数组: {test1[:k1]}")  # 期望: 长度: 2, 数组: [1, 2]

test2 = [0, 0, 1, 1, 1, 2, 2, 3, 3, 4]
k2 = removeDuplicates(test2)
print(f"长度: {k2}, 数组: {test2[:k2]}")  # 期望: 长度: 5, 数组: [0, 1, 2, 3, 4]

test3 = [1]
k3 = removeDuplicates(test3)
print(f"长度: {k3}, 数组: {test3[:k3]}")  # 期望: 长度: 1, 数组: [1]

test4 = [1, 1, 1, 1, 1]
k4 = removeDuplicates(test4)
print(f"长度: {k4}, 数组: {test4[:k4]}")  # 期望: 长度: 1, 数组: [1]
```

### 复杂度分析
- **时间复杂度**:O(n) — 快指针遍历数组一次,每个元素最多被访问两次(读和写)
  - 具体地说:如果输入规模 n=30000,最多需要 30000 次操作
- **空间复杂度**:O(1) — 只用了两个指针变量,原地修改

---

## 🐍 Pythonic 写法

利用 Python 的 `enumerate` 和列表推导式,可以更简洁:

```python
def removeDuplicates_pythonic(nums: List[int]) -> int:
    """
    Pythonic 写法:使用 enumerate 和条件判断
    """
    if not nums:
        return 0

    slow = 0
    for fast, num in enumerate(nums):
        if fast == 0 or num != nums[slow]:
            slow += 1
            nums[slow - 1] = num if fast == 0 else nums[fast]

    return slow


# 更极简的写法(利用切片赋值)
def removeDuplicates_slice(nums: List[int]) -> int:
    """
    使用切片去重(虽然不是严格原地,但很 Pythonic)
    """
    # 先去重
    unique = []
    for i, num in enumerate(nums):
        if i == 0 or num != nums[i - 1]:
            unique.append(num)

    # 切片赋值,修改原数组
    nums[:len(unique)] = unique
    return len(unique)
```

对于这道题,标准的快慢指针写法最清晰,建议在面试中使用。

> ⚠️ **面试建议**:这道题的标准解法就是快慢指针,代码简洁且高效。面试时直接给出这个解法,重点在于讲清楚**为什么可以原地修改**以及**如何利用数组已排序的特性**。

---

## 📊 解法对比

| 维度 | 解法一:额外空间 | 解法二:快慢指针 |
|------|--------------|--------------|
| 时间复杂度 | O(n) | O(n) |
| 空间复杂度 | O(n) | **O(1)** |
| 代码难度 | 简单 | 简单 |
| 面试推荐 | ⭐ | ⭐⭐⭐ |
| 适用场景 | 不关心空间,需要保留原数组 | 原地去重,符合题目要求 |

**面试建议**:直接给出解法二(快慢指针),这是这道题的标准解法,面试官期望看到的就是这个。关键在于解释清楚双指针的移动逻辑。

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你原地删除有序数组中的重复元素,返回新长度。

**你**:(审题30秒)好的,这道题要求原地修改已排序的数组,删除重复元素。让我先确认一下理解:
- 输入是**已排序**的数组
- 需要**原地修改**,O(1)空间
- 返回新长度,前 k 个元素为不重复元素

我的思路是用**快慢指针**:
- 因为数组已排序,重复元素一定相邻,所以我只需比较相邻元素
- 用 `slow` 指针维护"不重复区域"的末尾,用 `fast` 指针扫描整个数组
- 当 `nums[fast] != nums[slow]` 时,说明找到新元素,就把它写入 `nums[slow+1]`,然后 `slow++`
- 最后返回 `slow + 1` 作为新长度

时间复杂度 O(n),空间复杂度 O(1),符合要求。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def removeDuplicates(nums):
    if not nums:
        return 0

    slow = 0  # 慢指针指向不重复区域末尾

    for fast in range(1, len(nums)):  # 快指针从第二个元素开始
        if nums[fast] != nums[slow]:  # 发现新元素
            slow += 1  # 慢指针前进
            nums[slow] = nums[fast]  # 写入新元素

    return slow + 1  # 返回长度(索引+1)
```

关键点:
1. `slow` 从 0 开始,因为第一个元素一定保留
2. `fast` 从 1 开始,与 `slow` 比较
3. 只有遇到新元素才更新 `slow` 位置

**面试官**:测试一下?

**你**:用示例 `[0,0,1,1,1,2,2,3,3,4]` 走一遍:
- 初始:`slow=0` (指向第一个0),`fast=1`
- `fast=1`:nums[1]=0 等于 nums[0]=0,跳过
- `fast=2`:nums[2]=1 不等于 nums[0]=0,`slow++` 变1,`nums[1]=1`
- `fast=3,4`:都是1,跳过
- `fast=5`:nums[5]=2 不等于 nums[1]=1,`slow++` 变2,`nums[2]=2`
- 以此类推...
- 最后 `slow=4`,返回 5,前5个元素为 `[0,1,2,3,4]`,正确!

再测边界情况 `[1,1,1,1]`:
- `slow=0`,`fast` 从1扫到3,都等于1,跳过
- 返回 `slow+1=1`,结果 `[1]`,正确!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果数组未排序呢?" | 需要先排序(O(n log n)),或者用哈希表去重(O(n)时间 O(n)空间)。但排序会改变原数组顺序,哈希表会用额外空间,需要权衡。 |
| "能否保留最多 k 个重复元素?" | 可以!改成 `if slow < k-1 or nums[fast] != nums[slow - (k-1)]`,即比较当前元素与"k个位置之前"的元素。LeetCode 80 就是保留最多2个的变体。 |
| "为什么 slow 从 0 开始?" | 因为第一个元素一定保留,`slow` 指向不重复区域的末尾,初始只有第一个元素(索引0),所以从0开始。 |
| "能否用递归实现?" | 可以但不推荐。递归需要 O(n) 栈空间,不符合 O(1) 空间要求,且性能较差。迭代更直观高效。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:enumerate 优雅地获取索引和值
for i, num in enumerate(nums):
    print(f"索引 {i}: 值 {num}")

# 技巧2:边界条件简洁判断
if not nums or len(nums) == 0:  # 等价于 if len(nums) == 0
    return 0

# 简化为:
if not nums:  # 空列表的布尔值为 False
    return 0

# 技巧3:range 的灵活使用
for fast in range(1, len(nums)):  # 从索引1开始到len(nums)-1
    pass
```

### 💡 底层原理(选读)

> **为什么快慢指针可以原地修改?**
>
> 关键在于**写入位置永远不会超过读取位置**:
> - `slow` 指向待写入位置,`fast` 指向待读取位置
> - 因为有重复元素,`slow` 的增长速度 ≤ `fast`
> - 所以 `nums[slow]` 的位置要么是已经读过的(可以安全覆盖),要么是当前正在读的(相同位置)
> - 不会出现"还没读就被覆盖"的情况
>
> **为什么数组排序很重要?**
>
> - 如果数组未排序,重复元素可能分散在各处,如 `[1,3,1,2,3]`
> - 快慢指针只比较相邻元素,会漏掉不相邻的重复
> - 排序后重复元素聚集,如 `[1,1,2,3,3]`,一次扫描就能去重
> - 这就是为什么题目**强调已排序**

### 算法模式卡片 📐
- **模式名称**:快慢指针(原地去重/移除)
- **适用条件**:
  - 数组原地操作(O(1)空间)
  - 需要移除/去重/筛选元素
  - 通常数组已排序(方便比较相邻元素)
- **识别关键词**:"原地"、"删除"、"去重"、"移除"、"已排序"
- **模板代码**:
```python
def remove_pattern(nums: List[int]) -> int:
    """快慢指针原地移除/去重模板"""
    if not nums:
        return 0

    slow = 0  # 慢指针:写入位置

    for fast in range(len(nums)):  # 快指针:读取位置
        # 判断是否保留当前元素
        if should_keep(nums, fast, slow):
            nums[slow] = nums[fast]  # 写入
            slow += 1  # 慢指针前进

    return slow  # 返回新长度
```

### 易错点 ⚠️
1. **返回值写成 `slow` 而不是 `slow + 1`**
   - ❌ 错误:`return slow`
   - ⚠️ 为什么错:`slow` 是索引(从0开始),长度要 +1。例如 `[1,2,3]`,最后 `slow=2`,但长度是3
   - ✅ 正确:`return slow + 1`

2. **fast 从 0 开始而不是 1**
   - ❌ 错误:`for fast in range(len(nums))`
   - ⚠️ 为什么错:`fast=0` 时会与 `slow=0` 比较同一个元素,导致第一个元素被重复写入
   - ✅ 正确:`for fast in range(1, len(nums))`

3. **忘记边界检查**
   - ❌ 错误:直接进入循环,不检查空数组
   - ⚠️ 为什么错:空数组时 `nums[0]` 会索引越界
   - ✅ 正确:在开头加 `if not nums: return 0`

4. **slow 更新顺序错误**
   - ❌ 错误:`nums[slow] = nums[fast]; slow += 1` (先写再移动,漏掉第一个元素)
   - ⚠️ 为什么错:应该先移动再写,让 `slow` 指向下一个空位
   - ✅ 正确:`slow += 1; nums[slow] = nums[fast]`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:数据库去重查询** — SQL 中的 `SELECT DISTINCT` 在执行时,如果数据已按某列排序(建立了索引),数据库会使用类似快慢指针的算法,扫描一遍即可去重,避免使用哈希表的额外空间。

- **场景2:日志文件去重** — 运维系统处理海量日志时,如果日志按时间戳排序,可以用快慢指针算法原地去重相同的错误日志,节省存储空间。例如同一秒内的重复异常只保留一条。

- **场景3:流式数据去重** — 在数据流处理(如 Kafka 消费者)中,如果消息按某字段排序,可以用滑动窗口+快慢指针去除连续重复消息,减少下游系统压力。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 80. 删除有序数组中的重复项 II | Medium | 快慢指针变体 | 允许重复最多2次,比较 `nums[fast]` 和 `nums[slow-1]` |
| LeetCode 27. 移除元素 | Easy | 快慢指针 | 移除指定值,不要求排序 |
| LeetCode 283. 移动零 | Easy | 快慢指针 | 把所有0移到末尾,非0元素保持相对顺序 |
| LeetCode 88. 合并两个有序数组 | Easy | 双指针归并 | 从后往前用双指针合并,避免覆盖 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个有序数组,删除重复元素,但**每个元素最多保留2次**。例如 `[1,1,1,2,2,3]` → `[1,1,2,2,3]`(长度5)。

要求:原地修改,O(1)空间,返回新长度。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

把判断条件从"与 slow 位置比较"改为"与 slow-1 位置比较"!这样可以保留最多2个重复元素。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def removeDuplicates_keep_two(nums: List[int]) -> int:
    """
    保留最多2个重复元素
    核心:比较 nums[fast] 和 nums[slow-1]
    """
    if len(nums) <= 2:
        return len(nums)  # 长度<=2,一定满足条件

    slow = 2  # 前两个元素一定保留,从索引2开始

    for fast in range(2, len(nums)):
        # 如果当前元素 != slow-2 位置的元素,说明不会出现"连续3个相同"
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1

    return slow


# 测试
test = [1, 1, 1, 2, 2, 3]
k = removeDuplicates_keep_two(test)
print(f"长度: {k}, 数组: {test[:k]}")  # 期望: 长度: 5, 数组: [1, 1, 2, 2, 3]

test2 = [0, 0, 1, 1, 1, 1, 2, 3, 3]
k2 = removeDuplicates_keep_two(test2)
print(f"长度: {k2}, 数组: {test2[:k2]}")  # 期望: 长度: 7, 数组: [0, 0, 1, 1, 2, 3, 3]
```

**核心思路**:
- 前2个元素一定保留,所以 `slow` 从2开始
- 判断 `nums[fast]` 是否等于 `nums[slow-2]`:
  - 如果不等于,说明即使写入也不会出现"连续3个相同",可以保留
  - 如果等于,说明已经有2个相同的了,跳过这个元素
- 这个模式可以推广:**保留最多 k 个重复元素,就比较 `nums[fast]` 和 `nums[slow-k]`**

这展示了算法模式的**推广能力**——理解了快慢指针的本质,就能轻松解决各种变体问题!

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
