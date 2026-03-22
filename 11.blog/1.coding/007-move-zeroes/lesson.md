# 📖 第7课:移动零

> **模块**:双指针 | **难度**:Easy ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/move-zeroes/
> **前置知识**:无(双指针入门题)
> **预计学习时间**:15分钟

---

## 🎯 题目描述

给定一个数组 `nums`,编写一个函数将所有 `0` 移动到数组的末尾,同时保持非零元素的**相对顺序**。

**注意**:必须在**原地**对数组进行操作,不能使用额外的数组空间。

**示例:**
```
输入:nums = [0,1,0,3,12]
输出:[1,3,12,0,0]
```

```
输入:nums = [0]
输出:[0]
```

**约束条件:**
- 1 ≤ nums.length ≤ 10⁴
- -2³¹ ≤ nums[i] ≤ 2³¹ - 1

**进阶**:你能尽量减少操作次数吗?

---

## 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单个元素 | [0] 或 [1] | [0] 或 [1] | 基本功能 |
| 全是零 | [0,0,0] | [0,0,0] | 特殊情况 |
| 无零 | [1,2,3] | [1,2,3] | 无需移动 |
| 零在开头 | [0,0,1,2] | [1,2,0,0] | 零的移动 |
| 零在末尾 | [1,2,0,0] | [1,2,0,0] | 已经有序 |
| 交替出现 | [0,1,0,2,0,3] | [1,2,3,0,0,0] | 保持相对顺序 |

---

## 💡 思路引导

### 生活化比喻

> 想象你在整理一排书架(数组),要把所有空位(0)挪到最右边,但书的相对顺序不能变。
>
> 🐌 **笨办法**:你拿个箱子,先把所有书拿下来放进箱子(非零元素),记住顺序,然后再把书按原顺序放回书架左边,最后空位自然在右边。但这需要一个额外的箱子(额外空间)。
>
> 🚀 **聪明办法**:你用两只手(双指针):
> - **左手(slow)**:指向下一本书应该放的位置
> - **右手(fast)**:逐个检查每个位置
>
> 右手遇到书时,就把书移到左手位置,然后左手右移一格。右手扫完后,左手右边的都是空位!

### 关键洞察

**快慢指针:slow指向下一个非零元素的目标位置,fast遍历数组找非零元素!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组,可能包含0
- **输出**:原地修改数组,将0移到末尾
- **限制**:
  - 必须原地操作(O(1)空间)
  - 保持非零元素相对顺序
  - 尽量减少操作次数

### Step 2:先想笨办法(借助额外空间)

最直接的想法:
1. 遍历数组,把所有非零元素按顺序放到新数组
2. 再把剩余位置填0
3. 将新数组的值复制回原数组

```python
# 伪代码
non_zero = [x for x in nums if x != 0]
for i in range(len(non_zero)):
    nums[i] = non_zero[i]
for i in range(len(non_zero), len(nums)):
    nums[i] = 0
```

- 时间复杂度:O(n)
- 空间复杂度:O(n) — 需要额外数组
- 瓶颈在哪:**违反了"原地操作"的要求**

### Step 3:瓶颈分析 → 优化方向

题目要求原地操作,不能用额外空间。我们需要在遍历过程中**边找边放**。

核心问题:**如何用O(1)空间完成元素的移动?**

优化思路:用**两个指针**:
- `slow`:指向下一个非零元素应该放置的位置
- `fast`:遍历数组,寻找非零元素

当 `fast` 找到非零元素时,放到 `slow` 位置,然后 `slow++`

### Step 4:选择武器
- 选用:**快慢双指针(同向双指针)**
- 理由:
  - slow 记录写入位置,fast 扫描数组
  - 两个指针都向右移动,永不回头
  - O(1) 空间,O(n) 时间

> 🔑 **模式识别提示**:当题目要求"原地操作"+"保持顺序",考虑"快慢指针"

---

## 🔑 解法一:双指针覆盖法(标准)

### 思路

使用快慢指针:
- `slow`:指向下一个非零元素应该放的位置
- `fast`:遍历整个数组

遍历两遍:
1. 第一遍:将所有非零元素移到数组前面
2. 第二遍:将 slow 后面的位置全部填0

### 图解过程

```
初始数组:[0, 1, 0, 3, 12]
         ↑  ↑
       slow fast

Step 1:fast=0 指向 0,跳过
  [0, 1, 0, 3, 12]
   ↑     ↑
 slow  fast

Step 2:fast=1 指向 1(非零),放到 slow=0 位置
  [1, 1, 0, 3, 12]
      ↑     ↑
    slow  fast
  slow++ → slow=1

Step 3:fast=2 指向 0,跳过
  [1, 1, 0, 3, 12]
      ↑        ↑
    slow     fast

Step 4:fast=3 指向 3(非零),放到 slow=1 位置
  [1, 3, 0, 3, 12]
         ↑     ↑
       slow  fast
  slow++ → slow=2

Step 5:fast=4 指向 12(非零),放到 slow=2 位置
  [1, 3, 12, 3, 12]
            ↑      (fast结束)
          slow

第二遍:将 slow=2 之后的位置填0
  [1, 3, 12, 0, 0]
```

### Python代码

```python
from typing import List


def moveZeroes(nums: List[int]) -> None:
    """
    解法一:快慢指针(双遍历)
    思路:第一遍移动非零元素,第二遍填充0
    """
    n = len(nums)
    slow = 0  # 指向下一个非零元素应该放的位置

    # 第一遍:将所有非零元素移到前面
    for fast in range(n):
        if nums[fast] != 0:
            nums[slow] = nums[fast]
            slow += 1

    # 第二遍:将 slow 后面的位置全部填0
    for i in range(slow, n):
        nums[i] = 0


# ✅ 测试
nums1 = [0, 1, 0, 3, 12]
moveZeroes(nums1)
print(nums1)  # 期望输出:[1, 3, 12, 0, 0]

nums2 = [0]
moveZeroes(nums2)
print(nums2)  # 期望输出:[0]

nums3 = [1, 2, 3]
moveZeroes(nums3)
print(nums3)  # 期望输出:[1, 2, 3]
```

### 复杂度分析
- **时间复杂度**:O(n) — 两次遍历,2n次操作
- **空间复杂度**:O(1) — 只用了两个指针变量

### 优缺点
- ✅ 逻辑清晰,易于理解
- ✅ 满足原地操作要求
- ❌ 需要两遍遍历 → 可以优化为一遍!

---

## ⚡ 解法二:快慢指针交换法(优化)

### 优化思路

解法一需要两遍遍历。其实可以在一遍遍历中完成:
- 当 `fast` 指向非零元素时,与 `slow` 位置**交换**
- 这样每次交换都保证 slow 之前全是非零元素

> 💡 **关键想法**:用交换代替覆盖+填充,一次遍历搞定!

### 图解过程

```
初始数组:[0, 1, 0, 3, 12]
         ↑  ↑
       slow fast

Step 1:fast=0, nums[0]=0, 是0,fast++
  [0, 1, 0, 3, 12]
   ↑     ↑
 slow  fast

Step 2:fast=1, nums[1]=1, 非零,交换 nums[slow] 和 nums[fast]
  交换 nums[0]=0 和 nums[1]=1
  [1, 0, 0, 3, 12]
      ↑     ↑
    slow  fast
  slow++, fast++

Step 3:fast=2, nums[2]=0, 是0,fast++
  [1, 0, 0, 3, 12]
      ↑        ↑
    slow     fast

Step 4:fast=3, nums[3]=3, 非零,交换 nums[slow] 和 nums[fast]
  交换 nums[1]=0 和 nums[3]=3
  [1, 3, 0, 0, 12]
         ↑     ↑
       slow  fast
  slow++, fast++

Step 5:fast=4, nums[4]=12, 非零,交换 nums[slow] 和 nums[fast]
  交换 nums[2]=0 和 nums[4]=12
  [1, 3, 12, 0, 0]
            ↑      (结束)
          slow

结果:[1, 3, 12, 0, 0]
```

### Python代码

```python
from typing import List


def moveZeroes_v2(nums: List[int]) -> None:
    """
    解法二:快慢指针(一遍遍历+交换)
    思路:遇到非零元素时,与 slow 位置交换
    """
    slow = 0  # 指向下一个非零元素应该放的位置

    # 一次遍历完成
    for fast in range(len(nums)):
        if nums[fast] != 0:
            # 交换 slow 和 fast 位置的元素
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


# ✅ 测试
nums1 = [0, 1, 0, 3, 12]
moveZeroes_v2(nums1)
print(nums1)  # 期望输出:[1, 3, 12, 0, 0]

nums2 = [0, 0, 1]
moveZeroes_v2(nums2)
print(nums2)  # 期望输出:[1, 0, 0]
```

### 复杂度分析
- **时间复杂度**:O(n) — 一次遍历,最多n次交换
- **空间复杂度**:O(1) — 只用了指针变量

---

## 🚀 解法三:优化交换(减少不必要交换)

### 优化思路

解法二在 slow 和 fast 相同时也会交换自己,这是不必要的。
优化:只有 `slow != fast` 时才交换。

### Python代码

```python
from typing import List


def moveZeroes_v3(nums: List[int]) -> None:
    """
    解法三:优化版快慢指针
    思路:避免自己和自己交换
    """
    slow = 0

    for fast in range(len(nums)):
        if nums[fast] != 0:
            if slow != fast:  # 只有位置不同时才交换
                nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1


# ✅ 测试
nums1 = [0, 1, 0, 3, 12]
moveZeroes_v3(nums1)
print(nums1)  # 期望输出:[1, 3, 12, 0, 0]

nums2 = [1, 2, 3, 4, 5]
moveZeroes_v3(nums2)
print(nums2)  # 期望输出:[1, 2, 3, 4, 5] (无交换)
```

### 复杂度分析
- **时间复杂度**:O(n) — 一次遍历
- **空间复杂度**:O(1)
- **优化**:减少了不必要的自我交换

---

## 🐍 Pythonic 写法

使用列表推导式(但会违反原地操作要求):

```python
def moveZeroes_pythonic(nums):
    """Pythonic 写法(不符合原地操作要求)"""
    non_zero = [x for x in nums if x != 0]
    zeros = [0] * (len(nums) - len(non_zero))
    nums[:] = non_zero + zeros  # 修改原列表

# 测试
nums = [0, 1, 0, 3, 12]
moveZeroes_pythonic(nums)
print(nums)  # [1, 3, 12, 0, 0]
```

虽然简洁,但本质上是解法一,且使用了额外空间。

> ⚠️ **面试建议**:先写解法二展示双指针思路,再提解法三说明优化点。
> 面试官更看重你对**快慢指针**的理解,而非代码简洁度。

---

## 📊 解法对比

| 维度 | 解法一:双遍历 | 解法二:一遍交换 | 解法三:优化交换 |
|------|-------------|---------------|---------------|
| 时间复杂度 | O(n) | O(n) | O(n) |
| 空间复杂度 | O(1) | O(1) | O(1) |
| 遍历次数 | 2次 | 1次 | 1次 |
| 操作次数 | ~2n | ≤n | <n |
| 代码难度 | 简单 | 简单 | 简单 |
| 面试推荐 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**面试建议**:首选解法二或解法三,体现对快慢指针的熟练掌握。

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求将数组中的所有0移到末尾,保持非零元素的相对顺序,并且要原地操作。

我的第一个想法是创建一个新数组,先放非零元素,再填0,最后复制回原数组。但这需要 O(n) 额外空间,违反了原地操作要求。

我们可以用**快慢双指针**优化:
- slow 指针指向下一个非零元素应该放的位置
- fast 指针遍历数组,寻找非零元素

当 fast 找到非零元素时,与 slow 位置交换,然后 slow 右移。一次遍历就能完成,时间 O(n),空间 O(1)。

**面试官**:很好,请写一下代码。

**你**:(边写边说)我用两个指针 slow 和 fast,都从0开始。遍历数组,如果 nums[fast] 不为0,就交换 nums[slow] 和 nums[fast],然后 slow++。这样保证 slow 左边全是非零元素。(写下解法二的代码)

**面试官**:测试一下?

**你**:用示例 [0,1,0,3,12] 走一遍:
- fast=0, nums[0]=0,跳过
- fast=1, nums[1]=1,非零,交换 nums[0] 和 nums[1],数组变为 [1,0,0,3,12],slow=1
- fast=2, nums[2]=0,跳过
- fast=3, nums[3]=3,非零,交换 nums[1] 和 nums[3],数组变为 [1,3,0,0,12],slow=2
- fast=4, nums[4]=12,非零,交换 nums[2] 和 nums[4],数组变为 [1,3,12,0,0],slow=3
- 结果:[1,3,12,0,0],正确!

**面试官**:能不能减少操作次数?

**你**:可以!当 slow 和 fast 相同时,交换是多余的。我们加一个判断 `if slow != fast` 再交换,这样当数组前面都是非零元素时,不会做无效交换。(写下解法三的代码)

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么用交换而不是赋值?" | 交换能一次遍历完成,赋值需要两次遍历(先移动非零,再填0) |
| "slow 的含义是什么?" | slow 是下一个非零元素的目标位置,也是当前已处理的非零元素个数 |
| "如果要把非零元素移到末尾?" | slow 从右往左移,fast 从右往左扫,逻辑不变 |
| "能不能不用交换?" | 可以用解法一的两遍遍历,但操作次数更多 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍

```python
# 技巧1:Python 交换 — 不需要临时变量
a, b = b, a  # 内部实现是元组打包解包

# 技巧2:列表切片赋值 — 原地修改
nums[:] = new_list  # 修改原列表,而非创建新引用
nums = new_list     # 这只是改变了 nums 的引用!

# 技巧3:列表推导式过滤
non_zero = [x for x in nums if x != 0]

# 技巧4:count() 统计
zero_count = nums.count(0)
```

### 💡 底层原理(选读)

> **为什么快慢指针能保证相对顺序?**
>
> 因为 slow 和 fast 都是从左往右移动,永不回头:
> - fast 按顺序扫描每个元素
> - slow 按顺序分配位置
> - 先遇到的非零元素先被放到前面,所以相对顺序不变
>
> **交换 vs 覆盖?**
> - 解法一:先覆盖(非零移到前面),再填充(后面补0)
> - 解法二:直接交换,一步到位
> - 交换的好处:只需一次遍历,且对称操作更直观

### 算法模式卡片 📐

- **模式名称**:快慢双指针(同向双指针)
- **适用条件**:
  - 需要原地操作数组
  - 保持元素相对顺序
  - 区分两类元素(如零/非零、奇/偶)
- **识别关键词**:"原地操作"、"移动"、"保持顺序"、"删除重复"
- **核心思想**:slow 记录写入位置,fast 扫描读取
- **模板代码**:

```python
def two_pointer_template(nums):
    """快慢指针通用模板"""
    slow = 0  # 写指针:下一个满足条件元素的目标位置

    for fast in range(len(nums)):  # 读指针:遍历数组
        if condition(nums[fast]):  # 判断是否满足条件
            nums[slow] = nums[fast]  # 或者交换
            slow += 1

    # slow 左边是满足条件的元素
    # slow 右边是不满足条件的元素
    return slow  # slow 也是满足条件的元素个数
```

### 易错点 ⚠️

1. **忘记原地操作**
   - 错误:创建新数组 `result = []`,违反要求
   - 正确:直接修改 `nums`,使用 `nums[:] = ...` 或交换

2. **相对顺序被破坏**
   - 错误:从后往前填充非零元素,顺序会反
   - 正确:从前往后处理,保持遍历顺序

3. **多余的自我交换**
   - 不影响正确性,但增加操作次数
   - 优化:加判断 `if slow != fast`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:日志过滤** — 在日志处理系统中,原地删除空行或无效记录,保持有效记录的时间顺序。

- **场景2:数据清洗** — 在数据分析中,原地移除缺失值(如 NaN),保持数据的原始顺序,节省内存。

- **场景3:内存整理** — 在内存管理中,垃圾回收后将有效对象整理到内存前部,释放后部空间,类似"标记-清除-整理"算法。

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 27. 移除元素 | Easy | 快慢指针 | 和本题几乎一样,移除特定值而非0 |
| LeetCode 26. 删除有序数组中的重复项 | Easy | 快慢指针 | 保留一个,删除重复,原地操作 |
| LeetCode 80. 删除有序数组中的重复项II | Medium | 快慢指针 | 保留至多两个,更复杂的条件 |
| LeetCode 844. 比较含退格的字符串 | Easy | 双指针 | 模拟退格操作,从后往前双指针 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个数组,将所有负数移到数组开头,正数移到数组末尾,保持相对顺序。要求原地操作。

例如:[1, -2, 3, -4, 5] → [-2, -4, 1, 3, 5]

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

还是快慢指针!只是条件从"非零"改为"负数"。slow 指向下一个负数应该放的位置。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def moveNegatives(nums):
    """将负数移到前面,正数移到后面"""
    slow = 0  # 下一个负数的目标位置

    for fast in range(len(nums)):
        if nums[fast] < 0:  # 找到负数
            nums[slow], nums[fast] = nums[fast], nums[slow]
            slow += 1

    return nums


# 测试
print(moveNegatives([1, -2, 3, -4, 5]))  # [-2, -4, 1, 3, 5]
print(moveNegatives([-1, -2, -3]))       # [-1, -2, -3]
print(moveNegatives([1, 2, 3]))          # [1, 2, 3]
```

**核心思路**:
- 与"移动零"完全一样的模式
- 只是判断条件从 `!= 0` 改为 `< 0`
- slow 左边是负数,右边是正数

这就是快慢指针的威力:模板固定,条件可变!

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
