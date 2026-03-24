> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第55课:搜索插入位置

> **模块**:二分查找 | **难度**:Easy ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/search-insert-position/
> **前置知识**:第54课 二分查找
> **预计学习时间**:15分钟

---

## 🎯 题目描述

给定一个排序数组和一个目标值,在数组中找到目标值,并返回其索引。如果目标值不存在于数组中,返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 O(log n) 的算法。

**示例:**
```
输入:nums = [1,3,5,6], target = 5
输出:2
解释:5 在数组中,索引为2

输入:nums = [1,3,5,6], target = 2
输出:1
解释:2 应该插入在索引1的位置
```

**约束条件:**
- 1 ≤ nums.length ≤ 10^4
- -10^4 ≤ nums[i] ≤ 10^4
- nums 为无重复元素的升序排列数组
- -10^4 ≤ target ≤ 10^4

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1], target=1 | 0 | 单元素精确匹配 |
| 插入头部 | nums=[1,3,5], target=0 | 0 | 插入位置在开头 |
| 插入尾部 | nums=[1,3,5], target=7 | 3 | 插入位置在末尾 |
| 插入中间 | nums=[1,3,5,6], target=4 | 2 | 插入位置在中间 |
| 精确匹配 | nums=[1,3,5,6], target=3 | 1 | 目标值存在 |

---

## 💡 思路引导

### 生活化比喻
> 想象你要把一本新书插入到书架上,书架上的书已经按照书名字母排序好了。
>
> 🐌 **笨办法**:从头开始逐本检查,直到找到第一本字母比新书大的书,把新书插在它前面。如果书架有1000本书,最坏情况要检查1000次。
>
> 🚀 **聪明办法**:用"二分查找"思想 — 先看中间的书,如果新书比它小,就只看左半边,否则看右半边。每次都能排除一半的书,1000本书最多只需要检查10次(log₂1000 ≈ 10)。

### 关键洞察
**本题的核心是"左边界二分查找" — 找到第一个大于等于target的位置,这个位置就是插入位置!**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:有序数组nums + 目标值target
- **输出**:返回索引(如果找到) 或 插入位置(如果不存在)
- **限制**:必须用O(log n)算法 → 提示用二分查找

### Step 2:先想笨办法(暴力法)
从头到尾遍历数组,找到第一个大于等于target的位置:
```python
for i in range(len(nums)):
    if nums[i] >= target:
        return i
return len(nums)  # 如果都小于target,插入末尾
```
- 时间复杂度:O(n)
- 瓶颈在哪:没有利用"有序"这个特性,逐个检查太慢

### Step 3:瓶颈分析 → 优化方向
暴力法中每次只排除1个元素,效率低下。
- 核心问题:"有序数组"的信息被浪费了
- 优化思路:能不能每次排除一半的元素? → 用二分查找

### Step 4:选择武器
- 选用:**左边界二分查找**
- 理由:我们要找的是"第一个大于等于target的位置",这正是左边界二分的定义

> 🔑 **模式识别提示**:当题目出现"有序数组 + O(log n)要求",优先考虑"二分查找"

---

## 🔑 解法一:标准二分查找(直觉法)

### 思路
使用标准二分查找尝试找到target,如果没找到,根据最后的left指针位置返回插入点。

### 图解过程

```
示例: nums = [1,3,5,6], target = 5

初始状态:
  L           M           R
  ↓           ↓           ↓
[ 1,    3,    5,    6 ]
mid=1, nums[1]=3 < 5, 往右找

第2轮:
              L     M     R
              ↓     ↓     ↓
[ 1,    3,    5,    6 ]
mid=2, nums[2]=5 == 5, 找到了! 返回2


示例2: nums = [1,3,5,6], target = 2

初始状态:
  L           M           R
  ↓           ↓           ↓
[ 1,    3,    5,    6 ]
mid=1, nums[1]=3 > 2, 往左找

第2轮:
  L     R
  ↓     ↓
[ 1,    3,    5,    6 ]
  M=0
mid=0, nums[0]=1 < 2, 往右找

第3轮:
        L
        R
        ↓
[ 1,    3,    5,    6 ]
left > right, 循环结束, 返回left=1 (插入位置)
```

### Python代码

```python
from typing import List


def searchInsert(nums: List[int], target: int) -> int:
    """
    解法一:标准二分查找
    思路:用二分查找,如果找到返回索引,没找到返回left指针位置
    """
    left, right = 0, len(nums) - 1

    while left <= right:
        mid = left + (right - left) // 2  # 防止溢出

        if nums[mid] == target:
            return mid  # 找到目标值
        elif nums[mid] < target:
            left = mid + 1  # 目标在右半边
        else:
            right = mid - 1  # 目标在左半边

    # 循环结束时, left就是插入位置
    return left


# ✅ 测试
print(searchInsert([1, 3, 5, 6], 5))  # 期望输出:2
print(searchInsert([1, 3, 5, 6], 2))  # 期望输出:1
print(searchInsert([1, 3, 5, 6], 7))  # 期望输出:4
print(searchInsert([1, 3, 5, 6], 0))  # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(log n) — 每次循环排除一半元素
  - 具体地说:如果输入规模 n=10000,大约需要 log₂10000 ≈ 14 次比较
- **空间复杂度**:O(1) — 只用了几个指针变量

### 优缺点
- ✅ 思路直观,容易理解
- ✅ 时间复杂度已达最优O(log n)
- ⚠️ 需要理解为什么循环结束时left就是插入位置

---

## 🏆 解法二:左边界二分查找(最优解)

### 优化思路
用"左边界二分"的标准模板,直接找到"第一个大于等于target的位置",代码更简洁统一。

> 💡 **关键想法**:插入位置 = 第一个大于等于target的索引,这正是左边界二分的定义!

### 图解过程

```
示例: nums = [1,3,5,6], target = 2

左边界二分的核心思想:
"找到第一个 >= target 的位置"

  L                       R
  ↓                       ↓
[ 1,    3,    5,    6 ] (范围: 左闭右开 [0, 4))
        M=2
nums[2]=5 >= 2, 可能是答案, 继续往左找

  L           R
  ↓           ↓
[ 1,    3,    5,    6 ] (范围: [0, 2))
  M=1
nums[1]=3 >= 2, 可能是答案, 继续往左找

  L     R
  ↓     ↓
[ 1,    3,    5,    6 ] (范围: [0, 1))
  M=0
nums[0]=1 < 2, 不是答案, 往右找

        L
        R
        ↓
[ 1,    3,    5,    6 ] (范围: [1, 1))
left == right, 结束, 返回left=1

结论: 第一个 >= 2 的位置是索引1 (元素3)
```

### Python代码

```python
def searchInsert_v2(nums: List[int], target: int) -> int:
    """
    解法二:左边界二分查找 (最优解)
    思路:找到第一个大于等于target的位置
    """
    left, right = 0, len(nums)  # 注意: right = len(nums), 左闭右开区间

    while left < right:  # 注意: 不带等号
        mid = left + (right - left) // 2

        if nums[mid] < target:
            left = mid + 1  # [mid+1, right) 区间继续找
        else:
            # nums[mid] >= target, mid可能是答案, 保留它
            right = mid  # [left, mid) 区间继续找

    # 循环结束时 left == right, 就是插入位置
    return left


# ✅ 测试
print(searchInsert_v2([1, 3, 5, 6], 5))  # 期望输出:2
print(searchInsert_v2([1, 3, 5, 6], 2))  # 期望输出:1
print(searchInsert_v2([1, 3, 5, 6], 7))  # 期望输出:4
print(searchInsert_v2([1, 3, 5, 6], 0))  # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(log n) — 与解法一相同
- **空间复杂度**:O(1) — 与解法一相同

**为什么是最优解**:
- 时间O(log n)已经是理论最优(必须至少检查log n个元素来定位)
- 空间O(1)也是最优(原地操作)
- 代码模板统一,适用于所有边界查找问题(左边界、右边界)

---

## 🐍 Pythonic 写法

利用 Python 的 `bisect` 模块(标准库内置的二分查找):

```python
import bisect

def searchInsert_pythonic(nums: List[int], target: int) -> int:
    """
    Pythonic写法:使用bisect.bisect_left
    bisect_left(nums, target) 返回第一个 >= target 的位置
    """
    return bisect.bisect_left(nums, target)


# ✅ 测试
print(searchInsert_pythonic([1, 3, 5, 6], 5))  # 期望输出:2
print(searchInsert_pythonic([1, 3, 5, 6], 2))  # 期望输出:1
```

**说明**:
- `bisect.bisect_left(nums, target)` 本质就是左边界二分查找
- 一行代码解决问题,但面试时仍需手写以展示算法理解

> ⚠️ **面试建议**:先手写解法二展示二分查找功底,通过后再提`bisect`展示对Python标准库的了解。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:标准二分 | 🏆 解法二:左边界二分(最优) | Pythonic:bisect |
|------|--------------|------------------------|----------------|
| 时间复杂度 | O(log n) | **O(log n)** | O(log n) |
| 空间复杂度 | O(1) | **O(1)** | O(1) |
| 代码难度 | 中等(需理解返回值) | **简单(模板统一)** | 极简(一行) |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 | ⭐(辅助) |
| 适用场景 | 精确查找为主 | **所有边界查找问题** | Python快速实现 |

**为什么解法二是最优**:
- 时间空间都已达理论最优O(log n) / O(1)
- 代码模板统一,可以直接套用到"查找左边界/右边界"等变体题
- 逻辑清晰:"找第一个>=target" 直接对应题意

**面试建议**:
1. 先用30秒口述暴力法思路(O(n)遍历),表明你理解题意
2. 立即优化到🏆解法二(左边界二分),重点讲解"为什么left就是插入位置"
3. 手动测试边界用例:target比所有元素小/大、target在中间不存在、target存在
4. 如果时间充裕,提一下bisect模块展示Python功底

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求在有序数组中找到target或返回插入位置,并且要求O(log n)时间复杂度。让我先想一下...

我的第一个想法是暴力遍历,从头到尾找第一个大于等于target的位置,时间复杂度是O(n)。

不过题目要求O(log n),这提示我们要用二分查找。核心洞察是:插入位置就是"第一个大于等于target的索引",这正好是左边界二分查找的定义。

**面试官**:很好,请写一下代码。

**你**:(边写边说)我用左闭右开区间[left, right)来写,这样边界处理更统一:
- 初始化left=0, right=len(nums)
- 每次取中点mid,如果nums[mid] < target,说明答案在右边,更新left=mid+1
- 如果nums[mid] >= target,说明mid可能是答案,保留它,更新right=mid
- 循环结束时left就是答案

**面试官**:测试一下?

**你**:用示例[1,3,5,6], target=2走一遍:
- 初始left=0, right=4, mid=2, nums[2]=5 >= 2, 更新right=2
- left=0, right=2, mid=1, nums[1]=3 >= 2, 更新right=1
- left=0, right=1, mid=0, nums[0]=1 < 2, 更新left=1
- left=1, right=1, 循环结束, 返回1 ✅

再测边界情况target=7:
- 会一直往右找,最终left=4,正好是数组长度,表示插入末尾 ✅

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间O(log n)已经是理论最优,因为二分每次排除一半,不可能更快。空间O(1)也是最优。 |
| "为什么循环结束时left就是插入位置?" | 因为我们维护的不变量是:[left, right)区间内所有元素都 < target, right及右边的元素都 >= target。循环结束时left==right,就是第一个 >= target 的位置。 |
| "如果有重复元素怎么办?" | 当前解法会返回第一个target的位置,如果要找最后一个,用右边界二分。 |
| "能用Python标准库吗?" | 可以用bisect.bisect_left(nums, target),一行解决,但手写更能展示算法理解。 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:防止整数溢出的中点计算
mid = left + (right - left) // 2  # 推荐,防止 left+right 溢出
# 而不是: mid = (left + right) // 2

# 技巧2:bisect模块 — Python内置二分查找
import bisect
pos = bisect.bisect_left(nums, target)   # 左边界:第一个 >= target
pos = bisect.bisect_right(nums, target)  # 右边界:第一个 > target
bisect.insort_left(nums, target)         # 插入并保持有序
```

### 💡 底层原理(选读)

> **为什么二分查找这么快?**
>
> 二分查找的时间复杂度O(log n)来自于"每次排除一半"的策略:
> - 1000个元素 → 500 → 250 → 125 → 63 → 32 → 16 → 8 → 4 → 2 → 1
> - 只需要10次就能定位,而遍历需要1000次
> - 对于10^4个元素,二分只需14次,对于10^9个元素,只需30次!
>
> **左闭右开 vs 左闭右闭?**
> - 左闭右开[left, right):循环条件`left < right`,更新`right = mid`
> - 左闭右闭[left, right]:循环条件`left <= right`,更新`right = mid - 1`
> - 两种都正确,推荐左闭右开,因为边界处理更统一(right直接等于长度)

### 算法模式卡片 📐
- **模式名称**:左边界二分查找
- **适用条件**:有序数组 + 查找"第一个满足条件"的元素
- **识别关键词**:"有序"、"第一个大于等于"、"插入位置"、"O(log n)"
- **模板代码**:
```python
def left_bound(nums: List[int], target: int) -> int:
    """左边界二分:找第一个 >= target 的位置"""
    left, right = 0, len(nums)  # 左闭右开
    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid  # 保留mid,继续往左找
    return left
```

### 易错点 ⚠️
1. **边界条件混淆**
   - 错误:`while left <= right` 配合 `right = len(nums)` → 会越界
   - 正确:左闭右开用`<`,左闭右闭用`<=`

2. **返回值理解错误**
   - 错误:以为left只在找到target时才是正确答案
   - 正确:循环结束时,left永远是"第一个 >= target 的位置",找不到时就是插入点

3. **mid计算溢出**(Python不会溢出,但C++/Java会)
   - 错误:`mid = (left + right) // 2` → left+right可能溢出
   - 正确:`mid = left + (right - left) // 2`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:数据库索引 — MySQL的B+树索引本质就是多路平衡二分查找,将O(n)的顺序扫描优化为O(log n)
- **场景2**:Git bisect命令 — 用二分法快速定位引入bug的提交,从几千个commit中只需十几次就能找到
- **场景3**:版本控制 — 在有序的版本号列表中快速定位某个功能首次出现的版本
- **场景4**:推荐系统 — 在按评分排序的商品列表中,快速找到第一个评分>=4星的商品

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 34. 在排序数组中查找元素的首末位置 | Medium | 左右边界二分 | 分别用左边界和右边界二分查找 |
| LeetCode 69. x的平方根 | Easy | 二分查找答案 | 在[0, x]区间二分查找答案 |
| LeetCode 278. 第一个错误版本 | Easy | 左边界二分 | 找第一个返回true的版本 |
| LeetCode 153. 寻找旋转排序数组中的最小值 | Medium | 二分变体 | 判断哪半边有序来调整指针 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定有序数组nums和目标值target,找到target在数组中最后一次出现的位置,如果不存在返回-1。例如nums=[5,7,7,8,8,10], target=8,返回4。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

这是"右边界二分查找" — 找到最后一个等于target的位置,即"最后一个 <= target 的位置"。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def searchLast(nums: List[int], target: int) -> int:
    """
    右边界二分:找最后一个 <= target 的位置
    """
    left, right = 0, len(nums)

    while left < right:
        mid = left + (right - left) // 2
        if nums[mid] <= target:
            left = mid + 1  # 继续往右找
        else:
            right = mid

    # left-1 是最后一个 <= target 的位置
    if left > 0 and nums[left - 1] == target:
        return left - 1
    return -1


# 测试
print(searchLast([5, 7, 7, 8, 8, 10], 8))  # 输出:4
```

核心思路:右边界二分找"最后一个 <= target",然后检查是否等于target。注意返回的是`left-1`而不是`left`。

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
