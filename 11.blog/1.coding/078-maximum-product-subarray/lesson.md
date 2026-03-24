> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第78课:乘积最大子数组

> **模块**:动态规划 | **难度**:Medium ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/maximum-product-subarray/
> **前置知识**:第73课(打家劫舍)、第77课(最长递增子序列)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个整数数组 nums,找到一个具有最大乘积的连续子数组,返回该子数组的乘积。

**示例:**
```
输入:nums = [2,3,-2,4]
输出:6
解释:子数组 [2,3] 的乘积最大为 6
```

**约束条件:**
- 1 ≤ nums.length ≤ 2×10⁴
- -10 ≤ nums[i] ≤ 10
- 保证答案在 32 位整数范围内

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[2] | 2 | 单元素处理 |
| 全正数 | nums=[2,3,4] | 24 | 连续相乘 |
| 含单个负数 | nums=[2,3,-2,4] | 6 | 跳过负数 |
| 偶数个负数 | nums=[-2,3,-4] | 24 | 负负得正 |
| 含零 | nums=[2,0,-3,4] | 4 | 零分割数组 |
| 全负数 | nums=[-2,-3,-4] | 12 | 取偶数个 |
| 大规模 | n=20000 | — | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在玩一个"连续翻倍"的游戏,每次可以选择继续翻倍或重新开始。
>
> 🐌 **笨办法**:枚举所有连续子数组(共n²个),逐个计算乘积,记录最大值——这需要O(n³)时间(两层循环枚举+一层循环计算乘积)。
>
> 🚀 **聪明办法**:遍历数组时,维护"到当前位置的最大乘积"和"最小乘积"。为什么要最小?因为负数可能翻盘!当遇到负数时,之前的"最小负数"乘以当前负数,反而变成"最大正数"。
>
> 🎯 **关键洞察**:遇到负数时,"最大"和"最小"身份互换!

### 关键洞察
**负数的特殊性:之前的最小值(可能是负数)×当前负数 = 最大正数,所以必须同时跟踪最大和最小值。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:整数数组 nums,元素可正可负可为0,长度 n
- **输出**:最大乘积(整数),注意是连续子数组
- **限制**:子数组必须连续,不能跳过元素

### Step 2:先想笨办法(暴力法)
枚举所有起点i和终点j(i ≤ j),计算每个子数组的乘积,记录最大值。
- 时间复杂度:O(n²) — 两层循环枚举,每次O(1)计算(维护累乘)
- 瓶颈在哪:n=20000时需要2亿次比较操作

### Step 3:瓶颈分析 → 优化方向
观察暴力法:对于每个位置i,我们重复计算了很多"包含前面元素"的乘积。
- 核心问题:无法利用"前面算过的结果"
- 特殊难点:负数会让大小关系翻转(最大变最小,最小变最大)
- 优化思路:能不能用DP记住"到前一个位置的最大/最小乘积"?

### Step 4:选择武器
- 选用:**动态规划(同时维护最大和最小值)**
- 理由:
  - 最优子结构:当前位置的最大乘积 = max(当前数, 前最大×当前数, 前最小×当前数)
  - 关键技巧:因为负数翻转大小关系,必须同时跟踪最大最小值
  - 一次遍历O(n)解决

> 🔑 **模式识别提示**:当题目出现"最大/最小"且涉及"乘法"(符号会变)时,考虑"同时维护最大最小值的DP"

---

## 🔑 解法一:双变量DP(直觉法)

### 思路
维护两个变量:当前最大乘积max_prod和当前最小乘积min_prod。遍历数组时:
- 如果当前数为正:max_prod继续扩大,min_prod继续缩小
- 如果当前数为负:max_prod和min_prod互换(最小负数×负数=最大正数)
- 如果当前数为0:重置为0

### 图解过程

```
输入:nums = [2, 3, -2, 4]

初始化:
  max_prod = nums[0] = 2
  min_prod = nums[0] = 2
  result = 2

Step 1:遍历nums[1]=3(正数)
  备份max_prod=2
  max_prod = max(3, 2×3, 2×3) = 6
  min_prod = min(3, 2×3, 2×3) = 3
  result = max(2, 6) = 6

  当前状态:[2,3] 最大乘积=6

Step 2:遍历nums[2]=-2(负数!)
  备份max_prod=6, min_prod=3
  max_prod = max(-2, 6×(-2), 3×(-2)) = max(-2, -12, -6) = -2
  min_prod = min(-2, 6×(-2), 3×(-2)) = min(-2, -12, -6) = -12
  result = max(6, -2) = 6 (保持)

  当前状态:跳过负数,[2,3]仍是最优

Step 3:遍历nums[3]=4(正数)
  备份max_prod=-2, min_prod=-12
  max_prod = max(4, -2×4, -12×4) = max(4, -8, -48) = 4
  min_prod = min(4, -2×4, -12×4) = min(4, -8, -48) = -48
  result = max(6, 4) = 6

  最终结果:6 (子数组[2,3])
```

**负数翻转示例:**
```
输入:nums = [-2, 3, -4]

Step 1:nums[0]=-2
  max_prod = -2
  min_prod = -2
  result = -2

Step 2:nums[1]=3
  max_prod = max(3, -2×3, -2×3) = 3
  min_prod = min(3, -2×3, -2×3) = -6
  result = 3

Step 3:nums[2]=-4(负数翻转!)
  备份max_prod=3, min_prod=-6
  max_prod = max(-4, 3×(-4), -6×(-4)) = max(-4, -12, 24) = 24 ✓
                                      ↑
                                最小值×负数=最大值!
  min_prod = min(-4, 3×(-4), -6×(-4)) = -12
  result = 24

  最终结果:24 (完整数组[-2,3,-4])
```

### Python代码

```python
from typing import List


def maxProduct_dp(nums: List[int]) -> int:
    """
    解法一:双变量DP(标准解法)
    思路:同时维护当前最大和最小乘积,遇负数时可能翻转
    """
    if not nums:
        return 0

    # 初始化:以第一个元素开始
    max_prod = min_prod = result = nums[0]

    # 从第二个元素开始遍历
    for num in nums[1:]:
        # 遇到负数时,最大最小值会互换,所以先备份
        prev_max = max_prod
        prev_min = min_prod

        # 当前最大值 = max(当前数单独, 前最大×当前, 前最小×当前)
        max_prod = max(num, prev_max * num, prev_min * num)

        # 当前最小值 = min(当前数单独, 前最大×当前, 前最小×当前)
        min_prod = min(num, prev_max * num, prev_min * num)

        # 更新全局最大值
        result = max(result, max_prod)

    return result


# ✅ 测试
print(maxProduct_dp([2,3,-2,4]))      # 期望输出:6
print(maxProduct_dp([-2,3,-4]))       # 期望输出:24
print(maxProduct_dp([0,2]))           # 期望输出:2
print(maxProduct_dp([-2,0,-1]))       # 期望输出:0
```

### 复杂度分析
- **时间复杂度**:O(n) — 一次遍历,每个元素访问一次
  - 具体地说:如果输入规模 n=20000,只需 20000 次操作,约 0.0002秒
- **空间复杂度**:O(1) — 只用3个变量(max_prod, min_prod, result)

### 优缺点
- ✅ 思路清晰,代码简洁(15行)
- ✅ 时间空间都已最优(O(n)和O(1))
- ✅ 一次遍历,适合流式数据
- ⚠️ 需要理解"为什么要维护最小值"(负数翻转)

---

## 🏆 解法二:状态压缩优化(最优解)

### 优化思路
解法一已经很优了,但代码可以更简洁。关键观察:**遇到负数时,max和min会互换**,所以可以先判断符号,提前交换。

> 💡 **关键想法**:当遇到负数时,交换max_prod和min_prod,后续逻辑统一处理

### Python代码

```python
def maxProduct_optimal(nums: List[int]) -> int:
    """
    🏆 解法二:状态压缩优化(最优解)
    思路:遇负数时交换最大最小值,简化更新逻辑
    """
    if not nums:
        return 0

    max_prod = min_prod = result = nums[0]

    for num in nums[1:]:
        # 如果当前数为负数,交换max和min(因为负负得正)
        if num < 0:
            max_prod, min_prod = min_prod, max_prod

        # 更新当前最大值和最小值
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)

        # 更新全局最大值
        result = max(result, max_prod)

    return result


# ✅ 测试
print(maxProduct_optimal([2,3,-2,4]))      # 期望输出:6
print(maxProduct_optimal([-2,3,-4]))       # 期望输出:24
print(maxProduct_optimal([0,2]))           # 期望输出:2
```

### 复杂度分析
- **时间复杂度**:O(n) — 与解法一相同
- **空间复杂度**:O(1) — 仍然只用3个变量

**为什么更优?**
- 代码更简洁(10行 vs 15行)
- 逻辑更清晰:提前处理负数情况
- 性能完全一致,只是写法优化

---

## 🐍 Pythonic 写法

利用 Python 的 reduce 和 lambda 实现函数式风格:

```python
from functools import reduce

def maxProduct_functional(nums: List[int]) -> int:
    """函数式编程风格:用reduce累积状态"""
    def update_state(state, num):
        max_p, min_p, res = state
        if num < 0:
            max_p, min_p = min_p, max_p
        max_p = max(num, max_p * num)
        min_p = min(num, min_p * num)
        return (max_p, min_p, max(res, max_p))

    # 初始状态:(max_prod, min_prod, result)
    final_state = reduce(update_state, nums[1:], (nums[0], nums[0], nums[0]))
    return final_state[2]
```

**解释**:
- `reduce(func, iterable, init)`:累积应用函数func到序列元素
- 状态三元组 (max_prod, min_prod, result) 在遍历中不断更新
- 最后返回result(索引2)

> ⚠️ **面试建议**:先写解法二的标准版本(清晰易懂),再提函数式写法展示Python功底。面试官更看重**思考过程**。

---

## 📊 解法对比

| 维度 | 解法一:标准DP | 🏆 解法二:优化版(最优) |
|------|-------------|---------------------|
| 时间复杂度 | **O(n)** | **O(n)** |
| 空间复杂度 | **O(1)** | **O(1)** |
| 代码难度 | 简单(显式备份) | **更简洁(提前交换)** |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 通用 | **首选,逻辑更清晰** |

**为什么是最优解**:
- 时间O(n)已是理论最优(至少要遍历一遍)
- 空间O(1)无额外开销
- 代码简洁,逻辑清晰

**面试建议**:
1. 先花30秒说明暴力法O(n²),表明你理解问题
2. 立即优化到🏆解法二,重点讲解"为什么要维护最大和最小值"
3. 强调关键点:**负数会让最大最小翻转,提前交换简化逻辑**
4. 手动测试含负数的边界用例,展示深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下"乘积最大子数组"问题。

**你**:(审题30秒)好的,这道题要求找出连续子数组的最大乘积。让我先想一下...

我的第一个想法是枚举所有起点和终点,计算每个子数组的乘积,时间复杂度是 O(n²)。

不过我们可以用动态规划优化到 O(n)。核心思路是维护两个变量:**当前最大乘积max_prod和当前最小乘积min_prod**。为什么要最小值?因为遇到负数时,之前的最小值(可能是负数)乘以当前负数,反而变成最大正数!

具体做法:遍历数组时,如果当前数为负数,先交换max_prod和min_prod,然后更新它们为"当前数单独"或"之前乘积×当前数"的较优值。

**面试官**:很好,请写一下代码。

**你**:(边写边说)我初始化max_prod和min_prod为第一个元素,result记录全局最大值。从第二个元素开始遍历,如果遇到负数,先交换max和min(因为负负得正)。然后更新max_prod为max(当前数,max_prod×当前数),min_prod类似。每次更新result为历史最大值。最后返回result。

**面试官**:测试一下?

**你**:用示例 [2,3,-2,4] 走一遍:
- 初始:max=2,min=2,result=2
- 遍历3:max=6,min=3,result=6
- 遍历-2(负数):交换后max=-2,min=-12,result保持6
- 遍历4:max=4,min=-48,result=6
最终输出6,对应子数组[2,3]。再测边界[-2,3,-4]:
- 初始:max=-2,min=-2,result=-2
- 遍历3:max=3,min=-6,result=3
- 遍历-4(负数):交换后max=-6×(-4)=24,result=24
输出24,对应完整数组。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间O(n)已是理论最优(必须遍历一遍),空间O(1)也无法再优化 |
| "为什么要同时维护最大最小?" | 因为负数乘法会翻转大小关系:最小负数×负数=最大正数。只维护最大值会漏掉这种情况 |
| "遇到0怎么办?" | 乘以0后,max_prod和min_prod都变为0,相当于从下一个位置重新开始计算,逻辑自动处理 |
| "能处理超大数吗?" | 题目保证答案在32位整数内。如果超范围,需要用Python的大整数或取模 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:多变量同时赋值交换 — 无需临时变量
a, b = b, a  # 交换a和b的值

# 技巧2:链式比较简化判断 — 更Pythonic
if -10 <= num <= 10:  # 等价于 num >= -10 and num <= 10
    pass

# 技巧3:max/min支持多参数 — 简化逻辑
max_val = max(a, b, c)  # 返回三者最大值
```

### 💡 底层原理(选读)

> **为什么负数让问题变复杂?**
>
> 1. **正数的单调性**:如果数组全是正数,乘积越多越大,直接全部相乘即可
>
> 2. **负数打破单调性**:
>    - 奇数个负数:乘积为负,不如跳过部分负数
>    - 偶数个负数:负负得正,乘积可能很大
>
> 3. **0的分割作用**:遇到0后乘积归零,相当于将数组分段
>
> 4. **DP的妙处**:同时跟踪最大最小值,巧妙利用负数翻转:
>    ```
>    max_new = max(当前数, 前max×当前, 前min×当前)
>                                      ↑
>                           当前为负时,前min×负=最大正数!
>    ```

### 算法模式卡片 📐
- **模式名称**:动态规划+最大最小双向维护
- **适用条件**:
  - 求连续子数组的"最大/最小"值
  - 涉及乘法运算(符号会变化)
  - 需要考虑负数翻转大小关系
- **识别关键词**:"连续子数组"、"最大乘积"、"含负数"
- **模板代码**:
```python
def maxProduct(nums):
    max_prod = min_prod = result = nums[0]
    for num in nums[1:]:
        if num < 0:
            max_prod, min_prod = min_prod, max_prod
        max_prod = max(num, max_prod * num)
        min_prod = min(num, min_prod * num)
        result = max(result, max_prod)
    return result
```

### 易错点 ⚠️
1. **错误:只维护最大值,忽略最小值**
   - 原因:负数×负数=正数,之前的最小负值可能翻盘
   - 正确做法:同时维护max_prod和min_prod
   - 示例:[-2, 3, -4],如果只维护最大值,会错过 -2×-4=8 的情况

2. **错误:忘记交换max和min**
   - 原因:遇到负数时,最大最小会互换
   - 正确做法:在负数时先 `max_prod, min_prod = min_prod, max_prod`

3. **错误:初始化max_prod=0或1**
   - 原因:第一个元素可能就是答案(如单元素数组)
   - 正确做法:初始化为 `nums[0]`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:金融分析中的"最大收益率计算"
  - 股票日收益率可能为负,求连续交易日的最大收益率乘积

- **场景2**:推荐系统中的"连续行为价值评分"
  - 用户连续操作的价值可正可负,求最大价值片段

- **场景3**:信号处理中的"连续脉冲峰值检测"
  - 信号强度乘积,负信号翻转相位

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 53. 最大子数组和 | Medium | DP(只需维护最大) | 加法无符号翻转,简化版 |
| LeetCode 628. 三个数的最大乘积 | Easy | 贪心+排序 | 考虑最大×次大 vs 最小×次小 |
| LeetCode 238. 除自身以外数组的乘积 | Medium | 前缀积+后缀积 | 类似思想,分段处理 |
| LeetCode 1567. 乘积为正数的最长子数组长度 | Medium | DP变体 | 维护正负乘积的最长长度 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定数组nums,找到乘积为正数的最长连续子数组的长度。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

类似本题,维护"当前最长正乘积长度"和"当前最长负乘积长度",遇负数时交换。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def getMaxLen(nums):
    """
    乘积为正数的最长子数组长度
    思路:维护正乘积和负乘积的最长长度
    """
    pos_len = neg_len = 0  # 当前正/负乘积的长度
    max_len = 0

    for num in nums:
        if num == 0:
            # 遇到0,重置
            pos_len = neg_len = 0
        elif num > 0:
            # 正数:正长度+1,负长度继续(如果存在)
            pos_len += 1
            neg_len = neg_len + 1 if neg_len > 0 else 0
        else:  # num < 0
            # 负数:正负交换
            new_pos = neg_len + 1 if neg_len > 0 else 0
            new_neg = pos_len + 1
            pos_len, neg_len = new_pos, new_neg

        max_len = max(max_len, pos_len)

    return max_len

# 测试
print(getMaxLen([1,-2,-3,4]))   # 输出:4 (整个数组)
print(getMaxLen([0,1,-2,-3,-4])) # 输出:3 ([-2,-3,-4])
```

**核心思路**:不维护乘积值本身,只维护长度。负数时交换正负长度,零时重置。

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
