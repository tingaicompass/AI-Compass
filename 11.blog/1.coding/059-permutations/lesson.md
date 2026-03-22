# 📖 第59课:全排列

> **模块**:回溯算法 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/permutations/
> **前置知识**:无(回溯算法模块入门题)
> **预计学习时间**:25分钟

---

## 🎯 题目描述

给定一个不含重复数字的整数数组 nums,返回其所有可能的全排列。你可以按任意顺序返回答案。

**示例:**
```
输入:nums = [1,2,3]
输出:[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
解释:一共有6种不同的排列方式
```

**约束条件:**
- 1 <= nums.length <= 6
- -10 <= nums[i] <= 10
- nums 中的所有整数互不相同

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1] | [[1]] | 基本功能 |
| 两个元素 | nums=[1,2] | [[1,2],[2,1]] | 递归终止 |
| 负数 | nums=[-1,0,1] | 6种排列 | 负数处理 |
| 最大规模 | n=6 | 720种排列 | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你是一个摄影师,要给3个明星(A、B、C)拍合照,但每个明星都想站不同的位置。
>
> 🐌 **笨办法**:你先让A站第一位,然后B、C轮流站剩下的位置拍照;再让B站第一位,A、C轮流站剩下的位置...这样一个个试,但你总是忘记哪些组合拍过了,可能重复拍同一张照片。
>
> 🚀 **聪明办法**:你拿一个签到表(used数组),每次让一个明星站定后,就在表上打勾"已占用",后面的明星只能选没打勾的位置;拍完这张照后,把这个明星的勾去掉,让下一个明星站这个位置,这样保证不会重复,也不会遗漏!

### 关键洞察
**这是一个"填空问题":有n个位置,每个位置从剩余数字中选一个,用used数组标记"谁被用过"。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:[1,2,3] 三个不同的数字
- **输出**:所有排列 [[1,2,3], [1,3,2], ...]
- **限制**:数字不重复,需要返回所有可能的排列(顺序不同算不同排列)

### Step 2:先想笨办法(暴力法)
用三层循环枚举所有组合,第一层选第一个数字,第二层选第二个数字...
- 时间复杂度:O(n^n) 大量重复枚举
- 瓶颈在哪:无法灵活处理"已选择"和"未选择"的状态,代码难以扩展到不同长度的数组

### Step 3:瓶颈分析 → 优化方向
暴力法的核心问题:无法系统地管理"选择状态"
- 核心问题:"每一步选了谁,后面不能再选同一个数字"
- 优化思路:用回溯算法+used数组跟踪状态

### Step 4:选择武器
- 选用:**回溯算法 + used数组**
- 理由:回溯天然支持"选择 → 递归 → 撤销"的试错过程,used数组标记哪些数字已使用

> 🔑 **模式识别提示**:当题目出现"所有排列/组合/子集",优先考虑"回溯算法"

---

## 🔑 解法一:回溯 + used数组标记(标准解法)

### 思路
用一个 used 布尔数组标记每个数字是否已被选入当前排列。每次递归从头扫描所有数字,跳过已使用的,选择一个未使用的数字加入路径,递归处理剩余位置,然后撤销选择(回溯)。

### 图解过程

```
示例:nums = [1, 2, 3]

决策树(每层选择一个数字加入排列):

                       []
           /           |            \
         [1]          [2]           [3]
        /   \        /   \         /   \
    [1,2] [1,3]  [2,1] [2,3]   [3,1] [3,2]
      |     |      |     |       |     |
  [1,2,3][1,3,2][2,1,3][2,3,1][3,1,2][3,2,1]  ← 叶子节点(6个排列)

每个分支:
- 选择:从 nums 中选一个未用过的数字 nums[i]
- 约束:used[i] == False
- 递归:path.append(nums[i]), used[i] = True, 继续填下一个位置
- 撤销:path.pop(), used[i] = False, 回到上一层尝试其他选择

Step 1: path=[], used=[F,F,F], 选1 → path=[1], used=[T,F,F]
Step 2: path=[1], 选2 → path=[1,2], used=[T,T,F]
Step 3: path=[1,2], 选3 → path=[1,2,3], used=[T,T,T] → 收集结果
Step 4: 回溯:path=[1,2], used=[T,T,F], 无其他选择 → 继续回溯
Step 5: 回溯:path=[1], used=[T,F,F], 选3 → path=[1,3], used=[T,F,T]
...依次遍历整棵树
```

**边界情况演示:nums = [1]**
```
决策树:
    []
    |
   [1]  ← 立即收集结果

结果:[[1]]
```

### Python代码

```python
from typing import List


def permute(nums: List[int]) -> List[List[int]]:
    """
    解法一:回溯 + used数组标记
    思路:用布尔数组标记哪些数字已被使用,递归构建所有排列
    """
    result = []
    n = len(nums)

    def backtrack(path: List[int], used: List[bool]):
        # 递归终止条件:排列已包含所有数字
        if len(path) == n:
            result.append(path[:])  # 必须拷贝,否则后续修改会影响已保存结果
            return

        # 遍历所有数字
        for i in range(n):
            # 剪枝:跳过已使用的数字
            if used[i]:
                continue

            # 选择:将 nums[i] 加入当前排列
            path.append(nums[i])
            used[i] = True

            # 递归:继续填下一个位置
            backtrack(path, used)

            # 撤销选择(回溯):恢复状态,尝试其他选择
            path.pop()
            used[i] = False

    backtrack([], [False] * n)
    return result


# ✅ 测试
print(permute([1, 2, 3]))  # 期望输出:[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
print(permute([1]))        # 期望输出:[[1]]
print(permute([0, 1]))     # 期望输出:[[0,1],[1,0]]
```

### 复杂度分析
- **时间复杂度**:O(n! × n) — 生成 n! 个排列,每个排列需要 O(n) 时间复制到结果
  - 具体地说:如果 n=3,需要生成 3!=6 个排列,每个排列复制需要 3 次操作,总共约 18 次操作
  - 如果 n=6,需要生成 6!=720 个排列,总操作数约 720×6=4320 次
- **空间复杂度**:O(n) — 递归栈深度 n + used数组 n + path数组 n

### 优缺点
- ✅ 逻辑清晰,used数组直观标记状态
- ✅ 易于理解和调试,面试推荐
- ❌ 需要额外 O(n) 空间维护 used 数组(可优化)

---

## 🏆 解法二:回溯 + 交换元素(最优解,空间O(1))

### 优化思路
解法一需要 used 数组标记,能否省掉这个数组?关键洞察:可以通过**交换元素位置**来实现"选择"和"撤销"。

> 💡 **关键想法**:把数组分为"已选择"和"未选择"两部分,用一个指针 start 分隔。每次将未选择部分的某个元素交换到 start 位置(表示选择它),递归处理后再交换回来(撤销)。

### 图解过程

```
示例:nums = [1, 2, 3]

交换策略:
start=0: 依次将 nums[0], nums[1], nums[2] 与自己或后面的元素交换
start=1: 依次将 nums[1], nums[2] 与自己或后面的元素交换
start=2: 只剩 nums[2],收集结果

执行过程:
初始:[1, 2, 3], start=0
  交换 nums[0]↔nums[0]: [1, 2, 3], 选1, 递归 start=1
    交换 nums[1]↔nums[1]: [1, 2, 3], 选2, 递归 start=2
      交换 nums[2]↔nums[2]: [1, 2, 3], 选3, 收集 [1,2,3]
    交换 nums[1]↔nums[2]: [1, 3, 2], 选3, 递归 start=2
      收集 [1,3,2]
    恢复交换: [1, 2, 3]
  交换 nums[0]↔nums[1]: [2, 1, 3], 选2, 递归 start=1
    交换 nums[1]↔nums[1]: [2, 1, 3], 选1, 递归 start=2
      收集 [2,1,3]
    交换 nums[1]↔nums[2]: [2, 3, 1], 选3, 递归 start=2
      收集 [2,3,1]
    恢复交换: [2, 1, 3]
  恢复交换: [1, 2, 3]
  交换 nums[0]↔nums[2]: [3, 2, 1], 选3, 递归 start=1
    ...
    收集 [3,1,2], [3,2,1]
```

### Python代码

```python
def permute_swap(nums: List[int]) -> List[List[int]]:
    """
    解法二:回溯 + 原地交换(最优解)
    思路:通过交换元素避免使用 used 数组,空间更优
    """
    result = []

    def backtrack(start: int):
        # 递归终止:所有位置都已确定
        if start == len(nums):
            result.append(nums[:])  # 拷贝当前排列
            return

        # 从 start 位置开始,依次尝试将每个元素放到 start 位置
        for i in range(start, len(nums)):
            # 选择:将 nums[i] 交换到 start 位置
            nums[start], nums[i] = nums[i], nums[start]

            # 递归:处理下一个位置
            backtrack(start + 1)

            # 撤销选择:恢复原数组
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result


# ✅ 测试
print(permute_swap([1, 2, 3]))  # 期望输出:[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
print(permute_swap([1]))        # 期望输出:[[1]]
```

### 复杂度分析
- **时间复杂度**:O(n! × n) — 同解法一,生成 n! 个排列
- **空间复杂度**:O(n) — 仅递归栈,不需要 used 数组 ← **空间更优**

---

## 🐍 Pythonic 写法

利用 Python 的 itertools.permutations 库函数:

```python
from itertools import permutations

def permute_pythonic(nums: List[int]) -> List[List[int]]:
    """Pythonic写法:使用标准库"""
    return [list(p) for p in permutations(nums)]
```

这个写法底层也是回溯实现,但代码极简。

> ⚠️ **面试建议**:先手写回溯展示算法能力,最后可以提一句"工程中可以用 itertools.permutations",展示对标准库的了解。面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:used数组 | 🏆 解法二:交换元素(最优) | Pythonic写法 |
|------|--------------|----------------------|------------|
| 时间复杂度 | O(n! × n) | **O(n! × n)** | O(n! × n) |
| 空间复杂度 | O(n) used数组 + O(n) 栈 | **O(n)** ← 仅递归栈 | O(n! × n) 结果存储 |
| 代码难度 | 简单 | 中等 | 极简 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** | ⭐ |
| 适用场景 | 初学者,易于理解 | **面试首选,空间最优** | 工程快速实现 |

**为什么解法二是最优解**:
- 时间复杂度已达理论下限(必须生成所有 n! 个排列)
- 空间优化到极致(避免 used 数组,仅用递归栈)
- 面试中展示对回溯本质的深刻理解

**面试建议**:
1. 先用2分钟口述解法一的思路(used数组标记),表明你理解回溯基本框架
2. 立即优化到🏆解法二(交换元素),展示空间优化能力
3. **重点讲解回溯三要素**:"选择(交换)、递归、撤销(交换回来)"
4. 手动在示例 [1,2] 上走一遍递归树,展示对算法的深入理解

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下全排列问题。

**你**:(审题30秒)好的,这道题要求返回数组的所有排列。让我先想一下...我的第一个想法是用回溯算法,因为需要枚举所有可能的组合。可以用一个 used 数组标记哪些数字已经被选入当前排列,时间复杂度是 O(n! × n)。不过我们可以优化空间,通过交换元素来避免 used 数组,核心思路是每次将未选择的元素交换到当前位置。

**面试官**:很好,请写一下优化后的代码。

**你**:(边写边说)我们定义一个 backtrack 函数,参数是 start 位置。递归终止条件是 start 到达数组末尾,此时收集当前排列。然后从 start 到末尾遍历,每次将 nums[i] 交换到 start 位置表示选择它,递归处理 start+1,回溯时再交换回来恢复状态。

**面试官**:测试一下?

**你**:用示例 [1,2] 走一遍...start=0时,先选1留在位置0,递归start=1选2,得到[1,2];回溯后交换1和2,得到[2,1]。再测一个边界情况 [1],直接返回 [[1]]。结果正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间已经是 O(n!) 最优(必须生成所有排列),空间已优化到 O(n) 仅递归栈,无法进一步优化 |
| "如果数组包含重复元素呢?" | 需要排序+剪枝,在for循环中跳过重复元素:if i>start and nums[i]==nums[i-1]: continue |
| "能不用递归吗?" | 可以用迭代+栈模拟递归,但代码更复杂,实际面试中递归更清晰 |
| "实际工程中怎么用?" | Python 可以直接用 itertools.permutations,C++ 用 std::next_permutation |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:列表的浅拷贝 vs 深拷贝
result.append(path[:])   # ✅ 浅拷贝,创建新列表
result.append(path)      # ❌ 错误!只是引用,后续修改会影响结果

# 技巧2:列表原地交换
nums[i], nums[j] = nums[j], nums[i]  # Python 的优雅交换

# 技巧3:itertools.permutations 生成器
from itertools import permutations
list(permutations([1,2,3]))  # 返回元组列表
```

### 💡 底层原理(选读)

> **为什么回溯能遍历所有排列?**
>
> 回溯本质是深度优先搜索(DFS)遍历决策树。全排列的决策树有 n 层,每层从剩余元素中选一个,叶子节点就是一个完整排列。通过"选择 → 递归 → 撤销"的模式,可以系统地遍历整棵树的所有路径。
>
> **时间复杂度为什么是 O(n! × n)?**
> - 共有 n! 个排列(叶子节点数量)
> - 每个排列需要 O(n) 时间复制到结果数组
> - 总时间 = 排列数 × 每个排列的处理时间 = n! × n

### 算法模式卡片 📐
- **模式名称**:回溯算法(Backtracking)
- **适用条件**:需要枚举所有可能的排列/组合/子集,或在约束条件下搜索解
- **识别关键词**:"所有排列"、"所有组合"、"所有子集"、"路径搜索"、"N皇后"
- **模板代码**:
```python
def backtrack(路径, 选择列表):
    if 满足终止条件:
        收集结果
        return

    for 选择 in 选择列表:
        if 不满足约束:
            continue  # 剪枝
        做选择  # 修改状态
        backtrack(路径, 新的选择列表)  # 递归
        撤销选择  # 恢复状态
```

### 易错点 ⚠️
1. **忘记拷贝path** — `result.append(path)` 只保存引用,后续修改会影响结果。正确做法:`result.append(path[:])`
2. **忘记撤销选择** — 回溯的核心是"撤销",必须在递归后恢复状态:`path.pop()`, `used[i] = False`
3. **交换后忘记恢复** — 解法二中,`backtrack()` 后必须再次交换回来:`nums[start], nums[i] = nums[i], nums[start]`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:任务调度系统中,生成所有可能的任务执行顺序,找最优方案
- **场景2**:旅行路线规划,枚举所有城市访问顺序,结合TSP算法找最短路径
- **场景3**:密码破解,枚举所有字符排列(结合剪枝提高效率)

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 47. 全排列 II | Medium | 回溯+去重 | 先排序,剪枝时跳过重复元素 |
| LeetCode 77. 组合 | Medium | 回溯+剪枝 | 用 start 参数避免重复组合 |
| LeetCode 78. 子集 | Medium | 回溯 | 每个节点都收集结果,不只是叶子节点 |
| LeetCode 22. 括号生成 | Medium | 回溯+约束 | 左括号数 >= 右括号数作为剪枝条件 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个字符串,返回其所有不重复的全排列。例如输入 "aab",输出 ["aab", "aba", "baa"]。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

先排序字符串,然后在回溯时添加剪枝条件:if i > 0 and s[i] == s[i-1] and not used[i-1]: continue

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def permute_unique(s: str) -> List[str]:
    """含重复字符的全排列"""
    s = sorted(s)  # 排序使重复字符相邻
    result = []
    n = len(s)

    def backtrack(path: List[str], used: List[bool]):
        if len(path) == n:
            result.append(''.join(path))
            return

        for i in range(n):
            if used[i]:
                continue
            # 剪枝:跳过重复字符(关键:前一个相同字符未使用时才跳过)
            if i > 0 and s[i] == s[i-1] and not used[i-1]:
                continue

            path.append(s[i])
            used[i] = True
            backtrack(path, used)
            path.pop()
            used[i] = False

    backtrack([], [False] * n)
    return result
```

核心思路:排序后,重复字符相邻。剪枝条件 `s[i] == s[i-1] and not used[i-1]` 保证同一组重复字符按顺序使用,避免重复排列。

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
