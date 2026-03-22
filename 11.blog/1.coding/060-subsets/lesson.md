# 📖 第60课:子集

> **模块**:回溯算法 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/subsets/
> **前置知识**:第59课 全排列(回溯基础)
> **预计学习时间**:20分钟

---

## 🎯 题目描述

给定一个整数数组 nums,数组中的元素互不相同。返回该数组所有可能的子集(幂集)。解集不能包含重复的子集,可以按任意顺序返回。

**示例:**
```
输入:nums = [1,2,3]
输出:[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
解释:包含空集在内,一共8个子集
```

**约束条件:**
- 1 <= nums.length <= 10
- -10 <= nums[i] <= 10
- nums 中的所有元素互不相同

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | nums=[1] | [[],[1]] | 基本功能 |
| 两个元素 | nums=[1,2] | [[],[1],[2],[1,2]] | 递归正确性 |
| 负数 | nums=[-1,0] | [[],[-1],[0],[-1,0]] | 负数处理 |
| 最大规模 | n=10 | 2^10=1024个子集 | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你要去旅行,行李箱里可以放3件物品(A、B、C),但空间有限,你可以选择带或不带每件物品。
>
> 🐌 **笨办法**:你先试"什么都不带",然后"只带A",再"只带B"..."只带C",再"带A和B"..."一个个枚举,很容易漏掉某些组合。
>
> 🚀 **聪明办法**:你站在每件物品前做选择:"带还是不带?"对A做选择后,递归处理B;对B做选择后,递归处理C。这样每条路径对应一个子集,自动覆盖所有 2^3=8 种可能!

### 关键洞察
**子集问题 = 每个元素都有"选"或"不选"两种决策,构成二叉决策树,所有路径就是所有子集。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:[1,2,3] 三个不同的数字
- **输出**:所有子集(包括空集) [[],[1],[2],[1,2],...]
- **限制**:元素不重复,子集不能重复,顺序无关([1,2] 和 [2,1] 算同一个)

### Step 2:先想笨办法(暴力法)
用位运算枚举:n个元素有 2^n 个子集,用 0 到 2^n-1 的二进制表示每个子集
- 时间复杂度:O(n × 2^n) 需要遍历所有二进制数
- 瓶颈在哪:不够直观,代码不易扩展到有重复元素的情况

### Step 3:瓶颈分析 → 优化方向
暴力法虽然可行,但回溯法更通用、更易理解
- 核心问题:如何系统地生成所有子集而不遗漏?
- 优化思路:用回溯算法,每个元素"选/不选"构成决策树

### Step 4:选择武器
- 选用:**回溯算法(选/不选决策树)**
- 理由:与全排列不同,子集问题每个元素只需决策"选或不选",不需要 used 数组,代码更简洁

> 🔑 **模式识别提示**:当题目出现"所有子集",优先考虑"回溯算法 + 选/不选决策"

---

## 🔑 解法一:回溯(选/不选决策树)

### 思路
从第一个元素开始,每次有两种选择:选它(加入当前子集)或不选它(跳过)。递归处理后续元素,每个节点都对应一个有效子集,收集所有节点的结果。

### 图解过程

```
示例:nums = [1, 2, 3]

决策树(每层决策一个元素是否加入子集):

                        []
                /                \
          选1 [1]                  不选1 []
          /      \                /        \
    选2[1,2]   不选2[1]      选2[2]      不选2[]
     /   \       /   \        /   \        /   \
选3 不选3 选3 不选3  选3 不选3  选3 不选3
[1,2,3][1,2][1,3][1] [2,3][2] [3] []

↑ 所有节点(8个)都是有效子集,不只是叶子节点!

关键区别 vs 全排列:
- 全排列:只有叶子节点是结果(路径长度必须=n)
- 子集:所有节点都是结果(路径长度可以是0~n)

执行过程:
Step 1: path=[], 收集 [] → 选1
Step 2: path=[1], 收集 [1] → 选2
Step 3: path=[1,2], 收集 [1,2] → 选3
Step 4: path=[1,2,3], 收集 [1,2,3] → 回溯,不选3
Step 5: path=[1,2], 已收集 → 回溯,不选2
Step 6: path=[1], 已收集 → 选3
Step 7: path=[1,3], 收集 [1,3] → 回溯,不选3
Step 8: path=[1], 已收集 → 回溯,不选1
Step 9: path=[], 已收集 → 选2
...依次遍历右子树
```

**边界情况演示:nums = [1]**
```
决策树:
       []
      /  \
    [1]  []

收集4次:[], [1], [1](回溯后), []  → 去重后2个子集:[], [1]
实际代码中每个节点只收集一次,共2个子集
```

### Python代码

```python
from typing import List


def subsets(nums: List[int]) -> List[List[int]]:
    """
    解法一:回溯(选/不选决策树)
    思路:每个元素两种选择,所有节点都是有效子集
    """
    result = []

    def backtrack(start: int, path: List[int]):
        # 关键:每个节点都是一个有效子集,立即收集
        result.append(path[:])  # 拷贝当前子集

        # 从 start 开始遍历,保证子集不重复(避免[1,2]和[2,1])
        for i in range(start, len(nums)):
            # 选择:将 nums[i] 加入子集
            path.append(nums[i])

            # 递归:处理后续元素(i+1保证不重复选)
            backtrack(i + 1, path)

            # 撤销选择:回溯,尝试"不选 nums[i]"的分支
            path.pop()

    backtrack(0, [])
    return result


# ✅ 测试
print(subsets([1, 2, 3]))  # 期望输出:[[],[1],[1,2],[1,2,3],[1,3],[2],[2,3],[3]]
print(subsets([1]))        # 期望输出:[[],[1]]
print(subsets([0]))        # 期望输出:[[],[0]]
```

### 复杂度分析
- **时间复杂度**:O(n × 2^n) — 生成 2^n 个子集,每个子集需要 O(n) 时间复制
  - 具体地说:如果 n=3,有 2^3=8 个子集,每个平均长度 1.5,总操作约 12 次
  - 如果 n=10,有 2^10=1024 个子集,总操作数约 5120 次
- **空间复杂度**:O(n) — 递归栈深度最多 n 层

### 优缺点
- ✅ 代码简洁,逻辑清晰,易于理解
- ✅ 通用性强,容易扩展到"含重复元素的子集"
- ✅ 时间空间已达最优(必须生成所有 2^n 个子集)

---

## 🏆 解法二:迭代法(最优解,更直观)

### 优化思路
回溯法虽然优雅,但还有更直观的迭代思路:从空集开始,每次加入一个新元素,将现有所有子集都"复制一份并添加新元素"。

> 💡 **关键想法**:子集的生成过程是增量式的:已有子集 + 新元素 = 新子集

### 图解过程

```
示例:nums = [1, 2, 3]

迭代生成过程:
初始:result = [[]]  (只有空集)

加入元素1:
  现有子集:[]
  复制并添加1:[] + [1] → result = [[], [1]]

加入元素2:
  现有子集:[], [1]
  复制并添加2:[] + [2], [1] + [2] → result = [[], [1], [2], [1,2]]

加入元素3:
  现有子集:[], [1], [2], [1,2]
  复制并添加3:[] + [3], [1] + [3], [2] + [3], [1,2] + [3]
  → result = [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]

每次加入新元素,子集数量翻倍:1 → 2 → 4 → 8
```

### Python代码

```python
def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """
    解法二:迭代法(最优解)
    思路:从空集开始,每次加入新元素,复制现有子集并添加新元素
    """
    result = [[]]  # 初始只有空集

    for num in nums:
        # 遍历当前所有子集,复制并添加新元素
        new_subsets = [subset + [num] for subset in result]
        result.extend(new_subsets)  # 将新子集加入结果

    return result


# ✅ 测试
print(subsets_iterative([1, 2, 3]))  # 期望输出:[[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
print(subsets_iterative([1]))        # 期望输出:[[],[1]]
```

### 复杂度分析
- **时间复杂度**:O(n × 2^n) — 同解法一,必须生成所有子集
- **空间复杂度**:O(1) — 不计结果数组,无递归栈 ← **空间更优**

---

## ⚡ 解法三:位运算枚举(巧妙但不推荐)

### 思路
n个元素有 2^n 个子集,可以用 0 到 2^n-1 的二进制数表示每个子集:二进制的第 i 位为 1 表示选 nums[i]。

### Python代码

```python
def subsets_bit(nums: List[int]) -> List[List[int]]:
    """
    解法三:位运算枚举
    思路:用二进制数表示每个子集
    """
    n = len(nums)
    result = []

    # 遍历 0 到 2^n - 1 的所有二进制数
    for mask in range(1 << n):  # 1 << n 就是 2^n
        subset = []
        for i in range(n):
            # 检查第 i 位是否为 1
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)

    return result


# ✅ 测试
print(subsets_bit([1, 2, 3]))  # 期望输出:8个子集
```

### 复杂度分析
- **时间复杂度**:O(n × 2^n)
- **空间复杂度**:O(1)

---

## 🐍 Pythonic 写法

利用 Python 的 itertools.combinations:

```python
from itertools import combinations

def subsets_pythonic(nums: List[int]) -> List[List[int]]:
    """Pythonic写法:使用标准库"""
    result = []
    for i in range(len(nums) + 1):  # 子集长度从0到n
        result.extend([list(c) for c in combinations(nums, i)])
    return result
```

更简洁的链式写法:
```python
from itertools import chain, combinations

def subsets_pythonic_v2(nums: List[int]) -> List[List[int]]:
    return [list(subset) for subset in chain.from_iterable(
        combinations(nums, r) for r in range(len(nums) + 1)
    )]
```

> ⚠️ **面试建议**:先写回溯或迭代法展示算法思维,再提 Pythonic 写法展示语言功底。

---

## 📊 解法对比

| 维度 | 解法一:回溯 | 🏆 解法二:迭代(最优) | 解法三:位运算 |
|------|-----------|------------------|------------|
| 时间复杂度 | O(n × 2^n) | **O(n × 2^n)** | O(n × 2^n) |
| 空间复杂度 | O(n) 递归栈 | **O(1)** ← 无递归 | O(1) |
| 代码难度 | 中等 | **简单** | 较难 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** | ⭐ |
| 适用场景 | 通用,易扩展 | **面试首选,最直观** | 不易理解 |

**为什么解法二是最优解**:
- 时间复杂度已达理论下限(必须生成所有 2^n 个子集)
- 空间优化到极致(无递归栈,仅 O(1) 辅助空间)
- 代码最简洁直观,面试中最容易写对

**面试建议**:
1. 先用1分钟口述回溯思路(选/不选决策树),展示对回溯的理解
2. 立即优化到🏆解法二(迭代法),强调"从空集开始,逐步添加元素"的直观思路
3. **重点对比与全排列的区别**:"全排列只收集叶子节点,子集收集所有节点"
4. 手动在 [1,2] 上演示迭代过程:[] → [[], [1]] → [[], [1], [2], [1,2]]

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下子集问题。

**你**:(审题30秒)好的,这道题要求返回数组的所有子集,包括空集。我的第一个想法是回溯,因为每个元素都有"选"或"不选"两种决策,可以构成决策树。不过有个更直观的迭代方法:从空集开始,每次加入一个新元素,把现有所有子集复制一份并添加新元素。这样时间是 O(n × 2^n),空间只需 O(1)。

**面试官**:很好,请写一下迭代的代码。

**你**:(边写边说)首先初始化结果为 [[]] 只包含空集。然后遍历数组每个元素,对于每个元素,我用列表推导式复制现有所有子集并添加当前元素,再扩展到结果中。

**面试官**:测试一下?

**你**:用示例 [1,2] 走一遍...初始 [[]],加入1得到 [[], [1]],加入2复制得到 [[2], [1,2]],合并后是 [[], [1], [2], [1,2]],正确。再测边界 [1],结果是 [[], [1]],也正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "还有更优解吗?" | 时间 O(n × 2^n) 已经是最优(必须生成所有子集),空间 O(1) 也已最优 |
| "如果数组包含重复元素呢?" | 需要先排序,然后在回溯时剪枝:跳过重复元素或控制重复元素的选择顺序 |
| "能否只生成特定长度的子集?" | 可以,在回溯或迭代时添加长度限制:if len(path) == k: collect |
| "这道题和全排列有什么区别?" | 全排列关心顺序(选谁),子集不关心顺序(选不选);全排列只收集叶子节点,子集收集所有节点 |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:列表推导式复制并添加元素
new_subsets = [subset + [num] for subset in result]

# 技巧2:extend vs append
result.extend(new_subsets)  # ✅ 扩展列表,添加多个元素
result.append(new_subsets)  # ❌ 添加整个列表作为单个元素

# 技巧3:位运算生成2的幂
1 << n  # 等价于 2^n
mask & (1 << i)  # 检查mask的第i位是否为1
```

### 💡 底层原理(选读)

> **为什么子集数量是 2^n?**
>
> 每个元素都有"选"或"不选"两种状态,n个元素独立决策,总共 2 × 2 × ... × 2 (n个2相乘) = 2^n 种组合。
>
> **回溯 vs 迭代的本质区别?**
> - 回溯:深度优先搜索决策树,递归实现,空间需要递归栈
> - 迭代:广度优先生成子集,增量式构建,空间仅需结果数组
>
> 两者时间复杂度相同,但迭代法更直观,空间更优。

### 算法模式卡片 📐
- **模式名称**:子集/组合回溯
- **适用条件**:需要枚举所有子集、组合,或在"选/不选"约束下搜索
- **识别关键词**:"所有子集"、"所有组合"、"k个元素的组合"、"选或不选"
- **模板代码**:
```python
def subsets_template(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])  # 收集当前子集
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)  # i+1避免重复
            path.pop()

    backtrack(0, [])
    return result
```

### 易错点 ⚠️
1. **只收集叶子节点** — 子集问题每个节点都是结果,必须在递归开始就 `result.append(path[:])`,而不是等到 `len(path) == n`
2. **迭代时直接修改 result** — 必须先生成 `new_subsets`,再 `extend`,否则会在遍历中修改列表导致无限循环
3. **忘记拷贝 path** — `result.append(path)` 只保存引用,后续修改会影响结果,正确做法:`result.append(path[:])`

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:推荐系统中,生成用户可能感兴趣的商品组合(购物车推荐)
- **场景2**:特征工程中,枚举特征的所有组合,找最优特征子集
- **场景3**:测试用例生成,枚举配置参数的所有组合,实现全覆盖测试

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 90. 子集 II | Medium | 回溯+去重 | 先排序,剪枝跳过重复元素 |
| LeetCode 77. 组合 | Medium | 回溯+剪枝 | 限制子集长度为 k,剪枝优化 |
| LeetCode 39. 组合总和 | Medium | 回溯+剪枝 | 元素可重复选,sum达标时收集 |
| LeetCode 216. 组合总和 III | Medium | 回溯+约束 | 限制子集长度和元素和 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一个可能包含重复数字的整数数组 nums,返回所有不重复的子集。例如输入 [1,2,2],输出 [[],[1],[1,2],[1,2,2],[2],[2,2]]。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

先排序使重复元素相邻,然后在回溯时添加剪枝:if i > start and nums[i] == nums[i-1]: continue

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def subsets_with_dup(nums: List[int]) -> List[List[int]]:
    """含重复元素的子集"""
    nums.sort()  # 排序使重复元素相邻
    result = []

    def backtrack(start: int, path: List[int]):
        result.append(path[:])

        for i in range(start, len(nums)):
            # 剪枝:跳过重复元素(同一层递归中,相同元素只选第一个)
            if i > start and nums[i] == nums[i - 1]:
                continue

            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

核心思路:排序后,在同一层递归中跳过重复元素(`i > start` 保证是同层),避免生成重复子集。例如 [1,2,2],第一个2可以选,第二个2在同层会被跳过。

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
