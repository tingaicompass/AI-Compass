# 📖 第62课:组合总和

> **模块**:回溯算法 | **难度**:Medium ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/combination-sum/
> **前置知识**:第59课(全排列)、第60课(子集)、第61课(电话号码字母组合)
> **预计学习时间**:30分钟

---

## 🎯 题目描述

给定一个无重复元素的正整数数组 candidates 和一个目标整数 target,找出 candidates 中所有可以使数字和为 target 的组合。candidates 中的同一个数字可以无限制重复被选取。

**示例:**
```
输入:candidates = [2,3,6,7], target = 7
输出:[[2,2,3],[7]]
解释:
2 和 3 可以形成一组候选,2 + 2 + 3 = 7(注意2可以使用两次)
7 也是一个候选,7 = 7
```

**示例2:**
```
输入:candidates = [2,3,5], target = 8
输出:[[2,2,2,2],[2,3,3],[3,5]]
解释:
可以用4个2,或2+3+3,或3+5,共3种组合
```

**示例3:**
```
输入:candidates = [2], target = 1
输出:[]
解释:没有组合可以凑出1(2太大了)
```

**约束条件:**
- 1 <= candidates.length <= 30 — 候选数最多30个
- 2 <= candidates[i] <= 40 — 候选数都是正整数
- candidates 中元素互不相同 — 无重复
- 1 <= target <= 40 — 目标和范围

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 无法凑出 | candidates=[2], target=1 | [] | 剪枝终止 |
| 精确匹配 | candidates=[7], target=7 | [[7]] | 单个元素 |
| 需要重复 | candidates=[2,3], target=8 | [[2,2,2,2],[2,3,3],[3,3,2]] | 元素可重复 |
| 大数组 | candidates=[2,...,40](30个), target=40 | 多种组合 | 性能测试 |
| 最小target | candidates=[2,3,5], target=1 | [] | 边界处理 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在超市购物,手里只有7元,货架上有2元、3元、6元、7元的商品,每种商品数量无限。你想知道有哪些购物方案能正好花完这7元。
>
> 🐌 **笨办法**:盲目尝试所有可能的组合——先拿2元的,再拿2元的,再拿2元的...哎呀超了,退一个2元的,改拿3元的...这样没有章法,会尝试很多重复的无效组合。
>
> 🚀 **聪明办法**:
> 1. **先排序**:把商品按价格从小到大排列 [2,3,6,7]
> 2. **从前往后尝试**:先尝试便宜的,比如拿2元的,剩余5元继续递归
> 3. **剪枝优化**:如果当前商品价格已经大于剩余金额,后面更贵的商品就不用看了(因为已排序)
> 4. **避免重复**:规定每次只能选当前位置或之后的商品,这样 [2,3,3] 和 [3,2,3] 只会生成一个

### 关键洞察
**这是一个带剪枝优化的回溯问题。关键点有三:1) 元素可重复选取,所以递归时index不+1; 2) 排序后可以剪枝,遇到过大元素直接break; 3) 用start参数避免生成重复组合。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:正整数数组 candidates(无重复),目标整数 target
- **输出**:所有和为 target 的组合(结果集合)
- **限制**:同一个数字可以被无限次选取,但组合不能重复(如[2,3,3]和[3,2,3]视为同一组合)

### Step 2:先想笨办法(暴力回溯)
用回溯枚举所有可能:每个位置都尝试选取 candidates 中的每个数字,直到和等于 target。
- 时间复杂度:O(n^(target/min)) — 最坏情况,如 candidates=[1], target=100,需要尝试100层
- 瓶颈在哪:
  1. 会生成重复组合(如先选2再选3 vs 先选3再选2)
  2. 当剩余值很小时,还要尝试大数字(明显不可能)
  3. 当剩余值为负时,还继续递归(浪费计算)

### Step 3:瓶颈分析 → 优化方向
笨办法的三大问题:
1. **重复组合** — 可以规定"每次只能选当前位置及之后的数字"来去重
2. **无效尝试** — 如果当前数字 > 剩余值,后续更大的数字也不用试了(需要排序)
3. **负数递归** — 在递归前判断 `remaining < 0` 直接return

### Step 4:选择武器
- 选用:**回溯算法 + 剪枝优化 + 排序**
- 理由:
  1. 回溯能枚举所有组合
  2. 排序后可以用 break 剪枝,避免无效尝试
  3. start 参数避免重复组合
  4. 提前判断剩余值,减少递归层数

> 🔑 **模式识别提示**:当题目出现"组合总和"、"元素可重复"、"找所有方案",优先考虑"回溯+剪枝"

---

## 🔑 解法一:朴素回溯(无剪枝)

### 思路
直接用回溯框架:
1. 从 index 位置开始遍历 candidates
2. 选择当前数字,递归处理剩余值 `target - candidate`
3. 因为可重复,递归时 index 不变(允许再次选当前数字)
4. 用 start 参数避免重复组合

### 图解过程

```
示例:candidates = [2,3,6,7], target = 7

                    root(7)
           /    /     |      \
         2(5) 3(4)  6(1)    7(0)✓ ← 直接找到[7]
        /||\  /|\    |
      2 3 6 7...    ×(1<6,无法继续)

详细展开左子树 root→2(剩余5):
        2(5)
      / | \ \
    2(3) 3(2) 6 7
   /|\   |
  2 3 6  3(-1)×
  |  ×
 2(−1)×

有效路径:
1. root→7 → [7] ✓
2. root→2→2→3 → [2,2,3] ✓

剪枝前的无效尝试:
- root→2→2→2→2(-1) — 过头了
- root→2→3→3(-1) — 过头了
- root→3→6(负数) — 过头了
```

### Python代码

```python
from typing import List


def combinationSum(candidates: List[int], target: int) -> List[List[int]]:
    """
    解法一:朴素回溯
    思路:从start位置开始尝试每个候选数,可重复选取
    """
    result = []

    def backtrack(start: int, path: List[int], remaining: int):
        """
        start: 当前可选的起始位置(避免重复组合)
        path: 当前已选数字
        remaining: 剩余需要凑的和
        """
        # 递归终止条件
        if remaining == 0:
            result.append(path[:])  # 找到有效组合
            return
        if remaining < 0:
            return  # 超了,回退

        # 从start开始遍历候选数
        for i in range(start, len(candidates)):
            num = candidates[i]
            path.append(num)                    # 选择
            backtrack(i, path, remaining - num)  # 递归(注意:i不是i+1,因为可重复)
            path.pop()                          # 撤销

    backtrack(0, [], target)
    return result


# ✅ 测试
print(combinationSum([2, 3, 6, 7], 7))   # 期望输出:[[2,2,3],[7]]
print(combinationSum([2, 3, 5], 8))      # 期望输出:[[2,2,2,2],[2,3,3],[3,5]]
print(combinationSum([2], 1))            # 期望输出:[]
```

### 复杂度分析
- **时间复杂度**:O(n^(target/min)) — n是candidates长度,最坏情况指数级
  - 具体地说:如果 candidates=[2,3], target=8,树的高度最多8/2=4层,每层最多2个分支,约2^4=16次递归
  - 实际中会因为 remaining<0 提前终止,但最坏情况仍是指数级
- **空间复杂度**:O(target/min) — 递归栈深度,最深是不断选最小值

### 优缺点
- ✅ 代码简洁,逻辑清晰
- ✅ 正确处理了元素可重复和避免重复组合
- ❌ 没有剪枝,会尝试很多明显过大的数字
- ❌ 性能较差,引出优化方向

---

## 🏆 解法二:排序+剪枝优化(最优解)

### 优化思路
在解法一的基础上加两个优化:
1. **排序**:先对 candidates 排序,使得小的数字在前
2. **剪枝**:当 `candidates[i] > remaining` 时,因为数组已排序,后面的数字更大,直接 break 跳出循环

> 💡 **关键想法**:排序是剪枝的前提——有序后才能"遇到过大元素就停止"

### 图解过程

```
示例:candidates = [2,3,6,7], target = 7 (已排序)

                    root(7)
           /    /     |      \
         2(5) 3(4)  6(1)    7(0)✓
        /||   /|     |
      2 3 6  3 6    ×(6>1,剪枝break)
      |  ×   |
     2 (3>3? 继续)
     |  3(0)✓
    2(1)
    | (3>1,剪枝break)
    ×

剪枝效果:
- 在 2(5)→3(2) 这层,本来要尝试 6,7,但因为 6>2,直接break
- 在 6(1) 这层,6>1,直接break,不再尝试7
- 大幅减少递归次数

有效路径(同解法一):
1. [7] ✓
2. [2,2,3] ✓
```

**排序前 vs 排序后的区别:**
```
未排序 [7,3,6,2], target=7:
- 先尝试7,直接命中[7]
- 再尝试3→3→... 很多无效尝试
- 后面还要尝试6(已经知道不可能了,因为3+3+6>7)

已排序 [2,3,6,7], target=7:
- 从小到大尝试,当发现6>剩余值时,立即break
- 避免尝试后续所有更大的数字
```

### Python代码

```python
def combinationSum_v2(candidates: List[int], target: int) -> List[List[int]]:
    """
    解法二:排序+剪枝优化(最优解)
    思路:先排序,遇到过大元素直接break
    """
    candidates.sort()  # 关键:排序,使剪枝有效
    result = []

    def backtrack(start: int, path: List[int], remaining: int):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            num = candidates[i]

            # 剪枝1:如果当前数字已经大于剩余值,后面更大的数字也不可能,直接break
            if num > remaining:
                break  # 注意:是break不是continue,因为数组已排序

            path.append(num)
            backtrack(i, path, remaining - num)  # i不变,允许重复选
            path.pop()

    backtrack(0, [], target)
    return result


# ✅ 测试
print(combinationSum_v2([2, 3, 6, 7], 7))   # 期望输出:[[2,2,3],[7]]
print(combinationSum_v2([2, 3, 5], 8))      # 期望输出:[[2,2,2,2],[2,3,3],[3,5]]
print(combinationSum_v2([2], 1))            # 期望输出:[]
print(combinationSum_v2([8, 7, 4, 3], 11))  # 测试排序效果:输出[[3,4,4],[3,8],[4,7]]
```

### 复杂度分析
- **时间复杂度**:O(n^(target/min)) — 理论上界不变,但实际剪枝后快很多
  - 排序的O(n log n)可忽略(n≤30)
  - 剪枝效果显著:candidates=[2,3,100,200], target=10时,尝试100和200的分支会被直接剪掉
  - 具体地说:如果 candidates=[2,3,6,7], target=7,朴素回溯约30+次递归,剪枝后仅10+次
- **空间复杂度**:O(target/min) — 递归栈深度不变

### 为什么是最优解
1. **时间最优**:剪枝后避免大量无效递归,实战中比朴素回溯快5-10倍
2. **空间最优**:O(target/min)已是理论下限(必须递归到叶子节点)
3. **代码简洁**:只需加一行 `candidates.sort()` 和一个 `if break`
4. **通用性强**:这个剪枝技巧适用于所有"组合总和"类问题

---

## ⚡ 解法三:记忆化回溯(减少重复计算)

### 优化思路
进一步优化:用哈希表记录 `(start, remaining)` 已经计算过的结果,避免重复子问题。

> 💡 **关键想法**:虽然回溯问题通常无法记忆化(因为路径不同),但可以记忆"从start开始凑remaining的所有方案"

### 图解过程

```
示例:candidates = [2,3], target = 8

                root(start=0, remaining=8)
               /                        \
        2(0,6)                          3(1,5)
        /     \                         /    \
    2(0,4)   3(1,3)                 3(1,2)  ×
    /   \     /                      /
  2(0,2) 3  3(1,0)✓               3(1,-1)×
  /
2(0,0)✓

记忆化效果:
- 如果多次递归到 backtrack(0, 6),只计算一次,结果缓存为 [[2,2,2],[2,2],[3,3]]
- 后续命中缓存,直接返回
```

注意:本题由于路径长度短,记忆化收益不大,主要用于理解思想。

### Python代码

```python
def combinationSum_v3(candidates: List[int], target: int) -> List[List[int]]:
    """
    解法三:记忆化回溯
    思路:缓存(start,remaining)的结果,避免重复计算
    """
    candidates.sort()
    memo = {}  # key: (start, remaining), value: 符合条件的组合列表

    def backtrack(start: int, remaining: int) -> List[List[int]]:
        # 查缓存
        if (start, remaining) in memo:
            return memo[(start, remaining)]

        # 递归终止
        if remaining == 0:
            return [[]]
        if remaining < 0:
            return []

        result = []
        for i in range(start, len(candidates)):
            num = candidates[i]
            if num > remaining:
                break  # 剪枝

            # 递归获取子问题的所有方案
            sub_combinations = backtrack(i, remaining - num)
            # 在每个子方案前加上当前数字
            for combination in sub_combinations:
                result.append([num] + combination)

        # 存入缓存
        memo[(start, remaining)] = result
        return result

    return backtrack(0, target)


# ✅ 测试
print(combinationSum_v3([2, 3, 6, 7], 7))   # 期望输出:[[2,2,3],[7]]
print(combinationSum_v3([2, 3, 5], 8))      # 期望输出:[[2,2,2,2],[2,3,3],[3,5]]
```

### 复杂度分析
- **时间复杂度**:O(n × target) — 最多有 n×target 个不同的 (start, remaining) 状态
- **空间复杂度**:O(n × target) — 缓存占用空间

### 优缺点
- ✅ 避免重复计算,适合子问题重叠多的场景
- ❌ 本题子问题重叠较少,记忆化收益不大
- ❌ 代码稍复杂,面试时不推荐(除非数据规模极大)

---

## 🐍 Pythonic 写法

利用生成器和递归简化代码(偏函数式风格):

```python
def combinationSum_pythonic(candidates: List[int], target: int) -> List[List[int]]:
    """Pythonic写法:更简洁的递归表达"""
    candidates.sort()

    def backtrack(start, remaining):
        if remaining == 0:
            return [[]]  # 返回包含空列表的列表,表示一种方案
        result = []
        for i in range(start, len(candidates)):
            if candidates[i] > remaining:
                break
            # 递归获取子问题,拼接当前数字
            result += [[candidates[i]] + combo
                       for combo in backtrack(i, remaining - candidates[i])]
        return result

    return backtrack(0, target)


# ✅ 测试
print(combinationSum_pythonic([2, 3, 6, 7], 7))
```

这个写法用了:
1. **列表推导式** + **递归**:一行完成"递归+拼接"
2. **返回空列表的列表 `[[]]`**:表示"有一种方案(空方案)",方便递归拼接
3. **+=** 合并子结果

> ⚠️ **面试建议**:先写清晰版本(解法二)展示思路和剪枝技巧,再提Pythonic写法展示Python功底。
> 面试官最看重的是**剪枝思想**,而非代码简洁度。

---

## 📊 解法对比

| 维度 | 解法一:朴素回溯 | 🏆 解法二:排序+剪枝(最优) | 解法三:记忆化 |
|------|--------------|---------------------|------------|
| 时间复杂度 | O(n^(t/m)) | **O(n^(t/m))** ← 剪枝后实战快5-10倍 | O(n×t) |
| 空间复杂度 | O(t/m) | **O(t/m)** ← 仅递归栈 | O(n×t) |
| 代码难度 | 简单 | **简单** ← 只加2行代码 | 中等 |
| 面试推荐 | ⭐ | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | 理解回溯框架 | **面试首选,性能最优** | 子问题重叠极多 |

注:t=target, m=min(candidates), n=len(candidates)

**为什么解法二是最优解**:
- 排序+剪枝是本题的标准解法,面试必考
- 只需一行排序和一个break,代码改动极小,收益极大
- 实战中性能提升显著(尤其candidates有大数时)
- 剪枝思想通用,适用于所有组合总和类问题

**面试建议**:
1. 先口述思路:"这是组合问题,用回溯,关键是元素可重复和剪枝优化"
2. 写出🏆解法二,边写边强调:**"排序是为了剪枝,遇到过大元素直接break"**
3. 重点解释两个技巧:
   - **index不变(i不是i+1)**:允许重复选取当前元素
   - **break不是continue**:因为数组已排序,后续元素更大
4. 手动测试边界:target=1(无法凑出)、只有一个元素、包含大数字
5. 追问时分析剪枝效果:用具体例子说明减少了多少次递归

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你解决一下这道题。

**你**:(审题30秒)好的,这道题要求找出所有和为target的组合,候选数可以重复使用,但组合不能重复。让我先分析一下...

这是典型的回溯问题。关键点有两个:
1. **元素可重复**:递归时index不变,允许再次选当前数字
2. **避免重复组合**:用start参数确保只选当前位置及之后的数字

我的优化思路是:**先排序,然后剪枝**。当遇到 `candidates[i] > remaining` 时,因为数组已排序,后面的数字更大,可以直接break跳出循环,大幅减少无效递归。

时间复杂度最坏是指数级 O(n^(target/min)),但剪枝后实际快很多。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
def combinationSum(candidates, target):
    candidates.sort()  # 关键:排序,为剪枝做准备
    result = []

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])  # 找到有效组合
            return

        for i in range(start, len(candidates)):
            num = candidates[i]

            # 剪枝:当前数字已经大于剩余值,后面更大的也不可能
            if num > remaining:
                break  # 注意是break,不是continue!

            path.append(num)
            backtrack(i, path, remaining - num)  # i不变,允许重复
            path.pop()

    backtrack(0, [], target)
    return result
```

我这里用了两个技巧:
1. **递归时传i而不是i+1**:这样可以重复选取当前数字,比如 [2,2,3]
2. **break而不是continue**:因为数组已排序,如果当前数字过大,后面的更大,没必要继续尝试

**面试官**:测试一下?

**你**:用示例 candidates=[2,3,6,7], target=7 走一遍...(手动模拟)
- 先尝试2:2→2→2→2=8,超了,回退
- 2→2→3=7,找到第一个组合 [2,2,3]
- 回退后尝试3:3+6>7,break剪枝,跳过6和7
- 直接尝试7:7=7,找到第二个组合 [7]
- 共2种组合,符合预期

再测边界情况:candidates=[2], target=1,因为2>1,直接break,返回空列表,正确。

**面试官**:如果不排序会怎样?

**你**:如果不排序,比如 candidates=[7,3,6,2], target=7:
- 当尝试到3→6时,6>剩余值,但我们无法判断后面的2是否可行
- 无法用break剪枝,只能全部尝试,性能差很多
- 结果仍然正确,但时间复杂度实战中会差5-10倍

所以排序是这道题优化的关键,牺牲O(n log n)换来大量剪枝,绝对值得。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "如果元素不能重复使用呢?" | "递归时传i+1而不是i,这样每个元素最多用一次。其他逻辑不变。(对应LC 40组合总和II)" |
| "如果candidates有重复元素呢?" | "需要先排序,然后在循环中加去重逻辑:if i>start and candidates[i]==candidates[i-1]: continue,跳过重复元素。(对应LC 40)" |
| "如果target很大,如10000?" | "剪枝效果会更明显。如果candidates最小值是2,最多递归5000层,可能栈溢出,需要考虑迭代DP或记忆化。" |
| "能用DP解决吗?" | "可以,这是完全背包问题的变种。dp[i]表示凑出和i的所有组合,但DP难以记录所有路径,回溯更直观。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:排序为剪枝铺路 — 关键优化
candidates.sort()  # 从小到大排序
for num in candidates:
    if num > remaining:
        break  # 排序后可以break,省去后续所有尝试

# 技巧2:break vs continue的区别
# break:直接跳出循环(用于排序后的剪枝)
# continue:跳过本次,继续下次(用于去重)

# 技巧3:递归参数传i还是i+1
backtrack(i, ...)      # 传i:当前元素可重复选取
backtrack(i+1, ...)    # 传i+1:当前元素不可重复选取
```

### 💡 底层原理(选读)

> **为什么排序能加速剪枝?**
>
> - **有序性**是剪枝的前提:只有当数组有序时,才能根据"当前元素过大"推断"后续元素也过大"
> - 如果数组无序,比如 [5, 2, 3],当remaining=4,遇到5>4时,不能break,因为后面还有2和3可能满足
> - 排序后变为 [2, 3, 5],当remaining=4,遇到5>4时,可以确定break,因为后续只会更大
>
> **这类剪枝的数学依据**:
> - 前提:数组有序(单调递增)
> - 推理:如果 candidates[i] > remaining,则 candidates[j] > remaining (对所有 j>i)
> - 结论:可以提前终止循环(break)
>
> **剪枝效果量化**:
> - 不排序:candidates=[3,100,200,2], target=5 → 尝试3,100(×),200(×),2,共4次
> - 排序后:candidates=[2,3,100,200], target=5 → 尝试2,3,遇到100>5直接break,仅2次
> - 候选数越多、差距越大,剪枝效果越明显

### 算法模式卡片 📐
- **模式名称**:回溯+剪枝(排序优化)
- **适用条件**:组合类问题,需要枚举所有方案,候选集合可排序
- **识别关键词**:"组合总和"、"元素可重复"、"找所有组合"
- **模板代码**:
```python
def combination_sum_template(candidates, target):
    candidates.sort()  # 步骤1:排序
    result = []

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return

        for i in range(start, len(candidates)):
            num = candidates[i]
            if num > remaining:  # 步骤2:剪枝
                break
            path.append(num)
            backtrack(i, path, remaining - num)  # i or i+1 取决于能否重复
            path.pop()

    backtrack(0, [], target)
    return result
```

### 易错点 ⚠️
1. **break vs continue混淆** — 排序后应该用break(提前终止),而不是continue(跳过本次)
2. **递归传参错误** — 元素可重复应传i,不可重复应传i+1,别搞反
3. **忘记排序** — 如果不排序,break剪枝会出错,导致漏掉结果
4. **路径拷贝忘记** — result.append(path) 应该是 result.append(path[:]),否则path变化会影响已收集结果

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:凑单优惠系统** — 电商中"满减凑单",用户已选商品价格sum,还需凑够target,推荐哪些商品组合
- **场景2:资源分配** — 服务器有CPU核心数限制,多个任务需要不同核心数,如何组合任务使资源利用率最高
- **场景3:找零问题** — 收银系统中,用有限面额的纸币硬币凑出找零金额,枚举所有方案(实际中通常用贪心)
- **场景4:背包问题变种** — 完全背包中,枚举所有达到容量上限的物品组合方案

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 40. 组合总和 II | Medium | 回溯+去重 | 元素不能重复使用,且candidates有重复,需要去重逻辑 |
| LeetCode 216. 组合总和 III | Medium | 回溯+约束 | 只能用1-9,且恰好k个数,双重约束 |
| LeetCode 377. 组合总和 IV | Medium | 动态规划 | 求方案数,不需要具体路径,用DP更优 |
| LeetCode 322. 零钱兑换 | Medium | 完全背包DP | 求最少硬币数,DP比回溯效率高 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:假设题目改为"找出和为target的组合,但要求结果中组合按长度从小到大排序"。如何修改代码?

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

在收集结果时,不直接append到result,而是用字典 `{长度: [组合列表]}` 分组存储,最后按长度排序输出。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def combinationSum_sorted_by_length(candidates: List[int], target: int) -> List[List[int]]:
    candidates.sort()
    from collections import defaultdict
    length_map = defaultdict(list)  # {长度: 组合列表}

    def backtrack(start, path, remaining):
        if remaining == 0:
            length_map[len(path)].append(path[:])  # 按长度分组
            return

        for i in range(start, len(candidates)):
            num = candidates[i]
            if num > remaining:
                break
            path.append(num)
            backtrack(i, path, remaining - num)
            path.pop()

    backtrack(0, [], target)

    # 按长度从小到大输出
    result = []
    for length in sorted(length_map.keys()):
        result.extend(length_map[length])
    return result


# 测试
print(combinationSum_sorted_by_length([2, 3, 6, 7], 7))
# 输出:[[7], [2,2,3]] (长度1的在前,长度3的在后)
```

**核心思想**:在回溯框架不变的情况下,通过后处理(分组+排序)实现额外需求。这体现了算法的**模块化思维**——核心逻辑(回溯)和展示逻辑(排序)可以分离。

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
