> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 📖 第72课:杨辉三角

> **模块**:动态规划 | **难度**:Easy ⭐
> **LeetCode 链接**:https://leetcode.cn/problems/pascals-triangle/
> **前置知识**:无
> **预计学习时间**:20分钟

---

## 🎯 题目描述

给定一个非负整数 numRows,生成杨辉三角的前 numRows 行。

在杨辉三角中,每个数是它左上方和右上方的数的和。

**示例:**
```
输入:numRows = 5
输出:[[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]

可视化:
    1
   1 1
  1 2 1
 1 3 3 1
1 4 6 4 1
```

**约束条件:**
- 1 <= numRows <= 30
- 需要返回二维数组,不是打印

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 最小输入 | numRows=1 | [[1]] | 只有顶点 |
| 小规模 | numRows=2 | [[1],[1,1]] | 第二行有两个1 |
| 中等规模 | numRows=5 | [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]] | 验证递推逻辑 |
| 大规模 | numRows=30 | 30行三角形 | 性能边界 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在搭积木金字塔,从顶端的1个积木开始,每一层的积木数量都比上一层多1。关键规则是:**每个积木上的数字 = 它左上方的数字 + 右上方的数字**。
>
> 🐌 **笨办法**:用数学公式计算每个位置的值(组合数公式 C(n,k)),需要计算阶乘,容易溢出且效率不高。
>
> 🚀 **聪明办法**:观察规律——每一行的第一个和最后一个数字都是1,中间的数字都是"上一行相邻两个数字之和"。所以我们可以**逐行生成**:先在两端放1,中间的数字从上一行推导出来。这就是**二维动态规划的入门思想**。

### 关键洞察
**当前行的每个数字(除了首尾的1)= 上一行相邻两个数字之和**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:一个正整数 numRows,表示要生成的行数
- **输出**:一个二维列表,包含杨辉三角的前 numRows 行
- **限制**:每行第一个和最后一个数字是1,中间数字由上一行推导

### Step 2:先想笨办法(数学公式法)
杨辉三角第n行第k个数字的数学公式是组合数 C(n, k) = n! / (k! * (n-k)!),可以直接计算。
- 时间复杂度:O(numRows²) — 需要计算每个位置的组合数
- 瓶颈在哪:计算阶乘容易溢出,且需要重复计算,效率不高

### Step 3:瓶颈分析 → 优化方向
观察杨辉三角的规律:每一行的数字都可以从上一行推导出来,不需要每次重新计算阶乘。
- 核心问题:数学公式计算复杂,且没有利用前一行的结果
- 优化思路:能不能基于前一行直接生成当前行?→用**逐行递推**

### Step 4:选择武器
- 选用:**动态规划(二维DP)**
- 理由:当前行依赖上一行,具有明显的递推关系,适合DP自底向上生成

> 🔑 **模式识别提示**:当题目出现"第n行依赖第n-1行"、"逐层构建"时,优先考虑"动态规划"

---

## 🔑 解法一:逐行递推(标准DP)

### 思路
从第一行开始,逐行生成杨辉三角。每一行的首尾都是1,中间的数字通过访问上一行相邻两个位置相加得到。

### 图解过程

```
生成过程(以 numRows=5 为例):

第1行: [1]
       ↓
第2行: [1, 1]
       ↓ ↓ ↓
第3行: [1, 2, 1]
         ↗ ↖ ↗ ↖
       1+1=2
       ↓
第4行: [1, 3, 3, 1]
         ↗ ↖ ↗ ↖ ↗ ↖
       1+2=3  2+1=3
       ↓
第5行: [1, 4, 6, 4, 1]
         ↗ ↖ ↗ ↖ ↗ ↖ ↗ ↖
       1+3=4  3+3=6  3+1=4

规律总结:
  row[j] = prev_row[j-1] + prev_row[j]  (1 <= j < len(row)-1)
```

### Python代码

```python
from typing import List


def generate_pascal_triangle(numRows: int) -> List[List[int]]:
    """
    解法一:逐行递推(标准DP)
    思路:基于上一行生成当前行,首尾为1,中间为上一行相邻两数之和
    """
    result = []

    for i in range(numRows):
        # 创建当前行,长度为 i+1,初始化为1
        row = [1] * (i + 1)

        # 计算中间元素(第一个和最后一个保持为1)
        if i >= 2:  # 从第3行开始才有中间元素
            prev_row = result[i - 1]  # 上一行
            for j in range(1, i):  # j 从 1 到 i-1
                row[j] = prev_row[j - 1] + prev_row[j]

        result.append(row)

    return result


# ✅ 测试
print(generate_pascal_triangle(1))  # 期望输出: [[1]]
print(generate_pascal_triangle(3))  # 期望输出: [[1],[1,1],[1,2,1]]
print(generate_pascal_triangle(5))  # 期望输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
```

### 复杂度分析
- **时间复杂度**:O(numRows²) — 总共生成 1+2+3+...+numRows = numRows*(numRows+1)/2 个数字
  - 具体地说:如果 numRows=30,大约需要生成 30*31/2 = 465 个数字
- **空间复杂度**:O(numRows²) — 存储整个三角形的所有数字

### 优缺点
- ✅ 代码清晰,易于理解
- ✅ 利用递推关系,避免重复计算
- ⚠️ 空间复杂度无法优化(题目要求返回整个三角形)

---

## 🏆 解法二:优化写法(代码简化,最优解)

### 优化思路
观察解法一,我们可以在生成每一行时边遍历边计算,不需要单独判断 `i >= 2`。而且可以用更简洁的方式访问上一行。

> 💡 **关键想法**:简化代码逻辑,让循环更统一

### 图解过程

```
优化思路:
  对于每一行,先创建全是1的数组,然后只修改中间部分

生成第4行 [1, 3, 3, 1] 的过程:
  1. 创建: row = [1, 1, 1, 1]  (长度为4)
  2. 修改: row[1] = prev_row[0] + prev_row[1] = 1 + 2 = 3
          row[2] = prev_row[1] + prev_row[2] = 2 + 1 = 3
  3. 保持: row[0] = 1, row[3] = 1 不变
```

### Python代码

```python
def generate_optimized(numRows: int) -> List[List[int]]:
    """
    解法二:优化写法 — 🏆最优解
    思路:与解法一相同,但代码更简洁
    """
    result = []

    for i in range(numRows):
        # 当前行初始化为全1
        row = [1] * (i + 1)

        # 更新中间元素
        for j in range(1, i):
            row[j] = result[i - 1][j - 1] + result[i - 1][j]

        result.append(row)

    return result


# ✅ 测试
print(generate_optimized(5))  # 期望输出: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
```

### 复杂度分析
- **时间复杂度**:O(numRows²) — 与解法一相同
- **空间复杂度**:O(numRows²) — 与解法一相同

**为什么是最优解**:
- 时间复杂度已经最优(必须生成所有数字)
- 空间复杂度受题目限制(必须返回整个三角形)
- 代码更简洁,循环逻辑更清晰

---

## 🐍 Pythonic 写法

利用 Python 的列表推导式和 zip 函数,可以写出非常简洁的版本:

```python
# 方法一:列表推导式
def generate_pythonic(numRows: int) -> List[List[int]]:
    result = [[1]]
    for _ in range(numRows - 1):
        prev = result[-1]
        # 构造当前行: [1] + 中间部分 + [1]
        row = [1] + [prev[i] + prev[i + 1] for i in range(len(prev) - 1)] + [1]
        result.append(row)
    return result if numRows > 0 else []


# 方法二:使用 zip 妙用
def generate_zip(numRows: int) -> List[List[int]]:
    result = [[1]]
    for _ in range(numRows - 1):
        prev = result[-1]
        # zip([1,2,1], [0,1,2]) = [(1,0), (2,1), (1,2)]
        # 但我们需要 [1+2, 2+1] = [3, 3]
        # 技巧: zip(prev, prev[1:]) = [(1,2), (2,1)]
        row = [1] + [a + b for a, b in zip(prev, prev[1:])] + [1]
        result.append(row)
    return result if numRows > 0 else []
```

`zip(prev, prev[1:])` 的巧妙之处:
- `prev = [1, 2, 1]`
- `prev[1:] = [2, 1]`
- `zip(prev, prev[1:]) = [(1,2), (2,1)]` — 恰好是相邻两个数的配对!
- `[a+b for a,b in zip(...)]` = `[1+2, 2+1]` = `[3, 3]`

> ⚠️ **面试建议**:先写清晰版本展示思路,再提 Pythonic 写法展示语言功底。
> 面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:标准递推 | 🏆 解法二:优化写法(最优) |
|------|---------------|----------------------|
| 时间复杂度 | O(numRows²) | **O(numRows²)** ← 理论最优 |
| 空间复杂度 | O(numRows²) | **O(numRows²)** ← 题目要求 |
| 代码难度 | 简单 | 简单 |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 清晰展示逻辑 | **代码简洁,面试首选** |

**为什么解法二是最优解**:
- 时间复杂度 O(numRows²) 已经是理论最优(必须生成所有数字,无法更快)
- 空间复杂度受题目限制(必须返回整个三角形,无法更省)
- 代码更简洁,循环逻辑更统一,面试中更容易写对

**面试建议**:
1. 先用1分钟画图展示杨辉三角的递推规律:"每个数 = 左上 + 右上"
2. 立即写出🏆最优解的代码,强调"首尾为1,中间从上一行推导"
3. 手动模拟生成前3行的过程,展示对递推的理解
4. 测试边界用例(numRows=1),验证代码正确性
5. 如果时间允许,展示 Pythonic 写法(`zip`技巧),加分项

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你生成杨辉三角的前n行。

**你**:(审题30秒,画出示例)好的,杨辉三角的规律是:每一行的第一个和最后一个数字都是1,中间的数字是上一行相邻两个数字之和。比如第4行的3 = 上一行的1+2。

我的思路是逐行生成:从第1行 [1] 开始,每次基于上一行生成当前行。具体步骤是:
1. 创建长度为 i+1 的数组,全部初始化为1
2. 遍历中间位置(从1到i-1),计算 `row[j] = prev_row[j-1] + prev_row[j]`
3. 将当前行加入结果

时间复杂度 O(n²),空间复杂度 O(n²),都是最优的。

**面试官**:很好,请写代码。

**你**:(边写边说)
```python
def generate(numRows):
    result = []
    for i in range(numRows):
        row = [1] * (i + 1)  # 先全部填1
        for j in range(1, i):  # 更新中间元素
            row[j] = result[i - 1][j - 1] + result[i - 1][j]
        result.append(row)
    return result
```

**面试官**:测试一下?

**你**:用 numRows=3 走一遍:
- i=0: row=[1], result=[[1]]
- i=1: row=[1,1], result=[[1],[1,1]]
- i=2: row=[1,1,1], 更新 row[1]=result[1][0]+result[1][1]=1+1=2, 得到 [1,2,1], result=[[1],[1,1],[1,2,1]]
结果正确。边界情况 numRows=1 返回 [[1]],也正确。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "能不能只返回第n行,而不生成整个三角形?" | "可以!只需要维护当前行和上一行,空间优化到 O(n)。核心思想是用两个数组交替更新,或者用一个数组从后往前更新(避免覆盖)。" |
| "如何计算杨辉三角第n行第k个数?" | "可以用组合数公式 C(n,k),或者用递推关系从第1行一直算到第n行。如果只要一个数,用公式更快。" |
| "杨辉三角有什么应用?" | "杨辉三角在数学中就是组合数表,应用很广:二项式展开系数、概率计算(如抛硬币)、组合优化问题等。" |
| "能否用递归实现?" | "可以,递归计算 C(n,k) = C(n-1,k-1) + C(n-1,k),但需要记忆化避免重复计算,本质上和迭代DP相同。" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:列表推导式 — 简洁生成列表
row = [1] + [prev[i] + prev[i + 1] for i in range(len(prev) - 1)] + [1]

# 技巧2:zip 妙用 — 生成相邻元素配对
# zip([1,2,1], [2,1]) = [(1,2), (2,1)]
row = [1] + [a + b for a, b in zip(prev, prev[1:])] + [1]

# 技巧3:列表切片 — prev[1:] 从第2个元素到末尾
prev = [1, 2, 1]
prev[1:] = [2, 1]  # 去掉第一个元素
```

### 💡 底层原理(选读)

> **为什么杨辉三角和组合数有关?**
>
> 杨辉三角第n行第k个数字(从0开始计数)恰好是组合数 C(n, k) = n! / (k! * (n-k)!)。
>
> 组合数的递推关系是:C(n, k) = C(n-1, k-1) + C(n-1, k)
> 这恰好对应杨辉三角的"左上+右上"规则!
>
> **应用举例**:
> - 二项式展开:(a+b)^n 的系数就是杨辉三角第n行
>   - (a+b)² = 1·a² + 2·ab + 1·b² → 系数 [1, 2, 1]
>   - (a+b)³ = 1·a³ + 3·a²b + 3·ab² + 1·b³ → 系数 [1, 3, 3, 1]
>
> - 概率计算:抛3次硬币,恰好2次正面的概率 = C(3,2)/2³ = 3/8

### 算法模式卡片 📐
- **模式名称**:二维DP(逐行递推)
- **适用条件**:当前行依赖上一行,需要生成多行结果
- **识别关键词**:"杨辉三角"、"帕斯卡三角"、"逐行生成"、"上一行推导当前行"
- **模板代码**:
```python
def generate_rows(n):
    result = []
    for i in range(n):
        # 初始化当前行
        row = [initial_value] * (i + 1)

        # 基于上一行更新当前行
        if i > 0:
            prev_row = result[i - 1]
            for j in range(1, i):
                row[j] = transition_function(prev_row, j)

        result.append(row)
    return result
```

### 易错点 ⚠️
1. **索引越界**:访问 `result[i - 1]` 时忘记判断 i > 0,导致访问 result[-1] 拿到错误的最后一行
   - **正确做法**:确保 i >= 1 时才访问上一行,或者从 i=1 开始循环

2. **边界处理错误**:忘记处理 numRows=0 或 numRows=1 的情况
   - **正确做法**:在函数开头判断 `if numRows == 0: return []`

3. **修改了首尾元素**:循环范围写成 `range(0, i+1)` 导致覆盖了首尾的1
   - **正确做法**:循环范围应该是 `range(1, i)`,只修改中间元素

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:数据分析 — 计算多项式展开系数,用于信号处理中的滤波器设计(如二项式滤波器)
- **场景2**:概率计算 — 在金融风控中计算多次独立事件的组合概率(如贷款违约概率)
- **场景3**:图形渲染 — 贝塞尔曲线的系数计算使用杨辉三角中的组合数

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 119. 杨辉三角 II | Easy | DP空间优化 | 只返回第n行,可以用O(n)空间的滚动数组 |
| LeetCode 120. 三角形最小路径和 | Medium | DP递推 | 类似杨辉三角的结构,但求最小路径和而非生成三角形 |
| LeetCode 931. 下降路径最小和 | Medium | 二维DP | 在二维矩阵中从上到下找最小路径,递推关系类似 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:如何只用 O(n) 空间生成杨辉三角的第 n 行?(不需要返回整个三角形)

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

只需要维护当前行和上一行,用两个数组交替更新;或者用一个数组从后往前更新(避免覆盖)。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def getRow(rowIndex: int) -> List[int]:
    """
    只返回杨辉三角的第 rowIndex 行(从0开始计数)
    空间优化到 O(n)
    """
    # 从后往前更新,避免覆盖
    row = [1] * (rowIndex + 1)

    for i in range(2, rowIndex + 1):
        # 从后往前更新中间元素
        for j in range(i - 1, 0, -1):
            row[j] = row[j] + row[j - 1]

    return row


# 测试
print(getRow(3))  # 输出: [1, 3, 3, 1]
print(getRow(4))  # 输出: [1, 4, 6, 4, 1]
```

**核心思路**:
1. 初始化长度为 rowIndex+1 的数组,全部填1
2. 从第2行开始逐行更新(第0行和第1行都是全1,不需要更新)
3. **关键技巧**:从后往前更新 `row[j] = row[j] + row[j-1]`
   - 为什么从后往前?如果从前往后,`row[j]` 被更新后,计算 `row[j+1]` 时需要的旧值 `row[j]` 已经被覆盖了
   - 从后往前更新时,`row[j-1]` 还是旧值,不会被覆盖

**举例**:生成第3行 [1, 3, 3, 1]
- 初始: row = [1, 1, 1, 1]
- i=2: j=1, row[1] = row[1] + row[0] = 1+1 = 2 → [1, 2, 1, 1]
- i=3: j=2, row[2] = row[2] + row[1] = 1+2 = 3 → [1, 2, 3, 1]
        j=1, row[1] = row[1] + row[0] = 2+1 = 3 → [1, 3, 3, 1]

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
