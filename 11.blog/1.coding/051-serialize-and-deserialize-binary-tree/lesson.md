# 📖 第51课:序列化与反序列化

> **模块**:二叉树 | **难度**:Hard ⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/
> **前置知识**:第39课(二叉树中序遍历)、第44课(层序遍历)、第47课(前序+中序构造树)
> **预计学习时间**:35分钟

---

## 🎯 题目描述

设计一个算法,将二叉树序列化成字符串,并能将字符串反序列化恢复成原二叉树。你可以使用任何序列化方法,只要保证二叉树能被正确地序列化和反序列化即可。

**示例:**
```
输入:root = [1,2,3,null,null,4,5]
     1
    / \
   2   3
      / \
     4   5

序列化:"1,2,null,null,3,4,null,null,5,null,null"
反序列化:根据字符串重建上面的树
```

**约束条件:**
- 树中节点数在 [0, 10^4] 范围内
- -1000 <= Node.val <= 1000
- 必须支持包含null的完整结构信息

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 序列化结果 | 考察点 |
|---------|------|-----------|--------|
| 空树 | root=null | "null" | 边界处理 |
| 单节点 | root=[1] | "1,null,null" | 基本功能 |
| 只有左子树 | root=[1,2,null] | "1,2,null,null,null" | 非完全树 |
| 完全二叉树 | root=[1,2,3,4,5,6,7] | 正常序列化 | 标准情况 |
| 包含负数 | root=[-1,0,1] | "-1,0,null,null,1,null,null" | 负数处理 |

---

## 💡 思路引导

### 生活化比喻
> 想象你要把一个复杂的家族树"打包快递"到另一个城市,然后在那边完整还原。
>
> 🐌 **笨办法**:拍照发过去——但照片无法表达空节点的位置信息,收到照片后无法确定树的确切结构。
>
> 🚀 **聪明办法**:像读一本书一样按顺序报出每个位置的信息,包括"这里是空的"。比如:"根是1,左孩子是2,2的左孩子是空,2的右孩子是空,根的右孩子是3..."。这样对方就能完整还原整棵树了。

### 关键洞察
**序列化的关键是保存"结构信息",即每个节点的左右孩子位置,包括null;反序列化时按相同顺序读取即可还原。**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树的根节点 root
- **输出**:
  - 序列化:返回字符串(表示树的结构)
  - 反序列化:从字符串重建树,返回根节点
- **限制**:
  - 必须保存结构信息(包括null)
  - 序列化和反序列化必须是互逆操作

### Step 2:先想笨办法(层序遍历+队列)
用BFS层序遍历,把每个节点(包括null)都存入字符串,反序列化时也按层序还原。
- 时间复杂度:O(n)
- 问题:会产生大量的"null"占位符,字符串很长

### Step 3:瓶颈分析 → 优化方向
笨办法的问题:
- 层序遍历会在最后一层产生大量null
- 字符串冗长

优化思路:
- **前序遍历**:更紧凑,因为可以通过递归自然地处理null,不需要显式存储所有末尾的null
- **递归设计**:序列化和反序列化都用递归,代码简洁

### Step 4:选择武器
- 选用:**前序DFS序列化 + 递归反序列化**
- 理由:
  1. 前序遍历(根→左→右)顺序清晰,容易理解
  2. 递归写法简洁,代码量少
  3. 相比层序遍历,前序遍历的字符串更短(不需要补全所有层的null)

> 🔑 **模式识别提示**:当题目需要"保存和恢复树结构"时,考虑"前序DFS+递归"或"层序BFS+队列"

---

## 🔑 解法一:前序DFS序列化(递归法)

### 思路
按照前序遍历(根→左→右)的顺序,把每个节点的值(包括null)用逗号分隔存入字符串。反序列化时按相同顺序递归构建树。

### 图解过程

```
示例:
     1
    / \
   2   3
      / \
     4   5

步骤1:前序遍历序列化
访问顺序:1 → 2 → 2的左(null) → 2的右(null) → 3 → 4 → 4的左(null) → ...

序列化字符串:"1,2,null,null,3,4,null,null,5,null,null"

解读:
1          ← 根节点1
 2         ← 1的左孩子2
  null    ← 2的左孩子null
  null    ← 2的右孩子null
 3         ← 1的右孩子3
  4        ← 3的左孩子4
   null   ← 4的左孩子null
   null   ← 4的右孩子null
  5        ← 3的右孩子5
   null   ← 5的左孩子null
   null   ← 5的右孩子null

步骤2:反序列化(按相同前序顺序读取)

队列:["1","2","null","null","3","4","null","null","5","null","null"]

递归构建:
- 读"1",创建根节点1
  - 递归构建左子树:读"2",创建节点2
    - 递归构建2的左子树:读"null",返回None
    - 递归构建2的右子树:读"null",返回None
  - 递归构建右子树:读"3",创建节点3
    - 递归构建3的左子树:读"4",创建节点4
      - 读"null",返回None
      - 读"null",返回None
    - 递归构建3的右子树:读"5",创建节点5
      - 读"null",返回None
      - 读"null",返回None

最终还原出原树!
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Codec:
    """
    解法一:前序DFS序列化 + 递归反序列化
    思路:前序遍历保存节点值,null用占位符表示
    """

    def serialize(self, root: Optional[TreeNode]) -> str:
        """将树序列化为字符串"""
        result = []

        def dfs(node):
            if not node:
                result.append("null")  # 空节点用"null"表示
                return

            # 前序遍历:根 → 左 → 右
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(result)  # 用逗号连接

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """从字符串反序列化为树"""
        values = iter(data.split(","))  # 创建迭代器

        def build():
            val = next(values)  # 按顺序取下一个值
            if val == "null":
                return None

            # 前序遍历:根 → 左 → 右
            node = TreeNode(int(val))
            node.left = build()   # 递归构建左子树
            node.right = build()  # 递归构建右子树
            return node

        return build()


# ✅ 测试
codec = Codec()

# 测试1:正常树
root1 = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
serialized1 = codec.serialize(root1)
print(f"序列化:{serialized1}")  # "1,2,null,null,3,4,null,null,5,null,null"

deserialized1 = codec.deserialize(serialized1)
print(f"反序列化验证:{codec.serialize(deserialized1)}")  # 应该与原字符串相同

# 测试2:空树
root2 = None
serialized2 = codec.serialize(root2)
print(f"空树序列化:{serialized2}")  # "null"

# 测试3:单节点
root3 = TreeNode(1)
serialized3 = codec.serialize(root3)
print(f"单节点序列化:{serialized3}")  # "1,null,null"
```

### 复杂度分析
- **时间复杂度**:O(n) — 序列化和反序列化都遍历每个节点一次
  - 具体地说:如果树有1000个节点,序列化需要1000次操作,反序列化也需要1000次
- **空间复杂度**:
  - 序列化:O(n) 用于存储结果字符串
  - 反序列化:O(h) 递归栈深度 + O(n) 分割后的列表

### 优缺点
- ✅ 代码简洁,递归逻辑清晰
- ✅ 前序遍历易于理解和实现
- ✅ 字符串相对紧凑(比层序少很多null)
- ❌ 需要理解递归和迭代器的配合

---

## 🏆 解法二:层序BFS序列化(队列法,最优解)

### 优化思路
用BFS层序遍历进行序列化,这样可以更直观地看到树的层次结构。虽然会产生一些额外的null,但逻辑更直观,且更容易扩展到其他应用场景(如树的可视化)。

> 💡 **关键想法**:层序遍历用队列实现,序列化和反序列化都用队列,逻辑对称,易于理解

### 图解过程

```
示例:
     1
    / \
   2   3
      / \
     4   5

步骤1:层序遍历序列化

队列初始:[1]
输出:1
子节点入队:[2, 3]

队列:[2, 3]
输出:2, null, null (2的左右孩子)
子节点入队:无(都是null)

队列:[3]
输出:3
子节点入队:[4, 5]

队列:[4, 5]
输出:4, null, null (4的左右孩子)

队列:[5]
输出:5, null, null (5的左右孩子)

队列:空,结束

序列化字符串:"1,2,3,null,null,4,5,null,null,null,null"

步骤2:反序列化(按层序还原)

分割:["1","2","3","null","null","4","5","null","null","null","null"]
索引:i=0

i=0:创建根节点1,队列=[1]
i=1,2:节点1的左孩子2,右孩子3,队列=[2,3]
i=3,4:节点2的左孩子null,右孩子null
i=5,6:节点3的左孩子4,右孩子5,队列=[4,5]
i=7,8:节点4的左孩子null,右孩子null
i=9,10:节点5的左孩子null,右孩子null

完成!
```

### Python代码

```python
from typing import Optional
from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Codec2:
    """
    解法二:层序BFS序列化(最优解)
    思路:用队列进行层序遍历,逻辑直观
    """

    def serialize(self, root: Optional[TreeNode]) -> str:
        """BFS层序遍历序列化"""
        if not root:
            return "null"

        result = []
        queue = deque([root])

        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.append(node.left)   # 即使是None也要入队
                queue.append(node.right)
            else:
                result.append("null")

        return ",".join(result)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """BFS层序遍历反序列化"""
        if data == "null":
            return None

        values = data.split(",")
        root = TreeNode(int(values[0]))
        queue = deque([root])
        i = 1  # 从第二个值开始

        while queue:
            node = queue.popleft()

            # 处理左孩子
            if i < len(values) and values[i] != "null":
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1

            # 处理右孩子
            if i < len(values) and values[i] != "null":
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)
            i += 1

        return root


# ✅ 测试
codec2 = Codec2()

# 测试1:正常树
root1 = TreeNode(1, TreeNode(2), TreeNode(3, TreeNode(4), TreeNode(5)))
serialized1 = codec2.serialize(root1)
print(f"BFS序列化:{serialized1}")

deserialized1 = codec2.deserialize(serialized1)
print(f"BFS反序列化验证:{codec2.serialize(deserialized1)}")

# 测试2:只有左子树
root2 = TreeNode(1, TreeNode(2, TreeNode(3)))
serialized2 = codec2.serialize(root2)
print(f"左子树序列化:{serialized2}")
```

### 复杂度分析
- **时间复杂度**:O(n) — BFS遍历每个节点一次
- **空间复杂度**:
  - 序列化:O(n) 队列和结果字符串
  - 反序列化:O(n) 队列和分割后的列表

### 优缺点
- ✅ 逻辑直观:按层序遍历,容易理解
- ✅ 便于调试:可以直接看到每一层的节点
- ✅ 易扩展:可以方便地改为打印层次结构
- ✅ 面试推荐:队列操作标准,不易出错

---

## 🐍 Pythonic 写法

利用Python的生成器和迭代器让代码更简洁:

```python
class CodecPythonic:
    """Pythonic风格:使用生成器"""

    def serialize(self, root: Optional[TreeNode]) -> str:
        """用生成器简化前序遍历"""
        def gen(node):
            if node:
                yield str(node.val)
                yield from gen(node.left)
                yield from gen(node.right)
            else:
                yield "null"

        return ",".join(gen(root))

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """用迭代器自动移动指针"""
        def build(vals):
            val = next(vals)
            if val == "null":
                return None
            node = TreeNode(int(val))
            node.left = build(vals)
            node.right = build(vals)
            return node

        return build(iter(data.split(",")))
```

**解释**:
- `yield from` 可以优雅地递归生成值
- `iter()` 创建迭代器,`next()` 自动维护位置

> ⚠️ **面试建议**:先写标准版本展示思路,再提Pythonic写法展示语言功底。面试官更看重你的**思考过程**,而非代码行数。

---

## 📊 解法对比

| 维度 | 解法一:前序DFS | 🏆 解法二:层序BFS(最优) |
|------|-------------|---------------------|
| 时间复杂度 | O(n) | **O(n)** ← 相同 |
| 空间复杂度 | O(h)递归栈 | **O(w)** 队列最大宽度 |
| 字符串长度 | 较短 | 稍长(更多null) |
| 代码难度 | 中等(递归+迭代器) | **简单(纯队列操作)** |
| 面试推荐 | ⭐⭐ | **⭐⭐⭐** ← 首选 |
| 适用场景 | 字符串紧凑性要求高 | **通用,易理解和扩展** |

**为什么层序BFS是最优解**:
- 逻辑直观:按层遍历符合人的思维习惯,代码不易出错
- 易于扩展:可以方便地改为打印层次结构、可视化树等
- 面试友好:队列操作标准,面试官容易理解你的思路

**面试建议**:
1. 优先讲解层序BFS方法(解法二),因为逻辑最清晰
2. 强调关键点:
   - 序列化:null节点也要入队,保证结构信息完整
   - 反序列化:用索引i依次读取左右孩子
3. 如果时间充裕,可以提及前序DFS方法作为优化(字符串更短)
4. 手动trace一个小例子,展示序列化→反序列化的完整过程

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请设计二叉树的序列化和反序列化算法。

**你**:(审题30秒)好的,这道题的核心是:
1. 序列化要保存**结构信息**,包括哪里是null
2. 反序列化要能按相同顺序还原树

我的方案是用**层序BFS**:
- 序列化:用队列层序遍历,每个节点(包括null)都存入字符串
- 反序列化:用队列,按顺序读取值,依次连接左右孩子

时间复杂度O(n),每个节点访问一次。

**面试官**:很好,请写一下代码。

**你**:(边写边说)
```python
from collections import deque

class Codec:
    def serialize(self, root):
        if not root:
            return "null"

        result = []
        queue = deque([root])

        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.append(node.left)   # null也要入队
                queue.append(node.right)
            else:
                result.append("null")

        return ",".join(result)

    def deserialize(self, data):
        if data == "null":
            return None

        values = data.split(",")
        root = TreeNode(int(values[0]))
        queue = deque([root])
        i = 1

        while queue:
            node = queue.popleft()

            # 左孩子
            if values[i] != "null":
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1

            # 右孩子
            if values[i] != "null":
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)
            i += 1

        return root
```

关键点:
1. 序列化时,null节点也要加入队列,确保位置信息不丢失
2. 反序列化时,用索引i依次读取左右孩子的值

**面试官**:测试一下?

**你**:用示例[1,2,3,null,null,4,5]:
- 序列化:"1,2,3,null,null,4,5,null,null,null,null"
- 反序列化:
  - i=0:创建根1,队列=[1]
  - i=1,2:1的左孩子2,右孩子3,队列=[2,3]
  - i=3,4:2的左右孩子都是null
  - i=5,6:3的左孩子4,右孩子5,队列=[4,5]
  - i=7,8,9,10:4和5的左右孩子都是null
- 还原成功!

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "为什么用层序而不是前序?" | "层序更直观,按层处理符合思维习惯;前序也可以,字符串会更短,但递归逻辑稍复杂" |
| "如果要求字符串尽可能短?" | "可以用前序DFS,减少末尾的null;或者用后序遍历去掉尾部null,但反序列化会更复杂" |
| "能不能用中序遍历?" | "不能!中序无法唯一确定树的结构,比如[1,null,2]和[2,null,1]的中序都是[null,1,null,2],但结构不同" |
| "空间能优化吗?" | "序列化必须O(n)存储结果;反序列化的队列是必需的,无法优化,整体空间O(n)是最优的" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:用iter()和next()优雅地遍历
values = iter(data.split(","))
val = next(values)  # 自动移动到下一个,无需维护索引

# 技巧2:deque的popleft()是O(1)
from collections import deque
queue = deque([1, 2, 3])
queue.popleft()  # O(1),比list.pop(0)的O(n)快

# 技巧3:join连接字符串比+高效
result = []
for val in values:
    result.append(str(val))
return ",".join(result)  # O(n),比逐个+拼接的O(n²)快
```

### 💡 底层原理(选读)

**为什么中序遍历不能用于序列化?**

反例:
```
树1:  1          树2:    2
       \                /
        2              1

中序遍历:1,2         中序遍历:1,2  (相同!)
```

两棵不同的树,中序遍历结果相同,无法区分。

**为什么前序和层序可以?**

前序遍历:[根,左,右],记录null后,可以唯一确定树:
- 树1前序:1,null,2,null,null
- 树2前序:2,1,null,null,null (不同!)

层序遍历:按层记录,包括null,也能唯一确定。

**结论**:序列化需要的是能唯一确定树结构的遍历方式,前序、后序、层序都可以,但中序不行。

### 算法模式卡片 📐
- **模式名称**:树的序列化与反序列化
- **适用条件**:
  - 需要保存树的完整结构信息
  - 需要在不同系统间传输树数据
  - 需要持久化存储树
- **识别关键词**:
  - "序列化"
  - "持久化"
  - "深拷贝树"
  - "树的存储和恢复"
- **模板代码**:
```python
# 层序BFS模板
from collections import deque

class Codec:
    def serialize(self, root):
        if not root:
            return "null"
        result, queue = [], deque([root])
        while queue:
            node = queue.popleft()
            if node:
                result.append(str(node.val))
                queue.extend([node.left, node.right])
            else:
                result.append("null")
        return ",".join(result)

    def deserialize(self, data):
        if data == "null":
            return None
        values = data.split(",")
        root = TreeNode(int(values[0]))
        queue, i = deque([root]), 1
        while queue:
            node = queue.popleft()
            if values[i] != "null":
                node.left = TreeNode(int(values[i]))
                queue.append(node.left)
            i += 1
            if values[i] != "null":
                node.right = TreeNode(int(values[i]))
                queue.append(node.right)
            i += 1
        return root
```

### 易错点 ⚠️
1. **序列化时忘记处理null**
   - 错误:只存储非空节点的值
   - 问题:反序列化时无法确定树的结构
   - 正确:null节点也要用占位符(如"null")表示

2. **反序列化时索引越界**
   - 错误:忘记检查 `i < len(values)`
   - 问题:如果字符串格式不正确会导致IndexError
   - 正确:在访问values[i]前先判断 `if i < len(values)`

3. **用中序遍历序列化**
   - 错误:以为任何遍历方式都可以
   - 问题:中序无法唯一确定树结构
   - 正确:只能用前序、后序或层序

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1:Redis持久化**
  - 问题:Redis的数据结构(如Sorted Set底层的跳表)需要持久化到磁盘
  - 应用:用类似的序列化方法,将内存中的树状结构转换为字节流存储

- **场景2:网络传输**
  - 问题:微服务间需要传输复杂的层级数据(如组织架构树)
  - 应用:序列化为JSON字符串传输,接收方反序列化恢复

- **场景3:深拷贝**
  - 问题:需要完整复制一棵树,包括所有节点
  - 应用:序列化→反序列化,自动实现深拷贝

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 428. 序列化N叉树 | Hard | N叉树序列化 | 用层序BFS,每个节点记录子节点数量 |
| LeetCode 449. 序列化BST | Medium | BST序列化优化 | 利用BST性质,可以省略null,用前序+范围判断反序列化 |
| LeetCode 652. 寻找重复子树 | Medium | 树哈希 | 用序列化的字符串作为树的哈希值,找重复 |
| LeetCode 331. 验证序列化 | Medium | 序列化验证 | 不需要真的建树,用栈模拟验证格式 |
| LeetCode 1008. 前序遍历构造BST | Medium | 前序+BST | 前序遍历+范围约束反序列化BST |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定一棵二叉搜索树(BST),设计序列化和反序列化算法。由于BST有特殊性质(左<根<右),能否设计一个更紧凑的序列化方案(不需要存储null)?

示例:
```
输入:root = [2,1,3]
     2
    / \
   1   3

你的序列化:"2,1,3" (没有null!)
能否仅用这3个数字反序列化还原?
```

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

BST的前序遍历是唯一的!利用BST的性质:在前序遍历中,对于每个节点,左子树的所有值都小于它,右子树的所有值都大于它。反序列化时用范围约束递归构建。

</details>

<details>
<summary>✅ 参考答案</summary>

```python
class CodecBST:
    """BST的紧凑序列化(无需null占位符)"""

    def serialize(self, root: TreeNode) -> str:
        """前序遍历,不需要存储null"""
        result = []

        def preorder(node):
            if node:
                result.append(str(node.val))
                preorder(node.left)
                preorder(node.right)

        preorder(root)
        return ",".join(result)

    def deserialize(self, data: str) -> TreeNode:
        """用范围约束反序列化BST"""
        if not data:
            return None

        values = iter(map(int, data.split(",")))

        def build(lower, upper):
            """构建值在(lower, upper)范围内的子树"""
            val = next(values, None)
            if val is None or val < lower or val > upper:
                return None

            node = TreeNode(val)
            node.left = build(lower, val)    # 左子树值 < val
            node.right = build(val, upper)   # 右子树值 > val
            return node

        return build(float('-inf'), float('inf'))


# 测试
codec = CodecBST()
root = TreeNode(2, TreeNode(1), TreeNode(3))
serialized = codec.serialize(root)
print(f"BST序列化:{serialized}")  # "2,1,3" (只有3个数!)

deserialized = codec.deserialize(serialized)
print(f"验证:{codec.serialize(deserialized)}")  # "2,1,3"
```

**核心思路**:
- BST的前序遍历 + 范围约束可以唯一确定树的结构,无需null占位符
- 序列化:直接前序遍历,只记录值
- 反序列化:用递归+范围约束(lower, upper),如果当前值不在范围内,说明不属于这个子树,回退

**优势**:字符串长度从O(2n-1)(包含null)减少到O(n)(纯值)

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
