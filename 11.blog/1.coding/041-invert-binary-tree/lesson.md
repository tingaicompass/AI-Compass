# 📖 第41课:翻转二叉树

> **模块**:二叉树 | **难度**:Easy ⭐⭐⭐
> **LeetCode 链接**:https://leetcode.cn/problems/invert-binary-tree/
> **前置知识**:第40课(二叉树最大深度)
> **预计学习时间**:15分钟

---

## 🎯 题目描述

给定一个二叉树的根节点,将该树进行镜像翻转。也就是说,交换每个节点的左右子树。

**示例:**
```
原始树:        翻转后:
    4            4
   / \          / \
  2   7        7   2
 / \ / \      / \ / \
1  3 6  9    9  6 3  1

输入:root = [4,2,7,1,3,6,9]
输出:[4,7,2,9,6,3,1]
解释:每个节点的左右子树都进行了交换
```

**约束条件:**
- 树中节点数量范围是 [0, 100]
- -100 <= Node.val <= 100

---

### 🧪 边界用例(面试必考)

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 空树 | root=None | None | 递归出口 |
| 单节点 | root=[1] | [1] | 基本功能 |
| 只有左子树 | root=[1,2] | [1,null,2] | 单侧翻转 |
| 只有右子树 | root=[1,null,2] | [1,2] | 单侧翻转 |
| 完全二叉树 | root=[1,2,3,4,5,6,7] | [1,3,2,7,6,5,4] | 对称翻转 |
| 已翻转的树 | 翻转后再翻转 | 原树 | 幂等性 |

---

## 💡 思路引导

### 生活化比喻
> 想象你在整理一面照片墙,要把所有照片左右镜像翻转。
>
> 🐌 **笨办法**:先用相机把整面墙拍下来,再用Photoshop镜像翻转,最后重新打印粘贴。这需要额外空间存储副本。
>
> 🚀 **聪明办法**:从最顶上的照片开始,交换它左右两边的照片。然后递归地对左边区域和右边区域做同样的操作。就像递归地解决子问题,**原地交换**,不需要额外副本。

### 关键洞察
**翻转一棵树 = 交换根的左右子树 + 递归翻转左子树 + 递归翻转右子树**

---

## 🧠 解题思维链

> 这一节模拟你在面试中"从零开始思考"的过程。

### Step 1:理解题目 → 锁定输入输出
- **输入**:二叉树根节点 TreeNode
- **输出**:翻转后的树的根节点(原地修改)
- **限制**:节点数≤100,规模小,递归深度安全

### Step 2:先想笨办法(新建树)
遍历原树,对每个节点创建新节点,新节点的左子树指向原节点的右子树副本,右子树指向左子树副本。
- 时间复杂度:O(n)
- 瓶颈在哪:需要O(n)额外空间创建副本

### Step 3:瓶颈分析 → 优化方向
其实不需要创建新树,可以原地修改!
- 核心问题:"如何确保交换后不影响后续遍历?"
- 优化思路:用递归的后序遍历,先处理子树再交换当前节点

### Step 4:选择武器
- 选用:**DFS递归(后序遍历)**
- 理由:递归天然适合树的变换,后序遍历确保自底向上安全交换

> 🔑 **模式识别提示**:当题目涉及"树的结构变换",优先考虑"递归修改"

---

## 🏆 解法一:DFS递归(最优解)

### 思路
递归定义:
1. 空树翻转后仍是空树(递归出口)
2. 非空树:先递归翻转左右子树,再交换它们

这是**后序遍历**:先处理子树,再处理当前节点。

### 图解过程

```
示例:
原树:     4              Step1: 递归到叶子
        / \
       2   7
      / \ / \
     1  3 6  9

递归调用顺序(后序):
invertTree(4)
├─ invertTree(2)
│  ├─ invertTree(1) → 叶子,返回1
│  └─ invertTree(3) → 叶子,返回3
│  交换2的左右子树: 2的左=3, 右=1
│  返回2
│
└─ invertTree(7)
   ├─ invertTree(6) → 叶子,返回6
   └─ invertTree(9) → 叶子,返回9
   交换7的左右子树: 7的左=9, 右=6
   返回7

交换4的左右子树: 4的左=7, 右=2
返回4

最终结果:
    4
   / \
  7   2
 / \ / \
9  6 3  1  ✅
```

### Python代码

```python
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    解法一:DFS递归(最优解)
    思路:后序遍历,递归翻转左右子树后交换
    """
    # 递归出口:空节点无需翻转
    if not root:
        return None

    # 递归翻转左右子树
    left = invertTree(root.left)
    right = invertTree(root.right)

    # 交换左右子树(后序位置)
    root.left = right
    root.right = left

    return root


# ✅ 测试
# 构建示例树:    4
#              / \
#             2   7
#            / \ / \
#           1  3 6  9
root1 = TreeNode(4)
root1.left = TreeNode(2, TreeNode(1), TreeNode(3))
root1.right = TreeNode(7, TreeNode(6), TreeNode(9))

result = invertTree(root1)
# 验证:中序遍历原树[1,2,3,4,6,7,9] 翻转后应该是[9,7,6,4,3,2,1]
def inorder(node):
    return inorder(node.left) + [node.val] + inorder(node.right) if node else []
print(inorder(result))  # 期望:[9,7,6,4,3,2,1]

# 边界测试
print(invertTree(None))  # 期望:None
single = TreeNode(1)
invertTree(single)
print(single.val)  # 期望:1(单节点不变)
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点访问恰好一次
  - 具体地说:100个节点需要100次递归调用
- **空间复杂度**:O(h) — 递归栈深度等于树高h
  - 平衡树:O(log n),如100个节点约7层
  - 最坏链状树:O(n),但题目限制n≤100,栈深度安全

### 优缺点
- ✅ 代码极简,仅5行
- ✅ 原地修改,不需额外空间
- ✅ 时间O(n)最优

---

## ⚡ 解法二:BFS层序遍历(迭代法)

### 优化思路
递归虽简洁,但有些面试官要求"不用递归"。改用队列的BFS,逐层交换每个节点的左右子树。

> 💡 **关键想法**:用队列遍历每个节点,遇到就交换其左右孩子

### 图解过程

```
示例:    4
        / \
       2   7
      / \ / \
     1  3 6  9

BFS过程:
初始: queue=[4]

Step1: 出队4, 交换4的左右
       4.left=7, 4.right=2
       入队7,2
       queue=[7,2]

Step2: 出队7, 交换7的左右
       7.left=9, 7.right=6
       入队9,6
       queue=[2,9,6]

Step3: 出队2, 交换2的左右
       2.left=3, 2.right=1
       入队3,1
       queue=[9,6,3,1]

Step4: 出队9,6,3,1都是叶子,无需交换

最终:   4
       / \
      7   2
     / \ / \
    9  6 3  1  ✅
```

### Python代码

```python
from collections import deque


def invertTree_bfs(root: Optional[TreeNode]) -> Optional[TreeNode]:
    """
    解法二:BFS迭代
    思路:队列层序遍历,逐个交换节点的左右子树
    """
    if not root:
        return None

    queue = deque([root])

    while queue:
        node = queue.popleft()

        # 交换当前节点的左右子树
        node.left, node.right = node.right, node.left

        # 将非空子节点加入队列
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return root


# ✅ 测试
root2 = TreeNode(4)
root2.left = TreeNode(2, TreeNode(1), TreeNode(3))
root2.right = TreeNode(7, TreeNode(6), TreeNode(9))
result2 = invertTree_bfs(root2)
print(inorder(result2))  # 期望:[9,7,6,4,3,2,1]
```

### 复杂度分析
- **时间复杂度**:O(n) — 每个节点入队出队各一次
- **空间复杂度**:O(w) — w为树的最大宽度,完全树最坏O(n/2)=O(n)

---

## 🐍 Pythonic 写法

利用Python的元组解包实现优雅的交换:

```python
# 精简版(推荐)
def invertTree_pythonic(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None
    # Python的多重赋值,一行完成交换和递归
    root.left, root.right = invertTree_pythonic(root.right), invertTree_pythonic(root.left)
    return root

# 一行lambda版(不推荐,可读性差)
invertTree_oneline = lambda r: (setattr(r, 'left', invertTree_oneline(r.right)) or
                                  setattr(r, 'right', invertTree_oneline(r.left)) or r) if r else None
```

> ⚠️ **面试建议**:先写清晰版本展示思路,通过后可以说"Python可以用元组解包简化交换",但不要一开始就写一行版(面试官可能觉得你在炫技)。

---

## 📊 解法对比

| 维度 | 🏆 解法一:DFS递归 | 解法二:BFS迭代 |
|------|-----------------|--------------|
| 时间复杂度 | **O(n)** ← 最优 | O(n) |
| 空间复杂度 | **O(h) 约O(log n)** | O(w) 约O(n) |
| 代码难度 | **简单(5行)** | 中等 |
| 面试推荐 | **⭐⭐⭐** ← 首选 | ⭐⭐ |
| 适用场景 | **通用,最简洁** | 避免递归要求 |
| 原地修改 | ✅ | ✅ |

**为什么解法一是最优解**:
- 时间O(n)已是理论最优(必须访问所有节点)
- 空间O(log n)优于BFS的O(n)
- 代码仅5行,最简洁直观
- 完美展示递归思维

**面试建议**:
1. 直接说出🏆最优解思路:"递归翻转左右子树,然后交换"
2. 写代码时强调后序遍历顺序:"先处理子树,再交换当前节点"
3. 手动模拟一个3层树的翻转过程
4. 主动测试边界:"空树返回None,单节点返回自己"
5. 如被问"能否迭代?",给出解法二BFS方案

---

## 🎤 面试现场

> 模拟面试中的完整对话流程,帮你练习"边想边说"。

**面试官**:请你翻转一棵二叉树。

**你**:(审题10秒)好的,这道题要求镜像翻转,也就是交换每个节点的左右子树。

我的思路是用**递归**。对于每个节点,先递归翻转它的左右子树,然后交换这两个子树。这是后序遍历的思想,时间复杂度O(n),空间复杂度O(h)。

**面试官**:很好,请写代码。

**你**:(边写边说)
```python
def invertTree(root):
    # 递归出口:空节点无需翻转
    if not root:
        return None
    # 先递归翻转左右子树
    left = invertTree(root.left)
    right = invertTree(root.right)
    # 交换左右子树
    root.left = right
    root.right = left
    return root
```

**面试官**:为什么要先递归再交换?

**你**:这是后序遍历的思想。如果先交换再递归,会出现混乱:交换后root.left指向的是原来的右子树,递归root.left就会去处理原右子树,逻辑就乱了。而后序遍历先处理完子树,拿到翻转后的结果,再交换,逻辑清晰。

**面试官**:测试一下?

**你**:用示例树[4,2,7,1,3,6,9]...
- 最底层:节点1,3,6,9都是叶子,翻转后不变
- 节点2:翻转后左孩子变成3,右孩子变成1
- 节点7:翻转后左孩子变成9,右孩子变成6
- 根节点4:翻转后左孩子变成7,右孩子变成2
- 最终结果[4,7,2,9,6,3,1] ✅

边界:空树返回None,代码第2行处理了 ✅

**面试官**:不错!能不用递归吗?

**你**:可以用BFS。用队列遍历每个节点,逐个交换其左右孩子。时间仍是O(n),但空间可能O(n),代码也更长,所以递归是首选。

### 高频追问

| 追问 | 应答策略 |
|------|---------|
| "交换能一行写完吗?" | "可以:`root.left, root.right = invertTree(root.right), invertTree(root.left)`,Python的元组解包特性" |
| "为什么不用前序遍历?" | "前序也可以!先交换再递归,只要注意交换后root.left指向的是原右子树即可。后序更直观" |
| "翻转两次会怎样?" | "会恢复原样,因为镜像的镜像就是原图,这个操作是幂等的" |
| "如果树很大会栈溢出吗?" | "题目限制n≤100,栈深度最多100,Python默认约1000,安全。实际项目中可用BFS迭代避免" |

---

## 🎓 知识点总结

### Python技巧卡片 🐍
```python
# 技巧1:元组解包实现优雅交换 — 不需临时变量
a, b = b, a  # 交换a和b
root.left, root.right = root.right, root.left

# 技巧2:递归返回值链式调用
root.left = invertTree(root.right)  # 一边递归一边赋值

# 技巧3:原地修改vs返回新对象
# 原地修改:适合树的结构变换,节省空间
# 返回新对象:适合需要保留原数据
```

### 💡 底层原理(选读)

> **为什么后序遍历适合树的变换?**
>
> 三种遍历顺序的处理时机:
> - **前序**(根→左→右):先处理当前节点,再递归子树 → 适合**自顶向下传递信息**
> - **中序**(左→根→右):先处理左子树,再当前,再右 → 适合**BST有序输出**
> - **后序**(左→右→根):先处理子树,最后当前 → 适合**自底向上汇总**,如计算深度、翻转
>
> 翻转需要先得到子树的翻转结果,再交换,符合后序的"先子后父"特性。

> **Python的元组解包为何能避免临时变量?**
>
> ```python
> a, b = b, a
> ```
> 底层实现:
> 1. 先计算右边表达式,构造元组`(b的值, a的值)`
> 2. 再解包赋值给左边的a和b
>
> 相当于:
> ```python
> temp = (b, a)  # 元组打包
> a = temp[0]    # 解包
> b = temp[1]
> ```
> 但元组是在栈上构造的,比显式临时变量更高效。

### 算法模式卡片 📐
- **模式名称**:树的后序遍历变换
- **适用条件**:需要基于子树结果修改当前节点
- **识别关键词**:"翻转"、"变换"、"基于子树计算"
- **模板代码**:
```python
def transform(root):
    if not root:
        return None

    # 先递归处理子树
    left_result = transform(root.left)
    right_result = transform(root.right)

    # 基于子树结果修改当前节点(后序位置)
    root.left = modify(right_result)
    root.right = modify(left_result)

    return root
```

### 易错点 ⚠️
1. **交换时机错误**
   - ❌ 错误:先交换再递归 `root.left, root.right = root.right, root.left; invertTree(root.left)` 会导致递归的是原右子树
   - ✅ 正确:先递归保存结果,再交换

2. **忘记返回root**
   - ❌ 错误:只交换不返回,调用方拿不到修改后的树
   - ✅ 正确:函数末尾`return root`

3. **误以为需要新建树**
   - ❌ 错误:创建新节点复制,浪费空间
   - ✅ 正确:原地修改指针即可

---

## 🏗️ 工程实战(选读)

> 这个算法思想在真实项目中的应用,让你知道"学了有什么用"。

- **场景1**:图像镜像翻转
  ```python
  # OpenCV的cv2.flip(img, 1)水平翻转
  # 底层就是树形像素块的递归翻转优化
  ```

- **场景2**:UI布局镜像(RTL语言)
  ```python
  # 阿拉伯语/希伯来语从右往左书写
  # 需要镜像翻转整个UI组件树
  # React Native的I18nManager.forceRTL()内部就用类似递归
  ```

- **场景3**:DNA序列反向互补
  ```python
  # 生物信息学:DNA双螺旋的互补配对
  # ATCG → TAGC(反向+互补)
  # 可以看作树形分子结构的镜像变换
  ```

---

## 🏋️ 举一反三

完成本课后,试试这些同类题目来巩固知识:

| 题目 | 难度 | 相关知识点 | 提示 |
|------|------|-----------|------|
| LeetCode 101. 对称二叉树 | Easy | DFS递归 | 本题的应用:判断树是否等于自己的镜像 |
| LeetCode 951. 翻转等价二叉树 | Medium | DFS递归 | 允许交换任意节点的子树,判断能否相等 |
| LeetCode 971. 翻转二叉树以匹配 | Medium | DFS+贪心 | 给定前序遍历,最少翻转次数 |
| LeetCode 617. 合并二叉树 | Easy | DFS递归 | 类似思想:递归处理两棵树的对应节点 |

---

## 📝 课后小测

试试这道变体题,不要看答案,自己先想5分钟!

**题目**:给定两个二叉树root1和root2,判断root1翻转后是否等于root2。

<details>
<summary>💡 提示(实在想不出来再点开)</summary>

两种思路:
1. 直接翻转root1,然后判断两树是否相同
2. 递归判断:root1的左子树翻转后等于root2的右子树,且root1的右子树翻转后等于root2的左子树

</details>

<details>
<summary>✅ 参考答案</summary>

```python
def isFlipEquivalent(root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
    """
    判断两树是否翻转等价
    思路:递归判断翻转后的对应关系
    """
    # 都为空,等价
    if not root1 and not root2:
        return True
    # 一个空一个非空,不等价
    if not root1 or not root2:
        return False
    # 值不同,不等价
    if root1.val != root2.val:
        return False

    # 两种情况:
    # 1. 不翻转:左对左,右对右
    no_flip = (isFlipEquivalent(root1.left, root2.left) and
               isFlipEquivalent(root1.right, root2.right))

    # 2. 翻转:左对右,右对左
    flip = (isFlipEquivalent(root1.left, root2.right) and
            isFlipEquivalent(root1.right, root2.left))

    return no_flip or flip
```

**核心思路**:
- 每个节点有两种选择:翻转或不翻转
- 递归判断两种情况是否有一种成立
- 时间O(n),空间O(h)

</details>

---

## 💬 趣闻

这道题因为一个真实故事而出名:**Homebrew的作者Max Howell在Google面试时被这道题难住了,最终没通过**。他在Twitter上吐槽:"Google: 90% of our engineers use the software you wrote (Homebrew), but you can't invert a binary tree on a whiteboard so fuck off."

这个故事引发了业界大讨论:算法题到底能不能衡量工程能力?但无论如何,这道题成了最著名的面试题之一。

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
