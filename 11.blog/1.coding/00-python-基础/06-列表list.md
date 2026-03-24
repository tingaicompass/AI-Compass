> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 06 - 列表list:Python中最常用的数据结构

> **学习目标**: 掌握列表的创建、访问、修改和常用方法

---

## 📖 知识点讲解

### 什么是列表?

列表(list)就是Python中的**数组**,可以存储一系列元素。

**生活化比喻**:
> 列表就像一个**有序的收纳盒**,每个格子里可以放东西,盒子会自动编号(0, 1, 2...)。你可以:
> - 往盒子里加东西(`append`)
> - 从盒子里拿出东西(`pop`)
> - 查看某个格子里的东西(`nums[0]`)
> - 给盒子排序(`sort`)

### 列表 vs 其他语言的数组

| 特性 | Python列表 | Java数组 |
|------|-----------|---------|
| 长度固定? | ❌ 动态扩展 | ✅ 固定长度 |
| 类型一致? | ❌ 可混合类型 | ✅ 必须同类型 |
| 创建方式 | `[1, 2, 3]` | `new int[]{1,2,3}` |

---

## 💻 代码示例

### 示例1:创建列表

```python
# 方法1:直接用方括号 []
nums = [1, 2, 3, 4, 5]
print(nums)  # [1, 2, 3, 4, 5]

# 方法2:空列表
empty_list = []

# 方法3:list()函数
another_list = list()  # []

# 方法4:初始化固定大小的列表(常用于算法题)
zeros = [0] * 5  # [0, 0, 0, 0, 0]
matrix = [[0] * 3 for _ in range(2)]  # [[0,0,0], [0,0,0]]

# 方法5:混合类型(Python特有)
mixed = [1, "hello", True, 3.14]  # 可以混合不同类型
```

---

### 示例2:访问元素(索引和切片)

```python
nums = [10, 20, 30, 40, 50]

# 正向索引(从0开始)
print(nums[0])   # 10 (第一个元素)
print(nums[2])   # 30 (第三个元素)

# 负向索引(从-1开始)
print(nums[-1])  # 50 (最后一个元素)
print(nums[-2])  # 40 (倒数第二个)

# 切片[start:end](不包含end)
print(nums[1:3])   # [20, 30] (索引1到2)
print(nums[:3])    # [10, 20, 30] (开头到索引2)
print(nums[2:])    # [30, 40, 50] (索引2到结尾)
print(nums[:])     # [10, 20, 30, 40, 50] (复制整个列表)

# 切片[start:end:step](带步长)
print(nums[::2])   # [10, 30, 50] (每隔一个取一个)
print(nums[::-1])  # [50, 40, 30, 20, 10] (反转列表,超常用!)
```

---

### 示例3:修改列表

```python
nums = [1, 2, 3]

# 修改单个元素
nums[0] = 10
print(nums)  # [10, 2, 3]

# 添加元素到末尾
nums.append(4)
print(nums)  # [10, 2, 3, 4]

# 在指定位置插入
nums.insert(1, 15)  # 在索引1插入15
print(nums)  # [10, 15, 2, 3, 4]

# 删除末尾元素(返回被删除的值)
last = nums.pop()
print(last)  # 4
print(nums)  # [10, 15, 2, 3]

# 删除指定索引的元素
nums.pop(1)  # 删除索引1的元素(15)
print(nums)  # [10, 2, 3]

# 删除指定值的第一个出现
nums.remove(2)  # 删除值为2的元素
print(nums)  # [10, 3]

# 清空列表
nums.clear()
print(nums)  # []
```

---

### 示例4:列表常用方法

```python
nums = [3, 1, 4, 1, 5, 9, 2, 6]

# 长度
print(len(nums))  # 8

# 排序(原地排序,修改原列表)
nums.sort()
print(nums)  # [1, 1, 2, 3, 4, 5, 6, 9]

# 反转
nums.reverse()
print(nums)  # [9, 6, 5, 4, 3, 2, 1, 1]

# 获取最大/最小值
print(max(nums))  # 9
print(min(nums))  # 1

# 求和
print(sum(nums))  # 31

# 计数(某个值出现的次数)
print(nums.count(1))  # 2

# 查找某个值的索引(第一次出现)
print(nums.index(5))  # 2
```

---

### 示例5:列表作为栈(LIFO - 后进先出)

```python
# 栈:先进后出,像一摞盘子
stack = []

# 入栈(push)
stack.append(1)
stack.append(2)
stack.append(3)
print(stack)  # [1, 2, 3]

# 出栈(pop)
top = stack.pop()  # 3
print(top)
print(stack)  # [1, 2]

# 查看栈顶(不删除)
if stack:
    print(stack[-1])  # 2
```

---

### 示例6:列表推导式(List Comprehension)

```python
# 传统方式:创建平方数列表
squares = []
for i in range(5):
    squares.append(i ** 2)
print(squares)  # [0, 1, 4, 9, 16]

# 列表推导式:一行搞定!
squares = [i ** 2 for i in range(5)]
print(squares)  # [0, 1, 4, 9, 16]

# 带条件的列表推导式
evens = [i for i in range(10) if i % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# 嵌套列表推导式
matrix = [[i * j for j in range(3)] for i in range(3)]
print(matrix)  # [[0, 0, 0], [0, 1, 2], [0, 2, 4]]
```

---

## 🎯 在算法题中的应用

列表是**最最常用**的数据结构,107道题几乎全部用到!

### 应用场景1:存储结果

**来自第1课:两数之和**
```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]  # ← 返回列表
        seen[num] = i
    return []  # ← 返回空列表
```

### 应用场景2:作为栈使用

**来自第33课:有效的括号**
```python
def isValid(s):
    stack = []  # ← 列表作为栈
    pairs = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in pairs:  # 右括号
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()  # ← 出栈
        else:  # 左括号
            stack.append(char)  # ← 入栈

    return len(stack) == 0
```

### 应用场景3:双指针操作

**来自第7课:移动零**
```python
def moveZeroes(nums):
    slow = 0  # 慢指针
    for fast in range(len(nums)):  # 快指针
        if nums[fast] != 0:
            nums[slow], nums[fast] = nums[fast], nums[slow]  # ← 交换
            slow += 1
```

### 应用场景4:初始化DP数组

**来自第71课:爬楼梯**
```python
def climbStairs(n):
    if n <= 2:
        return n

    dp = [0] * (n + 1)  # ← 初始化列表
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]
```

### 应用场景5:排序后双指针

**来自第9课:三数之和**
```python
def threeSum(nums):
    nums.sort()  # ← 列表排序
    result = []

    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total == 0:
                result.append([nums[i], nums[left], nums[right]])  # ← 添加结果
                left += 1
                right -= 1
            elif total < 0:
                left += 1
            else:
                right -= 1

    return result
```

---

## 🏋️ 快速练习

### 练习1:反转列表

有一个列表`[1, 2, 3, 4, 5]`,用**两种方法**反转它:

<details>
<summary>点击查看答案</summary>

```python
nums = [1, 2, 3, 4, 5]

# 方法1:切片(创建新列表)
reversed_nums = nums[::-1]
print(reversed_nums)  # [5, 4, 3, 2, 1]

# 方法2:reverse()方法(原地反转)
nums.reverse()
print(nums)  # [5, 4, 3, 2, 1]

# 方法3:双指针手动反转(算法题常用)
left, right = 0, len(nums) - 1
while left < right:
    nums[left], nums[right] = nums[right], nums[left]
    left += 1
    right -= 1
```

</details>

---

### 练习2:找出列表中的最大值和它的索引

```python
nums = [3, 7, 2, 9, 1]
# 找出最大值9和它的索引3
```

<details>
<summary>点击查看答案</summary>

```python
nums = [3, 7, 2, 9, 1]

# 方法1:使用内置函数
max_val = max(nums)
max_idx = nums.index(max_val)
print(f"最大值: {max_val}, 索引: {max_idx}")  # 最大值: 9, 索引: 3

# 方法2:手动遍历(算法题常用思路)
max_val = nums[0]
max_idx = 0
for i in range(1, len(nums)):
    if nums[i] > max_val:
        max_val = nums[i]
        max_idx = i
print(f"最大值: {max_val}, 索引: {max_idx}")
```

</details>

---

### 练习3:删除列表中的重复元素

给定`[1, 2, 2, 3, 3, 3, 4]`,返回`[1, 2, 3, 4]`

<details>
<summary>点击查看答案</summary>

```python
nums = [1, 2, 2, 3, 3, 3, 4]

# 方法1:转成set再转回list(无序)
unique = list(set(nums))
print(unique)  # [1, 2, 3, 4] (可能顺序不同)

# 方法2:保持顺序(算法题常用)
result = []
for num in nums:
    if num not in result:
        result.append(num)
print(result)  # [1, 2, 3, 4]

# 方法3:使用dict.fromkeys(保持顺序,Python 3.7+)
unique = list(dict.fromkeys(nums))
print(unique)  # [1, 2, 3, 4]
```

</details>

---

## 🎓 小结

✅ **创建列表**: `[]`, `list()`, `[0] * n`
✅ **访问元素**: 正向索引`[0]`, 负向索引`[-1]`, 切片`[1:3]`
✅ **修改列表**: `append`, `pop`, `insert`, `remove`
✅ **常用方法**: `sort`, `reverse`, `len`, `max`, `min`, `sum`
✅ **作为栈**: `append`入栈, `pop`出栈
✅ **列表推导式**: `[x for x in ...]`

**最重要的操作**(必须记住):
- `nums[i]` - 访问元素
- `nums.append(x)` - 添加元素
- `nums.pop()` - 删除并返回最后一个元素
- `nums[::-1]` - 反转列表
- `len(nums)` - 列表长度

**下一步**: [07-字典dict.md](./07-字典dict.md)

---

*列表是Python的基石,熟练掌握它就成功了一半!* 🚀

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
