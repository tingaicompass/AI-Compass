# 13 - collections模块:高级数据结构

> **学习目标**: 掌握Counter, defaultdict, deque

---

## 💻 代码示例

### 1. Counter - 计数器

```python
from collections import Counter

# 统计元素出现次数
nums = [1, 2, 2, 3, 3, 3]
count = Counter(nums)
print(count)  # Counter({3: 3, 2: 2, 1: 1})

# 统计字符
s = "hello"
char_count = Counter(s)
print(char_count)  # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})

# 访问计数
print(count[2])  # 2
print(count[10])  # 0 (不存在返回0,不报错)

# 最常见的元素
print(char_count.most_common(2))  # [('l', 2), ('h', 1)]

# Counter运算
c1 = Counter("hello")
c2 = Counter("world")
print(c1 + c2)  # 相加
print(c1 - c2)  # 相减
```

### 2. defaultdict - 默认值字典

```python
from collections import defaultdict

# 普通字典
d = {}
# d["key"].append(1)  # ❌ KeyError

# defaultdict自动创建默认值
d = defaultdict(list)  # 默认值是list
d["key"].append(1)
d["key"].append(2)
print(d)  # {'key': [1, 2]}

# 默认值是int(用于计数)
count = defaultdict(int)
for char in "hello":
    count[char] += 1  # 不存在的键默认为0
print(dict(count))  # {'h': 1, 'e': 1, 'l': 2, 'o': 1}

# 默认值是set
groups = defaultdict(set)
groups["A"].add(1)
groups["A"].add(2)
print(dict(groups))  # {'A': {1, 2}}
```

### 3. deque - 双端队列

```python
from collections import deque

# 创建队列
q = deque([1, 2, 3])

# 右端操作(O(1))
q.append(4)       # [1, 2, 3, 4]
print(q.pop())    # 4, 队列变成[1, 2, 3]

# 左端操作(O(1))
q.appendleft(0)   # [0, 1, 2, 3]
print(q.popleft())  # 0, 队列变成[1, 2, 3]

# 作为队列(FIFO)
queue = deque()
queue.append(1)  # 入队
queue.append(2)
print(queue.popleft())  # 1 出队

# 作为栈(LIFO)
stack = deque()
stack.append(1)  # 入栈
stack.append(2)
print(stack.pop())  # 2 出栈

# 限制长度
recent = deque(maxlen=3)
for i in range(5):
    recent.append(i)
print(recent)  # deque([2, 3, 4])
```

---

## 🎯 在算法题中的应用

```python
# Counter: 字母异位词
from collections import Counter

def isAnagram(s, t):
    return Counter(s) == Counter(t)

# defaultdict: 分组
from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for word in strs:
        key = "".join(sorted(word))
        groups[key].append(word)
    return list(groups.values())

# deque: BFS层序遍历
from collections import deque

def levelOrder(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result

# deque: 滑动窗口最大值(单调队列)
def maxSlidingWindow(nums, k):
    from collections import deque
    dq = deque()
    result = []

    for i, num in enumerate(nums):
        # 移除队首过期元素
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 移除队尾小于当前元素的
        while dq and nums[dq[-1]] < num:
            dq.pop()

        dq.append(i)

        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

---

## 🎓 小结

✅ **Counter**: 统计元素出现次数
✅ **defaultdict**: 自动创建默认值,避免KeyError
✅ **deque**: 双端队列,两端操作都是O(1)

**使用场景**:
- Counter: 计数、频率统计
- defaultdict: 分组、建图
- deque: BFS、滑动窗口、单调队列

**下一步**: [14-heapq模块.md](./14-heapq模块.md)

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
