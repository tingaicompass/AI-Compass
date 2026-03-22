# 14 - heapq模块:堆操作

> **学习目标**: 掌握Python的堆(优先队列)操作

---

## 💻 代码示例

```python
import heapq

# Python的堆是最小堆
nums = [3, 1, 4, 1, 5, 9, 2, 6]

# 将列表转换为堆
heapq.heapify(nums)
print(nums)  # [1, 1, 2, 3, 5, 9, 4, 6]

# 弹出最小元素
min_val = heapq.heappop(nums)
print(min_val)  # 1

# 添加元素
heapq.heappush(nums, 0)
print(heapq.heappop(nums))  # 0

# 替换堆顶
heapq.heapreplace(nums, 10)  # 弹出最小值,然后添加10

# 获取最大/最小的K个元素
nums = [3, 1, 4, 1, 5, 9, 2, 6]
print(heapq.nlargest(3, nums))   # [9, 6, 5]
print(heapq.nsmallest(3, nums))  # [1, 1, 2]

# 最大堆(取负数)
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -8)
print(-heapq.heappop(max_heap))  # 8 (最大值)
```

---

## 🎯 在算法题中的应用

```python
# 第97课:前K个高频元素
import heapq
from collections import Counter

def topKFrequent(nums, k):
    count = Counter(nums)
    return heapq.nlargest(k, count.keys(), key=count.get)

# 第98课:数据流中位数(对顶堆)
class MedianFinder:
    def __init__(self):
        self.small = []  # 最大堆(存较小的一半)
        self.large = []  # 最小堆(存较大的一半)

    def addNum(self, num):
        if len(self.small) == len(self.large):
            heapq.heappush(self.large, -heapq.heappushpop(self.small, -num))
        else:
            heapq.heappush(self.small, -heapq.heappushpop(self.large, num))

    def findMedian(self):
        if len(self.small) == len(self.large):
            return (self.large[0] - self.small[0]) / 2
        else:
            return self.large[0]

# 第99课:合并K个升序链表
def mergeKLists(lists):
    import heapq
    heap = []

    # 初始化堆
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    current = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        current.next = node
        current = current.next

        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

---

## 🎓 小结

✅ Python的堆是**最小堆**
✅ `heapify(list)` 将列表转换为堆
✅ `heappush(heap, item)` 添加元素
✅ `heappop(heap)` 弹出最小元素
✅ `nlargest(k, iterable)` 最大的K个
✅ `nsmallest(k, iterable)` 最小的K个

**实现最大堆**: 存储负数

**下一步**: [16-高级特性.md](./16-高级特性.md)

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
