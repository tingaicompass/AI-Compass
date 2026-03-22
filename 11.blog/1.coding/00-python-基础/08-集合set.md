# 08 - 集合set:去重和快速查找

> **学习目标**: 掌握集合的创建和应用

---

## 💻 代码示例

### 1. 创建集合

```python
# 方法1:花括号
s = {1, 2, 3, 4, 5}
print(s)  # {1, 2, 3, 4, 5}

# 方法2:set()函数
s = set([1, 2, 2, 3, 3, 3])  # 自动去重
print(s)  # {1, 2, 3}

# 空集合(⚠️ 不能用{},那是空字典)
empty = set()

# 从字符串创建
chars = set("hello")
print(chars)  # {'h', 'e', 'l', 'o'}
```

### 2. 集合操作

```python
s = {1, 2, 3}

# 添加元素
s.add(4)
print(s)  # {1, 2, 3, 4}

# 删除元素
s.remove(2)  # 如果元素不存在会报错
print(s)  # {1, 3, 4}

s.discard(10)  # 元素不存在也不报错

# 检查成员
print(1 in s)   # True
print(10 in s)  # False

# 长度
print(len(s))  # 3
```

### 3. 集合运算

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

# 并集
print(a | b)  # {1, 2, 3, 4, 5, 6}
print(a.union(b))

# 交集
print(a & b)  # {3, 4}
print(a.intersection(b))

# 差集
print(a - b)  # {1, 2}
print(a.difference(b))

# 对称差集
print(a ^ b)  # {1, 2, 5, 6}
```

---

## 🎯 在算法题中的应用

```python
# 第3课:最长连续序列
def longestConsecutive(nums):
    num_set = set(nums)  # O(1)查找
    longest = 0

    for num in num_set:
        # 只从序列起点开始
        if num - 1 not in num_set:
            current = num
            length = 1

            while current + 1 in num_set:
                current += 1
                length += 1

            longest = max(longest, length)

    return longest

# 去重
nums = [1, 2, 2, 3, 3, 3]
unique = list(set(nums))  # [1, 2, 3]
```

---

## 🎓 小结

✅ `{1, 2, 3}` 创建集合
✅ `s.add(x)` 添加元素
✅ `x in s` O(1)查找
✅ `|` `&` `-` `^` 集合运算
✅ 自动去重

**核心优势**: O(1)查找,类似字典但只存值不存键值对

**下一步**: [09-字符串str.md](./09-字符串str.md)

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
