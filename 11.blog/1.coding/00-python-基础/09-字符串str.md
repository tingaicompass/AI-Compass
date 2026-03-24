> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 09 - 字符串str:文本处理

> **学习目标**: 掌握字符串操作方法

---

## 💻 代码示例

### 1. 字符串基础

```python
s = "Hello, World!"

# 长度
print(len(s))  # 13

# 访问字符
print(s[0])   # H
print(s[-1])  # !

# 切片
print(s[0:5])   # Hello
print(s[7:])    # World!
print(s[::-1])  # !dlroW ,olleH (反转)

# 字符串是不可变的
# s[0] = 'h'  # ❌ 报错
```

### 2. 字符串方法

```python
s = "hello world"

# 大小写
print(s.upper())       # HELLO WORLD
print(s.lower())       # hello world
print(s.capitalize())  # Hello world
print(s.title())       # Hello World

# 查找
print(s.find("world"))  # 6 (找不到返回-1)
print(s.index("world")) # 6 (找不到报错)
print(s.count("l"))     # 3

# 替换
print(s.replace("world", "Python"))  # hello Python

# 分割
words = s.split()  # 默认按空格分割
print(words)  # ['hello', 'world']

csv = "a,b,c"
parts = csv.split(",")
print(parts)  # ['a', 'b', 'c']

# 连接
words = ["hello", "world"]
print(" ".join(words))  # hello world
print("-".join(words))  # hello-world

# 去除空白
s = "  hello  "
print(s.strip())   # "hello"
print(s.lstrip())  # "hello  "
print(s.rstrip())  # "  hello"

# 判断
print("hello".startswith("he"))  # True
print("hello".endswith("lo"))    # True
print("123".isdigit())           # True
print("abc".isalpha())           # True
```

### 3. 字符串格式化

```python
name = "Alice"
age = 20

# f-string(Python 3.6+,推荐)
print(f"我是{name},今年{age}岁")

# format方法
print("我是{},今年{}岁".format(name, age))
print("我是{n},今年{a}岁".format(n=name, a=age))

# %格式化(旧式)
print("我是%s,今年%d岁" % (name, age))
```

---

## 🎯 在算法题中的应用

```python
# 第2课:字母异位词分组
def groupAnagrams(strs):
    groups = {}
    for word in strs:
        key = "".join(sorted(word))  # 字符串排序
        groups[key] = groups.get(key, []) + [word]
    return list(groups.values())

# 第18课:最长回文子串
def longestPalindrome(s):
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]

    result = ""
    for i in range(len(s)):
        # 奇数长度回文
        odd = expand(i, i)
        # 偶数长度回文
        even = expand(i, i+1)
        result = max(result, odd, even, key=len)
    return result

# 反转字符串
s = "hello"
reversed_s = s[::-1]  # olleh
```

---

## 🎓 小结

✅ 字符串是**不可变的**
✅ 切片`s[start:end]`, `s[::-1]`反转
✅ 常用方法:`split`, `join`, `strip`, `replace`
✅ 判断:`startswith`, `endswith`, `isdigit`
✅ f-string格式化

**下一步**: [10-元组tuple.md](./10-元组tuple.md)

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
