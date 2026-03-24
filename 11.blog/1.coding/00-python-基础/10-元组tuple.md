> 想系统提升编程能力、查看更完整的学习路线，欢迎访问 AI Compass：https://github.com/tingaicompass/AI-Compass
> 仓库持续更新刷题题解、Python 基础和 AI 实战内容，适合想高效进阶的你。

# 10 - 元组tuple:不可变序列

> **学习目标**: 理解元组的特性和使用场景

---

## 💻 代码示例

```python
# 创建元组
t = (1, 2, 3)
print(t)  # (1, 2, 3)

# 单元素元组(注意逗号!)
t1 = (1,)   # 元组
t2 = (1)    # 整数

# 访问元素
print(t[0])   # 1
print(t[-1])  # 3

# 元组是不可变的
# t[0] = 10  # ❌ 报错

# 元组解包
a, b, c = (1, 2, 3)
print(a, b, c)  # 1 2 3

# 交换变量
x, y = 10, 20
x, y = y, x  # 利用元组解包
print(x, y)  # 20 10

# 函数返回多个值
def get_name_age():
    return "Alice", 20

name, age = get_name_age()
```

---

## 🎯 在算法题中的应用

```python
# 坐标表示
point = (3, 4)
x, y = point

# 字典项
d = {"a": 1, "b": 2}
for key, value in d.items():  # items()返回元组
    print(key, value)

# 多值排序
students = [("Alice", 90), ("Bob", 85), ("Charlie", 90)]
students.sort(key=lambda x: (-x[1], x[0]))  # 按分数降序,姓名升序
```

---

## 🎓 小结

✅ 元组是**不可变的**列表
✅ `(1, 2, 3)` 创建
✅ 元组解包:`a, b = (1, 2)`
✅ 用于返回多个值、字典的items()

**下一步**: [11-推导式.md](./11-推导式.md)

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
