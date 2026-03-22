# 📖 第20课：回文子串

> **模块**：字符串 | **难度**：Medium ⭐⭐
> **LeetCode 链接**：https://leetcode.cn/problems/palindromic-substrings/
> **前置知识**：第18课（最长回文子串）、中心扩展法
> **预计学习时间**：25分钟

---

## 🎯 题目描述

给你一个字符串 `s`，请你统计并返回这个字符串中**回文子串**的数目。

**回文字符串**是正着读和倒着读都一样的字符串。

**子字符串**是字符串中连续的字符序列。

**示例1：**
```
输入：s = "abc"
输出：3
解释：三个回文子串: "a", "b", "c"
```

**示例2：**
```
输入：s = "aaa"
输出：6
解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
```

**约束条件：**
- `1 <= s.length <= 1000`
- `s` 由小写英文字母组成

---

### 🧪 边界用例（面试必考）

| 用例类型 | 输入 | 期望输出 | 考察点 |
|---------|------|---------|--------|
| 单字符 | `s = "a"` | `1` | 最小输入，单字符是回文 |
| 无长回文 | `s = "abc"` | `3` | 只有单字符回文 |
| 全相同 | `s = "aaa"` | `6` | `"a"×3 + "aa"×2 + "aaa"×1 = 6` |
| 奇数回文 | `s = "aba"` | `4` | `"a", "b", "a", "aba"` |
| 偶数回文 | `s = "abba"` | `6` | `"a", "b", "b", "a", "bb", "abba"` |

---

## 💡 思路引导

### 生活化比喻
> 想象你在一本书里找"回文句子"，不是要找最长的那句，而是要**数出一共有多少句**...
>
> 🐌 **笨办法**：枚举所有可能的子串，逐个检查是否为回文，每找到一个就计数+1。效率极低。
>
> 🚀 **聪明办法**：从每个可能的"中心点"出发，向两边扩展，**每扩展成功一次，计数就+1**。

### 关键洞察
**本题和第18课的区别：第18课求"最长"，本题求"数量"。核心方法相同——中心扩展法！**

---

## 🧠 解题思维链

### Step 1：理解题目 → 锁定输入输出
- **输入**：字符串 `s`，长度 1 ~ 1000
- **输出**：回文子串的总数量（整数）

### Step 2：先想笨办法（暴力法）
枚举所有子串（起点i，终点j），逐个检查是否回文，是的话计数+1。
- 时间复杂度：O(n³)

### Step 3：瓶颈分析 → 优化方向
回文的关键特征：**中心对称！**
- 优化思路：**从每个中心出发，向两边扩展，每扩展成功一次就是一个回文**

### Step 4：选择武器
- 选用：**中心扩展法**
- 时间复杂度降到 O(n²)

---

## 🔑 解法一：中心扩展法（推荐⭐⭐⭐）

### Python代码

```python
def count_substrings(s: str) -> int:
    """
    解法一：中心扩展法
    思路：从每个中心向两边扩展，每扩展成功一次计数+1
    """
    def expand_around_center(left: int, right: int) -> int:
        """从中心(left, right)向两边扩展，返回回文数量"""
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count

    total = 0
    for i in range(len(s)):
        # 奇数长度回文
        total += expand_around_center(i, i)
        # 偶数长度回文
        total += expand_around_center(i, i + 1)
    return total


# ✅ 测试
print(count_substrings("abc"))   # 3
print(count_substrings("aaa"))   # 6
```

### 复杂度分析
- **时间复杂度**：O(n²)
- **空间复杂度**：O(1)

---

## 📊 解法对比

推荐使用**中心扩展法**（时空最优，O(n²) 时间，O(1) 空间）

---

## 🎤 面试现场

**你**：这道题和"最长回文子串"类似，都用中心扩展法，但目标不同——之前求最长，现在求数量。

从每个中心出发向两边扩展，每扩展成功一次（左右字符相同）就找到一个回文，计数+1。

时间 O(n²)，空间 O(1)。

---

## 🎓 知识点总结

### 算法模式卡片 📐
```python
def count_palindromes(s: str) -> int:
    """中心扩展计数模板"""
    def expand(left, right):
        count = 0
        while left >= 0 and right < len(s) and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
        return count

    total = 0
    for i in range(len(s)):
        total += expand(i, i)       # 奇数回文
        total += expand(i, i + 1)   # 偶数回文
    return total
```

---

## 🏋️ 举一反三

| 题目 | 难度 | 提示 |
|------|------|------|
| LeetCode 5. 最长回文子串 | Medium | 本题的姊妹题，求最长而非数量 |
| LeetCode 516. 最长回文子序列 | Medium | 子序列可以不连续，需要DP |

---

💪 **继续加油！**

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
