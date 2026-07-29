# 安装 AI-Compass 本地知识库 Skill

本说明面向执行安装任务的 Coding Agent。目标是让同一份 AI-Compass 工作副本同时被 Codex 和 Claude Code 发现为本地 Skill。

## 安装步骤

1. 确认 AI-Compass 已克隆到本机。若未克隆，使用默认位置：

   ```bash
   git clone https://github.com/tingaicompass/AI-Compass.git "$HOME/AI-Compass"
   ```

2. 进入该工作副本并执行根目录安装脚本：

   ```bash
   cd "$HOME/AI-Compass"
   bash ./install.sh
   ```

   如果仓库已在其他位置，保留该工作副本位置并从其根目录执行 `bash ./install.sh`；不要复制整仓内容到 Skill 目录。

3. 验证两个软链接都指向同一份工作副本：

   ```bash
   test -L "$HOME/.codex/skills/ai-compass"
   test -L "$HOME/.claude/skills/ai-compass"
   ```

4. 向用户报告工作副本路径、两个验证结果，并提示重启 Codex、Claude Code 使新 Skill 生效。

## 后续更新

在工作副本中执行以下命令同步内容：

```bash
git pull --ff-only
```

软链接会继续指向同一份工作副本，无需重新安装。若 `git pull --ff-only` 因本地修改而停止，保留修改并向用户说明冲突，不要覆盖或删除本地内容。

## 安全约束

- `install.sh` 只创建或更新 `~/.codex/skills/ai-compass` 与 `~/.claude/skills/ai-compass` 两个软链接。
- 若任一目标是普通文件或目录，脚本会停止；安装 Agent 应报告该路径，而不是删除或覆盖它。
