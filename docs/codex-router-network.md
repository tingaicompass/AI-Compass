# Codex 外部路由网络排障

## 适用场景

当 `ai-compass-weekly-highlight-sync` 的 hybrid 路由出现以下错误时，先按本页排障：

```text
socket.gaierror: [Errno 8] nodename nor servname provided, or not known
模型裁决失败，已回退规则结果
```

## 2026-10-18 事件结论

本次周报同步中，Codex 进程无法解析远端 OpenAI-compatible 路由端点；同一时间 Dify Docker 容器的模型请求曾持续返回 `200 OK`。本机还观察到：

- macOS 配置了名为“速博士”的 VPN 服务，但 `scutil --nc list` 显示该服务为 `Disconnected`。
- `utun4` 仍存在，并有分流路由；Wi-Fi 的系统 DNS 文件显示为 `192.168.5.1`。
- Codex 沙箱内 `socket.getaddrinfo()` 对远端端点和 `example.com` 都失败，`dig` 还受到 `Operation not permitted` 限制。

因此不能仅凭这次错误断言“VPN 一定坏了”。当前更可靠的判断是：**Codex 受限进程没有可用的 DNS/网络能力；VPN 断开可能是系统侧的附加因素，但需要在普通 Terminal 中复测后才能确认。** 不要在没有复测的情况下强行覆盖 VPN 的 DNS。

## 先判断问题在哪一层

以下命令应在 macOS 普通 Terminal 中执行，而不是只在 Codex 沙箱里执行：

```bash
HOST=ws-ozv555vq5mlbiepv.cn-beijing.maas.aliyuncs.com
dscacheutil -q host -a name "$HOST"
dscacheutil -q host -a name example.com
scutil --nc list
route -n get default
```

判断方式：

1. 如果普通 Terminal 能解析两个域名，而 Codex 不能：这是 Codex 网络权限/沙箱路径问题，不要修改 macOS DNS。使用已批准的网络环境运行模型复核，或直接启用 rules fallback。
2. 如果普通 Terminal 也不能解析：先重新连接当前 VPN；若仍失败，再检查 Wi-Fi DNS、路由器 DNS 和 VPN 的 DNS 分流配置。
3. 如果只有模型端点失败而 `example.com` 正常：检查 VPN 是否把该域名解析到不可达的分流网段，或暂时使用可访问的路由端点。

## 不阻塞周报同步的固定做法

在运行同步前先做一次 DNS 预检：

```bash
python3 - <<'PY'
import socket

host = "ws-ozv555vq5mlbiepv.cn-beijing.maas.aliyuncs.com"
socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
print("router DNS: ok")
PY
```

预检失败时，使用仓库内的安全包装脚本。它会自动在 DNS 正常时选择 hybrid，在 DNS 失败时选择 rules，避免每个条目重复等待模型重试：

```bash
cd /Users/ting/code/AI-Compass
./tools/sync_weekly_highlight_safe.sh \
  --source-file "/path/to/weekly_report_YYYYMMDD_HHMMSS.md" \
  --featured-brief-file "/Users/ting/code/AI-Compass/weeklyHighlights/featuredBriefs/<issue>.json" \
  --published-at 2026-10-18
```

这不会跳过专题同步，也不会减少条目覆盖；DNS 失败时它只是不调用外部模型做边界裁决。脚本仍会输出 Top 3、主专题、关联专题和路由诊断，后续网络恢复后可以用同一源文件重跑以刷新模型裁决结果。

## VPN 侧修复顺序

只有在普通 Terminal 的 DNS 预检也失败时才执行：

1. 在系统 VPN 客户端中重新连接当前 VPN，确认 `scutil --nc list` 显示 `Connected`。
2. 重新运行 `dscacheutil` 检查；不要先删除现有分流路由。
3. 如果 VPN 明确要求自带 DNS，优先恢复 VPN 提供的 DNS；不要直接用公共 DNS 替换内部域名解析。
4. 只有确认不依赖内部域名时，才考虑临时设置公共 DNS，并在验证后恢复原配置。
5. DNS 配置变更后刷新缓存：

```bash
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
```

## 交付前检查

- `topic_sync_summary.expected_sections == synced_sections`
- `routing_diagnostics` 中记录 rules/model/fallback 数量和 warning
- 外部模型不可用时，没有留下长时间运行的重试进程
- 记录实际网络层：Codex 沙箱、普通 Terminal、Docker 容器，不把三者的结果混为一谈
