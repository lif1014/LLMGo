---
name: cctl
description: 使用 cctl 命令行工具登录 Cybertron，并通过 CLI 查询/创建任务、管理镜像、项目和资源池。当用户明确要求「用 CLI / cctl 操作」时使用。
version: v0.0.22
---

## 注意事情：
你只允許提交--priority NORMAL的任務，不允許任何kill掉別人任務的行為。
--cluster 請用beijing
卡池可通过cctl pool list --own查詢，只允許用model-autosft卡池
成立devspace時不允許使用GPU，即成立devspace GPU數目必須為0；如要用GPU，請通过job / pytorchjob來完成

# cctl — Cybertron CLI

`cctl` 是 Cybertron 平台命令行工具，源码位于 `cmd/cctl/`。版本通过 `-ldflags` 在编译时注入。

> 当前发布版本：`v0.0.22`

## 1. 当前可用顶层命令

当前 root command 已挂载的命令只有这些：

```bash
cctl login
cctl logout
cctl profile
cctl version
cctl changelog
cctl update
cctl skill

cctl task
cctl job
cctl pytorchjob
cctl notebook
cctl devspace

cctl image
cctl filesystem
cctl project
cctl pool
cctl node
cctl cluster
cctl billing-account
cctl file-task

cctl ops
cctl diagnosis
```

## 2. 全局约定

- 配置目录：`~/.cybertronctl/`
- profile 存储：
  - 当前默认 profile：`~/.cybertronctl/current-profile`
  - 各 profile 凭证目录：`~/.cybertronctl/profiles/<profile>/`
  - 每个 profile 保存：`token`、`server`、`token_expire_at`
- 默认 profile 选择优先级：`--profile` > `CCTL_PROFILE` > `current-profile` > `modelbest`
- `cctl login` 的默认约定：若未显式传 `--profile`，则 `profile = env`
- 向后兼容：默认 `modelbest` profile 会回退读取旧的全局文件 `~/.cybertronctl/token`、`server`、`token_expire_at`
- server 选择优先级：`CCTL_SERVER` > 本地保存的 server
- token 选择优先级：`CYBERTRON_TOKEN` > 本地保存的 token
- 输出格式：
  - `-o table|json|yaml`
  - `-q, --quiet`：只输出 ID，一行一个
  - `-F, --fields`：只输出指定 JSON 字段，逗号分隔；设置后会强制走 JSON 输出
- 未显式指定 `-o` 时：
  - 终端里默认 `table`
  - 非 TTY 场景默认 `json`
- 非 TTY 场景下出错时，CLI 会输出结构化 JSON 错误，便于 agent/脚本消费

机器可读输出常用写法：

```bash
cctl job list -o json
cctl image list -q
cctl project list -F name,projectName,status
```

## 3. 认证

### 3.1 登录与登出

```bash
# 默认 modelbest，优先尝试浏览器登录
cctl login

# 指定环境
cctl login -e thunlp
cctl login -e omnibest
cctl login -e dev2

# 显式指定 profile
cctl login -e modelbest --profile prod
cctl login -e thunlp --profile thunlp
cctl login -e omnibest --profile omnibest
cctl login -e dev2 --profile dev2

# 强制手动粘贴 token
cctl login -m
cctl login --manual -e thunlp

# 直接保存 token
cctl login <token>
cctl login <token> -e thunlp

# 登出
cctl logout
cctl --profile thunlp logout

# profile 管理
cctl profile current
cctl profile list
cctl profile use thunlp

# 查看版本
cctl version
```

预定义环境：

| 环境名 | 地址 |
|--------|------|
| `modelbest` | `https://cybertron.modelbest.co` |
| `thunlp` | `http://cybertron.thunlp.org` |
| `omnibest` | `https://cybertron.omnibest.cn` |
| `dev2` | `https://cybertron.ali-dev.modelbest.co` |

### 3.2 认证行为说明

- `cctl login` 默认先走浏览器登录，底层依赖 `/auth/cli/session`
- 如果服务端还没实现 CLI session 登录，CLI 会自动回退到手动粘贴 token
- 登录成功后会把 token 和 server 保存到目标 profile，并把该 profile 设为新的默认 profile
- 若未显式传 `--profile`，CLI 会自动使用与 `--env` 同名的 profile
- 后续普通命令会按 `--profile` / `CCTL_PROFILE` / `current-profile` / `modelbest` 的顺序选择要使用的那套凭证
- 登录成功后会尝试调用 `/v2/clusters?page_size=1` 验证 token
- 若保存的是浏览器登录得到的 token，CLI 会基于 `token_expire_at` 在临近过期时自动刷新
- 若请求返回 `401`，CLI 也会尝试自动刷新后再重试一次
- 如果使用 `CYBERTRON_TOKEN` 环境变量注入 token，则不会做本地自动刷新

### 3.3 AI Agent 使用登录命令的约定

`cctl login` 的交互流程会输出两个机器可读的哨兵行，AI Agent **必须**用它们来控制等待节奏：

| 哨兵行 | 含义 | AI 应做什么 |
|--------|------|------------|
| `[cctl:waiting-for-human]` | 需要人类在浏览器完成授权，或粘贴 token | **停止向下执行**，提示人类操作，持续监听输出 |
| `[cctl:login-complete]` | 登录已验证成功，token 已保存 | 可以继续执行后续命令 |
| `[cctl:update-available]` | CLI 检测到新版本（任意命令执行后都可能出现在 stderr） | 读取紧随其后的 JSON，按 §3.4 处理；**不要阻塞当前任务** |

**推荐 Agent 写法**（使用 Await pattern）：

```
# 1. 在后台启动登录（block_until_ms: 0）
cctl login -e thunlp

# 2. 等待 waiting-for-human 信号，然后提示人类去浏览器操作
# pattern: \[cctl:waiting-for-human\]

# 3. 等待 login-complete 信号，再继续
# pattern: \[cctl:login-complete\]
```

**如果是纯脚本场景**（无人值守），跳过交互登录，直接传 token：

```bash
cctl login <token> -e modelbest   # 无任何交互，直接保存并验证
```

### 3.4 升级提示的 Agent 处理策略

任意 cctl 命令运行后，若本地缓存判定存在新版本，stderr 会追加 **哨兵行 + 一行 JSON**（只出现在非 TTY，即 agent / 管道场景）：

```
[cctl:update-available] current=v0.0.8 latest=v0.0.9
{"notice":"cctl-update-available","current":"v0.0.8","latest":"v0.0.9","binary_path":"/usr/local/bin/cctl","upgrade_command":"sudo cctl update","requires_sudo":true,"auto_upgradable":false,"platform":"linux","hint":"binary 位于 /usr/local/bin/cctl，升级需要 sudo"}
```

Agent 必须按以下规则处理：

1. **不要阻塞当前任务**：当前请求的主 stdout 输出仍然有效，升级提示是带外元信息；先把用户请求处理完。
2. **不要擅自执行升级**：即使 `auto_upgradable=true`，也**仅在用户明确同意后**才执行 `upgrade_command`。
3. **在最终回复末尾**附一段提示，至少包含：
   - 版本号 `current → latest`
   - `hint` 字段文本
   - 建议的 `upgrade_command`
4. **`requires_sudo=true`**：提示用户手动执行 `sudo cctl update`，agent 不要尝试在自己的 shell 里调 sudo。
5. **多次出现时去重**：批量调用 cctl 可能产生多条相同 notice，agent 只在最终回复里提示一次。
6. **禁用开关**：通过 `CCTL_NO_UPDATE_CHECK=1` 可完全关闭检查与提示，适用于 CI / 受限网络。

## 4. 任务命令

四种任务类型分别对应独立顶层命令，CLI 会自动注入 `kind`：

| 命令 | 别名 | 任务类型 |
|------|------|----------|
| `cctl job` | `jobs`, `j` | `BATCH` |
| `cctl pytorchjob` | `pytorchjobs`, `pj`, `ptj` | `PYTORCHJOB` |
| `cctl notebook` | `notebooks`, `nb` | `NOTEBOOK` |
| `cctl devspace` | `devspaces`, `ds` | `DEVSPACE` |

### 4.1 当前已挂载的子命令

```bash
# 列表
cctl job list [--project X] [--cluster X] [--pool X] [--entry KEYWORD] [--status X] [--own] [--limit 50] [--page 1] [--web]

# 详情
cctl job get <id-or-name> [--web]

# 创建
cctl job create ...
cctl pytorchjob create ...
cctl notebook create ...
cctl devspace create ...

# 停止（DEVSPACE 可选停止前先打可恢复快照）
cctl job stop <id-or-name> [--reason "..."] [--snapshot-before-stop best-effort|required]

# 日志
cctl job logs <id-or-name> [--pod <name>|all] [--page-size 300] [--page-token TOKEN] [--stream]

# 复制
cctl job copy <id-or-name> [--dry-run] [覆盖字段...]

# 服务暴露（查看 / 开启或修改端口 / 关闭 / 打开）
cctl job expose <id-or-name>
cctl job expose <id-or-name> --port 8666
cctl job expose <id-or-name> --disable
cctl job expose <id-or-name> --port 8666 --open

# Devspace 恢复（仅此 kind 挂载；依赖源任务已成功 auto snapshot）
cctl devspace restore <id-or-name> [--dry-run] [--yes] [--confirm tasks/1234]
```

通用别名：

- `list`：`ls`, `l`
- `get`：`g`, `describe`, `desc`
- `stop`：`kill`
- `logs`：`log`
- `copy`：`cp`, `clone`
- `expose`：`service`

### 4.2 列表与详情

```bash
# 过滤训练任务
cctl job list --project my-project --cluster paratera --status Running --own
cctl job list --entry "python train.py"   # 启动命令子串匹配（后端 EntryContains）

# 直接打开 Web 页面
cctl job list --web
cctl job get 1234 --web
```

- `list --web` 会直接打开任务列表页
- `get --web` 会优先打开 `dashboardUrl`，否则退化到 `/job-detail?id=<task-id>`

### 4.2.1 job / task list 与 `--status`

- 要列出**全部**任务：**不要**写 `--status All`；应**直接省略** `--status`。
- cctl 仅接受字面量 `--status all`、`--status All`、`--status ALL` 表示「不过滤状态」（与省略 `--status` 等价）；**不要**用 `EqualFold` 类拼写（如 `alL`），会报参数错误。
- 其它未识别的 `--status` token 会在 CLI **本地**返回 `invalid_argument`，不要误当成 PAT/网络问题或「平台一定没有任务」。

### 4.3 创建任务

所有类型通用的必填参数：

- `--project`
- `--cluster`
- `--image`
- `--resource-pool`

按任务类型追加要求：

| 参数 | `BATCH` | `PYTORCHJOB` | `NOTEBOOK` | `DEVSPACE` |
|------|---------|--------------|------------|------------|
| `--entry` | 必填 | 必填 | 非必填 | 非必填 |
| `--workers` | - | 必填 | - | - |
| `--duration` | - | - | 非必填，默认 240 分钟 | 非必填 |

常用可选参数：

- `--priority HIGH|NORMAL|PREEMPTABLE`
- `--gpu`
- `--gpu-model`
- `--cpu`
- `--memory`
- `--env KEY=VALUE`，可重复传
- `--requeue`
- `--namespace`
- `--code-type git`
- `--git-path`
- `--git-ref`
- `--description`：用户自定义的任务描述，任意文本，便于后续识别任务用途
- `--expose-port`：任务创建成功后立即暴露容器端口，取值 `1..65535`

典型示例：

```bash
# BATCH
cctl job create \
  --project my-project \
  --cluster gpu-cluster \
  --resource-pool training-pool \
  --image nvcr.io/nvidia/pytorch:24.01-py3 \
  --gpu 8 --gpu-model A100 \
  --entry "python train.py --epochs 100" \
  --description "MiniCPM-V 7B SFT 实验 2026-04-18"

# PYTORCHJOB
cctl pytorchjob create \
  --project my-project \
  --cluster gpu-cluster \
  --resource-pool training-pool \
  --image my-training:latest \
  --gpu 8 --gpu-model H100 \
  --workers 4 \
  --entry "torchrun --nproc_per_node=8 train.py"

# NOTEBOOK
cctl notebook create \
  --project my-project \
  --cluster gpu-cluster \
  --resource-pool training-pool \
  --image jupyter:latest \
  --gpu 1
```

创建行为说明：

- `--resource-pool` 可以传资源池 `ID`、`name` 或 `alias`
- `--cpu` / `--memory` 未显式给定时，CLI 可能调用 `/resources/recommend` 自动补全推荐值
- `NOTEBOOK` 未显式传 `--duration` 时，CLI 默认使用 240 分钟
- `--namespace` 未显式给定时，默认使用 `training` 命名空间
- `--code-type=git` 时必须补 `--git-path`
- `--expose-port` 不写则不暴露；写了会在任务创建成功后调用服务暴露接口配置端口
- 支持从 JSON 文件创建，指定 `-f` 后任务主体来自文件；`--billing-account-id` 可作为 fallback，`--expose-port` 可在创建成功后追加服务暴露

### 4.3.1 服务暴露

任务运行前后均可通过 CLI 配置服务暴露：

```bash
# 创建时配置暴露端口
cctl job create ... --expose-port 8666

# 查看当前暴露状态
cctl job expose 1234

# 开启或修改暴露端口
cctl job expose 1234 --port 9000

# 关闭服务暴露
cctl job expose 1234 --disable

# 打开暴露服务 URL
cctl job expose 1234 --open
```

说明：

- `expose` 子命令同样挂载在 `pytorchjob`、`notebook`、`devspace` 下。
- `--port` / `--expose-port` 必须为 `1..65535`。
- `--open` 可与 `--port` 组合，先更新端口再打开 URL。
- `-q` 只输出暴露服务 URL；未开启时为空输出。
- `copy --expose-port` 只在显式传入时开启暴露，默认不继承源任务端口。

任务 JSON 支持两种格式：

```json
{
  "task": {
    "kind": "BATCH",
    "project": "my-project",
    "cluster": "gpu-cluster",
    "resourcePool": "training-pool",
    "image": "nvcr.io/nvidia/pytorch:24.01-py3",
    "resources": {
      "gpuCount": 8,
      "gpuModel": "A100",
      "cpu": 32,
      "memory": "128Gi"
    },
    "entry": "python train.py"
  }
}
```

也支持直接传裸 `Task` 对象，不带 `"task"` 包装。

### 4.4 日志与停止

```bash
cctl job stop 1234 --reason "manual stop"
cctl devspace stop 5678 --snapshot-before-stop best-effort --yes
cctl devspace stop 5678 --snapshot-before-stop required --reason "checkpoint"
cctl job logs 1234 --page-size 200
cctl job logs 1234 --pod task-1234-worker-0
cctl job logs 1234 --pod all
cctl job logs 1234 --stream
```

- `stop`：v2 `POST /api/v2/tasks/<id>:stop`，body 可使用 **`snapshotBeforeStop`**：`SNAPSHOT_BEFORE_STOP_BEST_EFFORT` 或 `SNAPSHOT_BEFORE_STOP_REQUIRED`（也可用 cctl `--snapshot-before-stop`）。仅 **DEVSPACE** 生效；其余 kind 传非默认值会返回 `INVALID_ARGUMENT`。与 lifecycle idle release 同源 **auto_snapshot** 打点实现。**生命周期在 deadline 到期时仍仅对申请了 GPU 的 devspace 做自动快照并释放；0 GPU devspace 不会在到点时自动打点，可通过本 stop 快照参数显式保存可恢复快照。** `best-effort`：集群不支持快照 / 非 Running / 超时或快照失败时仍照常停止；`required`：必须 `succeeded` 后才终止，否则 `FAILED_PRECONDITION` 且不结束任务。**带快照策略的请求可能阻塞约半小时量级**，CLI 会使用约 **32 分钟** HTTP 读超时。
- `logs`：同时支持运行中任务和已停止/归档任务
- `--stream` 仅支持当前正在运行的任务，且一次只跟随单个 pod
- 未指定 `--pod` 时，默认取主 pod；传 `--pod all` 时会按 pod 分段输出
- 支持 `--page-size` / `--page-token` 翻页；CLI 会在 stderr 提示上一页/下一页 token
- 响应被截断时，CLI 会提示被省略的 pod 列表
- 当日志暂不可用时，CLI 会给出原因提示，并在服务端返回时补充 `log path`

### 4.5 复制任务

```bash
# 先预览
cctl job copy 1174 --image new-image:latest --dry-run

# 再提交
cctl job copy 1174 --image new-image:latest
```

`copy` 支持覆盖这些字段：

- `--image`
- `--entry`
- `--priority`
- `--gpu`
- `--gpu-model`
- `--cpu`
- `--memory`
- `--workers`
- `--duration`
- `--project`
- `--cluster`
- `--resource-pool`
- `--git-ref`
- `--description`
- `--expose-port`：创建复制任务后显式开启服务暴露；默认不继承源任务暴露端口
- `--dry-run`

未传 `--description` 时，copy 会继承源任务的描述；传了则覆盖。

### 4.6 Devspace 恢复与环境保存

**恢复（CLI）**

```bash
cctl devspace restore <id> --yes
# 或交互确认：按提示输入 tasks/<id>
cctl devspace restore <id> --confirm tasks/1234
```

- 调用 v2：`POST /api/v2/tasks/{id}:restore`（与 `cctl job stop` 同源鉴权）。
- 服务端要求：源任务为 `DEVSPACE`，且 `auto_snapshot_status == succeeded`、`auto_snapshot_id` 有效；否则会 `FAILED_PRECONDITION`。
- 成功后会**新建**一条 devspace 任务（新 id），镜像来自快照；不是原地重启 Pod。

**可恢复用的「环境保存」（平台自动快照）**

- 由 devspace **生命周期 worker** 在 **deadline 到期且任务仍在 Running** 时触发（**仅限申请了 GPU** 的 devspace；纯 CPU devspace **不会在到点时自动打快照释放**，可避免「到点强行关」，需要可恢复快照时请用 **`cctl devspace stop --snapshot-before-stop`**）。
- 集群需支持镜像快照（集群配置 `snapshot_url` 等）；完成后任务会被 idle release，快照字段写入 `auto_snapshot_*`，此时才可用 `restore` 或管控台「恢复」。

**主动打镜像快照（不等同于上述 auto_snapshot，不能单独满足 restore）**

若要在运行中把当前容器打成镜像快照记录（与 `restore` 用的 `auto_snapshot` 不是同一条链路）：

```bash
# v1
curl -X POST "$CCTL_SERVER/api/image_snapshot?job_id=<devspace_job_id>" \
  -H "Authorization: Bearer $CYBERTRON_TOKEN"

# v2
curl -X POST "$CCTL_SERVER/api/v2/image-snapshots" \
  -H "Authorization: Bearer $CYBERTRON_TOKEN" -H "Content-Type: application/json" \
  -d '{"job_id":<id>,"cluster":"<cluster_name>"}'
```

**Devspace 续期（无 cctl 子命令）**

- 运行中需定期 `POST /api/job/devspace/heartbeat?id=<id>` 或 `POST /api/v2/tasks/<id>:heartbeat` 延长 deadline；README 中「每次顺延 5 分钟」可能已过时，以部署环境 `DEVSPACE_HEARTBEAT_EXTEND_MINUTE` 为准（默认 240 分钟）。

## 5. 镜像命令

```bash
# 列表 / 详情
cctl image list [--usage training|inference] [--status X] [--group N] [--own] [--limit 50] [--page 1]
cctl image get <name-or-id>

# 创建
cctl image create \
  --image-name pytorch-24.01 \
  --url nvcr.io/nvidia/pytorch:24.01-py3 \
  --usage training \
  [--canonical-name "..."] \
  [--description "..."] \
  [--source "..."] \
  [--hub-url "..."] \
  [--group N]

# 从 JSON 创建
cctl image create -f image.json

# 更新
cctl image update <name-or-id> [--image-name ...] [--url ...] [--canonical-name ...] [--usage ...] [--description ...] [--source ...] [--hub-url ...] [--group N]

# 删除 / 同步
cctl image delete <name-or-id>
cctl image sync <name-or-id>
```

镜像 JSON 示例：

```json
{
  "image": {
    "imageName": "pytorch-24.01",
    "url": "nvcr.io/nvidia/pytorch:24.01-py3",
    "canonicalName": "nvidia/pytorch:24.01-py3",
    "usage": "training",
    "description": "PyTorch 24.01"
  }
}
```

也支持直接传裸 `Image` 对象。

- `image get <name-or-id>` 支持三类输入：
  - 业务名，例如 `pytorch-24.01`
  - 带 `/` 的业务名，例如 `llm-rl/ngc-2506_xinjiang_train_zhangzhu_snapshot:20260420073452`
  - 资源名或数字 ID，例如 `images/42`、`42`
- `image list` 的 JSON/YAML 输出会带 `next_page_token`；CLI 在有下一页时会提示 `Next page: --page <n>`
- `image get` 如果返回 `syncStatuses`，CLI 会额外打印按集群展开的同步状态表（cluster/status/file task/pod/updated）

## 6. 项目命令

```bash
cctl project list [--name X] [--own] [--limit 50]
cctl project get <name-or-id>
cctl project create --name my-project [--description "..."]
```

项目别名：

- `project`：`projects`, `proj`

## 7. 资源池命令

## 6.1 文件系统命令

```bash
cctl filesystem list [--kind file|object] [--only-with-capacity] [--limit 50] [--page 1]
cctl filesystem get <id-or-name>

# 文件系统下每个用户的存储用量
cctl filesystem users <id-or-name> [--top 20]

# 单个用户在文件系统下各一级目录的大小（类似 du -sh <dir>/*）
cctl filesystem folders <id-or-name> --user <user_id> \
  [--order name|size|mtime] [--reverse] [--page 1] [--limit 50]

# 同步删除文件系统上的条目（破坏性，自动写入 file_delete_request 审计）
cctl filesystem rm <cluster:/absolute/path> [<cluster:/absolute/path>...] [-y]
```

别名：

- `filesystem`：`filesystems`, `fs`
- `users`：`user-usage`, `usage`
- `folders`：`folder`, `dirs`, `du`, `entries`, `user-entries`
- `rm`：`delete`, `delete-entries`

说明：

- `list`/`get` 桥接 v2 接口：`/api/v2/filesystems`，`list` 返回 `used_bytes`、`capacity_bytes`、`usage_ratio`、`last_stat_time` 等聚合字段；`get` 支持数值 ID、`filesystem_name`、`alias` 或完整资源名 `filesystems/<id>`
- `users` 桥接 v1 接口：`GET /api/file_stat_request/filesystem/{id}/users`，按总使用量降序展示该文件系统下所有用户的 `total_size`、`file_count`、`last_stat_time`；数据来自每周一次的 stat 快照；表格不展示 `user_id`，JSON/YAML 输出仍保留
- `folders` 桥接 v1 接口：`GET /api/file_stat_request?filesystem_id=..&user_id=..`，展示某个用户在该文件系统下各一级目录的大小与 mtime；表格会输出可直接复制到同步、外发、备份、删除命令的 `URI` 列，格式为 `cluster:/absolute/path`；省略 `--user` 时默认查询当前登录用户
- `users`/`folders` 会自动先用 `<id-or-name>` 把文件系统解析成数值 ID 再请求
- `rm` 桥接 v1 同步直删 `DELETE /{cluster}/file?filesystem_id=..`，结果按 `succeeded/skipped/failed` 三栏返回；`skipped` 是因路径不归当前用户被跳过、不算失败；服务端会异步把成功清单写入 `file_delete_request` 审计表。同一次调用所有 URI 必须共享同一集群和文件系统；推荐 URI 为 `cluster:/absolute/path`，旧的 `cluster:filesystem/path` 仍兼容；`-y / --yes` 跳过交互式确认

## 7. 资源池命令

```bash
cctl pool list [--own] [--role training|inference]
cctl pool get <name|alias|id>
cctl pool staff list <pool>
cctl pool staff list embody-train2
cctl pool staff add <pool> <user> [--dry-run]
cctl pool staff remove <pool> <user> [--yes] [--dry-run]
```

资源池成员（权限）说明：

- `staff add/remove` 桥接 v1：`POST/DELETE /api/resource_pool/staff`
- `<user>` 支持 `ou_xxx`、姓名、邮箱（与 `cctl filesystem folders --user` 相同解析逻辑）
- `<pool>` 支持数字 id、name、alias
- `remove` 默认交互确认；脚本请加 `--yes` 或 `--confirm remove/<pool_id>/<user_id>`

## 7.1 节点命令

```bash
cctl node list <cluster> [--pool <name>] [--only-available] [--include-system-pods] [-o json|table|yaml]
cctl node cordon <node> --cluster <cluster> --reason "..." [--yes|--confirm <cluster>/<node>]
cctl node uncordon <node> --cluster <cluster> --reason "..." [--yes|--confirm <cluster>/<node>]
```

节点别名：

- `node`：`nodes`, `n`
- `list`：`ls`, `l`

说明：

- 桥接 v2 接口：`/api/v2/clusters/{cluster}/nodes`
- 默认展示 GPU 节点的资源池、Ready/Maintenance/NotReady 状态、GPU/CPU/内存用量、运行任务数量与最显眼任务
- `--pool` 按资源池名过滤；`--only-available` 仅展示仍有空闲 GPU 的节点；`--include-system-pods` 用于管理员诊断时把系统 Pod 也计入任务列表
- `-q` 只输出节点名，`-o json|yaml` 输出完整结构化响应，适合脚本和 Agent 消费
- `cordon` / `uncordon` 桥接 v2 接口：`POST /api/v2/clusters/{cluster}/nodes/{node}:cordon|uncordon`
- 节点 cordon 写操作要求当前用户同时具备 `/api/v2/clusters/nodes/cordon` write 权限，以及目标节点所属资源池 staff 权限；服务端会写入 `resource_operation_history`
- `cordon` / `uncordon` 默认需要交互确认；脚本中请使用 `--yes` 或 `--confirm <cluster>/<node>`，并务必在测试后执行 `uncordon` 恢复节点调度

## 7.2 集群命令（只读）

```bash
cctl cluster list [--all] [-o json|table|yaml]
cctl cluster get <name|alias|id> [-o json|table|yaml]
```

集群别名：

- `cluster`：`clusters`, `cl`
- `list`：`ls`, `l`
- `get`：`g`, `describe`, `desc`

说明：

- 桥接 v2 接口：`/api/v2/clusters` / `/api/v2/clusters/{id}`
- `cluster list` 默认过滤 `status=="Active"`；`--all` 包含 Deleted 等历史项
- `cluster list -o json` 输出 envelope：`{"clusters":[...],"total":N}`，字段一律 camelCase（`clusterName` / `prometheusAddress` / `dashboardUrl` / `kubeconfigPath` / `gitInitImageTag`）
- `cluster get` 输入解析顺序：纯数字 → `/v2/clusters/<id>`；非数字 → list 后按 `clusterName` / `alias` 不区分大小写匹配
- 当前只挂只读子命令；`create` / `update` / `delete` 仍走管控台

## 7.3 Billing Account 命令

```bash
cctl billing-account list [-o json|table|yaml]
cctl billing-account list -q
cctl ba ls -F billing_account_id,billing_account_name
```

别名：

- `billing-account`：`billing-accounts`, `ba`
- `list`：`ls`, `l`

说明：

- 桥接 v2 接口：`/api/v2/me/billing-accounts`
- 用于查询当前用户可用于任务归因的 `billing_account_id`
- 创建或复制任务时可通过 `--billing-account-id <ID>` 显式指定；若资源池绑定了固定 billing account，`cctl` 会自动补齐或给出可操作报错
- 服务端拒绝 billing account 时，CLI 会尽量补充 suggestion 和可选账户列表，便于 agent/脚本修正参数

## 8. 文件任务命令

```bash
# 列表 / 详情
cctl file-task list [--type X] [--status X] [--user-id X] [--limit 50] [--page-token TOKEN]
cctl file-task get <id-or-resource-name>

# 快捷创建
cctl file-task import <src> <dst> [--bandwidth 200M] [--path shard-0]
cctl file-task sync <src> <dst> [--bandwidth 200M] [--path shard-0]
cctl file-task export <src> <dst> [--bandwidth 200M] [--path shard-0]
cctl file-task backup <src> [--target <object-fs>] [--dest-path PATH] [--bandwidth 200M]

# 取消
cctl file-task cancel <id-or-resource-name>
```

别名：

- `file-task`：`filetask`, `filetasks`
- `list`：`ls`, `l`
- `get`：`g`, `describe`, `desc`
- `cancel`：`delete`, `del`, `rm`

说明：

- 当前已正式挂载的是 `list`、`get`、`import`、`sync`、`export`、`backup`、`cancel`
- 资源名格式为 `filetasks/<id>`
- 源/目标位置优先使用可复制 URI 语法：`cluster:/absolute/path`
- 例如：`gpu-a100:/datasets/user/alice/input`
- CLI 会先把 URI 解析成 filesystem ID + path，再调用 v2 `/v2/filetasks` 接口
- `import` 适用于对象存储导入，`sync` 适用于文件系统之间同步/迁移，`export` 适用于导出到对象存储，`backup` 适用于备份语义（`type=backup`，目标 bucket 可用 `--target` 指定，省略时使用系统默认）

## 9. Ops 只读历史命令

```bash
# 资源操作历史
cctl ops resource-operation-history --cluster <cluster> --since 24h
cctl ops resource-operation-history --operation-id <id>

# 资源状态历史与 as-of 快照
cctl ops resource-state-history --cluster <cluster> --node <node> --since 24h
cctl ops resource-state-history --cluster <cluster> --node <node> --at "2026-06-03T10:00:00Z"
cctl ops resource-state-history --cluster <cluster> --at "2026-06-03T10:00:00Z"
```

说明：

- `resource-operation-history` 对应 `/api/v2/resource-operation-history`，用于查询资源操作审计线
- `resource-state-history` 对应 `/api/v2/resource-state-history`，并在 `--at` 模式下转到单资源或整集群 as-of 快照接口
- `ops audit list/get/k8s` 为兼容别名，会提示迁移到 `resource-operation-history`
- 这些命令均为只读诊断命令，不改变集群或任务状态

## 10. 版本与自更新

```bash
# 查看完整版本信息（版本号、commit、构建时间）
cctl version

# 查看当前版本或指定版本的 changelog
cctl changelog
cctl changelog latest
cctl changelog v0.0.12

# 只输出版本号
cctl version --short

# 检查是否有新版本
cctl update --check

# 下载并安装最新版
cctl update
```

说明：

- 版本信息通过 `-ldflags` 在编译时注入（`version`、`commit`、`date`）
- `cctl changelog [version|latest]` 从发布服务器读取对应版本的 `CHANGELOG.md`；不传版本时默认读取当前版本（开发构建默认读取 `latest`）
- `cctl update` 从 OSS 下载最新二进制替换当前可执行文件
- 同时刷新所有已安装的 cctl skill 文件（`~/.cursor/skills/cctl/SKILL.md`、`~/.claude/skills/cctl/SKILL.md`、`~/.agents/skills/cctl/SKILL.md`，以及 binary 同级兼容路径），不会自动创建新目录
- 可用 `cctl skill list` 查看当前 skill 安装情况，用 `cctl skill sync` 单独刷新 skill 文件（不下载二进制）
- 可通过 `CCTL_OSS_BASE` 环境变量覆盖下载服务器地址
- Windows 暂不支持原地 self-update；请改用 `scripts/install_cctl.ps1` 重新安装

### 10.1 自动版本检查

除 `cctl update` / `cctl version` 以外，任意子命令执行时都会在后台异步探测最新版本：

- 首次运行（或缓存过期）会在后台请求 `CCTL_OSS_BASE/latest/VERSION`，结果写入 `~/.cybertronctl/version_check.json`
- 节流：两次探测至少间隔 **12h**
- 超时：单次探测硬上限 **2s**；主命令不会被阻塞
- 若发现 `latest > current`，命令执行完成后会在 **stderr** 输出提示：
  - **TTY**：一行人类可读文本 `[cctl] New version available: <cur> → <new>. <hint>`
  - **非 TTY**（agent / pipe）：`[cctl:update-available]` 哨兵行 + 一行 JSON，字段契约见 §3.4
- `stdout` 不会被污染，管道下游和 `-o json` 的合约保持不变
- 禁用：`export CCTL_NO_UPDATE_CHECK=1`
- 开发构建（`version == "dev"`）不触发任何检查/提示

## 11. 当前不要依赖的命令

以下命令目前只有快捷 (shortcut) 形态，没有显式的 `create` 子命令：

- `cctl file-task create ...` — 请使用 `cctl filetask sync|export|backup|import ...` 等快捷命令

`cctl cluster` 已有 `list` / `get`，但**没有** `create` / `update` / `delete`；集群写操作仍走管控台。

如果后续需要挂出未实现的命令（例如 notebook extend、`devspace heartbeat`、job queue-position），请先在 v2 API 落地或产品确认后再加 CLI。（`cctl devspace restore` 已对接 `POST /v2/tasks/*:restore`。）

## 12. 推荐工作流

### 12.1 登录后创建训练任务

```bash
cctl login
cctl pool list --own
cctl image list --usage training --limit 10
cctl job create \
  --project my-project \
  --cluster gpu-cluster \
  --resource-pool platform \
  --image nvcr.io/nvidia/pytorch:24.01-py3 \
  --gpu 8 \
  --entry "python train.py"
```

### 11.2 用机器可读方式筛选结果

```bash
cctl job list -F name,status,cluster
cctl image list -q
cctl project list -o yaml
```

