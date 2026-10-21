---
name: cctl
description: 为本项目在 Cybertron 查询自己的资源和任务，按用户指定的 beijing、model-autosft、NORMAL 约束提交计算 job。
---

本项目操作摘要依据用户提供的 `/home/lif/.local/bin/.cursor/skills/cctl/SKILL.md`（v0.0.22）与已成功任务整理；原机器路径不要求在 devspace 存在。

**固定约束：** cluster `beijing`、资源池 `model-autosft`、priority `NORMAL`；新建 devspace 必须 GPU=0。GPU 通过 job/pytorchjob。本期 Controller 使用单 GPU job。不得停止、删除、抢占他人任务。
只写 `/user/lyh4170`；现有模型和已获许可的来源配置只读。不改 `$HOME`，也不打印认证材料。

先 `CCTL_NO_UPDATE_CHECK=1 cctl pool list --own --no-input -o json`。
训练/评测优先交给 `scripts/rsi.py` 的冻结 manifest 与防重机制；一般不手写第二套提交。
需要独立诊断 CPU job 或部署自有依赖时，使用 `docs/rsi-submission-and-files.md` 的已验证参数模板和唯一 description。

- 常规查询和创建用参数数组传 `--no-input -o json`，不要拼接包含未转义代码的 shell。
- `--entry` 是远端 shell 文本，使用 `shlex.join` 包裹每个参数。不要将 JSON 编码当 shell 转义。
- 提交后保存 ID、description、资源、manifest SHA 和产物目录；断线先查已存在的任务。
- 本机已验证的日志格式为 `cctl job logs ID --pod job-rsi-ID-main-0 --page-size 200`，不附加 `-o json` 或 `--no-input`；先用 get 核对任务身份。
- `cctl file-task` 是远端文件间操作，不是本机上传命令；当前 CLI 没有已验证的 devspace exec。用户负责首次上传本机代码和模型。

来源示例与任务 ID、共享盘和使用命令见 `docs/rsi-submission-and-files.md`。配置复制已完成，不重复复制或覆盖来源。
