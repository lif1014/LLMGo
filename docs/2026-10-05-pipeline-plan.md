# LLMGo! 2026-10-05：双模型中国象棋可观察对局

这一版只做一件事：两个座位共用 Qwen3.5-4B，在网页上下中国象棋，并把每一步的完整思考、正式输出、实际走子和拒绝原因留下来。

## 这一版要看到什么

- 棋盘长得像网上的中国象棋：圆形棋子、九宫、楚河汉界
- 红方和黑方的思考分开放，互不可见
- 本局和全部对局的首步合法率、错误类型，直接显示在网页上
- 录像是开关。打开以后保存逐手 PNG 和 GIF，思考全文写进画面

明确不做：训练、Memory、Meta-Agent、Pikafish 对手、Elo。这些以后接在同一套接口上。

## 机器和 vLLM

| 项目 | 现状 |
| --- | --- |
| 模型 | `/home/test/testdata/models/Qwen3.5-4B` |
| 当前机器 | 主机名 `g28`，8 张 A800。写这份实现时显存几乎占满，所以没有启动模型 |
| 可用镜像 | 本机 `vllm/vllm-openai:v0.20.1`，代码里有 `qwen3_5`，也有 `--language-model-only` 和 `--reasoning-parser qwen3` |
| g84 | 有 `vllm/vllm-openai:v0.11.0`、`lyh/torch29-vllm012-cu128`（vLLM 0.12.0）、`posttrainbench-vllm-debug`（也是 0.11.0）。这些镜像里没有 `qwen3_5`，不能用来起这个 4B |
| 没有 | 当前 Python 环境里的 vLLM、系统里的 ffmpeg |

推理和对局分成两个进程，只通过本机 HTTP 通信。启动参数：

```shell
vllm serve /home/test/testdata/models/Qwen3.5-4B \
  --max-model-len 16384 \
  --reasoning-parser qwen3 \
  --language-model-only
```

单步生成上限是 12288 token，给提示留出余量。网页、jsonl 和录像都不裁思考。如果模型自己碰到长度上限，`finish_reason` 记为 `length`，已经生成的文字原样保留。

## 双方互不影响

每次请求只有两条消息：这一方的系统提示，和当前公开局面。不附带上一手的对话，更不附带对方的思考、原文或重试原因。

两边共享的只有棋盘和已经落下的走法。重试反馈写着「只有你自己看得到」，并且只放进这一方的下一次提示。红方和黑方是两个 Player 对象，温度也可以分别设置。

## 棋盘和输出

内部坐标用 ICCS：红方在下，a–i 从左到右，0–9 从下到上。红方右炮平中是 `h2e2`，中文记谱由程序译成「炮二平五」。

给模型的文本棋盘用 `（帅）` 表示红方、`【将】` 表示黑方，第 5 段和第 4 段之间写楚河汉界。网页和录像则画真正的圆形棋子。

正式输出：

```text
<move>h2e2</move>
```

解析不到 ICCS 时，再试中文记谱。每步最多 1 次首发加 2 次重试。错误类型包括 `parse_fail`、`empty_origin`、`not_own_piece`、`illegal_geometry`、`king_face`、`leaves_check`、`request_error`。三次都不合法，这一方判负。首步合法率和重试后的结果分开统计。

默认不把合法着法列表放进提示。网页上可以改成对照模式。

## 模块

```text
env        规则、FEN、中文记谱、文本棋盘
runtime    每个座位自己的提示和 vLLM / 脚本玩家
runner     对局循环
store      runs/<id>/events.jsonl
eval       summary.json，以及多局汇总
server     网页
render     圆形棋子画面和 GIF
```

依赖只向下：网页调用对局循环，对局循环调用规则和玩家。以后的 Pikafish、训练、Memory、Meta-Agent 分别做成新的 Player，或来读 jsonl。

## 网页怎么访问

网页就是这台服务器上的一个进程，监听 `0.0.0.0:7860`。不需要再部署到别的网站。

- 本机：`http://127.0.0.1:7860`
- 自己的电脑：`ssh -L 7860:127.0.0.1:7860 <用户>@g28`，然后打开同一个地址
- 内网：`http://g28:7860`

启动：

```shell
bash scripts/run_web.sh
bash scripts/serve_vllm.sh
```
