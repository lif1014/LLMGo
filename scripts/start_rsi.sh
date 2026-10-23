#!/usr/bin/env bash
# Start/resume the controller in the user's existing CPU devspace.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
campaign=chess-autonomous-v2
web=false
upgrade=false
action=start
usage() {
  cat <<'HELP'
用法：bash scripts/start_rsi.sh [--web] [--upgrade] [--campaign NAME]
      bash scripts/start_rsi.sh --status
      bash scripts/start_rsi.sh --logs
默认检查环境和Meta连接，首次初始化，然后后台启动/恢复同名loop。
--web 同时后台启动网页，端口默认7860；重复执行不会重启已有loop。
--upgrade 在旧任务已终止且loop退出后升级现有campaign，保留全部历史。
--status 查看持久状态；--logs 跟随Controller日志，Ctrl-C不停止loop。
可用 RSI_PYTHON 指定控制环境解释器，默认沿用已安装的远端专用环境。
HELP
}
while (($#)); do
  case "$1" in
    --web) web=true; shift ;;
    --upgrade) upgrade=true; shift ;;
    --campaign) [[ $# -ge 2 ]] || { usage; exit 2; }; campaign="$2"; shift 2 ;;
    --status) action=status; shift ;;
    --logs) action=logs; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$campaign" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$ ]] || { echo 'campaign名称无效' >&2; exit 2; }
rsi_python="${RSI_PYTHON:-/user/lyh4170/venvs/llmgo-rsi-control-20261022/bin/python}"
[[ -x "$rsi_python" ]] || { echo "控制Python不存在：$rsi_python；请在远端devspace执行。" >&2; exit 1; }
cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
site="$repo/configs/rsi/site.beijing.json"
settings="$("$rsi_python" - "$site" <<'PY'
import json,sys
from pathlib import Path
from llmgo.remote.cctl_env import cctl_environment
s=json.loads(Path(sys.argv[1]).read_text())
root=Path(s['user_root']).resolve()
if not root.is_dir():raise SystemExit('用户远端目录不存在；请在devspace执行。')
paths=[root,Path(s['campaign_root']),Path(s['codex_home']),Path(s.get('runtime_root',root/'llmgo-rsi-runtime-20261022'))]
for p in paths:
    if not p.resolve().is_relative_to(root) or '\n' in str(p):raise SystemExit('站点路径超出用户目录')
mode=s.get('executor_mode','embedded')
if mode not in ('embedded','external'):raise SystemExit('executor_mode无效')
print('\n'.join(map(str,[*paths,mode,cctl_environment().get('CCTL_SERVER','')])))
PY
)"
mapfile -t settings <<<"$settings"
user_root="${settings[0]}"
campaign_root="${settings[1]}/$campaign"
export CODEX_HOME="${settings[2]}"
runtime="${settings[3]}/launcher"
executor_mode="${settings[4]}"
if [[ -n "${settings[5]:-}" ]]; then export CCTL_SERVER="${settings[5]}"; fi
export PATH="$user_root/tools/bin:$PATH"
if [[ "$action" == status ]]; then
  exec "$rsi_python" scripts/rsi.py --site "$site" status "$campaign"
elif [[ "$action" == logs ]]; then
  exec tail -n 60 -f "$campaign_root/controller.log"
fi
command -v flock >/dev/null || { echo '缺少flock命令' >&2; exit 1; }
mkdir -p "$runtime"

start_executor() {
  [[ "$executor_mode" == external && -f "$campaign_root/campaign.json" ]] || return 0
  mkdir -p "$campaign_root/executor"
  if ! flock -n "$campaign_root/executor/service.lock" true; then
    echo 'Executor已运行'; return 0
  fi
  nohup "$rsi_python" -u scripts/rsi.py --site "$site" executor run "$campaign" \
    >>"$campaign_root/executor-service.log" 2>&1 </dev/null 9>&- &
  echo "Executor PID：$!"
}
start_web() {
  [[ "$web" == true ]] || return 0
  local port="${LLMGO_WEB_PORT:-7860}"
  [[ "$port" =~ ^[0-9]+$ && ${#port} -le 5 ]] && ((10#$port>=1 && 10#$port<=65535)) || { echo '网页端口无效' >&2; return 1; }
  if ! flock -n "$runtime/web-$port.lock" true; then
    echo "网页已运行：端口$port，访问/rsi"; return 0
  fi
  # Leave an occupied port alone; it can belong to another user task.
  if ! "$rsi_python" - "$port" <<'PY'
import socket,sys
with socket.socket() as s:
    try:s.bind(('0.0.0.0',int(sys.argv[1])))
    except OSError:raise SystemExit(1)
PY
  then
    echo "端口$port已占用，保留现有服务；可用LLMGO_WEB_PORT指定其他端口。"; return 0
  fi
  LLMGO_RSI_ROOT="${settings[1]}" PYTHON="$rsi_python" nohup \
    flock -n "$runtime/web-$port.lock" bash scripts/run_web.sh \
    >>"$runtime/web-$port.log" 2>&1 </dev/null 9>&- &
  local web_pid=$!
  sleep 1
  if ! kill -0 "$web_pid" 2>/dev/null; then
    echo "网页启动失败；查看：$runtime/web-$port.log；Controller不受影响。" >&2
    return 1
  fi
  echo "网页 PID：$web_pid；在平台映射端口$port后访问/rsi"
}

# Keep this lock in the controller child until it exits. PID files are not locks.
exec 9>"$runtime/$campaign.lock"
if ! flock -n 9; then
  echo "同名loop已在运行或启动中：$campaign"
  start_web
  exit 0
fi
if [[ -f "$campaign_root/campaign.json" ]] && ! flock -n "$campaign_root/controller.lock" true; then
  echo "同名Controller已运行：$campaign"
  start_executor
  start_web
  exit 0
fi
echo '检查控制环境、集群认证与Meta-Agent连接……'
"$rsi_python" scripts/rsi.py --site "$site" doctor --platform --codex-auth
"$rsi_python" scripts/check_meta_connection.py --site "$site" --sandbox-check
if [[ ! -f "$campaign_root/campaign.json" ]]; then
  "$rsi_python" scripts/rsi.py --site "$site" init "$campaign" >"$runtime/$campaign-init.json"
  echo '已创建campaign：L1–L4，每档10局，共40局初始基线。'
else
  if [[ "$upgrade" == true ]]; then
    "$rsi_python" scripts/rsi.py --site "$site" upgrade "$campaign" >"$runtime/$campaign-upgrade.json"
    echo "已升级现有campaign，保留旧任务、基线及参考Harness：$campaign"
  else
    echo "沿用已有campaign及冻结设置：$campaign"
  fi
fi
start_executor
nohup "$rsi_python" -u scripts/rsi.py --site "$site" run "$campaign" \
  >>"$campaign_root/controller.log" 2>&1 </dev/null &
controller_pid=$!
sleep 1
if ! kill -0 "$controller_pid" 2>/dev/null; then
  echo "Controller已退出；查看：$campaign_root/controller.log" >&2
  exit 1
fi
echo "Loop已在后台启动，PID：$controller_pid"
echo "日志：$campaign_root/controller.log"
start_web
