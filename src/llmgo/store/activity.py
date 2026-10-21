"""Bounded display projections. Raw journals remain the authoritative evidence."""
from __future__ import annotations

import json
from itertools import islice

VERSION = 1


def clipped(value, size=1800):
    text = value if isinstance(value, str) else str(value or "")
    if len(text) <= size:
        return text
    return text[:size//2] + "\n[显示已截取；原文按需读取]\n" + text[-size//2:]


def compact(value, budget=3200):
    """Limit traversal as well as output; never stringify a whole large payload."""
    remaining, nodes = [budget], [80]
    def visit(obj, depth=0):
        nodes[0] -= 1
        if remaining[0] <= 0 or nodes[0] < 0 or depth > 4:
            return "[…]"
        if isinstance(obj, str):
            text = clipped(obj, min(600, remaining[0]))
            remaining[0] -= len(text)
            return text
        if isinstance(obj, dict):
            return {str(k)[:100]: visit(v, depth+1) for k, v in islice(obj.items(), 16)}
        if isinstance(obj, (list, tuple)):
            return [visit(v, depth+1) for v in obj[:12]]
        return obj if obj is None or isinstance(obj, (int, float, bool)) else clipped(str(obj), 100)
    return visit(value)


def project(event):
    kind = event.get("kind", "unknown")
    if kind == "state":
        return None  # repeated full campaign snapshots are not activity messages
    body = event.get("body", {})
    body = body if isinstance(body, dict) else {"value": body}
    view = {"version": VERSION, "id": event.get("id"), "ts": event.get("ts"), "kind": kind,
            "channel": "research", "category": "event", "title": kind, "text": "",
            "details": compact(body)}
    if kind.startswith("meta_"):
        message = body.get("message", {})
        message = message if isinstance(message, dict) else {}
        params = message.get("params", {})
        params = params if isinstance(params, dict) else {}
        item = params.get("item", {})
        item = item if isinstance(item, dict) else {}
        turn = params.get("turn", {})
        turn = turn if isinstance(turn, dict) else {}
        method = message.get("method", "")
        lower = method.lower()
        view.update(channel="coach", method=method, thread_id=params.get("threadId", body.get("thread_id")),
                    turn_id=params.get("turnId", turn.get("id") if isinstance(turn, dict) else None),
                    item_id=params.get("itemId", item.get("id")))
        label = item.get("type", "")
        if kind == "meta_heartbeat":
            view.update(category="heartbeat", title="教练连接心跳")
        elif kind in ("meta_failure", "meta_stderr"):
            view.update(category="error", title="教练错误/诊断", text=clipped(body.get("error", body.get("text", message.get("text", "")))))
        elif kind == "meta_status":
            view.update(category="lifecycle", title="教练状态 · " + str(body.get("status", "")))
        elif kind == "meta_send":
            view.update(category="request", title="发送 · " + method)
        elif method == "error" or message.get("error") is not None:
            error = message.get("error", params.get("error", {}))
            text = error.get("message", "") if isinstance(error, dict) else error
            view.update(category="error", title="教练协议错误", text=clipped(text))
        elif "tokenusage" in lower:
            view.update(category="usage", title="教练 Token 用量", details=compact(params), usage=compact(params.get("tokenUsage", {})))
        elif "delta" in lower and isinstance(params.get("delta"), str):
            stream = "reasoning" if "reasoning" in lower else "tool" if "commandexecution" in lower else "content"
            view.update(category="text" if stream != "tool" else "tool", stream=stream,
                        mode="append", title={"reasoning": "可见思考摘要", "tool": "命令输出", "content": "教练文字"}[stream],
                        text=clipped(params["delta"]), text_truncated=len(params["delta"]) > 1800, details={})
        elif method in ("item/started", "item/completed"):
            state = "开始" if method.endswith("started") else "完成"
            is_tool = label in ("commandExecution", "mcpToolCall", "dynamicToolCall", "fileChange", "webSearch")
            view.update(category="tool" if is_tool else "text", title=f"{state} · {label}", details=compact(item))
            if label == "agentMessage" or is_tool:
                view["stream"] = "tool" if is_tool else "content"
            text = item.get("text", item.get("aggregatedOutput", ""))
            if isinstance(text, str) and text:
                view.update(text=clipped(text), mode="replace", text_truncated=len(text) > 1800)
            command = item.get("command")
            if command:
                shown = " ".join(map(str, command[:8])) if isinstance(command, (list, tuple)) else str(command)
                view["title"] += " · " + clipped(shown, 200)
            if item.get("exitCode") not in (None, 0) or item.get("status") in ("failed", "declined"):
                view["category"] = "error"
        elif method == "turn/completed":
            view.update(category="lifecycle", title="教练轮次结束 · " + str(turn.get("status", "")))
        else:
            view["title"] = method or kind
        view["stream_key"] = ":".join(str(view.get(k) or "") for k in ("thread_id", "turn_id", "item_id", "stream"))
    elif kind in {"command_started", "command", "poll_started", "poll_finished", "platform_observation",
                  "lifecycle_transition", "observation_error", "reconciliation_required", "artifact_invalid",
                  "log_collection_error", "submission_intent", "submission_uncertain", "task_submitted", "task_reconciled", "retry_guard"}:
        view.update(channel="executor", category="poll", job_id=body.get("job_id"), task_id=body.get("task_id"))
        if kind in ("command", "command_started"):
            view.update(category="command", title=("命令开始" if kind == "command_started" else "命令返回") + " · " + clipped(" ".join(map(str, body.get("argv", [])[:8])), 200),
                        text=clipped(body.get("stderr") or body.get("stdout") or ""))
            if kind == "command":
                view["title"] += f" · 退出 {body.get('returncode', body.get('error_type', '?'))} · {body.get('seconds', 0):.2f}秒"
                if body.get("error_type") or body.get("returncode") not in (None, 0):
                    view["category"] = "error"
        elif kind == "platform_observation":
            task = body.get("task", {})
            selected = {k: task.get(k) for k in ("id", "status", "description", "createTime", "updateTime", "resourcesTotal", "priority", "cluster", "project")}
            view.update(title=f"状态查询返回 · {task.get('id', '')} {task.get('status', '')}", task_id=task.get("id"), details=compact(selected))
        elif kind in ("poll_started", "poll_finished"):
            view["title"] = f"{'轮询开始' if kind == 'poll_started' else '轮询完成'} · {body.get('job_id', '')} · 第{body.get('checks', '?')}次"
            if kind == "poll_finished":
                view["title"] += f" · {body.get('platform_status', '?')} / {body.get('status', '?')} · 下次间隔{body.get('interval_seconds', '?')}秒"
        elif "error" in kind or kind in ("submission_uncertain", "reconciliation_required", "artifact_invalid"):
            view.update(category="error", title="执行异常 · " + kind)
    else:
        labels = {"hypothesis": "提出实验假设", "decision": "实验结论", "handoff": "研究交接", "research_note": "研究笔记",
                  "proposal_invalid": "候选校验失败", "experiment_failed": "实验失败",
                  "evaluation": "评测完成", "executor_transition": "任务状态变化", "explicit_retry": "显式恢复任务"}
        view["title"] = labels.get(kind, kind)
        if kind == "evaluation":
            view["evaluation"] = compact(body)
        if kind in ("proposal_invalid", "experiment_failed"):
            view["category"] = "error"
    # Keep the total serialized projection bounded even for unusual protocol data.
    if len(json.dumps(view, ensure_ascii=False).encode()) > 18000:
        view["details"] = {"note": "大事件仅显示摘要，请按需读取原文"}
    return view
