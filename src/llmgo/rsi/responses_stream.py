"""Repair observed Chat-to-Responses reasoning lifecycle omissions.

No model text, call ID, tool arguments, usage or terminal status is changed.
Original converter events are recorded separately by the bridge.
"""
from collections import Counter
from importlib.metadata import version

STREAM_VERSION = "reasoning-lifecycle-v1"


class StreamProtocolError(ValueError):
    pass


def prepare_chat_iterator(iterator):
    """Defer item initialization on empty/role-only chunks in pinned LiteLLM.

    Patch this response instance only, before iteration. Otherwise an empty
    first chunk permanently selects a message item, and later reasoning deltas
    reference an ID the converter never started. No installed file is changed;
    empty chunks still pass through its accumulation/usage handling.
    """
    cls = type(iterator)
    if (cls.__module__ != "litellm.responses.litellm_completion_transformation.streaming_iterator"
            or cls.__name__ != "LiteLLMCompletionStreamingIterator" or version("litellm") != "1.105.2"):
        return False
    original = iterator._ensure_output_item_for_chunk

    def ensure(chunk):
        if not iterator.sent_output_item_added_event and chunk.choices:
            choice = chunk.choices[0]
            delta = choice.delta
            meaningful = any(getattr(delta, field, None) for field in
                ("content", "reasoning_content", "thinking_blocks", "tool_calls", "function_call", "refusal", "annotations"))
            if not meaningful and choice.finish_reason is None:
                return
        return original(chunk)

    iterator._ensure_output_item_for_chunk = ensure
    return True


class ReasoningStream:
    def __init__(self):
        self.sequence = -1
        self.reasoning = {}
        self.parts = set()
        self.closed = set()
        self.repairs = []
        self.counts = Counter()

    def _number(self, event):
        # The pinned converter omits most sequence numbers and occasionally
        # resets them. Keep a monotone sequence even when inserting a start event.
        event = dict(event)
        supplied = event.get("sequence_number")
        expected = self.sequence+1
        self.sequence = max(expected, supplied) if type(supplied) is int else expected
        if supplied != self.sequence or type(supplied) is not int:
            self.counts["sequence_number"] += 1
        event["sequence_number"] = self.sequence
        return event

    def feed(self, event):
        if not isinstance(event, dict):
            raise StreamProtocolError("Responses stream event must be an object")
        event = dict(event)
        self.repairs = []
        output = []
        kind = event.get("type", "")
        if kind == "response.output_item.added" and event.get("item", {}).get("type") == "reasoning":
            item = dict(event["item"])
            item_id, index = item.get("id"), event.get("output_index")
            if not isinstance(item_id, str) or type(index) is not int or index < 0:
                raise StreamProtocolError("Reasoning item start lacks a valid ID/output_index")
            if item_id in self.reasoning:
                raise StreamProtocolError("Duplicate reasoning item start")
            if item.get("summary") is None:
                item["summary"] = []
                event["item"] = item
                self.counts["reasoning_summary_defaulted"] += 1
                self.repairs.append({"repair":"reasoning_summary_defaulted", "item_id":item_id,
                                     "output_index":index})
            elif not isinstance(item["summary"], list):
                raise StreamProtocolError("Reasoning summary must be an array")
            self.reasoning[item_id] = index
        elif kind.startswith("response.reasoning_summary_"):
            item_id, index, summary = event.get("item_id"), event.get("output_index"), event.get("summary_index")
            if (not isinstance(item_id, str) or item_id not in self.reasoning
                    or self.reasoning[item_id] != index or item_id in self.closed
                    or type(summary) is not int or summary < 0):
                # Do not invent identities for orphan or post-completion deltas.
                raise StreamProtocolError("Reasoning summary event has no matching active item/index")
            key = (item_id, summary)
            if kind == "response.reasoning_summary_part.added":
                if key in self.parts:
                    raise StreamProtocolError("Duplicate reasoning summary part start")
                self.parts.add(key)
            elif key not in self.parts:
                self.parts.add(key)
                added = {"type":"response.reasoning_summary_part.added", "item_id":item_id,
                    "output_index":index, "summary_index":summary, "part":{"type":"summary_text", "text":""}}
                output.append(self._number(added))
                self.counts["summary_part_added"] += 1
                self.repairs.append({"repair":"summary_part_added", "item_id":item_id,
                                     "output_index":index, "summary_index":summary})
        elif kind == "response.output_item.done":
            item = event.get("item", {})
            if item.get("type") == "reasoning":
                self.closed.add(item.get("id"))
        output.append(self._number(event))
        return output
