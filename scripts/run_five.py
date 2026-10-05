"""连续下五局，供网页回放。"""

import json
import time
import urllib.request

BASE = "http://127.0.0.1:7860"


def post(path, payload):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=30) as response:
        return json.load(response)


def main():
    ids = []
    for index in range(1, 6):
        created = post(
            "/api/games",
            {
                "backend": "vllm",
                "max_plies": 300,
                "max_retries": 2,
                "record_video": False,
                "prompt_policy": "blind",
                "temperature": 0.9,
                "red_temperature": 0.9,
                "black_temperature": 0.9,
                "max_tokens": 16384,
            },
        )
        game_id = created["id"]
        ids.append(game_id)
        print(f"start {index}/5 {game_id}", flush=True)
        while True:
            time.sleep(5)
            info = get(f"/api/games/{game_id}")
            if not info.get("live"):
                summary = info.get("summary") or {}
                print(
                    f"done {game_id} result={summary.get('result')} reason={summary.get('result_reason')} "
                    f"plies_hint={summary.get('first_try_total')}",
                    flush=True,
                )
                break
    print("IDS " + " ".join(ids), flush=True)


if __name__ == "__main__":
    main()
