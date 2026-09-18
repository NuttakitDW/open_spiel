#!/usr/bin/env python3
"""Protocol fixtures for the embedded poker-arena HU fixed-limit executable."""
import json
import subprocess
import sys


def message(tag, **fields):
    return {"t": tag, **fields}


HELLO = message(
    "hello", proto=1, game_id="holdem-fl", seat_count=2,
    starting_stack=2000, timeout_ms=1000,
    stakes={"kind": "blinds", "small_blind": 1, "big_blind": 2, "ante": 0},
    betting={"kind": "fixed-limit", "raise_cap": 4},
)
FIXTURE = [
    message("future-tag", future="ignored"),
    HELLO,
    message("hand-start", seat=0, hand_no=1),
    message("event", ev={"event": "hand-start", "hand_no": 1, "button": 0,
                         "stacks": [2000, 2000]}),
    message("event", ev={"event": "post", "seat": 0,
                         "kind": "small-blind", "amount": 1, "all_in": False}),
    message("event", ev={"event": "post", "seat": 1,
                         "kind": "big-blind", "amount": 2, "all_in": False}),
    message("event", ev={"event": "street-start", "street": 0,
                         "label": "preflop"}),
    message("event", ev={"event": "deal-hole", "seat": 0,
                         "cards": ["As", "Ks"], "count": 2}),
    message("event", ev={"event": "deal-hole", "seat": 1,
                         "cards": [], "count": 2}),
    message("act", seat=0, hand_no=1,
            decision={"kind": "wager", "fold": True, "check": False,
                      "call": 1, "raise": {"min_to": 4, "max_to": 4}}),
    message("match-end"),
]


def run(command, rows):
    data = "\n".join(json.dumps(row) if isinstance(row, dict) else row
                     for row in rows) + "\n"
    return subprocess.run(command, input=data, text=True,
                          capture_output=True, timeout=20)


def main(command):
    result = run(command, ["  ", *FIXTURE])
    assert result.returncode == 0, result.stderr
    lines = [json.loads(line) for line in result.stdout.splitlines()]
    assert lines[0] == {"t": "join"}
    assert lines[1]["t"] == "action"
    assert lines[1]["action"]["kind"] in ("fold", "call", "raise")
    assert "trained_hits=1 unseen=0 incompatible=0" in result.stderr, result.stderr

    bad = json.loads(json.dumps(HELLO))
    bad["seat_count"] = -1
    result = run(command, [bad])
    assert result.returncode != 0 and not result.stdout
    bad["seat_count"] = 2.5
    result = run(command, [bad])
    assert result.returncode != 0 and not result.stdout
    result = run(command, ["x" * 65537])
    assert result.returncode != 0 and "oversized" in result.stderr
    bad = json.loads(json.dumps(HELLO))
    bad["game_id"] = "holdem-nl"
    result = run(command, [bad])
    assert result.returncode != 0 and not result.stdout

    shallow = json.loads(json.dumps(HELLO))
    shallow["starting_stack"] = 20
    shallow_rows = list(FIXTURE)
    shallow_rows[1] = shallow
    shallow_rows[3] = message("event", ev={"event": "hand-start", "hand_no": 1,
                                             "button": 0, "stacks": [20, 20]})
    result = run(command, shallow_rows)
    assert result.returncode == 0 and "incompatible=1" in result.stderr, result.stderr
    print("wire fixtures passed; embedded policy hit and fallback verified")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: arena_hulhe_bot_test.py BOT [ARGS...]")
    main(sys.argv[1:])
