#!/usr/bin/env python3
"""
SessionAware — raw signal extractor for Claude Code session transcripts.

Extracts 11 structural signatures of the "look productive" gradient from
~/.claude/projects/**/*.jsonl or a directory of session export ZIPs/JSONLs.

Output: report_local.json  — full per-session detail (never upload this)
        report_upload.json — anonymised aggregate counts only (safe to share)

Usage:
    python sessionaware.py                        # scans default paths
    python sessionaware.py --dir /path/to/exports # scans a directory
    python sessionaware.py --file session.jsonl   # single file
    python sessionaware.py --help

No third-party dependencies. Requires Python 3.8+.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Signature IDs and human labels
# ---------------------------------------------------------------------------
SIGNATURES: dict[str, str] = {
    "S3":  "Redundant read — reading a file already read with no intervening edit",
    "S4":  "Turn split — consecutive assistant turns with no user turn between",
    "S5":  "Unanchored TodoWrite — TodoWrite call not followed by any action",
    "C3":  "Bash ping-pong — identical Bash command re-run (already had result)",
    "S8":  "Result restatement — assistant text reproduces prior tool result",
    "S9":  "Plan-execute echo — narrates intent, acts, narrates completion",
    "S10": "Redundant search — identical Grep/Glob re-run with no new writes",
    "S12": "Re-plan without delta — TodoWrite re-run with no new tool results between",
    "S13": "Scope declaration — lists files/steps immediately before doing them",
    "S14": "Acknowledgment without transition — filler turn with no subsequent action",
    "S15": "Repeat failed tool call — same tool+input re-issued after an error result",
}

# ---------------------------------------------------------------------------
# JSONL parsing
# ---------------------------------------------------------------------------

def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def _parse_content(content: Any) -> tuple[list[dict], list[str], list[str]]:
    """Return (tools, text_blocks, result_blocks) from a message content field."""
    tools: list[dict] = []
    texts: list[str] = []
    results: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "tool_use":
                tools.append({"name": block.get("name", ""), "input": block.get("input", {})})
            elif bt == "text":
                texts.append(block.get("text", ""))
            elif bt == "tool_result":
                raw = block.get("content", "")
                results.append(str(raw) if not isinstance(raw, str) else raw)
    elif isinstance(content, str):
        texts.append(content)
    return tools, texts, results


def load_messages(path: Path) -> list[dict]:
    """Load deduplicated user+assistant messages from a JSONL file."""
    seen: set[str] = set()
    msgs: list[dict] = []
    for obj in _iter_jsonl(path):
        t = obj.get("type")
        if t not in ("user", "assistant"):
            continue
        uid = obj.get("uuid") or obj.get("id") or ""
        if uid and uid in seen:
            continue
        if uid:
            seen.add(uid)
        tools, texts, results = _parse_content(obj.get("message", {}).get("content", ""))
        msgs.append({
            "role": t,
            "tools": tools,
            "texts": texts,
            "results": results,
            "text": " ".join(texts),
            "ts": obj.get("isoTimestamp", ""),
        })
    return msgs

# ---------------------------------------------------------------------------
# Signal extractors — one function per signature
# ---------------------------------------------------------------------------

def extract_S3(msgs: list[dict]) -> int:
    """Redundant read: Read on path P where P was already Read with no Edit/Write on P between."""
    reads: dict[str, int] = {}   # path -> last read index
    edits: dict[str, int] = {}   # path -> last edit index
    count = 0
    for i, m in enumerate(msgs):
        for t in m["tools"]:
            name = t["name"]
            fp = t["input"].get("file_path", "")
            if not fp:
                continue
            if name == "Read":
                if fp in reads:
                    last_edit = edits.get(fp, -1)
                    if last_edit < reads[fp]:
                        count += 1
                reads[fp] = i
            elif name in ("Edit", "Write"):
                edits[fp] = i
    return count


def extract_S4(msgs: list[dict]) -> int:
    """Turn split: consecutive assistant turns (no user turn between)."""
    count = 0
    for i in range(1, len(msgs)):
        if msgs[i]["role"] == "assistant" and msgs[i - 1]["role"] == "assistant":
            count += 1
    return count


def extract_S5(msgs: list[dict]) -> int:
    """Unanchored TodoWrite: TodoWrite-only turn not followed by a non-TodoWrite tool call."""
    count = 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        names = [t["name"] for t in m["tools"]]
        if "TodoWrite" not in names:
            continue
        if any(n != "TodoWrite" for n in names):
            continue  # anchored in same turn
        # look ahead for a non-TodoWrite tool in next assistant turns (before next user turn)
        anchored = False
        for j in range(i + 1, min(i + 5, len(msgs))):
            if msgs[j]["role"] == "user":
                break
            jnames = [t["name"] for t in msgs[j]["tools"]]
            if any(n != "TodoWrite" for n in jnames):
                anchored = True
                break
        if not anchored:
            count += 1
    return count


def extract_C3(msgs: list[dict]) -> int:
    """Bash ping-pong: identical Bash command run more than once."""
    cmd_counts: Counter[str] = Counter()
    for m in msgs:
        if m["role"] != "assistant":
            continue
        for t in m["tools"]:
            if t["name"] == "Bash":
                cmd = t["input"].get("command", "").strip()[:200]
                cmd_counts[cmd] += 1
    return sum(v - 1 for v in cmd_counts.values() if v > 1)


def extract_S8(msgs: list[dict]) -> int:
    """Result restatement: assistant text-only turn whose content substantially overlaps
    with the immediately preceding tool result."""
    count = 0
    for i in range(1, len(msgs)):
        m = msgs[i]
        prev = msgs[i - 1]
        if m["role"] != "assistant" or m["tools"]:
            continue
        if not m["text"].strip():
            continue
        # collect tool results from previous user turn
        result_text = " ".join(prev["results"]).lower()
        if not result_text:
            continue
        asst_words = set(m["text"].lower().split())
        result_words = set(result_text.split())
        if not result_words:
            continue
        overlap = len(asst_words & result_words) / max(1, min(len(asst_words), len(result_words)))
        if overlap > 0.55:
            count += 1
    return count


def extract_S9(msgs: list[dict]) -> int:
    """Plan-execute echo: text-only → tool-call → text-only sandwich (no user turns between)."""
    count = 0
    i = 0
    while i < len(msgs) - 2:
        a = msgs[i]
        b = msgs[i + 1]
        c = msgs[i + 2]
        if (a["role"] == "assistant" and not a["tools"] and a["text"].strip()
                and b["role"] == "assistant" and b["tools"]
                and c["role"] == "assistant" and not c["tools"] and c["text"].strip()):
            count += 1
            i += 3
        else:
            i += 1
    return count


def extract_S10(msgs: list[dict]) -> int:
    """Redundant search: identical Grep/Glob pattern re-run with no new writes between."""
    searches: dict[str, int] = {}   # pattern -> last search index
    last_write = -1
    count = 0
    for i, m in enumerate(msgs):
        for t in m["tools"]:
            name = t["name"]
            if name in ("Edit", "Write"):
                last_write = i
            elif name in ("Grep", "Glob"):
                pat = str(t["input"].get("pattern", "") or t["input"].get("pattern", ""))[:120]
                if pat in searches and searches[pat] > last_write:
                    count += 1
                searches[pat] = i
    return count


def extract_S12(msgs: list[dict]) -> int:
    """Re-plan without delta: TodoWrite re-run between same user turns with no new tool results."""
    count = 0
    todo_since_user = 0
    results_since_todo = 0
    for m in msgs:
        if m["role"] == "user":
            todo_since_user = 0
            results_since_todo = 0
        else:
            has_todo = any(t["name"] == "TodoWrite" for t in m["tools"])
            has_other = any(t["name"] != "TodoWrite" for t in m["tools"])
            if has_other:
                results_since_todo += 1
            if has_todo:
                if todo_since_user > 0 and results_since_todo == 0:
                    count += 1
                todo_since_user += 1
                results_since_todo = 0
    return count


_SCOPE_RE = re.compile(
    r"\b(i.?ll\s+(need\s+to\s+)?(modify|update|change|edit|fix|add|remove|create|delete|check)\b"
    r"|i\s+need\s+to\b"
    r"|we.?ll\s+need\s+to\b"
    r"|the\s+(following|these)\s+(files?|steps?|changes?)\b"
    r"|first[,\s]+i.?ll\b"
    r"|let\s+me\s+(now\s+)?(modify|update|check|read|fix|add|remove|create))",
    re.IGNORECASE,
)

def extract_S13(msgs: list[dict]) -> int:
    """Scope declaration: text-only assistant turn matching planning language,
    immediately followed by an assistant turn with Edit/Write tools."""
    count = 0
    for i in range(len(msgs) - 1):
        m = msgs[i]
        nxt = msgs[i + 1]
        if m["role"] != "assistant" or m["tools"]:
            continue
        if not _SCOPE_RE.search(m["text"]):
            continue
        if nxt["role"] == "assistant" and any(t["name"] in ("Edit", "Write") for t in nxt["tools"]):
            count += 1
    return count


_ACK_RE = re.compile(
    r"^\s*(got\s+it|understood|noted|makes?\s+sense|sure|okay|ok|right|"
    r"i\s+see|i\s+understand|fair\s+enough|absolutely|of\s+course|"
    r"acknowledged|confirm(ed)?)[.!,\s]*$",
    re.IGNORECASE,
)

def extract_S14(msgs: list[dict]) -> int:
    """Acknowledgment without transition: short text-only turn matching filler phrases,
    not followed by a tool call in the next assistant turn before the next user turn."""
    count = 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or m["tools"]:
            continue
        if not _ACK_RE.match(m["text"].strip()):
            continue
        # check: is the next assistant action substantive?
        has_action = False
        for j in range(i + 1, min(i + 4, len(msgs))):
            if msgs[j]["role"] == "user":
                break
            if msgs[j]["tools"]:
                has_action = True
                break
        if not has_action:
            count += 1
    return count

# ---------------------------------------------------------------------------
# Per-session analysis
# ---------------------------------------------------------------------------

_ERROR_RE = re.compile(
    r"\b(error|exception|traceback|exit code [1-9]|not found|no such file|"
    r"failed|failure|cannot|could not|unable to|invalid|unexpected|denied|"
    r"permission|unrecognized|does not exist|timed out|timeout)\b",
    re.IGNORECASE,
)

def _tool_fingerprint(tool: dict) -> str:
    """Stable key for a tool call: name + sorted input params (truncated)."""
    name = tool.get("name", "")
    inp = tool.get("input", {})
    parts = sorted(f"{k}={str(v)[:120]}" for k, v in inp.items())
    return name + "|" + "|".join(parts)


def extract_S15(msgs: list[dict]) -> int:
    """Repeat failed tool call: tool re-issued with same name+input after an error result
    appeared between the two calls. Counts each redundant re-issue, not each error."""
    count = 0
    # track: fingerprint -> (last_call_index, error_seen_since_last_call)
    state: dict[str, dict] = {}

    for i, m in enumerate(msgs):
        if m["role"] == "user":
            # scan tool results for error markers
            for res in m["results"]:
                if _ERROR_RE.search(res):
                    # mark all previously seen fingerprints as having an error after them
                    for fp in state:
                        state[fp]["error_after"] = True
        elif m["role"] == "assistant":
            for t in m["tools"]:
                fp = _tool_fingerprint(t)
                if fp in state and state[fp].get("error_after"):
                    count += 1
                state[fp] = {"idx": i, "error_after": False}

    return count


EXTRACTORS = {
    "S3":  extract_S3,
    "S4":  extract_S4,
    "S5":  extract_S5,
    "C3":  extract_C3,
    "S8":  extract_S8,
    "S9":  extract_S9,
    "S10": extract_S10,
    "S12": extract_S12,
    "S13": extract_S13,
    "S14": extract_S14,
    "S15": extract_S15,
}


def analyze_session(path: Path) -> dict:
    msgs = load_messages(path)
    if not msgs:
        return {}
    asst = sum(1 for m in msgs if m["role"] == "assistant")
    user = sum(1 for m in msgs if m["role"] == "user")
    signals = {sid: fn(msgs) for sid, fn in EXTRACTORS.items()}
    total_signals = sum(signals.values())
    return {
        "file": str(path),
        "assistant_turns": asst,
        "user_turns": user,
        "signals": signals,
        "total_signals": total_signals,
        "signal_rate": round(total_signals / max(1, asst), 4),
    }

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_jsonl_files(search_dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.jsonl"):
            found.append(p)
    return found


def extract_zip_jsonl(zip_path: Path, tmp_dir: Path) -> list[Path]:
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith(".jsonl"):
                target = tmp_dir / Path(name).name
                target.write_bytes(zf.read(name))
                extracted.append(target)
    return extracted


def find_all_targets(args: argparse.Namespace) -> list[Path]:
    targets: list[Path] = []

    if args.file:
        p = Path(args.file)
        if p.suffix == ".zip":
            import tempfile
            tmp = Path(tempfile.mkdtemp())
            targets.extend(extract_zip_jsonl(p, tmp))
        else:
            targets.append(p)
        return targets

    search_dirs: list[Path] = []
    if args.dir:
        search_dirs.append(Path(args.dir))
    else:
        # default: ~/.claude/projects and common export locations
        home = Path.home()
        search_dirs += [
            home / ".claude" / "projects",
            home / "Downloads",
        ]

    for d in search_dirs:
        if not d.exists():
            continue
        # direct jsonl files
        for p in d.rglob("*.jsonl"):
            targets.append(p)
        # zip exports
        for p in d.glob("session-export-*.zip"):
            import tempfile
            tmp = Path(tempfile.mkdtemp())
            try:
                targets.extend(extract_zip_jsonl(p, tmp))
            except Exception:
                pass

    return list(set(targets))

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_local_report(results: list[dict]) -> dict:
    return {
        "version": __version__,
        "sessions_analyzed": len(results),
        "sessions": results,
        "aggregate": _aggregate(results),
    }


def build_upload_report(results: list[dict]) -> dict:
    """Anonymised aggregate only — no paths, no text, no identifiers."""
    agg = _aggregate(results)
    # hash session count as a consistency check without revealing count exactly
    n = len(results)
    return {
        "version": __version__,
        "session_count_hash": hashlib.sha256(str(n).encode()).hexdigest()[:12],
        "aggregate": agg,
        "per_session": [
            {
                "assistant_turns": r["assistant_turns"],
                "user_turns": r["user_turns"],
                "signals": r["signals"],
                "total_signals": r["total_signals"],
                "signal_rate": r["signal_rate"],
            }
            for r in results
        ],
    }


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {}
    total_asst = sum(r["assistant_turns"] for r in results)
    totals = {sid: sum(r["signals"].get(sid, 0) for r in results) for sid in SIGNATURES}
    rates = {sid: round(v / max(1, total_asst), 4) for sid, v in totals.items()}
    return {
        "total_assistant_turns": total_asst,
        "total_user_turns": sum(r["user_turns"] for r in results),
        "signal_totals": totals,
        "signal_rates_per_assistant_turn": rates,
        "overall_signal_rate": round(sum(totals.values()) / max(1, total_asst), 4),
    }

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SessionAware — extract look-productive gradient signals from Claude Code transcripts"
    )
    parser.add_argument("--file", help="Single .jsonl or .zip file to analyze")
    parser.add_argument("--dir", help="Directory to scan for .jsonl files and session-export ZIPs")
    parser.add_argument("--out", default=".", help="Output directory for reports (default: .)")
    parser.add_argument("--no-upload", action="store_true", help="Skip generating report_upload.json")
    args = parser.parse_args()

    targets = find_all_targets(args)
    if not targets:
        print("No JSONL files found. Use --file or --dir to specify a location.", file=sys.stderr)
        sys.exit(1)

    print(f"SessionAware {__version__} — analyzing {len(targets)} transcript(s)...")

    results = []
    for i, t in enumerate(targets, 1):
        print(f"  [{i}/{len(targets)}] {t.name}", end="", flush=True)
        r = analyze_session(t)
        if r:
            results.append(r)
            print(f"  asst={r['assistant_turns']}  signals={r['total_signals']}  rate={r['signal_rate']}")
        else:
            print("  (skipped — no messages)")

    if not results:
        print("No sessions produced output.", file=sys.stderr)
        sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    local_path = out / "report_local.json"
    local_path.write_text(json.dumps(build_local_report(results), indent=2))
    print(f"\nLocal report:  {local_path}")

    if not args.no_upload:
        upload_path = out / "report_upload.json"
        upload_path.write_text(json.dumps(build_upload_report(results), indent=2))
        print(f"Upload report: {upload_path}")

    # print summary to stdout
    agg = _aggregate(results)
    print(f"\n{'='*60}")
    print(f"SESSIONS: {len(results)}   ASSISTANT TURNS: {agg['total_assistant_turns']}")
    print(f"{'='*60}")
    print(f"{'SIGNAL':<6}  {'COUNT':>7}  {'RATE/TURN':>10}  DESCRIPTION")
    print(f"{'-'*60}")
    for sid, label in SIGNATURES.items():
        count = agg["signal_totals"].get(sid, 0)
        rate = agg["signal_rates_per_assistant_turn"].get(sid, 0.0)
        short = label.split("—")[0].strip()
        print(f"{sid:<6}  {count:>7}  {rate:>10.4f}  {short}")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<6}  {sum(agg['signal_totals'].values()):>7}  {agg['overall_signal_rate']:>10.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
