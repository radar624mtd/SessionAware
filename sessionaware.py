#!/usr/bin/env python3
"""
SessionAware — raw signal extractor for Claude Code session transcripts.

Extracts 10 structural signatures of the "look productive" gradient from
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
from typing import Any, Set

__version__ = "0.2.0"

# ---------------------------------------------------------------------------
# Signature IDs and human labels
# ---------------------------------------------------------------------------
SIGNATURES: dict[str, str] = {
    "S1":  "Redundant read — Read of content already in context (sub-cases: exact, subsumed, overlap, split)",
    "S2":  "Unanchored TodoWrite — TodoWrite call not followed by any action",
    "S3":  "Bash ping-pong — identical Bash command re-run (already had result)",
    "S4":  "Result restatement — assistant text reproduces prior tool result",
    "S5":  "Narration turn — text-only turn wrapping tool calls (sub-cases: echo, preamble)",
    "S6":  "Redundant search — identical Grep/Glob re-run with no new writes",
    "S7":  "Re-plan without delta — TodoWrite re-run with no new tool results between",
    "S8":  "Acknowledgment without transition — filler turn with no subsequent action",
    "S9":  "Repeat failed tool call — same tool+input re-issued after an error result",
    "S10": "Known-fix failure — tool re-fired after an error whose solution was already in context",
}

# Arbitration priority: S10 (Known-fix) > S9 (Repeat fail) > S1 (Redundant read) > ...
# High-certainty signals take precedence over overlapping heuristic signals.
SIGNAL_PRIORITY = ["S10", "S9", "S3", "S1", "S6", "S7", "S2", "S8", "S4", "S5"]

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
                tools.append({"name": block.get("name", ""), "input": block.get("input", {})
})
            elif bt == "text":
                texts.append(block.get("text", ""))
            elif bt == "tool_result":
                raw = block.get("content", "")
                results.append(str(raw) if not isinstance(raw, str) else raw)
    elif isinstance(content, str):
        texts.append(content)
    return tools, texts, results


def _agent_key(obj: dict) -> str:
    """Return a stable agent discriminator: agentId if present, else sessionId, else '_unknown'."""
    return obj.get("agentId") or obj.get("sessionId") or "_unknown"


def _topo_sort_agent(records: list[dict]) -> list[dict]:
    """Return records in causal order via BFS topological sort on parentUuid links."""
    by_uuid: dict[str, dict] = {r["uuid"]: r for r in records if r["uuid"]}
    children: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r["uuid"] and r["parentUuid"] and r["parentUuid"] in by_uuid:
            children[r["parentUuid"]].append(r["uuid"])

    roots = [r for r in records if r["uuid"] and (
        not r["parentUuid"] or r["parentUuid"] not in by_uuid
    )]
    ordered: list[dict] = []
    visited: set[str] = set()
    queue = list(roots)
    while queue:
        node = queue.pop(0)
        uid = node["uuid"]
        if uid in visited:
            continue
        visited.add(uid)
        ordered.append(node)
        for child_uid in children.get(uid, []):
            if child_uid not in visited:
                queue.append(by_uuid[child_uid])

    for r in records:
        if r["uuid"] and r["uuid"] not in visited:
            ordered.append(r)
        elif not r["uuid"]:
            ordered.append(r)

    seen_uuids: set[str] = set()
    deduped: list[dict] = []
    for r in ordered:
        uid = r["uuid"]
        if uid and uid in seen_uuids:
            continue
        if uid:
            seen_uuids.add(uid)
        deduped.append(r)
    return deduped


def load_messages(path: Path) -> list[dict]:
    """Load user+assistant messages from a JSONL file, preserving causal order."""
    raw: list[dict] = []
    for obj in _iter_jsonl(path):
        t = obj.get("type")
        if t not in ("user", "assistant"):
            continue
        tools, texts, results = _parse_content(obj.get("message", {}).get("content", ""))
        raw.append({
            "role": t,
            "tools": tools,
            "texts": texts,
            "results": results,
            "text": " ".join(texts),
            "ts": obj.get("isoTimestamp", ""),
            "uuid": obj.get("uuid") or obj.get("id") or "",
            "parentUuid": obj.get("parentUuid") or "",
            "agent_id": _agent_key(obj),
        })

    by_agent: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_agent[r["agent_id"]].append(r)

    msgs: list[dict] = []
    for agent_records in by_agent.values():
        msgs.extend(_topo_sort_agent(agent_records))
    return msgs

# ---------------------------------------------------------------------------
# Signal extractors — return set of message indices
# ---------------------------------------------------------------------------

def _read_range(tool: dict) -> tuple[int, int | None]:
    """Return (offset, limit) for a Read call; limit=None means full-file semantics."""
    inp = tool.get("input", {})
    off = inp.get("offset", 0) or 0
    lim = inp.get("limit", None)
    try:
        off = int(off)
    except (TypeError, ValueError):
        off = 0
    try:
        lim = int(lim) if lim is not None else None
    except (TypeError, ValueError):
        lim = None
    return off, lim


def _range_overlaps_merged(merged: list[tuple[int, int]], cur: tuple[int, int | None]) -> bool:
    """Check if 'cur' range overlaps with any range in 'merged' (sorted non-overlapping list)."""
    cur_off, cur_lim = cur
    if cur_lim is None:  # full file read
        return bool(merged)
    cur_end = cur_off + cur_lim
    # Simple linear check for now, can binary search if merged is large
    for m_start, m_end in merged:
        if not (cur_end <= m_start or cur_off >= m_end):
            return True
    return False

def _merge_range(merged: list[tuple[int, int]], cur: tuple[int, int | None]):
    """Merge 'cur' into 'merged' and return new sorted non-overlapping list."""
    cur_off, cur_lim = cur
    if cur_lim is None:
        return [(0, float('inf'))]
    cur_end = cur_off + cur_lim
    new_ranges = merged + [(cur_off, cur_end)]
    new_ranges.sort()
    
    if not new_ranges:
        return []
    
    res = []
    s, e = new_ranges[0]
    for ns, ne in new_ranges[1:]:
        if ns <= e:
            e = max(e, ne)
        else:
            res.append((s, e))
            s, e = ns, ne
    res.append((s, e))
    return res


def extract_S1(msgs: list[dict]) -> set[int]:
    """Redundant read — Read of content already in context."""
    hits = set()
    # Path -> merged non-overlapping ranges already read
    path_merged: dict[str, list[tuple[int, int]]] = defaultdict(list)
    # Path -> (msg_idx, range) of most recent Read (for split detection)
    last_read: dict[str, tuple[int, tuple[int, int | None]]] = {}
    last_nonread_turn: int = -1

    for msg_idx, m in enumerate(msgs):
        has_nonread = any(t["name"] != "Read" for t in m["tools"])
        if has_nonread:
            last_nonread_turn = msg_idx

        for t in m["tools"]:
            if t["name"] != "Read":
                continue
            fp = t["input"].get("file_path", "")
            if not fp:
                continue
            cur = _read_range(t)
            merged = path_merged[fp]

            # 1. exact/subsumed/overlap via merged set
            if _range_overlaps_merged(merged, cur):
                hits.add(msg_idx)
            # 2. split — disjoint but contiguous with the most recent prior Read
            elif fp in last_read:
                prev_msg_idx, prev_range = last_read[fp]
                prev_off, prev_lim = prev_range
                cur_off, _ = cur
                if (prev_lim is not None and cur_off == prev_off + prev_lim 
                    and last_nonread_turn <= prev_msg_idx):
                    hits.add(msg_idx)

            path_merged[fp] = _merge_range(merged, cur)
            if m["role"] == "assistant":
                last_read[fp] = (msg_idx, cur)

    return hits


def extract_S2(msgs: list[dict]) -> set[int]:
    """Unanchored TodoWrite: TodoWrite-only turn not followed by a non-TodoWrite tool call."""
    hits = set()
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        names = [t["name"] for t in m["tools"]]
        if "TodoWrite" not in names or any(n != "TodoWrite" for n in names):
            continue
        anchored = False
        for j in range(i + 1, min(i + 5, len(msgs))):
            if msgs[j]["role"] == "user":
                break
            if any(n != "TodoWrite" for n in [t["name"] for t in msgs[j]["tools"]]):
                anchored = True
                break
        if not anchored:
            hits.add(i)
    return hits


_PS_RE = re.compile(r"(?:\bGet-|\bSet-|\bNew-|\$env:|\.ps1\b|-Command\b|Write-Host\b)")
_CMD_RE = re.compile(r"%[A-Z_]%|\bdir\s+/[a-z]\b", re.IGNORECASE)
_BASH_RE = re.compile(r"(?:\$\{?[A-Za-z_]\w*\}?|&&|\|\||2>&1|/c/|\$\()")

def _bash_dialect(cmd: str) -> str:
    if _PS_RE.search(cmd): return "powershell"
    if _CMD_RE.search(cmd): return "cmd"
    if _BASH_RE.search(cmd): return "bash"
    return "neutral"

def _bash_path_style(cmd: str) -> str:
    if re.search(r"/c/Users", cmd, re.IGNORECASE): return "posix"
    if re.search(r"C:\\\\Users|C:\\Users", cmd): return "windows_backslash"
    if re.search(r"C:/Users", cmd): return "windows_forward"
    return "none"

def _bash_fingerprint(cmd: str) -> tuple:
    return (_bash_dialect(cmd), _bash_path_style(cmd))

def extract_S3(msgs: list[dict]) -> set[int]:
    """Failure-context repeat: a Bash call that ignores correction already in context."""
    hits = set()
    failed_fps: dict[tuple, int] = {}
    succeeded_fps: dict[tuple, int] = {}
    window = 40

    bash_calls: list[tuple] = []
    seq = 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            for t in m["tools"]:
                if t["name"] != "Bash": continue
                cmd = t["input"].get("command", "").strip()
                if not cmd: continue
                fp = _bash_fingerprint(cmd)
                err = False
                if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
                    for res in msgs[i + 1]["results"]:
                        if _ERROR_RE.search(res):
                            err = True; break
                bash_calls.append((i, seq, fp, err))
                seq += 1

    for idx, (msg_i, seq_i, fp, err) in enumerate(bash_calls):
        if fp in failed_fps:
            prior_fail_seq = failed_fps[fp]
            if seq_i - prior_fail_seq <= window:
                corrected = False
                for fp2, succ_seq in succeeded_fps.items():
                    if fp2 != fp and prior_fail_seq < succ_seq < seq_i:
                        corrected = True; break
                if not corrected:
                    prior_msg_i = bash_calls[seq_i_lookup(bash_calls, prior_fail_seq)][0]
                    for k in range(prior_msg_i + 1, msg_i):
                        if msgs[k]["role"] == "user" and not msgs[k]["results"]:
                            corrected = True; break
                if corrected:
                    hits.add(msg_i)
        if err: failed_fps[fp] = seq_i
        else: succeeded_fps[fp] = seq_i
    return hits


def seq_i_lookup(bash_calls: list[tuple], seq: int) -> int:
    for i, bc in enumerate(bash_calls):
        if bc[1] == seq: return i
    return 0


_FORWARD_RE = re.compile(
    r"\b(let\s+me|i.?ll|i\s+will|next[,\s]|now\s+(let|i)|going\s+to|"
    r"i\s+need\s+to|we.?ll|we\s+need\s+to|first[,\s]|then\s+i)\b",
    re.IGNORECASE,
)

def extract_S4(msgs: list[dict]) -> set[int]:
    """Result restatement: text-only assistant turn recap."""
    hits = set()
    for i in range(1, len(msgs)):
        m = msgs[i]; prev = msgs[i - 1]
        if m["role"] != "assistant" or m["tools"] or not m["text"].strip(): continue
        if prev["role"] != "user" or not prev["results"]:
            continue
        text = m["text"].strip()
        if _FORWARD_RE.search(text.split(".")[0]): continue
        if len(text.split()) < 6: continue
        hits.add(i)
    return hits


def extract_S5(msgs: list[dict]) -> set[int]:
    """Narration turn — text-only assistant turn wrapping tool calls."""
    hits = set()
    n = len(msgs)
    i = 0
    while i < n:
        m = msgs[i]
        if m["role"] == "assistant" and not m["tools"] and m["text"].strip():
            saw_tool = False; j = i + 1
            while j < n:
                mj = msgs[j]
                if mj["role"] == "user":
                    if mj["results"]: j += 1; continue
                    else: break
                if mj["role"] == "assistant":
                    if mj["tools"]: saw_tool = True; j += 1; continue
                    if not mj["text"].strip(): j += 1; continue
                    if saw_tool: hits.add(j)
                    break
                j += 1
            i = j
        else: i += 1

    for i in range(n - 1):
        m = msgs[i]
        if m["role"] != "assistant" or m["tools"] or not m["text"].strip(): continue
        if not _SCOPE_RE.search(m["text"]):
            continue
        for j in range(i + 1, min(i + 6, n)):
            mj = msgs[j]
            if mj["role"] == "user" and not mj["results"]:
                break
            if mj["role"] == "assistant" and mj["tools"]:
                hits.add(i); break
    return hits


def extract_S6(msgs: list[dict]) -> set[int]:
    """Redundant search: identical Grep/Glob pattern re-run with no new writes between."""
    hits = set()
    searches: dict[str, int] = {}
    last_write = -1
    for i, m in enumerate(msgs):
        for t in m["tools"]:
            name = t["name"]
            if name in ("Edit", "Write"): last_write = i
            elif name in ("Grep", "Glob"):
                inp = t["input"]
                pat = str(inp.get("pattern", "") or inp.get("query", ""))[:120]
                if not pat: continue
                if pat in searches and searches[pat] > last_write:
                    hits.add(i)
                searches[pat] = i
    return hits


def extract_S7(msgs: list[dict]) -> set[int]:
    """Re-plan churn: TodoWrite re-issued between the same real user prompt turns."""
    hits = set()
    todos_in_block = 0
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            if not m["results"]:
                todos_in_block = 0
            continue
        if any(t["name"] == "TodoWrite" for t in m["tools"]):
            if todos_in_block > 0: hits.add(i)
            todos_in_block += 1
    return hits


_SCOPE_RE = re.compile(
    r"\b(i.?ll\s+(need\s+to\s+)?\w+"
    r"|i\s+need\s+to\b"
    r"|i\s+will\s+\w+"
    r"|we.?ll\s+(need\s+to\s+)?\w+"
    r"|the\s+(following|these)\s+(files?|steps?|changes?|things?)\b"
    r"|first[,\s:]+(i.?ll|let\s+me)\b"
    r"|(now\s+)?let\s+me\s+\w+"
    r"|here.?s\s+(the\s+)?(plan|situation|what|my\s+plan)\b"
    r"|my\s+plan\s+is\b"
    r"|going\s+to\s+\w+"
    r"|^\s*\d+\.\s+\w+)",
    re.IGNORECASE | re.MULTILINE,
)

_ACK_RE = re.compile(
    r"^\s*(got\s+it|understood|noted|makes?\s+sense|sure|okay|ok|right|"
    r"i\s+see|i\s+understand|fair\s+enough|absolutely|of\s+course|"
    r"acknowledged|confirm(ed)?|excellent|perfect|great|good|nice|"
    r"no\s+response\s+requested|you.?re\s+right|you\s+are\s+correct|"
    r"exactly|indeed|yes[.,!]?|correct[.,!]?)\b",
    re.IGNORECASE,
)

def extract_S8(msgs: list[dict]) -> set[int]:
    """Acknowledgment without transition: filler text turn."""
    hits = set()
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or m["tools"]: continue
        text = m["text"].strip()
        if not _ACK_RE.match(text): continue

        # Substance Filter:
        # 1. Brief enough to be filler (< 16 words)
        # 2. No technical formatting (markdown bold/code)
        words = text.split()
        if len(words) > 15: continue
        if "```" in text or "**" in text: continue

        has_action = False
        for j in range(i + 1, min(i + 4, len(msgs))):
            if msgs[j]["role"] == "user": break
            if msgs[j]["tools"]: has_action = True; break
        if not has_action: hits.add(i)
    return hits

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
    name = tool.get("name", "")
    inp = tool.get("input", {})
    parts = sorted(f"{k}={str(v)[:120]}" for k, v in inp.items())
    return name + "|" + "|".join(parts)


def extract_S9(msgs: list[dict]) -> set[int]:
    """Repeat failed tool call."""
    hits = set()
    state: dict[str, dict] = {}
    for i, m in enumerate(msgs):
        if m["role"] == "user":
            for res in m["results"]:
                if _ERROR_RE.search(res):
                    for fp in state:
                        state[fp]["error_after"] = True
        elif m["role"] == "assistant":
            for t in m["tools"]:
                fp = _tool_fingerprint(t)
                if fp in state and state[fp].get("error_after"): hits.add(i)
                state[fp] = {"idx": i, "error_after": False}
    return hits


_TRUNC_RE = re.compile(r"(exceeds maximum allowed (size|tokens)|use offset and limit|file too large)", re.IGNORECASE)
_WBR_RE = re.compile(r"read it first before writing|file has not been read yet", re.IGNORECASE)
_NOMATCH_RE = re.compile(r"old_string.*not found|no match found|not unique", re.IGNORECASE)
_PYSTORE_RE = re.compile(r"python was not found.*microsoft store", re.IGNORECASE)

def extract_S10(msgs: list[dict]) -> set[int]:
    """Known-fix failure: tool re-fired after an error whose solution was already in context."""
    hits = set()
    trunc_paths, wbr_paths, nomatch_paths, pystore_cmds = set(), set(), set(), set()
    read_paths = set()

    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            for t in m["tools"]:
                name, inp = t["name"], t["input"]
                if name == "Read":
                    fp = inp.get("file_path", "")
                    if fp:
                        read_paths.add(fp)
                        if fp in trunc_paths: hits.add(i)
                elif name in ("Edit", "Write"):
                    fp = inp.get("file_path", "")
                    if fp and (fp in wbr_paths or fp in nomatch_paths): hits.add(i)
                elif name == "Bash":
                    cmd = inp.get("command", "").strip()
                    norm = re.sub(r"\s+", " ", cmd)[:200]
                    if norm in pystore_cmds: hits.add(i)
        elif m["role"] == "user":
            prev_tools = msgs[i - 1]["tools"] if i > 0 else []
            for res in m["results"]:
                if _TRUNC_RE.search(res):
                    for t in prev_tools: 
                        if t["name"] == "Read": trunc_paths.add(t["input"].get("file_path", ""))
                if _WBR_RE.search(res):
                    for t in prev_tools:
                        if t["name"] in ("Edit", "Write"): wbr_paths.add(t["input"].get("file_path", ""))
                if _NOMATCH_RE.search(res):
                    for t in prev_tools:
                        if t["name"] == "Edit":
                            fp = t["input"].get("file_path", "")
                            if fp and fp in read_paths: nomatch_paths.add(fp)
                if _PYSTORE_RE.search(res):
                    for t in prev_tools:
                        if t["name"] == "Bash": pystore_cmds.add(re.sub(r"\s+", " ", t["input"].get("command", "").strip())[:200])
    return hits


EXTRACTORS = {
    "S1": extract_S1, "S2": extract_S2, "S3": extract_S3, "S4": extract_S4, "S5": extract_S5,
    "S6": extract_S6, "S7": extract_S7, "S8": extract_S8, "S9": extract_S9, "S10": extract_S10,
}


def analyze_session(path: Path) -> dict:
    msgs = load_messages(path)
    if not msgs: return {}
    asst = sum(1 for m in msgs if m["role"] == "assistant")
    user = sum(1 for m in msgs if m["role"] == "user")
    
    # 1. Collect all hits per signal
    all_hits: dict[str, set[int]] = {sid: fn(msgs) for sid, fn in EXTRACTORS.items()}
    
    # 2. Arbitrate: Winner-Takes-All per turn index based on priority
    # Each turn index i can contribute to at most one signal.
    final_hits: dict[str, int] = Counter()
    claimed_turns: set[int] = set()
    
    for sid in SIGNAL_PRIORITY:
        turns = all_hits[sid]
        unclaimed = turns - claimed_turns
        final_hits[sid] = len(unclaimed)
        claimed_turns.update(unclaimed)
    
    total_signals = sum(final_hits.values())
    return {
        "file": str(path),
        "assistant_turns": asst,
        "user_turns": user,
        "signals": dict(final_hits),
        "total_signals": total_signals,
        "signal_rate": round(total_signals / max(1, asst), 4),
    }

# ---------------------------------------------------------------------------
# File discovery & Report generation (mostly unchanged but with arbitration)
# ---------------------------------------------------------------------------

def find_jsonl_files(search_dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for d in search_dirs:
        if not d.exists(): continue
        for p in d.rglob("*.jsonl"): found.append(p)
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
        else: targets.append(p)
        return targets

    search_dirs: list[Path] = []
    if args.dir: search_dirs.append(Path(args.dir))
    else:
        home = Path.home()
        search_dirs += [home / ".claude" / "projects", home / "Downloads"]

    for d in search_dirs:
        if not d.exists(): continue
        for p in d.rglob("*.jsonl"): targets.append(p)
        for p in d.glob("session-export-*.zip"):
            import tempfile
            tmp = Path(tempfile.mkdtemp())
            try: targets.extend(extract_zip_jsonl(p, tmp))
            except Exception: pass

    seen_hashes: set[str] = set()
    deduped: list[Path] = []
    for p in targets:
        try:
            with open(p, "rb") as fh: digest = hashlib.md5(fh.read(8192)).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest); deduped.append(p)
        except OSError: deduped.append(p)
    return deduped

def build_local_report(results: list[dict]) -> dict:
    return {"version": __version__, "sessions_analyzed": len(results), "sessions": results, "aggregate": _aggregate(results)}

def build_upload_report(results: list[dict]) -> dict:
    agg = _aggregate(results)
    n = len(results)
    return {
        "version": __version__,
        "session_count_hash": hashlib.sha256(str(n).encode()).hexdigest()[:12],
        "aggregate": agg,
        "per_session": [
            {k: r[k] for k in ["assistant_turns", "user_turns", "signals", "total_signals", "signal_rate"]}
            for r in results
        ],
    }

def _aggregate(results: list[dict]) -> dict:
    if not results: return {}
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

def main() -> None:
    parser = argparse.ArgumentParser(description="SessionAware — extract look-productive gradient signals")
    parser.add_argument("--file", help="Single .jsonl or .zip file to analyze")
    parser.add_argument("--dir", help="Directory to scan")
    parser.add_argument("--out", default=".", help="Output directory")
    parser.add_argument("--no-upload", action="store_true", help="Skip report_upload.json")
    args = parser.parse_args()

    targets = find_all_targets(args)
    if not targets:
        print("No JSONL files found.", file=sys.stderr); sys.exit(1)

    print(f"SessionAware {__version__} — analyzing {len(targets)} transcript(s)...")
    results = []
    for i, t in enumerate(targets, 1):
        print(f"  [{i}/{len(targets)}] {t.name}", end="", flush=True)
        r = analyze_session(t)
        if r:
            results.append(r)
            print(f"  asst={r['assistant_turns']}  signals={r['total_signals']}  rate={r['signal_rate']}")
        else: print("  (skipped — no messages)")

    if not results:
        print("No sessions produced output.", file=sys.stderr); sys.exit(1)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report_local.json").write_text(json.dumps(build_local_report(results), indent=2))
    if not args.no_upload:
        (out / "report_upload.json").write_text(json.dumps(build_upload_report(results), indent=2))

    agg = _aggregate(results)
    print(f"\n{'='*60}\nSESSIONS: {len(results)}   ASSISTANT TURNS: {agg['total_assistant_turns']}\n{'='*60}")
    print(f"{ 'SIGNAL':<6}  {'COUNT':>7}  {'RATE/TURN':>10}  DESCRIPTION\n{'-'*60}")
    for sid in SIGNAL_PRIORITY:
        count = agg["signal_totals"].get(sid, 0)
        rate = agg["signal_rates_per_assistant_turn"].get(sid, 0.0)
        print(f"{sid:<6}  {count:>7}  {rate:>10.4f}  {SIGNATURES[sid].split('—')[0].strip()}")
    print(f"{'-'*60}\n{'TOTAL':<6}  {sum(agg['signal_totals'].values()):>7}  {agg['overall_signal_rate']:>10.4f}\n{'='*60}")

if __name__ == "__main__":
    main()
