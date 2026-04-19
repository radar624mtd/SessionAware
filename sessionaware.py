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
    "S1":  "Redundant read — Read of content already in context (sub-cases: exact, subsumed, overlap, split)",
    "S2":  "Unanchored TodoWrite — TodoWrite call not followed by any action",
    "S3":  "Bash ping-pong — identical Bash command re-run (already had result)",
    "S4":  "Result restatement — assistant text reproduces prior tool result",
    "S5":  "Narration turn — text-only turn wrapping tool calls (sub-cases: echo, preamble)",
    "S6":  "Redundant search — identical Grep/Glob re-run with no new writes",
    "S7":  "Re-plan without delta — TodoWrite re-run with no new tool results between",
    "S8":  "Acknowledgment without transition — filler turn with no subsequent action",
    "S9":  "Repeat failed tool call — same tool+input re-issued after an error result",
    "S10": "Known-fix failure — tool re-fired after an error whose solution was already in context (sub-cases: trunc, write_before_read, edit_no_match, py_not_found)",
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


def _range_contains(outer: tuple[int, int | None], inner: tuple[int, int | None]) -> bool:
    """Outer range fully contains inner range."""
    o_off, o_lim = outer
    i_off, i_lim = inner
    if o_lim is None:   # outer is full-file read
        return True
    if i_lim is None:   # inner is full-file but outer is partial
        return False
    return i_off >= o_off and (i_off + i_lim) <= (o_off + o_lim)


def _range_overlaps(r1: tuple[int, int | None], r2: tuple[int, int | None]) -> bool:
    a_off, a_lim = r1
    b_off, b_lim = r2
    if a_lim is None or b_lim is None:
        return True
    return not (a_off + a_lim <= b_off or b_off + b_lim <= a_off)


def extract_S1(msgs: list[dict]) -> int:
    """Redundant read — Read of content already in context. Four sub-cases,
    mutually exclusive per re-read (earliest match wins):

      exact     — path + offset + limit identical to a prior Read
      subsumed  — new range fully contained within a prior Read's range
      overlap   — ranges partially overlap (portion already in context)
      split     — disjoint but contiguous read in the immediately following
                  assistant turn (prior_end == cur_offset, no other tool call
                  between); could have been one Read call

    True pagination (disjoint reads across separated tool calls) is NOT counted."""
    count = 0
    path_reads: dict[str, list[tuple[int, int | None]]] = {}
    # track: for each file_path the (msg_idx, range) of the most recent Read,
    # and whether any non-Read tool call appeared in a turn between that Read
    # and the current position — used for split sub-case detection
    last_read: dict[str, tuple[int, tuple[int, int | None]]] = {}  # fp -> (msg_idx, range)
    last_nonread_turn: int = -1  # msg_idx of the most recent turn containing a non-Read tool

    for msg_idx, m in enumerate(msgs):
        # track any non-Read tool use so split detection can exclude turns
        # where meaningful work happened between two Reads
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
            priors = path_reads.get(fp, [])

            matched = False
            # 1. exact
            for p in priors:
                if p == cur:
                    count += 1
                    matched = True
                    break
            # 2. subsumed
            if not matched:
                for p in priors:
                    if _range_contains(p, cur):
                        count += 1
                        matched = True
                        break
            # 3. overlap (partial)
            if not matched:
                for p in priors:
                    if _range_overlaps(p, cur):
                        count += 1
                        matched = True
                        break
            # 4. split — disjoint but contiguous with the most recent prior Read
            #    of the same file, with no non-Read tool calls between them
            if not matched and fp in last_read:
                prev_msg_idx, prev_range = last_read[fp]
                prev_off, prev_lim = prev_range
                cur_off, _ = cur
                if (
                    prev_lim is not None
                    and cur_off == prev_off + prev_lim          # contiguous, no gap
                    and last_nonread_turn <= prev_msg_idx       # no other tool work between
                ):
                    count += 1
                    matched = True

            priors.append(cur)
            path_reads[fp] = priors
            if m["role"] == "assistant":
                last_read[fp] = (msg_idx, cur)

    return count


def extract_S2(msgs: list[dict]) -> int:
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


_PS_RE = re.compile(r"(?:\bGet-|\bSet-|\bNew-|\$env:|\.ps1\b|-Command\b|Write-Host\b)")
_CMD_RE = re.compile(r"%[A-Z_]+%|\bdir\s+/[a-z]\b", re.IGNORECASE)
_BASH_RE = re.compile(r"(?:\$\{?[A-Za-z_]\w*\}?|&&|\|\||2>&1|/c/|\$\()")

def _bash_dialect(cmd: str) -> str:
    if _PS_RE.search(cmd):
        return "powershell"
    if _CMD_RE.search(cmd):
        return "cmd"
    if _BASH_RE.search(cmd):
        return "bash"
    return "neutral"

def _bash_path_style(cmd: str) -> str:
    if re.search(r"/c/Users", cmd, re.IGNORECASE):
        return "posix"
    if re.search(r"C:\\\\Users|C:\\Users", cmd):
        return "windows_backslash"
    if re.search(r"C:/Users", cmd):
        return "windows_forward"
    return "none"

def _bash_fingerprint(cmd: str) -> tuple:
    return (_bash_dialect(cmd), _bash_path_style(cmd))

def extract_S3(msgs: list[dict]) -> int:
    """Failure-context repeat: a Bash call that exhibits an environment-context
    fingerprint (shell dialect, path style) that previously FAILED in this
    session, when a different fingerprint was seen to succeed (or correcting
    feedback appeared) between the failure and the repeat. This measures
    Claude ignoring correction already present in context."""
    count = 0
    failed_fps: dict[tuple, int] = {}     # fp -> idx of last failure
    succeeded_fps: dict[tuple, int] = {}  # fp -> idx of last success
    window = 40  # tool-call index span to consider repeat fresh

    bash_calls: list[tuple] = []  # (msg_idx, call_idx_in_seq, fp, cmd, error)
    seq = 0
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            for t in m["tools"]:
                if t["name"] != "Bash":
                    continue
                cmd = t["input"].get("command", "").strip()
                if not cmd:
                    continue
                fp = _bash_fingerprint(cmd)
                # look ahead one message for the tool_result and classify error
                err = False
                if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
                    for res in msgs[i + 1]["results"]:
                        if _ERROR_RE.search(res):
                            err = True
                            break
                bash_calls.append((i, seq, fp, err))
                seq += 1

    for idx, (msg_i, seq_i, fp, err) in enumerate(bash_calls):
        if fp in failed_fps:
            prior_fail_seq = failed_fps[fp]
            if seq_i - prior_fail_seq <= window:
                # check: was there an intervening success with a different fingerprint,
                # OR an intervening user message between the prior failure and now?
                corrected = False
                for fp2, succ_seq in succeeded_fps.items():
                    if fp2 != fp and prior_fail_seq < succ_seq < seq_i:
                        corrected = True
                        break
                if not corrected:
                    prior_msg_i = bash_calls[seq_i_lookup(bash_calls, prior_fail_seq)][0]
                    for k in range(prior_msg_i + 1, msg_i):
                        if msgs[k]["role"] == "user" and not msgs[k]["results"]:
                            corrected = True
                            break
                if corrected:
                    count += 1
        if err:
            failed_fps[fp] = seq_i
        else:
            succeeded_fps[fp] = seq_i
    return count


def seq_i_lookup(bash_calls: list[tuple], seq: int) -> int:
    for i, bc in enumerate(bash_calls):
        if bc[1] == seq:
            return i
    return 0


_FORWARD_RE = re.compile(
    r"\b(let\s+me|i.?ll|i\s+will|next[,\s]|now\s+(let|i)|going\s+to|"
    r"i\s+need\s+to|we.?ll|we\s+need\s+to|first[,\s]|then\s+i)\b",
    re.IGNORECASE,
)

def extract_S4(msgs: list[dict]) -> int:
    """Result restatement: text-only assistant turn immediately following a tool_result
    user turn, whose text is backward-looking (describes what just happened) rather
    than forward-looking (declares next action). Captures situation recaps."""
    count = 0
    for i in range(1, len(msgs)):
        m = msgs[i]
        prev = msgs[i - 1]
        if m["role"] != "assistant" or m["tools"] or not m["text"].strip():
            continue
        if prev["role"] != "user" or not prev["results"]:
            continue
        text = m["text"].strip()
        # skip pure scope/plan turns (those are S13)
        if _FORWARD_RE.search(text.split(".")[0]):  # forward verb in first sentence -> not a recap
            continue
        # require it reference the result substantively (not just one-word ack)
        if len(text.split()) < 6:
            continue
        count += 1
    return count


def extract_S5(msgs: list[dict]) -> int:
    """Narration turn — text-only assistant turn wrapping tool calls. Two sub-cases,
    mutually exclusive per turn index (echo wins over preamble):

      echo     — text-only closing turn of a tool sandwich: text → tools → TEXT
                 with no real user prompt between
      preamble — text-only turn with planning language immediately before a tool
                 call (was formerly S13 'Scope declaration')

    A single text-only turn between two tool sandwiches would be both 'echo' of
    the prior sandwich and 'preamble' of the next — counted once as 'echo'."""
    n = len(msgs)
    echo_idxs: set[int] = set()
    preamble_idxs: set[int] = set()

    # Pass 1 — echo detection (closing text-only turn of a sandwich)
    i = 0
    while i < n:
        m = msgs[i]
        if m["role"] == "assistant" and not m["tools"] and m["text"].strip():
            saw_tool = False
            j = i + 1
            while j < n:
                mj = msgs[j]
                if mj["role"] == "user":
                    if mj["results"]:
                        j += 1
                        continue
                    else:
                        break
                if mj["role"] == "assistant":
                    if mj["tools"]:
                        saw_tool = True
                        j += 1
                        continue
                    if not mj["text"].strip():
                        j += 1
                        continue
                    if saw_tool:
                        echo_idxs.add(j)
                    break
                j += 1
            i = j
        else:
            i += 1

    # Pass 2 — preamble detection (planning language before a tool call)
    for i in range(n - 1):
        if i in echo_idxs:
            continue  # already counted as echo
        m = msgs[i]
        if m["role"] != "assistant" or m["tools"] or not m["text"].strip():
            continue
        if not _SCOPE_RE.search(m["text"]):
            continue
        for j in range(i + 1, min(i + 6, n)):
            mj = msgs[j]
            if mj["role"] == "user" and not mj["results"]:
                break
            if mj["role"] == "assistant" and mj["tools"]:
                preamble_idxs.add(i)
                break

    return len(echo_idxs) + len(preamble_idxs)


def extract_S6(msgs: list[dict]) -> int:
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
                inp = t["input"]
                pat = str(inp.get("pattern", "") or inp.get("query", ""))[:120]
                if not pat:
                    continue
                if pat in searches and searches[pat] > last_write:
                    count += 1
                searches[pat] = i
    return count


def extract_S7(msgs: list[dict]) -> int:
    """Re-plan churn: TodoWrite re-issued between the same real user prompt turns.
    Counts every TodoWrite after the first within a user-prompt block."""
    count = 0
    todos_in_block = 0
    for m in msgs:
        if m["role"] == "user":
            # real user prompt resets; tool_result user turns do not
            if not m["results"]:
                todos_in_block = 0
            continue
        if any(t["name"] == "TodoWrite" for t in m["tools"]):
            if todos_in_block > 0:
                count += 1
            todos_in_block += 1
    return count


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
    r"exactly|indeed|yes[.,!]|correct[.,!])\b",
    re.IGNORECASE,
)

def extract_S8(msgs: list[dict]) -> int:
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


def extract_S9(msgs: list[dict]) -> int:
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


# --- S16 error-pattern matchers ---
_TRUNC_RE = re.compile(
    r"(exceeds maximum allowed (size|tokens)|use offset and limit|"
    r"file too large|content too large|too large to read|read specific portions)",
    re.IGNORECASE,
)
_WBR_RE = re.compile(
    r"read it first before writing|file has not been read yet",
    re.IGNORECASE,
)
_NOMATCH_RE = re.compile(
    r"old_string.*not found|string.*not found in file|no match found|"
    r"not unique in the file|edit.*not.*unique",
    re.IGNORECASE,
)
_PYSTORE_RE = re.compile(
    r"python was not found.*microsoft store|run without arguments to install",
    re.IGNORECASE,
)


def extract_S10(msgs: list[dict]) -> int:
    """Known-fix failure: tool re-fired after an error whose solution was already
    in context. Four sub-cases, each independently tracked to prevent double-counting:

      trunc           — Read after platform size-limit notice (fix: use offset/limit)
      write_before_read — Edit/Write after 'read it first' error (fix stated verbatim)
      edit_no_match   — Edit retry after 'old_string not found', when a prior Read
                        of the same file exists (correct text was already in context)
      py_not_found    — Bash re-run after 'Python was not found / Microsoft Store'
                        (wrong python invocation; fix inferrable from prior successes)

    A path/fingerprint is added to a sub-case set when its error fires. Every
    subsequent matching tool call on that path counts as one instance. A single
    retry turn can only contribute one count per sub-case, preventing overlap."""
    count = 0

    # sub-case state: path/fingerprint -> marked once error seen
    trunc_paths: set[str] = set()
    wbr_paths: set[str] = set()       # write_before_read
    nomatch_paths: set[str] = set()   # edit_no_match (requires prior Read)
    pystore_cmds: set[str] = set()    # py_not_found: normalised command fingerprint

    read_paths: set[str] = set()      # paths seen via Read (for edit_no_match strict check)

    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            for t in m["tools"]:
                name = t["name"]
                inp = t["input"]

                if name == "Read":
                    fp = inp.get("file_path", "")
                    if fp:
                        read_paths.add(fp)
                        if fp in trunc_paths:
                            count += 1  # trunc retry

                elif name in ("Edit", "Write"):
                    fp = inp.get("file_path", "")
                    if fp:
                        if fp in wbr_paths:
                            count += 1  # write_before_read retry
                        if fp in nomatch_paths:
                            count += 1  # edit_no_match retry

                elif name == "Bash":
                    cmd = inp.get("command", "").strip()
                    norm = re.sub(r"\s+", " ", cmd)[:200]
                    if norm in pystore_cmds:
                        count += 1  # py_not_found retry

        elif m["role"] == "user":
            # inspect results and mark the failing path/cmd from the prior assistant turn
            prev_tools = msgs[i - 1]["tools"] if i > 0 else []
            for res in m["results"]:
                if _TRUNC_RE.search(res):
                    for t in prev_tools:
                        if t["name"] == "Read":
                            fp = t["input"].get("file_path", "")
                            if fp:
                                trunc_paths.add(fp)

                if _WBR_RE.search(res):
                    for t in prev_tools:
                        if t["name"] in ("Edit", "Write"):
                            fp = t["input"].get("file_path", "")
                            if fp:
                                wbr_paths.add(fp)

                if _NOMATCH_RE.search(res):
                    for t in prev_tools:
                        if t["name"] == "Edit":
                            fp = t["input"].get("file_path", "")
                            if fp and fp in read_paths:
                                nomatch_paths.add(fp)

                if _PYSTORE_RE.search(res):
                    for t in prev_tools:
                        if t["name"] == "Bash":
                            cmd = t["input"].get("command", "").strip()
                            norm = re.sub(r"\s+", " ", cmd)[:200]
                            if norm:
                                pystore_cmds.add(norm)

    return count


EXTRACTORS = {
    "S1":  extract_S1,
    "S2":  extract_S2,
    "S3":  extract_S3,
    "S4":  extract_S4,
    "S5":  extract_S5,
    "S6":  extract_S6,
    "S7":  extract_S7,
    "S8":  extract_S8,
    "S9":  extract_S9,
    "S10": extract_S10,
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
