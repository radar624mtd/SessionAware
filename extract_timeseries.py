"""extract_timeseries.py — Full-fidelity ingest of Claude Code JSONL transcripts.

Design principles (each is an explicit audit trail decision):

1. LOSSLESS RAW LAYER
   Every field from the source JSONL is preserved or explicitly noted as
   discarded with a reason. Nothing is silently dropped.

2. UUID CAUSAL CHAIN
   parentUuid links form a directed acyclic graph. We reconstruct that graph
   and assign a topological turn_index, not a line-number index. This matters
   because sidechain agents and parallel tool calls can interleave in the file
   but represent different causal branches.

3. REAL TOKEN COUNTS — NOT HEURISTICS
   input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens
   come directly from the Anthropic API response embedded in each assistant
   message. These are the ground truth billing numbers. We never estimate.

4. CONTEXT WINDOW SIZE PER TURN
   effective_context_tokens = input_tokens + cache_read_input_tokens
   This is the full token load presented to the model at each turn — the
   quantity that drives compute cost and attention overhead.

5. SIGNAL FLAGS ARE PER-TURN, NOT PER-SESSION
   sessionaware.py signals are re-derived at the turn level using the same
   extractor logic but writing flags onto individual message records rather
   than accumulating session counts. This enables positional analysis:
   does signal density correlate with context window size? Do signals cluster
   early or late in sessions?

6. CONTEXT POLLUTION MEASUREMENT (DIRECT, NOT ASSUMED)
   For each signalled turn, we record context_tokens_at_signal and
   context_tokens_at_session_end. The difference (pollution_tail) is the
   measured — not assumed — amount of context those injected tokens occupied
   for the remaining session. This replaces the 0.15 attenuation heuristic
   from impact.py with real numbers.

7. SIDECHAIN / AGENT BRANCHING
   isSidechain field is preserved. Sidechain turns are flagged but included
   because they are billed identically. Analysis can filter or stratify on this.

8. DEDUPLICATION
   uuid-based dedup matches sessionaware.py behaviour. Synthetic messages
   (model == '<synthetic>') are flagged but retained — they represent
   limit-hit turns which are themselves an impact event.

9. REPRODUCIBILITY
   Output is deterministic: sessions sorted by file path, turns sorted by
   topological order within session. Same input always produces same output.

Output schema (one JSONL record per assistant turn):
  session_id        str   — stem of source filename (anonymised)
  session_file      str   — original filename (keep local, strip for sharing)
  turn_index        int   — 0-based topological position in session
  turn_index_asst   int   — 0-based count of assistant turns only
  uuid              str
  parent_uuid       str | null
  is_sidechain      bool
  timestamp         str   — ISO-8601, empty string if absent
  model             str
  stop_reason       str
  is_synthetic      bool  — model == '<synthetic>' or usage all zeros on short msg
  input_tokens      int   — fresh tokens this turn (not cached)
  output_tokens     int
  cache_read_tokens int   — tokens served from cache (part of context but not billed at full rate)
  cache_create_tokens int
  effective_context_tokens int  — input_tokens + cache_read_tokens (full context load)
  text_chars        int   — total characters in text blocks (proxy for output verbosity)
  tool_names        list[str]  — names of tools called in this turn
  tool_count        int
  result_chars      int   — total characters in tool results received (prior user turn)
  has_error_result  bool  — any result matches ERROR_RE
  signals           dict[str, bool]  — per-signal flag for this turn
  any_signal        bool
  signal_count      int
  context_at_signal int   — effective_context_tokens if any_signal else null
  session_total_asst_turns  int   — filled in post-pass
  session_total_tokens      int   — sum of effective_context_tokens for session
  turns_remaining           int   — session_total_asst_turns - turn_index_asst - 1
  pollution_tail_tokens     int   — sum of effective_context_tokens for all subsequent turns
                                    (direct measure of context pollution from this turn)

Usage:
    python extract_timeseries.py --dir C:/Users/radar/Downloads --out output/timeseries/
    python extract_timeseries.py --dir C:/Users/radar/Downloads --out output/timeseries/ --no-local

Produces:
    output/timeseries/turns_local.jsonl   — full records (contains file paths — keep local)
    output/timeseries/turns_upload.jsonl  — stripped of session_file, safe to share
    output/timeseries/ingest_log.json     — per-session parse stats and any warnings
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import sessionaware as sa

# ---------------------------------------------------------------------------
# Constants (identical to sessionaware.py so signal logic is consistent)
# ---------------------------------------------------------------------------

_ERROR_RE = sa._ERROR_RE
_SCOPE_RE = sa._SCOPE_RE
_ACK_RE   = sa._ACK_RE
_FORWARD_RE = sa._FORWARD_RE

SIGNAL_IDS = list(sa.SIGNATURES.keys())  # ['S1'..'S10']

# ---------------------------------------------------------------------------
# Raw JSONL parsing — preserves every field
# ---------------------------------------------------------------------------

def _iter_raw(path: Path) -> list[dict]:
    """Parse all JSONL lines; return list of raw objects. Never raises on bad lines."""
    records: list[dict] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                # log but continue — one bad line must not drop the session
                records.append({"_parse_error": str(exc), "_lineno": lineno})
    return records


def _extract_usage(msg: dict) -> dict[str, int]:
    """Pull all token count fields from a message dict."""
    u = msg.get("usage") or {}
    return {
        "input_tokens":       int(u.get("input_tokens", 0) or 0),
        "output_tokens":      int(u.get("output_tokens", 0) or 0),
        "cache_read_tokens":  int(u.get("cache_read_input_tokens", 0) or 0),
        "cache_create_tokens":int(u.get("cache_creation_input_tokens", 0) or 0),
    }


def _parse_content(content: Any) -> tuple[list[dict], list[str]]:
    """Return (tool_use_blocks, text_strings) from content field.
    tool_result blocks are returned separately via caller inspecting prior user turn."""
    tools: list[dict] = []
    texts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "tool_use":
                tools.append({
                    "name":  block.get("name", ""),
                    "input": block.get("input", {}),
                    "id":    block.get("id", ""),
                })
            elif bt == "text":
                texts.append(block.get("text", "") or "")
            # tool_result blocks appear in *user* turns — handled separately
    elif isinstance(content, str):
        texts.append(content)
    return tools, texts


def _parse_results(content: Any) -> tuple[list[str], bool]:
    """Extract tool result strings and error flag from a user message content."""
    results: list[str] = []
    has_error = False
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                raw = block.get("content", "")
                text = str(raw) if not isinstance(raw, str) else raw
                results.append(text)
                if _ERROR_RE.search(text):
                    has_error = True
    return results, has_error


# ---------------------------------------------------------------------------
# Causal chain reconstruction
# ---------------------------------------------------------------------------

def _build_chain(raw_records: list[dict]) -> list[dict]:
    """
    Reconstruct topological turn order from uuid/parentUuid graph.

    Key structural facts discovered from the data:
    - Some JSONL files contain ONLY sidechain records (isSidechain=True).
      These are subagent sessions billed as separate files. They are valid
      complete causal graphs, not fragments.
    - When a file mixes main-chain and sidechain records, the two graphs
      are SEPARATE (sidechain parentUuids never reference main-chain uuids).
      Each agentId forms its own independent DAG.
    - We therefore sort within each agentId sub-graph independently, then
      concatenate main-chain graph(s) first, sidechain graph(s) after.
      This gives monotonically increasing turn_index_asst within each graph.
    - The _agent_id field is added to each record so downstream analysis
      can stratify or filter by agent sub-graph.

    Returns: list of raw records in causal order, with _chain_index and
             _agent_id added.
    """
    seen_uuids: set[str] = set()
    msg_records: list[dict] = []
    for rec in raw_records:
        if "_parse_error" in rec:
            continue
        t = rec.get("type")
        if t not in ("user", "assistant"):
            continue
        uid = rec.get("uuid", "")
        if uid and uid in seen_uuids:
            continue
        if uid:
            seen_uuids.add(uid)
        msg_records.append(rec)

    # Group by agentId — each agentId is an independent causal graph.
    # Fallback: when agentId is absent, use sessionId (present in all records)
    # so that records from different sessions never collapse into one "_unknown" bucket.
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for rec in msg_records:
        aid = rec.get("agentId", "") or rec.get("sessionId", "") or "_unknown"
        by_agent[aid].append(rec)

    def _topo_sort_agent(recs: list[dict]) -> list[dict]:
        """Topological sort (DFS) of one agent's records.

        The conversation graph is a DAG, not a tree — a single node can be
        the parent of multiple children (parallel tool calls), which means
        BFS/DFS can reach the same node more than once. We deduplicate on
        uuid AFTER traversal to emit each node exactly once in causal order.
        """
        children_map: dict[str | None, list[dict]] = defaultdict(list)
        for rec in recs:
            children_map[rec.get("parentUuid")].append(rec)
        # Sort children by timestamp for determinism
        for kids in children_map.values():
            kids.sort(key=lambda r: r.get("timestamp", "") or "")
        ordered_agent: list[dict] = []
        emitted: set[str] = set()
        queue = list(children_map.get(None, []))
        while queue:
            node = queue.pop(0)
            uid = node.get("uuid", "")
            if uid and uid in emitted:
                continue          # DAG: already emitted via another path
            if uid:
                emitted.add(uid)
            ordered_agent.append(node)
            kids = children_map.get(uid, [])
            queue = kids + queue
        # Orphans: nodes whose parentUuid was not found in this agent's uuid set
        agent_uuids = {r.get("uuid") for r in recs}
        for rec in recs:
            uid = rec.get("uuid", "")
            if uid not in emitted:
                # parentUuid either missing or points outside this agent
                rec["_orphan"] = True
                ordered_agent.append(rec)
                if uid:
                    emitted.add(uid)
        return ordered_agent

    # Separate main-chain agents from sidechain agents
    # An agent is "sidechain" if ALL its records have isSidechain=True
    main_agents: list[tuple[str, list[dict]]] = []
    side_agents: list[tuple[str, list[dict]]] = []
    for aid, recs in by_agent.items():
        is_sc = all(rec.get("isSidechain", False) for rec in recs)
        if is_sc:
            side_agents.append((aid, recs))
        else:
            main_agents.append((aid, recs))

    # Sort agent groups by earliest timestamp for determinism
    def _agent_ts(item: tuple[str, list[dict]]) -> str:
        return min((r.get("timestamp") or "" for r in item[1]), default="")

    main_agents.sort(key=_agent_ts)
    side_agents.sort(key=_agent_ts)

    # Build final ordered list: main chains first, sidechains after
    ordered: list[dict] = []
    for aid, recs in main_agents + side_agents:
        sorted_recs = _topo_sort_agent(recs)
        for rec in sorted_recs:
            rec["_chain_index"] = len(ordered)
            rec["_agent_id"] = aid
            ordered.append(rec)

    return ordered


# ---------------------------------------------------------------------------
# Per-session signal derivation at turn level
# ---------------------------------------------------------------------------
# We re-implement signal detection in a streaming fashion that emits a boolean
# flag per (turn_index, signal_id) rather than a session count. The logic is
# identical to sessionaware.py extractors — we are not redefining signals,
# just changing output granularity.

def _derive_turn_signals(ordered: list[dict]) -> dict[int, dict[str, bool]]:
    """
    Run all signal extractors over the ordered message list and return
    a mapping: chain_index -> {signal_id: bool}.
    """
    sa_msgs = []
    chain_idx_map = []

    for rec in ordered:
        msg = rec.get("message", {})
        if not isinstance(msg, dict): continue
        role = rec.get("type", "")
        if role not in ("user", "assistant"): continue
        content = msg.get("content", [])
        tools, texts = _parse_content(content)
        results, has_err = _parse_results(content)

        sa_msgs.append({
            "role": role, "tools": tools, "texts": texts, "results": results,
            "text": " ".join(texts), "ts": rec.get("timestamp", ""),
            "_chain_index": rec.get("_chain_index", -1),
        })
        chain_idx_map.append(rec.get("_chain_index", -1))

    # 1. Collect raw hits from centralized extractors
    all_hits = {sid: sa.EXTRACTORS[sid](sa_msgs) for sid in SIGNAL_IDS}

    # 2. Arbitrate per turn index
    flags = defaultdict(lambda: {s: False for s in SIGNAL_IDS})
    claimed_turns = set()
    for sid in sa.SIGNAL_PRIORITY:
        turns = all_hits[sid]
        unclaimed = turns - claimed_turns
        for turn_idx in unclaimed:
            cidx = chain_idx_map[turn_idx]
            flags[cidx][sid] = True
        claimed_turns.update(unclaimed)

    return dict(flags)


# ---------------------------------------------------------------------------
# Session processor
# ---------------------------------------------------------------------------

def process_session(path: Path) -> tuple[list[dict], dict]:
    """
    Process one JSONL file.
    Returns (turn_records, log_entry).
    turn_records: one dict per assistant turn, fully populated.
    log_entry: parse stats and warnings for ingest_log.json.
    """
    log: dict[str, Any] = {
        "file": path.name,
        "raw_lines": 0,
        "parse_errors": 0,
        "total_records": 0,
        "assistant_turns": 0,
        "user_turns": 0,
        "orphaned_nodes": 0,
        "synthetic_turns": 0,
        "turns_with_tokens": 0,
        "warnings": [],
    }

    raw = _iter_raw(path)
    log["raw_lines"] = len(raw)
    log["parse_errors"] = sum(1 for r in raw if "_parse_error" in r)

    ordered = _build_chain(raw)
    log["total_records"] = len(ordered)
    log["orphaned_nodes"] = sum(1 for r in ordered if r.get("_orphan"))

    # Derive per-turn signal flags
    turn_signals = _derive_turn_signals(ordered)

    session_id = path.stem

    # First pass: build assistant turn records.
    # CRITICAL: turn_index_asst and pollution_tail must be computed
    # PER AGENT SUB-GRAPH, not across the merged ordered list.
    # Each agentId has its own context window — context does not bleed
    # between the main chain and sidechain agents.
    turn_records: list[dict] = []

    # Group ordered records by agent_id
    by_agent_ordered: dict[str, list[dict]] = defaultdict(list)
    for rec in ordered:
        aid = rec.get("_agent_id", "_unknown")
        by_agent_ordered[aid].append(rec)

    # Determine agent ordering: main agents (not all-sidechain) first
    def _is_sidechain_agent(recs: list[dict]) -> bool:
        return all(rec.get("isSidechain", False) for rec in recs)

    agent_order = sorted(
        by_agent_ordered.keys(),
        key=lambda a: (
            int(_is_sidechain_agent(by_agent_ordered[a])),  # main first
            min((r.get("timestamp") or "" for r in by_agent_ordered[a]), default=""),
        )
    )

    for aid in agent_order:
        agent_recs = by_agent_ordered[aid]
        agent_turn_records: list[dict] = []
        asst_turn_idx = 0
        prev_user_results: list[str] = []
        prev_user_error: bool = False

        for rec in agent_recs:
            msg = rec.get("message", {})
            if not isinstance(msg, dict):
                continue
            role = rec.get("type", "")
            if role == "user":
                log["user_turns"] += 1
                content = msg.get("content", [])
                prev_user_results, prev_user_error = _parse_results(content)
                continue
            if role != "assistant":
                continue

            log["assistant_turns"] += 1
            content = msg.get("content", [])
            tools, texts = _parse_content(content)
            usage = _extract_usage(msg)
            model = msg.get("model", "") or ""
            is_synthetic = (model == "<synthetic>") or (
                usage["input_tokens"] == 0
                and usage["output_tokens"] == 0
                and len(texts) == 1
                and len(texts[0]) < 200
            )
            if is_synthetic:
                log["synthetic_turns"] += 1

            effective_ctx = usage["input_tokens"] + usage["cache_read_tokens"]
            has_real_tokens = effective_ctx > 0 or usage["output_tokens"] > 0
            if has_real_tokens:
                log["turns_with_tokens"] += 1

            cidx = rec.get("_chain_index", -1)
            sig_flags = turn_signals.get(cidx, {s: False for s in SIGNAL_IDS})
            any_sig = any(sig_flags.values())
            sig_count = sum(sig_flags.values())

            record: dict[str, Any] = {
                # identity
                "session_id":         session_id,
                "session_file":       path.name,
                "agent_id":           aid,
                "is_sidechain_agent": _is_sidechain_agent(agent_recs),
                "turn_index":         rec.get("_chain_index", -1),
                "turn_index_asst":    asst_turn_idx,
                "uuid":               rec.get("uuid", ""),
                "parent_uuid":        rec.get("parentUuid"),
                "is_sidechain":       bool(rec.get("isSidechain", False)),
                "is_orphan":          bool(rec.get("_orphan", False)),
                # timing
                "timestamp":          rec.get("timestamp", "") or "",
                # model
                "model":              model,
                "stop_reason":        msg.get("stop_reason", "") or "",
                "is_synthetic":       is_synthetic,
                # tokens — ground truth from Anthropic API response
                "input_tokens":       usage["input_tokens"],
                "output_tokens":      usage["output_tokens"],
                "cache_read_tokens":  usage["cache_read_tokens"],
                "cache_create_tokens":usage["cache_create_tokens"],
                # effective_context = what the model actually attended over this turn
                "effective_context_tokens": effective_ctx,
                # content
                "text_chars":         sum(len(t) for t in texts),
                "tool_names":         [t["name"] for t in tools],
                "tool_count":         len(tools),
                # prior user turn context
                "result_chars":       sum(len(r) for r in prev_user_results),
                "has_error_result":   prev_user_error,
                # signals
                "signals":            sig_flags,
                "any_signal":         any_sig,
                "signal_count":       sig_count,
                "context_at_signal":  effective_ctx if any_sig else 0,
                # agent-level session fields — filled in second pass below
                "agent_total_asst_turns": 0,
                "agent_total_tokens":     0,
                "turns_remaining":        0,
                "pollution_tail_tokens":  0,
            }
            agent_turn_records.append(record)
            asst_turn_idx += 1

            prev_user_results = []
            prev_user_error = False

        # Second pass within this agent: compute suffix sums for pollution measurement.
        # pollution_tail_tokens for turn i = Σ effective_context_tokens for turns i+1..end
        # within THIS agent's context window. This is the direct measured quantity —
        # no heuristic attenuation. If a signal fires at turn i, the context it
        # injected rode in every subsequent turn's effective_context. The tail sum
        # is an upper bound on the pollution cost (it includes clean turns too, but
        # when used in regression as a covariate against signal presence, the
        # per-signal attributable fraction is identified statistically).
        n_agent = len(agent_turn_records)
        agent_ctx_total = sum(r["effective_context_tokens"] for r in agent_turn_records)

        ctx_suffix: list[int] = [0] * (n_agent + 1)
        for i in range(n_agent - 1, -1, -1):
            ctx_suffix[i] = ctx_suffix[i + 1] + agent_turn_records[i]["effective_context_tokens"]

        for i, r in enumerate(agent_turn_records):
            r["agent_total_asst_turns"] = n_agent
            r["agent_total_tokens"]     = agent_ctx_total
            r["turns_remaining"]        = n_agent - i - 1
            r["pollution_tail_tokens"]  = ctx_suffix[i + 1]

        turn_records.extend(agent_turn_records)

    if not turn_records:
        log["warnings"].append("no_assistant_turns")
    elif log["turns_with_tokens"] == 0:
        log["warnings"].append("no_real_token_counts")

    return turn_records, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full-fidelity time-series ingest of Claude Code JSONL transcripts"
    )
    ap.add_argument("--dir",      default="C:/Users/radar/Downloads",
                    help="Directory to scan for .jsonl files")
    ap.add_argument("--out",      default="output/timeseries",
                    help="Output directory")
    ap.add_argument("--no-local", action="store_true",
                    help="Skip writing turns_local.jsonl (which contains file paths)")
    args = ap.parse_args()

    targets = sa.find_all_targets(
        __import__("argparse").Namespace(file=None, dir=args.dir)
    )
    targets = sorted(targets)  # deterministic order

    if not targets:
        print("No JSONL files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(targets)} transcript(s)...")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    local_path  = out / "turns_local.jsonl"
    upload_path = out / "turns_upload.jsonl"
    log_path    = out / "ingest_log.json"

    all_logs: list[dict] = []
    total_turns = 0
    total_tokens = 0

    with (
        open(local_path,  "w", encoding="utf-8") as lf,
        open(upload_path, "w", encoding="utf-8") as uf,
    ):
        for i, t in enumerate(targets, 1):
            if i % 50 == 0 or i == len(targets):
                print(f"  [{i}/{len(targets)}] processed {total_turns} turns so far...")
            try:
                records, log = process_session(Path(t))
            except Exception as exc:
                all_logs.append({"file": Path(t).name, "error": str(exc)})
                continue

            all_logs.append(log)
            total_turns += len(records)
            total_tokens += sum(r["effective_context_tokens"] for r in records)

            for rec in records:
                line = json.dumps(rec, ensure_ascii=False)
                if not args.no_local:
                    lf.write(line + "\n")
                # upload version: strip session_file path
                upload_rec = {k: v for k, v in rec.items() if k != "session_file"}
                uf.write(json.dumps(upload_rec, ensure_ascii=False) + "\n")

    log_path.write_text(json.dumps(all_logs, indent=2))

    # Summary
    sessions_with_tokens = sum(1 for l in all_logs if l.get("turns_with_tokens", 0) > 0)
    sessions_with_signals = 0
    # quick recount from logs
    parse_errors = sum(l.get("parse_errors", 0) for l in all_logs)

    print(f"\n{'='*60}")
    print(f"  Transcripts processed : {len(all_logs)}")
    print(f"  Assistant turns       : {total_turns:,}")
    print(f"  Total effective tokens: {total_tokens:,}")
    print(f"  Sessions with tokens  : {sessions_with_tokens}")
    print(f"  Parse errors          : {parse_errors}")
    print(f"  turns_local.jsonl     : {local_path}")
    print(f"  turns_upload.jsonl    : {upload_path}")
    print(f"  ingest_log.json       : {log_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
