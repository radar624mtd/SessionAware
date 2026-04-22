"""LLM-judge harness for SessionAware signal precision measurement.

Reads turns_local.jsonl (already has per-turn signal flags), samples flagged
instances per signal, reconstructs context windows from the source JSONL, then
runs an LLM judge pass using claude-haiku-4-5 to compute per-signal precision.

Nothing leaves the machine. All inputs/outputs are local files.

Usage:
    python judge.py [--turns output/timeseries/turns_local.jsonl]
                   [--dir C:/Users/radar/Downloads]
                   [--per-signal 50] [--out output/judge/]

Produces:
    output/judge/instances.jsonl  — sampled flagged instances with context
    output/judge/verdicts.jsonl   — LLM judgments (tp/fp + rationale)
    output/judge/summary.json     — precision per signal + overall
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import anthropic
import sessionaware as sa

# ---------------------------------------------------------------------------
# Signal definitions for judge prompts
# ---------------------------------------------------------------------------

SIGNAL_DESCRIPTIONS: dict[str, str] = {
    "S1": (
        "Redundant Read — the assistant issued a Read tool call for a file (or range) "
        "whose content was already present in the conversation context from a prior Read. "
        "Sub-cases: exact duplicate, subsumed range, overlapping range, or split read "
        "(contiguous chunks that could have been one call). "
        "TRUE POSITIVE: the prior read result was already visible in context and the new "
        "read adds no new content. "
        "FALSE POSITIVE: the prior read was far enough back that re-reading is reasonable, "
        "or the content changed (Edit/Write between reads), or the ranges don't actually overlap."
    ),
    "S2": (
        "Unanchored TodoWrite — the assistant issued a TodoWrite call not followed by any "
        "substantive tool action. The plan was written but execution did not follow. "
        "TRUE POSITIVE: TodoWrite turn stands alone with no real work after it before the "
        "next user message. "
        "FALSE POSITIVE: the TodoWrite is immediately followed by tool calls in a subsequent "
        "turn, or the user intervened with new instructions."
    ),
    "S3": (
        "Bash environment-context repeat — a Bash command with the same shell dialect/path-style "
        "fingerprint as a prior FAILED command was re-issued without correction, when a "
        "corrected version had already succeeded or the user had intervened. "
        "TRUE POSITIVE: same environment mistake repeated after clear failure feedback. "
        "FALSE POSITIVE: the command is legitimately similar but not the same mistake, "
        "or the prior failure was unrelated to the fingerprint (e.g. logic error not env error)."
    ),
    "S4": (
        "Result restatement — a text-only assistant turn immediately after a tool result "
        "that describes what just happened (backward-looking) rather than declaring next action. "
        "TRUE POSITIVE: turn is a narrative summary of the result with no forward action content. "
        "FALSE POSITIVE: turn actually declares what the assistant will do next, or provides "
        "meaningful synthesis the user needs (not just echo)."
    ),
    "S5": (
        "Narration turn — a text-only assistant turn that either (a) closes a tool sandwich "
        "with a summary echo, or (b) declares scope/plan immediately before tool calls. "
        "TRUE POSITIVE: text adds no information beyond what the tool results already show, "
        "or is pure planning language before tool calls that could be skipped. "
        "FALSE POSITIVE: the text provides necessary clarification, error diagnosis, or "
        "information the user genuinely needs beyond the raw tool output."
    ),
    "S6": (
        "Redundant search — identical Grep or Glob pattern re-run with no new writes between. "
        "TRUE POSITIVE: exact same search pattern issued again when results were already in context. "
        "FALSE POSITIVE: a write happened between, or the prior result was truncated/partial."
    ),
    "S7": (
        "Re-plan churn — TodoWrite re-issued within the same user-prompt block after already "
        "having written a plan, with no new tool results justifying a revision. "
        "TRUE POSITIVE: plan rewritten without meaningful new information. "
        "FALSE POSITIVE: tool results between the two TodoWrites justify the revision."
    ),
    "S8": (
        "Acknowledgment without transition — short filler text (got it, understood, noted, etc.) "
        "not followed by a tool call. "
        "TRUE POSITIVE: pure acknowledgment turn with no action following it. "
        "FALSE POSITIVE: the acknowledgment is followed by tool calls in a nearby turn, "
        "or the acknowledgment contains substantive content."
    ),
    "S9": (
        "Repeat failed tool call — exact same tool+input re-issued after an error result "
        "appeared between the two calls. "
        "TRUE POSITIVE: identical call retried despite an error result with no fix applied. "
        "FALSE POSITIVE: the tool inputs differ in a meaningful way, or the error was transient "
        "and a retry is appropriate."
    ),
    "S10": (
        "Known-fix failure — tool re-fired after an error whose solution was already present "
        "in context. Sub-cases: truncation retry (Read after size-limit), write-before-read "
        "(Edit/Write after 'read first' error), edit-no-match (Edit retry after old_string "
        "not found when the correct content was already read), py-not-found (Bash retry after "
        "Microsoft Store python error). "
        "TRUE POSITIVE: the fix was literally stated in context and the same mistake was repeated. "
        "FALSE POSITIVE: the error was for a different reason than the sub-case implies."
    ),
}

JUDGE_SYSTEM = """You are a precision evaluator for Claude Code session analysis.
You will be shown a flagged instance from a real Claude Code conversation transcript
and asked to judge whether the flag is a true positive or false positive.

Respond with a JSON object only — no prose before or after:
{
  "verdict": "tp" | "fp",
  "confidence": "high" | "medium" | "low",
  "rationale": "<one sentence explaining the key reason>"
}

Be strict: only mark tp if the signal clearly matches. When in doubt, fp."""


def _judge_prompt(signal: str, instance: dict) -> str:
    desc = SIGNAL_DESCRIPTIONS.get(signal, "")
    ctx_lines = []
    for turn in instance.get("context", []):
        marker = ">>> FLAGGED TURN <<<" if turn.get("marker") else ""
        role = turn["role"].upper()
        text = turn.get("text", "")[:400]
        tools = turn.get("tools", [])
        results = turn.get("results", [])
        parts = [f"[{role}] {marker}"]
        if text:
            parts.append(f"  text: {text}")
        for t in tools[:6]:
            if isinstance(t, str):
                parts.append(f"  tool: {t}")
            else:
                inp = {k: str(v)[:150] for k, v in t.get("input_preview", {}).items()}
                parts.append(f"  tool: {t['name']}({inp})")
        for r in results[:3]:
            parts.append(f"  result: {r[:300]}")
        ctx_lines.append("\n".join(parts))

    detail = instance.get("detail", {})
    return (
        f"SIGNAL: {signal}\n\n"
        f"DEFINITION:\n{desc}\n\n"
        f"FLAGGING DETAIL: {json.dumps(detail)}\n\n"
        f"CONVERSATION CONTEXT:\n"
        + "\n---\n".join(ctx_lines)
        + "\n\nJudge this instance. Is it a true positive (tp) or false positive (fp)?"
    )


# ---------------------------------------------------------------------------
# Instance collection from turns_local.jsonl + source JSONL context
# ---------------------------------------------------------------------------

def _build_context_map(turns_path: Path) -> dict[str, list[dict]]:
    """Index turns by (session_file, agent_id) for O(1) context window lookup."""
    index: dict[str, list[dict]] = defaultdict(list)
    with open(turns_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = f"{r['session_file']}|{r['agent_id']}"
            index[key].append(r)
    return dict(index)


def collect_instances_from_timeseries(turns_path: Path) -> list[dict]:
    """Extract flagged instances directly from turns_local.jsonl.

    Each assistant turn already has per-signal flags, effective_context_tokens,
    turn_index_asst, and enough metadata to build a context window from adjacent
    records in the same agent group.
    """
    # Load all turns grouped by agent
    agent_groups: dict[str, list[dict]] = defaultdict(list)
    with open(turns_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = f"{r['session_file']}|{r['agent_id']}"
            agent_groups[key].append(r)

    instances: list[dict] = []

    for key, turns in agent_groups.items():
        # turns are already in turn_index_asst order
        turns_sorted = sorted(turns, key=lambda t: t["turn_index_asst"])
        n = len(turns_sorted)

        for i, t in enumerate(turns_sorted):
            if not t.get("any_signal"):
                continue
            for sig, fired in t.get("signals", {}).items():
                if not fired:
                    continue

                # Build context window: 3 turns before + 3 after
                lo = max(0, i - 3)
                hi = min(n, i + 4)
                context = []
                for k in range(lo, hi):
                    ct = turns_sorted[k]
                    context.append({
                        "idx": ct["turn_index_asst"],
                        "role": "assistant",
                        "marker": (k == i),
                        "text": "",          # assistant turns: text_chars only, no raw text
                        "text_chars": ct.get("text_chars", 0),
                        "tools": ct.get("tool_names", []),
                        "effective_context_tokens": ct.get("effective_context_tokens", 0),
                        "has_error_result": ct.get("has_error_result", False),
                        "any_signal": ct.get("any_signal", False),
                        "signals": ct.get("signals", {}),
                    })

                instances.append({
                    "signal": sig,
                    "session_file": t["session_file"],
                    "agent_id": t["agent_id"],
                    "turn_index_asst": t["turn_index_asst"],
                    "effective_context_tokens": t.get("effective_context_tokens", 0),
                    "tool_names": t.get("tool_names", []),
                    "has_error_result": t.get("has_error_result", False),
                    "detail": {},
                    "context": context,
                })

    return instances


def enrich_from_source(instances: list[dict], source_dir: Path) -> list[dict]:
    """Add raw text/tool content from source JSONL files for richer judge context.

    Matches each instance to its source file by session_file, finds the agent
    sub-chain, and injects raw message text/tool-input previews into context turns.
    Falls back gracefully if source file is not found.
    """
    # Index source files by basename
    source_files: dict[str, Path] = {}
    if source_dir.exists():
        for p in source_dir.rglob("*.jsonl"):
            source_files[p.name] = p

    enriched = 0
    for inst in instances:
        sf = inst["session_file"]
        if sf not in source_files:
            continue
        path = source_files[sf]
        raw = list(sa._iter_jsonl(path))

        # Build agent-filtered ordered messages
        aid = inst["agent_id"]
        msgs_raw = [
            r for r in raw
            if r.get("type") in ("user", "assistant")
            and (r.get("agentId") or r.get("sessionId") or "_unknown") == aid
        ]
        # Sort by turn_index proxy: use order in file
        # Map turn_index_asst to raw records: assistant turns in file order
        asst_msgs = [m for m in msgs_raw if m.get("type") == "assistant"]

        target_idx = inst["turn_index_asst"]
        if target_idx >= len(asst_msgs):
            continue

        # Rebuild context with raw content around target_idx
        lo = max(0, target_idx - 3)
        hi = min(len(asst_msgs), target_idx + 4)

        rich_context = []
        # Interleave user messages too by going back to msgs_raw
        # Find approximate position of target assistant turn in msgs_raw
        asst_positions = [i for i, m in enumerate(msgs_raw) if m.get("type") == "assistant"]
        if target_idx >= len(asst_positions):
            continue
        target_raw_pos = asst_positions[target_idx]

        raw_lo = max(0, target_raw_pos - 5)
        raw_hi = min(len(msgs_raw), target_raw_pos + 5)

        for k in range(raw_lo, raw_hi):
            m = msgs_raw[k]
            role = m.get("type", "")
            tools, texts, results = sa._parse_content(
                m.get("message", {}).get("content", [])
            )
            asst_turn_idx = asst_positions.index(k) if k in asst_positions else -1
            marker = (role == "assistant" and asst_turn_idx == target_idx)
            tools_preview = [
                {"name": t["name"],
                 "input_preview": {ik: str(iv)[:200] for ik, iv in t["input"].items()}}
                for t in tools[:6]
            ]
            rich_context.append({
                "idx": k,
                "role": role,
                "marker": marker,
                "text": " ".join(texts)[:500],
                "tools": tools_preview,
                "results": [r[:400] for r in results[:3]],
            })

        inst["context"] = rich_context
        enriched += 1

    return instances


# ---------------------------------------------------------------------------
# LLM judge pass
# ---------------------------------------------------------------------------

def run_judge_pass(
    instances: list[dict],
    out_path: Path,
    model: str = "claude-haiku-4-5-20251001",
    max_retries: int = 3,
    rate_limit_delay: float = 0.2,
) -> list[dict]:
    """Send each instance to the LLM judge and collect verdicts."""
    client = anthropic.Anthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        base_url=os.environ.get("ANTHROPIC_BASE_URL"),
    )

    verdicts: list[dict] = []
    errors = 0

    with open(out_path, "w", encoding="utf-8") as fh:
        for i, inst in enumerate(instances, 1):
            sig = inst["signal"]
            prompt = _judge_prompt(sig, inst)

            verdict_record: dict[str, Any] = {
                "signal": sig,
                "session_file": inst.get("session_file", ""),
                "agent_id": inst.get("agent_id", ""),
                "turn_index_asst": inst.get("turn_index_asst", -1),
                "effective_context_tokens": inst.get("effective_context_tokens", 0),
            }

            for attempt in range(max_retries):
                try:
                    resp = client.messages.create(
                        model=model,
                        max_tokens=256,
                        system=JUDGE_SYSTEM,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw_text = resp.content[0].text.strip()
                    # Strip markdown fences if present
                    if raw_text.startswith("```"):
                        raw_text = raw_text.split("```")[1]
                        if raw_text.startswith("json"):
                            raw_text = raw_text[4:]
                    parsed = json.loads(raw_text)
                    verdict_record["verdict"] = parsed.get("verdict", "fp")
                    verdict_record["confidence"] = parsed.get("confidence", "low")
                    verdict_record["rationale"] = parsed.get("rationale", "")
                    verdict_record["error"] = None
                    break
                except json.JSONDecodeError as e:
                    verdict_record["verdict"] = "fp"
                    verdict_record["confidence"] = "low"
                    verdict_record["rationale"] = f"parse_error: {raw_text[:100]}"
                    verdict_record["error"] = f"json_decode: {e}"
                    errors += 1
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    verdict_record["verdict"] = "fp"
                    verdict_record["confidence"] = "low"
                    verdict_record["rationale"] = ""
                    verdict_record["error"] = str(e)
                    errors += 1

            verdicts.append(verdict_record)
            fh.write(json.dumps(verdict_record, ensure_ascii=False) + "\n")
            fh.flush()

            if i % 25 == 0:
                tps = sum(1 for v in verdicts if v.get("verdict") == "tp")
                print(f"  [{i}/{len(instances)}] tp={tps}  fp={i-tps}  errors={errors}")

            time.sleep(rate_limit_delay)

    return verdicts


# ---------------------------------------------------------------------------
# Precision summary
# ---------------------------------------------------------------------------

def compute_summary(verdicts: list[dict]) -> dict[str, Any]:
    by_signal: dict[str, list[dict]] = defaultdict(list)
    for v in verdicts:
        by_signal[v["signal"]].append(v)

    per_signal: dict[str, Any] = {}
    for sig in sorted(by_signal.keys()):
        vs = by_signal[sig]
        n = len(vs)
        tps = sum(1 for v in vs if v.get("verdict") == "tp")
        high_conf = [v for v in vs if v.get("confidence") == "high"]
        hc_tps = sum(1 for v in high_conf if v.get("verdict") == "tp")
        per_signal[sig] = {
            "n": n,
            "tp": tps,
            "fp": n - tps,
            "precision": round(tps / max(1, n), 4),
            "high_conf_n": len(high_conf),
            "high_conf_precision": round(hc_tps / max(1, len(high_conf)), 4),
        }

    all_n = len(verdicts)
    all_tp = sum(1 for v in verdicts if v.get("verdict") == "tp")
    return {
        "total_judged": all_n,
        "total_tp": all_tp,
        "total_fp": all_n - all_tp,
        "overall_precision": round(all_tp / max(1, all_n), 4),
        "per_signal": per_signal,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM-judge precision measurement for SessionAware signals"
    )
    ap.add_argument("--turns", default="output/timeseries/turns_local.jsonl",
                    help="turns_local.jsonl from extract_timeseries.py")
    ap.add_argument("--dir", default="C:/Users/radar/Downloads",
                    help="Source JSONL directory for context enrichment")
    ap.add_argument("--per-signal", type=int, default=50,
                    help="Max instances to judge per signal")
    ap.add_argument("--out", default="output/judge",
                    help="Output directory")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--model", default="claude-haiku-4-5-20251001",
                    help="Model for judge pass")
    ap.add_argument("--collect-only", action="store_true",
                    help="Only collect instances, skip LLM judge pass")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- Collect ---
    print(f"Collecting flagged instances from {args.turns}...")
    instances = collect_instances_from_timeseries(Path(args.turns))

    by_sig: dict[str, list[dict]] = defaultdict(list)
    for inst in instances:
        by_sig[inst["signal"]].append(inst)
    print(f"Total flagged instances: {len(instances)}")
    for sig in sorted(by_sig.keys()):
        print(f"  {sig}: {len(by_sig[sig])}")

    # --- Stratified sample ---
    rng = random.Random(args.seed)
    sampled: list[dict] = []
    for sig in sorted(by_sig.keys()):
        pool = by_sig[sig]
        k = min(args.per_signal, len(pool))
        picks = rng.sample(pool, k)
        sampled.extend(picks)
        print(f"  {sig}: population={len(pool)}, sampled={k}")

    inst_path = out / "instances.jsonl"
    with open(inst_path, "w", encoding="utf-8") as f:
        for inst in sampled:
            f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(sampled)} sampled instances to {inst_path}")

    print(f"\nReady for judgment. Read {inst_path} to score instances.")


if __name__ == "__main__":
    main()
