# SessionAware

**AI providers are paid for every token generated. Users are charged for every token generated. These interests are only aligned when every token serves the user.**

SessionAware measures where they diverge.

It extracts 10 structural signals from Claude Code session transcripts — each one a category of billable token generation that provides no function to the user. The output is counts. The counts are the accountability.

---

## Signals

| ID | Name | Definition |
|---|---|---|
| S1 | Redundant read | `Read` of a range already in context: exact repeat, subsumed range, partial overlap, or unnecessary split across consecutive turns |
| S2 | Unanchored TodoWrite | `TodoWrite` call not followed by any tool action before the next user turn |
| S3 | Bash ping-pong | `Bash` command re-issued with the same environment fingerprint (shell dialect, path style) that previously failed, after a corrected fingerprint succeeded |
| S4 | Result restatement | Text-only assistant turn immediately after a tool result, where the text is backward-looking with no new information |
| S5 | Narration turn | Text-only assistant turn wrapping tool calls: either the closing echo of a tool sandwich (text → tools → **text**) or a preamble announcing an action immediately before taking it |
| S6 | Redundant search | Identical `Grep`/`Glob` pattern re-issued with no new writes between |
| S7 | Re-plan without delta | `TodoWrite` re-issued within the same user-turn block with no intervening tool results |
| S8 | Acknowledgment without transition | Text-only turn matching filler phrases (e.g. "Got it", "Understood") with no tool action before the next user turn |
| S9 | Repeat failed tool call | Tool re-issued with the same name and input after an error result appeared between calls |
| S10 | Known-fix failure | Tool re-fired after an error whose solution was already stated in context — sub-cases: truncated read (`trunc`), write before read (`write_before_read`), edit no-match (`edit_no_match`), wrong Python invocation (`py_not_found`) |

Each signal event enters the context window and remains for the duration of the session, present in every subsequent cache read.

---

## Usage

```bash
# Scan default locations (~/.claude/projects/ and ~/Downloads/)
python sessionaware.py

# Scan a directory of session export ZIPs or JSONL files
python sessionaware.py --dir ~/Downloads/

# Single file
python sessionaware.py --file session-export-1234.zip

# Specify output directory
python sessionaware.py --out ./results/
```

Requires Python 3.8+. No third-party dependencies.

---

## Output

| File | Contents | Share? |
|---|---|---|
| `report_local.json` | Full per-session detail including file paths | No |
| `report_upload.json` | Anonymised counts and rates only — no paths, no text, no identifiers | Yes |

---

## Contributing

Run the tool against your own transcripts. Open an issue and attach `report_upload.json`. No other information is needed or wanted.

---

## License

GPL-3.0
