# SessionAware

**AI providers are paid for every token generated. Users are charged for every token generated. These interests are only aligned when every token serves the user.**

SessionAware measures where they diverge.

It extracts 11 structural signals from Claude Code session transcripts — each one a category of billable token generation that provides no function to the user. The output is counts. The counts are the accountability.

---

## Signals

| ID | Event | Definition |
|---|---|---|
| S3 | Redundant read | `Read` on path P where P was already read with no intervening `Edit`/`Write` on P |
| S4 | Turn split | Consecutive assistant turns with no user turn between |
| S5 | Unanchored TodoWrite | `TodoWrite` call not followed by any tool action before next user turn |
| C3 | Bash ping-pong | Identical `Bash` command issued more than once in a session |
| S8 | Result restatement | Assistant text-only turn with >55% token overlap with immediately preceding tool result |
| S9 | Plan-execute echo | Text-only → tool-call → text-only assistant sandwich with no user turns between |
| S10 | Redundant search | Identical `Grep`/`Glob` pattern re-issued with no new writes between |
| S12 | Re-plan without delta | `TodoWrite` re-issued between same user turns with no intervening tool results |
| S13 | Scope declaration | Text-only turn matching planning language immediately before an `Edit`/`Write` turn |
| S14 | Acknowledgment without transition | Text-only turn matching filler phrases with no tool action before next user turn |
| S15 | Repeat failed tool call | Tool re-issued with same name and input after an error result appeared between calls |

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
