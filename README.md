# SessionAware

A raw signal extractor for Claude Code session transcripts.

Detects structural signatures of the "look productive" gradient — patterns where the model generates output despite already possessing the information required to act. No interpretation, no opinion. Counts only.

---

## What it measures

Ten structural signals, each independently detectable from transcript structure alone:

| ID | Signal |
|---|---|
| S3 | **Redundant read** — file read again with no intervening edit |
| S4 | **Turn split** — consecutive assistant messages with no user turn between |
| S5 | **Unanchored TodoWrite** — task list generated with no subsequent action |
| C3 | **Bash ping-pong** — identical shell command re-run (result already known) |
| S8 | **Result restatement** — assistant text reproduces prior tool output |
| S9 | **Plan-execute echo** — narrates intent, acts, narrates completion |
| S10 | **Redundant search** — identical Grep/Glob re-run with no new writes |
| S12 | **Re-plan without delta** — TodoWrite re-run with no new information received |
| S13 | **Scope declaration** — lists files/steps immediately before executing them |
| S14 | **Acknowledgment without transition** — filler turn with no subsequent action |

These are not judgments. They are counts of structural events. What the counts mean is for the data to show.

---

## What it does not do

- Does not read, transmit, or store any content from your transcripts
- Does not classify intent or quality
- Does not make claims about any vendor
- Does not require network access

---

## Usage

```bash
# Scan default locations (~/.claude/projects/ and ~/Downloads/)
python sessionaware.py

# Scan a specific directory of exports
python sessionaware.py --dir ~/Downloads/

# Single file or ZIP
python sessionaware.py --file session-export-1234.zip

# Output to a specific directory
python sessionaware.py --out ./results/
```

Requires Python 3.8+. No dependencies beyond the standard library.

---

## Output

Two files are generated:

**`report_local.json`** — full per-session detail including file paths. Keep this local.

**`report_upload.json`** — anonymised aggregate: turn counts, signal counts, rates per assistant turn. No paths, no text, no identifiers. Safe to share.

---

## Contributing data

If you want to contribute your `report_upload.json` to the population dataset, open an issue and attach the file. The aggregate will be updated as submissions arrive.

---

## License

GPL-3.0
