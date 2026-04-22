import json
from pathlib import Path
import sessionaware as sa
from collections import Counter

def create_mock_session(messages):
    records = []
    for i, (role, content) in enumerate(messages):
        if isinstance(content, str):
            content_blocks = [{"type": "text", "text": content}]
        else:
            content_blocks = content
            
        records.append({
            "type": role,
            "uuid": f"u{i}",
            "parentUuid": f"u{i-1}" if i > 0 else None,
            "message": {"content": content_blocks},
            "agentId": "agent1"
        })
    return records

def load_messages_from_records(records):
    msgs = []
    for r in records:
        tools, texts, results = sa._parse_content(r["message"]["content"])
        msgs.append({
            "role": r["type"], "tools": tools, "texts": texts, "results": results,
            "text": " ".join(texts), "uuid": r["uuid"], "parentUuid": r["parentUuid"] or "",
            "agent_id": "agent1"
        })
    return msgs

def test_signal(name, records, expected_count):
    msgs = load_messages_from_records(records)
    extractor = sa.EXTRACTORS[name]
    actual_hits = extractor(msgs)
    actual_count = len(actual_hits)
    status = "PASS" if actual_count == expected_count else "FAIL"
    print(f"[{status}] {name}: expected {expected_count}, got {actual_count}")
    return actual_hits

def run_all_tests():
    print("Running Signal Logic Validation (v0.2.0)...\n")

    # --- S1: Redundant Read ---
    s1_records = create_mock_session([
        ("assistant", [{"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}}]),
        ("user", [{"type": "tool_result", "content": "hello"}]),
        ("assistant", [{"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}}]),
    ])
    test_signal("S1", s1_records, 1)

    # --- S2: Unanchored TodoWrite ---
    s2_records = create_mock_session([
        ("user", "Hello"),
        ("assistant", [{"type": "tool_use", "name": "TodoWrite", "input": {"todo": "fix bug"}}]),
        ("user", "Wait"),
    ])
    test_signal("S2", s2_records, 1)

    # --- S3: Bash ping-pong ---
    s3_records = create_mock_session([
        ("assistant", [{"type": "tool_use", "name": "Bash", "input": {"command": "ls -R"}}]),
        ("user", [{"type": "tool_result", "content": "error ls not found"}]), # removed colon to trigger \b
        ("user", "Try dir"),
        ("assistant", [{"type": "tool_use", "name": "Bash", "input": {"command": "ls -R"}}]),
    ])
    test_signal("S3", s3_records, 1)

    # --- S4: Result restatement ---
    s4_records = create_mock_session([
        ("assistant", [{"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}}]),
        ("user", [{"type": "tool_result", "content": "line 1\nline 2"}]),
        ("assistant", "The file contains line 1 and line 2 as expected.") 
    ])
    test_signal("S4", s4_records, 1)

    # --- S5: Narration turn ---
    s5_echo = create_mock_session([
        ("assistant", "I will read the file."),
        ("assistant", [{"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}}]),
        ("user", [{"type": "tool_result", "content": "..."}]),
        ("assistant", "I have read the file.")
    ])
    test_signal("S5", s5_echo, 2)

    # --- S6: Redundant search ---
    s6_records = create_mock_session([
        ("assistant", [{"type": "tool_use", "name": "Grep", "input": {"pattern": "foo"}}]),
        ("user", [{"type": "tool_result", "content": "..."}]),
        ("assistant", [{"type": "tool_use", "name": "Grep", "input": {"pattern": "foo"}}]),
    ])
    test_signal("S6", s6_records, 1)

    # --- S7: Re-plan without delta ---
    s7_records = create_mock_session([
        ("user", "Fix it"),
        ("assistant", [{"type": "tool_use", "name": "TodoWrite", "input": {"todo": "step 1"}}]),
        ("assistant", [{"type": "tool_use", "name": "TodoWrite", "input": {"todo": "step 1 corrected"}}]),
    ])
    test_signal("S7", s7_records, 1)

    # --- S8: Acknowledgment without transition ---
    s8_records = create_mock_session([
        ("user", "Update the tests"),
        ("assistant", "Got it."),
        ("assistant", "Still thinking...\n"), # intervening turn
        ("user", "Actually, wait...")
    ])
    test_signal("S8", s8_records, 1)

    # --- S9 vs S10 Overlap and Arbitration Test ---
    s9_s10_records = create_mock_session([
        ("assistant", [{"type": "tool_use", "name": "Read", "input": {"file_path": "big.log"}}]),
        ("user", [{"type": "tool_result", "content": "error file too large"}]),
        ("assistant", [{"type": "tool_use", "name": "Read", "input": {"file_path": "big.log"}}]),
    ])
    
    print("\nChecking Arbitration (Deduplication):")
    msgs = load_messages_from_records(s9_s10_records)
    hits_s9 = sa.extract_S9(msgs)
    hits_s10 = sa.extract_S10(msgs)
    
    print(f"  Raw S9 hits: {hits_s9}")
    print(f"  Raw S10 hits: {hits_s10}")
    
    combined = Counter()
    claimed = set()
    for sid in sa.SIGNAL_PRIORITY:
        if sid == "S9": turns = hits_s9
        elif sid == "S10": turns = hits_s10
        else: continue
        unclaimed = turns - claimed
        combined[sid] = len(unclaimed)
        claimed.update(unclaimed)
    
    print(f"  Arbitrated: S10={combined['S10']}, S9={combined['S9']}")
    if combined["S10"] == 1 and combined["S9"] == 0:
        print("[PASS] Arbitration correctly prioritized S10 over S9.")
    else:
        print("[FAIL] Arbitration failed to deduplicate.")

if __name__ == "__main__":
    run_all_tests()
