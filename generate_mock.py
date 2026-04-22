import json

mock_data = [
    {
        "type": "assistant", 
        "uuid": "1", 
        "message": {"content": [{"type": "text", "text": "I will start by reading the file."}]}, 
        "agentId": "a1", 
        "isoTimestamp": "2026-04-21T12:00:00Z"
    },
    {
        "type": "assistant", 
        "uuid": "2", 
        "parentUuid": "1", 
        "message": {"content": [{"type": "tool_use", "name": "Read", "input": {"file_path": "x.py"}}]}, 
        "agentId": "a1", 
        "isoTimestamp": "2026-04-21T12:01:00Z"
    },
    {
        "type": "user",
        "uuid": "3",
        "parentUuid": "2",
        "message": {"content": [{"type": "tool_result", "content": "print('hello')"}]}, 
        "agentId": "a1",
        "isoTimestamp": "2026-04-21T12:01:30Z"
    },
    {
        "type": "assistant",
        "uuid": "4",
        "parentUuid": "3",
        "message": {"content": [{"type": "text", "text": "I have finished reading x.py."}]}, 
        "agentId": "a1",
        "isoTimestamp": "2026-04-21T12:02:00Z"
    }
]

with open("mock.jsonl", "w", encoding="utf-8") as f:
    for entry in mock_data:
        f.write(json.dumps(entry) + "\n")
print("mock.jsonl created.")
