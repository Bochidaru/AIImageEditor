import json

from inpaint_core.models import MemoryTracker


def test_memory_tracker_exports_json(tmp_path):
    tracker = MemoryTracker(enabled=False)
    with tracker.measure("test_stage"):
        sum(range(10))

    output_path = tracker.save_json(tmp_path / "memory.json")
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert "generated_at" in payload
    assert "test_stage" in payload["stages"]
    assert payload["stages"]["test_stage"]["elapsed_ms"] >= 0
    assert payload["stages"]["test_stage"]["peak_allocated_mb"] is None
