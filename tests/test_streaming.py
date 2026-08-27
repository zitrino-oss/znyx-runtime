"""Streaming evaluator: text is only released after its window has been
evaluated and allowed (release-after-verdict)."""
from znyx_core.engine.streaming import StreamingEvaluator

POLICY = {"secrets": {"enabled": True}}


def _drive(evaluator, text, chunk_size=20):
    events = []
    for i in range(0, len(text), chunk_size):
        events.extend(evaluator.push(text[i:i + chunk_size]))
    events.extend(evaluator.flush())
    return events


class TestCleanStream:
    def test_all_text_released_and_allowed(self):
        text = "The quick brown fox jumps over the lazy dog. " * 6
        ev = StreamingEvaluator(POLICY, context="output", window_size=60, overlap=10)
        events = _drive(ev, text, chunk_size=25)

        done = events[-1]
        assert done["event"] == "done"
        assert done["data"]["final_decision"] == "ALLOW"
        assert done["data"]["released_text_length"] == len(text)
        assert ev.released_text == text
        assert not ev.is_blocked

        released = "".join(e["data"]["text"] for e in events if e["event"] == "chunk")
        assert released == text

    def test_verdict_always_precedes_release(self):
        # At every point in the event stream, at least as many windows must
        # have been evaluated as chunks released.
        text = "All work and no play makes for a very dull day indeed. " * 5
        ev = StreamingEvaluator(POLICY, context="output", window_size=60, overlap=10)
        events = _drive(ev, text)

        verdicts = released = 0
        for event in events:
            if event["event"] == "guardrail":
                verdicts += 1
            elif event["event"] == "chunk":
                released += 1
                assert released <= verdicts

    def test_chunk_indexes_count_releases_in_order(self):
        text = "word " * 60
        ev = StreamingEvaluator(POLICY, context="output", window_size=60, overlap=10)
        events = _drive(ev, text)
        indexes = [e["data"]["chunk_index"] for e in events if e["event"] == "chunk"]
        assert indexes == list(range(1, len(indexes) + 1))


class TestBlockedStream:
    def _blocked_events(self, fake_pat):
        text = ("Here are your deployment notes for today. The token is "
                + fake_pat + " and it must stay private. " + "Trailing text. " * 5)
        ev = StreamingEvaluator(POLICY, context="output", window_size=60, overlap=10)
        return ev, _drive(ev, text), fake_pat

    def test_block_verdict_stops_the_stream(self, fake_pat):
        ev, events, _ = self._blocked_events(fake_pat)
        kinds = [e["event"] for e in events]
        assert "block" in kinds
        assert ev.is_blocked
        assert events[-1]["data"]["final_decision"] == "BLOCK"

    def test_blocked_content_is_never_released(self, fake_pat):
        ev, events, token = self._blocked_events(fake_pat)
        assert token not in ev.released_text
        released = "".join(e["data"]["text"] for e in events if e["event"] == "chunk")
        assert token not in released
        # Nothing is released once the block verdict lands.
        block_at = next(i for i, e in enumerate(events) if e["event"] == "block")
        assert all(e["event"] != "chunk" for e in events[block_at:])

    def test_block_event_carries_no_text_preview(self, fake_pat):
        _, events, _ = self._blocked_events(fake_pat)
        block = next(e for e in events if e["event"] == "block")
        assert "text_preview" not in block["data"]
        assert any(h["rule_id"].startswith("secrets.") for h in block["data"]["rule_hits"])

    def test_released_length_reported_in_done(self, fake_pat):
        ev, events, _ = self._blocked_events(fake_pat)
        done = events[-1]["data"]
        assert done["released_text_length"] == len(ev.released_text)
        assert done["released_text_length"] < done["full_text_length"]

    def test_push_after_block_returns_block_event(self, fake_pat):
        ev, _, _ = self._blocked_events(fake_pat)
        follow_up = ev.push("more text")
        assert [e["event"] for e in follow_up] == ["block"]

    def test_short_stream_block_surfaces_on_flush(self, fake_pat):
        # A stream shorter than one window is only evaluated at flush; its
        # BLOCK must reach the caller and nothing may be released.
        ev = StreamingEvaluator(POLICY, context="output", window_size=200, overlap=40)
        events = ev.push("token: " + fake_pat)
        assert events == []
        events = ev.flush()
        kinds = [e["event"] for e in events]
        assert "block" in kinds and "chunk" not in kinds
        assert events[-1]["data"]["final_decision"] == "BLOCK"
        assert ev.released_text == ""


class TestShortCleanStream:
    def test_flush_releases_evaluated_tail(self):
        ev = StreamingEvaluator(POLICY, context="output", window_size=200, overlap=40)
        assert ev.push("just a short reply") == []
        events = ev.flush()
        assert [e["event"] for e in events] == ["guardrail", "chunk", "done"]
        assert ev.released_text == "just a short reply"
        assert events[-1]["data"]["final_decision"] == "ALLOW"
