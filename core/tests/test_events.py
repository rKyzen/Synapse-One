"""Event bus tests."""

from __future__ import annotations

from synapse.events import EventBus, Events


def test_subscribe_and_publish():
    bus = EventBus()
    seen = []

    def handler(event):
        seen.append(event)

    bus.subscribe("foo", handler)
    bus.publish("foo", {"a": 1})
    assert len(seen) == 1
    assert seen[0].name == "foo"
    assert seen[0].payload == {"a": 1}


def test_wildcard():
    bus = EventBus()
    seen = []
    bus.subscribe("*", seen.append)
    bus.publish("anything")
    assert seen[0].name == "anything"


def test_unsubscribe():
    bus = EventBus()
    seen = []

    def handler(event):
        seen.append(event)

    bus.subscribe("foo", handler)
    bus.unsubscribe("foo", handler)
    bus.publish("foo")
    assert seen == []


def test_handler_error_is_isolated():
    bus = EventBus()
    ok = []

    def bad(event):
        raise RuntimeError("boom")

    def good(event):
        ok.append(event)

    bus.subscribe("foo", bad)
    bus.subscribe("foo", good)
    bus.publish("foo")
    assert len(ok) == 1


def test_canonical_names_are_unique():
    names = [v for v in vars(Events).values() if isinstance(v, str)]
    assert len(names) == len(set(names))
