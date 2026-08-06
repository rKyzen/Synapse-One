"""DI container tests."""

from __future__ import annotations

import pytest

from synapse.di import Container, ServiceNotFound


class Greeter:
    def __init__(self, name: str = "world") -> None:
        self.name = name

    def greet(self) -> str:
        return f"hello {self.name}"


class App:
    def __init__(self, greeter: Greeter) -> None:
        self.greeter = greeter


def test_resolve_unknown_raises():
    c = Container()
    with pytest.raises(ServiceNotFound):
        c.resolve(Greeter)


def test_factory_is_singleton_by_default():
    c = Container()
    c.register(Greeter, lambda: Greeter())
    a = c.resolve(Greeter)
    b = c.resolve(Greeter)
    assert a is b


def test_factory_can_close_over_dependencies():
    c = Container()
    c.register(Greeter, lambda: Greeter("synapse"))
    c.register(App, lambda: App(c.resolve(Greeter)))
    app = c.resolve(App)
    assert app.greeter.greet() == "hello synapse"


def test_bind_instance():
    c = Container()
    greeter = Greeter("bound")
    c.bind_instance(Greeter, greeter)
    assert c.resolve(Greeter) is greeter
