"""Tests for the task DAG domain types (topology, order, edges)."""

from __future__ import annotations

import pytest

from synapse.domain.enums import TaskKind
from synapse.domain.tasks import Task, TaskDAG


def task(tid: str, depends_on: list[str] | None = None) -> Task:
    return Task(id=tid, kind=TaskKind.GENERAL, description=tid, depends_on=depends_on or [])


def test_roots_and_leaves():
    dag = TaskDAG(tasks=[task("t1"), task("t2", depends_on=["t1"]), task("t3", depends_on=["t1"])])
    assert [t.id for t in dag.roots()] == ["t1"]
    assert sorted(t.id for t in dag.leaves()) == ["t2", "t3"]


def test_dependents_of():
    dag = TaskDAG(tasks=[task("t1"), task("t2", depends_on=["t1"]), task("t3", depends_on=["t1"])])
    assert sorted(t.id for t in dag.dependents_of("t1")) == ["t2", "t3"]


def test_topological_order_is_deterministic():
    dag = TaskDAG(tasks=[task("t3", depends_on=["t1"]), task("t2", depends_on=["t1"]), task("t1")])
    order = [t.id for t in dag.topological_order()]
    assert order == ["t1", "t2", "t3"]  # stable by id among ready tasks


def test_topological_order_chain():
    dag = TaskDAG(
        tasks=[
            task("a"),
            task("b", depends_on=["a"]),
            task("c", depends_on=["b"]),
            task("s", depends_on=["b", "c"]),
        ]
    )
    order = [t.id for t in dag.topological_order()]
    assert order.index("a") < order.index("b") < order.index("c") < order.index("s")


def test_cycle_detection():
    dag = TaskDAG(tasks=[task("a", depends_on=["b"]), task("b", depends_on=["a"])])
    with pytest.raises(ValueError, match="cycle"):
        dag.topological_order()


def test_unknown_dependency_raises():
    dag = TaskDAG(tasks=[task("a", depends_on=["ghost"])])
    with pytest.raises(ValueError, match="unknown task"):
        dag.topological_order()


def test_edges_serialization():
    dag = TaskDAG(tasks=[task("t1"), task("t2", depends_on=["t1"])])
    assert dag.edges() == [("t1", "t2")]


def test_add_is_idempotent():
    dag = TaskDAG()
    dag.add(task("t1"))
    dag.add(task("t1"))
    assert len(dag.tasks) == 1


def test_execution_graph_node_lookup():
    from synapse.domain import ExecutionGraph, GraphNode
    from synapse.domain.enums import TaskStatus

    graph = ExecutionGraph(nodes=[GraphNode(task_id="t1", kind=TaskKind.CODING, status=TaskStatus.COMPLETED)])
    assert graph.node("t1") is not None
    assert graph.node("nope") is None