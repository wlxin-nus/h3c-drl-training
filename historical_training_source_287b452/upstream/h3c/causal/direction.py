"""Qualitative sign composition and whole-program direction proof."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from h3c.causal.graph import Edge

AMBIGUOUS = "ambiguous"


def compose(*signs: int | str) -> int | str:
    result = 1
    for sign in signs:
        if sign == AMBIGUOUS:
            return AMBIGUOUS
        if sign == 0:
            return 0
        result *= int(sign)
    return result


def combine(signs: Iterable[int | str]) -> int | str:
    seen = {sign for sign in signs if sign != 0}
    if not seen:
        return 0
    if AMBIGUOUS in seen or len(seen) > 1:
        return AMBIGUOUS
    return seen.pop()


def path_sign(
    edges: Iterable[Edge], source: str, target: str, max_length: int = 4
) -> int | str | None:
    adjacency: dict[str, list[tuple[str, int]]] = {}
    for edge in edges:
        if edge.sign is None:
            continue
        adjacency.setdefault(edge.source, []).append((edge.target, edge.sign))
        if edge.undirected:
            adjacency.setdefault(edge.target, []).append((edge.source, edge.sign))
    if source == target:
        return 1
    found: list[int | str] = []

    def walk(node: str, sign: int | str, seen: set[str], length: int) -> None:
        if length >= max_length:
            return
        for next_node, next_sign in adjacency.get(node, []):
            if next_node in seen:
                continue
            composed = compose(sign, next_sign)
            if next_node == target:
                found.append(composed)
            else:
                walk(next_node, composed, seen | {next_node}, length + 1)

    walk(source, 1, {source}, 0)
    return None if not found else combine(found)


def consistent_program_direction_proof(
    effect: Mapping[str, Any], cited_edges: Iterable[Edge]
) -> dict[str, Any]:
    """Require one non-zero direction over the complete executable program."""
    directions = tuple(effect.get("directions", ()))
    if len(directions) != 1 or directions[0] == 0:
        raise ValueError(
            "one unified non-zero direction across the full executable program cannot be proved"
        )
    drive = int(directions[0])
    edges = tuple(cited_edges)
    nodes = sorted({edge.source for edge in edges} | {edge.target for edge in edges})
    expected_effects: list[dict[str, str]] = []
    for node in nodes:
        if node == "cooling_setpoint":
            continue
        path = path_sign(edges, "cooling_setpoint", node)
        if path is None or path == 0:
            continue
        if path == AMBIGUOUS:
            raise ValueError(f"cited signed edges give an ambiguous effect for {node}")
        implied = compose(drive, path)
        if implied == AMBIGUOUS or implied == 0:
            raise ValueError(f"cited signed edges do not determine an effect for {node}")
        expected_effects.append({"node": node, "direction": "up" if implied == 1 else "down"})
    if not expected_effects:
        raise ValueError("cited signed edges yield no non-zero deterministic effect")
    return {
        "mode": "global",
        "program_direction": "up" if drive == 1 else "down",
        "expected_effects": expected_effects,
        "witness_count": len(effect.get("witnesses", ())),
        "cited_edge_ids": [edge.identifier for edge in edges],
    }
