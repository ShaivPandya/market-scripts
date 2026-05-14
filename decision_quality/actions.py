"""Canonical action vocabulary for investment decisions."""

from __future__ import annotations

from typing import Literal

CanonicalAction = Literal[
    "buy",
    "add",
    "short",
    "sell",
    "trim",
    "reduce",
    "exit",
    "hedge",
    "rebalance",
    "hold",
    "watch",
    "research",
    "avoid",
    "do_nothing",
]

CANONICAL_ACTIONS: tuple[str, ...] = (
    "buy",
    "add",
    "short",
    "sell",
    "trim",
    "reduce",
    "exit",
    "hedge",
    "rebalance",
    "hold",
    "watch",
    "research",
    "avoid",
    "do_nothing",
)

ACTIONABLE_ACTIONS: set[str] = {
    "buy",
    "add",
    "short",
    "sell",
    "trim",
    "reduce",
    "exit",
    "hedge",
    "rebalance",
}

NON_ACTIONABLE_ACTIONS: set[str] = {"hold", "watch", "research", "avoid", "do_nothing"}


def normalize_action(value: object, *, fallback: str = "watch") -> str:
    action = str(value or fallback).strip().lower()
    return action if action in CANONICAL_ACTIONS else fallback


def is_actionable(value: object) -> bool:
    return normalize_action(value) in ACTIONABLE_ACTIONS
