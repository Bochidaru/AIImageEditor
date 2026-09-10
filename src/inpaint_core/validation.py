from __future__ import annotations


def require_exactly_one(a: object, b: object, name_a: str, name_b: str) -> None:
    """Raise ValueError unless exactly one of `a`/`b` is not None.

    Shared by every "provide exactly one of X or Y" input pair in this
    codebase (selection/mask, placement/mask, image/image_id) so the rule
    and its wording live in one place instead of being hand-copied per call
    site.
    """
    if (a is None) == (b is None):
        raise ValueError(f"Provide exactly one of {name_a} or {name_b}.")
