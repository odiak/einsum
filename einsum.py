from typing import Dict, Tuple
import numpy as np


def einsum(expr: str, *args: Tuple[np.ndarray, ...], **kwargs) -> np.ndarray:
    (a, b) = map(str.strip, expr.split("->"))
    a_ = list(
        map(lambda s: list(map(str.strip, s.split(","))), map(str.strip, a.split(";")))
    )
    b_ = list(map(str.strip, b.split(",")))
    chars = "abcdefghijklmnopqrstuvwxyz"
    char_map: Dict[str, str] = {}
    i = 0
    for cs in a_:
        for c in cs:
            if c not in char_map:
                char_map[c] = chars[i]
                i += 1
    for c in b_:
        if c not in char_map:
            char_map[c] = chars[i]
            i += 1
    expr_ = "->".join(
        [
            ",".join(map(lambda ss: "".join(map(lambda s: char_map[s], ss)), a_)),
            "".join(map(lambda s: char_map[s], b_)),
        ]
    )
    return np.einsum(expr_, *args, **kwargs)
