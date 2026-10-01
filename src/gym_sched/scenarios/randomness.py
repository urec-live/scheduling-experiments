"""Stable keyed randomness: no draw-order dependency or Python randomized hash()."""

import hashlib
import json

import numpy as np


def keyed_rng(seed: int, stream: str, *key: object) -> np.random.Generator:
    payload = json.dumps([seed, stream, *key], sort_keys=True, separators=(",", ":"))
    entropy = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")
    return np.random.default_rng(entropy)


def uniform(seed: int, stream: str, *key: object) -> float:
    return float(keyed_rng(seed, stream, *key).random())


def lognormal_factor(seed: int, stream: str, cv: float, *key: object) -> float:
    if cv == 0:
        return 1.0
    sigma = float(np.sqrt(np.log1p(cv * cv)))
    return float(keyed_rng(seed, stream, *key).lognormal(-sigma * sigma / 2, sigma))


def content_hash(value: object) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
