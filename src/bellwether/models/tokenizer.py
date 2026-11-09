"""Hashing tokenizer shared by the local server and the Triton Python backend.

Keep this file dependency free (numpy only): ``deploy/triton`` copies the same
logic into the Triton tokenizer model so both serving paths produce identical
token ids.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence

import numpy as np

from bellwether import MAX_TOKENS, VOCAB_SIZE

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def token_id(token: str) -> int:
    # id 0 is reserved for padding
    return zlib.crc32(token.encode("utf-8")) % (VOCAB_SIZE - 1) + 1


def encode(texts: Sequence[str], max_tokens: int = MAX_TOKENS) -> np.ndarray:
    ids = np.zeros((len(texts), max_tokens), dtype=np.int64)
    for i, text in enumerate(texts):
        toks = tokenize(text)[:max_tokens]
        if toks:
            ids[i, : len(toks)] = [token_id(t) for t in toks]
    return ids
