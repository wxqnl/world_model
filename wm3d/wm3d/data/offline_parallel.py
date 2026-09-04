"""Bounded ordered dispatch for CPU-only, offline data materialization."""
from collections import deque
from itertools import islice


def iter_chunks(values, size):
    iterator = iter(values)
    while True:
        chunk = tuple(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def ordered_results(pool, function, items, *, max_pending):
    iterator = iter(items)
    pending = deque()
    for item in islice(iterator, max_pending):
        pending.append(pool.submit(function, item))
    while pending:
        yield pending.popleft().result()
        for item in islice(iterator, 1):
            pending.append(pool.submit(function, item))
