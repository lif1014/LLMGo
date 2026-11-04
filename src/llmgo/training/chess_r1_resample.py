"""Bound refill work while preserving complete native GRPO prompt groups."""


def bounded_replacement_groups(pending, available, spare=8):
    """Refill missing groups with a small margin, capped by the loader batch."""
    if any(type(value) is not int or value < 0 for value in (pending, available, spare)):
        raise ValueError('Replacement group counts must be nonnegative integers')
    if pending and not available:
        raise ValueError('Cannot refill prompt groups from an empty batch')
    return min(available, pending+spare) if pending else 0
