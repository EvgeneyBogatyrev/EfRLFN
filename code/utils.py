def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value