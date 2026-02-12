def calculate_oklab_range(h: int, w: int) -> float:
    """Calculates the maximum possible Euclidean distance in Oklab space for a given area."""
    return (2.28 * h * w) ** 4
