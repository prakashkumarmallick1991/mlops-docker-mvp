import numpy as np


def calculate_psi(expected, actual, buckets=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, buckets + 1)
    )

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=breakpoints)
    actual_counts, _ = np.histogram(actual, bins=breakpoints)

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def drift_detected(psi, threshold=0.20):
    return psi >= threshold
