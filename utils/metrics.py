import numpy as np


def q_error(true: list, estimate: list) -> np.array:
    y_true = np.asarray(true).reshape((-1,))
    y_pred = np.asarray(estimate).reshape((-1,))

    assert y_true.shape == y_pred.shape, (
        "Shapes of true and predicted arrays must match."
    )

    qerror = np.maximum(
        (y_pred + 1) / (y_true + 1), (y_true + 1) / (y_pred + 1)
    )  # use +1 smoothing to avoid division by zero
    return qerror


def rmse(true: list, estimate: list) -> float:
    true = np.asarray(true).reshape((-1,))
    estimate = np.asarray(estimate).reshape((-1,))
    return np.sqrt(np.mean((true - estimate) ** 2))


def q_error_percentiles(q_errors: list, percentiles: list) -> dict:
    results = {}
    for p in percentiles:
        results[f"{p}th"] = np.percentile(q_errors, p)
    return results
