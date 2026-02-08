import numpy as np

# -------------------------------------------------
# 1. TEMPORAL CONSISTENCY ERROR (TCE)
# -------------------------------------------------
def temporal_consistency_error(pred_counts):
    """
    Measures stability of predicted crowd count over time.
    Lower is better.
    """
    pred_counts = np.array(pred_counts)
    if len(pred_counts) < 2:
        return 0.0

    diffs = np.abs(np.diff(pred_counts))
    return np.mean(diffs)


# -------------------------------------------------
# 2. MOTION–DENSITY CORRELATION (MDC)
# -------------------------------------------------
def motion_density_correlation(avg_densities, motion_scores):
    """
    Measures physical consistency between crowd density and motion.
    Higher positive correlation is better.
    """
    avg_densities = np.array(avg_densities)
    motion_scores = np.array(motion_scores)

    if len(avg_densities) < 2:
        return 0.0

    return np.corrcoef(avg_densities, motion_scores)[0, 1]


# -------------------------------------------------
# 3. RISK EVENT RESPONSIVENESS (RER)
# -------------------------------------------------
def risk_event_responsiveness(sri_values, threshold=0.005):
    """
    Measures how quickly the system responds to dangerous situations.
    Higher is better.
    """
    sri_values = np.array(sri_values)
    if len(sri_values) < 2:
        return 0.0

    deltas = []
    for i in range(1, len(sri_values)):
        if sri_values[i] > threshold:
            deltas.append(sri_values[i] - sri_values[i - 1])

    return np.mean(deltas) if len(deltas) > 0 else 0.0
