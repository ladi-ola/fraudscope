"""Transparent per-feature deviation explanations without external packages."""

import numpy as np


def top_contributors(matrix, names, flagged, top_k=3):
    """Rank features by robust deviation from their median in each flagged row."""
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    contributions = np.abs(matrix - median) / np.maximum(1.4826 * mad, 1e-6)
    return {
        int(row): [
            names[i] for i in np.argsort(-contributions[row], kind="stable")[:top_k]
        ]
        for row in flagged
    }
