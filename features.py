"""Feature engineering shared by the notebook and the Streamlit app.

Keeping this in a module (not inside the notebook) lets joblib load the saved
pipeline in the app.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL_COLS = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]
RAW_COLS = NUMERIC_COLS + CATEGORICAL_COLS + ["Weekend"]
ENGINEERED_COLS = ["TotalPages", "TotalDuration", "AvgProductDuration", "HasPageValue", "Weekend"]

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_TYPES = ["Returning_Visitor", "New_Visitor", "Other"]


class SessionFeatures(BaseEstimator, TransformerMixin):
    """Adds session-level features. Stateless, so it cannot leak test data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = pd.DataFrame(X).copy()
        d["Weekend"] = d["Weekend"].astype(int)
        d["TotalPages"] = d["Administrative"] + d["Informational"] + d["ProductRelated"]
        d["TotalDuration"] = (d["Administrative_Duration"] + d["Informational_Duration"]
                              + d["ProductRelated_Duration"])
        d["AvgProductDuration"] = d["ProductRelated_Duration"] / d["ProductRelated"].clip(lower=1)
        d["HasPageValue"] = (d["PageValues"] > 0).astype(int)
        for c in ["OperatingSystems", "Browser", "Region", "TrafficType"]:
            d[c] = d[c].astype(str)
        return d


def log1p_nonneg(X):
    """log(1 + x) for skewed counts and durations (negatives clipped to 0)."""
    return np.log1p(np.clip(X, 0, None))
