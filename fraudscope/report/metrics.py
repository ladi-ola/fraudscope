"""Short, consistent explanations for numbers shown in the app and reports."""


def metric_explanation(result, key):
    """Describe a computed value without changing its underlying statistic."""
    name = result.test_name if result is not None else ""
    if key == "statistic":
        if name.startswith("benford_"):
            return "Compares all digit counts with the expected pattern; it can grow with more records."
        if name.startswith("ks_"):
            return "The largest gap between the data and the chosen reference distribution."
        if name.startswith("kuiper_"):
            return "Combines the largest gaps above and below the chosen reference distribution."
        if name == "regression":
            return "Tests whether the selected predictors relate to the response; it is not a record-level anomaly measure."
    if key == "effect_size":
        if name.startswith("benford_"):
            return "Average absolute difference between observed and expected digit shares (MAD)."
        if name in ("ks_one_sample", "kuiper_one_sample"):
            return "Size of the distribution gap, measured on a zero-to-one scale."
        if name == "regression":
            return "Share of variation explained by the fitted relationship (R²)."
    descriptions = {
        "n": "Number of eligible observations used for this particular check.",
        "p_value": "Small values suggest the data differ from the chosen reference; not a probability of wrongdoing.",
        "p_value_method": "How this p-value was estimated: a mathematical approximation or seeded simulation.",
        "adjusted_p": "P-value corrected because several checks were run on the same dataset.",
        "reference_band": "Shaded chart range chosen for each digit. Changing it affects only the chart highlights.",
        "conformity": "A descriptive digit-pattern guideline, not a determination about individual rows.",
        "positive_orders_span": "How many powers of ten separate the smallest and largest positive values.",
        "ecdf_95_band": "Rough uncertainty guide for a cumulative distribution with this sample size.",
        "D_plus": "Largest point where the data's cumulative share is above the reference.",
        "D_minus": "Largest point where the data's cumulative share is below the reference.",
        "r_squared": "Share of the response's variation explained by the fitted relationship.",
        "adj_r_squared": "Model fit after allowing for the number of predictors.",
        "breusch_pagan": "Checks whether prediction errors become more or less spread out.",
        "normality": "Checks whether the prediction errors have the expected shape.",
        "flagged_rows": "Rows whose regression diagnostics suggest a closer look; they may be legitimate.",
        "index": "A relative summary of this dataset, not a probability.",
        "review_share": "Share of records above the score threshold, relative to this dataset.",
        "normalized_effect": "Combined size of distribution differences, scaled to zero through one.",
        "adjusted_significance": "Combined reference-test signal after accounting for multiple checks.",
        "threshold": "Relative score above which records are listed for review.",
        "reviewed_records": "Count of records that reach the review score.",
    }
    return descriptions.get(
        key, "A directly computed value from this column and check."
    )
