"""Presentation and export of positional record flags."""

import streamlit as st


def render_flagged_records(df, bundle):
    """Filter and sort flags without recomputing statistics."""
    positions = sorted(
        {p for result in bundle.results for p in result.flagged_indices}
        | (set(bundle.ml_output.flagged_indices) if bundle.ml_output else set())
    )
    flagged = df.iloc[positions].copy()
    flagged.insert(0, "row_position", positions)
    for result in bundle.results:
        if result.flagged_indices:
            flagged[f"flag:{result.test_name}:{result.column}"] = flagged[
                "row_position"
            ].isin(result.flagged_indices)
    if bundle.ml_output and bundle.ml_output.scores:
        flagged["relative_score"] = [bundle.ml_output.scores[p] for p in positions]
        flagged["tests_only_score"] = [bundle.ml_output.baseline[p] for p in positions]
        minimum = st.slider("Minimum relative score", 0.0, 1.0, 0.0, 0.05)
        flagged = flagged.loc[flagged["relative_score"] >= minimum]
    sort_by = st.selectbox("Sort records by", list(flagged.columns), index=0)
    flagged = flagged.sort_values(sort_by, ascending=sort_by != "relative_score")
    st.dataframe(flagged, width="stretch")
    st.download_button(
        "Download flagged rows CSV",
        flagged.to_csv(index=False),
        "flagged_rows.csv",
        "text/csv",
    )
