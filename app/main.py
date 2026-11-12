"""In-memory Streamlit interface; statistical logic lives in fraudscope.service."""

import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.theme import CSS
from fraudscope.applicability import benford_applicable
from fraudscope.report.export import findings_json, html_report, pdf_report
from fraudscope.report.figures import (
    digit_figure,
    plot_ecdf_vs_reference,
    plot_group_residuals,
    plot_residuals,
    score_figure,
)
from fraudscope.runner import Config, RegressionSpec
from fraudscope.service import run_analysis

MAX_BYTES = int(os.environ.get("ANALYSIS_MAX_MB", "50")) * 1024 * 1024
MAX_ROWS = int(os.environ.get("ANALYSIS_MAX_ROWS", "100000"))


@st.cache_data(show_spinner=False)
def analyze(data, config_json, digest, name):
    """Data and serialized config form the cache key; uploads stay in memory."""
    options = json.loads(config_json)
    if options.get("regression"):
        options["regression"] = RegressionSpec(**options["regression"])
    return run_analysis(data, Config(**options), data_hash=digest, dataset_name=name)


def sample_data(injected=False):
    rng = np.random.default_rng(17)
    values = 10 ** rng.uniform(1, 6, 900)
    if injected:
        values[:100] = 55500
        values[100:180] = np.round(values[100:180] / 100) * 100
    return pd.DataFrame({"amount": values, "sequence": np.arange(len(values))})


st.set_page_config(page_title="Statistical anomaly analysis", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.title("Statistical anomaly analysis")
st.caption(
    "Uploads are processed in memory only. No external network calls are made for your data."
)
uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])
sample_choice = st.selectbox(
    "Or try sample data", ["None", "Expected digit distribution", "Injected deviations"]
)
if st.button("Try with sample data"):
    st.session_state["use_sample"] = True
df = None
name = "sample"
raw = None
if uploaded is not None:
    name = uploaded.name
    raw = uploaded.getvalue()
    if len(raw) > MAX_BYTES:
        st.error(
            f"This file exceeds the {MAX_BYTES // (1024 * 1024)} MB upload limit. Choose a smaller file."
        )
    elif not raw:
        st.error("This file is empty. Choose a CSV or XLSX with data rows.")
    else:
        try:
            df = (
                pd.read_csv(io.BytesIO(raw))
                if name.lower().endswith(".csv")
                else pd.read_excel(io.BytesIO(raw))
            )
        except (ValueError, UnicodeError, pd.errors.ParserError, OSError) as exc:
            st.error(f"Could not read the file: {exc}")
elif sample_choice != "None" or st.session_state.get("use_sample"):
    df = sample_data(sample_choice == "Injected deviations")
    name = (
        "injected-sample"
        if sample_choice == "Injected deviations"
        else "expected-sample"
    )

if df is not None:
    if df.empty or len(df.columns) == 0:
        st.warning("No data rows and columns were found.")
    elif len(df) > MAX_ROWS:
        st.error(f"This dataset exceeds the {MAX_ROWS:,} row limit.")
    else:
        st.subheader("Preview and profile")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"{len(df):,} rows · {len(df.columns)} columns")
        st.dataframe(
            pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum()})
        )
        numeric = list(df.select_dtypes(include="number").columns)
        for col in numeric:
            ok, reasons = benford_applicable(df[col])
            st.caption(
                f"{col}: digit suitability {'OK' if ok else 'caution'} · {'; '.join(reasons) if reasons else 'No heuristic concerns'}"
            )
        if not numeric:
            st.warning("Choose a dataset with at least one numeric column.")
        else:
            with st.sidebar:
                st.header("Configure")
                columns = st.multiselect(
                    "Numeric columns", numeric, default=numeric[:1]
                )
                modes = st.multiselect(
                    "Digit modes",
                    ["first", "second", "first_two", "last_two"],
                    default=["first"],
                )
                reference = st.selectbox(
                    "KS / Kuiper reference", ["None", "Benford mantissas"]
                )
                response = st.selectbox(
                    "Regression response (optional)", ["None", *numeric]
                )
                predictors = (
                    st.multiselect(
                        "Regression predictors", [x for x in numeric if x != response]
                    )
                    if response != "None"
                    else []
                )
                group = (
                    st.selectbox(
                        "Regression group",
                        ["None", *[x for x in df.columns if x != response]],
                    )
                    if response != "None"
                    else "None"
                )
                ml_enabled = st.checkbox("Anomaly scoring", value=True)
                seed = st.number_input("Random seed", value=17, step=1)
            if (
                st.button("Run analysis", type="primary")
                or st.session_state.pop("use_sample", False)
            ) and columns:
                from dataclasses import asdict

                regression = (
                    RegressionSpec(
                        response, predictors, group=None if group == "None" else group
                    )
                    if response != "None" and predictors
                    else None
                )
                options = Config(
                    columns,
                    modes,
                    {col: "benford" for col in columns} if reference != "None" else {},
                    regression=regression,
                    random_state=int(seed),
                    ml_enabled=ml_enabled,
                )
                digest = hashlib.sha256(
                    raw
                    if raw is not None
                    else pd.util.hash_pandas_object(df, index=False).values.tobytes()
                ).hexdigest()
                with st.spinner("Running independent analyses…"):
                    st.session_state["bundle"] = analyze(
                        df, json.dumps(asdict(options)), digest, name
                    )
                    st.session_state["digest"] = digest
            bundle = st.session_state.get("bundle")
            if (
                bundle
                and bundle.run_metadata["dataset_name"] == name
                and bundle.run_metadata["sha256"]
                == (
                    hashlib.sha256(
                        raw
                        if raw is not None
                        else pd.util.hash_pandas_object(
                            df, index=False
                        ).values.tobytes()
                    ).hexdigest()
                )
            ):
                for result in bundle.results:
                    for warning in result.warnings:
                        st.warning(f"{result.test_name} / {result.column}: {warning}")
                tabs = st.tabs(
                    [
                        "Overview",
                        "Benford",
                        "KS and Kuiper",
                        "Regression",
                        "Anomaly scoring",
                        "Flagged records",
                        "Report",
                    ]
                )
                with tabs[0]:
                    index = bundle.ml_output.index if bundle.ml_output else None
                    st.metric(
                        "Relative anomaly index",
                        f"{index}/100" if index is not None else "Unavailable",
                    )
                    st.caption(
                        "A descriptive within-dataset indicator, not a probability or evidence of intent."
                    )
                    if bundle.ml_output:
                        st.json(bundle.ml_output.components)
                    for result in bundle.results:
                        st.metric(
                            f"{result.test_name}: {result.column}",
                            (
                                str(round(result.effect_size, 4))
                                if result.effect_size is not None
                                else "N/A"
                            ),
                        )
                    st.subheader("What stood out")
                    for finding in bundle.findings:
                        st.write(f"**{finding.severity}** · {finding.statement}")
                    with st.expander("Caveats"):
                        for result in bundle.results:
                            st.write(*result.caveats)
                with tabs[1]:
                    for result in bundle.results:
                        if (
                            result.test_name.startswith("benford_")
                            and result.p_value is not None
                        ):
                            st.subheader(f"{result.column} · {result.details['mode']}")
                            fig = digit_figure(result)
                            st.pyplot(fig)
                            import matplotlib.pyplot as plt

                            plt.close(fig)
                            st.caption(
                                "How to read this: compare observed bars with the expected line and shaded range."
                            )
                            st.write(
                                "MAD:",
                                result.details["mad"],
                                "·",
                                result.details["conformity"],
                            )
                            st.dataframe(
                                pd.DataFrame(
                                    {
                                        key: result.details[key]
                                        for key in (
                                            "bins",
                                            "observed",
                                            "expected",
                                            "z_scores",
                                        )
                                    }
                                )
                            )
                            with st.expander("Interpretation"):
                                st.write(*result.caveats)
                with tabs[2]:
                    for result in bundle.results:
                        if (
                            result.test_name in ("ks_one_sample", "kuiper_one_sample")
                            and result.p_value is not None
                        ):
                            st.subheader(result.test_name)
                            fig = plot_ecdf_vs_reference(result)
                            st.pyplot(fig)
                            import matplotlib.pyplot as plt

                            plt.close(fig)
                            st.caption(
                                "How to read this: inspect the maximum cumulative distance from the reference."
                            )
                            st.write(
                                "Statistic:",
                                result.statistic,
                                "· p:",
                                result.p_value,
                                "· method:",
                                result.p_value_method,
                            )
                with tabs[3]:
                    for result in bundle.results:
                        if (
                            result.test_name == "regression"
                            and result.p_value is not None
                        ):
                            fig = plot_residuals(result)
                            st.pyplot(fig)
                            import matplotlib.pyplot as plt

                            plt.close(fig)
                            st.caption(
                                "How to read this: inspect outlying residuals and influential points."
                            )
                            st.json(
                                {
                                    k: result.details[k]
                                    for k in ("r_squared", "breusch_pagan", "normality")
                                }
                            )
                            st.write("Flagged rows:", result.flagged_indices[:30])
                            if "groups" in result.details:
                                fig = plot_group_residuals(result)
                                st.pyplot(fig)
                                plt.close(fig)
                                st.caption(
                                    "How to read this: compare residual spreads across groups."
                                )
                with tabs[4]:
                    ml = bundle.ml_output
                    if ml and ml.scores:
                        fig = score_figure(ml.scores)
                        st.pyplot(fig)
                        import matplotlib.pyplot as plt

                        plt.close(fig)
                        st.caption(
                            "How to read this: higher ranks are unusual relative to other rows here."
                        )
                        st.dataframe(
                            pd.DataFrame(
                                {
                                    "row_position": ml.flagged_indices,
                                    "score": [ml.scores[p] for p in ml.flagged_indices],
                                    "contributing_features": [
                                        ", ".join(ml.contributions[p])
                                        for p in ml.flagged_indices
                                    ],
                                }
                            )
                        )
                    else:
                        st.info(ml.warning if ml else "Anomaly scoring is off.")
                with tabs[5]:
                    positions = sorted(
                        {p for r in bundle.results for p in r.flagged_indices}
                        | (
                            set(bundle.ml_output.flagged_indices)
                            if bundle.ml_output
                            else set()
                        )
                    )
                    flagged = df.iloc[positions].copy()
                    flagged.insert(0, "row_position", positions)
                    for result in bundle.results:
                        if result.flagged_indices:
                            flagged[f"flag:{result.test_name}:{result.column}"] = (
                                flagged["row_position"].isin(result.flagged_indices)
                            )
                    if bundle.ml_output and bundle.ml_output.scores:
                        flagged["relative_score"] = [
                            bundle.ml_output.scores[p] for p in positions
                        ]
                        flagged["tests_only_score"] = [
                            bundle.ml_output.baseline[p] for p in positions
                        ]
                        min_score = st.slider(
                            "Minimum relative score", 0.0, 1.0, 0.0, 0.05
                        )
                        flagged = flagged.loc[flagged["relative_score"] >= min_score]
                    sort_by = st.selectbox(
                        "Sort records by", list(flagged.columns), index=0
                    )
                    flagged = flagged.sort_values(
                        sort_by, ascending=sort_by != "relative_score"
                    )
                    st.dataframe(flagged, use_container_width=True)
                    st.download_button(
                        "Download flagged rows CSV",
                        flagged.to_csv(index=False),
                        "flagged_rows.csv",
                        "text/csv",
                    )
                with tabs[6]:
                    st.download_button(
                        "Download HTML",
                        html_report(bundle),
                        "analysis.html",
                        "text/html",
                    )
                    st.download_button(
                        "Download PDF",
                        pdf_report(bundle),
                        "analysis.pdf",
                        "application/pdf",
                    )
                    st.download_button(
                        "Download findings.json",
                        findings_json(bundle),
                        "findings.json",
                        "application/json",
                    )
st.markdown(
    '<p class="footnote">Results indicate statistical anomalies, not proof of wrongdoing.</p>',
    unsafe_allow_html=True,
)
