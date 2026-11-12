"""In-memory Streamlit interface; statistical logic lives in fraudscope.service."""

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.components.records import render_flagged_records
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
from fraudscope.service import profile_dataset, run_analysis


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
    if not raw:
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
    df.columns = df.columns.astype(str).str.strip()
    if not df.columns.is_unique:
        st.error("Column names must be unique after removing surrounding whitespace.")
        df = None

if df is not None:
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(
            df[col]
        ):
            trimmed = df[col].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )
            numeric_values = pd.to_numeric(trimmed, errors="coerce")
            df[col] = (
                numeric_values
                if numeric_values.notna().sum() == trimmed.notna().sum()
                else trimmed
            )

if df is not None:
    if df.empty or len(df.columns) == 0:
        st.warning("No data rows and columns were found.")
    else:
        st.subheader("Preview and profile")
        st.dataframe(df.head(), width="stretch")
        st.write(f"{len(df):,} rows · {len(df.columns)} columns")
        st.dataframe(
            pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum()})
        )
        numeric = list(df.select_dtypes(include="number").columns)
        observed_profile = profile_dataset(df)["column_profiles"]
        for col in numeric:
            ok, reasons = benford_applicable(df[col])
            info = observed_profile.get(col, {})
            st.caption(
                f"{col}: {info.get('finite_count', 'unavailable')} finite, "
                f"{info.get('positive_count', 'unavailable')} positive; "
                f"range {info.get('finite_min', 'unavailable')} to {info.get('finite_max', 'unavailable')}; "
                f"positive magnitude span {info.get('positive_orders_span', 'unavailable')} orders · "
                f"digit suitability {'OK' if ok else 'caution'} · "
                f"{'; '.join(reasons) if reasons else 'No heuristic concerns'}"
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
                    index = (
                        bundle.ml_output.index
                        if bundle.ml_output and bundle.ml_output.scores
                        else None
                    )
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
                        st.subheader(f"{result.test_name}: {result.column}")
                        cards = st.columns(4)
                        cards[0].metric(
                            "Statistic",
                            (
                                f"{result.statistic:.4g}"
                                if isinstance(result.statistic, (int, float))
                                else str(result.statistic)
                            ),
                        )
                        cards[1].metric(
                            "Adjusted p",
                            (
                                f"{result.details['p_adjusted']:.4g}"
                                if "p_adjusted" in result.details
                                else "N/A"
                            ),
                        )
                        cards[2].metric(
                            "Effect size",
                            (
                                f"{result.effect_size:.4g}"
                                if result.effect_size is not None
                                else "N/A"
                            ),
                        )
                        cards[3].metric(
                            "Conformity", result.details.get("conformity", "N/A")
                        )
                    st.subheader("What stood out")
                    for finding in bundle.findings:
                        st.write(f"**{finding.severity}** · {finding.statement}")
                    with st.expander("Caveats"):
                        for result in bundle.results:
                            st.write(*result.caveats)
                with tabs[1]:
                    for i, result in enumerate(bundle.results):
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
                            with st.expander("Log-scale, deviation and mantissa views"):
                                for view, caption in (
                                    (
                                        "log",
                                        "The log scale makes small digit-bin proportions visible.",
                                    ),
                                    (
                                        "deviation",
                                        "Bars above zero exceed the expected proportion.",
                                    ),
                                    (
                                        "mantissa",
                                        "Compare the fractional log-magnitudes to uniform density.",
                                    ),
                                ):
                                    key = f"{i}:{view}"
                                    if key in bundle.figures:
                                        extra = bundle.figures[key]()
                                        st.pyplot(extra)
                                        plt.close(extra)
                                        st.caption("How to read this: " + caption)
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
                    render_flagged_records(df, bundle)
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
