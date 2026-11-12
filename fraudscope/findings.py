"""Deterministic, auditable statements backed by structured result numbers."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Finding:
    id: str
    section: str
    column: str | None
    metric: str
    value: float
    reference: float | None
    direction: str
    severity: str
    statement: str
    caveat_ids: list[str]
    figure_ref: str | None = None


def severity(value, notable, strong):
    """Inclusive, descriptive effect thresholds; not hypothesis-test cutoffs."""
    return "strong" if value >= strong else "notable" if value >= notable else "info"


class FindingsEngine:
    """Convert tests and optional scoring to rule-based, quantitative findings."""

    def build(self, results, ml_output=None):
        findings = []
        for i, result in enumerate(results):
            if result.p_value is None:
                continue
            section = result.test_name
            column = result.column
            caveats = [f"{i}:{j}" for j in range(len(result.caveats))]
            if section.startswith("benford_"):
                mad = result.details["mad"]
                thresholds = {
                    "first": (0.006, 0.015),
                    "second": (0.008, 0.012),
                    "first_two": (0.0012, 0.0022),
                    "last_two": (0.005, 0.01),
                }
                low, high = thresholds[result.details["mode"]]
                label = result.details["conformity"]
                findings.append(
                    Finding(
                        f"{i}:mad",
                        section,
                        column,
                        "MAD",
                        mad,
                        high,
                        "above" if mad >= high else "below",
                        severity(mad, low, high),
                        f"{column}: MAD {mad:.4f}; conformity {label}; n={result.n}.",
                        caveats,
                        f"{i}:primary",
                    )
                )
                for bin_, observed, expected, z in zip(
                    result.details["bins"],
                    result.details["observed"],
                    result.details["expected"],
                    result.details["z_scores"],
                ):
                    if z > 1.96:
                        findings.append(
                            Finding(
                                f"{i}:bin:{bin_}",
                                section,
                                column,
                                "digit_z",
                                float(z),
                                1.96,
                                "above",
                                severity(float(z), 1.96, 3),
                                f"{column} digit {bin_}: observed {observed:.2%}, expected {expected:.2%}, z={z:.2f}.",
                                caveats,
                                f"{i}:primary",
                            )
                        )
            elif section in ("ks_one_sample", "kuiper_one_sample"):
                value = float(result.statistic)
                bound = 0.2 if section.startswith("ks") else 0.3
                loc = (
                    result.details["x_plus"]
                    if result.details["D_plus"] >= result.details["D_minus"]
                    else result.details["x_minus"]
                )
                metric = "D" if section.startswith("ks") else "V"
                findings.append(
                    Finding(
                        f"{i}:{metric}",
                        section,
                        column,
                        metric,
                        value,
                        bound,
                        "above" if value >= bound else "below",
                        severity(value, bound / 2, bound),
                        f"{column}: {metric}={value:.4f} near value {loc:.4g}; p={result.p_value:.4g} ({result.p_value_method}); n={result.n}.",
                        caveats,
                        f"{i}:ecdf",
                    )
                )
            elif section == "regression":
                rate = len(result.flagged_indices) / result.n
                findings.append(
                    Finding(
                        f"{i}:rate",
                        section,
                        column,
                        "flag_rate",
                        rate,
                        0.1,
                        "above" if rate >= 0.1 else "below",
                        severity(rate, 0.05, 0.1),
                        f"{column}: R²={result.details['r_squared']:.4f}; {len(result.flagged_indices)}/{result.n} rows ({rate:.1%}) flagged by diagnostics.",
                        caveats,
                        f"{i}:residuals",
                    )
                )
        if ml_output is not None and ml_output.scores:
            share = ml_output.components["review_share"]
            findings.append(
                Finding(
                    "ml:share",
                    "anomaly_scoring",
                    None,
                    "review_share",
                    share,
                    ml_output.threshold,
                    "above" if share >= ml_output.threshold else "below",
                    "info",
                    f"{len(ml_output.flagged_indices)} records ({share:.1%}) score at or above {ml_output.threshold:.2f}; dataset index {ml_output.index:.2f}/100 (relative, not a probability).",
                    ["ml:ranking"],
                    "ml:scores",
                )
            )
        return findings


def findings_document(findings):
    """Stable schema for downstream consumers; text is generated only here."""
    return {"schema_version": "1.0", "findings": [asdict(f) for f in findings]}
