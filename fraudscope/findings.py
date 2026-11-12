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
                        f"For {column}, the average gap from the expected digit pattern is {mad:.4f} (MAD). The guideline label is {label}, based on {result.n} values.",
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
                                f"For {column}, digit {bin_} appears in {observed:.2%} of values, compared with {expected:.2%} expected (standardized difference {z:.2f}).",
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
                        f"For {column}, the largest distribution difference is {value:.4f} ({metric}) around value {loc:.4g}. The p-value is {result.p_value:.4g} (calculated using {result.p_value_method}), from {result.n} values.",
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
                        f"For {column}, the fitted relationship explains {result.details['r_squared']:.4f} of the variation (R²); {len(result.flagged_indices)} of {result.n} rows ({rate:.1%}) were marked for a closer look.",
                        caveats,
                        f"{i}:residuals",
                    )
                )
                for check in ("breusch_pagan", "normality"):
                    p = result.details[check]["p_value"]
                    findings.append(
                        Finding(
                            f"{i}:{check}",
                            section,
                            column,
                            check,
                            p,
                            0.05,
                            "below" if p < 0.05 else "above",
                            "notable" if p < 0.05 else "info",
                            f"For {column}, the {check.replace('_', ' ')} check gives a p-value of {p:.4g}, compared with 0.05 as a review guideline.",
                            caveats,
                            f"{i}:residuals",
                        )
                    )
                for group in result.details.get("flagged_groups", []):
                    item = result.details["groups"][group]
                    mean = item["mean_residual"]
                    findings.append(
                        Finding(
                            f"{i}:group:{group}",
                            section,
                            column,
                            "group_mean_residual",
                            mean,
                            0.0,
                            "above" if mean > 0 else "below",
                            "notable",
                            f"For {column}, group {group} differs from the model prediction by an average of {mean:.4g} across {item['n']} rows; this is worth checking in context.",
                            caveats,
                            f"{i}:groups",
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
                    f"{len(ml_output.flagged_indices)} records ({share:.1%}) have a relative score of at least {ml_output.threshold:.2f}. The overall index is {ml_output.index:.2f}/100; neither number is a probability.",
                    ["ml:ranking"],
                    "ml:scores",
                )
            )
            from collections import Counter

            top = Counter(
                name for names in ml_output.contributions.values() for name in names
            ).most_common(3)
            for rank, (feature, count) in enumerate(top):
                findings.append(
                    Finding(
                        f"ml:feature:{rank}",
                        "anomaly_scoring",
                        None,
                        "contributing_feature_count",
                        float(count),
                        0.0,
                        "above",
                        "info",
                        f"The measurement {feature} stood out for {count} reviewed records.",
                        ["ml:ranking"],
                        "ml:scores",
                    )
                )
        return findings


def findings_document(findings, results=()):
    """Stable schema for downstream consumers; text is generated only here."""
    caveats = {
        f"{i}:{j}": text
        for i, result in enumerate(results)
        for j, text in enumerate(result.caveats)
    }
    caveats["ml:ranking"] = (
        "Unusual records may reflect legitimate collection or entry conventions; scores are relative."
    )
    return {
        "schema_version": "1.0",
        "findings": [asdict(f) for f in findings],
        "caveats": caveats,
    }
