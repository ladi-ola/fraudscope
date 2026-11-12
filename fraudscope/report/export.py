"""Offline exports; images are embedded and uploaded data is never written."""

import io
import json
from importlib import resources

from jinja2 import BaseLoader, Environment
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from fraudscope.findings import findings_document

from .builder import build_report


def html_report(bundle):
    """Render a self-contained HTML report from structured numerical sections."""
    template = (
        resources.files("fraudscope.report")
        .joinpath("templates/report.html.j2")
        .read_text()
    )
    return (
        Environment(loader=BaseLoader(), autoescape=True)
        .from_string(template)
        .render(**build_report(bundle))
    )


def findings_json(bundle):
    return json.dumps(
        findings_document(bundle.findings, bundle.results), indent=2, allow_nan=False
    )


def pdf_report(bundle):
    """Create a styled PDF using only Python packages (ReportLab)."""
    import base64
    from xml.sax.saxutils import escape

    data = build_report(bundle)
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=(595, 842), leftMargin=42, rightMargin=42)
    styles = getSampleStyleSheet()
    story = []

    def paragraph(text, style="Normal"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 7))

    def picture(encoded, width=480):
        raw = io.BytesIO(base64.b64decode(encoded))
        reader = ImageReader(raw)
        w, h = reader.getSize()
        story.append(Image(raw, width=width, height=width * h / w))

    paragraph("Statistical anomaly analysis", "Title")
    paragraph(
        f"Dataset: {data['metadata']['dataset_name']} · {data['metadata']['timestamp']}"
    )
    paragraph("Results indicate statistical anomalies, not proof of wrongdoing.")
    paragraph(
        f"Anomaly index: {data['index']} / 100 (relative indicator, not a probability)"
        if data["index"] is not None
        else "Anomaly index unavailable for these data."
    )
    if data["scoring_warning"]:
        paragraph("Scoring note: " + data["scoring_warning"])
    paragraph(f"Components: {data['components']}")
    picture(data["cover"])
    paragraph("Executive summary", "Heading1")
    for finding in data["summary"]:
        paragraph(f"{finding.severity}: {finding.statement}")
    paragraph("Dataset profile and applicability", "Heading1")
    paragraph(
        "No fixed sample-size minimum applies. The profile gives eligible counts and observed value ranges. Smaller counts imply wider uncertainty; narrow ranges can limit the meaning of digit comparisons. The conservative 95% ECDF uncertainty bound is min(1, sqrt(log(40)/(2n))) for n independent finite observations, not a fitted-reference p-value."
    )
    paragraph(str(data["profile"]))
    for section in data["sections"]:
        paragraph(section["title"], "Heading1")
        paragraph("NUMBERS", "Heading2")
        rows = [["Metric", "Value"]] + [
            [str(k), str(v)[:250]] for k, v in section["numbers"].items()
        ]
        table = Table(rows, colWidths=[130, 350], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#287d9b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                ]
            )
        )
        story.append(table)
        paragraph("CHART", "Heading2")
        for _, image, caption in section["charts"]:
            picture(image)
            paragraph("How to read this: " + caption)
        paragraph("STATISTICAL READING", "Heading2")
        for reading in section["reading"] + section["caveats"]:
            paragraph(reading)
    paragraph("Flagged-records appendix", "Heading1")
    paragraph(str(data["flagged_rows"]))
    paragraph("Methodology and limitations", "Heading1")
    paragraph(
        "There is no fixed row-count or magnitude-span eligibility threshold. The uncertainty bound decreases with the number of independent observations. Benford chi-square calibration is unreliable if expected bin counts fall below five, and results are warned rather than suppressed. Benford MAD uses Nigrini descriptive cutoffs; KS D and Kuiper V use descriptive design scales of 0.2 and 0.3, not universal critical values. P-values may be asymptotic or Monte Carlo; valid run p-values use Benjamini-Hochberg adjustment. Isolation Forest is unsupervised and scores are within-dataset ranks. The index is 100 × (0.2 × review share + 0.5 × normalized distribution effect + 0.3 × adjusted significance)."
    )
    paragraph("Reproducibility", "Heading1")
    paragraph(str(data["metadata"]))
    paragraph("Packages: " + str(data["metadata"]["package_versions"]))
    doc.build(story)
    return output.getvalue()
