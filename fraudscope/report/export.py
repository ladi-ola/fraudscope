"""Offline exports; images are embedded and uploaded data is never written."""

import io
import json
from importlib import resources

from jinja2 import BaseLoader, Environment
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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
from .typography import font_path


def _pdf_image_size(image_width, image_height, max_width=480, max_height=650):
    """Fit report figures on one PDF page while preserving their proportions."""
    scale = min(max_width / image_width, max_height / image_height)
    return image_width * scale, image_height * scale


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
    bundled = font_path().is_file() and font_path(True).is_file()
    if bundled and "SourceCodePro" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("SourceCodePro", str(font_path())))
        pdfmetrics.registerFont(TTFont("SourceCodePro-Bold", str(font_path(True))))
    regular_font = "SourceCodePro" if bundled else "Helvetica"
    bold_font = "SourceCodePro-Bold" if bundled else "Helvetica-Bold"
    styles = getSampleStyleSheet()
    for name in ("Normal", "BodyText"):
        styles[name].fontName = regular_font
    for name in ("Title", "Heading1", "Heading2", "Heading3"):
        styles[name].fontName = bold_font
    story = []

    def paragraph(text, style="Normal"):
        story.append(Paragraph(escape(str(text)), styles[style]))
        story.append(Spacer(1, 7))

    def picture(encoded, width=480):
        raw = io.BytesIO(base64.b64decode(encoded))
        reader = ImageReader(raw)
        w, h = reader.getSize()
        display_width, display_height = _pdf_image_size(w, h, max_width=width)
        story.append(Image(raw, width=display_width, height=display_height))

    paragraph("Statistical anomaly analysis", "Title")
    paragraph(
        f"Dataset: {data['metadata']['dataset_name']} · {data['metadata']['timestamp']}"
    )
    paragraph(
        "Patterns in this report are worth reviewing, not proof of wrongdoing. Scores rank records within this dataset; they are not probabilities."
    )
    paragraph(
        f"Anomaly index: {data['index']} / 100 (relative indicator, not a probability)"
        if data["index"] is not None
        else "Anomaly index unavailable for these data."
    )
    if data["scoring_warning"]:
        paragraph("Scoring note: " + data["scoring_warning"])
    if data["components"]:
        parts = data["components"]
        paragraph(
            f"Index breakdown: {parts['review_share']:.1%} of records above the review score; combined size of differences {parts['normalized_effect']:.2f}/1; combined reference-test signal {parts['adjusted_significance']:.2f}/1."
        )
    picture(data["cover"])
    paragraph("Main points", "Heading1")
    for finding in data["summary"]:
        paragraph(f"{finding.severity}: {finding.statement}")
    paragraph("About the dataset", "Heading1")
    paragraph(
        "There is no fixed minimum number of rows. With fewer usable values, results are less precise. A narrow value range can make digit patterns less informative."
    )
    paragraph(
        f"{data['profile']['rows']} rows and {data['profile']['columns']} columns."
    )
    for name, profile in data["profile"]["column_profiles"].items():
        paragraph(
            f"{name}: {profile.get('finite_count', 'Unavailable')} usable numbers; {profile.get('positive_count', 'Unavailable')} positive; {profile['missing']} missing. Values range from {profile.get('finite_min')} to {profile.get('finite_max')}. Positive values range from {profile.get('positive_min')} to {profile.get('positive_max')} ({profile.get('positive_orders_span')} powers of ten). Sampling-uncertainty guide: +/- {profile.get('ecdf_95_band')}. Digit check: {profile['benford']}; reference check: {profile['continuous']}."
        )
        for reason in profile["benford_reasons"] + profile["continuous_reasons"]:
            paragraph("Keep in mind: " + reason)
    for section in data["sections"]:
        paragraph(section["title"], "Heading1")
        paragraph("NUMBERS - What we measured", "Heading2")
        paragraph(
            "A p-value describes how surprising a result would be if the chosen reference were a good match; it is not a probability of wrongdoing. The size of the difference shows how far the data are from that reference."
        )
        rows = [["Measure", "Value"]] + [
            [str(label), str(value)[:250]] for label, value in section["number_rows"]
        ]
        table = Table(rows, colWidths=[130, 350], repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#287d9b")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("FONTNAME", (0, 0), (-1, -1), regular_font),
                    ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ]
            )
        )
        story.append(table)
        if "bins" in section["numbers"]:
            numbers = section["numbers"]
            detail_rows = [["Digit", "Seen in data", "Expected", "Difference (z)"]]
            detail_rows += [
                [str(digit), f"{observed:.2%}", f"{expected:.2%}", f"{z:.2f}"]
                for digit, observed, expected, z in zip(
                    numbers["bins"],
                    numbers["observed"],
                    numbers["expected"],
                    numbers["z_scores"],
                )
            ]
            detail = Table(detail_rows, colWidths=[60, 130, 130, 130], repeatRows=1)
            detail.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("FONTNAME", (0, 0), (-1, -1), regular_font),
                        ("FONTNAME", (0, 0), (-1, 0), bold_font),
                    ]
                )
            )
            story.append(detail)
        paragraph("CHART - What the pictures show", "Heading2")
        for _, image, caption in section["charts"]:
            picture(image)
            paragraph("How to read this: " + caption)
        paragraph("STATISTICAL READING - What this may mean", "Heading2")
        for reading in section["reading"] + section["caveats"]:
            paragraph(reading)
    paragraph("Rows to review", "Heading1")
    paragraph(
        "Zero-based row positions in the uploaded data: " + str(data["flagged_rows"])
    )
    paragraph("How to interpret this report", "Heading1")
    paragraph(
        "Digit checks compare shares of each digit with a reference pattern. Distribution checks compare the overall shape of the data with a chosen reference. A relationship check compares observed values with model predictions. Scores rank unusual records only within this dataset. Smaller samples and rounding can affect results."
    )
    paragraph("Technical methods and thresholds", "Heading2")
    paragraph(
        "For n independent finite observations, the conservative 95% Dvoretzky-Kiefer-Wolfowitz ECDF bound is min(1, sqrt(log(40)/(2n))). Positive-value span is log10(max) - log10(min). Benford chi-square results are warned if expected digit counts fall below five; MAD labels follow Nigrini guidelines. KS D and Kuiper V use descriptive scales of 0.2 and 0.3, not universal thresholds. P-values use the method shown in each table and receive Benjamini-Hochberg adjustment across tests. The index is 100 x (0.2 x review share + 0.5 x normalized difference + 0.3 x adjusted reference-test signal)."
    )
    paragraph("How to reproduce this run", "Heading1")
    paragraph(str(data["metadata"]))
    paragraph("Packages: " + str(data["metadata"]["package_versions"]))
    doc.build(story)
    return output.getvalue()
