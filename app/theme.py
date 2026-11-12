"""Small UI theme shared with the report palette."""

from fraudscope.report.typography import font_face_css

CSS = f"""<style>{font_face_css()}
html, body, p, li, label, button, input, textarea,
[data-testid="stMarkdownContainer"], [data-testid="stMetric"],
[data-testid="stSidebar"], [data-testid="stDataFrame"]
{{font-family:'Source Code Pro',sans-serif !important}}
.block-container{{max-width:1250px;padding-top:2rem}}
div[data-testid="stMetric"]{{border:1px solid #a4b9c5;border-radius:10px;padding:1rem}}
.footnote{{color:#527385;border-top:1px solid #a4b9c5;margin-top:3rem;padding-top:1rem}}</style>"""
