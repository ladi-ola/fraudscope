"""Bundled offline typeface shared by the app, charts, and exported reports."""

import base64
from importlib import resources


def font_path(bold=False):
    """Return the locally packaged Source Code Pro TrueType font."""
    name = "SourceCodePro-Bold.ttf" if bold else "SourceCodePro-Regular.ttf"
    return resources.files("fraudscope.report").joinpath("fonts", name)


def font_face_css():
    """Embed both weights in CSS so a saved report needs no network or fonts."""
    rules = []
    for bold, weight in ((False, "400"), (True, "700")):
        path = font_path(bold)
        if not path.is_file():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        rules.append(
            "@font-face{font-family:'Source Code Pro';"
            f"src:url(data:font/ttf;base64,{encoded}) format('truetype');"
            f"font-style:normal;font-weight:{weight};font-display:swap}}"
        )
    return "\n".join(rules)


def style_figure(fig):
    """Apply bundled fonts to one figure without changing global matplotlib state."""
    from matplotlib.font_manager import FontProperties

    for ax in fig.axes:
        texts = [
            ax.title,
            ax.xaxis.label,
            ax.yaxis.label,
            *ax.get_xticklabels(),
            *ax.get_yticklabels(),
            *ax.texts,
        ]
        legend = ax.get_legend()
        if legend:
            texts.extend(legend.get_texts())
            texts.append(legend.get_title())
        for text in texts:
            bold = text is ax.title or text.get_fontweight() in (
                "bold",
                "semibold",
                600,
                700,
            )
            path = font_path(bold)
            options = (
                {"fname": str(path)} if path.is_file() else {"family": "sans-serif"}
            )
            text.set_fontproperties(FontProperties(size=text.get_fontsize(), **options))
    for text in fig.texts:
        path = font_path()
        options = {"fname": str(path)} if path.is_file() else {"family": "sans-serif"}
        text.set_fontproperties(FontProperties(size=text.get_fontsize(), **options))
    return fig
