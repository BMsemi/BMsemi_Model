from __future__ import annotations

import html
import math
from pathlib import Path
from textwrap import fill

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path("results") / "figures"

COLORS = {
    "ink": "#111827",
    "navy": "#1b1b32",
    "panel": "#f4f5f7",
    "blue": "#cfe1f9",
    "blue_dark": "#2f67a1",
    "green": "#d8ecd5",
    "green_dark": "#155c24",
    "orange": "#d45a1a",
    "yellow": "#fff1bf",
    "purple": "#8a1977",
    "red": "#f7caca",
    "gray": "#e6e6e9",
    "line": "#222222",
}


def font_path(bold: bool = False) -> str | None:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


FONT_REGULAR = font_path(False)
FONT_BOLD = font_path(True)


def load_font(size: int, bold: bool, scale: int):
    path = FONT_BOLD if bold else FONT_REGULAR
    if path:
        return ImageFont.truetype(path, size * scale)
    return ImageFont.load_default()


def rgb(color: str):
    named = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "none": None,
    }
    if color in named:
        return named[color]
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


class Diagram:
    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        self.scale = 2
        self.img = Image.new("RGB", (width * self.scale, height * self.scale), "white")
        self.draw = ImageDraw.Draw(self.img)
        self.svg: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            "<defs>",
            '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">',
            '<path d="M0,0 L8,4 L0,8 Z" fill="context-stroke"/>',
            "</marker>",
            "</defs>",
            '<rect width="100%" height="100%" fill="white"/>',
        ]

    def box(self, xy):
        return tuple(int(round(v * self.scale)) for v in xy)

    def font(self, size: int, bold: bool = False):
        return load_font(size, bold, self.scale)

    def text(
        self,
        x: float,
        y: float,
        text: str,
        *,
        size: int = 11,
        color: str = "black",
        bold: bool = False,
        ha: str = "center",
        va: str = "center",
        width: float | None = None,
        rotation: float = 0,
    ):
        if width:
            text = fill(str(text), max(6, int(width / (size * 0.55))))
        lines = str(text).splitlines()
        font = self.font(size, bold)
        s = self.scale
        line_h = int(size * 1.22 * s)

        if rotation:
            bbs = [self.draw.textbbox((0, 0), line, font=font) for line in lines]
            tw = max((bb[2] - bb[0] for bb in bbs), default=1)
            th = max(line_h * len(lines), 1)
            pad = 12 * s
            patch = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (255, 255, 255, 0))
            pd = ImageDraw.Draw(patch)
            yy = pad
            for line, bb in zip(lines, bbs):
                lw = bb[2] - bb[0]
                pd.text(((tw + 2 * pad - lw) / 2, yy), line, fill=rgb(color), font=font)
                yy += line_h
            rot = patch.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
            self.img.paste(rot, (int(x * s - rot.size[0] / 2), int(y * s - rot.size[1] / 2)), rot)
            self.svg.append(
                f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-family="Arial, sans-serif" '
                f'font-weight="{"700" if bold else "400"}" text-anchor="middle" dominant-baseline="central" '
                f'transform="rotate({rotation} {x} {y})">{html.escape(text)}</text>'
            )
            return

        bbs = [self.draw.textbbox((0, 0), line, font=font) for line in lines]
        widths = [bb[2] - bb[0] for bb in bbs]
        total_h = line_h * len(lines)
        top = y * s if va == "top" else y * s - total_h if va == "bottom" else y * s - total_h / 2
        for i, line in enumerate(lines):
            if ha == "left":
                xx = x * s
            elif ha == "right":
                xx = x * s - widths[i]
            else:
                xx = x * s - widths[i] / 2
            self.draw.text((xx, top + i * line_h), line, fill=rgb(color), font=font)

        anchor = {"left": "start", "right": "end"}.get(ha, "middle")
        start_y = y + size if va == "top" else y - (len(lines) - 1) * size * 0.6
        tspans = []
        for i, line in enumerate(lines):
            dy = "0" if i == 0 else f"{size * 1.2}"
            tspans.append(f'<tspan x="{x}" dy="{dy}">{html.escape(line)}</tspan>')
        self.svg.append(
            f'<text x="{x}" y="{start_y}" font-size="{size}" fill="{color}" font-family="Arial, sans-serif" '
            f'font-weight="{"700" if bold else "400"}" text-anchor="{anchor}">{"".join(tspans)}</text>'
        )

    def rounded(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str = "",
        *,
        fc: str = "white",
        ec: str = COLORS["line"],
        lw: float = 1,
        radius: float = 8,
        size: int = 11,
        color: str = "black",
        bold: bool = False,
        rotation: float = 0,
        wrap: bool = True,
    ):
        outline = None if ec == "none" else rgb(ec)
        self.draw.rounded_rectangle(
            self.box((x, y, x + w, y + h)),
            radius=int(radius * self.scale),
            fill=rgb(fc),
            outline=outline,
            width=max(1, int(lw * self.scale)) if outline else 0,
        )
        stroke = "none" if ec == "none" else ec
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" ry="{radius}" '
            f'fill="{fc}" stroke="{stroke}" stroke-width="{lw}"/>'
        )
        if text:
            self.text(
                x + w / 2,
                y + h / 2,
                text,
                size=size,
                color=color,
                bold=bold,
                width=(w - 10 if wrap and not rotation else None),
                rotation=rotation,
            )

    def rect(self, x, y, w, h, text="", *, fc="white", ec=COLORS["line"], lw=1, size=11, bold=False, color="black"):
        outline = None if ec == "none" else rgb(ec)
        self.draw.rectangle(
            self.box((x, y, x + w, y + h)),
            fill=rgb(fc),
            outline=outline,
            width=max(1, int(lw * self.scale)) if outline else 0,
        )
        stroke = "none" if ec == "none" else ec
        self.svg.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fc}" stroke="{stroke}" stroke-width="{lw}"/>'
        )
        if text:
            self.text(x + w / 2, y + h / 2, text, size=size, bold=bold, color=color, width=w - 8)

    def line(self, x1, y1, x2, y2, *, color=COLORS["line"], lw=1.2, dashed=False):
        if dashed:
            dx, dy = x2 - x1, y2 - y1
            dist = max(1, math.hypot(dx, dy))
            pos = 0.0
            while pos < dist:
                end = min(dist, pos + 8)
                a, b = pos / dist, end / dist
                self.draw.line(
                    self.box((x1 + dx * a, y1 + dy * a, x1 + dx * b, y1 + dy * b)),
                    fill=rgb(color),
                    width=max(1, int(lw * self.scale)),
                )
                pos += 14
        else:
            self.draw.line(self.box((x1, y1, x2, y2)), fill=rgb(color), width=max(1, int(lw * self.scale)))
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{lw}"{dash}/>')

    def arrow(self, x1, y1, x2, y2, *, color=COLORS["line"], lw=1.2, dashed=False):
        if dashed:
            dx, dy = x2 - x1, y2 - y1
            dist = max(1, math.hypot(dx, dy))
            pos = 0.0
            while pos < dist:
                end = min(dist, pos + 8)
                a, b = pos / dist, end / dist
                self.draw.line(
                    self.box((x1 + dx * a, y1 + dy * a, x1 + dx * b, y1 + dy * b)),
                    fill=rgb(color),
                    width=max(1, int(lw * self.scale)),
                )
                pos += 14
        else:
            self.draw.line(self.box((x1, y1, x2, y2)), fill=rgb(color), width=max(1, int(lw * self.scale)))
        angle = math.atan2(y2 - y1, x2 - x1)
        head_len = 9
        head_w = 7
        pts = [
            (x2, y2),
            (x2 - head_len * math.cos(angle) + head_w * math.sin(angle) / 2,
             y2 - head_len * math.sin(angle) - head_w * math.cos(angle) / 2),
            (x2 - head_len * math.cos(angle) - head_w * math.sin(angle) / 2,
             y2 - head_len * math.sin(angle) + head_w * math.cos(angle) / 2),
        ]
        self.polygon(pts, fc=color, ec=color)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{lw}" marker-end="url(#arrow)"{dash}/>'
        )

    def polygon(self, pts, *, fc="black", ec="none", lw=1):
        self.draw.polygon([(int(x * self.scale), int(y * self.scale)) for x, y in pts], fill=rgb(fc), outline=None if ec == "none" else rgb(ec))
        stroke = "none" if ec == "none" else ec
        points = " ".join(f"{x},{y}" for x, y in pts)
        self.svg.append(f'<polygon points="{points}" fill="{fc}" stroke="{stroke}" stroke-width="{lw}"/>')

    def save(self, name: str):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        png = OUT_DIR / f"{name}.png"
        svg = OUT_DIR / f"{name}.svg"
        self.img.save(png)
        svg.write_text("\n".join(self.svg + ["</svg>"]), encoding="utf-8")
        return png, svg


def header(d: Diagram, title: str):
    d.rounded(60, 28, d.w - 120, 54, title, fc=COLORS["navy"], ec=COLORS["navy"], radius=5, color="white", size=20, bold=True)


def chip(d: Diagram, x, y, w, h, text, fc, *, size=11, color="black", bold=False):
    d.rounded(x, y, w, h, text, fc=fc, ec="#52708a", radius=6, size=size, color=color, bold=bold)


def transformer_block(d: Diagram, x, y):
    d.rounded(x, y, 315, 120, fc=COLORS["gray"], ec="none", radius=10)
    labels = [("Input", "black"), ("Attention", "#37528a"), ("MLP", "#f26c2c"), ("Norm", "#43a436")]
    bx = x + 36
    for i, (txt, fc) in enumerate(labels):
        d.rounded(bx + i * 70, y + 25, 36, 70, txt, fc=fc, ec="none", radius=5, color="white", size=12, rotation=90, wrap=False)
        if i < 3:
            d.arrow(bx + i * 70 + 36, y + 60, bx + (i + 1) * 70, y + 60)


def figure_transformer_dataflow():
    d = Diagram(1500, 890)
    d.text(70, 55, "BF16 x BF16 SAR-TDC transformer dataflow", size=20, bold=True, ha="left")

    d.rounded(70, 100, 250, 105, fc=COLORS["gray"], ec="none", radius=8)
    d.text(95, 152, "Token\nembedding", size=15, ha="left")
    d.polygon([(230, 115), (262, 135), (262, 170), (230, 190)], fc="#38548c")
    d.polygon([(300, 115), (268, 135), (268, 170), (300, 190)], fc="#38548c")
    transformer_block(d, 415, 92)
    transformer_block(d, 840, 92)
    d.arrow(320, 152, 415, 152)
    d.arrow(730, 152, 840, 152)
    d.arrow(1155, 152, 1220, 152)
    d.line(500, 212, 395, 295, color="#999999", dashed=True)
    d.line(1000, 212, 1240, 295, color="#999999", dashed=True)

    d.rounded(60, 310, 350, 500, fc=COLORS["panel"], ec="none", radius=10)
    d.text(80, 334, "BF16 weight path for Q/K/V/O", size=12, bold=True, ha="left")
    d.rounded(90, 375, 46, 350, "BF16\nweight\nbits", fc="black", ec="none", color="white", size=12, rotation=90, wrap=False)
    d.rounded(158, 375, 56, 350, "component split", fc=COLORS["purple"], ec="none", color="white", size=12, rotation=90, wrap=False)
    d.arrow(136, 550, 158, 550)
    d.rounded(245, 365, 120, 56, "sign", fc="#f5d5e9", ec="#b96aa7", size=12, bold=True)
    d.rounded(245, 440, 120, 56, "8-bit exponent", fc="#efe8ff", ec="#9c7adb", size=12, bold=True)
    d.rounded(245, 515, 120, 74, "8-bit significand\n(hidden 1 + frac)", fc=COLORS["blue"], ec="#52708a", size=11, bold=True)
    d.rounded(245, 620, 120, 92, "conservative storage:\n4 BF16 nibbles x\n2 polarity = 8 banks", fc=COLORS["yellow"], ec="#d0a43b", size=10)
    d.arrow(214, 393, 245, 393)
    d.arrow(214, 468, 245, 468)
    d.arrow(214, 552, 245, 552)

    d.rounded(430, 310, 500, 500, fc=COLORS["panel"], ec="none", radius=10)
    d.rounded(480, 335, 380, 32, "BF16 activation input", fc="black", ec="none", color="white", size=14)
    d.arrow(670, 367, 670, 392)
    chip(d, 500, 392, 340, 48, "extract sign, exponent, 8-bit significand", "#49b7df", size=13, color="white", bold=True)
    d.arrow(670, 440, 670, 482)
    d.rounded(475, 482, 160, 62, "activation chunks:\nA_lo, A_hi", fc=COLORS["blue"], ec="#52708a", size=12, bold=True)
    d.rounded(700, 482, 160, 62, "weight chunks:\nW_lo, W_hi", fc=COLORS["blue"], ec="#52708a", size=12, bold=True)
    d.rounded(505, 590, 330, 70, "RRAM reads 4-bit significand chunk products\n(A_lo/A_hi) x (W_lo/W_hi)", fc=COLORS["green_dark"], ec="none", color="white", size=13, bold=True)
    d.arrow(555, 544, 555, 590)
    d.arrow(780, 544, 780, 590)
    d.rounded(505, 700, 330, 42, "SAR-TDC digitizes column partial sums", fc=COLORS["orange"], ec="none", color="white", size=13, bold=True)
    d.arrow(670, 660, 670, 700)
    d.text(445, 770, "RRAM cells sense chunk currents;\nthey do not store or multiply a whole BF16 value in one read.", size=11, ha="left", bold=True)

    d.rounded(890, 330, 45, 430, "sign/exponent sideband", fc=COLORS["purple"], ec="none", color="white", size=12, rotation=90, wrap=False)
    d.line(365, 393, 405, 393, color=COLORS["purple"], lw=1.0, dashed=True)
    d.line(405, 393, 405, 332, color=COLORS["purple"], lw=1.0, dashed=True)
    d.arrow(405, 332, 890, 332, color=COLORS["purple"], lw=1.0, dashed=True)
    d.text(878, 392, "metadata\nbus", size=9, color=COLORS["purple"], ha="center")
    d.rounded(878, 405, 12, 22, fc="#f5d5e9", ec=COLORS["purple"], radius=3)
    d.arrow(840, 416, 878, 416, color=COLORS["purple"], lw=1.4)

    d.rounded(965, 310, 475, 500, fc=COLORS["panel"], ec="none", radius=10)
    steps = [
        (1005, 345, 395, 44, "bucket by product exponent: E_a + E_w", COLORS["purple"]),
        (1005, 425, 395, 54, "32 read passes in demo:\n2 act chunks x 2 weight chunks x 8 exponent buckets", COLORS["yellow"]),
        (1005, 515, 395, 44, "digital shift by exponent + chunk offset", COLORS["green"]),
        (1005, 595, 395, 44, "accumulate aligned partials", COLORS["green"]),
        (1005, 675, 395, 54, "round/recombine to BF16 output", COLORS["blue"]),
    ]
    for i, (x, y, w, h, txt, fc) in enumerate(steps):
        chip(d, x, y, w, h, txt, fc, size=12, bold=True)
        if i < len(steps) - 1:
            d.arrow(x + w / 2, y + h, x + w / 2, steps[i + 1][1])
    d.line(670, 742, 670, 762)
    d.line(670, 762, 965, 762)
    d.line(965, 762, 965, 537)
    d.arrow(965, 537, 1005, 537)
    d.arrow(935, 367, 1005, 367, color=COLORS["purple"])
    d.text(75, 850, "Correct BF16 interpretation: BF16xBF16 is realized by chunked significand reads plus digital exponent scheduling, not by replacing INT8 with a one-pass BF16 analog MAC.", size=12, ha="left", bold=True)
    return d.save("bf16_sar_tdc_dataflow")


def figure_one_pe_path():
    d = Diagram(1600, 900)
    header(d, "BF16 x BF16 SAR-TDC One-PE Path")

    d.rounded(45, 115, 1510, 160, fc="white", ec=COLORS["navy"], lw=1.4, radius=8)
    d.rounded(45, 115, 1510, 38, "1. Scope", fc=COLORS["navy"], ec=COLORS["navy"], radius=8, color="white", bold=True, size=12)
    d.text(90, 178, "One PE workload: activation 1 x 10, weight 10 x 18, output 1 x 18.", size=16, bold=True, ha="left")
    d.text(90, 215, "BF16 operands are decomposed; sign/exponent stay digital, and only 4-bit significand chunks are sensed through RRAM.", size=16, bold=True, ha="left")
    d.text(90, 250, "This is the BF16 SAR-TDC model used by the one-PE comparison, separate from the default W7/A7 INT8 PWM executor path.", size=14, ha="left")

    d.rounded(45, 300, 1510, 205, fc="white", ec=COLORS["navy"], lw=1.4, radius=8)
    d.rounded(45, 300, 1510, 38, "2. Datapath", fc=COLORS["navy"], ec=COLORS["navy"], radius=8, color="white", bold=True, size=12)
    boxes = [
        (75, 370, 150, 60, "BF16 activation\nand weight", COLORS["blue"]),
        (265, 370, 150, 60, "component\nsplit", COLORS["yellow"]),
        (455, 370, 170, 60, "2 x 2 significand\nchunk products", COLORS["green"]),
        (665, 370, 170, 60, "bucket by\nE_a + E_w", "#efe8ff"),
        (875, 370, 170, 60, "RRAM chunk\nread", COLORS["green"]),
        (1085, 370, 170, 60, "SAR-TDC\npartial sums", COLORS["orange"]),
        (1295, 370, 170, 60, "align + round\nBF16 output", COLORS["blue"]),
    ]
    for i, (x, y, w, h, txt, fc) in enumerate(boxes):
        chip(d, x, y, w, h, txt, fc, size=11, bold=True, color="white" if fc == COLORS["orange"] else "black")
        if i < len(boxes) - 1:
            d.arrow(x + w, y + h / 2, boxes[i + 1][0], boxes[i + 1][1] + boxes[i + 1][3] / 2)
    d.rounded(265, 452, 570, 28, "metadata sideband: sign, exponent, product exponent bucket", fc="#efe8ff", ec="#9c7adb", radius=5, size=11, color=COLORS["purple"], bold=True)
    d.arrow(340, 430, 340, 452, color=COLORS["purple"], dashed=True)
    d.arrow(750, 452, 750, 430, color=COLORS["purple"], dashed=True)
    d.rounded(875, 452, 380, 28, "data path: 4-bit chunk currents -> digitized partials", fc="#e8f4e5", ec="#79a772", radius=5, size=11, color=COLORS["green_dark"], bold=True)

    d.rounded(45, 535, 760, 310, fc="white", ec=COLORS["navy"], lw=1.4, radius=8)
    d.rounded(45, 535, 760, 38, "3. Readout Accounting", fc=COLORS["navy"], ec=COLORS["navy"], radius=8, color="white", bold=True, size=12)
    rows = [
        ("Activation significand chunks", "2"),
        ("Weight significand chunks", "2"),
        ("Chunk products per exponent bucket", "4"),
        ("Observed exponent buckets in demo", "8"),
        ("Readout passes", "2 x 2 x 8 = 32"),
        ("SAR-TDC column conversions", "86"),
    ]
    table(d, 95, 610, [395, 225], 34, ["Quantity", "Value"], rows, value_color=COLORS["yellow"])

    d.rounded(845, 535, 710, 310, fc="white", ec=COLORS["navy"], lw=1.4, radius=8)
    d.rounded(845, 535, 710, 38, "4. One-PE Cost Model", fc=COLORS["navy"], ec=COLORS["navy"], radius=8, color="white", bold=True, size=12)
    rows = [
        ("Logical weights", "180"),
        ("Physical 4-bit cells for weights", "1,440"),
        ("Differential banks per weight", "8"),
        ("Latency", "2,283 cycles"),
        ("Energy", "1.5455e-6 J"),
        ("Area", "707.6 um^2"),
        ("Max abs error vs BF16", "0.00390625"),
    ]
    table(d, 900, 596, [360, 230], 27, ["Metric", "BF16 SAR-TDC"], rows, value_color=COLORS["yellow"], size=9)
    d.text(905, 828, "The 8-bank cell count is the conservative full-BF16 storage setting in the current cost table.", size=10, ha="left", bold=True)
    return d.save("bf16_sar_tdc_one_pe_path")


def table(d: Diagram, x, y, widths, row_h, headers, rows, *, value_color=COLORS["yellow"], size=10):
    d.rect(x, y, sum(widths), row_h, fc="#e6ebff", ec="#99a5dd")
    cx = x
    for i, h in enumerate(headers):
        d.rect(cx, y, widths[i], row_h, h, fc="#e6ebff", ec="#99a5dd", bold=True, size=size)
        cx += widths[i]
    cy = y + row_h
    for left, right in rows:
        d.rect(x, cy, widths[0], row_h, left, fc="white", ec="#cfd3dd", size=size)
        d.rect(x + widths[0], cy, widths[1], row_h, right, fc=value_color, ec="#cfd3dd", size=size, bold=True)
        cy += row_h


def figure_cell_comparison():
    d = Diagram(1450, 960)
    d.text(70, 55, "Cell-level readout comparison: INT8 PWM vs BF16 SAR-TDC", size=20, bold=True, ha="left")

    d.rounded(70, 105, 560, 60, "Existing INT8 PWM cost-model path", fc=COLORS["red"], ec="#c35d5d", radius=8, size=15, bold=True)
    left = [
        ("BF16 tensor boundary", COLORS["blue"]),
        ("W/A block quantization to signed INT8 codes", COLORS["yellow"]),
        ("INT8 weight -> low/high 4-bit nibbles", COLORS["green"]),
        ("differential banks: pos/neg x low/high = 4", COLORS["green"]),
        ("one logical tile readout unit in mapping cost model", COLORS["blue"]),
    ]
    y = 210
    for i, (txt, fc) in enumerate(left):
        d.rect(140, y + i * 88, 420, 52, txt, fc=fc, ec="#7aa273" if fc == COLORS["green"] else "#c35d5d", size=11, bold=True)
        if i < len(left) - 1:
            d.arrow(350, y + i * 88 + 52, 350, y + (i + 1) * 88)

    d.rounded(805, 105, 560, 60, "BF16 x BF16 SAR-TDC one-PE path", fc=COLORS["green"], ec="#73a46f", radius=8, size=15, bold=True)
    right = [
        ("BF16 = sign + exponent + fraction", COLORS["blue"]),
        ("normal significand = hidden 1 + 7 fraction bits", COLORS["yellow"]),
        ("split significand into two 4-bit chunks", COLORS["green"]),
        ("for each product exponent bucket: 2 x 2 chunk reads", COLORS["green"]),
        ("8 buckets in demo -> 32 readout passes", COLORS["yellow"]),
        ("SAR-TDC partials -> digital shift/accumulate -> BF16", COLORS["blue"]),
    ]
    y = 190
    for i, (txt, fc) in enumerate(right):
        d.rect(875, y + i * 76, 420, 48, txt, fc=fc, ec="#7aa273", size=10, bold=True)
        if i < len(right) - 1:
            d.arrow(1085, y + i * 76 + 48, 1085, y + (i + 1) * 76)
    d.line(1325, 260, 1325, 540, color=COLORS["purple"], lw=1.3)
    d.arrow(1325, 540, 1295, 540, color=COLORS["purple"])
    d.text(1340, 390, "sign/exponent\nmetadata bypass", size=10, color=COLORS["purple"], ha="left")

    d.rounded(80, 675, 1280, 235, fc="white", ec=COLORS["navy"], lw=1.3, radius=8)
    d.text(100, 703, "Same deterministic 1 x 10 by 10 x 18 workload", size=14, bold=True, ha="left")
    comparison = [
        ("Physical 4-bit cells", "720", "1,440"),
        ("Readout passes", "1", "32"),
        ("Latency cycles", "78", "2,283"),
        ("Energy", "4.839e-8 J", "1.546e-6 J"),
        ("Area", "478.8 um^2", "707.6 um^2"),
        ("Mean abs error vs BF16", "7.60e-4", "1.14e-3"),
    ]
    d.rect(120, 740, 1180, 28, fc="#e6ebff", ec="#99a5dd")
    d.rect(120, 740, 430, 28, "Metric", fc="#e6ebff", ec="#99a5dd", bold=True, size=10)
    d.rect(550, 740, 330, 28, "INT8 PWM", fc="#e6ebff", ec="#99a5dd", bold=True, size=10)
    d.rect(880, 740, 420, 28, "BF16 SAR-TDC", fc="#e6ebff", ec="#99a5dd", bold=True, size=10)
    cy = 768
    for metric, a, b in comparison:
        d.rect(120, cy, 430, 22, metric, fc="white", ec="#cfd3dd", size=9)
        d.rect(550, cy, 330, 22, a, fc="#fff6dc", ec="#cfd3dd", size=9, bold=True)
        d.rect(880, cy, 420, 22, b, fc="#fff1bf", ec="#cfd3dd", size=9, bold=True)
        cy += 22
    d.text(95, 935, "Key correction: BF16 increases readout passes because exponent buckets prevent one unscaled analog sum across rows.", size=12, bold=True, ha="left")
    return d.save("bf16_sar_tdc_int8_comparison")


def main():
    outputs = []
    for maker in (figure_transformer_dataflow, figure_one_pe_path, figure_cell_comparison):
        outputs.extend(maker())
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
