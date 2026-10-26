"""Build an editable PowerPoint replica of the Qwen NLPC/LPC hardware flow diagram."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
BLUE = RGBColor(0x1F, 0x4E, 0x79)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
HIGHLIGHT = RGBColor(0x2E, 0x74, 0xB5)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_title(slide, text, y=Inches(0.15)):
    tb = slide.shapes.add_textbox(Inches(0.3), y, Inches(12.7), Inches(0.6))
    tf = tb.text_frame
    tf.text = text
    p = tf.paragraphs[0]
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = BLUE
    return tb


def add_box(slide, text, x, y, w, h, *, fill=WHITE, line=BLACK, font_size=10,
            dashed=False, bold=False, line_width=Pt(1.25)):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = line
    shp.line.width = line_width
    if dashed:
        ln = shp.line._get_or_add_ln()
        prst = ln.makeelement(qn('a:prstDash'), {'val': 'dash'})
        ln.append(prst)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(2)
    tf.margin_right = Pt(2)
    tf.margin_top = Pt(1)
    tf.margin_bottom = Pt(1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.text = text
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = BLACK
    return shp


def add_group_box(slide, label, x, y, w, h):
    """Outer container rectangle (e.g. MEM/CTRL, LPC, NLPC) with a label above the top edge."""
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.fill.background()
    shp.line.color.rgb = BLACK
    shp.line.width = Pt(1.5)
    shp.shadow.inherit = False
    tb = slide.shapes.add_textbox(x, y - Inches(0.28), w, Inches(0.26))
    p = tb.text_frame.paragraphs[0]
    p.text = label
    p.alignment = PP_ALIGN.CENTER
    p.font.size = Pt(11)
    p.font.bold = True
    return shp


def h_arrow(slide, x1, y, x2):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y, x2, y)
    conn.line.color.rgb = BLACK
    conn.line.width = Pt(1.25)
    conn.line._get_or_add_ln().append(
        conn.line._get_or_add_ln().makeelement(qn('a:tailEnd'), {'type': 'triangle'})
    )
    return conn


def elbow_arrow(slide, x1, y1, x2, y2):
    gap_y = (y1 + y2) // 2
    gutter_x = x2 - Inches(0.2)
    points = (
        (x1, y1),
        (x1, gap_y),
        (gutter_x, gap_y),
        (gutter_x, y2),
    )
    for start, end in zip(points, points[1:]):
        conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, *start, *end)
        conn.line.color.rgb = BLACK
        conn.line.width = Pt(1.25)
    return h_arrow(slide, gutter_x, y2, x2)


# ---------------------------------------------------------------------------
# Slide 1 - Title
# ---------------------------------------------------------------------------
s1 = prs.slides.add_slide(BLANK)
tb = s1.shapes.add_textbox(Inches(0.8), Inches(2.6), Inches(11.7), Inches(1.2))
p = tb.text_frame.paragraphs[0]
p.text = "Qwen Transformer Block — NLPC / LPC Hardware Flow"
p.font.size = Pt(36)
p.font.bold = True
p.font.color.rgb = BLUE
p.alignment = PP_ALIGN.CENTER
tb2 = s1.shapes.add_textbox(Inches(0.8), Inches(3.7), Inches(11.7), Inches(0.8))
p2 = tb2.text_frame.paragraphs[0]
p2.text = "End-to-end token flow, per-block non-linear (NLPC) vs. linear crossbar (LPC) engine split"
p2.font.size = Pt(16)
p2.font.color.rgb = BLACK
p2.alignment = PP_ALIGN.CENTER

# ---------------------------------------------------------------------------
# Slide 2 - Full architecture flow (3 rows, matches the source diagram)
# ---------------------------------------------------------------------------
s2 = prs.slides.add_slide(BLANK)
add_title(s2, "End-to-End Flow: Token → Attention → FFN → Sampling")

BH = Inches(0.62)
GAP = Inches(0.14)

# ---- Row 1 ----
row1_y = Inches(0.85)
row1_items = [
    ("Input prompt", Inches(1.0)),
    ("Tokenization", Inches(1.1)),
]
x = Inches(0.3)
boxes_row1 = []
for label, w in row1_items:
    b = add_box(s2, label, x, row1_y, w, BH)
    boxes_row1.append(b)
    x += w + GAP

# MEM/CTRL group: Embedding lookup + Hidden vector buffer
mem1_w = Inches(2.75)
mem1_x = x
add_group_box(s2, "MEM/CTRL", mem1_x, row1_y - Inches(0.02), mem1_w, BH + Inches(0.04))
inner_w = Inches(1.3)
emb_box = add_box(s2, "Embedding lookup", mem1_x + Inches(0.08), row1_y, inner_w, BH, font_size=9)
hid_box = add_box(s2, "Hidden vector buffer", mem1_x + Inches(0.08) + inner_w + Inches(0.08), row1_y, inner_w, BH, font_size=9)
h_arrow(s2, emb_box.left + emb_box.width, emb_box.top + emb_box.height // 2, hid_box.left)
x = mem1_x + mem1_w + GAP

row1_rest = [
    ("RMS Norm (NLPC)", Inches(1.35)),
    ("Q/K/V projection\non PE array", Inches(1.55)),
    ("qnorm/knorm (NLPC)", Inches(1.5)),
    ("RoPE (NLPC)", Inches(1.3)),
]
for label, w in row1_rest:
    b = add_box(s2, label, x, row1_y, w, BH)
    boxes_row1.append(b)
    x += w + GAP
rope_box = boxes_row1[-1]

# arrows across row 1 (skip the mem-group internal pair already connected)
seq_row1 = [boxes_row1[0], boxes_row1[1], emb_box]  # input->tok->mem group start
for a, b in zip(seq_row1, seq_row1[1:]):
    h_arrow(s2, a.left + a.width, a.top + a.height // 2, b.left)
seq_after_mem = [hid_box] + boxes_row1[2:]
for a, b in zip(seq_after_mem, seq_after_mem[1:]):
    h_arrow(s2, a.left + a.width, a.top + a.height // 2, b.left)

# ---- Row 2 ----
row2_y = Inches(2.35)
kv_w = Inches(1.5)
add_group_box(s2, "MEM/CTRL", Inches(0.3), row2_y - Inches(0.02), kv_w + Inches(0.16), BH + Inches(0.04))
kv_box = add_box(s2, "KV cache indexing", Inches(0.38), row2_y, kv_w, BH, font_size=9)

row2_rest = [
    ("Q * K^T\nPE array", Inches(1.3)),
    ("Causal mask (NLPC)", Inches(1.45)),
    ("softmax (NLPC)", Inches(1.3)),
    ("softmax * v\nPE array", Inches(1.3)),
    ("output projection\n(PE array)", Inches(1.5)),
    ("Residual add 1", Inches(1.3)),
    ("RMSNorm\npre-FFN (NLPC)", Inches(1.75)),
]
x = kv_box.left + kv_box.width + Inches(0.24)
boxes_row2 = [kv_box]
for label, w in row2_rest:
    b = add_box(s2, label, x, row2_y, w, BH)
    boxes_row2.append(b)
    x += w + GAP
for a, b in zip(boxes_row2, boxes_row2[1:]):
    h_arrow(s2, a.left + a.width, a.top + a.height // 2, b.left)
prenorm_box = boxes_row2[-1]

# wrap arrow: RoPE (end of row1) -> KV cache indexing (start of row2)
elbow_arrow(s2, rope_box.left + rope_box.width // 2, rope_box.top + rope_box.height,
            kv_box.left, kv_box.top + kv_box.height // 2)

# ---- Row 3 ----
row3_y = Inches(3.85)
row3_items = [
    ("Gate/up projection\nPE array", Inches(1.35)),
    ("SiLU activation\n(NLPC)", Inches(1.25)),
    ("SwiGLU multiply\n(NLPC)", Inches(1.35)),
    ("Down projection\nPE array", Inches(1.3)),
    ("Residual add 2", Inches(1.1)),
    ("Repeat blocks\n2-28", Inches(1.3)),
    ("Final RMSNorm\n(NLPC)", Inches(1.35)),
    ("LM head", Inches(0.9)),
    ("Top-K", Inches(0.8)),
]
x = Inches(0.3)
boxes_row3 = []
for i, (label, w) in enumerate(row3_items):
    dashed = (i == 2)  # SwiGLU multiply highlighted like the source diagram
    b = add_box(s2, label, x, row3_y, w, BH, dashed=dashed, line_width=Pt(1.75) if dashed else Pt(1.25))
    boxes_row3.append(b)
    x += w + GAP
for a, b in zip(boxes_row3, boxes_row3[1:]):
    h_arrow(s2, a.left + a.width, a.top + a.height // 2, b.left)

# wrap arrow: pre-FFN RMSNorm (end of row2) -> Gate up projection (start of row3)
elbow_arrow(s2, prenorm_box.left + prenorm_box.width // 2, prenorm_box.top + prenorm_box.height,
            boxes_row3[0].left, boxes_row3[0].top + boxes_row3[0].height // 2)

# Legend note
note = s2.shapes.add_textbox(Inches(0.3), Inches(4.75), Inches(12.5), Inches(0.4))
p = note.text_frame.paragraphs[0]
p.text = "NLPC = Non-Linear Processing Core (vector ALU engine)   |   PE array = ReRAM crossbar linear-processing core (LPC)"
p.font.size = Pt(11)
p.font.italic = True
p.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

# ---------------------------------------------------------------------------
# Slide 3 - LPC internal block
# ---------------------------------------------------------------------------
s3 = prs.slides.add_slide(BLANK)
add_title(s3, "LPC (Linear Processing Core) — PE Internal Datapath")

lpc_x, lpc_y, lpc_w, lpc_h = Inches(2.0), Inches(1.4), Inches(9.3), Inches(4.6)
add_group_box(s3, "LPC", lpc_x, lpc_y, lpc_w, lpc_h)

add_box(s3, "Input\nbuffer", lpc_x + Inches(0.4), lpc_y + Inches(0.5), Inches(1.2), Inches(2.6), font_size=11)
add_box(s3, "32*32 crossbar array", lpc_x + Inches(2.0), lpc_y + Inches(0.5), Inches(2.9), Inches(2.6), font_size=12)
add_box(s3, "ADC/IV-sensing/\nSAR-TDC/Wear\nleveling", lpc_x + Inches(5.3), lpc_y + Inches(0.5), Inches(1.7), Inches(2.6), font_size=10)
add_box(s3, "Output\nbuffer", lpc_x + Inches(7.4), lpc_y + Inches(0.5), Inches(1.2), Inches(2.6), font_size=11)
add_box(s3, "Compute buffer", lpc_x + Inches(2.0), lpc_y + Inches(3.4), Inches(2.9), Inches(0.7), font_size=11)

# ---------------------------------------------------------------------------
# Slide 4 - NLPC internal block
# ---------------------------------------------------------------------------
s4 = prs.slides.add_slide(BLANK)
add_title(s4, "NLPC (Non-Linear Processing Core) — Internal Datapath")

nlpc_x, nlpc_y, nlpc_w, nlpc_h = Inches(2.6), Inches(1.4), Inches(8.1), Inches(4.6)
add_group_box(s4, "NLPC", nlpc_x, nlpc_y, nlpc_w, nlpc_h)

add_box(s4, "Input\nbuffer", nlpc_x + Inches(0.4), nlpc_y + Inches(0.5), Inches(1.2), Inches(2.6), font_size=11)
add_box(s4, "32*32 crossbar array", nlpc_x + Inches(2.0), nlpc_y + Inches(0.5), Inches(2.9), Inches(2.6), font_size=12)
out_box = add_box(s4, "Output\nbuffer", nlpc_x + Inches(5.3), nlpc_y + Inches(0.5), Inches(1.2), Inches(1.1),
                   fill=RGBColor(0xDD, 0xEB, 0xF7), line=HIGHLIGHT, line_width=Pt(2.0), font_size=11, bold=True)
add_box(s4, "Compute buffer", nlpc_x + Inches(2.0), nlpc_y + Inches(3.4), Inches(2.9), Inches(0.7), font_size=11)

note4 = s4.shapes.add_textbox(nlpc_x, nlpc_y + nlpc_h + Inches(0.2), nlpc_w, Inches(0.4))
p = note4.text_frame.paragraphs[0]
p.text = "Output buffer highlighted — staging point where NLPC results feed back into the next crossbar pass."
p.font.size = Pt(11)
p.font.italic = True

out_path = "results/figures/qwen_nlpc_architecture_final.pptx"
prs.save(out_path)
print("Saved:", out_path)
