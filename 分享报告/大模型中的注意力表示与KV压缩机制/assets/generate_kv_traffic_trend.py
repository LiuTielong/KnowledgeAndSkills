"""Generate the conceptual KV/weight read-volume figure and inline it into HTML.

Requires matplotlib and a Chinese font. The resulting HTML works offline.
Coordinates are illustrative: there are no numerical ticks, model parameters,
context limits, benchmark measurements, or calibrated crossover estimates.
"""

from pathlib import Path
import io
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


assets = Path(__file__).resolve().parent
html = assets.parent / "index.html"
fonts = font_manager.findSystemFonts()
preferred = ("PingFang.ttc", "NotoSansCJK-Regular.ttc", "msyh.ttc", "STHEITI.ttf", "Arial Unicode.ttf")
font_path = next((path for name in preferred for path in fonts if path.endswith(name)), None)
if font_path is None:
    raise RuntimeError("Install a Chinese font such as Noto Sans CJK to rebuild this figure.")
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams.update({
    "font.family": font_name, "font.size": 14, "svg.fonttype": "path",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9db3b2", "axes.labelcolor": "#526b70",
    "xtick.color": "#62757a", "savefig.facecolor": "white",
})
fig = plt.figure(figsize=(11.4, 4.25), dpi=72)
ax = fig.add_axes([0.08, 0.16, 0.89, 0.80])
crossover = 1 / 0.65  # Illustrative geometry only; never shown as a numeric threshold.
ax.axvspan(0, crossover, color="#eef3f7", zorder=0)
ax.axvspan(crossover, 6, color="#edf7f2", zorder=0)
ax.axvline(crossover, color="#b8cbc5", ls=(0, (3, 4)), lw=1)
ax.plot([0, 6], [1, 1], color="#688aa1", lw=2.6, ls=(0, (6, 4)))
ax.plot([0, 6], [0, 3.9], color="#117d73", lw=3)
ax.text(crossover / 2, 4.18, "权重读取主导", ha="center", color="#56788d", fontsize=14)
ax.text((crossover + 6) / 2, 4.18, "KV 读取主导", ha="center", color="#117d73", fontsize=16, fontweight="medium")
ax.text(4.2, 3.15, "KV Cache 读取", ha="center", color="#117d73", fontsize=16)
ax.text(4.65, 1.18, "权重读取（基本不变）", ha="center", color="#56788d", fontsize=14)
ax.set(xlim=(0, 6), ylim=(-0.06, 4.6), xlabel="上下文长度 S", ylabel="每步 Decode 读取量")
ax.set_xticks([0.25, 5.7], ["短上下文", "长上下文"])
ax.set_yticks([])
ax.tick_params(axis="both", length=0, pad=10)
ax.xaxis.labelpad = 12
ax.yaxis.labelpad = 14
ax.annotate("", xy=(6.08, -0.06), xytext=(5.88, -0.06), annotation_clip=False,
            arrowprops={"arrowstyle": "-|>", "color": "#9db3b2", "lw": 1})
buffer = io.StringIO()
fig.savefig(buffer, format="svg", metadata={"Date": None, "Title": "KV 与权重的解码读取量趋势示意"})
svg = buffer.getvalue()
svg = svg[svg.index("<svg"):]
svg = svg.replace("<svg ", '<svg id="traffic-trend-svg" class="traffic-svg" role="img" aria-label="趋势示意：固定模型、batch 和精度，权重读取为水平线，KV Cache 读取随上下文长度上升，长上下文时 KV 读取占主导；不标具体数值或实测交点" ', 1)
(assets / "kv-memory-traffic-trend.svg").write_text(svg, encoding="utf-8")
page = html.read_text(encoding="utf-8")
begin, end = "<!-- BEGIN KV TRAFFIC TREND -->", "<!-- END KV TRAFFIC TREND -->"
page, count = re.subn(re.escape(begin) + r".*?" + re.escape(end), lambda _: begin + "\n" + svg + "\n" + end, page, flags=re.S)
assert count == 1, "Expected exactly one chart insertion point"
html.write_text(page, encoding="utf-8")
plt.close(fig)
print("Conceptual chart generated and inlined; no model-specific numbers or context cap.")
