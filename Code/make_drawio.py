"""Writes an editable draw.io (diagrams.net) source of the architecture figure."""
import html, os

cells = []
_id = [2]
def nid():
    _id[0] += 1; return str(_id[0])

def vertex(label, x, y, w, h, style):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{html.escape(label, quote=True)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
    return i

def edge(src, dst, style, label=""):
    i = nid()
    cells.append(f'<mxCell id="{i}" value="{html.escape(label, quote=True)}" style="{style}" edge="1" parent="1" source="{src}" target="{dst}">'
                 f'<mxGeometry relative="1" as="geometry"/></mxCell>')
    return i

box = "rounded=1;whiteSpace=wrap;html=1;fontSize=11;fontFamily=Times New Roman;strokeColor=#000000;"
tier = "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fontStyle=1;align=left;verticalAlign=top;spacingLeft=8;spacingTop=4;strokeColor=#333333;"
arr = "endArrow=blockThin;endFill=1;html=1;strokeWidth=1.2;fontSize=10;fontFamily=Times New Roman;"
dash = "endArrow=blockThin;endFill=1;html=1;dashed=1;strokeWidth=1.2;fontSize=10;fontFamily=Times New Roman;"

cloud = vertex("Cloud f = K+1 (overflow host, +40 ms gateway link)", 230, 10, 340, 50, box + "fillColor=#E8EEF7;")
ctrl = vertex("SDN controller (control period Δ)", 20, 90, 760, 330, tier + "fillColor=#F3F3F3;")
gv = vertex("<b>Global view</b><br>load h<sub>f,t</sub>, backlog b<sub>f,t</sub>, link utilisation U<sub>ℓ,t</sub><br>attachments a<sub>i</sub>(t)", 40, 130, 210, 110, box + "fillColor=#FFFFFF;")
pred = vertex("<b>MC-dropout LSTM + residual bootstrap</b><br>K stochastic passes<br>⇒ samples ĥ<sup>(k)</sup><sub>f,t+1</sub>, k = 1..K", 280, 130, 240, 110, box + "fillColor=#FFF3D1;")
risk = vertex("<b>Risk engine</b><br>empirical CVaR<sub>β</sub> (β = upper-tail probability)<br>⇒ risk-adjusted load ĥ<sup>β</sup><sub>f</sub>", 550, 130, 210, 110, box + "fillColor=#FDE2E1;")
greedy = vertex("<b>Greedy joint orchestrator (Algorithm 1)</b><br>risk/memory feasibility, look-ahead gain vs. migration cost<br>⇒ placement x<sub>t</sub>, migrations M<sub>t</sub> (≤ T<sub>bud</sub>), routes R<sub>t</sub>", 40, 270, 720, 120, box + "fillColor=#E3F1E1;")
edge(gv, pred, arr); edge(pred, risk, arr); edge(risk, greedy, arr); edge(gv, greedy, arr)

fog = vertex("Fog tier: SDN switches + fog nodes", 20, 470, 760, 190, tier + "fillColor=#F9F9F9;")
fns = []
for k, (x, kind) in enumerate([(40, "micro"), (190, "macro"), (340, "micro"), (490, "micro"), (640, "macro")]):
    fns.append(vertex(f"FN {kind}<br>{'16' if kind=='micro' else '48'} GHz<br>C<sub>f</sub>, ε<sub>f</sub>", x, 530, 120, 70,
                      box + ("fillColor=#D9E8FF;" if kind == "micro" else "fillColor=#B9CFF2;")))
for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (2, 4)]:
    edge(fns[a], fns[b], "endArrow=none;html=1;strokeWidth=1.2;" + ("edgeStyle=orthogonalEdgeStyle;exitX=0.5;exitY=1;entryX=0.5;entryY=1;" if b - a == 2 else ""))
vertex("backhaul B<sub>ℓ</sub>, τ<sub>ℓ</sub> (fluid sharing)", 240, 622, 320, 24, "text;html=1;align=center;fontSize=10;fontFamily=Times New Roman;")
edge(fns[1], fns[2], arr + "strokeColor=#B8860B;exitX=1;exitY=0.2;entryX=0;entryY=0.2;", "migration r_s")
edge(greedy, fog, dash + "strokeColor=#1A5FB4;exitX=0.15;exitY=1;entryX=0.15;entryY=0;", "rules R_t, commands M_t")
edge(fog, greedy, dash + "strokeColor=#C01C28;exitX=0.85;exitY=0;entryX=0.85;entryY=1;", "telemetry")
edge(cloud, ctrl, "endArrow=blockThin;startArrow=blockThin;endFill=1;startFill=1;html=1;strokeWidth=1.2;exitX=0.5;exitY=1;entryX=0.5;entryY=0;", "gateway link")

dev = vertex("IoT devices (RWP / Gauss-Markov / trace); uplink to nearest AP", 20, 700, 760, 150, tier + "fillColor=#FBFBFB;")
for k, (x, tgt) in enumerate([(70, 0), (190, 1), (310, 2), (430, 2), (550, 3), (670, 4)]):
    d = vertex("i", x, 760, 36, 36, "ellipse;whiteSpace=wrap;html=1;fillColor=#FFE8C7;fontSize=11;fontFamily=Times New Roman;")
    edge(d, fns[tgt], arr + "strokeColor=#2A7F2A;")
vertex("ON/OFF Poisson tasks (d<sub>κ</sub>, w<sub>κ</sub>); service s<sub>i</sub> at host x<sub>s<sub>i</sub></sub>", 100, 812, 600, 24,
       "text;html=1;align=center;fontSize=10;fontFamily=Times New Roman;")

xml = ('<mxfile host="app.diagrams.net" modified="2026-10-29T00:00:00.000Z" agent="RAI-Edge make_drawio.py" version="24.0.0">'
       '<diagram id="raiedge-sysmodel" name="RAI-Edge system model"><mxGraphModel dx="1000" dy="700" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0">'
       '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram></mxfile>')
os.makedirs("../figures", exist_ok=True)
open("../figures/sysmodel_raiedge.drawio", "w").write(xml)
print("wrote drawio with", len(cells), "cells")
