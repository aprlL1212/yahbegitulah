"""Peningkatan visualisasi diagram gaya dalam dan deformasi dengan keterangan skala,
interpretasi nilai ekstrem, dan petunjuk arah/konvensi tanda."""
import numpy as np
import plotly.graph_objects as go
from . import engine as eg

COL = {"N": "#2f9e44", "V": "#e8590c", "M": "#1971c2"}
UNIT = {"N": "kN", "V": "kN", "M": "kN·m"}
NAME = {"N": "Gaya Aksial (N)", "V": "Gaya Geser (V)", "M": "Momen Lentur (M)"}
DESC = {
    "N": "Nilai (+) = Tarik, Nilai (-) = Tekan. Sangat penting untuk analisis stabilitas batang & bahaya tekuk (buckling).",
    "V": "Gaya lintang/geser penampang. Menentukan kebutuhan tulangan sengkang pada beton atau ketebalan web pada profil baja.",
    "M": "Momen lentur (digambar pada sisi serat tarik). Menentukan dimensi tinggi penampang dan luas tulangan lentur utama."
}


def _nodes(model):
    return {int(n["id"]): (float(n["x"]), float(n["y"])) for n in model["nodes"]}


def _size(nodes):
    p = np.array(list(nodes.values()))
    return max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)


def _rgba(h, a):
    return f"rgba({int(h[1:3],16)},{int(h[3:5],16)},{int(h[5:7],16)},{a})"


def _layout(fig, title, subtitle="", scale_info=""):
    # Gabungkan annotations yang sudah ada (label elemen, beban, dll)
    existing_annotations = list(fig.layout.annotations) if fig.layout.annotations else []
    
    if scale_info:
        existing_annotations.append(dict(
            text=f"📐 <b>Skala Diagram:</b> {scale_info}",
            xref="paper", yref="paper",
            x=0.01, y=0.98,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(148, 163, 184, 0.4)",
            borderwidth=1,
            borderpad=5,
            font=dict(size=11, color="#1e293b", family="Inter, sans-serif")
        ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color="#0f172a", family="Inter, sans-serif")),
        height=500,
        margin=dict(l=30, r=30, t=55, b=45),
        showlegend=False,
        xaxis=dict(title="Koordinat Global X (m)", zeroline=True, gridcolor="rgba(226, 232, 240, 0.7)", linecolor="#cbd5e1"),
        yaxis=dict(title="Koordinat Global Y (m)", scaleanchor="x", zeroline=True, gridcolor="rgba(226, 232, 240, 0.7)", linecolor="#cbd5e1"),
        hovermode="closest",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        annotations=existing_annotations
    )
    return fig


def _arrow(fig, x, y, fx, fy, text, color="#e03131", px=42):
    n = np.hypot(fx, fy)
    if n < 1e-12:
        return
    fig.add_annotation(
        x=x, y=y, ax=-px * fx / n, ay=px * fy / n,
        xref="x", yref="y", axref="pixel", ayref="pixel",
        showarrow=True, arrowhead=2, arrowwidth=2, arrowcolor=color,
        text=text, font=dict(color=color, size=11, family="sans-serif")
    )


def model_fig(model, title="Model Struktur, Tumpuan & Pembebanan"):
    nodes = _nodes(model)
    fig = go.Figure()

    # Gambar elemen
    for m in model["members"]:
        (x1, y1), (x2, y2) = nodes[m["ni"]], nodes[m["nj"]]
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2], mode="lines",
            line=dict(color="#343a40", width=5),
            hoverinfo="text",
            text=f"<b>Elemen {m['id']}</b><br>Bentang: {np.hypot(x2-x1, y2-y1):.2f} m<br>Dimensi: {m['b']:g} × {m['h']:g} mm<br>E = {m['E']:g} MPa"
        ))
        fig.add_annotation(
            x=(x1+x2)/2, y=(y1+y2)/2, text=f"<b>E{m['id']}</b>", showarrow=False,
            font=dict(size=11, color="#1971c2"), yshift=-14
        )

    # Gambar node
    fig.add_trace(go.Scatter(
        x=[p[0] for p in nodes.values()], y=[p[1] for p in nodes.values()],
        mode="markers+text", text=[f"N{i}" for i in nodes], textposition="top right",
        marker=dict(size=9, color="#1c7ed6", line=dict(color="#ffffff", width=2)),
        hoverinfo="text",
        hovertext=[f"Node {i}: ({nodes[i][0]:.3f}, {nodes[i][1]:.3f}) m" for i in nodes]
    ))

    # Tumpuan
    sym = {(1, 1, 1): "square", (1, 1, 0): "triangle-up", (0, 1, 0): "circle-open", (1, 0, 0): "triangle-left"}
    sps = [s for s in model["supports"] if s["node"] in nodes]
    if sps:
        fig.add_trace(go.Scatter(
            x=[nodes[s["node"]][0] for s in sps], y=[nodes[s["node"]][1] for s in sps], mode="markers",
            marker=dict(
                size=18, color="#2b8a3e",
                symbol=[sym.get((int(s["ux"]), int(s["uy"]), int(s["rz"])), "diamond") for s in sps],
                line=dict(width=2, color="#1b4b27")
            ),
            hoverinfo="text",
            text=[f"Tumpuan N{s['node']}: [Ux={int(s['ux'])}, Uy={int(s['uy'])}, Rz={int(s['rz'])}]" for s in sps]
        ))

    # Beban nodal
    for nl in model["nodal_loads"]:
        x, y = nodes[nl["node"]]
        _arrow(fig, x, y, nl["Fx"], nl["Fy"], f"{np.hypot(nl['Fx'], nl['Fy']):g} kN")
        if nl["M"]:
            fig.add_annotation(
                x=x, y=y, text=f"M={nl['M']:g} kN·m {'↺' if nl['M'] > 0 else '↻'}",
                showarrow=False, yshift=26, font=dict(color="#e03131", size=11, weight="bold")
            )

    # Beban elemen
    mem_map = {m["id"]: m for m in model["members"]}
    for ml in model["member_loads"]:
        m = mem_map[ml["member"]]
        (x1, y1), (x2, y2) = nodes[m["ni"]], nodes[m["nj"]]
        L, c, s = eg.geom((x1, y1), (x2, y2))
        gx, gy = eg.load_global(ml, c, s)
        if ml["type"] == "UDL":
            for t in np.linspace(0.1, 0.9, 5):
                _arrow(fig, x1 + t*(x2-x1), y1 + t*(y2-y1), gx, gy, f"{ml['value']:g} kN/m" if abs(t-0.5) < 1e-9 else "", "#f08c00", 30)
        else:
            _arrow(fig, x1 + c*ml["a"], y1 + s*ml["a"], gx, gy, f"{ml['value']:g} kN", "#f08c00")

    return _layout(fig, title)


def deformed_data(res, factor=1.0, n=21):
    nodes = res["nodes"]
    u = res["u"]
    dmax = max(np.abs(u[0::3]).max(), np.abs(u[1::3]).max())
    sc = (0.1 * _size(nodes) / dmax if dmax > 0 else 1.0) * factor
    curves = []
    for mem in res["members"].values():
        x = np.linspace(0, mem["L"], n)
        ul, vl = eg.local_disp(mem, x)
        c, s = mem["c"], mem["s"]
        x0, y0 = nodes[mem["ni"]]
        curves.append((x0 + c*x + sc*(ul*c - vl*s), y0 + s*x + sc*(ul*s + vl*c)))
    return sc, curves


def deformed_fig(res, factor=1.0):
    sc, curves = deformed_data(res, factor)
    fig = go.Figure()
    nodes = res["nodes"]
    u = res["u"]

    # Garis struktur asli (undeformed)
    for m in res["model"]["members"]:
        (x1, y1), (x2, y2) = nodes[m["ni"]], nodes[m["nj"]]
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y2], mode="lines",
            line=dict(color="#adb5bd", width=2.5, dash="dash"),
            hoverinfo="skip"
        ))

    # Kurva terdeformasi
    for X, Y in curves:
        fig.add_trace(go.Scatter(
            x=X, y=Y, mode="lines",
            line=dict(color="#e03131", width=4),
            hoverinfo="skip"
        ))

    # Tandai posisi node terdeformasi dengan nilai lendutan
    for k, nid in enumerate(res["ids"]):
        x0, y0 = nodes[nid]
        ux_val, uy_val = u[3*k]*1e3, u[3*k+1]*1e3
        x_def = x0 + sc * u[3*k]
        y_def = y0 + sc * u[3*k+1]
        fig.add_trace(go.Scatter(
            x=[x_def], y=[y_def], mode="markers",
            marker=dict(size=8, color="#e03131", line=dict(color="#ffffff", width=1.5)),
            hoverinfo="text",
            hovertext=f"<b>Node {nid}</b><br>Ux = {ux_val:.3f} mm<br>Uy = {uy_val:.3f} mm<br>Rz = {u[3*k+2]:.4f} rad"
        ))

    scale_str = f"Faktor Perbesaran Visual ≈ {sc:,.0f}× (Garis Abu-abu = Posisi Awal, Merah = Deformasi)"
    return _layout(fig, f"Visualisasi Deformasi Struktur Elastis", subtitle="Perpindahan diperbesar agar kelengkungan elastis terlihat jelas.", scale_info=scale_str)


def diagram_data(res, kind, factor=1.0):
    nodes = res["nodes"]
    raw = []
    for mem in res["members"].values():
        x = eg.sample_x(mem)
        N, V, M = eg.internal(mem, x)
        raw.append((mem, x, {"N": N, "V": V, "M": M}[kind]))
    vmax = max(np.abs(v).max() for _, _, v in raw) or 1.0
    sc = 0.15 * _size(nodes) / vmax * factor
    sign = -1 if kind == "M" else 1   # M digambar di sisi tarik
    items = []
    for mem, x, v in raw:
        x0, y0 = nodes[mem["ni"]]
        c, s = mem["c"], mem["s"]
        bx, by = x0 + c*x, y0 + s*x
        off = sign * v * sc
        items.append(dict(id=mem["id"], x=x, val=v, bx=bx, by=by, ox=bx - s*off, oy=by + c*off))
    return sc, items, vmax


def diagram_fig(res, kind, factor=1.0):
    sc, items, vmax = diagram_data(res, kind, factor)
    fig = go.Figure()
    col = COL[kind]

    for it in items:
        # Garis netral sumbu elemen
        fig.add_trace(go.Scatter(
            x=it["bx"], y=it["by"], mode="lines",
            line=dict(color="#495057", width=3), hoverinfo="skip"
        ))
        # Area arsiran bidang gaya
        fig.add_trace(go.Scatter(
            x=np.r_[it["bx"], it["ox"][::-1]], y=np.r_[it["by"], it["oy"][::-1]], mode="lines",
            fill="toself", fillcolor=_rgba(col, .25), line=dict(color=col, width=2), hoverinfo="skip"
        ))
        # Titik interaktif hover
        fig.add_trace(go.Scatter(
            x=it["ox"], y=it["oy"], mode="markers", marker=dict(size=6, opacity=0), hoverinfo="text",
            text=[f"<b>Elemen {it['id']}</b><br>Jarak x = {a:.3f} m<br><b>{kind} = {v:.3f} {UNIT[kind]}</b>" for a, v in zip(it["x"], it["val"])]
        ))
        # Label nilai ekstrem pada ujung dan puncak
        v = it["val"]
        for i in {0, len(v) - 1, int(np.abs(v).argmax())}:
            if abs(v[i]) > 1e-4:
                val_sign = "+" if v[i] > 0 else ""
                fig.add_annotation(
                    x=it["ox"][i], y=it["oy"][i],
                    text=f"<b>{val_sign}{v[i]:.2f}</b>",
                    showarrow=False,
                    font=dict(color=col, size=11, family="sans-serif"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=col,
                    borderwidth=1,
                    borderpad=2,
                    yshift=10
                )

    scale_text = f"1 unit tinggi grafis ≈ {(1.0/sc if sc > 0 else 1.0):.2f} {UNIT[kind]} | Nilai Puncak: {vmax:.2f} {UNIT[kind]}"
    title = f"{NAME[kind]} [{UNIT[kind]}]"
    return _layout(fig, title, subtitle=DESC[kind], scale_info=scale_text)
