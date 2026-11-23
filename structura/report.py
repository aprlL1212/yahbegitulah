"""Laporan PDF Resmi Analisis Struktur (ReportLab).
Dilengkapi tahapan proses perhitungan, asumsi analisis, interpretasi rekayasa,
dan verifikasi terhadap solusi analitis/kesetimbangan statika."""
import datetime
import io
import numpy as np
import pandas as pd
from reportlab.graphics.shapes import Drawing, Polygon, PolyLine, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether, PageBreak
from . import engine as eg, plots, insights

ASUMSI = [
    "Analisis Portal 2D menggunakan Metode Kekakuan Langsung (Direct Stiffness Method), K·u = F, elastis linier statik orde pertama.",
    "Elemen frame mengadopsi teori balok-kolom Euler-Bernoulli dengan 3 DOF per node (Ux, Uy, Rz). Deformasi geser diabaikan.",
    "Material bersifat isotropik dan homogen dengan Modulus Elastisitas E konstan sepanjang elemen.",
    "Penampang elemen dianggap persegi penuh dengan Luas A = b·h dan Momen Inersia I = b·h³/12.",
    "Semua sambungan antar-elemen (pertemuan balok dan kolom) diasumsikan kaku sempurna (rigid joint).",
    "Konvensi Gaya Dalam: Gaya aksial N bernilai positif untuk gaya tarik (+); Gaya geser V positif bila gaya sebelah kiri potongan mengarah ke atas (sumbu lokal y); Momen lentur M positif (sagging) dan digambar pada sisi serat tarik penampang.",
    "Aplikasi dirancang sebagai media pembelajaran analisis struktur dan mekanika rekayasa komputasional."
]


def _f(v):
    if isinstance(v, (float, np.floating)):
        return "0" if abs(v) < 1e-9 else f"{v:.4g}"
    return str(v)


def _tbl(df, col_widths=None):
    data = [list(df.columns)] + [[_f(v) for v in r] for r in df.itertuples(index=False)]
    t = Table(data, repeatRows=1, hAlign="LEFT", colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c3d7a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ced4da")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _drawing(res, kind, W=500, H=220):
    model = res["model"]
    nodes = res["nodes"]
    base = [(np.array([nodes[m["ni"]][0], nodes[m["nj"]][0]]), np.array([nodes[m["ni"]][1], nodes[m["nj"]][1]])) for m in model["members"]]
    layers, items = [(x, y, "base") for x, y in base], []
    if kind == "def":
        layers = [(x, y, "ghost") for x, y in base] + [(X, Y, "def") for X, Y in plots.deformed_data(res)[1]]
    elif kind in "NVM":
        items = plots.diagram_data(res, kind)[1]
        layers += [(np.r_[i["bx"], i["ox"][::-1]], np.r_[i["by"], i["oy"][::-1]], "diag") for i in items]
    xs = np.concatenate([l[0] for l in layers])
    ys = np.concatenate([l[1] for l in layers])
    dx, dy = max(np.ptp(xs), 1e-9), max(np.ptp(ys), 1e-9)
    sc = min((W - 60) / dx, (H - 50) / dy)
    tr = lambda x, y: ((W - dx*sc)/2 + (np.asarray(x) - xs.min())*sc, (H - dy*sc)/2 + (np.asarray(y) - ys.min())*sc)
    d = Drawing(W, H)
    for x, y, st in layers:
        pts = np.column_stack(tr(x, y)).ravel().tolist()
        if st == "diag":
            d.add(Polygon(pts, fillColor=colors.Color(.1, .4, .8, .2), strokeColor=colors.HexColor(plots.COL[kind]), strokeWidth=1.2))
        else:
            d.add(PolyLine(pts, strokeColor={"base": colors.HexColor("#343a40"), "ghost": colors.HexColor("#adb5bd"), "def": colors.HexColor("#e03131")}[st],
                           strokeWidth=2 if st != "ghost" else 1, strokeDashArray=[3, 2] if st == "ghost" else None))
    for sp in model["supports"]:
        X, Y = tr(*nodes[sp["node"]])
        d.add(Polygon([X, Y, X-5, Y-9, X+5, Y-9], fillColor=colors.HexColor("#2b8a3e"), strokeColor=colors.HexColor("#2b8a3e")))
    for it in items:
        v = it["val"]
        for i in {0, len(v)-1, int(np.abs(v).argmax())}:
            if abs(v[i]) > 1e-4:
                X, Y = tr(it["ox"][i], it["oy"][i])
                d.add(String(float(X) + 2, float(Y) + 3, f"{v[i]:.2f}", fontSize=7, fillColor=colors.HexColor(plots.COL[kind])))
    return d


def build_pdf(name, model, res):
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#1c3d7a"))
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#1971c2"), spaceBefore=10, spaceAfter=4)
    H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10, leading=14, textColor=colors.HexColor("#212529"), spaceBefore=6, spaceAfter=2)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=8.5, leading=12, textColor=colors.HexColor("#343a40"))
    B_Bold = ParagraphStyle("BB", parent=B, fontName="Helvetica-Bold")
    Callout = ParagraphStyle("Callout", parent=B, backColor=colors.HexColor("#e7f5ff"), borderColor=colors.HexColor("#339af0"),
                             borderWidth=1, borderPadding=6, spaceBefore=4, spaceAfter=6)

    t = eg.tables(res)
    calc_steps = insights.generate_calculation_steps(res)
    eng_insights = insights.generate_engineering_insights(res)
    benchmark_df = insights.compare_with_benchmarks(res)
    S = []

    # Header & Judul
    S += [
        Paragraph("<b>STRUCTURA — LAPORAN RESMI ANALISIS STRUKTUR PORTAL 2D</b>", H1),
        Paragraph(f"<b>Nama Proyek:</b> {name} &nbsp;|&nbsp; <b>Waktu Analisis:</b> {datetime.datetime.now():%d-%m-%Y %H:%M WIB}", B),
        Paragraph("<b>Metode:</b> Kekakuan Langsung (Direct Stiffness Method, K·u = F) &nbsp;|&nbsp; <b>Model:</b> Elastis Linier Statik Orde-1", B),
        Spacer(1, 8),
    ]

    # Asumsi Dasar
    S.append(Paragraph("1. Asumsi Analisis & Konvensi Rekayasa", H2))
    for a in ASUMSI:
        S.append(Paragraph(f"• {a}", B))
    S.append(Spacer(1, 6))

    # Data Input
    S.append(Paragraph("2. Data Masukan Geometri, Material & Pembebanan", H2))
    nd = pd.DataFrame(model["nodes"]).rename(columns={"id": "Node", "x": "x (m)", "y": "y (m)"})
    mm = []
    for m in model["members"]:
        E, A, I = eg.section(m)
        mm.append({
            "Elemen": m["id"], "Node i": m["ni"], "Node j": m["nj"],
            "E (MPa)": m["E"], "b (mm)": m["b"], "h (mm)": m["h"],
            "A (cm²)": A * 1e4, "I (cm⁴)": I * 1e8
        })
    sp = pd.DataFrame([{
        "Node": s["node"],
        "Ux": "Tertahan" if s["ux"] else "Bebas",
        "Uy": "Tertahan" if s["uy"] else "Bebas",
        "Rz": "Tertahan" if s["rz"] else "Bebas"
    } for s in model["supports"]])
    nl = pd.DataFrame(model["nodal_loads"]).rename(columns={"node": "Node", "Fx": "Fx (kN)", "Fy": "Fy (kN)", "M": "M (kN·m)"})
    ml = pd.DataFrame([{
        "Elemen": m["member"], "Jenis": m["type"], "Arah": m["dir"],
        "Nilai": f"{m['value']:g} " + ("kN/m" if m["type"] == "UDL" else "kN"),
        "Posisi a (m)": m["a"] if m["type"] == "Titik" else "-"
    } for m in model["member_loads"]])

    for title, df in (("Tabel Koordinat Node", nd), ("Tabel Elemen & Properti Penampang", pd.DataFrame(mm)),
                      ("Tabel Kondisi Batas Tumpuan", sp), ("Tabel Beban Titik Buhul", nl), ("Tabel Beban Bentang Elemen", ml)):
        if len(df):
            S += [Paragraph(f"<b>{title}</b>", H3), _tbl(df), Spacer(1, 4)]

    # Tahapan Proses Perhitungan (Transparansi)
    S.append(Paragraph("3. Transparansi Tahapan Proses Perhitungan Matriks", H2))
    S.append(Paragraph(
        f"Struktur memiliki total <b>{calc_steps['total_dof']} DOF</b> ({calc_steps['free_dof']} DOF Bebas, {calc_steps['restrained_dof']} DOF Tertahan Tumpuan). "
        "Matriks kekakuan lokal elemen ditransformasikan ke koordinat global menggunakan matriks rotasi T: <b>k_global = Tᵀ · k_lokal · T</b>, lalu dirakit menjadi Matriks Kekakuan Global <b>K</b>.",
        B
    ))
    S.append(Spacer(1, 4))
    S.append(Paragraph("<b>Properti Geometri & Koefisien Kekakuan Elemen:</b>", H3))
    S.append(_tbl(calc_steps["df_elem"]))
    S.append(Spacer(1, 4))

    if not calc_steps["df_fe"].empty and "f1 (N_i)" in calc_steps["df_fe"].columns:
        S.append(Paragraph("<b>Gaya Ujung Jepit (Fixed End Actions) Akibat Beban Bentang:</b>", H3))
        S.append(_tbl(calc_steps["df_fe"]))
        S.append(Spacer(1, 4))

    # Hasil Analisis
    S.append(Paragraph("4. Hasil Perhitungan Analisis Struktur", H2))
    for title, df in (("Perpindahan & Rotasi Titik Buhul (Node Displacements)", t["disp"]),
                      ("Reaksi Tumpuan (Support Reactions)", t["reac"]),
                      ("Gaya-Gaya Dalam Ujung Elemen (Member End Actions)", t["forces"]),
                      ("Nilai Ekstrem Gaya Dalam per Elemen", t["ext"])):
        S += [Paragraph(f"<b>{title}</b>", H3), _tbl(df), Spacer(1, 4)]

    # Interpretasi Rekayasa Otomatis
    S.append(Paragraph("5. Interpretasi & Makna Hasil Rekayasa", H2))
    for ins in eng_insights:
        S.append(Paragraph(f"<b>{ins['ikon']} {ins['kategori']}:</b> {ins['ringkasan']}", B_Bold))
        S.append(Paragraph(f"{ins['detail']}", B))
        S.append(Spacer(1, 3))

    # Pembandingan / Verifikasi
    S.append(Paragraph("6. Verifikasi & Pembandingan Ketepatan Solusi (Benchmark)", H2))
    eq = res["equilibrium"]
    eq_ok = max(abs(eq["SFx"]), abs(eq["SFy"]), abs(eq["SM"])) < 1e-5
    S.append(Paragraph(
        f"<b>Pemeriksaan Kesetimbangan Statika Global:</b><br/>"
        f"• ΣFx = {eq['SFx']:.2e} kN &nbsp;|&nbsp; ΣFy = {eq['SFy']:.2e} kN &nbsp;|&nbsp; ΣM(asal) = {eq['SM']:.2e} kN·m<br/>"
        f"• Residu Sistem K·u − F = {res['residual']:.2e}<br/>"
        f"<b>Status Validasi:</b> {'TERPENUHI (Statika Seimbang Sempurna) ✅' if eq_ok else 'PERINGATAN: Residu tidak nol ⚠️'}",
        Callout
    ))
    if not benchmark_df.empty:
        S.append(Paragraph("<b>Perbandingan Terhadap Solusi Teoritis Tertutup:</b>", H3))
        S.append(_tbl(benchmark_df))
        S.append(Spacer(1, 4))

    # Gambar Diagram
    S.append(Paragraph("7. Gambar & Diagram Hasil Analisis", H2))
    for kind, title in (("model", "Model Struktur & Pembebanan"),
                        ("def", "Bentuk Deformasi Struktur Elastis"),
                        ("N", "Diagram Bidang Gaya Aksial N (kN) - Hijau"),
                        ("V", "Diagram Bidang Gaya Geser V (kN) - Oranye"),
                        ("M", "Diagram Bidang Momen Lentur M (kN·m) - Biru (Sisi Serat Tarik)")):
        S.append(KeepTogether([
            Paragraph(f"<b>{title}</b>", H3),
            _drawing(res, kind),
            Spacer(1, 6)
        ]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm,
        topMargin=1.5*cm, bottomMargin=1.5*cm
    )
    doc.build(S)
    return buf.getvalue()
