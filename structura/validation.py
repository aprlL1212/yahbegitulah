"""Konversi tabel UI -> model dan validasi input menyeluruh."""
import numpy as np
import pandas as pd

SPEC = {
    "nodes": dict(num=["id", "x", "y"], req=["id", "x", "y"], ints=["id"]),
    "members": dict(num=["id", "ni", "nj", "E", "b", "h"], req=["id", "ni", "nj", "E", "b", "h"],
                    ints=["id", "ni", "nj"]),
    "supports": dict(num=["node"], req=["node"], ints=["node"], bool=["ux", "uy", "rz"]),
    "nodal_loads": dict(num=["node", "Fx", "Fy", "M"], req=["node"], ints=["node"], fill0=["Fx", "Fy", "M"]),
    "member_loads": dict(num=["member", "value", "a"], req=["member", "value"], ints=["member"],
                         fill0=["a"], txt={"type": "UDL", "dir": "Gravitasi"}),
}
LABEL = {"nodes": "Node", "members": "Elemen", "supports": "Tumpuan",
         "nodal_loads": "Beban titik buhul", "member_loads": "Beban elemen"}


def to_model(tables):
    errs, model = [], {}
    for key, sp in SPEC.items():
        df = tables[key].copy()
        for c in sp["num"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[~df[sp["req"]].isna().all(axis=1)]
        for c in sp.get("fill0", []): df[c] = df[c].fillna(0.0)
        for c in sp.get("bool", []): df[c] = df[c].fillna(False).astype(bool)
        for c, d in sp.get("txt", {}).items(): df[c] = df[c].fillna(d).astype(str)
        bad = df[sp["req"]].isna().any(axis=1)
        if bad.any():
            errs.append(f"Tabel {LABEL[key]}: ada {int(bad.sum())} baris belum lengkap atau bukan format angka yang valid.")
        recs = df[~bad].to_dict("records")
        for r in recs:
            for c in sp["ints"]: r[c] = int(round(r[c]))
        model[key] = recs
    return model, errs


def validate(model):
    """Return (errors, warnings). Melakukan validasi menyeluruh geometri, penampang, tumpuan, dan beban."""
    E, W = [], []
    nodes, mems = model["nodes"], model["members"]
    
    if not nodes:
        E.append("Belum ada node yang didefinisikan dalam model.")
    if not mems:
        E.append("Belum ada elemen batang (balok/kolom) dalam model.")
        
    nid = [n["id"] for n in nodes]
    mid = [m["id"] for m in mems]
    if len(set(nid)) != len(nid):
        E.append("Terdapat ID Node yang duplikat. Pastikan setiap titik memiliki nomor ID unik.")
    if len(set(mid)) != len(mid):
        E.append("Terdapat ID Elemen yang duplikat. Setiap batang harus memiliki nomor ID unik.")
        
    pts = {n["id"]: (float(n["x"]), float(n["y"])) for n in nodes}
    ids = list(pts)
    
    # Validasi koordinat bertumpuk
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            dist = np.hypot(pts[ids[a]][0] - pts[ids[b]][0], pts[ids[a]][1] - pts[ids[b]][1])
            if dist < 1e-5:
                E.append(f"Node {ids[a]} dan Node {ids[b]} berada pada posisi koordinat yang sama (jarak = {dist:.4f} m).")

    L = {}
    used = set()
    pairs = set()
    adj = {i: set() for i in ids}
    
    for m in mems:
        i, j = m["ni"], m["nj"]
        if i not in pts or j not in pts:
            E.append(f"Elemen {m['id']} menghubungkan node yang tidak terdaftar ({i} - {j}).")
            continue
        if i == j:
            E.append(f"Elemen {m['id']}: Node awal sama dengan node akhir ({i}). Elemen tidak memiliki panjang.")
            continue
            
        len_m = float(np.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]))
        L[m["id"]] = len_m
        if len_m < 1e-4:
            E.append(f"Elemen {m['id']} memiliki panjang nol ({len_m:.4f} m).")
        elif len_m > 100.0:
            W.append(f"Elemen {m['id']} memiliki bentang sangat panjang ({len_m:.1f} m). Pastikan satuan koordinat dalam meter.")

        # Validasi material dan penampang fisik
        for k, nm in (("E", "Modulus Elastisitas E"), ("b", "Lebar penampang b"), ("h", "Tinggi penampang h")):
            val = m[k]
            if val <= 0:
                E.append(f"Elemen {m['id']}: Nilai {nm} harus lebih besar dari 0 (saat ini: {val}).")
            elif k == "E" and val < 1000:
                W.append(f"Elemen {m['id']}: Nilai Modulus E = {val} MPa tergolong sangat rendah untuk material teknik sipil.")
            elif k in ("b", "h") and val > 5000:
                W.append(f"Elemen {m['id']}: Dimensi {nm} = {val} mm tergolong sangat besar. Pastikan satuan dalam milimeter.")

        # Rasio kelangsingan penampang b/h
        if m["b"] > 0 and m["h"] > 0:
            aspect = max(m["b"] / m["h"], m["h"] / m["b"])
            if aspect > 10:
                W.append(f"Elemen {m['id']}: Rasio dimensi b/h ({aspect:.1f}) sangat pipih; asumsi balok Euler-Bernoulli mungkin kurang ideal.")

        used |= {i, j}
        adj[i].add(j)
        adj[j].add(i)
        p = frozenset((i, j))
        if p in pairs:
            W.append(f"Terdapat elemen ganda yang menghubungkan pasangan Node {i} dan Node {j}.")
        pairs.add(p)

    # Node mengambang tanpa elemen
    for n in set(ids) - used:
        E.append(f"Node {n} berdiri sendiri dan tidak terhubung ke elemen batang mana pun.")

    # Cek struktur terfragmentasi (terpisah)
    if ids and used:
        seen, stack = set(), [next(iter(used))]
        while stack:
            k = stack.pop()
            if k not in seen:
                seen.add(k)
                stack += list(adj[k] - seen)
        if used - seen:
            W.append("Struktur terbagi menjadi beberapa bagian rangka yang terpisah; pastikan tiap rangka memiliki tumpuan yang stabil.")

    # Validasi tumpuan
    nres, seen_s = 0, set()
    for sp in model["supports"]:
        if sp["node"] not in pts:
            E.append(f"Tumpuan merujuk Node {sp['node']} yang tidak ada di tabel Node.")
            continue
        if sp["node"] in seen_s:
            E.append(f"Node {sp['node']} didaftarkan lebih dari satu kali pada tabel tumpuan.")
        seen_s.add(sp["node"])
        nres += int(sp["ux"]) + int(sp["uy"]) + int(sp["rz"])
        
    if nres < 3:
        E.append(f"Jumlah reaksi tumpuan = {nres}; minimal 3 agar struktur 2D stabil.")

    # Validasi beban
    for nl in model["nodal_loads"]:
        if nl["node"] not in pts:
            E.append(f"Beban titik buhul diterapkan pada Node {nl['node']} yang tidak terdaftar.")
            
    for ml in model["member_loads"]:
        if ml["member"] not in L:
            E.append(f"Beban elemen bekerja pada Elemen {ml['member']} yang tidak valid.")
            continue
        if ml["type"] not in ("UDL", "Titik"):
            E.append(f"Jenis beban pada Elemen {ml['member']} harus berupa 'UDL' atau 'Titik'.")
        if ml["dir"] not in ("Gravitasi", "Lokal-y", "Lokal-x"):
            E.append(f"Arah beban pada Elemen {ml['member']} harus salah satu dari: Gravitasi, Lokal-y, atau Lokal-x.")
        if ml["type"] == "Titik":
            a_pos = float(ml["a"])
            mem_l = L[ml["member"]]
            if not (0 < a_pos < mem_l):
                E.append(f"Beban titik pada elemen {ml['member']}: jarak a harus 0 < a < L ({mem_l:.3f} m).")

    if not model["nodal_loads"] and not model["member_loads"]:
        W.append("Tidak ada beban yang bekerja pada struktur; seluruh perpindahan dan gaya dalam akan bernilai 0.")

    return E, W
