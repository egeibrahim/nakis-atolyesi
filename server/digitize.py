"""Görsel -> Ink/Stitch SVG -> PES/DST.

Ink/Stitch is used as the stitch engine. This module turns a raster image into an
SVG whose shapes carry Ink/Stitch parameters (fill, satin column, running stitch),
then calls Ink/Stitch's own output extension to generate the embroidery file.
"""
import base64
import io
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import cv2
import numpy as np
import pystitch
from skimage.morphology import skeletonize

INKSTITCH_DIR = os.environ.get("INKSTITCH_DIR", "/opt/inkstitch")
INKSTITCH_PY = os.environ.get("INKSTITCH_PYTHON", "python")
STUBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stubs")
INK_TIMEOUT = int(os.environ.get("INKSTITCH_TIMEOUT", "240"))


@dataclass
class Params:
    width_mm: float = 100.0
    ncol: int = 5
    remove_bg: str = "edge"          # edge | all | off
    satin_mm: float = 5.0            # auto: strokes up to this width become satin
    min_detail_mm: float = 1.0
    row_mm: float = 0.4              # default fill row spacing
    max_stitch_mm: float = 4.0
    angle: str = "alt"               # alt | number
    underlay: bool = True
    overrides: list = field(default_factory=list)   # [{nx,ny,c:[r,g,b],s:{...}}]
    color_overrides: list = field(default_factory=list)  # [{c:[r,g,b],s:{...}}]

    @classmethod
    def from_dict(cls, d):
        p = cls()
        p.width_mm = float(np.clip(float(d.get("width", p.width_mm)), 10, 400))
        p.ncol = int(np.clip(int(d.get("ncol", p.ncol)), 1, 15))
        p.remove_bg = str(d.get("removeBg", p.remove_bg))
        p.satin_mm = float(d.get("satin", p.satin_mm))
        p.min_detail_mm = float(d.get("minDet", p.min_detail_mm))
        p.row_mm = float(d.get("row", p.row_mm))
        p.max_stitch_mm = float(d.get("maxst", p.max_stitch_mm))
        p.angle = str(d.get("angle", p.angle))
        p.underlay = bool(d.get("underlay", p.underlay))
        p.overrides = list(d.get("overrides", []))[:500]
        p.color_overrides = list(d.get("colorOverrides", []))[:50]
        return p


# ---------------------------------------------------------------- image -> labels

def _rgb2lab(rgb):
    return cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def quantize(img_bytes, p: Params):
    arr = np.frombuffer(img_bytes, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError("Görsel okunamadı. PNG, JPG veya WEBP yükleyin.")
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGRA)
    elif im.shape[2] == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2BGRA)
    rgba = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
    h0, w0 = rgba.shape[:2]
    m = 2000 / max(h0, w0)
    if m < 1:
        rgba = cv2.resize(rgba, (round(w0 * m), round(h0 * m)), interpolation=cv2.INTER_AREA)
    sh, sw = rgba.shape[:2]
    rgb = rgba[..., :3].astype(np.int32)
    on = rgba[..., 3] > 128

    if p.remove_bg != "off":
        bg = rgba[0, 0]
        if bg[3] > 128:
            like = (np.abs(rgb - bg[:3].astype(np.int32)).sum(-1) < 60)
            if p.remove_bg == "edge":
                n, lab = cv2.connectedComponents(like.astype(np.uint8), connectivity=4)
                border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
                border = border[border > 0]
                like = np.isin(lab, border)
            on &= ~like

    ys, xs = np.where(on)
    if len(xs) == 0:
        raise ValueError("Görselde dikilecek alan bulunamadı. Arka plan seçeneğini değiştirmeyi deneyin.")
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgb = rgba[y0:y1, x0:x1, :3]
    on = on[y0:y1, x0:x1]
    ch, cw = on.shape

    # work resolution: ~10 px/mm, capped
    W = int(min(2400, max(64, round(p.width_mm * 10))))
    PX = p.width_mm / W
    H = max(8, round(W * ch / cw))
    rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    cov = cv2.resize(on.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    on = cov > 0.7

    lab = _rgb2lab(rgb[on])
    K = min(p.ncol, len(lab))
    rng = np.random.default_rng(0)
    samp = lab[rng.choice(len(lab), min(40000, len(lab)), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    cv2.setRNGSeed(1)
    _, _, centers = cv2.kmeans(samp, K, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    merged = []
    for c in centers:
        if not any(np.sum((m_ - c) ** 2) < 144 for m_ in merged):
            merged.append(c)
    centers = np.array(merged, np.float32)
    K = len(centers)

    d = ((lab[:, None, :] - centers[None]) ** 2).sum(-1)
    lbl = np.full((H, W), -1, np.int32)
    lbl[on] = d.argmin(1)

    # soft mode filter: smooth each label's mask, keep the strongest (background included)
    k = max(3, int(round(0.3 / PX)) | 1)
    stack = [cv2.GaussianBlur((lbl == -1).astype(np.float32), (k, k), 0)]
    for i in range(K):
        stack.append(cv2.GaussianBlur((lbl == i).astype(np.float32), (k, k), 0))
    lbl = np.argmax(np.stack(stack), 0).astype(np.int32) - 1

    # merge islands smaller than the minimum detail into the nearest region
    min_area = (p.min_detail_mm / PX) ** 2 * 0.8
    unknown = np.zeros((H, W), bool)
    for i in range(K):
        n, cl, st, _ = cv2.connectedComponentsWithStats((lbl == i).astype(np.uint8), connectivity=4)
        small = np.where(st[1:, cv2.CC_STAT_AREA] < min_area)[0] + 1
        if len(small):
            unknown |= np.isin(cl, small)
    if unknown.any() and not unknown.all():
        # zero pixels (= known) are the sources; each gets its own label in raster order
        _, nearest = cv2.distanceTransformWithLabels(unknown.astype(np.uint8), cv2.DIST_L2, 5,
                                                     labelType=cv2.DIST_LABEL_PIXEL)
        zero_px = np.flatnonzero(~unknown.ravel())
        src = np.flatnonzero(unknown.ravel())
        idx = np.clip(nearest.ravel()[src] - 1, 0, len(zero_px) - 1)
        flat = lbl.ravel()
        flat[src] = flat[zero_px[idx]]
        lbl = flat.reshape(H, W)

    rgb_c = np.zeros((K, 3))
    for i in range(K):
        sel = lbl == i
        if sel.any():
            rgb_c[i] = rgb[sel].mean(0)
    light = centers[:, 0]
    return lbl, rgb_c, light, PX, W, H


# ---------------------------------------------------------------- geometry helpers

def _fmt(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _poly_d(pts, PX, close=True):
    s = "M" + " L".join(f"{_fmt(x * PX)},{_fmt(y * PX)}" for x, y in pts)
    return s + ("Z" if close else "")


def contours_d(mask, PX):
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    parts = []
    for c in cs:
        if len(c) < 4:
            continue
        a = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2).astype(float) + 0.5
        if len(a) >= 3:
            parts.append(_poly_d(a, PX))
    return " ".join(parts)


def outer_contour_d(mask, PX):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return ""
    c = max(cs, key=cv2.contourArea)
    a = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2).astype(float) + 0.5
    return _poly_d(a, PX)


# --- satin columns from a raster stroke: skeleton -> centre lines -> two rails each

_NB = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def _skeleton_edges(skel):
    pts = set(zip(*np.nonzero(skel.T)))  # (x, y)

    def nbrs(p):
        x, y = p
        return [(x + dx, y + dy) for dx, dy in _NB if (x + dx, y + dy) in pts]

    deg = {p: len(nbrs(p)) for p in pts}
    nodes = {p for p, d in deg.items() if d != 2}
    edges, seen = [], set()
    for n in nodes:
        for nb in nbrs(n):
            if (n, nb) in seen:
                continue
            path = [n, nb]
            seen.add((n, nb))
            prev, cur = n, nb
            while cur not in nodes:
                nxt = [q for q in nbrs(cur) if q != prev and q not in path[-3:]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            seen.add((cur, prev))
            edges.append(path)
    # pure loops (e.g. an "O"): every pixel has degree 2
    rest = pts - {q for e in edges for q in e}
    while rest:
        start = next(iter(rest))
        path, prev, cur = [start], None, start
        while True:
            nxt = [q for q in nbrs(cur) if q != prev and q not in path[-3:]]
            if not nxt or nxt[0] == start:
                break
            prev, cur = cur, nxt[0]
            if cur in path:
                break
            path.append(cur)
        rest -= set(path)
        if len(path) > 8:
            half = len(path) // 2
            edges.append(path[: half + 1])
            edges.append(path[half:] + [path[0]])
    return edges, deg


def _resample(path, step):
    p = np.array(path, float)
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return p[[0, -1]] if len(p) > 1 else p
    t = np.arange(0, s[-1], step)
    t = np.append(t, s[-1])
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def _smooth(a, w=5):
    if len(a) < w + 2:
        return a
    k = np.ones(w) / w
    out = a.copy()
    for j in range(a.shape[1]):
        pad = np.concatenate([np.repeat(a[:1, j], w // 2), a[:, j], np.repeat(a[-1:, j], w // 2)])
        out[:, j] = np.convolve(pad, k, "valid")
    out[0], out[-1] = a[0], a[-1]
    return out


def _ray(mask, p, d, maxlen):
    H, W = mask.shape
    t = 0.0
    while t < maxlen:
        x, y = p[0] + d[0] * t, p[1] + d[1] * t
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= W or yi >= H or not mask[yi, xi]:
            return max(0.0, t - 0.5)
        t += 0.5
    return maxlen


def satin_columns(mask, dt, PX):
    """Return list of (left_rail, right_rail, rungs) in px coordinates."""
    skel = skeletonize(mask > 0)
    edges, deg = _skeleton_edges(skel)
    if not edges:
        return []
    min_len = 1.0 / PX
    step = 0.4 / PX
    cols = []
    for e in edges:
        p = np.array(e, float)
        length = np.hypot(*np.diff(p, axis=0).T).sum() if len(p) > 1 else 0
        if length < min_len and len(edges) > 1:
            continue
        r = _smooth(_resample(e, step), 5)
        if len(r) < 2:
            continue

        def extend(end_pt, direction):
            n = np.hypot(*direction)
            if n == 0:
                return []
            dvec = direction / n
            t = _ray(mask, end_pt, dvec, 20 / PX)
            return [end_pt + dvec * t] if t > 0.5 else []

        head = extend(r[0], r[0] - r[min(2, len(r) - 1)]) if deg.get(e[0], 0) == 1 else []
        tail = extend(r[-1], r[-1] - r[max(-3, -len(r))]) if deg.get(e[-1], 0) == 1 else []
        r = np.array(head[::-1] + list(r) + tail)
        L, R = [], []
        for i in range(len(r)):
            a, b = r[max(0, i - 2)], r[min(len(r) - 1, i + 2)]
            tg = b - a
            n = np.hypot(*tg)
            if n == 0:
                continue
            tg /= n
            nm = np.array([-tg[1], tg[0]])
            xi, yi = int(round(r[i][0])), int(round(r[i][1]))
            hw = dt[min(max(yi, 0), dt.shape[0] - 1), min(max(xi, 0), dt.shape[1] - 1)]
            cap = max(1.5, hw * 1.6 + 1)
            dl, dr = _ray(mask, r[i], nm, cap), _ray(mask, r[i], -nm, cap)
            L.append(r[i] + nm * dl)
            R.append(r[i] - nm * dr)
        if len(L) < 2:
            continue
        L, R = _smooth(np.array(L), 3), _smooth(np.array(R), 3)
        rungs = []
        for i in range(0, len(L), max(1, int(3 / 0.4))):
            dv = L[i] - R[i]
            n = np.hypot(*dv)
            if n < 1:
                continue
            dv /= n
            rungs.append((L[i] + dv * 0.6, R[i] - dv * 0.6))
        cols.append((L, R, rungs))
    return cols


# ---------------------------------------------------------------- SVG assembly

def _settings_for(comp_mask, k, rgb_c, p: Params, W, H):
    def near_k(c):
        return int(np.argmin(((rgb_c - np.array(c, float)) ** 2).sum(1)))

    for e in p.overrides:
        try:
            if near_k(e["c"]) != k:
                continue
            x = min(W - 1, int(round(float(e["nx"]) * W)))
            y = min(H - 1, int(round(float(e["ny"]) * H)))
            if comp_mask[y, x]:
                return e["s"]
        except (KeyError, TypeError, ValueError):
            continue
    for e in p.color_overrides:
        try:
            if near_k(e["c"]) == k:
                return e["s"]
        except (KeyError, TypeError, ValueError):
            continue
    return None


def build_svg(lbl, rgb_c, light, PX, W, H, p: Params):
    K = len(rgb_c)
    areas = [(lbl == i).sum() for i in range(K)]
    ids = [i for i in range(K) if areas[i] > 0]
    base = max(ids, key=lambda i: areas[i])
    order = [base] + sorted([i for i in ids if i != base], key=lambda i: -light[i])
    min_area = (p.min_detail_mm / PX) ** 2 * 0.8
    out, stats = [], {"satin": 0, "fill": 0, "run": 0}
    for oi, k in enumerate(order):
        col = "#%02x%02x%02x" % tuple(int(round(v)) for v in rgb_c[k])
        n, cl, st, _ = cv2.connectedComponentsWithStats((lbl == k).astype(np.uint8), connectivity=8)
        group = []
        for ci in range(1, n):
            if st[ci, cv2.CC_STAT_AREA] < min_area:
                continue
            x, y, w, h = st[ci, :4]
            pad = 2
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            m = (cl[y0:y1, x0:x1] == ci).astype(np.uint8)
            full = np.zeros((H, W), bool)
            full[y0:y1, x0:x1] = m > 0
            s = _settings_for(full, k, rgb_c, p, W, H) or {}
            dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            width_mm = 2 * dt.max() * PX
            typ = s.get("type", "auto")
            if typ == "auto":
                typ = "satin" if (p.satin_mm > 0 and width_mm <= p.satin_mm) else "fill"
            if typ == "skip":
                continue
            den = float(s.get("den", 1 / p.row_mm))
            pull = float(s.get("pull", 0.2))
            under = bool(s.get("under", p.underlay))
            if p.angle == "alt":
                ang = [0, 45, -45, 90][oi % 4]
            else:
                ang = float(p.angle)
            ang = float(s.get("ang", ang))
            off = (x0, y0)

            def shift(d_px_pts):
                return d_px_pts + np.array(off, float)

            if typ == "run":
                pitch = float(s.get("pitch", 2.0))
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for c in cs:
                    a = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2).astype(float) + 0.5 + np.array(off)
                    if len(a) < 3:
                        continue
                    group.append(
                        f'<path d="{_poly_d(a, PX)}" style="fill:none;stroke:{col};stroke-width:0.2" '
                        f'inkstitch:running_stitch_length_mm="{_fmt(pitch)}"/>')
                stats["run"] += 1
                continue

            if typ == "satin":
                cols = satin_columns(m, dt, PX)
                if cols:
                    for L, R, rungs in cols:
                        L, R = shift(L), shift(R)
                        d = _poly_d(L, PX, False) + " " + _poly_d(R, PX, False)
                        for a, b in rungs:
                            d += " " + _poly_d(np.array([a, b]) + np.array(off, float), PX, False)
                        group.append(
                            f'<path d="{d}" style="fill:none;stroke:{col};stroke-width:0.1" '
                            f'inkstitch:satin_column="True" '
                            f'inkstitch:zigzag_spacing_mm="{_fmt(max(0.2, 1 / den))}" '
                            f'inkstitch:pull_compensation_mm="{_fmt(pull)}" '
                            f'inkstitch:contour_underlay="{under}" '
                            f'inkstitch:center_walk_underlay="{under}"/>')
                    stats["satin"] += 1
                    continue
                typ = "fill"  # could not build a column: fall back to fill

            cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            parts = []
            for c in cs:
                a = cv2.approxPolyDP(c, 0.6, True).reshape(-1, 2).astype(float) + 0.5 + np.array(off)
                if len(a) >= 3:
                    parts.append(_poly_d(a, PX))
            if not parts:
                continue
            group.append(
                f'<path d="{" ".join(parts)}" style="fill:{col};fill-rule:evenodd;stroke:none" '
                f'inkstitch:angle="{_fmt(ang)}" inkstitch:row_spacing_mm="{_fmt(max(0.15, 1 / den))}" '
                f'inkstitch:max_stitch_length_mm="{_fmt(p.max_stitch_mm)}" '
                f'inkstitch:expand_mm="{_fmt(pull)}" inkstitch:fill_underlay="{under}"/>')
            stats["fill"] += 1
        if group:
            out.append(f'<g inkscape:groupmode="layer" inkscape:label="Renk {oi + 1} {col}" id="renk{oi + 1}">' + "".join(group) + "</g>")
    wmm, hmm = W * PX, H * PX
    svg = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           '<svg xmlns="http://www.w3.org/2000/svg" '
           'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
           'xmlns:sodipodi="http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd" '
           'xmlns:inkstitch="http://inkstitch.org/namespace" '
           f'width="{_fmt(wmm)}mm" height="{_fmt(hmm)}mm" viewBox="0 0 {_fmt(wmm)} {_fmt(hmm)}">'
           + "".join(out) + "</svg>")
    return svg, stats


# ---------------------------------------------------------------- Ink/Stitch

def run_inkstitch(svg_text, fmt="pes"):
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "tasarim.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_text)
        env = dict(os.environ)
        env["PYTHONPATH"] = STUBS_DIR + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [INKSTITCH_PY, os.path.join(INKSTITCH_DIR, "inkstitch.py"), "--extension=output", f"--format={fmt}", path],
            cwd=INKSTITCH_DIR, env=env, capture_output=True, timeout=INK_TIMEOUT, stdin=subprocess.DEVNULL)
        if proc.returncode != 0 or not proc.stdout:
            raise RuntimeError("Ink/Stitch hata verdi: " + proc.stderr.decode("utf-8", "replace")[-800:])
        return proc.stdout


def pattern_payload(pes_bytes):
    pat = pystitch.read_pes(io.BytesIO(pes_bytes))
    dst = io.BytesIO()
    pystitch.write_dst(pat, dst)
    threads = [t.hex_color() for t in pat.threadlist]
    stitches = []
    cmds = {pystitch.STITCH: "S", pystitch.JUMP: "J", pystitch.TRIM: "T", pystitch.COLOR_CHANGE: "C", pystitch.STOP: "C"}
    trims = 0
    for x, y, c in pat.stitches:
        c &= pystitch.COMMAND_MASK
        t = cmds.get(c)
        if t is None:
            continue
        if t == "T":
            trims += 1
        stitches.append([round(x / 10, 2), round(y / 10, 2), t])
    count = sum(1 for s in stitches if s[2] == "S")
    return {"threads": threads, "stitches": stitches, "count": count, "trims": trims,
            "dst": base64.b64encode(dst.getvalue()).decode()}


def digitize(img_bytes, params: dict):
    p = Params.from_dict(params)
    lbl, rgb_c, light, PX, W, H = quantize(img_bytes, p)
    svg, stats = build_svg(lbl, rgb_c, light, PX, W, H, p)
    try:
        pes = run_inkstitch(svg)
    except RuntimeError:
        # a satin column Ink/Stitch could not handle: retry with satins drawn as fills
        p.satin_mm = 0
        for e in p.overrides + p.color_overrides:
            if isinstance(e.get("s"), dict) and e["s"].get("type") == "satin":
                e["s"]["type"] = "fill"
        svg, stats = build_svg(lbl, rgb_c, light, PX, W, H, p)
        stats["satin_fallback"] = True
        pes = run_inkstitch(svg)
    payload = pattern_payload(pes)
    payload.update({"pes": base64.b64encode(pes).decode(), "svg": svg, "stats": stats,
                    "size_mm": [round(W * PX, 1), round(H * PX, 1)]})
    return payload
