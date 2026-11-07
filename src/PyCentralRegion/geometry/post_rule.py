"""The post rule: posts on the gap edges between the beam's turns, placed from a tracked run of the bare electrodes.

On one edge of a gap (``builder.gap_edge``) the crossings of the run's survivors are taken at the edge and at the edge
moved post_size_mm into its electrode, as arclength s along the edge (the second projected back onto it), and grouped
by the particle's pass, 1 + floor(unwrapped azimuth / 360 deg), the readout's numbering: band p is what crosses in pass
p. (Pooled over all passes the crossings fill the edge without gaps: the turns overlap in radius at the 100 % level, so
a split at gaps in the crossings fails.) A band counts when its crossings number at least min_share times the
survivors: a survivor crosses both lines, so a band every survivor crosses once has a share of 2, and min_share 0.2
asks about 10 % of them. Its edges are its quantiles lo_q and 1 - lo_q.

Row k of [post_rule] is the free stretch between bands k and k + 1, clear_mm kept to each band's edge, on the gaps of
``gaps``. It is split at the edge's bends (``split_at_bends``), and every piece of at least post_size_mm is a post of
that length. An inner post sits inside the edge's first band, on the gaps of ``inner_gaps``: its outer end inner_clear_mm
inside the band's lower edge, inner_len_mm long at most, never below radius inner_rmin_mm or arclength inner_smin_mm.
A stretch too short for a post, or an edge whose side cannot be told, is noted in the edge's record and gets none.
"""
import pickle
from pathlib import Path

import numpy as np

from ..config import section, to_dict
from . import builder as B
from .options import SIDES

TRACK_DATA = 'track_data.pkl'


def load_tracks(path, r_max_mm, z_max_mm):
    """The recorded tracks of a run's final bunch: (xyz (S, N, 3) mm, live (S, N), survived (N,)). live: from birth to
    the metal hit, finite and inside the loss limits (a track's records after it left them are frozen copies);
    survived: live at the last record, no metal hit."""
    with open(path, 'rb') as fh:
        tr = pickle.load(fh)['final']['tracks']
    steps = np.asarray(tr['steps']).astype(np.int64)
    xyz = np.asarray(tr['xyz'], dtype=float) * 1e3
    stop = np.asarray(tr['stop_step']).astype(np.int64)
    birth = np.asarray(tr['birth_step']).astype(np.int64)
    live = (steps[:, None] >= birth[None, :]) & (steps[:, None] < stop[None, :]) & np.isfinite(xyz).all(axis=2)
    out = (np.hypot(xyz[..., 0], xyz[..., 1]) > r_max_mm) | (np.abs(xyz[..., 2]) > z_max_mm)
    live &= ~np.maximum.accumulate(out, axis=0)
    return xyz, live, (stop >= int(tr['n_steps'])) & live[-1]


def offset_polyline(edge, poly, d, what, owner):
    """The edge moved d [mm] into its electrode (per-vertex mean of the adjacent segments' inward normals)."""
    seg = np.diff(edge, axis=0)
    t = seg / np.linalg.norm(seg, axis=1)[:, None]
    n = np.array([B._into(poly, edge[k] + 0.5 * seg[k], t[k], what, owner) for k in range(len(seg))])
    nv = np.vstack([n[:1], 0.5 * (n[:-1] + n[1:]), n[-1:]])
    nv /= np.linalg.norm(nv, axis=1)[:, None]
    return edge + d * nv


def crossings(edge, xyz, live):
    """Every crossing of the polyline ``edge`` (M, 2) by every particle's live track segment: arrays (particle, record,
    point (n, 2), edge arclength). Exact segment-segment intersection (as builder.edge_crossings), all at once."""
    ok = live[:-1] & live[1:]
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    P0, P1 = xyz[:-1, :, :2], xyz[1:, :, :2]
    lo, hi = edge.min(axis=0) - 1e-9, edge.max(axis=0) + 1e-9
    near = ok & (np.minimum(P0[..., 0], P1[..., 0]) <= hi[0]) & (np.maximum(P0[..., 0], P1[..., 0]) >= lo[0]) \
        & (np.minimum(P0[..., 1], P1[..., 1]) <= hi[1]) & (np.maximum(P0[..., 1], P1[..., 1]) >= lo[1])
    kk, nn = np.nonzero(near)
    a, r = P0[kk, nn], P1[kk, nn] - P0[kk, nn]
    who, rec, pts, arc = [], [], [], []
    for m in range(len(seg)):
        c, sv = edge[m], seg[m]
        den = r[:, 0] * sv[1] - r[:, 1] * sv[0]
        good = np.abs(den) > 1e-12
        safe = np.where(good, den, 1.0)
        d = c - a
        t = np.where(good, (d[:, 0] * sv[1] - d[:, 1] * sv[0]) / safe, np.nan)
        u = np.where(good, (d[:, 0] * r[:, 1] - d[:, 1] * r[:, 0]) / safe, np.nan)
        hit = np.nonzero(good & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1))[0]
        who.append(nn[hit])
        rec.append(kk[hit])
        pts.append(a[hit] + t[hit, None] * r[hit])
        arc.append(sl[m] + u[hit] * seg_len[m])
    return np.concatenate(who), np.concatenate(rec), np.concatenate(pts), np.concatenate(arc)


def project_s(edge, pts):
    """Arclength of the nearest point on the polyline for each point (n, 2)."""
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    best_d = np.full(len(pts), np.inf)
    best_s = np.zeros(len(pts))
    for m in range(len(seg)):
        u = np.clip(((pts - edge[m]) @ seg[m]) / seg_len[m] ** 2, 0.0, 1.0)
        q = edge[m] + u[:, None] * seg[m]
        dd = np.linalg.norm(pts - q, axis=1)
        b = dd < best_d
        best_d[b], best_s[b] = dd[b], sl[m] + u[b] * seg_len[m]
    return best_s


def unwrapped_azimuth(xyz, live):
    """Each particle's azimuth [deg] unwrapped along its live records, starting in (0, 360) (nan where not live)."""
    phi = np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))
    d = (np.diff(phi, axis=0) + 180.0) % 360.0 - 180.0
    d[~(live[:-1] & live[1:])] = 0.0
    cum = np.vstack([np.zeros((1, phi.shape[1])), np.cumsum(d, axis=0)])
    n = np.arange(phi.shape[1])
    first = np.argmax(live, axis=0)
    phi0 = phi[first, n] % 360.0
    out = phi0[None, :] + cum - cum[first, n][None, :]
    out[~live] = np.nan
    return out


def bands_by_pass(s, passes, n_part, min_share, lo_q):
    """Crossings grouped by pass: {p: dict(n, share, e_lo, e_hi, p50)} for the passes with at least min_share of the
    n_part particles counted; e_lo / e_hi the band's quantiles lo_q and 1 - lo_q."""
    out = {}
    for p in np.unique(passes):
        q = np.asarray(s)[passes == p]
        share = q.size / n_part
        if share >= min_share:
            out[int(p)] = dict(n=int(q.size), share=float(share), e_lo=float(np.quantile(q, lo_q)),
                               e_hi=float(np.quantile(q, 1.0 - lo_q)), p50=float(np.median(q)))
    return out


def split_at_bends(edge, lo_s, hi_s, size, margin=0.25, round_ends=True):
    """The free stretch [lo_s, hi_s] of an edge as straight pieces [(a, b)], split at the edge's vertices inside it, so
    that every post lies on one straight segment (a straight stadium over a bend sticks out of its electrode at a convex
    corner, or leaves its face off the edge at a concave one). At a split each piece with round ends runs on past the
    vertex by delta = size/2 (1 - tan(theta/2)) - margin (theta the bend): its round cap then stays inside the
    electrode at a convex corner (it would stick out by size/2 (1 - cos theta) at delta = size/2) while the two caps
    overlap. A square end (``round_ends`` False: geometry.post_shape 'square') stops at the vertex, where it stays inside
    at either kind of corner. Pieces shorter than ``size`` are dropped."""
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    cuts = [(k, float(sl[k])) for k in range(1, len(sl) - 1) if lo_s + 1e-6 < sl[k] < hi_s - 1e-6]
    if not cuts:
        return [(lo_s, hi_s)]
    ends = [lo_s] + [s for _, s in cuts] + [hi_s]
    ext = []
    for k, _ in cuts:
        t0, t1 = seg[k - 1] / seg_len[k - 1], seg[k] / seg_len[k]
        th = float(np.arccos(np.clip(t0 @ t1, -1.0, 1.0)))
        ext.append(max(0.0, 0.5 * size * (1.0 - np.tan(0.5 * th)) - margin) if round_ends else 0.0)
    out = []
    for i in range(len(ends) - 1):
        a = ends[i] - (ext[i - 1] if i > 0 else 0.0)
        b = ends[i + 1] + (ext[i] if i < len(cuts) else 0.0)
        a, b = max(a, lo_s), min(b, hi_s)
        if b - a >= size:
            out.append((a, b))
    return out


def s_at_radius(edge, r_min):
    """The first arclength of the polyline at which its radius reaches r_min (0 if it starts beyond; the edge's length
    if it never does)."""
    seg = np.diff(edge, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    sl = np.concatenate([[0.0], np.cumsum(seg_len)])
    r = np.hypot(edge[:, 0], edge[:, 1])
    if r[0] >= r_min:
        return 0.0
    for k in range(len(seg)):
        if r[k + 1] >= r_min:
            u = np.linspace(0.0, 1.0, 2001)
            rr = np.hypot(*(edge[k][:, None] + seg[k][:, None] * u))
            return float(sl[k] + seg_len[k] * u[np.argmax(rr >= r_min)])
    return float(sl[-1])


def edge_crossings_by_pass(edge, poly, owner, size, xyz, live, phiu, what):
    """The survivors' crossings of one edge: (arclengths, passes) at the edge and at the edge moved ``size`` into its
    electrode. xyz, live, phiu: of the survivors only."""
    off = offset_polyline(edge, poly, size, what, owner)
    n0, k0, _, s0 = crossings(edge, xyz, live)
    n1, k1, p1, _ = crossings(off, xyz, live)
    s1 = project_s(edge, p1) if len(p1) else np.zeros(0)
    return (np.concatenate([s0, s1]),
            (1 + np.floor(phiu[np.concatenate([k0, k1]), np.concatenate([n0, n1])] / 360.0)).astype(int))


def place_posts(rule, edges, xyz, live, survived, size, round_ends=True):
    """The rule on tracks (``load_tracks``): (posts, records). ``edges``: {(gap, side): (edge polyline, its electrode's
    footprint polygon, the electrode's label)}, each edge in mm; ``size``: the post's depth and least length;
    ``round_ends``: the posts' shape has round ends (``split_at_bends``).

    posts: one dict per post, {gap, side, s_mm, length_mm (as geometry.posts takes them), between_bands [k, k + 1] (an
    inner post: [0, first band]), r_mm}, in the order of ``edges``, an edge's inner post first. records: one per edge,
    {'gap', 'side', 'owner', 'bands': {pass: dict(n, share, e_lo, e_hi, p50, r_e_lo, r_e_hi, r_p50)}, 'skipped': [why]}."""
    posts, records = [], []
    n_surv = int(np.count_nonzero(survived))
    xyz, live = xyz[:, survived], live[:, survived]
    phiu = unwrapped_azimuth(xyz, live) if n_surv else None
    if n_surv:                                  # the azimuth each particle gains from its first live record to its last
        idx, first, last = np.arange(n_surv), np.argmax(live, axis=0), len(live) - 1 - np.argmax(live[::-1], axis=0)
        if np.nanmedian(phiu[last, idx] - phiu[first, idx]) < 0.0:
            raise NotImplementedError("the beam turns clockwise: the rule numbers the passes of a counterclockwise one")
    for (gap, side), (edge, poly, owner) in edges.items():
        edge = np.asarray(edge, dtype=float)
        what = f"gap {gap} {side}"
        rec = {'gap': int(gap), 'side': side, 'owner': owner, 'bands': {}, 'skipped': []}
        records.append(rec)
        rows = [k for k in rule.rows if gap in rule.gaps]
        wants_inner = gap in rule.inner_gaps
        if not n_surv:
            rec['skipped'].append('no survivors')
            continue
        try:
            s_all, pas = edge_crossings_by_pass(edge, poly, owner, size, xyz, live, phiu, what)
        except RuntimeError as ex:                     # an edge whose inward side cannot be told gets no posts
            rec['skipped'].append(str(ex))
            continue
        bands = bands_by_pass(s_all, pas, n_surv, rule.min_share, rule.lo_q)
        rec['bands'] = bands
        for b in bands.values():
            for key in ('e_lo', 'e_hi', 'p50'):
                b['r_' + key] = float(np.hypot(*B._edge_point(edge, b[key])[0]))
        stretches = []
        if wants_inner:
            if bands:
                k1 = min(bands)
                hi_s = float(np.quantile(s_all[pas == k1], rule.lo_q)) - rule.inner_clear_mm
                lo_s = max(hi_s - rule.inner_len_mm, s_at_radius(edge, rule.inner_rmin_mm), rule.inner_smin_mm)
                stretches.append((lo_s, hi_s, (0, k1), 'inner'))
            else:
                rec['skipped'].append(f"inner: no band with >= {rule.min_share:g}")
        for k in rows:
            if k in bands and k + 1 in bands:
                stretches.append((bands[k]['e_hi'] + rule.clear_mm, bands[k + 1]['e_lo'] - rule.clear_mm, (k, k + 1),
                                  f"row {k}"))
            else:
                rec['skipped'].append(f"row {k}: only passes {sorted(bands)} have >= {rule.min_share:g}")
        edge_len = float(np.sum(np.linalg.norm(np.diff(edge, axis=0), axis=1)))
        for lo_s, hi_s, pair, label in stretches:
            if hi_s - lo_s < size:
                rec['skipped'].append(f"{label}: {hi_s - lo_s:.1f} mm free after the clearances")
            elif not (0.0 <= lo_s and hi_s <= edge_len):
                rec['skipped'].append(f"{label}: off the edge")
            else:
                for pa, pb in split_at_bends(edge, lo_s, hi_s, size, round_ends=round_ends):
                    s_c, length = 0.5 * (pa + pb), pb - pa
                    p, t, _ = B._edge_point(edge, s_c)
                    try:
                        B._into(poly, p, t, f"{what} {label}", owner)
                    except RuntimeError as ex:
                        rec['skipped'].append(str(ex))
                        continue
                    posts.append({'gap': int(gap), 'side': side, 's_mm': round(float(s_c), 3),
                                  'length_mm': round(float(length), 3), 'between_bands': list(pair),
                                  'r_mm': round(float(np.hypot(*p)), 3)})
    return posts, records


def posts_from_run(cfg, rule=None, run_dir=None):
    """The rule on the run in ``run_dir`` (default run.out_dir) of the electrodes of cfg, as ``geometry.build.layout``
    lays them out: {'posts', 'edges' (the records of ``place_posts``), 'n_tracks', 'n_survivors', 'rule'}."""
    from .build import layout
    rule = rule or section(cfg, 'post_rule')
    run_dir = Path(run_dir or section(cfg, 'run').out_dir)
    mc, g = section(cfg, 'machine'), section(cfg, 'geometry')
    lay = layout(cfg, verbose=False)
    edges = {(gap, side): B.gap_edge(lay['foot'], lay['dummy'], gap, side)
             for gap in range(2 * len(lay['foot'])) for side in SIDES if gap in rule.gaps or gap in rule.inner_gaps}
    xyz, live, survived = load_tracks(run_dir / TRACK_DATA, mc.r_max_m * 1e3, mc.z_max_m * 1e3)
    posts, records = place_posts(rule, edges, xyz, live, survived, float(g.post_size_mm),
                                 round_ends=g.post_shape != 'square')
    return {'posts': posts, 'edges': records, 'n_tracks': int(xyz.shape[1]), 'n_survivors': int(np.count_nonzero(survived)),
            'rule': to_dict(rule)}
