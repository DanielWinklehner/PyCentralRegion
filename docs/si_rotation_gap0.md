# Spiral-inflector rotation and gap-0 position (gap-0 orientation held)

Two `[geometry]` quantities that the corner rule used to tie to the dee opening. Each is a number a MOBO campaign can
vary (`kind = "geometry"`), and each is 0 by default, which builds exactly what the code built before they existed:

| key | unit | meaning (0 = today) | modes |
|---|---|---|---|
| `si_rotation_offset_deg` | deg, + ccw | the spiral-inflector assembly turned off the corner rule's angle | `si_rotation = "corner"` |
| `gap0_leave_mm` | mm, + outward | where gap 0 leaves the exit plate, along the plate from its home point | `gap_lines = "connected"` |

The third quantity of the request, gap 0's **orientation** (its first segment tilted off the plate, with a grounded
nose), is **held** until it is asked for again. The last section says what it would take and what it would cost.

**Readings chosen.** Daniel's sentence ("separation of SI housing rotation and gap0 position and orientation") can be
read two ways for the rotation and for the position. This note takes the first reading of each:

- *Rotation* is an **offset from the corner rule**, so 0 is today's build at every dee opening. The alternative, an
  absolute angle free of the opening, is `si_rotation = "fixed"` with `[inflector] rotation_deg`. That mode stays as it
  is, gap 0 included (see "The fixed mode" below).
- *Position* is the **point on the plate** where gap 0 leaves it, as an arclength along the turned plate line. The
  alternative, gap 0's radial ray turned free of the dee opening (an asymmetric dee 0), is not done.

The two are independent: the offset carries the leave point with the housing, and the leave point then moves along
the turned plate.

## Definitions

Conventions: machine frame, mm, seen from +z; angles are + counter-clockwise; `R(a)` turns a vector by `a` about the
machine axis.

From the export's `exit_plane_spec.json`:
- the plate line: the plate's outer face, unrotated, shifted by half the local gap-law width along its downstream
  normal `n`;
- its outward unit direction `d0`;
- the plate's outer corner `C0`;
- the export's gap-0 azimuth `a_x`.

Gap 0's ray is at azimuth `t0` (its base angle, set by the dee opening), with unit vector `e`.

1. **Corner-rule angle** `rot_c = t0 - a_x`. This is what `si_rotation = "corner"` turns the inflector by today: the
   export's placement of the plate corner relative to gap 0's ray, carried to another opening.
2. **Assembly rotation** `rot = rot_c + si_rotation_offset_deg`.
   - Everything of the inflector turns by `rot`: the housing outline and STEP, the exit aperture, the emitting plane
     (beam channel), the inflector's E-field and the hand-off bunch.
   - The build records `rot` in the model (`params['si_rotation_deg']`), and the tracker adopts it
     (`simulate3d.follow_model_si_rotation`, unchanged).
   - The turned plate direction is `d = R(rot) d0`.
3. **Home point** `H`. Today's leave point `x_c` is where the plate line turned by `rot_c` meets gap 0's ray. `H` is
   that point carried with the assembly: `H = R(si_rotation_offset_deg) x_c`. It sits at the same place on the plate
   for every offset.
4. **Leave point** `L = H + gap0_leave_mm * d`. It lies on the turned plate line: + towards (and past) the plate's
   outer corner, - towards the exit aperture.
5. **Gap 0** is the polyline `L - x_L d -> L -> P1 -> P2 -> its ray`.
   - Its first segment runs along the plate. `x_L = L.e`, so the first point lies well inside the housing.
   - `P1` and `P2` come from gap 0's `[[geometry.gap_line]]` item, as before.
   - In gap 0's own frame, `P0 = L` is `(x_L, y_L)`, with `y_L = e x L` (0 at the defaults).
   - `y1_max` is now the largest `|y1|` for which both `+|y1|` and `-|y1|` keep the bends within the limits. At
     `y_L = 0` this is the old even rule, to the bit.

```
                           dee 0                         P1 ----- P2 ------ ray (t0) --->
                                                    ___/
                                               ___/
        ______________________________________L .  .  .  .  . C (plate corner)    gap 0's line
                                              |                                    beyond L: L -> P1 -> P2
  =========[ exit aperture ]==================|==================C====   plate face (housing), direction d --->
                     |
     H = the home point on the plate; L = H + gap0_leave_mm d; the offset turns all of it (and the plate) about the axis
```

## Valid ranges

| key | config check | build check | campaign box (suggested) |
|---|---|---|---|
| `si_rotation_offset_deg` | finite; `si_rotation = "corner"` | gap 0's connected line must exist (below) | about -6 .. +6 |
| `gap0_leave_mm` | finite; connected lines | `L` outward of the exit aperture, `x_L < r1` | from just outward of the aperture's outer end |

The build checks raise `ConfigError`. `pcr build` then stops with the message, and in a campaign `pcr evaluate`
records the evaluation as failed with that message (no crash). The checks run only while either key is non-zero, so a
design at the defaults builds exactly as before in every mode. The config checks run at load and at a campaign's
preflight (`check_overrides`, at every corner of the box).

## What each does to the first turn

- **Offset.** The beam leaves the housing at the same place relative to the plate and meets gap 0's first segment
  there, so the first gap crossing and its energy gain do not change. The hand-off turns by the offset about the axis
  while gaps 1, 2 and 3 stay put, so the first turn reaches them earlier (+) or later (-): about the offset times the
  harmonic in RF phase, plus the change in path length. Beyond `L`, gap 0's line runs to `P1` and `P2` on its own ray,
  so its outer part bends by a different angle at `L`.
- **Leave point.** The first crossing stays where it is as long as `L` stays outward of the aperture. Moving `L`
  changes the length of gap 0 on the plate, and with it where later turns cross gap 0 (beyond `L`) and at what angle.
  It also moves dee 0's gap-0 edge from `L` out to `P1`.

## Refused, and why

| when | message (start) |
|---|---|
| offset != 0 with `si_rotation = "fixed"` | `geometry.si_rotation_offset_deg = ... turns the spiral inflector off the corner rule; with si_rotation = 'fixed' the turn is [inflector] rotation_deg alone ...` |
| leave != 0 with `gap_lines = "seed"` | `geometry.gap0_leave_mm = ... places gap 0's connected line; the seed's gap 0 ...` |
| `L` inside or inward of the exit aperture | `gap 0 would leave the exit plate ... mm inward of the exit aperture's outer end ...` |
| `L` at or beyond `r1` along gap 0's ray | `a connected gap line needs P0 < r1 < r2 ...` (as before) |
| no `y1` keeps the bend at `P1` within `bend_max_deg` (`L` far off the ray) | `gap 0: from its leave point ... no P1 at r1 = ... keeps the bend at P1 ...` |
| bend at `L` > 75 deg | `gap 0 bends by ... deg where it leaves the exit plate ...` |

## The fixed mode

`si_rotation = "fixed"` is unchanged: the assembly turns by `[inflector] rotation_deg`, and gap 0's home point is where
that turned plate meets gap 0's ray. In the exports seen so far the plate crosses the ray at 10-16 deg, so that point
runs along the plate several times faster than the turn (tens of mm for a few degrees), and the build fails once it
passes `r1`. Campaigns that want the assembly turned with gap 0 held on the plate should use `si_rotation = "corner"`
with `si_rotation_offset_deg`. `gap0_leave_mm` works in both modes, measured from its mode's home point.

## Outputs

`<prefix>_build.json` keeps everything it had (its `geometry` table now lists the two keys) and adds `leave` to gap 0's
`gap_lines` record:
- `H` and `L`;
- the offset and the leave arclength;
- `L` along and across gap 0's ray;
- the arclengths of the plate corner and the exit aperture from `L`.

The model's `params['si_rotation_deg']` is `rot`, as before. A MOBO campaign's geometry key leaves each of the two keys
out while it is 0, so existing campaign stores keep their keys.

## Held: gap 0's orientation (a tilt off the plate)

Not in this change. It is described here so the decision to build it can be taken on its cost.

**What it would be.** A third key, `gap0_tilt_deg` (>= 0, connected lines, star ground):
- gap 0's first segment turns clockwise about `L`, `d1 = R(-tilt) d`, so the accelerating field at the first crossing
  turns from `n` towards `+d`, radially outward along the plate;
- the segment's inner end swings off the plate into dee 0. Between the plate face and gap 0's dummy edge (dee 0's
  gap-0 edge offset by the local gap-law width) a wedge opens. Its depth `u` inward of `L` is about `u tan(tilt)`, and
  it must be ground metal, or the median plane sees a gap wider than designed;
- the wedge is filled, full height, by a grounded **nose** fused into the star ground before the inflector's exit
  opening is cut, so the opening (the exit aperture swept along the emitting plane's normal) is the beam's channel
  through the nose;
- the first-turn effect would be an outward kick of about `sin(tilt)` of the gap's momentum gain, after a drift
  through the nose. An earlier estimate: about 0.38 deg of exit angle per degree of tilt. Its purpose is to correct a
  poor inflector exit angle.

**What it would take.**
- Builder, about 120 lines:
  - the tilted first segment;
  - the nose polygon (the existing `gap0_face_polygon`, restricted to the edge inward of `L`, reaching
    `u_max tan(tilt) + 5 mm`);
  - its fuse into the star ground before the exit-opening cut;
  - three build checks: no housing within reach, the nose closer to a dee than the star ground's gap floor, and the
    bend at `L`.
- Config check and MOBO plumbing, about 15 lines.
- Tests: 2D layout properties and a 3D gmsh test that the nose is metal beside the opening and vacuum in it.
- Not buildable with the dummy-wedge ground (`ground = "dummies"`) or on the seed's gap lines.

**What it would cost beyond the code.**
- The inflector's static E-field map comes from the export and does not contain the nose. Near the exit, the field a
  tilted design tracks through would be wrong until the inflector is re-exported with the nose.
- The exit opening through the nose is a straight sweep of the aperture. A beam that curves inside the nose's channel
  can graze its walls, so each design needs a check that the channel clears the beam.
- Each tilt changes the ground near the hand-off. Nothing here measures what that does to the beam: it needs a
  tracking study before a campaign varies it.

A full implementation along these lines, with its tests and a build-only dry run, was written before the hold. It is
kept, unreviewed, on branch `claude/si-gap0-tilt-held` (commit 07d1e9f, the two keys above included) and is not part
of this change.
