# Spiral-inflector rotation, gap-0 position and gap-0 orientation

Three `[geometry]` quantities that the corner rule used to tie to the dee opening, each a number a MOBO campaign can
vary (`kind = "geometry"`), each 0 by default, which builds exactly what the code built before they existed:

| key | unit | meaning (0 = today) | modes |
|---|---|---|---|
| `si_rotation_offset_deg` | deg, + ccw | the spiral-inflector assembly turned off the corner rule's angle | `si_rotation = "corner"` |
| `gap0_leave_mm` | mm, + outward | where gap 0 leaves the exit plate, along the plate from its home point | `gap_lines = "connected"` |
| `gap0_tilt_deg` | deg, >= 0 | gap 0's first segment turned off the plate about that point; a grounded nose fills the wedge | `gap_lines = "connected"`, `ground = "star"` |

**Readings chosen.** Daniel's sentence ("separation of SI housing rotation and gap0 position and orientation") allows two
readings of two of the quantities; this note takes the first of each:

- *Rotation* is an **offset from the corner rule**, so 0 is today's build at every dee opening. (Alternative: an
  absolute angle free of the opening; that is `si_rotation = "fixed"` with `[inflector] rotation_deg`, which stays as it
  is, gap 0 included: see "The fixed mode" below.)
- *Position* is the **point on the plate** where gap 0 leaves it, as an arclength along the turned plate line.
  (Alternative: gap 0's radial ray turned free of the dee opening, an asymmetric dee 0; not done.)

The tilt's pivot is that leave point, so the three are independent: the offset carries the leave point with the
housing, the leave point does not move when the segment tilts.

## Definitions

Machine frame, mm, seen from +z; angles + counter-clockwise. `R(a)` turns a vector by `a` about the machine axis.
From the export's `exit_plane_spec.json`: the plate line (the plate's outer face shifted by half the local gap-law
width along its downstream normal `n`, unrotated) with its outward unit direction `d0`, the plate's outer corner `C0`
and the export's gap-0 azimuth `a_x`. Gap 0's ray is at azimuth `t0` (its base angle, set by the dee opening), unit
vector `e`.

1. **Corner-rule angle** `rot_c = t0 - a_x` (what `si_rotation = "corner"` turns the inflector by today: the export's
   placement of the plate corner relative to gap 0's ray, carried to another opening).
2. **Assembly rotation** `rot = rot_c + si_rotation_offset_deg`. The housing outline and STEP, the exit aperture, the
   emitting plane (beam channel), the inflector's E-field and the hand-off bunch all turn by `rot`: the build records
   it in the model (`params['si_rotation_deg']`) and the tracker adopts it (`simulate3d.follow_model_si_rotation`,
   unchanged). Turned plate direction `d = R(rot) d0`.
3. **Home point** `H`: where the plate line turned by `rot_c` meets gap 0's ray (today's leave point, `x_c`), carried
   with the assembly: `H = R(si_rotation_offset_deg) x_c`. It sits at the same place on the plate for every offset.
4. **Leave point** `L = H + gap0_leave_mm * d`: on the turned plate line, + towards (and past) the plate's outer
   corner, - towards the exit aperture.
5. **First segment** through `L` with direction `d1 = R(-gap0_tilt_deg) d`. Gap 0 is the polyline
   `L - x_L d1 -> L -> P1 -> P2 -> its ray` (`x_L = L.e`; the first point lies well inside the housing), with `P1`,
   `P2` from its `[[geometry.gap_line]]` item as before. `P0 = L` in gap 0's own frame is `(x_L, y_L)` with
   `y_L = e x L` (0 at defaults), so `y1_max` is now the largest `|y1|` for which both `+|y1|` and `-|y1|` keep the
   bends within the limits; at `y_L = 0` this is the old even rule, to the bit.

```
                           dee 0                         P1 ----- P2 ------ ray (t0) --->
                                                    ___/
            tilted first segment (d1)          ___/
        ______________________________________L .  .  .  .  . C (plate corner)    gap 0's line
   ____/   ^ nose (ground, full height):  /////|            beyond L: L -> P1 -> P2
  /        | wedge between the plate      /////|
           | face and gap 0's dummy edge  /////|
  =========[ exit aperture ]===========================C====   plate face (housing), direction d --->
                     |
     H = the home point on the plate; L = H + gap0_leave_mm d; tilt = the angle between d1 and d (clockwise +)
```

**Sign of the tilt.** `gap0_tilt_deg > 0` turns the first segment clockwise, so its inner end swings off the plate into
dee 0 and gap 0's normal (the accelerating field) turns from `n` towards `+d`, radially outward along the plate. A
negative tilt would run the segment's inner end into the housing and is refused.

**The nose.** With a tilt the ground side of gap 0's first segment leaves the plate face: the wedge between the plate
face and gap 0's dummy edge (dee 0's gap-0 edge offset by the local gap-law width) is filled, full height, with ground
metal reaching 2 mm into the housing, inward of `L` wherever the housing lies within that wedge's depth plus 5 mm. It is
fused into the star ground before the inflector's exit opening is cut, so the opening (the exit aperture swept along
the emitting plane's normal) is the beam's channel through it. Its depth at arclength `u` inward of `L` is about
`u tan(gap0_tilt_deg)`: the beam leaves the aperture, drifts through that much grounded channel and then meets the
tilted gap. `gap0_face_mm` (the existing face beyond the plate corner) can be combined with it.

## Valid ranges

| key | config check | build check | campaign box (suggested) |
|---|---|---|---|
| `si_rotation_offset_deg` | any float; `si_rotation = "corner"` | gap 0's connected line must exist (below) | about -6 .. +6 |
| `gap0_leave_mm` | any float; connected lines | `L` outward of the exit aperture, `x_L < r1` | from just outward of the aperture's outer end |
| `gap0_tilt_deg` | `0 <= tilt < 45`; connected; star | bend at `L` <= 75 deg; the nose keeps a gap off every dee | 0 .. 15 |

The build checks raise `ConfigError`: `pcr build` stops with the message, and in a campaign `pcr evaluate` records the
evaluation as failed with that message (no crash). They run while any of the three keys is non-zero, so a design at
the defaults builds exactly as before in every mode. The config checks run at load and at a campaign's preflight
(`check_overrides`, every corner of the box).

## What each does to the first turn

- **Offset.** The beam leaves the housing at the same place relative to the plate and meets gap 0's first segment
  there, so the first gap crossing and its energy gain do not change; the hand-off turns by the offset about the axis
  while gaps 1, 2, 3 stay put, so the first turn reaches them earlier (+) or later (-): about the offset times the
  harmonic in RF phase, plus the change of path length. Beyond `L` gap 0's line runs to `P1`, `P2` on its own ray, so
  its outer part bends by a different angle at `L`.
- **Leave point.** The first crossing stays where it is while `L` stays outward of the aperture; moving `L` changes
  the length of gap 0 on the plate and so where later turns cross gap 0 (beyond `L`) and the angle they cross it at.
  It also moves dee 0's gap-0 edge from `L` out to `P1`.
- **Tilt.** The first crossing's field turns by the tilt: an outward kick of about `sin(tilt)` of the gap's momentum
  gain, so the beam leaves gap 0 at a larger angle to the tangent (an earlier estimate: ~0.38 deg of exit angle per
  degree of tilt), after a drift through the nose. Dee 0's tip narrows by about `u_tip tan(tilt)` (`u_tip` its
  distance inward of `L`); a negative `gap0_leave_mm` shortens that lever and thins the nose where the beam crosses it.

## Refused, and why

| when | message (start) |
|---|---|
| offset != 0 with `si_rotation = "fixed"` | `geometry.si_rotation_offset_deg = ... turns the spiral inflector off the corner rule; with si_rotation = 'fixed' the turn is [inflector] rotation_deg alone ...` |
| leave / tilt != 0 with `gap_lines = "seed"` | `geometry.gap0_leave_mm = ... place gap 0's connected line; the seed's gap 0 ...` |
| tilt < 0 or >= 45 | `geometry.gap0_tilt_deg = ...: 0 <= tilt < 45 deg ...` |
| tilt > 0 with `ground = "dummies"` | `geometry.gap0_tilt_deg = ... needs ground = 'star': the grounded nose ...` |
| `L` inside or inward of the exit aperture | `gap 0 would leave the exit plate ... mm inward of the exit aperture's outer end ...` |
| `L` at or beyond `r1` along gap 0's ray | `a connected gap line needs P0 < r1 < r2 ...` (as before) |
| no `y1` keeps the bend at `P1` within `bend_max_deg` (`L` far off the ray) | `gap 0: from its leave point ... no P1 at r1 = ... keeps the bend at P1 ...` |
| bend at `L` > 75 deg | `gap 0 bends by ... deg where it leaves the exit plate ...` |
| no housing within the wedge's depth + 5 mm of the tilted segment | `geometry.gap0_tilt_deg = ...: the housing is nowhere within ...` |
| the nose closer to a dee than the star ground's gap floor | `the gap-0 nose comes within less than the gap floor ...` |

Not built: a tilted gap 0 with the dummy-wedge ground (`ground = "dummies"`), and a tilt on the seed's gap lines.

## The fixed mode

`si_rotation = "fixed"` is unchanged: the assembly turns by `[inflector] rotation_deg` and gap 0's home point is where
that turned plate meets gap 0's ray. As the plate crosses the ray at 10-16 deg in the exports seen so far, that point
runs along the plate several times faster than the turn (tens of mm for a few degrees), and the build fails once it
passes `r1`. Campaigns that want the assembly turned with gap 0 held on the plate use `si_rotation = "corner"` with
`si_rotation_offset_deg`; `gap0_leave_mm` and `gap0_tilt_deg` work in both modes, from their mode's home point.

## Outputs

`<prefix>_build.json` keeps everything it had (its `geometry` table now lists the three keys) and adds, in gap 0's
`gap_lines` record, `leave` (`H`, `L`, the offset, the leave arclength, the tilt, `L` along and across gap 0's ray, the
plate corner's and the exit aperture's arclengths from `L`) and, with a tilt, a top-level `gap0_nose` (its plan-view
area, radii and outline). The model's `params['si_rotation_deg']` is `rot` as before. The geometry key of a MOBO
campaign leaves each of the three keys out while it is 0, so existing campaign stores keep their keys.
