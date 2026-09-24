# Felt-Radius Transformer: Derivation, Coefficients & Sanity Check

**Solving the surface felt radius $R_{\text{felt}}$ from an Allen-style intensity attenuation model, focal depth $h$, and the Lambert $W$ function**

> **Project context.** This function was written for my ML project, where it is used as a **physics-informed feature transformer**: it converts raw catalog fields `(magnitude, depth, region)` into a physically meaningful quantity (an estimated felt radius in km) that downstream models consume. It is **not** intended to be a stand-alone, full-proof felt-area prediction model. Section 6 tests it on that basis: is it accurate enough to be a *useful feature*, not a *final predictor*.

---

## Contents

1. [Governing equation](#1-governing-equation)
2. [Reduction to standard form](#2-reduction-to-standard-form)
3. [Closed-form solution via Lambert W](#3-closed-form-solution-via-lambert-w)
4. [Extracting the surface radius](#4-extracting-the-surface-radius)
5. [Regional coefficients](#5-regional-coefficients)
6. [Sanity check against known earthquakes](#6-sanity-check-against-known-earthquakes)
7. [Is it good enough for ML use?](#7-is-it-good-enough-for-ml-use)
8. [References](#8-references)

---

## 1. Governing equation

The model is a macroseismic **intensity prediction equation (IPE)** in the family of Allen, Wald & Worden (2012) [1], where Modified Mercalli Intensity (MMI) decays with magnitude-dependent geometric spreading and anelastic damping:

$$
MMI = c_1 + c_2 M - c_3 \log_{10}(R_{\text{eff}}) - c_4 R_{\text{eff}}
$$

**Definitions**

| Symbol | Meaning |
| --- | --- |
| $M$ | Moment magnitude ($M_w$) |
| $h$ | Focal depth from the catalog (km) |
| $MMI$ | Target intensity threshold (default $2.5$, roughly the "felt" boundary, MMI II–III) |
| $R_0(M) = 10^{-0.281 + 0.251M}$ | Near-field finite-fault saturation term (km) |
| $R_{\text{eff}} = \sqrt{R_{\text{felt}}^2 + R_0(M)^2 + h^2}$ | Effective 3D distance (km) |
| $R_{\text{felt}}$ | Horizontal (surface) distance from the epicenter at which $MMI$ is reached (km) |
| $c_1, c_2, c_3, c_4$ | Regional constants (Section 5) |

The physical roles of the terms:

- $c_1 + c_2 M$: source-strength term (intensity grows roughly linearly with magnitude).
- $c_3 \log_{10} R_{\text{eff}}$: **geometric spreading** (energy spreading over a growing wavefront).
- $c_4 R_{\text{eff}}$: **anelastic attenuation** (energy absorbed by the crust; linear in distance).

The $R_0$ term stops the log term from diverging at short distance and acts as a crude stand-in for finite-fault (extended source) effects, which matter because a large rupture is not a point source.

---

## 2. Reduction to standard form

We want the distance at which the predicted intensity equals the target. Move the distance terms to the left:

$$
c_4 R_{\text{eff}} + c_3 \log_{10}(R_{\text{eff}}) = c_1 + c_2 M - MMI
$$

Divide by $c_4$:

$$
R_{\text{eff}} + \frac{c_3}{c_4}\log_{10}(R_{\text{eff}}) = \frac{c_1 + c_2 M - MMI}{c_4}
$$

Define

$$
x = R_{\text{eff}}, \qquad a = \frac{c_3}{c_4}, \qquad b = \frac{c_1 + c_2 M - MMI}{c_4}
$$

which gives the standard form

$$
\boxed{\,x + a\log_{10}(x) = b\,}
$$

This is a transcendental equation (a polynomial-plus-log mix), so it has no elementary closed form, but it does have one in terms of the Lambert $W$ function.

---

## 3. Closed-form solution via Lambert W

The Lambert $W$ function is defined as the inverse of $f(Y) = Ye^{Y}$, i.e. $W(Z)e^{W(Z)} = Z$ [2].

**A. Convert to natural logarithms.** Using $\log_{10}x = \ln x / \ln 10$:

$$
x + a\frac{\ln x}{\ln 10} = b
\;\;\Longrightarrow\;\;
\frac{x\ln 10}{a} + \ln x = \frac{b\ln 10}{a}
$$

**B. Exponentiate both sides.**

$$
x\, e^{\,x\ln 10 / a} = e^{\,b\ln 10 / a} = 10^{\,b/a}
$$

**C. Match the $Ye^{Y}=Z$ form.** Multiply both sides by $\ln 10 / a$ so the prefactor equals the exponent:

$$
\underbrace{\left[\frac{x\ln 10}{a}\right]}_{Y} e^{\underbrace{\left[\frac{x\ln 10}{a}\right]}_{Y}}
= \underbrace{\frac{\ln 10}{a}\,10^{\,b/a}}_{Z}
$$

**D. Apply $W_0$ and isolate $x$.**

$$
\boxed{\,R_{\text{eff}} = \frac{a}{\ln 10}\; W_0\!\left(\frac{\ln 10}{a}\, 10^{\,b/a}\right)\,}
$$

**Why the principal branch is safe.** Since $c_3, c_4 > 0$ we have $a > 0$, so the argument $Z > 0$. On $Z > 0$, $W_0$ is real, single-valued, and monotonic, so the intensity-vs-distance curve crosses the target exactly once and the solution is unique.

**Implementation note.** In code: `z = (ln10 / a) * 10.0 ** (b / a)` then `R_eff = (a / ln10) * lambertw(z).real`. If $b/a$ ever gets very large (roughly above 300), `10.0 ** (b/a)` overflows in float64. That is not reachable with $M \le 9.5$ and the coefficients below, but if coefficients are ever fitted or perturbed, evaluate $W$ in log-space instead.

---

## 4. Extracting the surface radius

From $R_{\text{eff}}^2 = R_{\text{felt}}^2 + R_0(M)^2 + h^2$:

$$
\boxed{\,R_{\text{felt}} = \sqrt{R_{\text{eff}}^2 - R_0(M)^2 - h^2}\,}
$$

**Edge case.** If $R_{\text{eff}}^2 \le R_0(M)^2 + h^2$, the function returns $R_{\text{felt}} = 0.0$ km. Physically, the target intensity is already reached at (or below) the focal/saturation distance, so it is not exceeded anywhere at the surface (a deep or small event whose surface shaking never reaches the threshold).

---

## 5. Regional coefficients

The code selects $(c_1, c_2, c_3, c_4)$ by tectonic region. Unknown region strings silently fall back to `"active"`.

| Region key | Tectonic setting / examples | $c_1$ | $c_2$ | $c_3$ | $c_4$ | $a = c_3/c_4$ |
| --- | --- | --- | --- | --- | --- | --- |
| `active` | Active crustal regions (Himalayas, Alpine belt, NZ) | 2.08 | 1.04 | 2.09 | 0.0019 | 1100.0 |
| `stable` | Generic stable continental crust (cratons, shields) | 1.85 | 1.15 | 1.80 | 0.0006 | 3000.0 |
| `stable_cena` | Central & Eastern North America | 1.139 | 1.093 | 1.150 | 0.00197 | 583.8 |
| `subduction` | Megathrust zones (Cascadia, Japan) | 2.45 | 0.98 | 2.15 | 0.0012 | 1791.7 |
| `strike_slip` | Transform faults (California, Anatolia) | 1.95 | 1.05 | 2.20 | 0.0025 | 880.0 |
| `volcanic` | Rift / volcanic zones (Iceland) | 2.10 | 1.02 | 2.35 | 0.0041 | 573.2 |

**Reading the table**

- $c_1, c_2$ set how strong shaking is at the source for a given magnitude.
- $c_3$ controls geometric spreading; $c_4$ controls anelastic loss per km. Higher $c_4$ generally means shaking dies out faster (the warm, fractured crust of active and volcanic regions), while a low $c_4$ lets shaking carry over continental distances (stable cratons).
- $c_4$ alone does **not** determine felt range. `stable_cena` has a $c_4$ similar to `active`, but its $c_1$–$c_3$ combination still produces a large felt radius (about 825 km for Mw 5.8 in the test below). Judge a region by the full coefficient set, not one column.

**Provenance caveat (please read).**

- The published Allen et al. (2012) IPEs are for **active crustal regions**, calibrated for $M_w$ 5.0–7.9, intensity ≥ II, and distances < 300 km [1]. The published model uses a more elaborate functional form than the four-parameter version here (the OpenQuake implementation, for example, lists coefficients `c0–c4, m1, m2, s1–s3` [3]).
- The `active` and `stable` rows come from the reference I was working from and are attributed there to Allen et al. (2012). I have **not** independently verified these exact numbers against the paper's tables.
- `stable_cena` follows the general approach of Atkinson & Wald (2007) [4] and the North American IPEs of Atkinson, Worden & Wald (2014) [5], but the exact numbers should be treated as **project-defined** until traced to a table.
- `subduction`, `strike_slip`, and `volcanic` are **project-defined heuristic settings**, not published coefficients.

In this document the model is therefore best described as *"an Allen-style attenuation model with project-tuned regional coefficients"*, not as a faithful reimplementation of the published equations.

---

## 6. Sanity check against known earthquakes

**Setup.** Seven well-known events, target $MMI = 2.5$, comparing the predicted $R_{\text{felt}}$ against approximate observed felt radii (ranges entered by hand; see caveats below).

| Event | $M_w$ | Depth (km) | Region | Predicted (km) | Observed (km) | vs. range midpoint | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2015 Gorkha (Nepal) | 7.8 | 15.0 | `active` | 833.2 | ~800–1000 | −7.4% | Within range |
| 2011 Mineral (Virginia, USA) | 5.8 | 6.0 | `stable_cena` | 824.5 | ~800–1000 | −8.4% | Within range |
| 2011 Tohoku (Japan) | 9.1 | 29.0 | `subduction` | 1629.9 | > 1200 | n/a (lower bound only) | Consistent |
| 2023 Kahramanmaraş (Turkey) | 7.8 | 10.0 | `strike_slip` | 604.9 | ~700–900 | −24.4% | **Under-predicts** |
| 1989 Loma Prieta (California) | 6.9 | 19.0 | `strike_slip` | 392.5 | ~350–450 | −1.9% | Within range |
| 2021 Fagradalsfjall (Iceland) | 5.7 | 5.0 | `volcanic` | 122.0 | ~100–150 | −2.4% | Within range |
| 2010 Christchurch (New Zealand) | 6.2 | 5.0 | `active` | 359.8 | ~250–350 | +19.9% | **Slightly over-predicts** |

**Summary statistics (the six events with two-sided ranges, vs. midpoint)**

| Metric | Value |
| --- | --- |
| Events within or consistent with observed range | 5 of 7 |
| Mean absolute percentage error | about 10.7% |
| Mean signed error (bias) | about −4.1% (slight under-prediction overall) |
| Worst cases | Kahramanmaraş (−24%), Christchurch (+20%) |

**What the test shows**

- **Order of magnitude and regional ordering are right.** Predictions span about 120 km (Iceland, volcanic) to about 1600 km (Tohoku), and the ranking follows the expected physics: stable crust and megathrusts carry far, warm or volcanic crust does not.
- **The stable-vs-active contrast is captured.** A Mw 5.8 event in `stable_cena` is predicted to be felt about as far as a Mw 7.8 event in `active` (both about 825 km), which matches the well-documented low attenuation of eastern North America.
- **Misses are moderate, not catastrophic.** The two misses are roughly 20–25% off, in opposite directions, so there is no single systematic failure.

**Caveats on the test itself**

- **Small sample.** Seven events cannot establish accuracy; they can only expose gross failures.
- **Ranges are loose.** The "observed" values are approximate ranges entered by hand. "Felt radius" depends on the intensity threshold used, population density, and how many people file reports (for example on USGS "Did You Feel It?"). They should be traced to specific event pages before being quoted as ground truth.
- **Possible in-sample tuning.** If the `subduction`, `strike_slip`, or `volcanic` coefficients were adjusted while looking at these same events, the agreement is optimistic. A held-out set is needed.
- **Extrapolation.** Felt radii of 800+ km at MMI 2.5 lie far beyond the < 300 km distance range and ≥ II intensity range that Allen et al. calibrated on [1], and Tohoku (Mw 9.1) is beyond the Mw 7.9 upper limit. Good agreement there is encouraging but is extrapolation, not validation.

---

## 7. Is it good enough for ML use?

**Short answer: yes as a feature transformer, no as a standalone predictor.**

**Why it is good enough as a feature**

1. **Right physics, right shape.** The output is monotonic in magnitude, decays sensibly with depth, and separates tectonic regimes, so it injects domain structure that a small dataset would struggle to learn from raw `(M, depth)` alone.
2. **Errors are small relative to the natural spread.** An error of about 10% on average (up to about 25%) is comparable to the inherent variability in observed felt areas. A downstream model can absorb a smooth, bounded bias like this.
3. **Errors are not chaotic.** Deviations look region- and event-specific rather than random, which is exactly what a learned residual (or a per-region correction) can fix.
4. **Deterministic and cheap.** A closed-form Lambert $W$ evaluation is fast and vectorizable, with no iterative solver failures.

**Why it is not good enough on its own**

1. It misses individual events by 20–25%, which is too much for a final "answer" but acceptable for an input feature.
2. Site effects, rupture directivity, fault geometry, and local geology are absent.
3. Coefficients for three of the six regions are heuristic, and the validation set is small.

**Recommendations before relying on it in the pipeline**

- **Validate on a larger, independently sourced set** (for example USGS DYFI and ShakeMap event pages), with a train/held-out split by event so coefficients are not tuned on test events.
- **Feed the model both the raw inputs and the transformer output** (magnitude, depth, region, and $R_{\text{felt}}$), so the downstream learner can correct where the physics is off rather than being forced to trust it.
- **Consider a log transform** (`log1p(R_felt)`), since the radius spans two orders of magnitude.
- **Handle the zero case deliberately.** Returning `0.0` for `rad_sq <= 0` creates a hard discontinuity. Consider adding a binary "felt at surface" flag alongside the radius.
- **Do not silently default the region.** An unknown region currently becomes `active`. For ML use, raise an error or emit a "region unknown" indicator instead.
- **Expose $MMI_{\text{target}}$ as a parameter and check sensitivity.** Because of the exponential-like decay, small changes in the threshold shift the radius substantially; computing several thresholds (for example 2.5, 3.5, 4.5) can make a richer feature set.
- **Quantify uncertainty.** IPEs have intrinsic scatter; if the downstream task allows, propagate a rough spread rather than a single number.

---

## 8. References

[1] Allen, T. I., Wald, D. J., & Worden, C. B. (2012). Intensity attenuation for active crustal regions. *Journal of Seismology*, 16, 409–433. https://doi.org/10.1007/s10950-012-9278-7

[2] Corless, R. M., Gonnet, G. H., Hare, D. E. G., Jeffrey, D. J., & Knuth, D. E. (1996). On the Lambert W function. *Advances in Computational Mathematics*, 5, 329–359.

[3] OpenQuake Engine documentation, `openquake.hazardlib.gsim.allen_2012_ipe` (implementation of the Allen, Wald & Worden 2012 GSIM). https://docs.openquake.org/

[4] Atkinson, G. M., & Wald, D. J. (2007). "Did You Feel It?" intensity data: a surprisingly good measure of earthquake ground motion. *Seismological Research Letters*, 78, 362–368. https://doi.org/10.1785/gssrl.78.3.362

[5] Atkinson, G. M., Worden, C. B., & Wald, D. J. (2014). Intensity prediction equations for North America. *Bulletin of the Seismological Society of America*, 104, 3084–3093.

**Data sources for validation (to be cited per event once the observed radii are traced):** USGS Earthquake Hazards Program event pages and "Did You Feel It?" community intensity maps.