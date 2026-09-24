# Felt-Radius Feature: Solving Surface Distance $R_{\text{felt}}$ with an Intensity Attenuation Model and the Lambert W Function

## Purpose and Scope

This note documents the `compute_felt_radius_v1` feature extractor used in my earthquake-forecasting ML project. Raw catalog entries only give a point (epicenter), a magnitude and a focal depth. That is not enough to say *which locations were actually affected*. This function converts each event into a physically motivated **felt radius**: the horizontal distance from the epicenter at which shaking is predicted to fall to a chosen Modified Mercalli Intensity (MMI) threshold.

The radius is then used as an engineered feature (a spatial extent for each event) so that the downstream transformer receives a "reliable enough" footprint of affected areas instead of a bare point and a magnitude.

**What this is not:** a hazard model or a ground-truth shaking map. It is a compact, deterministic, physics-informed preprocessing step. Section 7 shows how far it can be trusted.

**Sources and credit**

* **Model structure.** The two-term distance decay (geometric spreading plus a linear damping term, in intensity units) is a simplified form of published macroseismic intensity prediction equations (IPEs), notably **Allen, T. I., Wald, D. J., & Worden, C. B. (2012), "Intensity attenuation for active crustal regions", *Journal of Seismology*, 16, 409–433.** The published equations are more elaborate (Section 5.3), so this function is an adaptation, not a re-implementation.
* **Coefficient data.** Where a published IPE exists, the regional constants in Section 5 are **fitted to that published equation**; they are not copied from a table, because no published paper uses exactly this four-coefficient form. Provenance is stated per region.
    * **Atkinson, G. M., & Wald, D. J. (2007), ""Did You Feel It?" Intensity Data: A Surprisingly Good Measure of Earthquake Ground Motion", *Seismological Research Letters*, 78(3), 362–368**: California and Central & Eastern US (CEUS).
    * **Allen et al. (2012)**: active crustal regions (coefficients as implemented in the GEM OpenQuake hazardlib).
* **Lambert W.** Corless, R. M., et al. (1996), "On the Lambert W function", *Advances in Computational Mathematics*, 5, 329–359. Evaluated with `scipy.special.lambertw`.
* **Reference felt extents** in Section 7 are approximate, hand-compiled ranges of the kind reported on USGS event pages and "Did You Feel It?" (DYFI) maps.

---

## 1. Governing Equation

$$MMI = c_1 + c_2 M - c_3 \log_{10}(R_{\text{eff}}) - c_4 R_{\text{eff}}$$

The first distance term ($c_3\log_{10}R_{\text{eff}}$) captures geometric spreading of the wavefront. The second ($c_4 R_{\text{eff}}$) captures losses that grow linearly with distance (anelastic damping and scattering).

**Definitions**

* $M$: earthquake magnitude.
* $h$: focal depth from the catalog (km).
* $MMI$: target intensity threshold (e.g. $2.5$ for "felt by people").
* $R_0(M) = 10^{-0.281 + 0.251M}$: near-source saturation term (km), which grows with magnitude and accounts for the finite size of the rupture. This is a project-specific choice; Allen et al. use a different magnitude-dependent term.
* $R_{\text{eff}} = \sqrt{R_{\text{felt}}^2 + R_0(M)^2 + h^2}$: effective 3D distance (km).
* $R_{\text{felt}}$: horizontal surface distance from the epicenter (km). This is the quantity we want.
* $c_1, c_2, c_3, c_4$: regional constants (Section 5).

---

## 2. Reduction to Standard Form

Move the distance terms to the left and the rest to the right:

$$c_4 R_{\text{eff}} + c_3 \log_{10}(R_{\text{eff}}) = c_1 + c_2 M - MMI$$

Divide by $c_4$ and substitute

$$x = R_{\text{eff}}, \qquad a = \frac{c_3}{c_4}, \qquad b = \frac{c_1 + c_2 M - MMI}{c_4}$$

to get

$$x + a\log_{10}(x) = b$$

This mixes a linear and a logarithmic term in $x$, so it has no solution in elementary functions. It is, however, exactly the shape the Lambert W function solves.

---

## 3. Solving for $R_{\text{eff}}$ with Lambert W

The Lambert W function inverts $Y e^{Y} = Z$, i.e. $Y = W(Z)$.

**A. Switch to natural log and rescale.** Using $\log_{10}x = \ln x / \ln 10$ and multiplying through by $\ln(10)/a$:

$$\frac{x\ln 10}{a} + \ln x = \frac{b\ln 10}{a}$$

**B. Exponentiate both sides.** Since $e^{\ln x} = x$ and $e^{(b\ln 10)/a} = 10^{b/a}$:

$$x \, e^{\frac{x\ln 10}{a}} = 10^{b/a}$$

**C. Match the $Ye^Y = Z$ form.** Multiply both sides by $\ln(10)/a$ so the coefficient in front of $x$ matches the exponent:

$$\underbrace{\frac{x\ln 10}{a}}_{Y}\; e^{\overbrace{\frac{x\ln 10}{a}}^{Y}} = \underbrace{\frac{\ln 10}{a}\,10^{b/a}}_{Z}$$

**D. Apply $W_0$ and solve for $x$:**

$$R_{\text{eff}} = \frac{a}{\ln 10}\; W_0\!\left(\frac{\ln 10}{a}\,10^{b/a}\right)$$

Because $a > 0$, the argument $Z$ is positive, so there is exactly one real solution and it lies on the principal branch $W_0$. No iterative root-finding is needed, and the solution is vectorizable across a whole catalog.

**Limiting case.** When $c_4 \to 0$ the linear term vanishes, $a \to \infty$, and the solution tends to the pure power law $R_{\text{eff}} \to 10^{(c_1 + c_2 M - MMI)/c_3}$. The formula stays numerically stable in that limit, which matters for the `active` setting (Section 5).

---

## 4. Extract the Surface Felt Radius

Invert the definition of $R_{\text{eff}}$:

$$R_{\text{felt}} = \sqrt{R_{\text{eff}}^2 - R_0(M)^2 - h^2}$$

**Edge case:** if $R_{\text{eff}}^2 \le R_0(M)^2 + h^2$, the predicted intensity is below the target even directly above the source, because depth and saturation already account for the whole effective distance. No surface distance satisfies the threshold, so the function returns $R_{\text{felt}} = 0$. This happens for small, deep events.

**Worked example (2015 Gorkha, $M_w$ 7.8, $h = 15$ km, MMI = 2.5).** For the provisional constants used before literature fitting ($c = 2.08,\ 1.04,\ 2.09,\ 0.0019$):

| Quantity | Value |
| --- | --- |
| $a = c_3/c_4$ | $2.09 / 0.0019 = 1100$ |
| $b$ | $(2.08 + 1.04\cdot 7.8 - 2.5)/0.0019 \approx 4048.4$ |
| $Z = \frac{\ln 10}{a}10^{b/a}$ | $\approx 10.03$ |
| $R_{\text{eff}}$ | $\approx 834.7$ km |
| $R_0(M)$ | $\approx 47.5$ km |
| $R_{\text{felt}}$ | $\sqrt{834.7^2 - 47.5^2 - 15^2} \approx 833.2$ km |

Substituting $R_{\text{eff}}$ back into the governing equation reproduces the target intensity to numerical precision, which confirms the algebra. The same algebra applies unchanged to the literature-fitted constants in Section 5.

---

## 5. Regional Coefficient Configuration

### 5.1 Configuration table

These parameters account for regional crustal age, thermal structure, and tectonic style. Rows marked **Fitted** are initialised from a published IPE (Section 5.2). Rows marked **Provisional** could not be tied to a published MMI equation and are carried over unchanged from the first version of the function.

| Region Key | Tectonic Regime / Example Region | Baseline $c_1$ | Magnitude Scaling $c_2$ | Spreading $c_3$ | Absorption $c_4$ | Source / Status |
| --- | --- | --- | --- | --- | --- | --- |
| `"active"` | Active crustal regions (Himalaya, Alpine Belt, NZ) | $1.819$ | $1.428$ | $3.061$ | $0.00001$ | **Fitted** to Allen et al. (2012), hypocentral-distance version |
| `"stable"` | Standard cratons / shield regions | $1.85$ | $1.15$ | $1.80$ | $0.00060$ | **Provisional** (no source found) |
| `"stable_cena"` | Central & Eastern North America | $1.139$ | $1.093$ | $1.150$ | $0.00197$ | **Fitted** to Atkinson & Wald (2007), CEUS |
| `"subduction"` | Cascadia, Japan megathrust, Chile | $2.45$ | $0.98$ | $2.15$ | $0.00120$ | **Provisional** (no source found) |
| `"strike_slip"` | California crustal faults (key name kept from v1) | $1.885$ | $1.066$ | $2.160$ | $0.00217$ | **Fitted** to Atkinson & Wald (2007), California |
| `"volcanic"` | Iceland, East African Rift | $2.10$ | $1.02$ | $2.35$ | $0.00410$ | **Provisional** (no source found) |

**What the fitted rows do and do not mean**

* $c_3$ and $c_4$ are **curve-fit parameters** of a two-term surrogate, not directly measured geometric-spreading or quality-factor values. Different published equations trade the two terms off against each other, so comparing $c_4$ across rows as a physical "damping" ranking is not reliable.
* For `active`, $c_4$ sits at the lower bound of the fit ($10^{-5}$), meaning the Allen et al. equation decays as a pure power law over the fitted range and supports no linear damping term. Numerically this makes $a$ very large (about $3\times10^5$); Section 3 explains why the solution remains stable.
* For the published models themselves, Atkinson & Wald report that MMI is about 1 unit higher in the CEUS than in California at near-fault distances (< 30 km), and typically 1.5 to 2 units higher at distances beyond 100–200 km. Their regression coefficients do not show that contrast cleanly because of the extra shape terms in their equation.
* The `stable`, `subduction` and `volcanic` rows reflect qualitative expectations (cold rigid crust transmits shaking far, hot fluid-rich crust damps it quickly). They are placeholders, not results.

### 5.2 How the fitted values were obtained

Each published IPE was evaluated on a grid of magnitude and distance, and the four constants of the surrogate

$$MMI = c_1 + c_2 M - c_3\log_{10}R - c_4 R$$

were found by bounded linear least squares ($c_3 \ge 0$, $c_4 \ge 10^{-5}$, so that $a>0$ and the Lambert W solution exists).

* **Grid:** $M = 4.5$ to $8.5$ in steps of $0.25$; distance $10$ to $2500$ km, log-spaced (300 points).
* **Fit window:** only grid points where the published MMI lies between $1.5$ and $5.5$, since the feature targets the felt boundary rather than damaging intensities.
* **Distance mapping:** $R_{\text{eff}}$ is used as the published model's distance argument (hypocentral distance for Allen et al.; fault/hypocentral distance $D$ for Atkinson & Wald, which apply their own effective depth internally).
* **Fit quality (RMS residual against the published curve inside the window):** Allen et al. $0.02$ MMI units; Atkinson & Wald California $0.11$; Atkinson & Wald CEUS $0.15$.

Because the surrogate has fewer shape parameters than the published equations, it can differ from them by several percent in predicted radius (Section 7 shows this explicitly).

### 5.3 Published equations used as the reference data

**Atkinson & Wald (2007), Eq. 1 and Table 1** (values read directly from the paper):

$$MMI = c_1 + c_2 (M-6) + c_3 (M-6)^2 + c_4 \log_{10} R + c_5 R + c_6 B + c_7 M \log_{10} R,\quad R=\sqrt{D^2+h^2}$$

with $B = 0$ for $R \le R_t$ and $B = \log_{10}(R/R_t)$ for $R > R_t$.

| Coefficient | California | Central & Eastern U.S. |
| --- | --- | --- |
| $c_1$ | 12.27 | 11.72 |
| $c_2$ | 2.270 | 2.36 |
| $c_3$ | 0.1304 | 0.1155 |
| $c_4$ | −1.30 | −0.44 |
| $c_5$ | −0.000707 | −0.002044 |
| $c_6$ | 1.95 | 2.31 |
| $c_7$ | −0.577 | −0.479 |
| $h$ (km) | 14.0 | 17.0 |
| $R_t$ (km) | 30.0 | 80.0 |

Their data are DYFI reports supplemented with historical MMI observations, and the stated residual standard deviation is about 0.4 MMI units. I checked my implementation against statements in the paper: a CEUS $M$ 4 event at 300 km and a California $M$ 6 event at 300 km give similar MMI (2.3 vs 2.2), and CEUS near-fault MMI is about 1 unit higher than California's.

**Allen, Wald & Worden (2012), active crustal regions, hypocentral-distance version.** Values below are as implemented in the GEM OpenQuake hazardlib (I did not read the paper's own tables):

$$MMI = c_0 + c_1 M + c_2 \ln\sqrt{R^2 + r_m^2} + c_4 \ln(R/50)\ [R>50\text{ km}],\qquad r_m = m_1 + m_2 e^{M-5}$$

| $c_0$ | $c_1$ | $c_2$ | $c_4$ | $m_1$ | $m_2$ |
| --- | --- | --- | --- | --- | --- |
| 2.085 | 1.428 | −1.402 | 0.078 | −0.209 | 2.042 |

### 5.4 Gaps and candidate sources

* **No published MMI equation found** for generic stable crust, subduction interface events, or Icelandic/rift volcanic settings. Searches for these returned peak-ground-motion equations (PGA/PSA) rather than intensity equations. A two-step route (ground-motion equation plus a ground-motion-to-intensity conversion such as Worden et al., 2012, *BSSA* 102(1), 204–221) is possible but was not attempted.
* **Nepal Himalaya.** Prajapati, S. K., Dadhich, H. K., & Chopra, S. (2017), "Isoseismal map of the 2015 Nepal earthquake and its relationships with ground-motion parameters, distance and magnitude", *J. Asian Earth Sci.*, 133, 24–37 (doi:10.1016/j.jseaes.2016.07.013), derives an intensity attenuation relation for the Nepal Himalaya. The coefficients were behind a paywall and were not read here. It is the most relevant source to check for the `active` row.

---

## 6. Implementation

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import lambertw

def compute_felt_radius_v1(
	magnitude: float,
	depth_km: float,
	target_mmi: float = 2.5,
	region: str = "active"
) -> float:
	"""
	Computes surface felt radius R_felt (km): the epicentral distance at which
	ground shaking is predicted to reach `target_mmi`, using a regional
	tectonic attenuation model.
	"""
	# 1. Master Configuration Dictionaries
	#    "FITTED"      = least-squares fit to a published IPE (see Section 5.2)
	#    "PROVISIONAL" = no published MMI source found; placeholder values
	coefficient_configurations = {
		"active":      (1.819, 1.428, 3.061, 0.00001),   # FITTED: Allen, Wald & Worden (2012), Rhypo version
		"stable":      (1.85, 1.15, 1.80, 0.0006),       # PROVISIONAL: stable continental crust
		"stable_cena": (1.139, 1.093, 1.150, 0.00197),   # FITTED: Atkinson & Wald (2007), CEUS
		"subduction":  (2.45, 0.98, 2.15, 0.0012),       # PROVISIONAL: Cascadia, Japan megathrust
		"strike_slip": (1.885, 1.066, 2.160, 0.00217),   # FITTED: Atkinson & Wald (2007), California
		"volcanic":    (2.10, 1.02, 2.35, 0.0041)        # PROVISIONAL: Iceland / Rifts
	}

	# 2. Extract settings with safety defaults
	c1, c2, c3, c4 = coefficient_configurations.get(region, coefficient_configurations["active"])

	# 3. Finite-fault saturation term
	R0 = 10 ** (-0.281 + 0.251 * magnitude)

	# 4. Lambert W setup for 3D effective distance
	a = c3 / c4
	b = (c1 + c2 * magnitude - target_mmi) / c4

	ln10 = np.log(10)
	z = (ln10 / a) * (10.0 ** (b / a))
	R_eff = (a / ln10) * lambertw(z).real

	# 5. Pythagorean depth / finite-fault correction to surface distance
	rad_sq = R_eff**2 - R0**2 - (depth_km ** 2)

	if rad_sq <= 0:
		return 0.0

	return float(np.sqrt(rad_sq))
```

Only the coefficient dictionary changed relative to the first version; the algorithm is identical.

---

## 7. Sanity Check Against Known Events

The function was run on seven well-known earthquakes spanning the regional settings, and the predictions were compared with approximate reported felt extents.

> **How to read this section:** this is a **sanity check, not a proof or a calibration.** The reference values are approximate ranges compiled by hand and are not tied to a specific MMI contour, the sample is small, and no coefficients were fitted to these events. The aim is only to see whether outputs are in the right physical ballpark and respond sensibly to magnitude, depth and tectonic setting.

The validation script is the same as before (catalog of seven events, `TARGET_MMI = 2.5`). Output with the configuration in Section 6:

```text
=== EARTHQUAKE FELT RADIUS VALIDATION (Target MMI = 2.5) ===
               Earthquake Event  Mag (Mw)  Depth (km)      Region  Predicted Radius (km) Approx. Reported Felt Radius (km)
            2015 Gorkha (Nepal)       7.8        15.0      active                 2558.0                       ~800 - 1000
   2011 Mineral (Virginia, USA)       5.8         6.0 stable_cena                  824.5                       ~800 - 1000
            2011 Tohoku (Japan)       9.1        29.0  subduction                 1629.9                            > 1200
    2023 Kahramanmaraş (Turkey)       7.8        10.0 strike_slip                  708.4                        ~700 - 900
  1989 Loma Prieta (California)       6.9        19.0 strike_slip                  456.5                        ~350 - 450
  2021 Fagradalsfjall (Iceland)       5.7         5.0    volcanic                  122.0                       ~100 - 150
2010 Christchurch (New Zealand)       6.2         5.0      active                  465.6                       ~250 - 350
```

### Comparison with the provisional constants and the published equations

The table below adds two columns: the original provisional constants ("v1"), and the felt radius obtained by solving the **published IPE itself** for MMI = 2.5 (numerically, converting hypocentral distance to surface distance with the catalog depth). The last column shows how faithfully the four-coefficient surrogate reproduces the published model. Tohoku and Fagradalsfjall have no published IPE in the set, so they still use provisional constants.

| Event | Reported (km) | v1 provisional (km) | Literature-fitted (km) | Published IPE solved directly (km) |
| --- | --- | --- | --- | --- |
| Gorkha (`active`) | ~800 – 1000 | 833.2 | 2558.0 | 2614.1 |
| Mineral (`stable_cena`) | ~800 – 1000 | 1433.9 | 824.5 | 833.3 |
| Tohoku (`subduction`) | > 1200 | 1629.9 | 1629.9 (provisional) | n/a |
| Kahramanmaraş (`strike_slip`) | ~700 – 900 | 604.9 | 708.4 | 683.1 |
| Loma Prieta (`strike_slip`) | ~350 – 450 | 392.5 | 456.5 | 419.9 |
| Fagradalsfjall (`volcanic`) | ~100 – 150 | 122.0 | 122.0 (provisional) | n/a |
| Christchurch (`active`) | ~250 – 350 | 359.8 | 465.6 | 465.4 |

### Reading the results

* **The surrogate tracks the published models.** Within a few percent for most events (the largest gap is Loma Prieta, about 9 % above the direct solution). Differences from the reported ranges therefore mostly come from the published equations and the loose reference values, not from the Lambert W step.
* **Better with literature data:** Mineral moves from about 43 % above the reported range to inside it, and Kahramanmaraş moves from about 14 % below to the lower edge of the range. Loma Prieta is now marginally above the range (about 1 %).
* **Worse for active crust:** Christchurch is about 33 % above its range, and **Gorkha is roughly 2.5 times the reported range**. The provisional `active` constants happened to match Gorkha; the published Allen et al. equation does not at this threshold.
* **Threshold sensitivity for `active`.** For Gorkha the literature-fitted radius is 1766 km at MMI 3.0, 1217 km at MMI 3.5 and 837 km at MMI 4.0. So the published equation reproduces the reported range at about MMI 4, not 2.5. Two plausible readings: the reported extents correspond to a higher intensity than the "barely felt" threshold, or MMI 2.5 at 2000+ km lies outside the intensity and distance range these equations were fitted on. I could not confirm the data range of Allen et al. (2012), so this is worth checking.

**Conclusion.** With literature-derived constants, five of seven events fall inside or at the edge of the reported ranges (Tohoku and Fagradalsfjall rely on provisional values). The two outliers, Gorkha and Christchurch, are both in the `active` setting, which is the setting most relevant to this project. The predicted radii still scale correctly with magnitude, depth and tectonic regime, which is what an engineered input feature needs, and the transformer can absorb a consistent multiplicative bias. Two choices remain open for `active` before this is treated as settled:

1. keep the provisional constants for now and label them as such, or
2. keep the literature-fitted values but raise `target_mmi` (for example to 3.5–4.0) for active-crust events, or replace them with a Nepal-specific relation (Section 5.4).

---

## 8. Known Limitations

* **Isotropic footprint.** The result is a single circular radius. Rupture directivity, fault geometry and elongated felt areas for large events are not modeled.
* **Point-source treatment.** Finite-fault effects are approximated only through the saturation term $R_0(M)$. Because the published equations already contain their own finite-source terms, this adds a small extra correction (at most a few tens of km against radii of hundreds).
* **Surrogate, not the published equation.** The four-coefficient form cannot represent magnitude-squared terms, magnitude–distance interaction, or distance-dependent slope changes in the original equations. Fit errors are reported in Section 5.2.
* **Extrapolation to low intensities.** A "felt" threshold of MMI 2.5 at large distances may lie outside the range the source equations were fitted on.
* **No site effects.** Basin amplification, soft-soil response and topography are ignored, so local intensity can differ substantially from the radius-based estimate.
* **Catalog uncertainty propagates.** Errors in catalog magnitude and especially focal depth feed straight into $R_{\text{felt}}$, and depth error matters most for small, shallow events.
* **Single threshold.** One target MMI defines "felt"; other thresholds (e.g. damaging intensity, MMI ≥ 6) need to be chosen explicitly.
* **Incomplete literature coverage.** Three of six regional settings still use provisional constants (Section 5.4).