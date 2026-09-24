# Mathematical Derivation of the Advanced Felt Radius Model

> **Purpose note:** `compute_felt_radius_advanced` is not intended as a general-purpose seismological formula. It's a **feature-engineering transform** used to turn three raw catalog fields (magnitude, depth, region/soil label) into a single continuous, physically-motivated number, the predicted MMI-threshold felt radius, that can be fed into an earthquake ML model as a richer input feature than the raw fields alone. The derivation below documents *why* the transform is shaped the way it is, so the feature stays interpretable and debuggable, not because the output is meant to be an authoritative hazard estimate.

The goal is to solve for the maximum epicentral surface radius $R_{\text{felt}}$ (in km) where ground shaking reaches or exceeds a threshold Modified Mercalli Intensity ($MMI_{\text{target}}$), taking into account regional crustal attenuation and site-specific soil amplification.

---

### Step 1: Base Intensity Prediction Equation

We start with a generalized non-linear Intensity Prediction Equation (IPE) modeling $MMI$ as a function of Moment Magnitude ($M$) and 3D effective hypocentral distance ($R_{\text{eff}}$):

$$MMI_{\text{base}}(M, R_{\text{eff}}) = c_1 + c_2 M - c_3 \log_{10}(R_{\text{eff}}) - c_4 R_{\text{eff}}$$

Where:

* $c_1$: Constant regional baseline parameter representing source energy coupling.
* $c_2$: Magnitude scaling coefficient ($M$ scales logarithmically with seismic energy $E \propto 10^{1.5M}$).
* $c_3$: Geometric spreading coefficient. The $\log_{10}(R)$ term captures the roughly $1/R$ decay of body/surface wave amplitude as the wavefront spreads over an expanding area.
* $c_4$: Inelastic (anelastic) absorption coefficient. The linear-in-$R$ term captures the exponential energy loss from internal material friction and scattering ($\propto e^{-\gamma R}$). Taking $\log_{10}$ of an exponential decay turns it into a term linear in $R$, which is why $c_4$ multiplies $R$ directly rather than $\log_{10}(R)$.

This two-term shape, one $\log$ term for geometric spreading and one linear term for anelastic loss, is the standard structure used in empirical macroseismic IPEs (e.g. the Allen, Wald & Worden style regional intensity attenuation models). The six regional presets below are a compact, hand-tuned approximation of that family, not a literal reproduction of any single published coefficient set (see sourcing note under the table).

---

### Step 2: Site Amplification Correction ($\Delta MMI_{\text{site}}$)

Seismic waves expand from high-velocity bedrock into lower-velocity surface soils, causing wave impedance ($\rho \cdot V_s$) to drop. To conserve energy flux, wave amplitude increases.

Using the average shear-wave velocity in the top 30 meters ($V_{s30}$) relative to standard engineering baseline rock ($V_{\text{ref}} = 760\text{ m/s}$, the NEHRP B/C boundary):

$$\Delta MMI_{\text{site}} = b_{\text{site}} \log_{10}\left(\frac{V_{\text{ref}}}{V_{s30}}\right)$$

Where $b_{\text{site}} = 1.8$ is an empirical site scaling exponent (in the same spirit as the $V_{s30}$-based amplification terms used in GMICE/ShakeMap-style site corrections; see sourcing note below).

* If $V_{s30} < 760\text{ m/s}$ (soils/basins), $\Delta MMI_{\text{site}} > 0$ (shaking amplifies).
* If $V_{s30} > 760\text{ m/s}$ (hard rock), $\Delta MMI_{\text{site}} < 0$ (shaking de-amplifies).

The observed ground intensity $MMI_{\text{obs}}$ combines base radiation and local site effects:

$$MMI_{\text{obs}} = MMI_{\text{base}} + \Delta MMI_{\text{site}}$$

---

### Step 3: Setting the Target Intensity Boundary & Isolating Distance

We want the distance at which $MMI_{\text{obs}}$ first drops to exactly $MMI_{\text{target}}$, i.e. the outer boundary of the "felt" region. Setting $MMI_{\text{obs}} = MMI_{\text{target}}$:

$$MMI_{\text{base}} + \Delta MMI_{\text{site}} = MMI_{\text{target}}$$

$$MMI_{\text{base}} = MMI_{\text{target}} - \Delta MMI_{\text{site}}$$

We define this isolated value as the **adjusted target intensity** ($MMI_{\text{adj}}$), effectively, "what intensity would need to be predicted on baseline rock so that, once the local site correction is applied, the observed intensity equals the target":

$$MMI_{\text{adj}} \equiv MMI_{\text{target}} - \Delta MMI_{\text{site}}$$

Substituting into the IPE:

$$c_1 + c_2 M - c_3 \log_{10}(R_{\text{eff}}) - c_4 R_{\text{eff}} = MMI_{\text{adj}}$$

Rearranging to isolate the distance terms on the left:

$$c_3 \log_{10}(R_{\text{eff}}) + c_4 R_{\text{eff}} = c_1 + c_2 M - MMI_{\text{adj}}$$

Divide by $c_4$:

$$\left(\frac{c_3}{c_4}\right) \log_{10}(R_{\text{eff}}) + R_{\text{eff}} = \frac{c_1 + c_2 M - MMI_{\text{adj}}}{c_4}$$

To simplify the algebra, define two helper constants:

$$a = \frac{c_3}{c_4}, \quad b = \frac{c_1 + c_2 M - MMI_{\text{adj}}}{c_4}$$

so the equation becomes the transcendental form:

$$a \log_{10}(R_{\text{eff}}) + R_{\text{eff}} = b$$

Dividing through by $a$:

$$\log_{10}(R_{\text{eff}}) + \frac{R_{\text{eff}}}{a} = \frac{b}{a}$$

This is the point where ordinary algebra stops working: $R_{\text{eff}}$ appears both inside a logarithm and as a bare linear term, so there is no way to isolate it with elementary inverse operations (no combination of $\exp$, roots, or logs alone untangles a sum of $\log(x)$ and $x$). That's exactly the situation the Lambert $W$ function exists to solve.

---

### Step 4: Analytical Closed-Form Solution via Lambert $W$

The Lambert $W$ function is defined as the (multivalued, but here single-valued on the relevant branch) inverse of $f(W) = W e^W$, i.e. $W(z)$ is "the value you'd need to plug into $xe^x$ to get $z$ back out." Any equation that can be massaged into the shape $Xe^X = Z$ has closed-form solution $X = W(Z)$. The steps below are purely algebraic manipulation to force our equation into that shape.

**1. Convert base-10 logarithm to natural logarithm ($\ln$):**
Using $\log_{10}(x) = \frac{\ln(x)}{\ln(10)}$:

$$\frac{\ln(R_{\text{eff}})}{\ln(10)} + \frac{R_{\text{eff}}}{a} = \frac{b}{a}$$

**2. Clear denominators by multiplying both sides by $\ln(10)$:**

$$\ln(R_{\text{eff}}) + \left(\frac{\ln(10)}{a}\right) R_{\text{eff}} = \frac{b \ln(10)}{a}$$

**3. Exponentiate both sides:**

$$\exp\left[ \ln(R_{\text{eff}}) + \left(\frac{\ln(10)}{a}\right) R_{\text{eff}} \right] = \exp\left[ \frac{b \ln(10)}{a} \right]$$

Using $e^{u + v} = e^u \cdot e^v$ and $e^{\ln(x)} = x$, the $\ln(R_{\text{eff}})$ term collapses back to a bare $R_{\text{eff}}$ factor:

$$R_{\text{eff}} \cdot \exp\left( \frac{\ln(10)}{a} R_{\text{eff}} \right) = 10^{b/a}$$

**4. Multiply both sides by $\frac{\ln(10)}{a}$** so that the coefficient of $R_{\text{eff}}$ inside the exponent matches the coefficient outside it. This is the step that actually produces the $Xe^X$ shape:

$$\left[ \frac{\ln(10)}{a} R_{\text{eff}} \right] \cdot \exp\left[ \frac{\ln(10)}{a} R_{\text{eff}} \right] = \frac{\ln(10)}{a} \cdot 10^{b/a}$$

Let:

$$X = \frac{\ln(10)}{a} R_{\text{eff}}, \qquad Z = \frac{\ln(10)}{a} 10^{b/a}$$

so the relation is exactly:

$$X e^X = Z$$

**5. Apply the Lambert $W$ function** (taking $W$ of both sides is valid because $W$ is defined as the inverse of $x \mapsto xe^x$):

$$\frac{\ln(10)}{a} R_{\text{eff}} = W\left( \frac{\ln(10)}{a} 10^{b/a} \right)$$

**6. Isolate $R_{\text{eff}}$:**

$$R_{\text{eff}} = \frac{a}{\ln(10)} \cdot W\left( \frac{\ln(10)}{a} 10^{b/a} \right)$$

We take the principal branch $W_0$ (the real branch valid for arguments $z \ge -1/e$, which covers the physically sensible parameter ranges here); this is what `scipy.special.lambertw(z).real` returns.

---

### Step 5: Finite-Fault Saturation & Surface Projection

$R_{\text{eff}}$ is a 3D effective hypocentral distance. To map it down to a 2D epicentral surface radius $R_{\text{felt}}$, we remove the focal depth $h$ and a finite-fault saturation term $R_0$ via the Pythagorean relation:

$$R_{\text{eff}}^2 = R_{\text{felt}}^2 + R_0^2 + h^2$$

Where:

* $h$: Focal depth in kilometers.
* $R_0 = 10^{-0.281 + 0.251 M}$: An equivalent fault-rupture-dimension term. Its role is purely numerical: without it, $R_{\text{eff}} \to 0$ would force infinite/undefined ground motion at the hypocenter for large $M$, since a point-source model breaks down once the rupture itself is large. $R_0$ acts as a magnitude-dependent floor that keeps the geometry well-behaved.

Solving for $R_{\text{felt}}$:

$$R_{\text{felt}} = \sqrt{\max\left(0, \; R_{\text{eff}}^2 - R_0^2 - h^2\right)}$$

The $\max(0, \cdot)$ clamp handles the case where the depth/finite-fault terms alone already account for more attenuation than the target intensity allows. In that case, the site is never felt at that threshold at the surface, so the feature is bottomed out at $0$ km rather than becoming complex/undefined. $\blacksquare$

---

# Configuration Parameter Tables

**Sourcing note:** these two tables are a compact, hand-built parameterization, not a verbatim transcription of one paper's coefficient table. The overall *functional form*, a $c_1 + c_2 M - c_3\log_{10}(R) - c_4 R$ regional IPE, and a $V_{s30}$-referenced-to-760-m/s site term, follows the standard structure used in published macroseismic intensity attenuation work (e.g. Allen, Wald & Worden (2012), *"An Intensity Prediction Equation for the Central and Eastern United States and California,"* related IPE studies, and the $V_{s30}$-based GMICE/ShakeMap site-correction literature, e.g. Worden et al. (2012) and the broader NEHRP $V_{s30}$ site-classification convention). The specific per-region numeric values in these tables were assembled and tuned to be physically ordered and self-consistent. For example, `stable_cena` has the lowest $c_4$ because cratonic crust is known to transmit shaking unusually far, which is also the real-world basis for the 2011 Virginia earthquake being felt across much of the eastern US. These values were not pulled verbatim from a single citable source table. Treat the qualitative ordering as literature-grounded and the exact digits as an engineered approximation, worth revisiting against a primary IPE/GMICE paper's electronic supplement if you need the feature to match a specific published model exactly.

### 1. Regional Tectonic Attenuation Coefficients (`coefficient_configurations`)

These parameters account for regional crustal age, thermal structure, and tectonic style:

| Region Key | Tectonic Regime / Example Region | Baseline $c_1$ | Magnitude Scaling $c_2$ | Spreading $c_3$ | Absorption $c_4$ | Physical Characteristics |
| --- | --- | --- | --- | --- | --- | --- |
| `"active"` | Himalayas, Alpine Belt, Mediterranean | $2.08$ | $1.04$ | $2.09$ | $0.00190$ | Moderate attenuation in complex, folded mountain crust. |
| `"stable"` | Standard Cratons / Shield Regions | $1.85$ | $1.15$ | $1.80$ | $0.00060$ | Old, cold crust; low absorption lets seismic energy travel far. |
| `"stable_cena"` | Central & Eastern North America | $1.35$ | $1.21$ | $1.70$ | $0.00035$ | Ultra-low inelastic attenuation; massive felt areas. |
| `"subduction"` | Cascadia, Japan Megathrust, Chile | $2.45$ | $0.98$ | $2.15$ | $0.00120$ | Deep slab paths; strong geometrical dispersion. |
| `"strike_slip"` | San Andreas Fault, California | $1.95$ | $1.05$ | $2.20$ | $0.00250$ | High shallow crustal fracturing increases wave absorption. |
| `"volcanic"` | Iceland, East African Rift | $2.10$ | $1.02$ | $2.35$ | $0.00410$ | Hot, fluid-filled crust causes rapid wave energy loss. |

---

### 2. Soil Profile & Velocity Configurations (`soil_configurations`)

These parameters dictate site amplification based on upper $30\text{m}$ shear-wave velocity ($V_{s30}$), following the NEHRP-style site-class convention referenced to the $760\text{ m/s}$ B/C boundary:

| Soil Key | Description / Geological Setting | $V_{s30}$ (m/s) | $\Delta MMI_{\text{site}}$ vs 760m/s | Effect on Shaking Radius |
| --- | --- | --- | --- | --- |
| `"hard_rock"` | Unweathered granite, basalt, metamorphic bedrock | $1500.0$ | $-0.53$ | **Reduces radius** (High velocity prevents wave trapping). |
| `"solid_rock"` | Standard baseline engineering rock (Class B/C boundary) | $760.0$ | $0.00$ | **Baseline** (No amplification adjustment). |
| `"soft_rock"` | Weathered rock, dense sandstones, stiff gravels | $560.0$ | $+0.24$ | **Slight increase** in felt extent. |
| `"stiff_soil"` | Stiff urban clay, dense sands, compacted fills | $270.0$ | $+0.81$ | **Moderate increase** (Common urban profile). |
| `"soft_basin"` | Deep river alluvium, lake deposits, sedimentary basins | $150.0$ | $+1.27$ | **Significant expansion** (Strong resonance/amplification). |
| `"swamp_mud"` | Very soft marsh muds, saturated bay clays | $100.0$ | $+1.59$ | **Maximum expansion** (Extreme low-frequency trapping). |

---

# Real-World Sanity Check (Not a Formal Validation)

The point of the note below is illustrative, not a rigorous accuracy claim. Formally validating this feature would mean comparing many events against DYFI-style felt reports; this is one hand-picked example that shows the output lands in a physically sensible range, which is a useful gut check while building the feature and worth recording alongside the code.

Running the function for the **2015 Gorkha (Nepal) earthquake** parameters, $M = 7.8$, depth $= 8.22\text{ km}$, region `"active"`, gives:

**MMI ≥ 3 felt boundary:**

| Rock Profile | Predicted Radius |
| --- | --- |
| `hard_rock` | 516.7 km |
| `solid_rock` | 672.1 km |
| `stiff_soil` | 939.2 km |
| `soft_basin` | 1104.1 km |

**MMI ≥ 4 felt boundary:**

| Rock Profile | Predicted Radius |
| --- | --- |
| `hard_rock` | 278.6 km |
| `solid_rock` | 395.6 km |
| `stiff_soil` | 614.2 km |
| `soft_basin` | 757.0 km |

The Gorkha earthquake's actual epicenter was roughly 80 km northwest of Kathmandu at a hypocentral depth of about 8.2 km, matching the parameters used here, and it was widely reported as felt across Nepal, northern India, southern Tibet, Bhutan, and Bangladesh, a footprint that spans several hundred to roughly a thousand kilometers from the source. The Kathmandu Valley itself sits on a deep, soft lacustrine (former lake-bed) sediment basin, which is well documented as a strong amplifier of shaking intensity there, both in 2015 and in the historical 1934 Nepal–Bihar earthquake. That geological fact lines up qualitatively with the model: the `soft_basin` profile is the one that produces the largest felt radius, and its MMI ≥ 3 to 4 range (roughly 750 to 1,100 km) is in the right ballpark for the actual multi-country felt extent, while the `hard_rock` profile (roughly 280 to 520 km) would clearly have understated it.

This is one event, hand-checked, not a systematic backtest, so treat it as a sanity check that the feature behaves the way the underlying physics would suggest, not as evidence the specific coefficients are well-calibrated. If this feature matters a lot to downstream model performance, it would be worth running the same comparison against a small batch of well-documented events with known DYFI felt-radius data before trusting it further.