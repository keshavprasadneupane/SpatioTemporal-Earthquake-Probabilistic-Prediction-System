### Derivation: Solving Felt Surface Distance $R_{\text{felt}}$ via Allen et al. (2012), Depth $h$, & Lambert W

---

#### 1. Governing Equation (USGS Intensity Prediction Model)

$$MMI = c_1 + c_2 M - c_3 \log_{10}(R_{\text{eff}}) - c_4 R_{\text{eff}}$$

**Definitions:**

* $M$: Earthquake magnitude.
* $h$: Focal depth from catalog (km).
* $MMI$: Target Modified Mercalli Intensity threshold (e.g., $MMI = 2.5$ for felt vibration).
* $R_0(M) = 10^{-0.281 + 0.251 M}$: Near-field magnitude saturation factor (km).
* $R_{\text{eff}} = \sqrt{R_{\text{felt}}^2 + R_0(M)^2 + h^2}$: Total 3D effective distance (km).
* $R_{\text{felt}}$: Horizontal surface felt radius from epicenter (km).
* $c_1, c_2, c_3, c_4$: Empirical regional constants.

---

#### 2. Reduction to Standard Form ($x + a\log_{10}(x) = b$)

Isolate 3D effective distance terms on the left side:

$$c_4 R_{\text{eff}} + c_3 \log_{10}(R_{\text{eff}}) = c_1 + c_2 M - MMI$$

Divide through by $c_4$:

$$R_{\text{eff}} + \left(\frac{c_3}{c_4}\right) \log_{10}(R_{\text{eff}}) = \frac{c_1 + c_2 M - MMI}{c_4}$$

Substitute intermediate variables:

* Let $x = R_{\text{eff}}$
* Let $a = \frac{c_3}{c_4}$
* Let $b = \frac{c_1 + c_2 M - MMI}{c_4}$

$$\implies x + a \log_{10}(x) = b$$

---

#### 3. Solving for $x$ ($R_{\text{eff}}$) via Lambert W Function

**A. Convert $\log_{10}$ to natural log ($\ln$):**

$$x + a \frac{\ln(x)}{\ln(10)} = b$$

Multiply by $\frac{\ln(10)}{a}$:

$$\frac{x \ln(10)}{a} + \ln(x) = \frac{b \ln(10)}{a}$$

**B. Exponentiate both sides:**

$$e^{\frac{x \ln(10)}{a} + \ln(x)} = e^{\frac{b \ln(10)}{a}}$$

$$x \cdot e^{\frac{x \ln(10)}{a}} = 10^{b/a}$$

**C. Transform to $Y e^Y = Z$ form:**

Multiply both sides by $\frac{\ln(10)}{a}$:

$$\left[\frac{x \ln(10)}{a}\right] e^{\left[\frac{x \ln(10)}{a}\right]} = \frac{\ln(10)}{a} \cdot 10^{b/a}$$

**D. Apply Lambert W Function ($W_0$):**

$$\frac{x \ln(10)}{a} = W_0 \left( \frac{\ln(10)}{a} \cdot 10^{b/a} \right)$$

Isolate $x$ ($R_{\text{eff}}$):

$$R_{\text{eff}} = \frac{a}{\ln(10)} W_0 \left( \frac{\ln(10)}{a} \cdot 10^{b/a} \right)$$

*(Note: Since $a > 0$, the argument $z > 0$, guaranteeing a unique real root on the principal branch $W_0$.)*

---

#### 4. Extract Surface Felt Radius ($R_{\text{felt}}$)

Using $R_{\text{eff}} = \sqrt{R_{\text{felt}}^2 + R_0(M)^2 + h^2}$:

$$R_{\text{felt}} = \sqrt{R_{\text{eff}}^2 - R_0(M)^2 - h^2}$$

* **Condition:** If $R_{\text{eff}}^2 \le R_0(M)^2 + h^2$, then $R_{\text{felt}} = 0.0\text{ km}$ (the target intensity $MMI$ is absorbed vertically by depth $h$ or saturation $R_0$ before producing surface vibration at that threshold).

---

#### 5. Empirical Regional Coefficients (Allen, Wald, & Worden, 2012)

The coefficients $c_1, c_2, c_3, c_4$ depend on the tectonic environment of the crust:

| Tectonic Setting | Description & Examples | $c_1$ (Baseline) | $c_2$ (Mag Scaling) | $c_3$ (Geom Spreading) | $c_4$ (Anelastic Damping) |
| --- | --- | --- | --- | --- | --- |
| **Active Tectonic Crust (ACR)** | Warm, highly fractured crust (California, Japan, Himalayas, NZ) | $2.08$ | $1.04$ | $2.09$ | $0.0019$ |
| **Stable Continental Crust (SCR)** | Cold, unbroken shield/craton (Central/Eastern US, Australia, Baltic) | $1.85$ | $1.15$ | $1.80$ | $0.0006$ |

---

#### Physical Impact on $a = \frac{c_3}{c_4}$:

* **Active Crust (ACR):** $a = \frac{2.09}{0.0019} \approx 1100.0$
* Higher $c_4$ ($0.0019$) means rapid thermal and mechanical friction absorption. Shaking dies down much closer to the epicenter.


* **Stable Crust (SCR):** $a = \frac{1.80}{0.0006} = 3000.0$
* Lower $c_4$ ($0.0006$) allows seismic waves to travel through rigid, unbroken rock over continental distances with minimal energy loss.