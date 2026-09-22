# Earthquake Felt Radius Estimation Using Modified Mercalli Intensity (MMI)

## Purpose

To estimate the approximate **surface radius** where an earthquake can be felt, we use a seismic intensity attenuation relationship based on:

* Earthquake Magnitude (`M`)
* Earthquake Depth (`h`)
* Modified Mercalli Intensity (`MMI`)

The resulting radius can be used to determine:

* Potentially affected districts
* Spatial influence regions
* Earthquake impact visualization
* Feature engineering for spatiotemporal forecasting models

---

# Step 1: Intensity Attenuation Equation

The relationship between earthquake magnitude, distance, and perceived shaking intensity is approximated by:

```math
MMI = 1.41 + 1.68M - 0.00345R - 2.08\log_{10}(R)
```

Where:

| Variable | Description                 |
| -------- | --------------------------- |
| `MMI`    | Modified Mercalli Intensity |
| `M`      | Earthquake Magnitude        |
| `R`      | Hypocentral Distance (km)   |

---

# Step 2: Hypocentral Distance

The attenuation equation uses the distance from the earthquake source to the observer.

This distance is called the **Hypocentral Distance** and is calculated as:

```math
R = \sqrt{d^2 + h^2}
```

Where:

| Variable | Description                          |
| -------- | ------------------------------------ |
| `R`      | Hypocentral Distance (km)            |
| `d`      | Surface Distance from Epicenter (km) |
| `h`      | Earthquake Depth (km)                |

---

# Step 3: Deriving Surface Radius

Since the forecasting system requires the surface influence radius, we rearrange the equation:

```math
d = \sqrt{R^2 - h^2}
```

Where:

| Variable | Description               |
| -------- | ------------------------- |
| `d`      | Surface Felt Radius (km)  |
| `R`      | Hypocentral Distance (km) |
| `h`      | Earthquake Depth (km)     |

---

# Procedure

Given:

* Magnitude `M`
* Depth `h`
* Target intensity threshold `MMI`

1. Solve the attenuation equation for `R`.
2. Compute the surface radius using:

```math
d = \sqrt{R^2 - h^2}
```

3. Use `d` as the estimated radius where shaking of at least the chosen intensity may be experienced.

---

# Modified Mercalli Intensity (MMI) Scale

The Modified Mercalli Intensity scale describes how strongly an earthquake is felt by people and structures.

| MMI  | Description         | Typical Effects                               |
| ---- | ------------------- | --------------------------------------------- |
| I    | Not Felt            | Detected only by sensitive instruments        |
| II   | Weak                | Felt by a few people at rest                  |
| III  | Weak but Noticeable | Similar to vibration from a passing truck     |
| IV   | Light               | Windows and dishes may rattle                 |
| V    | Moderate            | Felt by nearly everyone indoors               |
| VI   | Strong              | Minor plaster cracks, objects may fall        |
| VII  | Very Strong         | Damage to poorly constructed buildings        |
| VIII | Severe              | Partial structural damage                     |
| IX   | Violent             | Significant structural damage                 |
| X    | Extreme             | Many masonry structures destroyed             |
| XI   | Catastrophic        | Few structures remain standing                |
| XII  | Total Destruction   | Ground deformation and widespread devastation |

---

# Recommended Thresholds

| Threshold | Meaning            | Suggested Use                           |
| --------- | ------------------ | --------------------------------------- |
| MMI ≥ II  | Barely Felt        | Maximum perception radius               |
| MMI ≥ III | Clearly Felt       | Recommended for STEPFS influence radius |
| MMI ≥ IV  | Noticeable Shaking | Population impact analysis              |
| MMI ≥ V   | Moderate Shaking   | Risk assessment                         |
| MMI ≥ VI  | Strong Shaking     | Potential damage zone                   |

For district-level earthquake forecasting, **MMI ≥ III** is often a practical threshold because it represents the region where people are likely to notice the earthquake.

---


### Key Features to Extract

| Extracted Feature | Formula / Logic | Physical Meaning |
| --- | --- | --- |
| **`felt_radius_km`** | $R_{\text{felt}}$ solved via Lambert W ($MMI=2.5$) | Boundary distance where humans perceive vibration. |
| **`felt_area_sqkm`** | $A_{\text{felt}} = \pi \cdot R_{\text{felt}}^2$ | Linearizes logarithmic magnitude into total geographic coverage. |
| **`max_mmi_at_epicenter`** | $MMI_{\text{max}}$ at $R_{\text{eff}} = \sqrt{R_0(M)^2 + h^2}$ | Peak intensity accounting for focal depth `depth`. |
| **`rolling_felt_area_30d`** | $\sum A_{\text{felt}}$ over rolling time window | Quantifies regional seismic disturbance over time. |

