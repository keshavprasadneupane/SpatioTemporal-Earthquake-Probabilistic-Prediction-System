# STEPPS Important Dataset Columns

This reference only includes columns that are likely to provide meaningful value for intensity modeling, attenuation relationships, felt-radius estimation, and district-impact generation.

---

# Tier 1 — Essential Features

These should be investigated first and are likely to explain the majority of intensity variation.

| Column                | Meaning                                    | Why It Matters                                                         |
| --------------------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| `Intensity`           | Observed Modified Mercalli Intensity (MMI) | Primary target variable                                                |
| `Preferred_Magnitude` | Earthquake magnitude                       | Strongest predictor of shaking intensity                               |
| `Depth`               | Hypocenter depth (km)                      | Controls attenuation and surface shaking                               |
| `Rhypo`               | Hypocentral distance (km)                  | Physical distance traveled by seismic waves                            |
| `Regime`              | Broad tectonic setting                     | Different tectonic environments exhibit different attenuation behavior |

**Recommended first model:**

```python
[
    "Preferred_Magnitude",
    "Depth",
    "Rhypo",
    "Regime"
]
```

**Physical interpretation:**

```text
Magnitude  → How much energy was released?
Depth      → How deep was the source?
Distance   → How far did the seismic waves travel?
Regime     → What tectonic environment generated and propagated the waves?
```

Together these variables describe the earthquake source and the large-scale propagation environment.

---

# Tier 2 — Strong Additions

These features often explain a substantial portion of the remaining variance after Tier 1.

| Column          | Meaning                                       | Why It Matters                                     |
| --------------- | --------------------------------------------- | -------------------------------------------------- |
| `Thompson_VS30` | Average shear-wave velocity of the upper 30 m | Captures local soil amplification and site effects |
| `Sub_Regime`    | Detailed tectonic classification              | Refines attenuation behavior within a regime       |

**Recommended second model:**

```python
[
    "Preferred_Magnitude",
    "Depth",
    "Rhypo",
    "Regime",
    "Thompson_VS30",
    "Sub_Regime"
]
```

**Physical interpretation:**

```text
VS30        → Local site amplification
Sub_Regime  → Regional tectonic differences
```

These variables describe why two nearby locations may experience different shaking despite having similar earthquake parameters.

---

# Tier 3 — Potential Refinements

Only investigate these after understanding the behavior of Tier 1 and Tier 2 features.

| Column          | Meaning                       | Why It Matters                                 |
| --------------- | ----------------------------- | ---------------------------------------------- |
| `Rrup_Combined` | Distance to rupture surface   | Preferred distance metric in many modern GMPEs |
| `Repi`          | Epicentral distance           | Alternative distance measure                   |
| `Rjb_Combined`  | Joyner-Boore distance         | Widely used engineering distance metric        |
| `Lithology_1`   | Primary geological material   | Additional site-response information           |
| `Lithology_2`   | Secondary geological material | Additional site-response information           |

**Research Goal:**

Determine whether alternative distance metrics outperform `Rhypo` and whether geological classifications improve predictive performance beyond VS30.

---

# Not Recommended Initially

The following variables are useful for specialized engineering-seismology studies but are unlikely to provide substantial value during the early stages of STEPPS.

| Columns                                                           |
| ----------------------------------------------------------------- |
| `RxFF`                                                            |
| `RyFF`                                                            |
| `Ry0FF`                                                           |
| `PGA_g`                                                           |
| `PGV_cms`                                                         |
| Event identifiers                                                 |
| Geological time classifications (`Eon`, `Era`, `Period`, `Epoch`) |
| References and metadata fields                                    |

**Why not PGA/PGV?**

For attenuation modeling:

```text
Magnitude
Depth
Distance
      ↓
PGA / PGV
      ↓
MMI
```

Since PGA and PGV are already downstream products of the earthquake, including them can obscure the relationship you're trying to learn between earthquake parameters and observed intensity.

---

# Suggested Research Order

## Stage 1 — Baseline Attenuation Model

```python
[
    "Preferred_Magnitude",
    "Depth",
    "Rhypo",
    "Regime"
]
```

Questions to answer:

* Does intensity decrease linearly with distance?
* Does it decrease with log(distance)?
* How important is depth?
* Do different tectonic regimes require different coefficients?

---

## Stage 2 — Site Effects

```python
[
    "Preferred_Magnitude",
    "Depth",
    "Rhypo",
    "Regime",
    "Thompson_VS30",
    "Sub_Regime"
]
```

Questions to answer:

* How much variance does VS30 explain?
* Does Sub_Regime improve performance over Regime alone?

---

## Stage 3 — Alternative Distance Metrics

Experiment with replacing or adding:

```python
[
    "Rrup_Combined",
    "Repi",
    "Rjb_Combined"
]
```

Questions to answer:

* Is Rhypo truly the best distance metric?
* Do modern rupture-based distances outperform hypocentral distance?

---

## Stage 4 — Geological Refinements

```python
[
    "Lithology_1",
    "Lithology_2"
]
```

Questions to answer:

* Do geological units add information beyond VS30?
* Are there lithology-specific amplification patterns?

---

# Expected Relative Importance

```text
Magnitude      ████████████████████
Distance       ██████████████████
Regime         ██████████
Depth          ████████
VS30           ██████
Sub_Regime     ████
Lithology      ██
Fault Geometry █
```

---

# Working Hypothesis for STEPPS

A strong initial attenuation relationship will likely be driven primarily by:

```text
Magnitude
Distance
Regime
Depth
```

with:

```text
VS30
Sub_Regime
```

providing the largest secondary improvements.

The remaining variables should be viewed as refinements and only investigated after a solid baseline model has been established.
