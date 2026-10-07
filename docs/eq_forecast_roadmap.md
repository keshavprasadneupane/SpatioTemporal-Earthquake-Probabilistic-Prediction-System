# Nepal District Earthquake Probability Forecasting System

## Introduction

The objective of this project is to estimate the probability that a district in Nepal will experience shaking above a specified intensity level within a future time horizon.

Rather than attempting to predict the exact occurrence of earthquakes, the system produces probabilistic forecasts similar to weather forecasts. For example:

> There is a 12% probability that District X will experience shaking of at least MMI 4 during the next 30 days.

This distinction is important. Earthquake occurrence remains fundamentally uncertain, particularly over short time scales. The goal is therefore to estimate changing hazard levels and quantify uncertainty rather than predict individual events. 

---

# Project Inputs

The forecasting system is built using four primary data sources.

## Earthquake Catalog

The earthquake catalog contains the historical seismic record:

```text
time
latitude
longitude
depth
mag
magType
nst
gap
dmin
rms
horizontalError
depthError
magError
magNst
```

Each record represents a single earthquake event and serves as the foundation for all temporal and spatial analyses. 

## District Dataset

Each district contains spatial information and neighborhood relationships:

```text
id
province
district
neighbour_ids
centroid_lat
centroid_long
boundary information
```

These data allow seismic activity to be aggregated and forecasted at the district level. 

## MMI–PGA Lookup Table

A lookup table converts Modified Mercalli Intensity (MMI) values into median Peak Ground Acceleration (PGA) values.

The base implementation uses:

```text
MMI = 2.5
```

which represents the minimum perceptible shaking level. Higher thresholds such as MMI 4, MMI 5, and MMI 6 can later be used for stronger shaking forecasts. 

## Rhypo Prediction Models

Two neural-network models estimate hypocentral distance from earthquake characteristics:

### Model 0

Used when:

```text
Magnitude ≤ 5
```

### Model 1

Used when:

```text
Magnitude > 5
```

Inputs:

```text
log10(PGA)
Magnitude
log10(Vs30)
log10(Depth)
```

Output:

```text
log10(Rhypo)
```

Current performance:

| Model | Test R² |
| ----- | ------- |
| M ≤ 5 | ~0.62   |
| M > 5 | ~0.70   |

Residual uncertainty is approximately:

```text
σ ≈ 0.20 (small events)
σ ≈ 0.15 (larger events)
```

These residuals later become part of the probabilistic footprint calculation. 

---

# Defining the Forecasting Problem

The target prediction can be expressed as:

```text
P(at least one event causes shaking ≥ MMI X
in district D
during future interval (t, t + Δ])
```

where:

* X is an MMI threshold
* D is a district
* Δ is the forecast horizon

Recommended forecast horizons:

```text
7 days
30 days
90 days
180 dats
```

Recommended intensity thresholds:

```text
MMI 2.5
MMI 4
MMI 5
MMI 6
```

These combinations create the forecasting targets that the system will learn to predict. 

---

# Project Structure

A clean project structure helps separate data processing, modeling, and evaluation.

```text
eq-forecast/

data/
    raw/
    interim/
    processed/

notebooks/

src/
    catalog.py
    districts.py
    footprint.py
    panel.py
    baselines.py
    models.py
    evaluate.py

configs/
    config.yaml

README.md
```

Configuration values such as MMI thresholds, forecast horizons, and model paths should be stored in a centralized configuration file. 

---

# Building the Earthquake Footprint Engine

The footprint engine transforms an earthquake into a district-level shaking probability.

This is one of the most important components of the entire system.

## Step 1: Select the Appropriate Rhypo Model

```python
if magnitude <= 5:
    model = MODEL_0
else:
    model = MODEL_1
```

## Step 2: Convert MMI to PGA

```python
log_pga = log10(pga_lookup[mmi])
```

## Step 3: Predict Hypocentral Distance

```python
log_rhypo = model.predict(...)
```

## Step 4: Convert Back to Distance

```python
rhypo = 10 ** log_rhypo
```

## Step 5: Compute Epicentral Radius

```python
repi = sqrt(max(rhypo**2 - depth**2, 0))
```

The resulting radius estimates how far from the epicenter the chosen intensity level may be felt. 

---

# Probabilistic District Membership

Using a hard radius cutoff introduces unrealistic boundaries.

Instead, the model uses uncertainty from Rhypo prediction residuals.

For a district located at hypocentral distance \(d_{hypo}\):

```text
p_affect =
1 - Φ(
    (log10(d_hypo) - log_rhypo) / σ
)
```

where:

* Φ is the normal cumulative distribution function
* σ is the Rhypo residual standard deviation

This converts the footprint into a smooth probability surface.

Each district receives:

```text
hard_flag
p_affect
```

allowing both deterministic and probabilistic analyses. 

---

# Event–District Table

After processing all earthquakes, an event–district table is created.

Each row represents one earthquake affecting one district.

```text
event_id
time
mag
depth
district_id
d_epi
d_hypo
repi
p_affect
hard_flag
mmi_level
```

This table becomes the central dataset for forecasting. 

---

# Building Forecast Features

The forecasting model requires features that summarize seismic activity before a given time.

## Activity Features

Examples:

```text
Events during last 1 day
Events during last 7 days
Events during last 30 days
Events during last 90 days
Events during last 365 days
Σ p_affect
```

## Magnitude Features

Examples:

```text
Maximum magnitude
Number of M≥4 events
Number of M≥5 events
Number of M≥6 events
Energy release
Benioff strain
```

## Recency Features

Examples:

```text
Days since last M≥4 event
Days since last M≥5 event
Days since last M≥6 event
```

## Rate Features

Examples:

```text
30-day rate
365-day rate
Rate ratios
Local b-value
```

## Spatial Features

Features computed from neighboring districts:

```text
1-hop neighbors
2-hop neighbors
Regional aggregates
```

## Static Features

Examples:

```text
Vs30
Latitude
Longitude
Province
Distance to major faults
```

## Aftershock Features

An ETAS-inspired aftershock term can be included:

```text
Σ 10^(α(M−Mc)) / (t−ti+c)^p
```

This captures elevated short-term hazard after large earthquakes. 

---

# Forecast Labels

For each district and forecast horizon:

```text
y = 1
```

if at least one future earthquake affects the district during the prediction window.

Otherwise:

```text
y = 0
```

A soft-label alternative may also be used:

```text
1 − Π(1 − p_affect)
```

which preserves footprint uncertainty. 

---

# Baseline Models

Before using machine learning, simple baseline models must be established.

## Poisson Baseline

District event rate:

```text
λ = events / time
```

Probability:

```text
P = 1 − exp(−λΔ)
```

This serves as the primary benchmark for medium- and long-term forecasts. 

## Gutenberg–Richter Baseline

Uses:

```text
λ(M ≥ m) = 10^(a − bm)
```

combined with spatial smoothing and footprint calculations. 

## ETAS Baseline

For short-term forecasting, an Epidemic-Type Aftershock Sequence model is recommended.

ETAS explicitly models aftershock triggering and is often the strongest benchmark for days-to-weeks forecasting horizons. 

---

# Machine Learning Forecasting

After baseline construction, machine learning models can be trained.

A gradient boosting framework such as:

```text
LightGBM
XGBoost
```

is recommended as the first predictive model.

Inputs:

* Activity features
* Magnitude features
* Recency features
* Spatial features
* Static features
* Baseline probabilities

Output:

```text
P(shaking ≥ MMI X in district D during horizon Δ)
```

Predictions should always be calibrated using methods such as:

```text
Isotonic Regression
Platt Scaling
```

to ensure probabilistic accuracy. 

---

# Model Evaluation

Evaluation must respect temporal ordering.

Training and validation periods should be separated chronologically:

```text
Train: 2000–2012
Validate: 2013–2015

Train: 2000–2015
Validate: 2016–2018
```

A temporal gap equal to the forecast horizon should be inserted to prevent leakage. 

## Primary Metrics

### Log Loss

Measures probabilistic accuracy.

### Brier Score

Measures calibration and forecast quality.

### Information Gain

Measures improvement relative to a Poisson baseline.

### Reliability Diagrams

Visualize calibration quality.

ROC-AUC and PR-AUC should be treated as secondary metrics. 

---

# Long-Term Hazard Forecasting

For horizons spanning months to years, the problem becomes a hazard estimation problem rather than a short-term forecasting problem.

The recommended workflow is:

1. Decluster the catalog.
2. Estimate Gutenberg–Richter parameters.
3. Build spatially smoothed seismicity rates.
4. Convert rates into district probabilities.

Using:

```text
P = 1 − exp(−λΔ)
```

provides a probabilistically consistent estimate of long-term seismic hazard. 

---

# Deployment

The final system should expose a simple interface:

```python
forecast(
    district_id,
    as_of_time,
    horizons,
    mmi
)
```

which returns:

```text
Probability
Confidence interval
Forecast horizon
MMI threshold
```

A FastAPI service and district-level visualization dashboard can be built on top of this forecasting engine. 

---

# Responsible Use

This system estimates the probability of future shaking, not future earthquakes.

Outputs should always be presented as:

> Statistical probability of experiencing shaking at or above MMI X during the next Δ period.

and never as:

> Prediction that an earthquake will occur.

Probabilities should always be accompanied by uncertainty estimates and long-term baseline rates. 

---

