# Net Demand Forecasting During the Sobriety Period

Forecasting of daily **net electricity demand** during the sobriety period, using a hybrid approach that combines Generalized Additive Models (GAM/BAM), ARIMA and LightGBM. The goal is to estimate the **conditional 80th percentile** of demand, evaluated with the **pinball loss (τ = 0.8)**.

**Academic project:** developed as part of the first year (M1) of the Master's in Mathematics and Artificial Intelligence at Paris-Saclay University. This work achieved the **best score in the class** in the forecasting challenge and the **highest grade of the year group (18/20)**.

**Authors:** Jordy Saltos and Juan Carlos Pajares

---

## Problem Statement

- **Task:** supervised regression with temporal structure (daily time series forecasting).
- **Target:** `Net_demand`, predicted for the test period from the information in the training set.
- **Metric:** pinball loss for quantile τ = 0.8. This asymmetric loss penalizes underestimation more than overestimation, which matters in energy systems where under-forecasting demand carries higher operational risk.
- **Validation:** chronological split (no shuffling). Training on years ≤ 2020 and validation on years > 2020. Block cross-validation is used for the additive models.

## Data

Daily observations with the following groups of predictors:

| Group | Examples |
|---|---|
| Temporal | `Date`, `toy` (time of year), `DLS` (daylight saving) |
| Meteorological | `Temp`, `Temp_s95`, `Temp_s99`, min/max temperatures, `Solar_power`, `Nebulosity`, `Wind`, `Wind_power` |
| Lagged | `Net_demand.1`, `Net_demand.7`, `Load.1`, `Load.7` |
| Calendar / social | `WeekDays`, `Month`, `BH` (bank holidays), school holiday zones, `Summer_break`, `Christmas_break` |

## Methodology

### 1. Feature engineering
- Harmonization of train/test feature spaces and a numeric time index (`Time`).
- Pandemic indicator (`Is_pandemic`, 2020–2021) to capture regime shifts.
- Calendar features: quarter, weekend, month start/end, holiday indicators.
- Cyclical encoding of `toy` (sine/cosine).
- Nonlinear transformations (log, sqrt, squared, cubic), pairwise **interaction terms** and **ratio features**.
- Final design matrix of **652 explanatory variables**, later reduced by feature selection.

### 2. Models compared
1. **Linear regression:** baseline, with a residual-quantile adjustment.
2. **LightGBM:** with mean and with quantile (τ = 0.8) objectives, plus feature selection by gain importance (13 variables retained).
3. **BAM / GAM:** smooth terms and tensor-product interactions built from the LightGBM-selected variables, with basis dimensions tuned via `gam.check()` and 10-fold block cross-validation.
4. **GAM + ARIMA:** ARIMA(3,0,3)(1,0,0)[7] fitted on out-of-sample GAM residuals to capture remaining linear serial dependence.
5. **GAM + LightGBM on residuals (final model):** a quantile LightGBM (τ = 0.8) learns the nonlinear structure left in the GAM residuals.

### 3. Final hybrid forecast

$$\tilde{Y}_t = \hat{f}_{GAM}(X_t) + \hat{r}_t^{LightGBM}$$

Residuals are generated out-of-sample with block cross-validation to avoid look-ahead bias.

## Results

Pinball loss (τ = 0.8), lower is better:

| Model | Validation | Test |
|---|---|---|
| Linear regression (reduced) | 784.09 | – |
| Linear regression + residual quantile | 514.29 | 541.86 |
| LightGBM (conditional mean) | 491.24 | 510.84 |
| LightGBM (quantile τ = 0.8) | 439.38 | 560.16 |
| LightGBM (13 selected variables) + quantile correction | 413.14 | 453.02 |
| BAM (first approach) | 464.80 | 371.49 |
| BAM (updated equation) | 419.08 | 300.91 |
| BAM (basis dimension adjusted) | – | 294.55 |
| GAM + ARIMA (with quantile correction) | 286.31 | 290.60 |
| **GAM + LightGBM on residuals (final)** | – | **271.60** |

### Key findings
- Nonlinear models (LightGBM, GAM) clearly outperform the linear baseline.
- **Distribution drift** between train and test hurts models trained directly on the 0.8 quantile (better on validation, worse on test), so the conditional-mean specification with a quantile correction proved more robust.
- Modeling GAM residuals with ARIMA helps, but the remaining structure is nonlinear, and a LightGBM residual model gives the best result.

### Future work
- Explore interactions among more than three variables in the GAM.
- Online aggregation of experts (e.g. with the `opera` package) to combine the best models.

## Repository Structure

```
.
├── Net_Demand_Forecasting_Informe_Final_Github.Rmd   # Full report (code + analysis)
├── Data/
│   ├── train.csv
│   └── test.csv
├── R/
│   └── score.R                                        # Pinball loss function
└── header.html                                        # HTML header for the report
```

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
```
2. Install the required R packages:
```r
   install.packages(c(
     "tidyverse", "highcharter", "broom", "ranger", "corrplot", "ggcorrplot",
     "data.table", "lightgbm", "lubridate", "mgcv", "yarrr", "magrittr",
     "forecast", "qgam", "viking", "patchwork", "bookdown"
   ))
```
3. Open the `.Rmd` file in RStudio and click **Knit**, or run:
```r
   rmarkdown::render("Net_Demand_Forecasting_Informe_Final_Github.Rmd")
```

## Tech Stack

**R** · `mgcv` (GAM/BAM) · `lightgbm` · `forecast` (ARIMA) · `tidyverse` · `highcharter` · `bookdown`

## Notes

- Test pinball loss is approximated using the lagged variable `Net_demand.1` of the test set to reconstruct the observed values.
- Datasets are not redistributed here unless included in `Data/`.
