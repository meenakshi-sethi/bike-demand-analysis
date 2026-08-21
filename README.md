# Bike Demand Forecasting for a Shared-Mobility Relaunch

A multiple linear regression model that explains and predicts daily bike-rental
demand for a US bike-sharing operator, built to answer a concrete business
question: **which levers actually move demand, and by how much?**

## Business context

BoomBikes, a US bike-sharing operator, saw revenue drop sharply during the
COVID-19 lockdowns. Ahead of the market reopening, the company needed to
understand what drives day-to-day rental demand — weather, seasonality,
calendar effects — so it could time marketing spend, fleet servicing, and
expansion plans around the factors that actually matter, rather than guessing.

This project models daily rental counts (`cnt`) from two years (2018–2019) of
weather and calendar data and turns the resulting model into specific,
actionable recommendations.

## Data

`data/day.csv` — 730 daily records with season, month, weekday, holiday/working-day
flags, weather situation, temperature, humidity, windspeed, and rental counts
(casual, registered, total). Field definitions are in
[`data/data_dictionary.md`](data/data_dictionary.md).

Source: Fanaee-T, H. and Gama, J., "Event labeling combining ensemble detectors
and background knowledge," *Progress in Artificial Intelligence* (2013).

## Approach

1. **Data preparation** — recoded numeric categoricals (`season`, `weathersit`,
   `mnth`, `weekday`) to proper categorical types and one-hot encoded them
   (`drop_first=True` to avoid the dummy-variable trap); dropped `casual` and
   `registered` to avoid leaking the target (`cnt = casual + registered`).
2. **EDA** — visualized numeric predictors (pair plots, correlation heatmap)
   and categorical predictors (box plots against `cnt`) to check relationships
   and spot multicollinearity risks before modeling.
3. **Feature selection** — built the model iteratively (`lr1` → `lr6`),
   dropping one variable per round based on **high VIF (multicollinearity)**
   or **high p-value (statistical insignificance)**, re-checking VIF after
   each drop, until every remaining feature was both significant and
   independent.
4. **Assumption checks** — validated the final model against the linear
   regression assumptions: normally distributed residuals, homoscedasticity,
   linearity, and low multicollinearity (VIF).
5. **Evaluation** — scored the final model on a held-out test set using
   R² and adjusted R² (not just training performance).

## Result

The final model (`lr6`) uses 10 features and generalizes well from train to test:

| Metric | Train | Test |
|---|---|---|
| R² | 0.824 | 0.820 |
| Adjusted R² | 0.821 | 0.812 |

**cnt ≈ 0.084 + 0.231·yr + 0.043·workingday + 0.564·temp − 0.155·windspeed
+ 0.083·season_summer + 0.129·season_winter + 0.095·mnth_sep
+ 0.057·weekday_sat − 0.075·weathersit_mist − 0.307·weathersit_snow_rain**

*(all variables scaled 0–1; coefficients are directly comparable)*

### What actually drives demand

- **Temperature is the single biggest lever** — by a wide margin the strongest
  predictor of daily demand.
- **Year-over-year growth is real and large** — 2019 demand was structurally
  higher than 2018, independent of weather, confirming the business is
  gaining organic traction and not just riding good weather.
- **Bad weather is costly, not just inconvenient** — light snow/rain suppresses
  demand more than any other single factor drags it down.
- **Summer and winter both outperform spring**, and **September** and
  **Saturdays** see an extra demand bump on top of seasonal effects.

### Recommendations

1. **Weight marketing spend toward September and the summer/winter shoulder
   months**, when demand is already trending up — that's where incremental
   spend converts best.
2. **Use light-snow/rain days for fleet maintenance**, not promotions — demand
   drops sharply regardless of marketing, so this is the lowest-cost window
   to service the fleet.
3. **Don't over-invest in weekday-specific promotions** — the model finds a
   real Saturday bump but no broader weekday effect, so weekend-timed offers
   are the better lever than day-of-week discounting generally.
4. **Trust the year-over-year growth trend** when planning capacity for the
   reopening — the `yr` effect suggests demand recovery won't just track
   weather back to normal, it will exceed prior-year levels.

## Repo structure

```
data/           day.csv + data dictionary
notebooks/      bike_demand_regression.ipynb — full analysis, model build, evaluation
```

## Running it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/bike_demand_regression.ipynb
```

## Next steps

- Track the `lr1`→`lr6` model iterations with MLflow for a reproducible
  experiment log instead of notebook cell history.
- Wrap the final model in a small Streamlit app so demand can be predicted
  interactively from season/weather/day-type inputs.
- Compare the OLS/VIF-selected model against a regularized (Ridge/Lasso) or
  tree-based (Random Forest) baseline to sanity-check feature selection and
  quantify any accuracy/interpretability trade-off.
