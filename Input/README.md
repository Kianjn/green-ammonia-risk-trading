# Model inputs

These files are what the solver reads. They are already reduced to eight representative days per weather label.

| Path | Role |
|------|------|
| `timeseries_<label>.csv` | 192 rows (8 days × 24 hours): `SOLAR`, `WIND`, `LOAD_E`, `LOAD_H`, `LOAD_EP`, all in [0, 1] |
| `output_<label>/decision_variables_short.csv` | Representative-day weights. Column `weights` sums to 365 and is `W[jd, jy]` in the model |
| `output_<label>/decision_variables.csv` | Assignment of each calendar day to a representative day |
| `output_<label>/ordering_variable.csv` | Same assignment as a one-hot matrix. Loaded, not used in the optimisation |
| `weather_scenario_summary.json` | Source year, capacity factors, demand statistics, and weights for each label |

`Scenarios.weather_years` in `Data/data.yaml` selects the labels. The default is `[1, 2, 3, 4, 5]`.

Labels are scenario indices, not calendar years in which agents reinvest. The mapping to ERA5 source years, the common Netherlands capacity-factor calibration, the temperature-coupled demand model, and the representative-day weights are documented in [../docs/TECHNICAL.md](../docs/TECHNICAL.md) under **Uncertainty set**.
