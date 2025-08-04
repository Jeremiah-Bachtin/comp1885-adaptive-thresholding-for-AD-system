# Dataset Metadata – `comp1884_train_scores.csv`

**Source:** Hourly historical (Mar 2017 - Jan 2025) weather data retrieved from [Open-Meteo](https://open-meteo.com), including: <br>
`timestamp`, `temperature_2m`, `surface_pressure`, `wind_speed_10m`, and `precipitation` <br> 
These are supplemented with outputs from the COMP1884 anomaly detection pipeline. <br>
**Model:** IFS (Integrated Forecasting System)  
**Location:** London Heathrow (London timezone)  
**Licence:** Public domain (refer to Open-Meteo usage terms)

| Column Name             | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `timestamp`            | Hourly timestamp (London Time, DST-aware)                                  |
| `temperature_2m`       | Raw temperature at 2 metres above ground (°C)                              |
| `surface_pressure`     | Raw surface pressure (hPa)                                                  |
| `wind_speed_10m`       | Raw wind speed at 10 metres above ground (km/h)                            |
| `precipitation`        | Raw precipitation (mm)                                                     |
| `temperature_2m_z`     | 60-day rolling z-score of temperature (IF input)                           |
| `surface_pressure_z`   | 60-day rolling z-score of pressure (IF input)                              |
| `wind_r`               | Wind smoothed over 3 hours and IQR-scaled over 60 days (IF input)          |
| `precip_log`           | log1p-transformed precipitation                                             |
| `precip_z_12h`         | 12-hour rolling z-score of `precip_log` (IF input)                         |
| `precip_z_24h`         | 24-hour rolling z-score of `precip_log` (IF input)                         |
| `hour`                 | Hour of day (0–23)                                                          |
| `hour_sin`             | Sine encoding of hour (LSTM-AE input)                                      |
| `hour_cos`             | Cosine encoding of hour (LSTM-AE input)                                    |
| `month`                | Month of year (1–12)                                                        |
| `month_sin`            | Sine encoding of month (LSTM-AE input)                                     |
| `month_cos`            | Cosine encoding of month (LSTM-AE input)                                   |
| `if_score`             | Isolation Forest anomaly score (lower = more anomalous)                    |
| `is_if_anomaly`        | Binary IF anomaly flag (1 = anomaly, based on 3rd percentile threshold)     |
| `used_in_lstm_training`| Indicates whether the timestamp was used to construct LSTM-AE training seq |
| `lstm_score`           | Aggregated reconstruction error from LSTM-AE                               |
| `is_lstm_anomaly`      | Binary LSTM-AE anomaly flag (1 = anomaly, based on 95th percentile cutoff) |
| `anomaly_label`        | Final hybrid label: Normal / Point / Pattern / Compound anomaly            |

> This metadata supports FAIR data principles: findability, accessibility, interoperability, and reusability.
