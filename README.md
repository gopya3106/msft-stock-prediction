# Stock Price Prediction using Supervised Machine Learning

This project predicts the next-week return of Microsoft (MSFT) from its own recent returns and those of correlated assets (IBM, Alphabet, USD/JPY, GBP/USD, S&P 500, Dow Jones, VIX). We compare XGBoost, Keras, PyTorch with fastai, PyTorch with TabNet and PyTorch Lightning against simple baselines (train-mean forecast, linear regression, Ridge), using RMSE, MAE, R², out-of-sample R² and directional accuracy. We also report expected next-week returns for MSFT, IBM and GOOGL.

**Main finding:** weekly returns are almost unpredictable from these features. All models land within about 0.0007 RMSE of the train-mean forecast (0.0363 to 0.0370, excluding the overfit default XGBoost), so no model is clearly better than the others.

## Team
- Member 1: pipeline, models, methodology
- Member 2: EDA, introduction, README
- Member 3: results, comparison, conclusion, slides

## Data
Daily adjusted closing prices from Yahoo Finance (yfinance), January 2010 to October 2026, for MSFT (target), IBM, GOOGL, USD/JPY, GBP/USD, S&P 500, Dow Jones and VIX. After aligning to MSFT trading days (forward-fill of up to 3 days, never back-fill) there are 4,230 daily rows, resampled to 876 weekly (Friday) returns. Returns are used instead of prices because prices are non-stationary (ADF p = 0.993) while returns are stationary (p < 0.0001). Features are the current and 4 previous weekly returns of all 8 series (40 features); a second dataset adds 5 technical indicators for MSFT (45 features). The target is MSFT's return in the following week.

## Repository structure
- `data/` raw and processed datasets
- `notebooks/` 01 extraction, 02 preprocessing, 03 EDA, 04 baselines, 05 XGBoost, 06 Keras, 07 fastai, 08 TabNet, 09 Lightning, 10 comparison, 11 expected return, 12 all frameworks on all stocks
- `src/` `utils.py` (chronological split and shared metrics), `make_report.py`, `make_ppt.py`
- `results/` metrics and expected-return CSVs, ratings
- `figures/` plots used in the report
- `report/` written report
- `presentation/` slides

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Notebooks 01-05, 10 and 11 run locally. Notebooks 06-09 and 12 were run in Google Colab, and their packages (TensorFlow/Keras, fastai, pytorch-tabnet, lightning) are not in `requirements.txt`. To rerun them, upload `src/utils.py` and the CSVs from `data/` to Colab.

## How to reproduce
Run the notebooks in numeric order. 01 writes `data/raw_close_prices.csv`; 02 writes the two feature datasets; 03 writes EDA figures; 04-09 write one results CSV each (`results/*_results.csv`) and figures; 10 writes `results/final_comparison_*.csv` and the comparison chart; 11 and 12 write the expected-return files. Then run `python src/make_report.py` and `python src/make_ppt.py`.

## Method summary
- Chronological 80/20 split (train to June 2023, test June 2023 to October 2026), no shuffling.
- Scalers and target statistics are fit on training data only; the target is shifted so no feature uses future information.
- Tuning uses `TimeSeriesSplit` on the training set only. Neural networks use the last 15% of training weeks as a chronological validation set for early stopping, and predictions are averaged over 3 seeds.

## Key results (test period, 40 base features)

| Model | RMSE | MAE | R² | R²_OOS | Dir. acc. |
|---|---|---|---|---|---|
| fastai tabular | 0.0363 | 0.0261 | 0.0218 | 0.0230 | 0.509 |
| PyTorch Lightning MLP | 0.0363 | 0.0260 | 0.0217 | 0.0229 | 0.474 |
| Linear regression | 0.0364 | 0.0265 | 0.0167 | 0.0179 | 0.451 |
| Keras MLP | 0.0366 | 0.0261 | 0.0071 | 0.0083 | 0.526 |
| Ridge | 0.0366 | 0.0260 | 0.0060 | 0.0072 | 0.509 |
| XGBoost (tuned) | 0.0366 | 0.0261 | 0.0059 | 0.0071 | 0.543 |
| Train mean (baseline) | 0.0367 | 0.0260 | -0.0012 | 0.0000 | 0.520 |
| TabNet | 0.0370 | 0.0260 | -0.0175 | -0.0163 | 0.503 |
| XGBoost (default) | 0.0381 | 0.0278 | -0.0808 | -0.0795 | 0.491 |

Differences between models are within noise on 175 test weeks, and no directional accuracy is meaningfully above the 52% obtained by always predicting "up". Default XGBoost overfits; shallow, regularized trees fix this. Technical indicators did not help consistently. Expected next-week returns for MSFT, IBM and GOOGL are close to each stock's historical mean, and the models' R²_OOS values (zero or negative for IBM and GOOGL) mean they should not be read as reliable forecasts.

## Known limitations
- Lightning Flash could not be installed (it requires numpy<1.24, which does not build on current Python), so plain PyTorch Lightning, the library Flash is built on, was used.
- One chronological split; the test period contains unusually large weeks (for example +21.8% on 2026-07-31).
- No fundamentals or news features; small samples for deep models (about 700 training weeks).
- Annualized figures (weekly x 52) are rough scalings, not forecasts of yearly returns.
