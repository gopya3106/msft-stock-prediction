"""Run from the project root:  python src/make_report.py
Builds report/report.docx with full text, your figures and your results tables."""
import os
import pandas as pd
from docx import Document
from docx.shared import Inches

doc = Document()
n_fig = [0]
n_tab = [0]


def H(t, lvl=1):
    doc.add_heading(t, lvl)


def P(t):
    doc.add_paragraph(t)


def fig(name, caption):
    path = f"figures/{name}"
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(6))
    else:
        P(f"[missing figure: {path}]")
    n_fig[0] += 1
    P(f"Figure {n_fig[0]}: {caption}")


def tbl(csv, title, index_col=0, digits=4):
    path = f"results/{csv}"
    if not os.path.exists(path):
        P(f"[missing table: {path}]")
        return
    df = pd.read_csv(path, index_col=index_col).reset_index()
    n_tab[0] += 1
    P(f"Table {n_tab[0]}: {title}")
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for j, c in enumerate(df.columns):
        t.rows[0].cells[j].text = str(c)
    for row in df.itertuples(index=False):
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = f"{v:.{digits}f}" if isinstance(v, float) else str(v)


doc.add_heading("Stock Price Prediction using Supervised Machine Learning", 0)
P("Team: [names]    Course: [course]    Date: [date]")
P("Drafted with AI assistance and reviewed by the team, as permitted by the evaluator.")

# ---------------------------------------------------------------- 1
H("1. Introduction")
P("Predicting stock prices is one of the hardest problems in finance. Prices respond to company "
  "results, news, monetary policy and the behaviour of other investors, and much of this information "
  "is already reflected in the price, so what remains to be predicted is mostly noise. Machine "
  "learning methods learn patterns from historical data without a fixed formula, which makes them a "
  "natural candidate for this task.")
P("In this project we predict the weekly return of Microsoft (MSFT) stock. We model returns rather "
  "than prices because prices trend and are non-stationary, while returns fluctuate around a stable "
  "mean and can be modeled with standard regression. A weekly horizon reduces day-to-day noise "
  "while still giving several hundred observations from 2010 onward.")
P("Of the feature families used in practice (correlated assets, technical indicators and fundamental "
  "analysis), we focus on correlated assets, as the project brief specifies. A company depends on its "
  "competitors, the broader market, currencies and risk sentiment, so MSFT's return may be related "
  "to the returns of IBM, Alphabet (GOOGL), USD/JPY, GBP/USD, the S&P 500, the Dow Jones index and "
  "the VIX. As an extension we also test whether five technical indicators of MSFT add information.")
P("We compare five frameworks (XGBoost, Keras, PyTorch with fastai, PyTorch with TabNet and PyTorch "
  "Lightning) against three baselines (train-mean forecast, linear regression and Ridge). The "
  "comparison covers accuracy (RMSE, MAE, R², out-of-sample R² and directional accuracy) and three "
  "qualitative criteria: simplicity, transparency and efficacy. We also report expected next-week "
  "returns for MSFT, IBM and GOOGL.")

# ---------------------------------------------------------------- 2
H("2. Methodology")
H("2.1 Data", 2)
P("Daily adjusted closing prices were downloaded from Yahoo Finance with the yfinance library for "
  "MSFT (target), IBM, GOOGL, USD/JPY, GBP/USD, the S&P 500, the Dow Jones index and the VIX, from "
  "January 2010 to October 2026. The raw file has 4,384 daily rows. MSFT's trading days were used as "
  "the master calendar, which removed days when only the currency markets were open. Remaining gaps "
  "of up to three days were forward-filled, because a forward-fill uses only information that was "
  "available at the time; back-filling would use future data. This left 4,230 clean daily rows.")
P("Prices were resampled to the last trading day of each week (Friday-ending) and converted to "
  "weekly percentage returns, giving 876 weekly observations. An incomplete final week is dropped.")
H("2.2 Target and features", 2)
P("The target is MSFT's return in the week after the observation date. The base feature set contains "
  "the current week's return and the four previous weeks' returns for all eight series, 40 features "
  "in total, giving 871 usable samples. A second dataset adds five MSFT technical indicators "
  "(ratios of price to its 4-week and 12-week moving averages and to a 12-week exponential moving "
  "average, 4-week momentum and 4-week volatility), giving 45 features and 865 samples.")
H("2.3 Leakage controls and validation", 2)
P("Several measures prevent look-ahead leakage. The target is shifted by one week so that all "
  "features are known at the close of the Friday on which the prediction is made. The data are split "
  "chronologically, with the first 80% of weeks for training (February 2010 to June 2023, 696 weeks) "
  "and the last 20% for testing (June 2023 to October 2026, 175 weeks), with no shuffling. Scalers "
  "and target statistics are fit on training data only. Hyperparameters are tuned with "
  "TimeSeriesSplit, which always validates on weeks later than the ones it trains on. For the neural "
  "networks, the last 15% of the training weeks form a chronological validation set for early "
  "stopping.")
H("2.4 Models", 2)
P("Baselines: the train-mean forecast (predicts the average training return), a zero forecast, "
  "ordinary linear regression and Ridge regression. Ridge's penalty was tuned on a logarithmic grid "
  "with TimeSeriesSplit and selected the largest value in the range (alpha about 2,976), meaning "
  "heavy shrinkage.")
P("XGBoost: a default model and a tuned model. The tuned model was chosen by randomized search "
  "(15 combinations, 3 time-ordered folds) over tree depth, number of trees, learning rate, row and "
  "column subsampling, minimum child weight and L2 regularization. The selected settings used "
  "shallow trees (depth 3), 100 trees, learning rate 0.02, subsampling of 0.6 and strong L2 "
  "regularization (50).")
P("Keras: a feed-forward network with two hidden layers, ReLU activations, dropout and L2 "
  "regularization, trained with Adam and early stopping. Three sizes were compared on the "
  "validation set and the medium network (32 and 16 units) was chosen. The target was standardized "
  "and predictions converted back to return units.")
P("fastai: a tabular learner with layers of 32 and 16 units, dropout of 0.3, weight decay of 0.1 and "
  "the one-cycle schedule, with early stopping on validation loss. Explicit chronological splits "
  "replaced fastai's default random split.")
P("TabNet: pytorch-tabnet's regressor with 3 decision steps, 8-dimensional decision and attention "
  "layers and a sparsity penalty, trained with early stopping. Its attention masks provide "
  "feature importance.")
P("PyTorch Lightning: the brief asks for Lightning Flash. Flash is no longer maintained and its "
  "installation failed because it pins numpy below 1.24, which cannot be built on the current Python "
  "version (see the figure below). We therefore used plain PyTorch Lightning, the library on which "
  "Flash is built, with the same network design as Keras and fastai.")
P("All neural models were trained with three random seeds and their predictions averaged, because "
  "results on noisy data vary considerably between seeds. Keras, fastai, TabNet and Lightning "
  "(notebooks 06 to 09 and 12) were run in Google Colab because of package installation conflicts "
  "on the local machine; the rest ran locally.")
fig("flash_install_error.png", "Installation failure of Lightning Flash (numpy<1.24 does not build).")
H("2.5 Evaluation metrics", 2)
P("RMSE and MAE measure the typical size of prediction errors in return units. R² compares the "
  "model with predicting the test-set mean; because this is not available when forecasting, we "
  "also report out-of-sample R² (R²_OOS), which compares the model with the train-mean forecast "
  "and is therefore the more honest measure. Positive values mean the model beats the train mean, "
  "negative values mean it does worse. Directional accuracy is the share of weeks in which the sign "
  "of the return was predicted correctly; it must be compared with the share of up weeks in the test "
  "set (52% for MSFT), since always predicting 'up' achieves that without any skill.")

# ---------------------------------------------------------------- 3
H("3. Exploratory Data Analysis")
P("MSFT's weekly return has a mean of 0.41% and a standard deviation of 3.3%. Its distribution is "
  "slightly right-skewed (skewness 0.34) with fat tails (excess kurtosis 3.57), and the "
  "Jarque-Bera test rejects normality decisively (p about 4e-104). The Q-Q plot shows the same: the "
  "middle follows the normal line while the extreme weeks lie far outside it. The largest moves are "
  "real events: the COVID-19 crash in February and March 2020, earnings reactions in 2013 and 2015, "
  "and large weeks in 2026. Fat tails matter because RMSE is dominated by a few extreme weeks.")
fig("01_weekly_prices.png", "Weekly closing prices of all eight series.")
fig("02_msft_return_dist.png", "MSFT weekly return distribution and Q-Q plot against the normal.")
P("The Augmented Dickey-Fuller test supports modeling returns: MSFT's price has a p-value of 0.993 "
  "(non-stationary), whereas weekly returns have a statistic of -32.3 and a p-value below 0.0001 "
  "(stationary).")
P("Same-week correlations with MSFT are strong for the S&P 500 (0.67), Dow Jones (0.59) and GOOGL "
  "(0.55), and negative for the VIX (-0.49). The S&P 500 and Dow are almost collinear (0.95), which "
  "affects interpretation of linear coefficients. However, same-week correlation cannot be used for "
  "prediction, because those values are not known before the week they describe.")
fig("03_corr_heatmap.png", "Correlation of same-week returns.")
P("The correlations that matter are those between lagged features and next week's return. They are "
  "all small: the strongest is -0.099 (GOOGL's current-week return), and the top features are mostly "
  "negative, hinting at slight short-term mean reversion. With 871 samples, only correlations beyond "
  "about 0.066 are distinguishable from noise, and with 40 features about two would pass by chance, "
  "so we expect very little predictive power.")
fig("04_feature_target_corr.png", "Fifteen features most correlated with next-week MSFT return.")
fig("05_rolling_vol.png", "Rolling 26-week annualized volatility of MSFT.")
P("Volatility changes over time and clusters in turbulent periods. The ACF and PACF of returns show "
  "almost no autocorrelation, which is consistent with weak predictability, while squared returns "
  "show more structure (volatility clustering). Returns are therefore hard to predict, even though "
  "their risk is somewhat predictable.")
fig("06_acf_pacf.png", "ACF and PACF of MSFT returns and squared returns.")

# ---------------------------------------------------------------- 4
H("4. Results")
P("Table 1 reports test-period results with the base features, sorted by RMSE. The train-mean "
  "forecast has an RMSE of 0.0367. The best models (fastai and Lightning, 0.0363) improve on it by "
  "about 1%, and R²_OOS reaches only 0.023, so roughly 2% of the variance is explained at best. "
  "Excluding default XGBoost, the spread between best and worst RMSE is just 0.0007, which is within "
  "the noise expected from 175 test weeks, so we cannot name a clearly superior model.")
tbl("final_comparison_base.csv", "Test metrics, base features (sorted by RMSE).")
P("Default XGBoost overfits badly (R² = -0.081, worse than the mean), but shallow, strongly "
  "regularized trees found by time-ordered tuning bring it to the level of Ridge (RMSE 0.0366). "
  "TabNet, the largest architecture, is the only tuned model worse than the train mean (R²_OOS = "
  "-0.016), which is plausible for a flexible model with about 700 training samples.")
P("Directional accuracy ranges from 45% to 54% and none of it is meaningfully above the 52% obtained "
  "by always predicting 'up'. With 175 weeks, one standard error is about 3.8 percentage points. "
  "Linear regression has the second-best R²_OOS but only 45% directional accuracy, so smaller "
  "squared errors did not translate into direction skill.")
fig("19_model_comparison.png", "Out-of-sample R² and directional accuracy for all models.")
tbl("final_comparison_tech.csv", "Test metrics with technical indicators (sorted by RMSE).")
P("Technical indicators did not help consistently. R²_OOS fell for eight of the ten models, "
  "including fastai (-0.043) and Lightning (-0.031). TabNet is the only model that improved "
  "(+0.043), but its sign flip between datasets with otherwise identical settings points to "
  "instability rather than a genuine effect of the indicators.")
for name, cap in [("07_ridge_pred_vs_actual.png", "Ridge predictions versus actual returns."),
                  ("10_xgb_pred_vs_actual.png", "XGBoost predictions versus actual returns."),
                  ("12_keras_pred_vs_actual.png", "Keras predictions versus actual returns."),
                  ("13_fastai_pred_vs_actual.png", "fastai predictions versus actual returns."),
                  ("16_tabnet_pred_vs_actual.png", "TabNet predictions versus actual returns."),
                  ("18_lightning_pred_vs_actual.png", "Lightning predictions versus actual returns.")]:
    fig(name, cap)
P("All models produce an almost flat prediction line close to the mean, while actual returns "
  "fluctuate widely, including the +21.8% week of 31 July 2026. The models effectively predict "
  "the average return, which is what the near-zero R² values imply.")
fig("08_xgb_feature_importance.png", "XGBoost feature importance (tuned, base data).")
fig("09_xgb_shap.png", "SHAP summary for the tuned XGBoost model.")
fig("15_tabnet_importance.png", "TabNet attention-based feature importance.")
P("Importance is spread thinly over many features in both XGBoost and TabNet, consistent with the "
  "weak correlations found in the EDA.")
fig("11_keras_loss_curve.png", "Keras training and validation loss.")
fig("14_fastai_loss_curve.png", "fastai training and validation loss.")
fig("17_tabnet_loss_curve.png", "TabNet training curve.")
P("In every network, training loss keeps falling while validation loss stays flat or rises, and early "
  "stopping triggers within a few dozen epochs. This is the signature of fitting noise: there is "
  "little signal to learn.")

H("4.1 Expected returns", 2)
P("We refit each model on all available weeks and forecast the return for the week after the latest "
  "complete week. Table 3 gives the test-period metrics for MSFT, IBM and GOOGL for all six "
  "frameworks, and Table 4 gives the forecasts (as decimals, so 0.0075 is 0.75%) next to each "
  "stock's historical mean weekly return.")
tbl("multi_stock_metrics_all.csv", "Test metrics for MSFT, IBM and GOOGL.", index_col=[0, 1])
tbl("expected_returns_all_frameworks.csv",
    "Expected next-week return versus historical mean (decimals).", index_col=[0, 1], digits=5)
fig("20_expected_returns.png", "Ridge and XGBoost expected returns versus historical mean.")
P("For IBM and GOOGL, R²_OOS is zero or negative for every framework, so no model beats the "
  "train mean. The forecasts differ widely between models: for MSFT they range from 0.16% (TabNet) "
  "to 0.80% (Lightning) per week, and for IBM TabNet forecasts -0.46% while Keras forecasts +0.70%, "
  "against historical means of 0.43%, 0.21% and 0.44% for MSFT, IBM and GOOGL. When models with "
  "essentially no skill disagree this much, the differences mainly reflect model noise. The "
  "historical mean is the most defensible single estimate of expected return. Annualized figures "
  "(weekly x 52) are rough scalings and not forecasts of yearly return.")

# ---------------------------------------------------------------- 5
H("5. Comparison: Simplicity, Transparency and Efficacy")
P("Scores run from 1 (worst) to 5 (best) and reflect our experience in this project. Simplicity "
  "reflects code and setup effort and the installation problems we met; transparency reflects how "
  "well a prediction can be explained; efficacy reflects accuracy on the test set, bearing in mind "
  "that the differences between models are within noise.")
ratings = [
    ("Linear regression", 5, "Scaler plus one fit", 5, "Coefficients (but collinear S&P/Dow)", 3,
     "RMSE 0.0364, best R²_OOS among simple models, but 45% direction accuracy"),
    ("Ridge", 5, "One extra tuned parameter", 5, "Coefficients, shrunk", 3,
     "RMSE 0.0366, R²_OOS 0.007"),
    ("XGBoost (tuned)", 4, "No scaling, but needs a hyperparameter search; import issues locally", 4,
     "Feature importance and SHAP", 3, "RMSE 0.0366; default version overfit (R² -0.081)"),
    ("Keras MLP", 3, "Scaling, target standardization, validation split, seeds; Colab", 2,
     "No built-in explanation", 3, "RMSE 0.0366, R²_OOS 0.008"),
    ("fastai tabular", 3, "High-level API but argument errors and recorder issues", 2,
     "No built-in explanation", 3, "RMSE 0.0363, best R²_OOS 0.023 (within noise)"),
    ("TabNet", 2, "Extra package, most settings, slowest to run", 3,
     "Attention masks, though importance is diffuse", 2,
     "R²_OOS -0.016 on base, flipped sign on tech: unstable"),
    ("PyTorch Lightning", 2, "Most boilerplate; Flash would not install", 2,
     "No built-in explanation", 3, "RMSE 0.0363, R²_OOS 0.023, 47% direction accuracy"),
]
t = doc.add_table(rows=1, cols=7)
t.style = "Light Grid Accent 1"
for j, c in enumerate(["Model", "Simplicity", "Why", "Transparency", "Why", "Efficacy", "Why"]):
    t.rows[0].cells[j].text = c
for r in ratings:
    cells = t.add_row().cells
    for j, v in enumerate(r):
        cells[j].text = str(v)
P("Linear models are the simplest and most transparent and perform as well as anything else, "
  "so for this problem added model complexity buys nothing. Among the frameworks, XGBoost offers the "
  "best balance of tuning effort and explanation tools, TabNet offers built-in interpretability "
  "but the least stable results, and Keras, fastai and Lightning are architecturally almost "
  "identical, which is why their results agree closely.")

# ---------------------------------------------------------------- 6
H("6. Conclusion and Summary")
P("We built a leakage-controlled pipeline to predict MSFT's next-week return from lagged returns of "
  "correlated assets and compared five frameworks against baselines. All models perform within "
  "about 0.0007 RMSE of the train-mean forecast, with out-of-sample R² between -0.016 and 0.023 for "
  "all tuned models, and directional accuracy indistinguishable from always predicting 'up'. Weekly "
  "returns are therefore almost unpredictable from these features, which is consistent with "
  "efficient markets and with the weak correlations, absent autocorrelation and training-versus-"
  "validation gap we observed. Default XGBoost overfit but was fixed by regularization; technical "
  "indicators did not help consistently; and expected-return forecasts for MSFT, IBM and GOOGL are "
  "close to historical means and should not be treated as reliable.")
P("Limitations: a single chronological split, a test period containing unusually large weeks, no "
  "fundamental or news information, small samples for the deep models, and Flash being unavailable. "
  "Future work could use rolling-window (walk-forward) validation, add fundamentals and news "
  "sentiment, predict volatility or direction directly, and test longer horizons.")

os.makedirs("report", exist_ok=True)
doc.save("report/report.docx")
print("saved report/report.docx")
