import pandas as pd
import numpy as py
import pypfopt as pypo
import os

CLOSE_PRISES_PATH = os.path.join('datasets','marketData','adj_close_prices.csv')
WEIGHTS_PATH = os.path.join('datasets','groundTruth')

def markowitzBenchmatk():
    dataframe = pd.read_csv(CLOSE_PRISES_PATH, sep = ",")

    date_column_name = dataframe.columns[0] 
    prices_df = dataframe.set_index(date_column_name)

    mu = pypo.expected_returns.mean_historical_return(prices_df)
    S = pypo.risk_models.sample_cov(prices_df)

    ef = pypo.EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    expected_return, volatility, sharpe = ef.portfolio_performance(verbose=True)

    weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weights'])

    weights_df.to_csv(os.path.join(WEIGHTS_PATH,'weights.csv'))

    return expected_return, volatility, sharpe


markowitzBenchmatk()