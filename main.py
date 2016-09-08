#!/usr/bin/python
import os

from patterns.annotate import make_candel_plot
from patterns.generate import build_plots_for_prediction, build_plots_for_training
from patterns.labels import get_hs_text
from patterns.model import get_best_chart, build_predictions
from config import ROOT_PATH
from twitter_api import tweet_chart

"""
Make sure you have the file path setup:

-plots
    -results
        -posted_plots
        -prediction
        -train
    -current
    -template
"""

if __name__ == "__main__":
    # Clear all tmp folders out to make sure I have a clean run
    SAVE_RESULTS_PATH = ROOT_PATH + "results/posted_plots/"

    del_paths = ['results/prediction/',
                 'current/']
    for del_path in del_paths:
        filelist = [f for f in os.listdir(ROOT_PATH+ del_path) if f.endswith(".jpg")]
        for f in filelist:
            os.remove(ROOT_PATH + del_path + f)

    # Update price folder pics

    tickers = ['F', 'GOOG', 'MSFT','AAPL','VRX','GS','C']
    build_plots_for_training(tickers) # this take a while and does not need to be performed each run, only once
    tickers = ['F', 'GOOG', 'MSFT','AAPL','VRX','GS','C','CSCO','CBS','XOM','V','T','AIG','AMZN','ALL','ADS','LNT']
    build_plots_for_prediction(tickers)

    # Run through both models
    build_predictions()

    # Determine the best chart to send out
    ticker, hs_prices = get_best_chart(force_models_agree=False)

    if hs_prices.empty is False:
        # Make pretty, could add some trend lines

        file_name = make_candel_plot(hs_prices, sec=ticker, save_loc=SAVE_RESULTS_PATH)

        # Add text
        text = get_hs_text(ticker)

        # Set out tweet!!
        tweet_chart(file_name, text)