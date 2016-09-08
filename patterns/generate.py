import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import pandas.io.data as web
from config import ROOT_PATH


plt.matplotlib.use('Agg')
np.random.seed(98)
# Mater DIR to save all plots
plt.ioff()


# Head and shoulders example to start
def generate_skeleton(type = 'hs'):
    """
    Build a basic chart pattern template.

    >>> generate_skeleton(type = 'hs')[0].plot()
    """
    # TODO: randomized lengths between each leg
    # extra data points
    # noise adn then smoothing
    slope = np.random.randint(-50, 50)

    # idx = np.arange(0, length)
    ratio = np.random.randint(60, 90) / 100.0

    # e3 = head and e1/e5 are the shoulders
    e0, e1, e2, e3, e4, e5, e6 = [100] * 7

    e1 *= np.random.randn()/10/3 + ratio
    e5 *= np.random.randn()/10/3 + ratio

    e2 = e1 * (np.random.randn()/10/4 + ratio)
    e4 = e5 * (np.random.randn()/10/4 + ratio)

    e0 = e2
    e6 = e4

    t = pd.Series([e0, e1, e2, e3, e4, e5, e6])
    # apply slope
    res = t + np.cumsum([x*slope/10/7 for x in np.arange(0, 7)])
    return res, ratio, slope


def stretch_skeleton(skeleton):
    """
    Take a raw pattern and fill in data points
    >>> stretch_skeleton(generate_skeleton(type = 'hs')[0]).interpolate().plot()
    """
    total_length = 100

    s = pd.Series(index=np.arange(0, total_length))
    i = 0
    num_moves = len(skeleton)
    # Stretch the TS
    for n, v in skeleton.iteritems():
        s.ix[i] = v
        i += int(total_length/num_moves+1)

    return s


def apply_random_walks(s):
    """
    >>> sk = stretch_skeleton(generate_skeleton(type = 'hs')[0])
    >>> apply_random_walks(sk).plot()
    """
    total_length = 100
    # Add white noise
    anchor_points = s.dropna()
    num_moves = len(anchor_points)
    _a = 0
    _jump_len = total_length / num_moves
    for n, v in s.iteritems():
        if _a == 0 or n == 0:
            _a += 1
            continue

        if n in anchor_points.index:
            _a += 1
            continue

        if n > max(anchor_points.index):
            s.iloc[n] = np.random.normal(0, 1) + s.iloc[n-1]
            continue

        move_size = anchor_points.iloc[_a] - anchor_points.iloc[_a-1]
        _m = np.random.normal(move_size/_jump_len, abs(move_size/_jump_len))
        s.iloc[n] = _m + s.iloc[n-1]
    return s


def gen_pipe(save=True):
    """ Build a save the results for a single chart. """
    skeleton, ratio, slope = generate_skeleton()
    s = stretch_skeleton(skeleton)
    res = apply_random_walks(s)

    if save:
        # fig = plt.figure()
        fig, ax = plt.subplots(1,1)
        fig.patch.set_visible(False)
        ax.axis('off')

        plt.plot(res.dropna().values, color='black', linewidth=3)
        id = 'hs_' + str(ratio) + '_' + str(slope)
        plt.savefig("{}template/target_plots_{}.jpg".format(ROOT_PATH, id), tight_layout=True)
        plt.close(fig)
    else:
        res


def build_smoothed_snapshots(prices, name, forward= True, save_folder='random', win=3):
    """
    Loop through a single security and build a bunch of random snapshots of the data.

    :param prices:
    :return:

    Usage:
    >>> prices =
    >>> name = 'F'
    """
    idx = prices.index
    for n in [20, 50, 100]:
        step = n / 3

        if forward:
            steps = np.arange(0, len(idx), step)
        else:
            steps = np.arange(len(idx), len(idx)-n, -step)

        for i in steps:
            if forward:
                _p = prices.iloc[i:i+n]
            else:
                _p = prices.iloc[i-n:-1]
            _p_avg = _p.rolling(win, min_periods=2).mean().reset_index(drop=True)
            fig, ax = plt.subplots(1,1)
            plt.plot(_p_avg.dropna().values, color='black', linewidth=3)
            fig.patch.set_visible(False)
            ax.axis('off')
            id = name.replace('usd',"") + '_' + str(i) + '_' + str(i+n)
            plt.savefig("{}/{}/random_plots_{}.jpg".format(ROOT_PATH,
                                                           save_folder, id))
            plt.xlim((0, len(_p)))
            plt.close(fig)


def build_plots_for_prediction(tickers):
    """

    :param tickers:
    :return:
    >>> tickers = ['F','GOOG']
    """
    prices = web.DataReader(tickers, 'yahoo', dt.datetime(2014, 1, 1), dt.date.today())
    for ticker in tickers:
        build_smoothed_snapshots(prices.ix[:,:,ticker]['Adj Close'],
                                 ticker, forward=False, save_folder='current')

def build_plots_for_training(tickers):
    """
    :param tickers:
    :return:
    >>> tickers = ['F','GOOG']
    """
    prices = web.DataReader(tickers, 'yahoo', dt.datetime(1990, 1, 1), dt.datetime(2014, 1, 1))
    for ticker in tickers:
        build_smoothed_snapshots(prices.ix[:,:,ticker]['Adj Close'],
                                 ticker, forward=True, save_folder='random')


if __name__ == "__main__1":

    # plt.ion()
    plt.ioff()
    [gen_pipe(save=True) for x in np.arange(1000)]

    # Example:
    skeleton, ratio, slope = generate_skeleton()
    s = stretch_skeleton(skeleton)
    s.interpolate().plot()
    apply_random_walks(s).plot()


    # Build plots useing for training models and then plots for prediction using the trained model.
    tickers = ['F', 'AAPL', 'MSFT']
    build_plots_for_training(tickers)
    build_plots_for_prediction(tickers)

