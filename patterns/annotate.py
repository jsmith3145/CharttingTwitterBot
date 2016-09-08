import copy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from matplotlib.finance import candlestick_ohlc

# http://matplotlib.org/examples/pylab_examples/finance_demo.html

def make_candel_plot(plotting_prices, sec, save_loc=None):
    prices = copy.copy(plotting_prices)
    s_date = str(plotting_prices.index[0].date())
    e_date = str(plotting_prices.index[-1].date())
    prices.index = [matplotlib.dates.date2num(i) for i in prices.index]
    mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
    alldays = DayLocator()              # minor ticks on the days
    weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
    dayFormatter = DateFormatter('%d')      # e.g., 12


    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.xaxis.set_major_locator(mondays)
    ax.xaxis.set_minor_locator(alldays)
    ax.xaxis.set_major_formatter(weekFormatter)

    _p = prices.dropna()[['Open','Close','High','Low']].reset_index()
    matplotlib.finance.candlestick_ochl(ax, _p.values, width=0.7)
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=70, horizontalalignment='right')
    fig.suptitle(sec)
    fig.tight_layout()

    if save_loc is not None:
        file_name = save_loc + sec + "_" + s_date + "_" + e_date + '.jpg'
        fig.savefig(file_name)
        plt.close()
    return file_name

if __name__ =="__main__":
    pass