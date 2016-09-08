# CharttingTwitterBot

A fun toy example of using ML to find fake technical patterns in stock market data. 

Requirements are the following
    1. tensorflow
    2. sklear
    3.  numpy
    4.  pandas
    5.  TwitterAPI

You must enter in your twitter credentials in twitter_api.py


Make sure you have the file path setup(or change the root):

1. plots
    1. results
        1. posted_plots
        1. prediction
        1. train
    2. current (most recent prices to make predictions on)
    3. random (random snapshots for historical training data)
    4. template (where the random head and shoulders are stored)



You will want to add more tickers to the main.py
ticker list. ~30 should be a good start, they must
be pulled from yahoo.
