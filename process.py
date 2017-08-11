
# This script generates csv files in the "data" directory with processed information about each stock. For each stock, it makes a new document with the same name in the "data" directory, with the format:  date, open, high, low, close, volume, sma5, sma10, sma20, sma50, sma100, sma200 . They are all given in percentages, except for the volume and date, which are left as they are. None of the data is normalized.


import os
from os import path
import csv

def get_percentage(new_price, old_price):
    """
    Returns the percentage increase/decrease of the new price new_price based on the old price old_price.
    """
    diff = new_price - old_price

    return (diff / old_price) * 100

def calculate_moving_averages(closing_prices, last_closing_price):
    """
    Calculates the SMA 5, 10, 20, 50, 100 and 200 with a list of all the closing prices.
    """
    moving_averages = []
    moving_averages_perc = []

    # calculating moving averages
    sma5 = sum(closing_prices[0:5]) / 5.0
    sma10 = sum(closing_prices[0:10]) / 10.0
    sma20 = sum(closing_prices[0:20]) / 20.0
    sma50 = sum(closing_prices[0:50]) / 50.0
    sma100 = sum(closing_prices[0:100]) / 100.0
    sma200 = sum(closing_prices[0:200]) / 200.0

    moving_averages.extend([sma5, sma10, sma20, sma50, sma100, sma200])

    for sma in moving_averages:
        perc = get_percentage(sma, last_closing_price)
        moving_averages_perc.insert(0, perc)

    return moving_averages, moving_averages_perc




# get the names of all csv data files
files = os.listdir(path.join(os.getcwd(), "raw_data/daily/"))
csv_files = [f for f in files if f.endswith(".csv")]

try:
    os.mkdir("data")
except OSError:
    print "Data directory already exists. Overwriting..."

# remember that the format of the csv files are "date,time,open,high,low,close,volume". We want to focus on the open, high, low, close, and volume.

# From the data, we want to obtain the percentage increases (based on the previous day's closing price) as well as the simple moving averages for 5, 10, 20, 50, 100 and 200 days. This should give our learning algorithm more features to work with.


for doc in csv_files:

    print "Processing", doc, "..."

    # open original csv file to read from
    fh = open(path.join("raw_data/daily", doc), "r")
    csv_reader = csv.reader(fh, delimiter=',')

    # open new csv file to write processed data to
    wh = open(path.join("data", doc), "w")
    csv_writer = csv.writer(wh, delimiter=',')

    # will repeatedly get updated with the last closing price, to calculate percentage increases and decreases.
    last_closing_price = None
    last_closing_price_percentage = None
    last_open_price_percentage = None

    # will get filled with the previous 200 closing prices
    closing_prices = []

    # row to write is outside so that subsequent loops can access it
    row_to_write = []

    for row in csv_reader:

        # getting necessary variables
        date = row[0]
        price_open = float(row[2])
        price_high = float(row[3])
        price_low = float(row[4])
        price_close = float(row[5])
        volume = float(row[6])

        # if this is the first closing price, then skip and update
        # if we don't have enough data to calculate the simple moving averages yet, then skip and update
        if len(closing_prices) < 200 or last_closing_price == None:
            closing_prices.insert(0, price_close)
            last_closing_price = price_close

            # the first time around, this will be zero, but it will fix itself afterwards because of the "< 200" condition
            last_closing_price_percentage = get_percentage(price_close, last_closing_price)
            last_open_price_percentage = get_percentage(price_open, last_closing_price)
            continue


        # calculating open price change as percentage
        price_open_perc = get_percentage(price_open, last_closing_price)
        #price_high_perc = get_percentage(price_high, last_closing_price)
        #price_low_perc = get_percentage(price_low, last_closing_price)
        price_close_perc = get_percentage(price_close, last_closing_price)

        smas, smas_perc = calculate_moving_averages(closing_prices, last_closing_price)



        # if the percentage is positive, then we write a "1" to the previous row, or a "0" if negative (WHICH AT THIS POINT HAS NOT YET BEEN WRITTEN).
        # the percentage profit is the desired profit to be made from buying at the OPEN PRICE of the day and then selling the NEXT DAY at the CLOSING PRICE.
        percentage_profit = 0.1
        gain_past_day = last_closing_price_percentage - last_open_price_percentage
        # calculating profit made in past day
        profit = (1 + (1 * (gain_past_day  / 100)))
        # calculating profit made next day at sell
        profit = profit + (profit * (price_close_perc / 100))
        # making it a truth value
        profit = profit >= 1 + (percentage_profit / 100)

        if profit and len(row_to_write) > 0:
            row_to_write.append(1)
        elif len(row_to_write) > 0:
            row_to_write.append(0)

        # once the previous row has been written to file, clear it all and begin filling it up again with the new data
        csv_writer.writerow(row_to_write)
        del row_to_write[:]

        # begin filling up again
        row_to_write.append(date)

        # adding the open percentage, and the volume. In real life, we will only have the open price available to us to make the buy/sell decision
        row_to_write.extend([price_open_perc, volume])

        # adding the percentages of sma5, sma10, sma20, sma50, sma100, sma200
        row_to_write.extend(smas_perc)

        # updating variables
        last_closing_price = price_close
        last_closing_price_percentage = price_close_perc # should always be able to do this since last_closing_price variable is guaranteed here
        last_open_price_percentage = price_open_perc

        closing_prices.insert(0, price_close)
        closing_prices.pop()

    print doc, "written successfully."
