# World Cup Predictions

REQUIREMENTS: 
Python 3.5+ //
Tensorflow //
Keras

The easiest way is probably to install Anaconda package and: 
pip install tensorflow //
pip install keras

This is a small project in python to try to predict results of group stage of World Cup and later maybe a winner. So far it looks alright since I don't have that much data (750 record is not enough to predict much). It had accuracy of around 60-65% so far based only on historical World Cup results from last 7 World Cups, national team results from last 10-15 games of each participant and past season WhoScored results for each player participating. Project uses Keras (Tensorflow), Selenium to scrape data, sklearn for one hot encoding.

Usage: python predict.py $path$ (something like C:/Users/john/Downloads/ etc.) 
