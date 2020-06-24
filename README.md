# Keras_Stock_Prediction
Repository that uses the Keras/TensorFlow Deep Learning library to make stock price predictions


It includes a python program that allows a user to select stocks and ETFs, and then historical data on these stocks and ETFs is used to train a multi-layered Keras Sequential Model, which is used to predict the future behavior of a stock or ETF specified by the user. Once the model has been trained, then the program allows the user to input current information on the stocks/ETFs that were used to train this model, and then the program uses the deep-learning model to make a prediction based on user input. 

The repository also includes a .sh file that allows a user to feed input to the python program automatically. If you run this shell script, which is also known as a heredoc, then all input required for the program will be sent using this file. The format of the heredoc is as follows:

python3 SPY_Prediction.py << EOF *tells terminal to run stock prediction program* \
1 *number of stocks that will be included*\
'MSFT' *ticker for every stock, each new stock on new line*\
10 *number of ETFs that will be included*\
'VGT' *ticker for every stock, each new ETF on new line*\
'RYH'\
'FNCL'\
'VDC'\
'VDE'\
'FREL'\
'VIS'\
'VAW'\
'VPU'\
'SPY'\
'SPY' *ticker of stock/ETF that you want to predict outcome for*\
2.7 *current information on each corresponding stock and ETF that was used as input*\
0.3\
-0.67\
0.018\
-0.03\
0.24\
1.663\
EOF *marks end of file*
