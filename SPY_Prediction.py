'''
Program that will take data on different Stocks and/or ETFs and will attempt
to use this data to make predictions on the future closing price of the SPY Sector
Spider S&P 500 ETF using a deep-learning regression model with Keras.
'''
from functools import reduce
import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt



def get_Stocks_and_ETFs_From_User():
    '''
    Function that will get a list of ETFs and/or stocks from the user that will
    be used as input data on the Machine Learning regression model

    return: dict containing the ticker for all desired stocks and ETFs. Stocks
            and ETFs are stored differently so that they can be more easily
            found in data files
    '''

    num_stocks = int(input("Please enter number of stocks to include:"))

    #accumulate list of stocks and ETFS
    stocks_and_ETFs = {}
    stocks = []
    for i in range(num_stocks):
        stock = input('Please enter ticker of stock:').lower()
        stocks.append(stock)

    #add to dict
    stocks_and_ETFs['Stocks'] = stocks

    num_ETFs = int(input("Please enter number of ETFs to include:"))
    ETFs = []
    for i in range(num_ETFs):
        ETF = input('Please enter ticker of ETF:').lower()
        ETFs.append(ETF)

    stocks_and_ETFs['ETFs'] = ETFs

    return stocks_and_ETFs



def merge_DataFrames(dataframe_list):
    '''
    Function that will merge all of the dataframes into one single dataframe
    with all of the data

    return: single pandas dataframe with all desired information
    '''


    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],
                                            how='inner'), dataframe_list)

    return df_merged



def get_Dataframe(a_type, ticker):
    '''
    Function that will return a dataframe with only the date and change in price
    for a stock on a particular date


    @a_type: tells whether ticker is stock or ETF
    @ticker: ticker of stock/ETF

    return: Dataframe with Date and Change in price column
    '''
    ticker = ticker.strip("'")

    #build filepath string
    filepath_str = 'Stock_csvs/Stock_and_ETF_Data/' + a_type + '/' \
                                        + ticker + '.us.txt'

    #handle incorrect user input
    try:
        df = read_csv(filepath_str, usecols=['Date', 'Open', 'Close'])
    except FileNotFoundError:
        print("File for "+ ticker.upper() +" does not exist")
        return pd.DataFrame()

    #calculate change column
    df[ticker.upper() + ' Change'] = df['Close'] - df['Open']

    df_add = df[['Date', ticker.upper() + ' Change']]

    return df_add




def get_stock_and_ETF_data(user_input):
    '''
    Function that will ensure that the program has data on all of the stocks and
    ETFs that the user has created. This function will read in the data from
    each stock/ETFs respective CSV.

    @user_input: dict of tickers of desired Stocks and ETFs to be used in ML
                 model

    return: if valid input: list of pandas dataframes with all of the data from
    all of desired stocks
            if NOT valid input: empty list
    '''
    dataframe_list = []

    #/Users/ScottEberle/Desktop/Projects/TF_and_Keras/Stock_csvs/Stock_and_ETF_Data/Stocks/a.us.txt

    for asset_type in user_input.keys(): #stocks and ETFs
        for ticker in user_input[asset_type]:   #each individual one
            financial_data = get_Dataframe(asset_type, ticker)
            if financial_data.empty:
                return []
            #otherwise append to the end
            dataframe_list.append(financial_data)

    #merge all other dataframes into one
    # TODO: Edit this so that it asks for user input on what stock to predict

    merged_Frame = merge_DataFrames(dataframe_list)


    return merged_Frame




def format_Data_For_Model(stock_ETF_data):
    '''
    Function that will format the data into numpy ndarrays. The function will
    put all data from the desired stocks and ETFs into an input array X and then
    put data from the SPY ETF into a corresponding ndarray y. y, however, will
    be offset by one day because we want to use the behavior of these stocks and
    ETFs to predict the change in the SPY for the NEXT day.

    @stock_ETF_data: List containing two pandas data frames. the first is for
                     input data and the second is for output data

    return: X (ndarray) which is input to the ML regression model
            y (ndarray) which is the output for the ML regression model
    '''

    #get which is the desired predicted price change from the user
    desired_output = input( "Please enter ticker for stock that is to be predicted").strip("'")

    #format for columns
    desired_output += ' Change'

    #trim so just input data
    input_data = stock_ETF_data.drop(['Date', desired_output], axis=1)

    #to numpy array
    X = input_data.to_numpy()

    #clip last one so dimensions match
    X = X[:-1, :]

    #just desired output
    output_data = stock_ETF_data[[desired_output]].to_numpy()
    #delete first row so that today's stock and ETF readings are matched w
    #TOMORROW's SPY change
    y = output_data[1:,:]


    return X, y, desired_output



def build_Keras_model(number_of_features=10):
    #TODO: implement customizatino of the size and shape of the model
    '''
    Function that will build a sequantial Keras regression model

    return: The created Sequantial model
    '''

    model = Sequential()
    #input layer
    model.add(Dense(9, input_dim=number_of_features, kernel_initializer='normal', activation='relu'))
    #hidden layer
    model.add(Dense(5, kernel_initializer='normal', activation='relu'))
    #dropout for regularization
    model.add(Dropout(0.4))
    #hidden layer
    model.add(Dense(3, kernel_initializer='normal', activation='relu'))
    #output layer
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    return model



def get_Prediction_Data(price_change_data, output_label):
    '''
    Function that will take in a dataframe with all of the data and then
    determine what inputs would be needed from the user to make a prediction
    based on the sequantial model generated by the program. This function will
    then get those inputs from the user and return them in a format that can be
    used to predict an outcome based on this user input.

    @price_change_data: dataframe containing all of the data on the change in
                        prices of particular stocks/ETFs

    return: ndarray that will serve as input to keras model
    '''

    #get rid of date and spy change columns
    just_input = price_change_data.drop(columns = ['Date', output_label])

    #get all columns
    columns = list(just_input)

    #accumulate input from user
    input_data = []

    for column_name in columns:
        value = float(input("Please enter real change of price for " +
                                        column_name + ":"))
        input_data.append(value)


    #convert to numpy array
    return np.asarray(input_data)



def main():

    #initially, we have no data
    model_data = pd.DataFrame()

    #while we have no data
    while model_data.empty:

        #determine which stocks/ETFs should the ml model use
        stock_and_ETF_dict = get_Stocks_and_ETFs_From_User()

        if len(stock_and_ETF_dict['Stocks']) != 0 or len(
                                            stock_and_ETF_dict['ETFs']) != 0:
        #check that we have the data on these stocks if yes read in the data, otherwise get new input
            model_data = get_stock_and_ETF_data(stock_and_ETF_dict)




    #now we have stock and ETF data
    #merge all data into one dataframe need to find out how to offset date by 1
    X, y, output_title = format_Data_For_Model(model_data)

    # TODO: DYNAMICALLY CREATE SEQUANTIAL MODEL?

    #standardize data using pipeline
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn = build_Keras_model,
        number_of_features=X.shape[1], epochs=200, batch_size=5, verbose=0)))


    #cross validate to check accuracy of predictions
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=5)
    results = cross_val_score(pipeline, X, y, cv=kfold)
    print("Wider: %.2f (%.2f) Mean Squared Error" % (results.mean(), results.std()))

    #actually fit the data so that it can be used to make predictions
    pipeline.fit(X,y)


    #user input for current data
    data_to_predict = get_Prediction_Data(model_data, output_title)

    print()
    print("___________________________________________________________________")
    print("Prediction for:", output_title, pipeline.predict(data_to_predict.reshape(1, -1)))



if __name__ == '__main__':
    main()
