'''
Program that will take data on different Stocks and/or ETFs and will attempt
to use this data to make predictions on the closing price of the SPY Sector
Spider S&P 500 ETF using a deep-learning regression model with Keras.
'''
from functools import reduce
import numpy as np
import pandas as pd
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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
    all of desired stocks along with dataframe from SPY data
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
    merged_Frame = merge_DataFrames(dataframe_list + [get_Dataframe('ETFs', 'spy')])


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

    #trim so just input data
    input_data = stock_ETF_data.drop(['Date', 'SPY Change'], axis=1)

    #to numpy array
    X = input_data.to_numpy()

    #clip last one so dimensions match
    X = X[:-1, :]

    #just desired output
    output_data = stock_ETF_data[['SPY Change']].to_numpy()
    #delete first row so that today's stock and ETF readings are matched w
    #TOMORROW's SPY change
    y = output_data[1:,:]

    print()
    print("Output Data")
    print("---------------------------------------------------")
    print(X.shape, y.shape)
    print("---------------------------------------------------")




    return np.ndarray([]), np.ndarray([])



def build_Keras_model():
    #TODO: implement customizatino of the size and shape of the model
    '''
    Function that will build a sequantial Keras regression model

    return: The created Sequantial model
    '''

    model = Sequential()





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
    X, y = format_Data_For_Model(model_data)

    #create function for building keras sequential model - let user input how bit to make it -> do later
    build_Keras_model()

    #standardize data
    estimators = []
    estimators.append(('standardize', StandardScaler()))
    estimators.append(('mlp', KerasRegressor(build_fn = build_Keras_model, epochs=100, batch_size=5, verbose=0)))


    #cross validate to check accuracy of predictions
    pipeline = Pipeline(estimators)
    kfold = KFold(n_splits=10)
    results = cross_val_score(pipeline, X, y, cv=kfold)
    print("Wider: %.2f (%.2f) MSE" % (results.mean(), results.std()))






if __name__ == '__main__':
    main()
