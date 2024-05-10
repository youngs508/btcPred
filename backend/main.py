
## in command prompt run: uvicorn main:app --reload
##  python3 -m uvicorn main:app
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
import re
from typing import Union
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sklearn.linear_model import LinearRegression

global_var = None
dataSamples = None
index = 0

app = FastAPI()
# Allow requests from all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["DNT", "User-Agent", "X-Requested-With", "If-Modified-Since", "Cache-Control", "Content-Type", "Range"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


class LinearRegressionRegularization:
    def __init__(self, max_iter=6,  learningRate=90, random_state=None):
        self.max_iter_ = max_iter
        self.alpha = learningRate
        self.random_state_ = random_state
        self.w_ = np.random.randint(0, 1, 17)
        self.w0 = 0
        
    def _split(self, X, y, ratio, random_state):
        header = X.columns
        # remove header for shuffling 
        x_data = X.values

        random.seed(random_state)
        train_ratio = ratio
        test_ratio = 1 - train_ratio
        
        total_data_sample = len(X)
        train_samples = int(total_data_sample * train_ratio)
        test_samples = total_data_sample - train_samples

        random.shuffle(x_data)
        shuffled_data = pd.DataFrame(x_data, columns=header)

        train_data = shuffled_data.head(train_samples)
        test_data = shuffled_data.head(test_samples)
        
        return train_data, test_data
        
    def _predict(self, x_train, y_train):
        #TO-DO : Add Regularization 
        
        x_train_array = x_train.to_numpy()
        m = len(y_train) # number of features 
        #Initial weights 
        tempWeights = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
        tempW0 = 0
        y_pred_btc = tempW0
        
        for idx in range (self.max_iter_): 
            for i in range (m):
                summation = []
                xi = x_train_array[i, :]
                y_pred_btc = tempW0
                for k in range (17):
                    y_pred_btc += self.w_[k] * xi[k]
                summation.append(y_pred_btc - y_train)

                derivative = 2/m * sum(summation)
                tempW0 = self.w0 - (self.alpha * derivative)

            for j in range (17):
                for i in range (m): 
                    summation = []
                    xi = x_train_array[i, :]
                    y_pred_btc = tempW0 
                    for k in range (17):
                        y_pred_btc += self.w_[k] * xi[k]  

                    summation.append(y_pred_btc - y_train)
                derivative = 2/m * sum(summation)
                tempWeights.append(self.w_[j] - (self.alpha * derivative))    

            #The previous iteration 
            last_17 = tempWeights[-17:]

            #Assigning new weight values to old weights
            for n in range (17):
                tempWeights[n] = last_17[n]
            self.w0 = tempW0
            #next iteration

        final_weights = tempWeights[:17]
        self.w_ = final_weights
        
        return y_pred_btc
        
class Utils:  

    def _updateTrainingData(self, x_train):
        num_days = 7  # Number of days to lag
        lagged_columns = []

        for i in x_train.columns:
            if i != 'Date':
                for lag in range(1, num_days + 1):
                    new_column_name = f'{lag}DaysAgo_{i}'
                    lagged_column = x_train[i].shift(lag)
                    lagged_columns.append(lagged_column)

        # Concatenate all new columns at once
        lagged_df = pd.concat(lagged_columns, axis=1)
        lagged_df.columns = [f'{lag}DaysAgo_{col}' for lag in range(1, num_days + 1) for col in x_train.columns if col != 'Date']

        # Combine lagged_df with x_train
        x_train = pd.concat([x_train[['Date']], lagged_df], axis=1)

        return x_train
    
    def _updateDate(self, x_train):
        x_train["Date"]
        x_train['Date'] = x_train['Date'].astype(str)
        
        for k in range (len(x_train["Date"])):
            size = len(x_train["Date"][k])
            string = x_train["Date"][k]
            substring_to_remove = ".0"
            x_train.loc[k]['Date'] = string.replace(substring_to_remove, "")

        for k in range (len(x_train["Date"])):
            size = len(x_train["Date"][k])
            string = x_train["Date"][k]
            x_train.loc[k]['Date'] = string[:4] + "-" + string[4:]
            #print(result)

        for k in range (len(x_train["Date"])):
            size = len(x_train["Date"][k])
            string = x_train["Date"][k]
            x_train.loc[k]['Date'] = string[:7] + "-" + string[7:]
        
        return x_train['Date']
    
    def _replaceNanY(self, y_train):
        column_means_7days =  y_train.tail(7).mean()
        df_filled = y_train_btcHigh.fillna(column_means_7days)
        y_train = df_filled
        
        return y_train
    
    def _replaceNanX(self, x_train):
        column_means = x_train.mean()
        df_filled = x_train.fillna(column_means)
        x_train = df_filled
        
        return x_train
    
    def _newTrainingData(self, new_row, x_train):
        index = len(x_train)
        x_train.loc[index] = new_row
        x_train_temp = x_train
        return x_train_temp

    
    def _newTrainLabel(self, y_train):
        column_means_7days = y_train.tail(7).mean()
        new_row = pd.Series({'btcHigh': column_means_7days})
        y_train_temp = y_train.append(new_row, ignore_index=True)
        return y_train_temp
    
class Predictor: 
    
    X_ = []
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.y_btcHigh_pred = []
        self.x_btcHigh_train = []
        self.y_btcHigh_train = []
        Predictor.X_ = self.X
        
    def getAllPredictionsBTCHigh(self):
        return self.y_btcHigh_pred
    
    def getAllXTrainBTCHigh(self):
        return self.x_btcHigh_train
    
    def getAllYTrainBTCHigh(self):
        return self.y_btcHigh_train
    
    def _predictNextSevenDays(self):
        today = datetime.now().date()
        tomorrow = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        secondDay = (today + timedelta(days=2)).strftime('%Y-%m-%d')
        threeDay = (today + timedelta(days=3)).strftime('%Y-%m-%d')
        fourthDay = (today + timedelta(days=4)).strftime('%Y-%m-%d')
        fifthDay = (today + timedelta(days=5)).strftime('%Y-%m-%d')
        sixthDay = (today + timedelta(days=6)).strftime('%Y-%m-%d')
        seventhDay = (today + timedelta(days=7)).strftime('%Y-%m-%d')
        self._predictWithLinearRegression(tomorrow)
        self._predictWithLinearRegression(secondDay)
        self._predictWithLinearRegression(threeDay)
        self._predictWithLinearRegression(fourthDay)
        self._predictWithLinearRegression(fifthDay)
        self._predictWithLinearRegression(sixthDay)
        self._predictWithLinearRegression(seventhDay)
        
        return Predictor.X_.tail(15), Predictor.X_.tail(15)
    
    
    def _predictWithLinearRegression(self, date=None):
    
        model = LinearRegressionRegularization()
        utils = Utils()
        
        #New training data
        if (date!=None): 
            self.X = utils._newTrainingData({'Date': date}, self.X)
            Predictor.X_ = self.X
            
        # Split the data into training subsets
        train_data, test_data, = model._split(self.X, self.y, ratio=0.7, random_state=123 )
        X_train_btcHigh = train_data.drop(columns=['btcHigh'])
        y_train_btcHigh = train_data['btcHigh']
        #Adjust training data 
        X_train_btcHigh = utils._updateTrainingData(X_train_btcHigh)
        X_train_btcHigh = utils._replaceNanX(X_train_btcHigh)
        has_nan = X_train_btcHigh.isna().any().any()
        #if (has_nan != True):
            #Predict
        #    y_pred_btcHigh = model._predict(X_train_btcHigh, y_train_btcHigh)
        #else:
        #    print ("Training Data contains Nan values, ", has_nan)
        pd.options.mode.chained_assignment = None  #Hide warning
        X_train_btcHigh["Date"] = utils._updateDate(X_train_btcHigh)
    
         # Predict High
        X_train_btcHigh = train_data.drop(columns=['btcHigh'])
        y_train_btcHigh = train_data['btcHigh']
        model = LinearRegression()
        model.fit(X_train_btcHigh, y_train_btcHigh)
        y_pred_btcHigh = model.predict(X_train_btcHigh)
        
        # Predict Low
        X_train_btcLow = train_data.drop(columns=['btcLow'])
        y_train_btcLow = train_data['btcLow']
        model = LinearRegression()
        model.fit(X_train_btcLow, y_train_btcLow)
        y_pred_btcLow = model.predict(X_train_btcLow)
        
        # Predict Low
        X_train_btcOpen = train_data.drop(columns=['btcOpen'])
        y_train_btcOpen = train_data['btcOpen']
        model = LinearRegression()
        model.fit(X_train_btcOpen, y_train_btcOpen)
        y_pred_btcOpen = model.predict(X_train_btcOpen)
        
        # Predict Low
        X_train_btcClose = train_data.drop(columns=['btcClose'])
        y_train_btcClose = train_data['btcClose']
        model = LinearRegression()
        model.fit(X_train_btcClose, y_train_btcClose)
        y_pred_btcClose = model.predict(X_train_btcClose)
    
        y_pred_btcHigh = pd.DataFrame(y_pred_btcHigh)
        y_pred_btcLow = pd.DataFrame(y_pred_btcLow)
        y_pred_btcOpen = pd.DataFrame(y_pred_btcOpen)
        y_pred_btcClose = pd.DataFrame(y_pred_btcClose)
        
        self.y_btcHigh_pred = y_pred_btcHigh 
        self.x_btcHigh_train = X_train_btcHigh 
        self.y_btcHigh_train = y_train_btcHigh 
        
        
        y_pred_btcHigh_ = y_pred_btcHigh.iloc[-1:].iloc[-1:].values.flatten()
        y_pred_btcLow_ = y_pred_btcLow.iloc[-1:].iloc[-1:].values.flatten()
        y_pred_btcOpen_ = y_pred_btcOpen.iloc[-1:].iloc[-1:].values.flatten()
        y_pred_btcClose_ = y_pred_btcClose.iloc[-1:].iloc[-1:].values.flatten()
        
        return self.X.tail(15), y_pred_btcHigh_[0], y_pred_btcLow_[0], y_pred_btcOpen_[0],y_pred_btcClose_[0]



@app.get("/greetings")
async def read_root():
    return {"message": "Hello, World!"}

def is_valid_date_format(date_string):
    date_regex = r'^\d{4}-\d{2}-\d{2}$'
    return bool(re.match(date_regex, date_string))

@app.get("/predict/{date1}/{date2}/{date3}/{date4}/{date5}/{date6}/{date7}")
async def predict(date1: str, date2: str, date3: str, date4: str, date5: str, date6: str, date7: str):
    global global_var
    global index
    dates = [date1, date2, date3, date4, date5, date6, date7]
    columns_to_keep = ['Date', 'btcHigh', 'btcLow', 'btcOpen', 'btcClose']    # Specify the columns you want to keep
    for date in dates:
        if is_valid_date_format(date):
            X0, predHigh, predLow, predOpen, predClose = predict(date)
            if global_var is None:
                global_var = X0.tail()
        
            X0, predHigh, predLow, predOpen, predClose  = predict(date)
            new_row = {'Date': date, 'btcHigh':predHigh, 'btcLow':predLow, 'btcOpen':predOpen, 'btcClose':predClose }
            if (index == 0):
                global_var.drop(global_var.index[-1], inplace=True)
                columns_to_drop = [col for col in X0.columns if col not in columns_to_keep]
                global_var.drop(columns=columns_to_drop, inplace=True)
                index += 1
            #global_var = global_var.append(new_row, ignore_index=True)
            i = len(global_var)
            global_var.loc[i] = new_row
            #x_train_temp = x_train
            #for col in global_var.columns:
            #   global_var.at[global_var.index[-1], col] = new_row[col]
            global_var.fillna(method='ffill', inplace=True)
        else:
            return {"Validation": "Invalid Date format. Expected YYYY-MM-DD!"}

    for col in columns_to_keep:
        if pd.api.types.is_numeric_dtype(global_var[col]):
            global_var[col] = global_var[col].round(2)

    indicator = global_var['btcOpen'].tail(6)
    indicatorDate = global_var['Date'].tail(6)
  

    # Assuming global_var['Open'].tail(6) contains the prices at different times
    prices = global_var['btcOpen'].tail(6).to_dict()
    dates = global_var['Date'].tail(6).to_dict()
    insight1 = ""
    insight2 = ""
    balance = ""

    balance = 100000 
    shares = 0     
    last_price = None
    bought = False 

    # Iterate over the prices
    for time, price in sorted(prices.items()):
        if last_price is not None:
            if price < last_price and not bought:
                # Buy using all available balance
                shares_to_buy = balance / price 
                shares += shares_to_buy
                balance -= shares_to_buy * price
                bought = True
                insight1 = (f"Date: ({dates[time]}): Bought {shares_to_buy} shares at {price}. Current balance: {balance}")
            # Check if the price decreased since last time
            elif price > last_price:
                # Sell if the price decreased
                balance += shares * price  # Sell all shares
                shares = 0
                insight2 = f"Date: ({dates[time]}): Sold all shares at {price} . Current balance: {balance}"
        last_price = price

    return {"predictions": global_var.tail(7), "insight1": insight1, "insight2": insight2}


def predict(date):
    global dataSamples
    if dataSamples is None:  # Check if dataSamples is None
        dataSamples = handleData()  # Initialize dataSamples if it's None

    X = dataSamples
    y = dataSamples['btcHigh']
    predictor = Predictor(X,y)
    predicted_price = predictor._predictWithLinearRegression(date)
    return predicted_price


def handleData():
    try:
        btc, eth, spy = readData()
        if btc is None:
            return {"message": "Failed to read data from CSV files"}

        btc, eth, spy = updateDataframe(btc, eth, spy)

        df = mergeColumn(btc, eth, spy)
        df = updateDateColumn(df)
        global dataframe
        dataframe = df

        return dataframe

    except Exception as e:
        return {"error": str(e)}

    except Exception as e:
        print("An error occurred:", e)
        return {"message": "An error occurred while processing the data"}

@app.get("/readData")
def readData():
    try:
        btc = pd.read_csv('BTC-USD.csv')
        eth = pd.read_csv('ETH-USD.csv')
        spy = pd.read_csv('SPY.csv')
        return btc, eth, spy
    except FileNotFoundError:
        print("One or more CSV files not found.")
        return None, None, None
    except pd.errors.ParserError:
        print("Error parsing CSV files.")
        return None, None, None

    return {"btc": btc, "q": eth}

def updateDataframe(btc, eth, spy):
    for i in btc.columns:
        if i != 'Date':
            btc['btc' + i] = btc[i]
            btc = btc.drop(i, axis=1)

    for i in eth.columns:
        if i != 'Date':
            eth['eth' + i] = eth[i]
            eth = eth.drop(i, axis=1)

    for i in spy.columns:
        if i != 'Date':
            spy['spy' + i] = spy[i]
            spy = spy.drop(i, axis=1)

    return btc, eth, spy

def updateDateColumn(df):
    dateColumn = df["Date"]
    df["Date"] = df["Date"].str.replace("-", '')
    df["Date"] = pd.to_numeric(df['Date'], errors='coerce', downcast="integer")
    return df

def mergeColumn(btc, eth, spy):
    df = pd.merge(btc, eth, on="Date")
    df = pd.merge(df, spy, on='Date')
    return df
