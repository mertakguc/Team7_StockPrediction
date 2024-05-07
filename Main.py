import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from StockMarketData import StockMarketData
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def ParseJSONToDataFrame(data):
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index') # parse Json to DataFrame
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index) 
    df = df.astype(float)  # converting data to float values
    return df

def CreateNormalizedData(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df) #Normalized data with mix max scaler algoriythm
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index) # parse normalized data to df
    return scaled_df

def CreateDataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

stockMarketData = StockMarketData("savedJSONFile.json") #if json file doesn't exist, it creates new json file

data = stockMarketData.ExtractDataJSON()
df = ParseJSONToDataFrame(data)
scaled_df = CreateNormalizedData(df)


# splitting data to test and train.
train_size = int(len(scaled_df) * 0.8)
test_size = len(scaled_df) - train_size
train, test = scaled_df.iloc[0:train_size], scaled_df.iloc[train_size:len(scaled_df)]


window_size = 3 # using window for LSTM 

# windowing test and train data
X_train, y_train = CreateDataset(train[['close']], train.close, window_size)
X_test, y_test = CreateDataset(test[['close']], test.close, window_size)

# LSTM model
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=1)