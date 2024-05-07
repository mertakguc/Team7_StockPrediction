import pandas as pd

def ParseJSONToDataFrame(data):
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index') # parse Json to DataFrame
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df.index = pd.to_datetime(df.index) 
    df = df.astype(float)  # converting data to float values
    return df

def CreateDataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)