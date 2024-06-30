import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import yfinance as yf

def download_stock_data(stock_name, start_date, end_date):
    stock = yf.Ticker(stock_name)
    df = stock.history(start=start_date, end=end_date, interval='1d')
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    # Remove timezone from date
    df['ds'] = df['ds'].dt.tz_localize(None)
    return df

def forecast_plot(forecast, df):
    begining_last_3_months = df.iloc[-1]['ds'] - pd.DateOffset(months=3)
    print(begining_last_3_months)

    forecast_last_3_months = forecast.merge(df, on='ds', how='left')
    forecast_last_3_months = forecast_last_3_months[forecast_last_3_months['ds'] > begining_last_3_months]

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2 = ax.twinx()

    twin1.spines['right'].set_position(('outward', 60))
    twin2.spines['right'].set_position(('outward', 120))

    p1, = ax.plot(forecast_last_3_months['ds'], forecast_last_3_months['yhat'], label='Predicted price', color='y')
    p2, = twin1.plot(forecast_last_3_months['ds'], forecast_last_3_months['y'], label='Real price', color='b')
    twin2.plot(forecast_last_3_months['ds'], forecast_last_3_months['yhat_upper'], label='Upper predicted price', color='g', alpha=0.5)
    twin2.plot(forecast_last_3_months['ds'], forecast_last_3_months['yhat_lower'], label='Lower predicted price', color='g', alpha=0.5)
    plt.fill_between(forecast_last_3_months['ds'], forecast_last_3_months['yhat_upper'], forecast_last_3_months['yhat_lower'], color='g', alpha=0.1)

    fig.legend(handles=[p1, p2])
    plt.gcf().autofmt_xdate()
    plt.gca().set_title('Stock price prediction AAPL')
    fig.savefig('forecast.png')

m = Prophet()

# . Download and prepare stock data
df = download_stock_data('AAPL', '2021-01-01', '2024-3-01')

# . Train the model
df_without_last_3_months = df[df['ds'] < '2023-12-31']
m.fit(df_without_last_3_months)

# . Make predictions
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

forecast_plot(forecast, df)
