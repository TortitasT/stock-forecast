import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
import yfinance as yf
import subprocess

def download_stock_data(stock_name, start_date, end_date):
    stock = yf.Ticker(stock_name)
    df = stock.history(start=start_date, end=end_date, interval='1d')
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    # Remove timezone from date
    df['ds'] = df['ds'].dt.tz_localize(None)
    # Fill missing dates with previous value
    df = df.resample('D', on='ds').last().fillna(method='ffill').reset_index()
    return df

def forecast_plot(forecast, df, filename = 'forecast.png'):
    begining_last_months = df.iloc[-1]['ds'] - pd.DateOffset(months=3)
    forecast_last_months = forecast.merge(df, on='ds', how='left')
    forecast_last_months = forecast_last_months[forecast_last_months['ds'] > begining_last_months]

    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2 = ax.twinx()

    twin1.spines['right'].set_position(('outward', 60))
    twin2.spines['right'].set_position(('outward', 120))

    p1, = ax.plot(forecast_last_months['ds'], forecast_last_months['yhat'], label='Predicted price', color='y')
    p2, = twin1.plot(forecast_last_months['ds'], forecast_last_months['y'], label='Real price', color='b')
    twin2.plot(forecast_last_months['ds'], forecast_last_months['yhat_upper'], label='Upper predicted price', color='g', alpha=0.5)
    twin2.plot(forecast_last_months['ds'], forecast_last_months['yhat_lower'], label='Lower predicted price', color='g', alpha=0.5)
    plt.fill_between(forecast_last_months['ds'], forecast_last_months['yhat_upper'], forecast_last_months['yhat_lower'], color='g', alpha=0.1)

    fig.legend(handles=[p1, p2])
    plt.gcf().autofmt_xdate()
    plt.gca().set_title('Stock price prediction AAPL')
    fig.savefig(filename)

def backtest(stock, start_date, end_date):
    m = Prophet()

    # Download and prepare stock data
    df = download_stock_data(stock, '2000-01-01', end_date)
    df.to_csv(f'{stock}.csv', index=False)

    # Train the model
    df_without_observable_part = df[df['ds'] < start_date]
    m.fit(df_without_observable_part)

    # Make predictions
    periods = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    future = m.make_future_dataframe(periods)
    forecast = m.predict(future)

    filename = f'forecast_{stock}.png'
    forecast_plot(forecast, df, filename)

    subprocess.run(['open', filename])

# En este pasa algo super extraño
# backtest('AAPL', '2024-06-01', '2024-06-28')

backtest('BIDU', '2024-05-01', '2024-07-28')
