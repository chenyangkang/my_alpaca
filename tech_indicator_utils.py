import pandas as pd
import numpy as np


class tech_indivator_calculator():
    def __init__():
        pass

    def calculate_sma(df, column='close', period=20):
        return df[column].rolling(window=period).mean()

    def calculate_ema(df, column='close', period=20):
        return df[column].ewm(span=period, adjust=False).mean()

    def calculate_bollinger_bands(df, column='close', period=20, num_std=2):
        rolling_mean = df[column].rolling(window=period).mean()
        rolling_std = df[column].rolling(window=period).std()
        upper_band = rolling_mean + num_std * rolling_std
        lower_band = rolling_mean - num_std * rolling_std
        return upper_band, lower_band

    def calculate_macd(df, column='close', span_short=12, span_long=26, span_signal=9):
        exp_short = df[column].ewm(span=span_short, adjust=False).mean()
        exp_long = df[column].ewm(span=span_long, adjust=False).mean()
        macd = exp_short - exp_long
        signal = macd.ewm(span=span_signal, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def calculate_rsi(df, column='close', period=14):
        diff = df[column].diff()
        up = diff.where(diff > 0, 0)
        down = -diff.where(diff < 0, 0)
        ema_up = up.rolling(window=period).mean()
        ema_down = down.rolling(window=period).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_adx(df, period=14):
        
        def calculate_true_range(df):
            prev_close = df['close'].shift()
            high_minus_low = df['high'] - df['low']
            high_minus_prev_close = np.abs(df['high'] - prev_close)
            low_minus_prev_close = np.abs(df['low'] - prev_close)
            tr = pd.DataFrame({'high_low': high_minus_low, 'high_prev_close': high_minus_prev_close, 'low_prev_close': low_minus_prev_close})
            true_range = tr.max(axis=1)
            return true_range
        
        tr = calculate_true_range(df)
        atr = tr.rolling(window=period).mean()
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move.values > 0), up_move, 0), index=up_move.index)
        minus_dm = pd.Series(np.where((down_move > up_move.values) & (down_move > 0), down_move, 0), index=down_move.index)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).apply(lambda x: np.average(x, weights=atr[x.index]))
        return adx


    def calculate_stochastic_oscillator(df, period=14, smooth_window=3):
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=smooth_window).mean()
        return k, d

    def calculate_williams_r(df, period=14):
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        wr = -100 * ((high_max - df['close']) / (high_max - low_min))
        return wr

    def calculate_mfi(df, period=14):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        pos_mf = positive_flow.rolling(window=period).sum()
        neg_mf = negative_flow.rolling(window=period).sum()
        mfr = pos_mf / neg_mf
        mfi = 100 - (100 / (1 + mfr))
        return mfi

    def calculate_obv(df):
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv



















