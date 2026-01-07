import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf
import matplotlib.dates as mdates
import helpers.util as util
from datetime import datetime
import uuid
import os
import shared_vars as sv
from PIL import Image
import talib
import numpy as np
from matplotlib.ticker import MultipleLocator
from matplotlib.dates import MonthLocator

binance_dark = {
    "base_mpl_style": "dark_background",
    "marketcolors": {
        "candle": {"up": "#3dc985", "down": "#ef4f60"},  
        "edge": {"up": "#3dc985", "down": "#ef4f60"},  
        "wick": {"up": "#3dc985", "down": "#ef4f60"},  
        "ohlc": {"up": "green", "down": "red"},
        "volume": {"up": "#247252", "down": "#82333f"},  
        "vcedge": {"up": "green", "down": "red"},  
        "vcdopcod": False,
        "alpha": 1,
    },
    "mavcolors": ("#ad7739", "#a63ab2", "#62b8ba"),
    "facecolor": "#1b1f24",
    "gridcolor": "#2c2e31",
    "gridstyle": "--",
    "y_on_right": True,
    "rc": {
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.edgecolor": "#474d56",
        "axes.titlecolor": "red",
        "figure.facecolor": "#161a1e",
        "figure.titlesize": "x-large",
        "figure.titleweight": "semibold",
    },
    "base_mpf_style": "binance-dark",
}

def draw_candlesticks_positions(candles: list, trades: list, title: str):
    images = []
    for trade in trades:
        coin = trade['coin']
        open_time = float(trade['open_time'])
        close_time = float(trade['close_time'])
        direction = trade['signal']
        profit = trade['profit']
        data_s = trade['data_s']
        index_open = next((i for i, v in enumerate(candles) if float(v[0]) == open_time), None)
        index_close = next((i for i, v in enumerate(candles) if float(v[0]) == close_time), None)
        if index_close == None or index_open == None:
            continue
        # if profit > 0:
        #     continue
        plot_candles = candles[index_open-10:index_close+11]
        
        df = pd.DataFrame(plot_candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)

        markers_buy = [np.nan]*len(df)
        markers_sell = [np.nan]*len(df)
        
        if direction == 1:
            markers_buy[10] = df['Low'].iloc[10]
            markers_sell[-11] = df['High'].iloc[-11]
        else:
            markers_buy[10] = df['High'].iloc[10]
            markers_sell[-11] = df['Low'].iloc[-11]


        addplot_buy = None
        addplot_sell = None
        if direction == 1:
            addplot_buy = mpf.make_addplot(markers_sell, panel=0, type='scatter', markersize=200, color='w', marker='v')
            addplot_sell = mpf.make_addplot(markers_buy, panel=0, type='scatter', markersize=200, color='w', marker='^')
        else:
            addplot_buy = mpf.make_addplot(markers_buy, panel=0, type='scatter', markersize=200, color='w', marker='v')
            addplot_sell = mpf.make_addplot(markers_sell, panel=0, type='scatter', markersize=200, color='w', marker='^')
        tt = f'{data_s} sg:{direction} pr:{profit}'
        if not os.path.exists(f'_pic/{coin}/'):
            os.makedirs(f'_pic/{coin}/')
        uid = uuid.uuid4()
        filename = f'_pic/{coin}/{uid}.png'
        mpf.plot(df, type='candle', style=binance_dark, title=tt, addplot=[addplot_buy, addplot_sell], savefig=filename)
        images.append(Image.open(filename))

    # Combine 6 images into one
    while len(images) >= 6:
        new_img = Image.new('RGB', (3 * images[0].width, 2 * images[0].height))
        for i in range(6):
            new_img.paste(images[i], ((i % 3) * images[i].width, (i // 3) * images[i].height))
            os.remove(images[i].filename)  # remove the image after pasting it
        new_img.save(f'_pic/{coin}/combined_{uuid.uuid4()}.png')
        images = images[6:]

    # If there are remaining images, save them as well
    if images:
        new_img = Image.new('RGB', (3 * images[0].width, 2 * images[0].height))
        for i in range(len(images)):
            new_img.paste(images[i], ((i % 3) * images[i].width, (i // 3) * images[i].height))
            os.remove(images[i].filename)  # remove the image after pasting it
        new_img.save(f'_pic/{coin}/combined_{uuid.uuid4()}.png')

def draw_candlesticks(candles: list, type_labels: str, mark_index: int):
    # Convert the candlesticks data into a pandas DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)
    figsize = (10, 6)
    # Plot the candlestick chart using mpf.plot()
    fig, axlist = mpf.plot(df, type='candle', style=binance_dark, title=type_labels, returnfig=True, figsize=figsize)

    if type_labels == 'up':
        axlist[0].annotate('MARK', (mark_index, df.iloc[mark_index]['open']), xytext=(mark_index, df.iloc[mark_index]['open']-10),
                    arrowprops=dict(facecolor='black', arrowstyle='->'))
    elif type_labels == 'down':
        axlist[0].annotate('MARK', (mark_index, df.iloc[mark_index]['open']), xytext=(mark_index, df.iloc[mark_index]['open']+10),
                        arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Display the chart
    mpf.show()

def plot_time_series(data_list: list, save_pic: bool, points: int, dont_show: bool, data_items: dict, data_items_2: dict):
    path = f'_pic/{datetime.now().date().strftime("%Y-%m-%d")}'
    timestamps = [item['open_time'] for item in data_list]
    values = [item['saldo'] for item in data_list]
    report = ''
    # Преобразование timestamp в формат даты
    dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
    if len(dates) >= 2:
        if dates[0].year < 2017:
            dates[0] = dates[1]
    # Создание графика
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(MonthLocator(interval=2))  # каждый второй месяц
    ax.yaxis.set_major_locator(MultipleLocator(15))  # больше горизонтальных линий
    plt.xticks(rotation=45)  # Поворот дат для лучшей читаемости
    periods = range(1, len(values) + 1)
    cell_text = []
    for key, value in data_items.items():
        cell_text.append([key, '', value])
    table = plt.table(cellText=cell_text,
                  loc='upper left',
                  edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(0.8, 0.8)
    table.auto_set_column_width([0, 1, 2])

    if len(data_items_2)>0:
        cell_text2 = []
        for key, value in data_items_2.items():
            cell_text2.append([key, '', value])
        table2 = plt.table(cellText=cell_text2,
                    loc='lower right',
                    edges='open')
        table2.auto_set_font_size(False)
        table2.set_fontsize(8)
        table2.scale(0.8, 0.8)
        table2.auto_set_column_width([0, 1, 2])

    # Построение графика
    ax.plot(dates, values, linewidth=0.8)
    
    # Добавление подписей и заголовка
    plt.xlabel('Period')
    plt.ylabel('Value')
    plt.title(f"{report}", fontsize=5.5)

    # Add periods close to the dates
    # for i, (date, value) in enumerate(zip(dates, values)):
    #     if i % points == 0:
    #         ax.text(date, value, f"{i}", verticalalignment='top', horizontalalignment='center', fontsize=9, color='red')
    
    # Add grid lines
    ax.grid(True)
    
    # Отображение графика
    if not dont_show:
        plt.tight_layout()

    if save_pic:
        if not os.path.exists(path):
            os.makedirs(path)
        end_path = f'{path}/{datetime.now().timestamp()}{sv.unique_ident}.png'
        plt.savefig(end_path)
        plt.close(fig)
        return end_path
    if not dont_show:
        plt.show()
        plt.close(fig)
    return None


def save_candlesticks_pic(candles: list, path: str):
    # Convert the candlesticks data into a pandas DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = df['timestamp'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)

    # Define the style dictionary
    my_style = mpf.make_mpf_style(base_mpf_style='binance', gridstyle='')

    # Plot the candlestick chart using mpf.plot()
    mpf.plot(df, type='candle', style=my_style, axisoff=True, figratio=(4,4), savefig=path)


def plot_profit(data):
    # Разделение данных на даты и профиты
    timestamps = [item['close_time'] for item in data]
    dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
    profits = [item['profit'] for item in data]

    # Определение цветов для столбцов
    colors = ['green' if profit >= 0 else 'red' for profit in profits]

    # Создание столбчатой диаграммы
    plt.bar(dates, profits, color=colors)
    plt.xlabel('Close Time')
    plt.ylabel('Profit')
    plt.title('Profit by Close Time')

    # Настройка отображения дат вертикально и с более частыми метками
    plt.xticks(rotation=90, fontsize=6)
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(nbins=len(dates)))

    # Создание директории, если она не существует
    path = f'_pic/{datetime.now().date().strftime("%Y-%m-%d")}'
    os.makedirs(path, exist_ok=True)

    # Сохранение графика
    file_path = os.path.join(path, f'{datetime.now().timestamp()}{sv.unique_ident}.png')
    plt.savefig(file_path)
    plt.close()

    return file_path


def plot_types(data):
    # Extracting profits, types of signals, and close times
    profits = [item['profit'] for item in data]
    types_of_signal = [item['type_of_signal'] for item in data]
    close_times = [datetime.fromtimestamp(item['close_time'] / 1000) for item in data]

    # Defining colors for each type of signal
    color_map = {
        'ham_1a': 'blue',
        'ham_2a': 'red',
        'ham_5a': 'orange',
        'ham_5b': 'purple',
        'ham_60c': 'brown',
        'ham_usdc_1': 'pink',
        'ham_usdc': 'yellow',
        'ham_long': 'cyan',
        'long_1': 'black',
        'short_1': 'red',
        'long_2': 'blue',
        'short_2': 'orange'
    }
    colors = [color_map[type_signal] for type_signal in types_of_signal]

    # Creating the bar chart
    bars = plt.bar(range(len(profits)), profits, color=colors)
    plt.xlabel('Position Index')
    plt.ylabel('Profit')
    plt.title('Profit by Position Index')

    # Adding legend
    legend_labels = list(color_map.keys())
    legend_colors = [color_map[label] for label in legend_labels]
    patches = [plt.Line2D([0], [0], color=color, lw=4) for color in legend_colors]
    plt.legend(patches, legend_labels, title="Type of Signal")

    # Adding dates to the x-axis every 10 bars
    ln_pos = len(data)
    freq = 25 if ln_pos > 500 else 10 if ln_pos > 120 else 5
    for i in range(0, len(profits), freq):
        plt.text(i, min(profits) - (max(profits) - min(profits)) * 0.05, close_times[i].strftime('%Y-%m-%d'), 
                 rotation=90, fontsize=4.5, ha='center')

    # Creating directory if it doesn't exist
    path = f'_pic/{datetime.now().date().strftime("%Y-%m-%d")}'
    os.makedirs(path, exist_ok=True)

    # Saving the plot
    file_path = os.path.join(path, f'{datetime.now().timestamp()}.png')
    plt.savefig(file_path)
    plt.close()

    return file_path

def save_candlesticks_pic_2(candles: list, inset_candles: list, path: str):
    # Convert the main candlesticks data into a pandas DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = df['timestamp'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)

    # Convert the inset candlesticks data into a pandas DataFrame
    inset_df = pd.DataFrame(inset_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    inset_df['timestamp'] = inset_df['timestamp'].astype(int)
    inset_df['timestamp'] = pd.to_datetime(inset_df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    inset_df.set_index('timestamp', inplace=True)

    # Define the style dictionary
    my_style = mpf.make_mpf_style(base_mpf_style='binance', gridstyle='', y_on_right=False)

    # Create the main plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    

    # Plot the main candlesticks
    mpf.plot(df, type='candle', style=my_style, ax=ax1, axisoff=True)
    # ax1.set_title(util.get_ident_type(sv.signal.type_os_signal))
    # Plot the inset candlesticks
    mpf.plot(inset_df, type='candle', style=my_style, ax=ax2, axisoff=True)
    ax2.set_title(util.get_viz_time(int(datetime.fromtimestamp(candles[-1][0]/1000).hour)))
    # Remove the frame and ticks from both plots
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Save the figure
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_candlesticks_pic_3(candles: list, inset_candles: list, inset_candles_2: list, path: str):
    # Convert the main candlesticks data into a pandas DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = df['timestamp'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)

    # Convert the first inset candlesticks data into a pandas DataFrame
    inset_df = pd.DataFrame(inset_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    inset_df['timestamp'] = inset_df['timestamp'].astype(int)
    inset_df['timestamp'] = pd.to_datetime(inset_df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    inset_df.set_index('timestamp', inplace=True)

    # Convert the second inset candlesticks data into a pandas DataFrame
    inset_df_2 = pd.DataFrame(inset_candles_2, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    inset_df_2['timestamp'] = inset_df_2['timestamp'].astype(int)
    inset_df_2['timestamp'] = pd.to_datetime(inset_df_2['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    inset_df_2.set_index('timestamp', inplace=True)

    # Define the style dictionary
    my_style = mpf.make_mpf_style(base_mpf_style='binance', gridstyle='', y_on_right=False)

    # Create the main plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax_main = fig.add_axes([0.05, 0.5, 0.9, 0.45])  # Main plot in the upper half
    mpf.plot(df, type='candle', style=my_style, ax=ax_main, axisoff=True)

    # Create the first inset plot if inset_df is not empty
    if not inset_df.empty:
        ax_inset = fig.add_axes([0.05, 0.05, 0.2, 0.45])  # First inset plot in the lower left
        mpf.plot(inset_df, type='candle', style=my_style, ax=ax_inset, axisoff=True)

        # Remove the frame and ticks from the first inset plot
        for spine in ax_inset.spines.values():
            spine.set_visible(False)
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])

    # Create the second inset plot if inset_df_2 is not empty
    if not inset_df_2.empty:
        ax_inset_2 = fig.add_axes([0.3, 0.05, 0.65, 0.45])  # Second inset plot in the lower right
        mpf.plot(inset_df_2, type='candle', style=my_style, ax=ax_inset_2, axisoff=True)

        # Remove the frame and ticks from the second inset plot
        for spine in ax_inset_2.spines.values():
            spine.set_visible(False)
        ax_inset_2.set_xticks([])
        ax_inset_2.set_yticks([])

    # Remove the frame and ticks from the main plot
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    ax_main.set_xticks([])
    ax_main.set_yticks([])

    # Remove the "Price" label from the y-axis of the main plot
    ax_main.yaxis.label.set_visible(False)

    # Remove the "Price" label from the y-axis of the inset plots
    if not inset_df.empty:
        ax_inset.yaxis.label.set_visible(False)
    if not inset_df_2.empty:
        ax_inset_2.yaxis.label.set_visible(False)

    # Save the figure
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_candlesticks_pic_2BB(candles: list, inset_candles: list, path: str):
    # Convert the main candlesticks data into a pandas DataFrame
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = df['timestamp'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)

    # Convert the inset candlesticks data into a pandas DataFrame
    inset_df = pd.DataFrame(inset_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    inset_df['timestamp'] = inset_df['timestamp'].astype(int)
    inset_df['timestamp'] = pd.to_datetime(inset_df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    inset_df.set_index('timestamp', inplace=True)

    # Define the style dictionary
    my_style = mpf.make_mpf_style(base_mpf_style='binance', gridstyle='', y_on_right=False)

    # Calculate Bollinger Bands for both datasets
    def calculate_bollinger_bands(df):
        upper_band, middle_band, lower_band = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        return upper_band, middle_band, lower_band

    # Create the main plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Calculate Bollinger Bands for the main plot
    upper_band, middle_band, lower_band = calculate_bollinger_bands(df)
    addplots_main = [
        mpf.make_addplot(upper_band, ax=ax1, color='red'),
        mpf.make_addplot(middle_band, ax=ax1, color='yellow'),
        mpf.make_addplot(lower_band, ax=ax1, color='blue')
    ]

    # Calculate Bollinger Bands for the inset plot
    upper_band_inset, middle_band_inset, lower_band_inset = calculate_bollinger_bands(inset_df)
    addplots_inset = [
        mpf.make_addplot(upper_band_inset, ax=ax2, color='red'),
        mpf.make_addplot(middle_band_inset, ax=ax2, color='yellow'),
        mpf.make_addplot(lower_band_inset, ax=ax2, color='blue')
    ]
    ax2.set_title(util.get_viz_time(int(datetime.fromtimestamp(candles[-1][0]/1000).hour)))

    # Plot the main candlesticks with Bollinger Bands
    mpf.plot(df, type='candle', style=my_style, ax=ax1, addplot=addplots_main, axisoff=True)

    # Plot the inset candlesticks with Bollinger Bands
    mpf.plot(inset_df, type='candle', style=my_style, ax=ax2, addplot=addplots_inset, axisoff=True)

    # Remove the frame and ticks from both plots
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    # Save the figure
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_candlesticks_pic_1(candles: list, path: str) -> None:
    # Таблица со свечами
    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df['timestamp'] = df['timestamp'].astype(int)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=False).dt.tz_localize('UTC').dt.tz_convert('Europe/London')
    df.set_index('timestamp', inplace=True)

    # Стиль
    my_style = mpf.make_mpf_style(base_mpf_style='binance', gridstyle='', y_on_right=False)

    # Плоскость
    fig, ax1 = plt.subplots(figsize=(10, 10))

    # Свечи без каких-либо подписей/осей
    mpf.plot(
        df,
        type='candle',
        style=my_style,
        ax=ax1,
        axisoff=True,
        ylabel=''  # убираем "Price"
    )

    # Доп. зачистка всего, кроме самих свечей
    ax1.set_ylabel('')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.yaxis.set_visible(False)
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Сохранение
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
