# -*- coding: utf-8 -*-
from typing import Tuple, Union, Sequence
import numpy as np

def classify_two_candlesticks(
    body_1: float, upper_1: float, lower_1: float,
    body_2: float, upper_2: float, lower_2: float
) -> int:
    """
    Классификация двух подряд идущих свечей (двухсвечный паттерн) в одно из 24 значений (1..24).

    Параметры:
        body_1, upper_1, lower_1 : float
            Параметры первой свечи:
                body_1  > 0  -> бычья свеча,
                body_1  < 0  -> медвежья свеча,
                body_1  ~ 0  -> doji (очень маленькое тело).
                upper_1      -> верхняя тень (если body>0, это процент над close, иначе над open)
                lower_1      -> нижняя тень (отрицательное число, берём abs(...) как длину)
        
        body_2, upper_2, lower_2 : float
            То же, но для второй свечи.

    Возвращает:
        int в диапазоне 1..24 – уникальный номер паттерна.

    -------------------------------
    Библиотека используемых паттернов (пример):
      1.  Bullish Engulfing
      2.  Bearish Engulfing
      3.  Bullish Harami
      4.  Bearish Harami
      5.  Piercing Pattern
      6.  Dark Cloud Cover
      7.  Tweezer Top
      8.  Tweezer Bottom
      9.  Bullish Outside
      10. Bearish Outside
      11. Bullish Inside
      12. Bearish Inside
      13. Bullish Doji Star
      14. Bearish Doji Star
      15. Matching High
      16. Matching Low
      17. Bullish Kicker
      18. Bearish Kicker
      19. Bullish Meeting Lines
      20. Bearish Meeting Lines
      21. Bullish Belt Hold
      22. Bearish Belt Hold
      23. Side-by-Side (Bullish)
      24. Side-by-Side (Bearish)
    -------------------------------

    Подход к реконструкции цены (условный):
      - Считаем, что первая свеча открылась по цене 1.0
      - Если body > 0: close_1 = 1.0 + body_1
        Иначе        : close_1 = 1.0 + body_1 (будет меньше 1, если body_1 < 0)
      - high_1 = max(open_1, close_1) + upper_1
      - low_1 = min(open_1, close_1) - abs(lower_1)

      Аналогично для второй свечи (open_2 = close_1) – чтобы учесть «природу» соседних свечей.
      Если нужно моделировать «гепы» (разрывы), можно open_2 = close_1 +/- некий шаг.
    """
    import math

    # === Шаг 1. Реконструкция (псевдо) цен свечей ===
    # Условимся, что первая свеча открывается по 1.0
    open_1 = 1.0
    close_1 = open_1 + body_1  # если body_1 > 0, закрытие выше открытия
    high_1 = max(open_1, close_1) + upper_1
    low_1 = min(open_1, close_1) - abs(lower_1)

    # Предположим, что вторая свеча открывается там же, где закрылась первая.
    # (В реальном рынке может быть геп: open_2 != close_1, но для упрощения возьмём так.)
    open_2 = close_1
    close_2 = open_2 + body_2
    high_2 = max(open_2, close_2) + upper_2
    low_2 = min(open_2, close_2) - abs(lower_2)

    # === Шаг 2. Определение свойств свечей (бычья / медвежья / doji) ===
    SMALL = 0.001  # порог, ниже которого тело считаем очень маленьким (doji)
    def get_candle_type(b: float) -> str:
        if abs(b) < SMALL:
            return "doji"
        return "bullish" if b > 0 else "bearish"

    candle1_type = get_candle_type(body_1)
    candle2_type = get_candle_type(body_2)

    # Размер тела (в условных единицах), чтобы оценивать "большое" или "маленькое" тело
    body1_size = abs(close_1 - open_1)
    body2_size = abs(close_2 - open_2)

    # === Шаг 3. Вспомогательные функции для сравнения ===
    def is_engulfing_bullish() -> bool:
        # Bullish Engulfing: первая свеча медвежья, вторая бычья,
        # причём тело второй полностью покрывает (engulf) тело первой
        if candle1_type == "bearish" and candle2_type == "bullish":
            return (open_2 < open_1) and (close_2 > close_1)
        return False

    def is_engulfing_bearish() -> bool:
        # Bearish Engulfing: первая свеча бычья, вторая медвежья,
        # причём тело второй полностью покрывает тело первой
        if candle1_type == "bullish" and candle2_type == "bearish":
            return (open_2 > open_1) and (close_2 < close_1)
        return False

    def is_harami_bullish() -> bool:
        # Bullish Harami: первая свеча медвежья (большая),
        # вторая (обычно бычья) внутри тела первой
        if candle1_type == "bearish" and candle2_type == "bullish":
            return (open_2 > open_1) and (close_2 < close_1)
        return False

    def is_harami_bearish() -> bool:
        # Bearish Harami: первая свеча бычья (большая),
        # вторая (обычно медвежья) внутри тела первой
        if candle1_type == "bullish" and candle2_type == "bearish":
            return (open_2 < open_1) and (close_2 > close_1)
        return False

    def is_piercing() -> bool:
        # Piercing Pattern: первая свеча медвежья,
        # вторая – бычья, открытие 2й ниже минимума 1й (геп вниз),
        # а закрытие выше середины тела 1й
        if candle1_type == "bearish" and candle2_type == "bullish":
            mid1 = open_1 + (body1_size / 2.0)  # середина тела первой
            # open_2 < low_1 (геп вниз), но у нас упрощённая логика без явных гэпов
            # Упростим до "open_2 < close_1" и "close_2 > mid тела первой"
            return (open_2 < close_1) and (close_2 > mid1)
        return False

    def is_dark_cloud() -> bool:
        # Dark Cloud Cover: первая свеча бычья,
        # вторая – медвежья, открытие 2й выше максимума 1й (геп вверх),
        # а закрытие ниже середины тела 1й
        if candle1_type == "bullish" and candle2_type == "bearish":
            mid1 = open_1 + (body1_size / 2.0)
            return (open_2 > close_1) and (close_2 < mid1)
        return False

    def is_tweezer_top() -> bool:
        # Tweezer Top: две свечи с примерно одинаковым верхом (high_1 ~ high_2)
        # Обычно первая свеча бычья, вторая – медвежья, но мы упростим до сходных high.
        return abs(high_1 - high_2) < 0.001

    def is_tweezer_bottom() -> bool:
        # Tweezer Bottom: две свечи с примерно одинаковым низом (low_1 ~ low_2)
        return abs(low_1 - low_2) < 0.001

    def is_outside_bullish() -> bool:
        # Bullish Outside: вторая свеча бычья,
        # и при этом high_2 > high_1 и low_2 < low_1 (полностью охватывает первую)
        return (candle2_type == "bullish") and (high_2 > high_1) and (low_2 < low_1)

    def is_outside_bearish() -> bool:
        return (candle2_type == "bearish") and (high_2 > high_1) and (low_2 < low_1)

    def is_inside_bullish() -> bool:
        # Bullish Inside: вторая свеча бычья и полностью внутри диапазона первой
        return (candle2_type == "bullish") and (high_2 < high_1) and (low_2 > low_1)

    def is_inside_bearish() -> bool:
        return (candle2_type == "bearish") and (high_2 < high_1) and (low_2 > low_1)

    def is_doji_star_bullish() -> bool:
        # Bullish Doji Star: первая свеча медвежья, вторая – doji, расположенная с «разрывом» (геп)
        # Упрощённо: candle2_type = doji, candle1_type = bearish
        # и high_2 < open_1 или low_2 > close_1 (в зависимости от направления) – будем считать условно
        if candle1_type == "bearish" and candle2_type == "doji":
            # Упростим логику гепа: open_2 > open_1 + некий порог
            return True
        return False

    def is_doji_star_bearish() -> bool:
        # Аналогично для первой бычьей, второй doji
        if candle1_type == "bullish" and candle2_type == "doji":
            return True
        return False

    def is_meeting_lines_bullish() -> bool:
        # Bullish Meeting Lines: первая свеча медвежья, вторая бычья
        # с открытием, равным или близким к закрытию первой.
        # И закрытие второй очень близко к закрытию первой.
        if candle1_type == "bearish" and candle2_type == "bullish":
            if abs(open_2 - close_1) < 0.001:
                return True
        return False

    def is_meeting_lines_bearish() -> bool:
        # То же зеркально
        if candle1_type == "bullish" and candle2_type == "bearish":
            if abs(open_2 - close_1) < 0.001:
                return True
        return False

    def is_belt_hold_bullish() -> bool:
        # Bullish Belt Hold: длинная бычья свеча с очень маленькой нижней тенью
        # Вторая (предположим) может быть нейтральной?
        # Упростим до: первая свеча (candle1_type == 'bullish'),
        # нижняя тень очень маленькая, а вторая просто тоже бычья.
        if candle1_type == "bullish" and abs(low_1 - min(open_1, close_1)) < 0.001:
            if candle2_type == "bullish":
                return True
        return False

    def is_belt_hold_bearish() -> bool:
        if candle1_type == "bearish" and abs(high_1 - max(open_1, close_1)) < 0.001:
            if candle2_type == "bearish":
                return True
        return False

    def is_side_by_side_bullish() -> bool:
        # Side-by-Side (Bullish): упрощённо - две бычьи свечи подряд с близкими тенями
        if candle1_type == "bullish" and candle2_type == "bullish":
            # Проверим, что high_1 ~ high_2 и/или low_1 ~ low_2
            if abs(high_1 - high_2) < 0.002 or abs(low_1 - low_2) < 0.002:
                return True
        return False

    def is_side_by_side_bearish() -> bool:
        if candle1_type == "bearish" and candle2_type == "bearish":
            if abs(high_1 - high_2) < 0.002 or abs(low_1 - low_2) < 0.002:
                return True
        return False

    # === Шаг 4. Определяем паттерн по набору проверок (упрощённый приоритет) ===
    # Для упрощения: сначала проверяем самые "узнаваемые" паттерны, далее переходим к другим.
    # На практике логику и приоритет можно менять.

    if is_engulfing_bullish():
        return 1
    if is_engulfing_bearish():
        return 2
    if is_harami_bullish():
        return 3
    if is_harami_bearish():
        return 4
    if is_piercing():
        return 5
    if is_dark_cloud():
        return 6
    if is_tweezer_top():
        return 7
    if is_tweezer_bottom():
        return 8
    if is_outside_bullish():
        return 9
    if is_outside_bearish():
        return 10
    if is_inside_bullish():
        return 11
    if is_inside_bearish():
        return 12
    if is_doji_star_bullish():
        return 13
    if is_doji_star_bearish():
        return 14
    if is_meeting_lines_bullish():
        return 19
    if is_meeting_lines_bearish():
        return 20
    if is_belt_hold_bullish():
        return 21
    if is_belt_hold_bearish():
        return 22
    if is_side_by_side_bullish():
        return 23
    if is_side_by_side_bearish():
        return 0

    # Если никакой паттерн не найден — можем вернуть что-то вроде "0" или "15..18" (не реализованные)
    # но для полноты возвращаем условные 15..18 (или 0). Ниже вариант с 15.
    return 15

def classify_candlestick(body: float, upper: float, lower: float) -> int:
    SMALL = 0.0025         # Threshold to consider a value negligible (0.5%)
    LONG_MULTIPLIER = 2.5   # Multiplier to decide if a shadow is "long" relative to the body

    # Convert lower shadow to positive for easier calculations
    lower = abs(lower)

    # Check for Doji: nearly zero body
    if abs(body) < SMALL:
        # Doji classification based on shadows
        if upper < SMALL and lower < SMALL:
            return 9      # Pattern 9: Doji with negligible shadows
        elif upper >= lower:
            if lower < SMALL:
                return 10  # Pattern 10: Doji with long upper shadow
            else:
                return 0  # Pattern 12: Doji with both shadows significant (upper dominant)
        else:  # lower > upper
            if upper < SMALL:
                return 11  # Pattern 11: Doji with long lower shadow
            else:
                return 0  # Pattern 12: Doji with both shadows significant (lower dominant)

    # For non-Doji candles (body is significant)
    is_bullish = body > 0
    abs_body = abs(body)
    is_upper_negligible = upper < SMALL
    is_lower_negligible = lower < SMALL
    is_upper_long = upper >= LONG_MULTIPLIER * abs_body
    is_lower_long = lower >= LONG_MULTIPLIER * abs_body

    if is_bullish:
        if is_upper_negligible and is_lower_negligible:
            return 1  # Pattern 1: Bullish Marubozu
        elif is_upper_long and is_lower_negligible:
            return 2  # Pattern 2: Bullish with long upper shadow
        elif is_lower_long and is_upper_negligible:
            return 3  # Pattern 3: Bullish with long lower shadow
        elif not is_upper_negligible and not is_lower_negligible:
            return 4  # Pattern 4: Bullish with both shadows significant
        else:
            # When one shadow exists but does not qualify as "long"
            if not is_upper_negligible:
                return 2  # Default to pattern 2 if only upper shadow is present
            elif not is_lower_negligible:
                return 3  # Default to pattern 3 if only lower shadow is present
            else:
                return 4
    else:
        # Bearish candle
        if is_upper_negligible and is_lower_negligible:
            return 5  # Pattern 5: Bearish Marubozu
        elif is_upper_long and is_lower_negligible:
            return 6  # Pattern 6: Bearish with long upper shadow
        elif is_lower_long and is_upper_negligible:
            return 7  # Pattern 7: Bearish with long lower shadow
        elif not is_upper_negligible and not is_lower_negligible:
            return 8  # Pattern 8: Bearish with both shadows significant
        else:
            if not is_upper_negligible:
                return 6
            elif not is_lower_negligible:
                return 7
            else:
                return 8

# -*- coding: utf-8 -*-
from typing import List
import numpy as np
import talib


def _extract_ohlc(candles: List[List[float]]):
    """
    Извлекает массивы OHLC из свечей вида:
    [timestamp_ms, open, high, low, close, volume]
    """
    arr = np.asarray(candles, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError("candles должен быть списком размерности Nx6: [ts, o, h, l, c, v].")
    o, h, l, c = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    return o, h, l, c


def rsi_category(
    candles: List[List[float]],
    period: int = 14,
    typical_low: float = 15.0,
    typical_high: float = 75.0,
) -> int:
    """
    Категория RSI (0..5) по последнему значению.
    Диапазон 15..75 делится на 6 равных бинов (по 10 пунктов).
    Значения ниже 15 → 0, выше 75 → 5.

    Возвращает: int в [0, 5]
    """
    _, _, _, close = _extract_ohlc(candles)
    if close.size < period:
        raise ValueError(f"Недостаточно свечей для RSI: нужно ≥ {period}.")

    rsi = talib.RSI(close, timeperiod=period)
    last_rsi = rsi[-1]
    if np.isnan(last_rsi):
        # На случай, если последние элементы NaN из-за короткого ряда
        last_valid = np.flatnonzero(~np.isnan(rsi))
        if last_valid.size == 0:
            raise ValueError("RSI не удалось вычислить (все значения NaN).")
        last_rsi = rsi[last_valid[-1]]

    width = (typical_high - typical_low) / 6.0
    if width <= 0:
        raise ValueError("Некорректный типовой диапазон для RSI.")

    bin_idx = int(np.floor((last_rsi - typical_low) / width))
    bin_idx = int(np.clip(bin_idx, 0, 5))
    return bin_idx


def atr_category(
    candles: List[List[float]],
    period: int = 14,
    low_q: float = 0.10,
    high_q: float = 0.90,
) -> int:
    """
    Категория ATR (0..5) по последнему значению.

    Под «самым частым диапазоном» применяем практичный эквивалент — «типичный диапазон»
    по квантилям ATR за весь доступный ряд: [low_q, high_q] (по умолчанию 10..90 перцентили),
    чтобы отбросить редкие выбросы. Этот диапазон делим на 6 равных зон.

    Возвращает: int в [0, 5]
    """
    _, high, low, close = _extract_ohlc(candles)
    n = close.size
    if n < period + 2:
        raise ValueError(f"Недостаточно свечей для ATR: нужно ≥ {period + 2}.")

    atr = talib.ATR(high, low, close, timeperiod=period)
    valid = atr[~np.isnan(atr)]
    if valid.size == 0:
        raise ValueError("ATR не удалось вычислить (все значения NaN).")

    last_atr = valid[-1]

    lo = float(np.quantile(valid, low_q))
    hi = float(np.quantile(valid, high_q))
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Не удалось определить квантильные границы ATR.")
    if hi <= lo:
        # Вся волатильность «сплющена» — считаем серединой
        return 3

    width = (hi - lo) / 6.0
    if width <= 0:
        return 3

    bin_idx = int(np.floor((last_atr - lo) / width))
    bin_idx = int(np.clip(bin_idx, 0, 5))
    return bin_idx


def bollinger_category(
    candles: List[List[float]],
    period: int = 20,
    nbdev: float = 2.0,
    center_tol_ratio: float = 0.10,
) -> int:
    """
    Категория по положению цены относительно полос Боллинджера (0..4) на последней свече.

    Категории:
      0 — цена ниже нижней ленты
      1 — между нижней и средней (над нижней)
      2 — «центр»: |close - middle| <= center_tol_ratio * (upper - lower)
      3 — между средней и верхней (под верхней)
      4 — выше верхней ленты

    center_tol_ratio = доля ширины канала, задающая «зону центра» (по умолчанию 10%).
    """
    _, _, _, close = _extract_ohlc(candles)
    if close.size < period + 2:
        raise ValueError(f"Недостаточно свечей для Bollinger Bands: нужно ≥ {period + 2}.")

    upper, middle, lower = talib.BBANDS(
        close,
        timeperiod=period,
        nbdevup=nbdev,
        nbdevdn=nbdev,
        matype=0,  # SMA
    )

    u, m, l = upper[-1], middle[-1], lower[-1]
    if np.isnan(u) or np.isnan(m) or np.isnan(l):
        # Берём последние валидные значения, если хвост NaN
        mask = (~np.isnan(upper)) & (~np.isnan(middle)) & (~np.isnan(lower))
        idxs = np.flatnonzero(mask)
        if idxs.size == 0:
            raise ValueError("Невозможно получить последние полосы Боллинджера (NaN).")
        u, m, l = upper[idxs[-1]], middle[idxs[-1]], lower[idxs[-1]]

    c = float(close[-1])
    band_w = float(u - l)

    if c < l:
        return 0
    if c > u:
        return 4

    # Зона центра — симметрично вокруг средней
    if band_w <= 0:
        return 2
    if abs(c - m) <= center_tol_ratio * band_w:
        return 2

    # Внутри канала, но не центр
    if c < m:
        return 1
    else:
        return 3




def candle_parts_pct(
    candle: Union[Sequence[float], np.ndarray],
    *,
    base: Union[str, float] = "open",   # "open" | "close" | "hl2" | "hlc3" | "ohlc4" | число
) -> Tuple[float, float, float]:
    """
    Возвращает (body_pct, upper_wick_pct, lower_wick_pct) в долях (1% = 0.01)
    для одной свечи [timestamp_ms, open, high, low, close, volume].

    Определения:
      body_pct  = (close - open) / denom      (со знаком; для красной свечи < 0)
      upper_pct = max(0, high - max(open, close)) / denom
      lower_pct = max(0, min(open, close) - low) / denom

    denom (база для процентов):
      base="open"  → denom = open
      base="close" → denom = close
      base="hl2"   → denom = (high + low) / 2
      base="hlc3"  → denom = (high + low + close) / 3
      base="ohlc4" → denom = (open + high + low + close) / 4
      base=float   → использовать заданное значение как фиксированный знаменатель
    """
    arr = np.asarray(candle, dtype=float).ravel()
    if arr.size < 6:
        raise ValueError("Ожидается свеча формата [ts, open, high, low, close, volume].")

    _, o, h, l, c, _ = arr[:6]
    # Починка редких аномалий: гарантируем корректный диапазон high/low относительно O/C
    if h < o or h < c:
        h = max(h, o, c)
    if l > o or l > c:
        l = min(l, o, c)

    # Выбор знаменателя
    if isinstance(base, (int, float)):
        denom = float(base)
    else:
        b = str(base).lower()
        if b == "open":
            denom = o
        elif b == "close":
            denom = c
        elif b == "hl2":
            denom = (h + l) / 2.0
        elif b == "hlc3":
            denom = (h + l + c) / 3.0
        elif b == "ohlc4":
            denom = (o + h + l + c) / 4.0
        else:
            raise ValueError('Недопустимое значение base. Используйте "open", "close", "hl2", "hlc3", "ohlc4" или число.')

    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Некорректный знаменатель (denom) для процентов.")

    body = (c - o) / denom
    upper = max(0.0, h - max(o, c)) / denom
    lower = max(0.0, min(o, c) - l) / denom

    return float(body), float(upper), float(lower)

