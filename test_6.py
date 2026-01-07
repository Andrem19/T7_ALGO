from __future__ import annotations

import json
import time
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


def get_latest_fear_greed_index(
    *,
    timeout_sec: int = 10,
    retries: int = 3,
    retry_sleep_sec: float = 1.5,
    user_agent: str = "fg-index-prod/1.0",
) -> int:
    """
    Возвращает самое свежее значение Crypto Fear & Greed Index (Alternative.me) как int.
    Ничего лишнего: только число.

    Бросает RuntimeError, если не удалось получить корректный ответ.
    """
    url = "https://api.alternative.me/fng/?limit=1&format=json"

    last_err: Optional[Exception] = None
    for attempt in range(1, max(1, int(retries)) + 1):
        try:
            req = Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/json",
                },
                method="GET",
            )

            with urlopen(req, timeout=timeout_sec) as resp:
                # Иногда resp.status может отсутствовать на старых питонах, поэтому аккуратно
                status = getattr(resp, "status", 200)
                if status != 200:
                    raise RuntimeError(f"HTTP status {status}")
                raw = resp.read()

            payload = json.loads(raw.decode("utf-8", errors="strict"))

            data = payload.get("data")
            if not isinstance(data, list) or not data:
                raise RuntimeError("Bad payload: missing 'data' list")

            item0 = data[0]
            if not isinstance(item0, dict) or "value" not in item0:
                raise RuntimeError("Bad payload: missing 'value'")

            val = int(item0["value"])
            # На всякий случай проверим диапазон (обычно 0..100)
            if val < 0 or val > 100:
                raise RuntimeError(f"Out of range value: {val}")

            return val

        except (HTTPError, URLError, ValueError, json.JSONDecodeError, RuntimeError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(retry_sleep_sec)

    raise RuntimeError(f"Failed to fetch latest Fear & Greed index: {last_err}") from last_err


idx = get_latest_fear_greed_index()
print(idx)