#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
import tensorflow as tf
from typing import Optional, List, Tuple

# =========================================================
# ============== ЗАГРУЗКА МОДЕЛИ И ЛЕЙБЛОВ ===============
# =========================================================

def load_trained_model(model_path: str) -> tf.keras.Model:
    """
    Загружает сохранённую Keras-модель (.keras или SavedModel).
    Компиляция не требуется для инференса (compile=False), чтобы
    не тянуть метрики/объекты обучения.

    :param model_path: Путь к модели (например: ".../model_final.keras")
    :return: tf.keras.Model
    """
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def load_labels(labels_json_path: str) -> List[str]:
    """
    Загружает список имён классов из labels.json, сохранённого вашим тренером.
    В вашем коде он создаётся как словарь {index: "name"}.

    :param labels_json_path: Путь к labels.json (например: ".../labels.json")
    :return: Список имён классов по индексу: labels[idx] -> "class_name"
    """
    with open(labels_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ключи в JSON — строки; упорядочим по индексу и вернём список
    items = sorted(((int(k), v) for k, v in data.items()), key=lambda kv: kv[0])
    return [v for _, v in items]


# =========================================================
# ================= ПРЕДОБРАБОТКА КАРТИНКИ ================
# =========================================================

def _preprocess_image_for_model(img_path: str, img_size: int) -> tf.Tensor:
    """
    Препроцессинг идентичен обучению:
      - tf.io.read_file + tf.image.decode_image(channels=3, expand_animations=False)
      - convert_image_dtype(..., tf.float32) -> [0,1]
      - resize_with_pad(img_size, img_size, BILINEAR)
      - добавляем batch dimension: (1, H, W, 3)

    :param img_path: Путь к изображению
    :param img_size: Целевой размер стороны (у вас IMG_SIZE = 180)
    :return: Тензор формы (1, img_size, img_size, 3), dtype float32
    """
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize_with_pad(img, img_size, img_size,
                                   method=tf.image.ResizeMethod.BILINEAR)
    img = tf.expand_dims(img, axis=0)
    return img


# =========================================================
# ===================== ПРЕДСКАЗАНИЕ ======================
# =========================================================

def predict_single_image(model: tf.keras.Model,
                         img_path: str,
                         img_size: int = 180,
                         labels: Optional[List[str]] = None,
                         print_topk: Optional[int] = None
                         ) -> Tuple[int, Optional[str], np.ndarray]:
    """
    Предсказывает класс по одному изображению, печатает вероятности по всем классам
    (или top-k, если указан print_topk), и возвращает:
      - индекс предсказанного класса,
      - имя класса (если передан labels), иначе None,
      - массив вероятностей (shape=(K,), сумма ≈ 1.0).

    :param model: Загруженная модель (load_trained_model(...))
    :param img_path: Путь к изображению
    :param img_size: Размер стороны для resize_with_pad; должен совпадать с обучением (180)
    :param labels: Список имён классов (load_labels(...)), опционально
    :param print_topk: Если задано, печатаем только top-k строк
    :return: (pred_idx, pred_name_or_None, probs_np)
    """
    # Подготовка входа
    x = _preprocess_image_for_model(img_path, img_size)

    # Прогноз
    probs = model.predict(x, verbose=0)  # (1, K)
    if probs.ndim == 2 and probs.shape[0] == 1:
        probs = probs[0]  # -> (K,)

    # На случай, если последним слоем не softmax (другой чекпойнт):
    if (probs.ndim == 1) and (np.any(probs < 0) or not np.isclose(probs.sum(), 1.0, atol=1e-3)):
        # Применим softmax для стабильности
        e = np.exp(probs - np.max(probs))
        probs = e / e.sum()

    pred_idx = int(np.argmax(probs))
    pred_name = (labels[pred_idx] if (labels is not None and pred_idx < len(labels)) else None)

    # # Красивый вывод вероятностей
    # if labels is not None:
    #     rows = list(zip(range(len(probs)), labels, probs))
    #     rows.sort(key=lambda t: t[2], reverse=True)
    #     k = print_topk if (print_topk is not None and print_topk > 0) else len(rows)
    #     for rank, (idx, name, p) in enumerate(rows[:k], 1):
    #         print(f"{rank:02d}. [{idx:>2}] {name:<30} — {p * 100:6.2f}%")
    # else:
    #     for idx, p in enumerate(probs):
    #         print(f"[{idx:>2}] — {p * 100:6.2f}%")

    return pred_idx, pred_name, probs


# =========================================================
# ===================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ==============
# =========================================================
# if __name__ == "__main__":
    # Пример:
    #   model_path  = "/home/jupiter/PYTHON/T4_ALGO/_models/1d/model_final.keras"
    #   labels_path = "/home/jupiter/PYTHON/T4_ALGO/_models/1d/labels.json"
    #   img_path    = "/path/to/any_image.png"
    #
    #   model  = load_trained_model(model_path)
    #   labels = load_labels(labels_path)
    #   pred_idx, pred_name, probs = predict_single_image(
    #       model, img_path, img_size=180, labels=labels, print_topk=10
    #   )
    #   print(f"\nПредсказание: idx={pred_idx}, class={pred_name}")
    # pass
