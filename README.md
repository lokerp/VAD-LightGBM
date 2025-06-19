# Обнаружение речи с использованием LightGBM

Репозиторий содержит реализацию модели обнаружения речи (Voice Activity Detection, VAD) на основе модели LightGBM, а также графическое приложение для инференса в реальном времени.

## Описание проекта

Проект решает задачу автоматического определения участков аудиосигнала, содержащих речь, отличая их от тишины, фонового шума и других неречевых звуков. Включает:
- Препроцессинг аудиоданных и извлечение признаков (MFCC, RMS, ZCR и др.).
- Обучение и оценку модели градиентного бустинга LightGBM.
- Реализацию инференса в реальном времени с возможностью выбора между тремя моделями:
  - Обученная здесь LightGBM.
  - WebRTC VAD.
  - Silero VAD.

## Структура репозитория

├── notebook.ipynb          # Анализ данных и обучение модели
├── app.py                  # GUI для инференса
├── models.pkl              # Обученная модель LightGBM
└── README.md

## Использованные датасеты

- QUT-NOISE-TIMIT.
- ESC-50.
- FSDnoisy18k.

## Извлекаемые признаки

- MFCC (Mel-Frequency Cepstral Coefficients) и их производные.
- RMS (Root Mean Square Energy).
- ZCR (Zero-Crossing Rate).
- ZRMSE (RMS / ZCR).

## Результаты модели
F1 $= 0.81$.

## Требования для запуска notebook.ipynb
- Python с библиотеками из requirements.txt.
- Датасеты:
  - QUT-NOISE-TIMIT (https://github.com/qutsaivt/QUT-NOISE)
  - ESC-50 (https://github.com/karolpiczak/ESC-50)
  - FSDnoisy18k (https://www.eduardofonseca.net/FSDnoisy18k)
  Необходимо прописать пути к ним в notebook.ipynb.

## Требования для запуска app.py
- Python с библиотеками из requirements.txt.
