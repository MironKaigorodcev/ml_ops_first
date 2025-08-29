# ❤️ Heart Disease Classification (ML Ops Demo)

## 📖 Описание проекта

Этот проект демонстрирует полный ML Ops цикл на задаче **классификации сердечных заболеваний** (датасет UCI Heart Disease).  

Мы построили пайплайн, который:
- обрабатывает данные,
- обучает несколько моделей (Logistic Regression, Random Forest, Gradient Boosting),
- логирует эксперименты и метрики в **MLflow**,
- управляет данными и артефактами через **DVC**.

Проект может служить шаблоном для продакшн-подхода к ML: воспроизводимость, версионирование и управление экспериментами.

---

## 📂 Структура проекта
ml_ops_first/
├── data/ # данные (raw, processed) – под DVC
│ └── heart.csv # исходный датасет UCI Heart Disease
├── models/ # обученные модели (.pkl) – под DVC
├── artifacts/ # артефакты препроцессинга, отчёты
├── src/
│ ├── data_loader.py # стадия prepare: предобработка данных
│ └── train_model.py # стадия train: обучение моделей
├── dvc.yaml # описание пайплайна (prepare → train)
├── params.yaml # (опц.) параметры обучения
├── requirements.txt # зависимости
└── README.md # документация


---

## ⚙️ Технологии

- **Python 3.12**
- [scikit-learn](https://scikit-learn.org/stable/) – ML-модели
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) – обработка данных
- [MLflow](https://mlflow.org/) – логирование экспериментов
- [DVC](https://dvc.org/) – управление данными и пайплайнами
- [Git](https://git-scm.com/) – контроль версий кода

---

## 📊 Датасет

- Источник: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Размер: ~300 записей
- Цель: бинарная классификация (`target`: есть заболевание сердца или нет)
- Признаки: возраст, пол, давление, холестерин, результаты ЭКГ и т.д.

---

## 🚀 Запуск проекта

1. **Установите зависимости**:
   ```bash
   pip install -r requirements.txt


