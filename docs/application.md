# Учебное приложение прогнозирования

Streamlit-приложение выводит два прямых прогноза: модуль упругости при растяжении в ГПа и прочность при растяжении в МПа. Пользователь задаёт 11 признаков; угол нашивки принимает только подтверждённые в наборе значения 0 или 90 градусов. Для признаков, у которых исходная схема не сообщает единицу измерения, интерфейс прямо это указывает.

## Артефакты моделей

В `models/model_manifest.json` сохранён контракт `schema_version=1`, метрики и относительные пути двух joblib-артефактов. Каждый артефакт является полностью обученным sklearn-совместимым pipeline и принимает `pandas.DataFrame` с 11 признаками в порядке из manifest. Для модуля используется baseline, для прочности — Gradient Boosting. При отсутствии или повреждении manifest/артефакта интерфейс показывает инструкцию вместо аварийного завершения.

## Нативный запуск в Windows

Нужна вся папка репозитория и CPython 3.12. Команда `python app/streamlit_app.py` не является правильным запуском: глобальный Python может не содержать Streamlit, пакет `composites` находится в каталоге `src`, а само приложение должен запускать модуль Streamlit. Python 3.13 не соответствует диапазону версий проекта.

### Установка один раз

Откройте CMD или PowerShell в корне `composites-property-prediction` и выполните `py -3.12 --version`. Если launcher `py` не находит Python 3.12, установите CPython 3.12 и замените `py -3.12` в следующей команде полным путём к его `python.exe`.

CMD:

```bat
py -3.12 -m venv "..\.venv-composites"
"..\.venv-composites\Scripts\python.exe" -m pip install -r requirements.lock.txt
```

PowerShell:

```powershell
py -3.12 -m venv "..\.venv-composites"
& "..\.venv-composites\Scripts\python.exe" -m pip install -r requirements.lock.txt
```

Зависимости устанавливаются из `requirements.lock.txt` во внешнее окружение рядом с проектом.

### Запуск каждый раз

CMD из корня проекта:

```bat
set "PYTHONPATH=%CD%\src"
set "PYTHONDONTWRITEBYTECODE=1"
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
"..\.venv-composites\Scripts\python.exe" -m streamlit run app\streamlit_app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
```

PowerShell из корня проекта:

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = 'false'
& "..\.venv-composites\Scripts\python.exe" -m streamlit run app/streamlit_app.py --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=false
```

Откройте <http://127.0.0.1:8501>. Оставьте окно терминала открытым; для остановки используйте `Ctrl+C`.

## Запуск в Docker

Команда из корня проекта в проверенном Docker-образе:

```powershell
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -p 127.0.0.1:8501:8501 -v "${PWD}:/workspace:ro" -w /workspace composites-thesis:local streamlit run app/streamlit_app.py --server.address=0.0.0.0 --browser.gatherUsageStats=false
```

Открыть <http://localhost:8501>. При необходимости переменная окружения `COMPOSITES_MODELS_DIR` позволяет указать другой каталог артефактов.

## Интерпретация результата

Поля заполнены медианами обучающих признаков из manifest. Это демонстрационный ввод, а не измерения нового образца; для содержательного прогноза значения нужно заменить фактическими характеристиками материала. Проверка в браузере медианного ввода с углом 90° дала 73,2073 ГПа для модуля и 2502,06 МПа для прочности. Первый прогноз постоянный, поскольку для этой задачи выбрана baseline-модель.

Выход за сохранённые минимумы и максимумы сопровождается предупреждением об экстраполяции, но не блокируется искусственной физической границей. RMSE, MAE и R² отложенного test показываются как общая характеристика модели; они не являются неопределённостью отдельного прогноза.

Результат демонстрирует работу моделей на ограниченном наборе данных. Он не заменяет лабораторные испытания и не подтверждает применимость к новым материалам, партиям или производственным режимам.
