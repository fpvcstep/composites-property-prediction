# Прогнозирование свойств композиционных материалов

Проект ВКР воспроизводит полный путь от двух исходных таблиц до оценки регрессионных моделей и Streamlit-приложения. Работа решает три задачи:

1. прогноз модуля упругости при растяжении по 11 признакам;
2. прогноз прочности при растяжении по тем же 11 признакам;
3. оценка отношения матрица–наполнитель по 12 остальным полям с помощью MLP.

Третья задача является обратной регрессией по известному полному профилю. Она не оптимизирует состав и не доказывает причинное влияние признаков.

Численные результаты окончательной оценки независимо проверены. Доступны [профиль автора](https://github.com/fpvcstep), [репозиторий](https://github.com/fpvcstep/composites-property-prediction) и [история изменений](https://github.com/fpvcstep/composites-property-prediction/commits/main).

## Данные и протокол

`data/raw/X_bp.xlsx` и `data/raw/X_nup.xlsx` объединяются INNER JOIN по проверенному исходному индексу. Результат содержит 1023 строки и 13 содержательных полей; 17 строк `X_nup.xlsx` не имеют пары. Пропусков, бесконечностей и полных дубликатов нет.

Единственное разбиение фиксируется заранее:

- train: 716 строк (70%);
- test: 307 строк (30%);
- внутри обучающей части: десятифолдовая `KFold`-кросс-валидация с перемешиванием и `random_state=42`.

Разведочный анализ, границы IQR, масштабирование, поиск гиперпараметров и выбор модели используют только обучающую часть. Все обучаемые преобразования находятся внутри `sklearn.pipeline.Pipeline` и заново оцениваются на обучающей части каждого фолда. Отложенная выборка используется один раз после фиксации решений.

## Модели

Для двух прямых задач сравниваются `DummyRegressor`, Ridge, Random Forest, Gradient Boosting и SVR. Основная метрика выбора — средняя RMSE по кросс-валидации; дополнительно рассчитываются MAE и R². Модель выбирается по заранее заданному критерию, поэтому простая константная модель сохраняется, если испытанные более сложные модели не уменьшают среднюю RMSE.

Для отношения матрица–наполнитель используется `MLPRegressor`, предусмотренная заданием. Её результат сопоставляется с константной моделью среднего. Наличие нейросети само по себе не означает более высокое качество.

## Структура проекта

- `src/composites/` — загрузка, схемы, split, EDA, preprocessing, обучение и inference;
- `scripts/` — воспроизводимые команды этапов исследования;
- `data/raw/` и `data/splits.json` — опубликованные исходники и зафиксированное разбиение;
- `reports/` — audit, EDA, train/CV и финальная evaluation;
- `figures/` — EDA и финальные графики оценки;
- `models/` — проверенные sklearn pipeline и manifest приложения;
- `notebooks/research.ipynb` — объяснение полного исследования без дублирования реализации;
- `app/streamlit_app.py` — два прямых прогноза;
- `docs/` — словарь данных, протоколы и материалы подготовки к защите;
- `tests/` — проверки контрактов данных, pipeline, обучения и приложения.

## Материалы проекта

- [Текст ВКР в формате DOCX](reports/thesis.docx)
- [Текст ВКР в формате PDF](reports/thesis.pdf)
- [Презентация в формате PPTX](presentation/defense.pptx)
- [Презентация в формате PDF](presentation/defense.pdf)
- [Исследовательский notebook](notebooks/research.ipynb)
- [Руководство по подготовке к защите](docs/defense_guide.md)
- [Текст доклада](docs/talk.md)
- [Сценарий демонстрации](docs/demo.md)

## Полное воспроизведение в Docker

Требуются Docker Desktop и режим Linux containers. Команды выполняются из корня проекта. Переменная `PYTHONDONTWRITEBYTECODE=1` и отключённый pytest cache не создают служебные кэши в каталоге проекта. Контейнер собирается по Dockerfile; изменяемый тег базового образа и незакреплённая версия pip ограничивают побитовую воспроизводимость среды при будущей сборке.

```powershell
docker build -t composites-thesis:local .

docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "${PWD}:/workspace" -w /workspace composites-thesis:local python -m pytest -q -p no:cacheprovider
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "${PWD}:/workspace" -w /workspace composites-thesis:local python scripts/audit_data.py
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "${PWD}:/workspace" -w /workspace composites-thesis:local python scripts/prepare_data.py
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "${PWD}:/workspace" -w /workspace composites-thesis:local python scripts/run_eda.py
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "${PWD}:/workspace" -w /workspace composites-thesis:local python scripts/train_models.py
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -v "${PWD}:/workspace" -w /workspace composites-thesis:local python scripts/evaluate_models.py
```

`evaluate_models.py` выполняют после выбора по обучающей части: он один раз оценивает зафиксированные артефакты на отложенной выборке и формирует `reports/evaluation`, `figures/evaluation` и `models/model_manifest.json`.

Повторный `train_models.py` выполняет полный поиск гиперпараметров на обучающей части и итоговое обучение выбранных конфигураций; это может занимать несколько минут. Для изучения используйте опубликованные результаты и notebook с выключенным флагом `RETRAIN_ALL_MODELS`.

## Запуск без установки проекта в его каталог

Docker — рекомендуемый путь. Если нужен CPython 3.12, виртуальное окружение создаётся рядом с проектом, а импорт исходников задаётся через `PYTHONPATH`. Это не создаёт `.venv`, `egg-info` или bytecode внутри итогового каталога.

```powershell
$project = (Get-Location).Path
$venv = Join-Path (Split-Path -Parent $project) '.venv-composites'
py -3.12 -m venv $venv
& "$venv\Scripts\python.exe" -m pip install --upgrade pip
& "$venv\Scripts\python.exe" -m pip install -r "$project\requirements.lock.txt"
$env:PYTHONPATH = "$project\src"
$env:PYTHONDONTWRITEBYTECODE = '1'
& "$venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider "$project\tests"
```

## Notebook

`notebooks/research.ipynb` читает опубликованные audit/EDA/modeling/evaluation-артефакты и поясняет каждый график. По умолчанию он не повторяет GridSearch. Выполнить notebook в контейнере и сохранить outputs можно без дополнительного служебного скрипта:

```powershell
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -e MPLBACKEND=Agg -v "${PWD}:/workspace" -w /workspace composites-thesis:local python -c "from pathlib import Path; import nbformat; from nbclient import NotebookClient; p=Path('notebooks/research.ipynb'); n=nbformat.read(p,as_version=4); NotebookClient(n,timeout=600,kernel_name='python3',resources={'metadata':{'path':'/workspace'}}).execute(); nbformat.write(n,p)"
```

## Streamlit-приложение

Приложение запускается из read-only bind mount:

```powershell
docker run --rm -e PYTHONDONTWRITEBYTECODE=1 -p 127.0.0.1:8501:8501 -v "${PWD}:/workspace:ro" -w /workspace composites-thesis:local streamlit run app/streamlit_app.py --server.address=0.0.0.0 --browser.gatherUsageStats=false
```

Откройте `http://localhost:8501`. Значения формы по умолчанию — медианы train, а не измеренный образец. Для содержательного прогноза введите фактические 11 характеристик. Выход за train min/max сопровождается предупреждением об экстраполяции. Приложение выдаёт модуль в ГПа и прочность в МПа; метрики test описывают модель в целом и не являются неопределённостью отдельного прогноза.

## Результаты

Train/CV-выбор опубликован в `reports/modeling`, финальная оценка — в `reports/evaluation`. Метрики выбранных моделей на test:

| Задача | Модель | RMSE | MAE | R² |
|---|---|---:|---:|---:|
| Модуль упругости при растяжении | baseline | 3,178 ГПа | 2,556 ГПа | -0,016 |
| Прочность при растяжении | Gradient Boosting | 471,410 МПа | 380,152 МПа | 0,008 |
| Отношение матрица–наполнитель | MLP | 0,944 | 0,777 | -0,040 |

Для модуля испытанные сложные модели не улучшили константную модель. Для прочности RMSE Gradient Boosting на отложенной выборке равна 471,410 МПа против 473,288 МПа у константной модели, однако R² остаётся около нуля. MLP хуже константной модели: её RMSE равна 0,944 против 0,926. Отрицательный R² означает, что сумма квадратов ошибок превышает сумму квадратов отклонений от среднего целевой переменной оцениваемой выборки. Это возможно и для `DummyRegressor`, который использует среднее обучающей, а не отложенной выборки. Полученные результаты относятся только к испытанным моделям, сеткам и случайно отложенной части данной таблицы; они не подтверждают практическую точность.

## Ограничения

Неизвестны происхождение наблюдений, партии и серии, типы матрицы и наполнителя, а также смысл части единиц. Случайное разбиение оценивает качество на отложенных строках той же таблицы при данном разбиении; перенос на новые классы материалов не проверен. Прогнозы не заменяют лабораторные испытания.

Публичный репозиторий: <https://github.com/fpvcstep/composites-property-prediction>.
