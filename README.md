# Прогнозирование свойств композиционных материалов

Учебный проект ВКР по загрузке, исследованию и моделированию свойств композитов. На текущем этапе реализованы воспроизводимая среда Python 3.12, строгая проверка двух исходных таблиц и их INNER JOIN по исходному индексу. Обучение моделей, статистическая очистка и EDA ещё не выполнялись.

## Данные

Разрешённые к публикации исходные файлы находятся в `data/raw/`. Их происхождение и контрольные суммы описаны в `data/raw/README.md`. Первая колонка каждого файла служит ключом соединения и не используется как признак.

## Запуск в Docker

Требуются Docker Desktop и Linux containers.

```powershell
docker build -t composites-thesis:local .
docker run --rm composites-thesis:local
docker run --rm -v "${PWD}:/workspace" -w /workspace composites-thesis:local python scripts/audit_data.py
```

Последняя команда обновляет `reports/data_audit.json` в рабочем каталоге.

## Запуск в обычной виртуальной среде

Требуется CPython 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.lock.txt
.venv\Scripts\python -m pip install --no-deps -e .
.venv\Scripts\python -m pytest -q
.venv\Scripts\python scripts/audit_data.py
```

## Результаты текущего этапа

- `reports/data_audit.json` — машиночитаемый аудит файлов и объединения;
- `docs/data_dictionary.md` — словарь полей, схемы целей и известные ограничения;
- `tests/test_data.py` — проверки индекса, схемы и ожидаемых потерь при соединении.

Неподтверждённые единицы и происхождение наблюдений не додумываются. Две колонки модуля упругости имеют разные имена и роли и не смешиваются.

`requirements.txt` фиксирует прямые зависимости проекта, а `requirements.lock.txt` — полный набор версий, фактически проверенный в Docker на Python 3.12.
