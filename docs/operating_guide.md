# Operating Guide

Пошаговая инструкция по запуску пайплайна на локальной машине.

## Требования

- Python 3.10 или новее
- Интернет-доступ (для обращения к LLM через OpenRouter; скачивание датасета — только при использовании канала URL)
- Минимум $1 на балансе OpenRouter (один полный прогон стоит ~$0.05)
- Рекомендуемая среда: VS Code с расширениями Python и Jupyter, или JupyterLab

## 1. Получение API-ключа OpenRouter

1. Регистрация: https://openrouter.ai
2. Пополнение баланса: https://openrouter.ai/credits (минимум $5)
3. Создание ключа: https://openrouter.ai/keys — **Create Key**, скопировать значение `sk-or-v1-...`

## 2. Подготовка окружения

```bash
cd readmission-risk-agents
python3 -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows CMD
.venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
```

Установка занимает 2–4 минуты.

## 3. Настройка секретов

```bash
cp config/settings.env.template config/settings.env
```

Открыть `config/settings.env` в редакторе, вставить значение в строку `OPENROUTER_API_KEY=`. Минимальная валидная конфигурация:

```
OPENROUTER_API_KEY=sk-or-v1-...
LANGCHAIN_API_KEY=
DATASET_URL=
```

## 4. Передача входных данных

Выбрать один из двух каналов.

### Канал 1. Ссылка на облачное хранилище

Загрузить файл в Google Drive, Dropbox или на публичный HTTPS-сервер. Вставить ссылку в `settings.env`:

```
DATASET_URL=https://drive.google.com/file/d/1abc.../view
```

Для Google Drive файл должен быть открыт для чтения по ссылке.

### Канал 2. Локальная папка

Положить файл (`.csv`, `.tsv`, `.xlsx` или `.zip`) в папку `input/` в корне проекта. Оставить `DATASET_URL` пустым.

Подробности по каналам и требованиям к формату — `data_sources.md`.

## 5. Запуск ноутбука

### Вариант A. VS Code

1. File → Open Folder → выбрать папку `readmission-risk-agents`
2. В Explorer открыть `notebooks/pipeline.ipynb`
3. В правом верхнем углу ноутбука нажать Select Kernel → выбрать `.venv`
4. Run All (двойная стрелка сверху) или прожимать ячейки по одной через Shift+Enter

### Вариант B. JupyterLab

```bash
jupyter lab notebooks/pipeline.ipynb
```

## 6. Контрольные точки

### После ячейки загрузки конфигурации

Должна появиться строка:

```
Settings loaded: OPENROUTER_API_KEY=set, tracing=false, data_source=URL
```

или

```
Settings loaded: OPENROUTER_API_KEY=set, tracing=false, data_source=local input/ folder
```

Поле `data_source` показывает, какой канал будет использован.

### Во время выполнения графа

В выводе ячейки `result = graph.invoke(initial_state)` появятся логи по этапам. Типичный прогон занимает 3–7 минут.

### После завершения

В папке `artifacts/` появятся выходные файлы. В VS Code они видны в Explorer после обновления.

## 7. Проверка результатов

- `artifacts/priority_records.csv` — открыть в Excel, VS Code или через `pandas.read_csv`
- `artifacts/run_report.html` — открыть в браузере (двойной клик на файле в системном файловом менеджере)
- `artifacts/best_metrics.json` — читается как обычный JSON

## 8. Повторный запуск

Пайплайн поддерживает сравнение результатов между запусками через долговременную память. При повторном запуске `load_memory` прочитает `artifacts/best_metrics.json` и сравнит новый результат с предыдущим.

Для чистого старта без сравнения — удалить `artifacts/best_metrics.json`.

## 9. Обновление входных данных

### Для канала URL

Чтобы форсировать повторное скачивание (если файл в облаке обновился):

```bash
rm data/*
```

При следующем запуске агент скачает файл заново.

### Для канала локальной папки

Положить новый файл в `input/`. Если в папке уже есть старый, агент всё равно выберет самый свежий по времени изменения — старый можно оставить как архив.

## Типичные проблемы

### `ModuleNotFoundError: langchain_openai`

Ядро Jupyter подключено не к `.venv`. В VS Code: Select Kernel → выбрать `.venv`.

### `AuthenticationError: 401 Unauthorized`

- Ключ неверный или содержит лишние символы (пробелы, кавычки)
- Не пополнен баланс OpenRouter
- Ключ отозван

Проверить баланс: https://openrouter.ai/credits.

### `FileNotFoundError: config/settings.env`

Ноутбук запущен не из корня проекта. Закрыть и открыть папку `readmission-risk-agents` заново.

### `Could not obtain dataset: No dataset source available`

Не задан ни один канал. Либо заполнить `DATASET_URL` в `settings.env`, либо положить файл в `input/`.

### `Could not obtain dataset: <URL error>`

Ошибка скачивания по URL. Возможные причины:
- Неверный URL
- Файл в Google Drive не открыт для чтения по ссылке
- Временные проблемы с облачным хранилищем
- Корпоративный firewall блокирует домен

Решения: проверить URL в браузере, открыть права доступа, переключиться на канал 2 (положить файл в `input/`).

### Пайплайн зависает на `graph.invoke`

Проблема на стороне OpenRouter или провайдера модели. Нажать Interrupt в Jupyter, подождать минуту, запустить ячейку повторно. Состояние в `artifacts/` не повреждается.

### `OpenRouter key exhausted: usage >= limit`

Ключ исчерпал установленный на него лимит. Варианты:
- Пополнить баланс на https://openrouter.ai/credits
- Поднять лимит ключа на https://openrouter.ai/keys (Edit → Credit Limit)

### `OpenRouter key check failed (401)`

Ключ отозван или введён неверно. Проверить значение `OPENROUTER_API_KEY` в `config/settings.env`. При необходимости создать новый ключ на https://openrouter.ai/keys.

### `OpenRouter unreachable`

Нет сетевого доступа к openrouter.ai. Проверить интернет-соединение. Если за корпоративным firewall — добавить домен в белый список.

### Preflight подменил модель на замену

Это нормальное поведение — не ошибка. В выводе ячейки preflight появится строка вида:
```
gemini: google/gemini-2.5-flash-lite -> UNAVAILABLE, replaced with google/gemini-2.5-flash
```

Это означает, что исходная модель снята с OpenRouter и автоматически заменена на совместимый аналог из резервного списка. Пайплайн продолжит работу со свежей моделью. Чтобы зафиксировать новый выбор — обновить идентификатор в вызове `preflight(...)` в коде ноутбука.


## Стоимость прогона

Оценка при текущей конфигурации (8–12 LLM-вызовов за прогон):

- GPT-4o: $0.01–0.03
- Gemini 2.5 Flash Lite: $0.001
- Claude 3.5 Haiku: $0.01–0.02

Итого ~$0.05 за полный прогон. $5 баланса хватает на ~100 прогонов.
