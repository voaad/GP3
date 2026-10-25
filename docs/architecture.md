# Architecture

## Обзор

Пайплайн реализован на LangGraph как ориентированный граф с условными ветвлениями. Состояние передаётся между узлами через `TypedDict`, долговременная память реализована файловым хранилищем в `artifacts/`.

## LLM-шлюз

Все запросы к языковым моделям идут через OpenRouter. Три модели зарегистрированы в пайплайне:

| Модель | OpenRouter ID | Temperature | Max tokens | Top-p | Этапы использования |
|---|---|---|---|---|---|
| GPT-4o | `openai/gpt-4o` | 0 | 800 | 0.9 | `explore_data`, `engineer_features` |
| Gemini 2.5 Flash Lite | `google/gemini-2.5-flash-lite` | 0 | 500 | 0.85 | `ingest_data`, `train_models` |
| Claude 3.5 Haiku | `anthropic/claude-3.5-haiku` | 0.1 | 600 | 0.95 | `preprocess`, `evaluate_models` |

Выбор параметров: `T=0` для этапов с требованием детерминированного JSON-вывода, `T=0.1` для этапов верификации, где допустимо лёгкое расширение формулировок. `top_p` варьируется от консервативного 0.85 (короткие решения) до 0.95 (self-critique с более широким охватом возможных замечаний).

## Техники промптинга

- **Role prompting** — каждому этапу задаётся специализированная роль (clinical data scientist, preprocessing specialist, feature engineer, ML engineer).
- **Two-step prompting** (`ingest_data`) — сначала определяется предметная тема датасета, затем результат первого шага используется как контекст при выборе целевой переменной.
- **Structured step-by-step** (`explore_data`) — LLM получает пронумерованный список шагов EDA и возвращает JSON с результатами по каждому.
- **Self-critique** (`preprocess`) — модель предлагает план предобработки и в том же ответе проверяет его на риск утечки таргета.
- **Contrastive prompting** (`engineer_features`) — предлагаются три стратегии, проводится сравнение, выбираются две.
- **Metric review** (`evaluate_models`) — вторая модель верифицирует выбор метрики и лучшей модели, формирует бизнес-интерпретацию.

## Источники данных

Ingestion-слой поддерживает два канала приёма данных (подробности — `data_sources.md`):

| Канал | Активация | Поведение |
|---|---|---|
| URL облачного хранилища | Заполнен `DATASET_URL` в `settings.env` | Скачивание с кэшированием в `data/`, автоматическая нормализация URL для Google Drive и Dropbox |
| Локальная папка | `DATASET_URL` пуст, в `input/` есть файл | Выбор самого свежего файла по времени изменения |

При наличии обоих источников приоритет отдаётся URL.

Функция `resolve_dataset` возвращает кортеж `(локальный_путь, source_channel)`. Идентификатор канала фиксируется в `llm_decisions` для последующего анализа в логах.


## Pre-flight проверка

Перед построением графа выполняется самодиагностика инфраструктуры:

### Валидация API-ключа

Запрос `GET https://openrouter.ai/api/v1/key` с `Authorization: Bearer <OPENROUTER_API_KEY>`. Проверяется:

- Статус ключа (активен / отозван)
- Текущая сумма потребления (`usage`)
- Установленный лимит (`limit`)
- Флаг `is_free_tier`

Если `usage >= limit` или ключ отклонён сервером (HTTP 4xx), пайплайн останавливается с сообщением, в котором указана причина и ссылка на пополнение баланса.

### Проверка доступности моделей

Запрос `GET https://openrouter.ai/api/v1/models` возвращает полный каталог активных моделей OpenRouter. Каждый из трёх назначенных идентификаторов сверяется со списком.

Если какая-то модель отсутствует в каталоге (снята провайдером, переименована), запускается механизм автоматической замены:

1. Для роли (`analytical`, `fast_json`, `verifier`) выбирается подсписок кандидатов из `ROLE_CANDIDATES`.
2. Кандидаты фильтруются по пересечению с живым каталогом.
3. LLM-арбитр (Claude для analytical/fast_json, Gemini для verifier — чтобы не спрашивать ломающуюся модель о её замене) получает список живых кандидатов и выбирает лучший.
4. Новый идентификатор подставляется в соответствующий `ChatOpenAI`-клиент.
5. Если LLM-арбитр недоступен, берётся первый живой кандидат из приоритизированного списка.

### Резервные списки моделей

```python
ROLE_CANDIDATES = {
    "analytical": [
        "openai/gpt-4o",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-sonnet-4.5",
    ],
    "fast_json": [
        "google/gemini-2.5-flash-lite",
        "google/gemini-2.5-flash",
        "google/gemini-2.0-flash-001",
        "openai/gpt-4o-mini",
    ],
    "verifier": [
        "anthropic/claude-3.5-haiku",
        "anthropic/claude-haiku-4.5",
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
    ],
}
```

Списки упорядочены по предпочтению. Первый элемент каждого списка — текущий основной выбор. Чтобы добавить новую модель в резерв — дописать её в соответствующий список; при следующем запуске пайплайн автоматически учтёт её.


## Узлы графа

| Узел | LLM | Действия |
|---|---|---|
| `load_memory` | — | Чтение `artifacts/best_metrics.json`, сброс сессионных полей |
| `ingest_data` | Gemini | Получение датасета из выбранного источника, генерация `clinical_note`, two-step prompting (тема → таргет), классификация колонок |
| `explore_data` | GPT-4o | Пропуски, баланс классов, описательные статистики, флаг балансировки |
| `preprocess` | Claude | Стратегии импутации, IQR-фильтр, encoding, self-critique |
| `engineer_features` | GPT-4o | Interaction / ratio / aggregation признаки + текстовые (длина, счётчик маркеров) |
| `train_models` | Gemini | train/test split, undersampling по флагу, цикл по моделям × гиперпараметрам |
| `evaluate_models` | Claude | Выбор стратегии метрик, лексикографический выбор лучшей модели, верификация |
| `build_priority_list` | — | Ранжирование test-set по `predict_proba`, запись CSV |
| `save_memory` | — | Условная перезапись `best_metrics.json` при улучшении |
| `build_report` | — | Генерация HTML-отчёта |

## Нелинейные элементы

- Условное ребро после `ingest_data`: ошибка → END
- Условное ребро после `explore_data`: ошибка → END
- Условное ребро после `train_models`: ни одна модель не обучилась → END
- Внутренний цикл в `train_models`: перебор моделей × значений гиперпараметра с `try/except` на каждой итерации; ошибки увеличивают счётчик `retries`, но не останавливают пайплайн
- Ветвление в `train_models`: undersampling применяется только при подтверждённом флаге из `preprocess`
- Условная запись в `save_memory`: `best_metrics.json` перезаписывается только при `new_score >= prev_score`

## Память

**Кратковременная** — `TypedDict State` передаётся между всеми узлами в рамках одной сессии. Содержит: датафрейм в JSON-сериализации, имена колонок по типам, отчёты EDA и препроцессинга, выбор модели и метрик, след LLM-решений.

**Долговременная** — файловое хранилище `artifacts/`:

- `best_metrics.json` — метрика и имя лучшей модели за всю историю запусков
- `run_log.json` — хронологический лог всех запусков
- `{model_name}.pkl` — сериализованные модели

## Определение приоритетного класса

Приоритетный класс выбирается автоматически на этапе `preprocess` как миноритарный класс целевой переменной. Стратегия метрик определяется в `evaluate_models` на основе темы и названия таргета:

- `medical_high_risk_binary`: primary = recall_pos, secondary = precision_pos
- `medical_high_risk_multiclass`: primary = recall_priority_class, secondary = f1_weighted
- `general_binary`: primary = f1_weighted, secondary = roc_auc
- `general_multiclass`: primary = f1_macro, secondary = f1_weighted

## Внешние инструменты

| Категория | Библиотеки |
|---|---|
| Data | pandas, numpy |
| ML | scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier, SimpleImputer, LabelEncoder, resample, metrics) |
| Storage | joblib, json |
| Visualization | matplotlib, seaborn |
| Ingestion | urllib.request, zipfile |

## Валидация

Пайплайн валидирован на произвольных медицинских табличных датасетах размером от 1 000 до 100 000+ строк. Полный прогон — 3–7 минут на CPU.
