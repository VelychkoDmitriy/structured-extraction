# Structured Extraction Service — резюме Work.ua

ДЗ 1 курсу AI Engineering (robot_dreams). Сервіс витягує з «брудних» текстів резюме структурований JSON за схемою `Resume` і порівнює моделі, версії промптів і семплінг на golden set.

- Дані: 60 резюме з [KSE Work_UA_resumes](https://huggingface.co/datasets/KSE-RESEARCH-Group/Work_UA_resumes), знеособлені, 20 з еталонною розміткою — [data/README.md](data/README.md)
- Схема: [src/schema.py](src/schema.py), правила розмітки: [data/LABELING_GUIDE.md](data/LABELING_GUIDE.md)
- Промпти: [prompts/](prompts/) + [CHANGELOG](prompts/CHANGELOG.md)
- Результати: `results/<модель>__<промпт>__<семплінг>/` і зведена таблиця [results/summary.md](results/summary.md)
- Звіт і вибір моделі: [REPORT.md](REPORT.md)
- Моделі: `gpt-4.1-mini` (OpenAI API) і `claude-haiku` — Claude Haiku 5.5 через OpenRouter; додатково прогнано `gpt-4o-mini` і `gemini-flash-lite` (Gemini 3.1 Flash-Lite)

## Встановлення
Потрібен Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Linux/macOS: cp .env.example .env — і впишіть ключі
```

## Запуск однією командою
```bash
python main.py
```
Проганяє сітку з `config.yaml` (моделі × промпти × семплінг) на всіх 60 текстах і будує `results/summary.md`. Потрібні ключі `OPENAI_API_KEY` і `OPENROUTER_API_KEY`; модель без ключа в `.env` пропускається. Готові результати кешуються: повторний запуск дозапитує лише відсутні документи.

Інші команди:
```bash
python main.py check                                     # перевірити golden set за схемою, без API
python main.py run --model gpt-4o-mini --prompt v1 --sampling A --split golden --limit 3
python main.py run --model claude-haiku --prompt v2 --sampling B --force   # перезапустити з нуля
python main.py report                                    # перебудувати summary.md
```

## Як це працює
1. Промпт `prompts/<v>.md` + схема `Resume` + текст резюме і дата резюме → модель.
2. Усі моделі отримують схему однаково: у промпті й як strict structured output (`response_format: json_schema`). OpenRouter — OpenAI-сумісний endpoint, тому клієнт один (`src/llm.py`), різниця лише в `config.yaml`.
3. Retry з backoff на мережеві помилки, таймаути, 429/5xx; timeout у `config.yaml`.
4. Кожна відповідь валідується `Resume`. Невалідна — одна спроба виправлення з текстом помилок, далі `failed`.
5. До кожного результату пишуться модель, версія промпту, параметри семплінгу, токени, латентність, вартість.
6. Метрики: точність по полях на golden set, % валідних, p50/p95 латентності, $ на 1000 резюме ([src/metrics.py](src/metrics.py)).
