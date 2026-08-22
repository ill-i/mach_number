# Автоматический HAWC+ / GETSOURCES / SF pipeline

Конвейер обрабатывает все multi-extension FITS из `data_hawc`, запускает
двухпроходный GETSOURCES на Stokes I, сравнивает поляризационные структурные
функции внутри и вне надёжных источников и собирает общий TeX/PDF-отчёт.

## Запуск

Из этой директории:

```bash
python run_hawc_pipeline.py
```

Ручные ответы GETSOURCES не требуются. Стандартный ввод закрыт, поэтому
подтверждение параметров принимает значение по умолчанию `yes`, а удаление
промежуточных файлов — безопасное значение `no`.

Полезные режимы:

```bash
# Только извлечь I, err_I, Q, err_Q, U, err_U, qual_I и подготовить Stokes I
python run_hawc_pipeline.py --prepare-only

# Повторить только SF и отчёт из уже готовых GETSOURCES-продуктов
python run_hawc_pipeline.py --skip-getsources

# Обработать один или несколько объектов
python run_hawc_pipeline.py --objects Fil1_RA_Original G24_RA_Original

# Использовать более полный reliable+tentative каталог для проверки устойчивости
python run_hawc_pipeline.py --skip-getsources --catalog relitent
```

## Что сохраняется

- `work/<object>/prepared/components/` — отдельные FITS для всех входных HDU;
- `work/<object>/prepared/getsources/` — Stokes I в MJy/sr и безопасная маска;
- `work/<object>/getsources_runs/` — четыре стадии GETSOURCES и полные логи;
- `report/objects/<object>/` — CSV, FITS-маски и диагностические PDF;
- `report/all_segment_results.csv` — единая таблица всех объектов;
- `report/hawc_pipeline_report.tex` и `.pdf` — итоговый отчёт;
- `pipeline_manifest.json` — параметры и статус каждого входного файла.

## Логика сегментов и источников

Сегменты определяются по дебиасированному поляризационному S/N. Адаптивный
порог площади равен `Smin = min_beams * pixels_per_beam`.

Для сегмента без пересечения с надёжным GETSOURCES-source строится только
исходная SF. Для сегмента с источником строятся четыре панели:

1. `with_sources`;
2. `without_sources`;
3. `sources_only`;
4. все три кривые вместе.

Каждая диагностическая панель использует логику `plot_sf_diagnostic` из
`SF_analysis.py`, включая линию случайного шума `Dpsi=0.5`.

GETSOURCES никогда не меняет Q и U: footprints используются только как
пространственная маска. Одно число Маха не интерпретируется для разорванной
маски; в таком случае SF сохраняется, а `MA_status=disconnected_mask`.

По умолчанию используются только `reliable` источники без дополнительного
расширения footprints. Сегменты, где sources-only занимает более 70% площади,
явно отмечаются в отчёте.

## Текущие параметры

- поляризационный порог: `S/N >= 3`;
- минимальная площадь: `10` лучей;
- GETSOURCES: `sreliable=5`, `stentative=3`;
- максимальный размер источника: `120 arcsec`;
- полный набор из `99` пространственных масштабов;
- `savespace=0` для сохранения промежуточных данных.

Для `Snake_RA_Original.fits` ключ `PIXSCAL=9.43` противоречит WCS, который
даёт `4.55 arcsec/pixel`; конвейер использует WCS. Отсутствующий `BEAM`
восстанавливается как номинальные `18.2 arcsec` для 216 мкм, и это записано в
metadata/отчёт.
