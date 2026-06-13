# Sketch Bomb — отчёт по времени ответа сервиса

Замер латентности веб-сервиса генерации «скетч → чистое изображение»
(`POST /api/generate`) с экспортом метрик в **Prometheus** и визуализацией в
**Grafana**.

- **Дата прогона:** 2026-06-13
- **Конфигурация:** `model=lightning`, `best_of=4`, `selector=siglip`, `remove_bg=true`
- **Вход:** 10 случайных скетчей из 10 случайных классов DomainNet
- **Запросов:** 10 успешных, 0 ошибок

## Сводные показатели времени ответа (end-to-end)

| Метрика | Значение, с |
|---|---|
| min  | 10.69 |
| p50  | 12.27 |
| mean | 12.46 |
| p90  | 14.05 |
| p95  | 14.35 |
| max  | 14.35 |

## Разбивка по стадиям пайплайна (среднее)

| Стадия | Среднее, с | Доля |
|---|---|---|
| classify (BeiT)        | 0.03  | 0.2 % |
| domainnet (поиск)      | 1.15  | 9.3 % |
| **generate (диффузия, ControlNet, best_of=4)** | **10.39** | **83.4 %** |
| rembg (BiRefNet)       | 0.71  | 5.7 % |
| score (SigLIP2 отбор)  | 0.01  | 0.0 % |
| background (финал)     | 0.15  | 1.2 % |

**Вывод:** время ответа практически полностью определяется стадией диффузии
(≈83 %), т.к. при `best_of=4` генерируется 4 кандидата. Остальные стадии
(классификация, поиск по DomainNet, удаление фона BiRefNet, отбор SigLIP2)
суммарно дают ~2 с. Главный рычаг ускорения — число кандидатов `best_of` и/или
выбор более быстрого пайплайна.

## Что собирается в метриках

Экспортёр (`webapp/bench_metrics.py`) отдаёт на `/metrics`:

| Метрика | Тип | Смысл |
|---|---|---|
| `sketch_request_duration_seconds`      | Histogram | полное время ответа |
| `sketch_stage_duration_seconds{stage}` | Histogram | время каждой стадии |
| `sketch_requests_total{status}`        | Counter   | число запросов ok/error |
| `sketch_siglip_score`                  | Histogram | качество выбранного кандидата |
| `sketch_last_request_duration_seconds` | Gauge     | последний замер |
| `sketch_benchmark_in_progress`         | Gauge     | идёт ли прогон |

Средний SigLIP2-скор выбранного кандидата за прогон: **0.316**.

## Архитектура сбора

```
bench_metrics.py (:19808 /metrics)  ──scrape 5s──>  Prometheus (:19090)  ──>  Grafana (:13000)
   10 скетчей → POST /api/generate                   TSDB, 15d retention        дашборд "Sketch Bomb — время ответа"
```

Все три сервиса под supervisor, слушают `127.0.0.1` (свободных внешних портов на
инстансе нет). Доступ — через SSH-форвард (см. ниже).

## Как воспроизвести

```bash
# одиночный прогон 10 скетчей с экспортом метрик
source /venv/main/bin/activate
cd /workspace/sketch_bomb
python webapp/bench_metrics.py -n 10 --keep-alive

# через supervisor (сервис sketch_bench, держит /metrics живым)
supervisorctl start sketch_bench
tail -f /var/log/portal/sketch_bench.log
```

Полезные параметры: `-n N` (число скетчей), `--model {lightning,sdxl,sd15}`,
`--best-of K`, `--selector {siglip,siglip_multi,kimi}`, `--seed S`
(воспроизводимая выборка), `--loop-every SEC` (непрерывный сбор).

## Доступ к Grafana / Prometheus (SSH-форвард)

С машины пользователя:

```bash
ssh -p <SSH_PORT> -L 13000:127.0.0.1:13000 -L 19090:127.0.0.1:19090 root@<PUBLIC_IP>
# Grafana:    http://localhost:13000  (дашборд «Sketch Bomb — время ответа», логин admin/admin)
# Prometheus: http://localhost:19090
```

Артефакты: дашборд `/opt/metrics-stack/dashboards/sketchbomb.json`,
сводка прогона `webapp/bench_results.json`, лог времени ответа
`/var/log/portal/sketch_bench.log`.
