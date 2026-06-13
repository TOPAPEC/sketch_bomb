#!/usr/bin/env python3
"""
bench_metrics.py — нагрузочный замер времени ответа сервиса Sketch Bomb
(скетч → чистое изображение) с экспортом метрик в формате Prometheus.

Что делает:
  1. Берёт N случайных скетчей из DomainNet (разные классы).
  2. Шлёт каждый в POST /api/generate, парсит SSE-поток.
  3. Логирует время ответа: суммарное и по стадиям пайплайна
     (classify → domainnet → generate → rembg → score → background).
  4. Отдаёт метрики на /metrics (Prometheus стягивает их → Grafana рисует).
  5. Пишет JSON-сводку для отчётности.

Метрики Prometheus:
  sketch_request_duration_seconds        Histogram  — полное время ответа
  sketch_stage_duration_seconds{stage}   Histogram  — время каждой стадии
  sketch_requests_total{status}          Counter    — ok / error
  sketch_siglip_score                    Histogram  — качество выбранного кандидата
  sketch_last_request_duration_seconds   Gauge      — последний замер
  sketch_benchmark_runs_total            Counter    — завершённые прогоны
  sketch_benchmark_in_progress           Gauge      — идёт ли прогон
"""
import argparse
import base64
import glob
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import requests
from PIL import Image
from prometheus_client import (
    Counter, Gauge, Histogram, start_http_server,
)

# ── Конфигурация по умолчанию ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOMAINNET_ROOT = PROJECT_ROOT / "data" / "domainnet" / "sketch"
DEFAULT_URL = "http://localhost:8000"
LOG_PATH = "/var/log/portal/sketch_bench.log"
SUMMARY_PATH = PROJECT_ROOT / "webapp" / "bench_results.json"

STAGES = ["classify", "domainnet", "generate", "rembg", "score", "background"]

# Бакеты подобраны под наблюдаемые задержки (одиночный запрос ~18 c):
TOTAL_BUCKETS = (1, 2, 5, 8, 10, 12, 15, 18, 20, 25, 30, 40, 60, float("inf"))
STAGE_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 18, 25, float("inf"))

# ── Метрики Prometheus ──────────────────────────────────────────────────────
REQ_DURATION = Histogram(
    "sketch_request_duration_seconds",
    "Полное время ответа сервиса на генерацию из скетча",
    buckets=TOTAL_BUCKETS,
)
STAGE_DURATION = Histogram(
    "sketch_stage_duration_seconds",
    "Время выполнения отдельной стадии пайплайна",
    ["stage"],
    buckets=STAGE_BUCKETS,
)
REQ_TOTAL = Counter(
    "sketch_requests_total",
    "Число запросов к сервису по статусу",
    ["status"],
)
SIGLIP_SCORE = Histogram(
    "sketch_siglip_score",
    "SigLIP2-оценка выбранного кандидата",
    buckets=(0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0),
)
LAST_DURATION = Gauge(
    "sketch_last_request_duration_seconds",
    "Время ответа последнего обработанного запроса",
)
RUNS_TOTAL = Counter(
    "sketch_benchmark_runs_total",
    "Число полностью завершённых прогонов бенчмарка",
)
IN_PROGRESS = Gauge(
    "sketch_benchmark_in_progress",
    "1 пока бенчмарк выполняется, иначе 0",
)

# ── Логирование времени ответа ────────────────────────────────────────────
log = logging.getLogger("sketch_bench")
log.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
log.addHandler(_sh)
# Под supervisor stdout уже перенаправляется в LOG_PATH. Отдельный FileHandler
# заводим только при автономном запуске (вне supervisor), чтобы не дублировать строки.
if not os.environ.get("SUPERVISOR_ENABLED"):
    try:
        _fh = logging.FileHandler(LOG_PATH)
        _fh.setFormatter(_fmt)
        log.addHandler(_fh)
    except OSError:
        pass  # каталог логов недоступен — пишем только в stdout


def pick_random_sketches(n, seed=None):
    """Выбрать n случайных скетчей из случайных классов DomainNet."""
    rng = random.Random(seed)
    classes = [d for d in DOMAINNET_ROOT.iterdir() if d.is_dir()] if DOMAINNET_ROOT.exists() else []
    if not classes:
        raise SystemExit(f"Нет данных DomainNet в {DOMAINNET_ROOT}")
    picks = []
    for _ in range(n):
        cls = rng.choice(classes)
        files = glob.glob(str(cls / "*.jpg")) + glob.glob(str(cls / "*.png"))
        if not files:
            continue
        picks.append((cls.name, rng.choice(files)))
    return picks


def encode_image(path):
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def run_one(url, image_b64, model, best_of, selector, remove_bg, timeout):
    """Один запрос к /api/generate. Возвращает dict с таймингами и метаданными."""
    payload = {
        "image": image_b64, "model": model, "best_of": best_of,
        "selector": selector, "remove_bg": remove_bg, "seed": -1,
    }
    stage_start = {}
    stage_dur = {}
    label = None
    siglip_pick = None
    t0 = time.perf_counter()
    r = requests.post(f"{url}/api/generate", json=payload, stream=True, timeout=timeout)
    r.raise_for_status()
    for raw in r.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", "replace")
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        ev = json.loads(data)
        et = ev.get("type")
        if et == "stage_start":
            stage_start[ev["stage"]] = time.perf_counter()
        elif et == "stage_done":
            s = ev["stage"]
            if s in stage_start:
                stage_dur[s] = time.perf_counter() - stage_start[s]
            d = ev.get("data", {})
            if s == "classify":
                label = d.get("label")
            elif s == "score":
                scores = d.get("siglip_scores") or []
                idx = d.get("pick_idx", 0)
                if scores and idx < len(scores):
                    siglip_pick = scores[idx]
        elif et == "error":
            raise RuntimeError(ev.get("message", "unknown error"))
    total = time.perf_counter() - t0
    return {
        "total_s": total, "stages": stage_dur,
        "label": label, "siglip_score": siglip_pick,
    }


def benchmark(args):
    picks = pick_random_sketches(args.n, seed=args.seed)
    log.info("Старт бенчмарка: n=%d model=%s best_of=%d selector=%s remove_bg=%s url=%s",
             len(picks), args.model, args.best_of, args.selector, args.remove_bg, args.url)
    IN_PROGRESS.set(1)
    results = []
    for i, (true_cls, path) in enumerate(picks, 1):
        fname = os.path.basename(path)
        try:
            b64 = encode_image(path)
            res = run_one(args.url, b64, args.model, args.best_of,
                          args.selector, args.remove_bg, args.timeout)
        except Exception as e:  # noqa: BLE001
            REQ_TOTAL.labels(status="error").inc()
            log.error("[%2d/%d] ОШИБКА class=%s file=%s: %s", i, len(picks), true_cls, fname, e)
            results.append({"seq": i, "true_class": true_cls, "file": fname,
                            "status": "error", "error": str(e)})
            continue

        REQ_TOTAL.labels(status="ok").inc()
        REQ_DURATION.observe(res["total_s"])
        LAST_DURATION.set(res["total_s"])
        for stg, dur in res["stages"].items():
            STAGE_DURATION.labels(stage=stg).observe(dur)
        if res["siglip_score"] is not None:
            SIGLIP_SCORE.observe(res["siglip_score"])

        stage_str = " ".join(f"{s}={res['stages'].get(s, 0):.2f}" for s in STAGES if s in res["stages"])
        log.info("[%2d/%d] OK class=%-18s pred=%-18s total=%5.2fs siglip=%s | %s",
                 i, len(picks), true_cls, res["label"], res["total_s"],
                 f"{res['siglip_score']:.3f}" if res['siglip_score'] is not None else "n/a", stage_str)
        results.append({
            "seq": i, "true_class": true_cls, "file": fname, "status": "ok",
            "total_s": round(res["total_s"], 3),
            "predicted_label": res["label"],
            "siglip_score": res["siglip_score"],
            "stages": {k: round(v, 3) for k, v in res["stages"].items()},
        })

    IN_PROGRESS.set(0)
    RUNS_TOTAL.inc()
    write_summary(args, results)
    return results


def write_summary(args, results):
    ok = [r for r in results if r["status"] == "ok"]
    totals = sorted(r["total_s"] for r in ok)

    def pct(p):
        if not totals:
            return None
        k = max(0, min(len(totals) - 1, int(round((p / 100) * (len(totals) - 1)))))
        return round(totals[k], 3)

    # средние по стадиям
    stage_avg = {}
    for s in STAGES:
        vals = [r["stages"].get(s) for r in ok if s in r.get("stages", {})]
        if vals:
            stage_avg[s] = round(sum(vals) / len(vals), 3)

    summary = {
        "config": {
            "n": args.n, "model": args.model, "best_of": args.best_of,
            "selector": args.selector, "remove_bg": args.remove_bg,
            "url": args.url, "seed": args.seed,
        },
        "count_ok": len(ok),
        "count_error": len(results) - len(ok),
        "total_seconds": {
            "min": totals[0] if totals else None,
            "max": totals[-1] if totals else None,
            "mean": round(sum(totals) / len(totals), 3) if totals else None,
            "p50": pct(50), "p90": pct(90), "p95": pct(95),
        },
        "stage_avg_seconds": stage_avg,
        "requests": results,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log.info("Сводка записана в %s", SUMMARY_PATH)
    t = summary["total_seconds"]
    if t["mean"] is not None:
        log.info("ИТОГО ok=%d err=%d | total mean=%.2fs p50=%.2fs p90=%.2fs min=%.2fs max=%.2fs",
                 summary["count_ok"], summary["count_error"],
                 t["mean"], t["p50"], t["p90"], t["min"], t["max"])


def main():
    ap = argparse.ArgumentParser(description="Бенчмарк времени ответа Sketch Bomb с экспортом в Prometheus")
    ap.add_argument("-n", type=int, default=10, help="число скетчей (по умолчанию 10)")
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default="lightning", choices=["sd15", "sdxl", "lightning"])
    ap.add_argument("--best-of", type=int, default=4)
    ap.add_argument("--selector", default="siglip", choices=["siglip", "siglip_multi", "kimi"])
    ap.add_argument("--remove-bg", dest="remove_bg", action="store_true", default=True)
    ap.add_argument("--no-remove-bg", dest="remove_bg", action="store_false")
    ap.add_argument("--seed", type=int, default=None, help="seed выборки скетчей (для воспроизводимости)")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--metrics-port", type=int, default=19808)
    ap.add_argument("--metrics-addr", default="127.0.0.1")
    ap.add_argument("--keep-alive", action="store_true",
                    help="после прогона держать /metrics живым (для Prometheus/Grafana)")
    ap.add_argument("--loop-every", type=int, default=0,
                    help="если >0 — повторять прогон каждые N секунд (непрерывный сбор)")
    args = ap.parse_args()

    start_http_server(args.metrics_port, addr=args.metrics_addr)
    log.info("Метрики Prometheus: http://%s:%d/metrics", args.metrics_addr, args.metrics_port)

    benchmark(args)

    if args.loop_every > 0:
        while True:
            time.sleep(args.loop_every)
            benchmark(args)
    elif args.keep_alive:
        log.info("Прогон завершён. Держу /metrics живым (Ctrl-C для выхода).")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    main()
