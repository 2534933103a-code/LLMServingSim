"""End-to-end vLLM benchmark + simulator validation.

``python -m bench run``
    Run a vLLM serving benchmark against a given dataset/model/engine config
    and write per-request timestamps + per-tick metrics under bench/results/.

``python -m bench run_disagg``
    Run a PD-separated vLLM benchmark via HTTP proxy (prefill → decode).
    Requires ``./bench/bench_disagg.sh`` to launch the prefill/decode/proxy
    services first.

``python -m bench validate``
    Compare a finished bench run against simulator output for the same
    dataset/cluster, produce throughput / running-waiting / latency-CDF
    plots and a numeric TTFT/TPOT/throughput summary.

Module map:
    __main__.py                 CLI dispatch (run / run_disagg / validate)
    core/                       internals
        runner.py               AsyncLLM driver (vLLM v1, colocated)
        runner_disagg.py        HTTP-based driver (PD-separated, via proxy)
        recorder.py             writes meta.json / requests.jsonl / timeseries.csv
        stat_logger.py          custom vLLM StatLoggerBase that fills timeseries
        validate.py             bench-vs-sim comparison
        plots.py                throughput / running-waiting / latency-CDF helpers
        logger.py               Rich-based logger + stdio capture
    bench.sh                    host-side ``python -m bench run`` wrapper
    bench_disagg.sh             host-side PD-separated (P2pNcclConnector) wrapper
    disagg_proxy.py             PD disagg proxy (P2pNcclConnector, quart-based)
    bench_nixl.sh               host-side Nixl PD-separated (XpYd) wrapper
    nixl_proxy.py               Nixl proxy server (httpx + fastapi, XpYd)
    validate.sh                 host-side ``python -m bench validate`` wrapper
    results/                    output: bench/results/<run_id>/

Output schema (one bench run)::

    bench/results/<run_id>/
      meta.json              run metadata (model, vLLM version, engine kwargs,
                             dataset hash, wall-clock start/end)
      requests.jsonl         one row per request: arrival_time, queued_ts,
                             scheduled_ts, first_token_ts, last_token_ts,
                             input_toks, output_toks
      timeseries.csv         per-tick: t, prompt_throughput, gen_throughput,
                             running, waiting, kv_cache_pct
"""

__version__ = "0.1.0"
