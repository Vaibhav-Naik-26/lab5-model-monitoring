import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

DEFAULT_PAYLOAD = {
    "features": [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0]
}


async def one_request(client: httpx.AsyncClient, url: str, payload: dict) -> float:
    started = time.perf_counter()
    response = await client.post(url, json=payload)
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    rank = (len(values) - 1) * p / 100
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    weight = rank - lo
    return values[lo] * (1 - weight) + values[hi] * weight


async def run(url: str, requests: int, concurrency: int) -> list[float]:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=30.0) as client:
        async def bounded() -> float:
            async with sem:
                return await one_request(client, url, DEFAULT_PAYLOAD)

        return await asyncio.gather(*(bounded() for _ in range(requests)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Full /predict URL")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    print(f"Warm-up: {args.warmup} requests")
    asyncio.run(run(args.url, args.warmup, min(args.concurrency, args.warmup)))

    print(
        f"Load test: {args.requests} requests at up to {args.concurrency} concurrent requests"
    )
    latencies = asyncio.run(run(args.url, args.requests, args.concurrency))

    result = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "p50_ms": round(percentile(latencies, 50), 2),
        "p95_ms": round(percentile(latencies, 95), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
    }

    out = Path("load_test_results.json")
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
