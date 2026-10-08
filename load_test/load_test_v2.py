import argparse, json, time
from concurrent.futures import ThreadPoolExecutor
import httpx
import numpy as np
from load_test import DEFAULT_PAYLOAD

p = argparse.ArgumentParser()
p.add_argument("--url", required=True)
p.add_argument("--requests", type=int, default=500)
p.add_argument("--concurrency", type=int, default=10)
p.add_argument("--warmup", type=int, default=20)
a = p.parse_args()

client = httpx.Client(timeout=30)

def one(_):
    # try up to 5 times if the network lookup fails
    for attempt in range(5):
        try:
            t = time.perf_counter()
            r = client.post(a.url, json=DEFAULT_PAYLOAD)
            ms = (time.perf_counter() - t) * 1000
            return ms if r.status_code == 200 else None
        except httpx.ConnectError:
            time.sleep(0.5)
    return None

print(f"Warm-up: {a.warmup} requests (excluded from results)")
for i in range(a.warmup):
    one(i)

print(f"Load test: {a.requests} requests, {a.concurrency} concurrent")
with ThreadPoolExecutor(a.concurrency) as ex:
    results = list(ex.map(one, range(a.requests)))

ok = [x for x in results if x is not None]
summary = {
    "requests": a.requests,
    "concurrency": a.concurrency,
    "successful": len(ok),
    "failed": a.requests - len(ok),
    "p50_ms": round(float(np.percentile(ok, 50)), 2),
    "p95_ms": round(float(np.percentile(ok, 95)), 2),
    "max_ms": round(max(ok), 2),
    "mean_ms": round(float(np.mean(ok)), 2),
}
print(json.dumps(summary, indent=2))
with open("load_test_results.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: load_test_results.json")
