from __future__ import annotations
from collections import defaultdict, deque
import threading
import time
from app.gpu_metrics import GPUMetrics


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.counters = defaultdict(int)
        self.samples = defaultdict(lambda: deque(maxlen=1000))
        self.inferences = defaultdict(lambda: deque(maxlen=1000))
        self.started = time.monotonic()
        self.gpu = GPUMetrics()

    def increment(self, key):
        with self.lock:
            self.counters[key] += 1

    def latency(self, key, seconds):
        with self.lock:
            self.samples[key].append(seconds * 1000)

    def inference(self, camera_id):
        with self.lock:
            self.inferences[camera_id].append(time.monotonic())

    def snapshot(self):
        import psutil
        now = time.monotonic()
        with self.lock:
            latencies = {}
            for key, values in self.samples.items():
                ordered = sorted(values)
                latencies[key] = {'count': len(ordered), 'p50_ms': ordered[int((len(ordered) - 1) * .5)],
                                  'p95_ms': ordered[int((len(ordered) - 1) * .95)]}
            result = {'uptime_seconds': now - self.started, 'counters': dict(self.counters), 'latencies': latencies,
                      'inference_fps': {key: sum(now - t < 5 for t in ts) / 5 for key, ts in self.inferences.items()}}
        result.update(cpu_percent=psutil.cpu_percent(), memory_percent=psutil.virtual_memory().percent,
                      process_memory_bytes=psutil.Process().memory_info().rss)
        result.update(self.gpu.sample())
        return result
