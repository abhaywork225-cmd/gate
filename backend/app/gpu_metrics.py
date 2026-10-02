from __future__ import annotations
"""Portable optional NVIDIA telemetry; unavailable fields remain null."""
import csv
import io
import os
import re
import shutil
import subprocess
import threading
import time


class GPUMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_sample = 0
        self.cached = {}

    def _sample_tegrastats(self):
        exe = shutil.which('tegrastats')
        if not exe:
            return None
        try:
            proc = subprocess.Popen([exe, '--interval', '250'], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            try:
                line = proc.stdout.readline()
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except Exception:
                    pass
            if not line:
                return None
            data = {'gpu_name': 'NVIDIA Jetson AGX Orin', 'gpu_utilization_percent': None, 'gpu_power_watts': None,
                    'gpu_power_limit_watts': None, 'gpu_temperature_c': None,
                    'gpu_memory_used_mb': None, 'gpu_memory_total_mb': None, 'gpu_source': 'tegrastats'}
            try:
                if os.path.isfile('/proc/device-tree/model'):
                    with open('/proc/device-tree/model', 'r') as f:
                        model = f.read().strip('\x00\n ')
                        if model:
                            data['gpu_name'] = model
            except Exception:
                pass
            gr3d = re.search(r'GR3D_FREQ\s+(\d+)%', line)
            if gr3d:
                data['gpu_utilization_percent'] = float(gr3d.group(1))
            gpu_temp = re.search(r'GPU@([0-9.]+)C', line)
            if gpu_temp:
                data['gpu_temperature_c'] = float(gpu_temp.group(1))
            ram = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
            if ram:
                data['gpu_memory_used_mb'] = float(ram.group(1))
                data['gpu_memory_total_mb'] = float(ram.group(2))
            power = re.search(r'(?:VDD_IN|VIN_SYS_5V0|POM_5V_IN)\s+(\d+)mW', line)
            if power:
                data['gpu_power_watts'] = float(power.group(1)) / 1000.0
            return data
        except Exception:
            return None

    def sample(self):
        with self.lock:
            if time.monotonic() - self.last_sample < 2:
                return dict(self.cached)
            data = {'gpu_name': None, 'gpu_utilization_percent': None, 'gpu_power_watts': None,
                    'gpu_power_limit_watts': None, 'gpu_temperature_c': None,
                    'gpu_memory_used_mb': None, 'gpu_memory_total_mb': None, 'gpu_source': 'unavailable'}
            executable = shutil.which('nvidia-smi')
            if executable:
                try:
                    result = subprocess.run([executable, '--query-gpu=name,utilization.gpu,power.draw,power.limit,temperature.gpu,memory.used,memory.total',
                                             '--format=csv,noheader,nounits', '--id=0'], capture_output=True, text=True, timeout=2,
                                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    if result.returncode == 0:
                        row = next(csv.reader(io.StringIO(result.stdout)))
                        data['gpu_name'] = row[0].strip()
                        fields = ['gpu_utilization_percent', 'gpu_power_watts', 'gpu_power_limit_watts',
                                  'gpu_temperature_c', 'gpu_memory_used_mb', 'gpu_memory_total_mb']
                        for key, value in zip(fields, row[1:]):
                            try:
                                data[key] = float(value.strip())
                            except ValueError:
                                pass
                        data['gpu_source'] = 'nvidia-smi'
                except (OSError, subprocess.TimeoutExpired, StopIteration):
                    pass
            if data['gpu_source'] == 'unavailable':
                tegra_data = self._sample_tegrastats()
                if tegra_data:
                    data = tegra_data
            self.last_sample, self.cached = time.monotonic(), data
            return dict(data)

