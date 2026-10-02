from __future__ import annotations
from app.camera.camera_worker import CameraWorker
from app.camera.rtsp_stream import OpenCVStream
from app.config.settings import resolve_source


class CameraManager:
    def __init__(self, cameras, settings, stream_factory=OpenCVStream, source_resolver=resolve_source):
        self.workers = {c.id: CameraWorker(c, source_resolver(c), settings, stream_factory) for c in cameras}

    def start(self):
        for worker in self.workers.values():
            worker.start()

    def stop(self):
        for worker in self.workers.values():
            worker.stop()
        return all([worker.join() for worker in self.workers.values()])

    def status(self):
        return [{**worker.config.model_dump(mode='json'), **worker.status()} for worker in self.workers.values()]
