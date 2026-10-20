from __future__ import annotations
from contextlib import asynccontextmanager
import asyncio
import functools
import logging
from pathlib import Path

if not hasattr(asyncio, 'to_thread'):
    async def _to_thread(func, /, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    asyncio.to_thread = _to_thread
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import SQLAlchemyError

from app.api import alerts, cameras, detections, events, health, videos
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository
from app.logging_setup import configure_logging
from app.runtime import GateRuntime
from app.snapshots.snapshot_manager import SnapshotManager

log = logging.getLogger('gate')


def create_app(settings=None, database=None, camera_configs=None, runtime_factory=GateRuntime, camera_sources=None,
               frontend_directory=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        configure_logging(settings.log_directory)
        db = database or Database(settings.database_url.get_secret_value())
        configs = camera_configs if camera_configs is not None else settings.cameras()
        sources = camera_sources if camera_sources is not None else settings.camera_sources(configs)
        repository = GateRepository(db)
        snapshots = SnapshotManager(settings.snapshot_directory, categorized=settings.categorized_snapshot_folders)
        # Expose the evidence root for legacy filesystem-only snapshot
        # recovery in the repository/API layer.
        db.snapshot_directory = snapshots.root
        runtime = runtime_factory(settings, configs, repository, snapshots, camera_sources=sources)
        app.state.database, app.state.repository = db, repository
        app.state.snapshots, app.state.runtime = snapshots, runtime
        app.state.startup_error = None
        try:
            if db.is_sqlite and settings.deployment_mode != 'test':
                await asyncio.to_thread(db.initialize)
            await asyncio.to_thread(db.check)
            await asyncio.to_thread(repository.sync_cameras, configs)
            log.info('application_started')
        except Exception as exc:
            app.state.startup_error = type(exc).__name__
            log.error('startup_failed', extra={'error_type': type(exc).__name__})
        # Input previews and manual replay remain available during a DB outage.
        if app.state.startup_error:
            runtime.settings.pipeline_enabled = False
        runtime.start()
        try:
            yield
        finally:
            stopped = await asyncio.to_thread(runtime.stop)
            if stopped:
                db.close()
            log.info('application_stopped')

    app = FastAPI(title='CFS Gate Module', version='0.1.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type', 'X-Filename'])

    @app.middleware('http')
    async def check_browser_origin(request: Request, call_next):
        origin = request.headers.get('origin')
        local_origin = str(request.base_url).rstrip('/')
        if request.method == 'POST' and origin and origin not in settings.cors_origins + [local_origin]:
            return JSONResponse({'detail': 'Request origin is not allowed'}, status_code=403)
        return await call_next(request)

    for router in (cameras.router, events.router, detections.router, health.router, videos.router, alerts.router):
        app.include_router(router)

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc):
        log.error('database_request_failed', extra={'error_type': type(exc).__name__})
        return JSONResponse({'detail': 'Database unavailable'}, status_code=503)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc):
        return JSONResponse({'detail': 'Invalid request parameters'}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc):
        log.error('api_request_failed', extra={'error_type': type(exc).__name__})
        return JSONResponse({'detail': 'Internal service error'}, status_code=500)

    # In deployment FastAPI serves the compiled Angular application. API and
    # documentation routes are registered first so the catch-all mount cannot
    # shadow them. Development may still use `npm start` with its API proxy.
    frontend = Path(frontend_directory) if frontend_directory is not None else Path(__file__).resolve().parents[2] / 'frontend' / 'dist' / 'gate' / 'browser'
    if (frontend / 'index.html').is_file():
        app.mount('/', StaticFiles(directory=frontend, html=True), name='frontend')

    return app


app = create_app()
