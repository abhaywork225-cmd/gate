from __future__ import annotations
from sqlalchemy import create_engine, event, inspect, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy.engine import make_url
from pathlib import Path
from app.database.models import Base, SchemaVersion

SCHEMA_VERSION = 1


class Database:
    def __init__(self, url: str):
        parsed = make_url(url)
        self.is_sqlite = parsed.get_backend_name() == 'sqlite'
        options = {'pool_pre_ping': True}
        if self.is_sqlite:
            if parsed.database and parsed.database != ':memory:':
                # Resolve relative paths against backend, regardless of caller cwd.
                location = Path(parsed.database)
                if not location.is_absolute():
                    location = Path(__file__).resolve().parents[2] / location
                location = location.resolve()
                location.parent.mkdir(parents=True, exist_ok=True)
                url = parsed.set(database=str(location))
            options['connect_args'] = {'check_same_thread': False, 'timeout': 10}
            if not parsed.database or parsed.database == ':memory:':
                options['poolclass'] = StaticPool
        else:
            options['connect_args'] = {'connect_timeout': 3, 'read_timeout': 5, 'write_timeout': 5}
            options['pool_recycle'] = 1800
        self.engine = create_engine(url, **options)
        if self.is_sqlite:
            @event.listens_for(self.engine, 'connect')
            def enable_foreign_keys(connection, _):
                connection.execute('PRAGMA foreign_keys=ON')
                connection.execute('PRAGMA busy_timeout=10000')
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA synchronous=FULL')
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def _migrate_columns(self):
        with self.engine.begin() as conn:
            insp = inspect(conn)
            if 'gate_events' in insp.get_table_names():
                cols = {c['name'] for c in insp.get_columns('gate_events')}
                if 'visit_id' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN visit_id VARCHAR(36)')
                if 'capture_policy' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN capture_policy VARCHAR(20)')
                if 'lane_id' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN lane_id VARCHAR(32)')
                if 'is_twin' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN is_twin BOOLEAN DEFAULT 0')
                if 'container_number_2' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN container_number_2 VARCHAR(11)')
                if 'iso_code_1' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN iso_code_1 VARCHAR(4)')
                if 'iso_code_2' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN iso_code_2 VARCHAR(4)')
                if 'cloud_synced' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN cloud_synced BOOLEAN DEFAULT 0')
                if 'cloud_synced_at' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN cloud_synced_at DATETIME')
                if 'cloud_event_id' not in cols:
                    conn.exec_driver_sql('ALTER TABLE gate_events ADD COLUMN cloud_event_id VARCHAR(64)')

    def initialize(self):
        existing = set(inspect(self.engine).get_table_names())
        if SchemaVersion.__tablename__ in existing:
            self.check()
        elif existing.intersection(Base.metadata.tables):
            raise RuntimeError('Unversioned Gate tables exist; review schema before initializing')
        Base.metadata.create_all(self.engine)
        self._migrate_columns()
        with self.sessions.begin() as session:
            version = session.get(SchemaVersion, 1)
            if version is None:
                session.add(SchemaVersion(id=1, version=SCHEMA_VERSION))
            elif version.version != SCHEMA_VERSION:
                raise RuntimeError('Database schema version mismatch; explicit migration required')

    def check(self):
        with self.sessions() as session:
            version = session.scalar(select(SchemaVersion.version).where(SchemaVersion.id == 1))
            if version != SCHEMA_VERSION:
                raise RuntimeError('Initialize the database schema before starting processing')

    def close(self):
        self.engine.dispose()
