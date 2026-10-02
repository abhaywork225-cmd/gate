from __future__ import annotations
"""Non-destructive schema v1 initialization: python -m app.database.initialize."""
from app.config.settings import Settings
from app.database.database import Database
from app.database.repositories.gate import GateRepository


def main():
    settings = Settings()
    db = Database(settings.database_url.get_secret_value())
    try:
        db.initialize()
        GateRepository(db).sync_cameras(settings.cameras())
        print('Gate schema v1 initialized. No legacy tables were modified.')
    except Exception as exc:
        # Driver exceptions may contain passwords or SQL parameters.
        raise SystemExit(f'Schema initialization failed ({type(exc).__name__}); verify database access/configuration.') from None
    finally:
        db.close()


if __name__ == '__main__':
    main()
