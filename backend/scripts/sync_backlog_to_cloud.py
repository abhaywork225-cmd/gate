#!/usr/bin/env python3
"""
Sync Backlog to Prosper Smart Yard Cloud
Safely flushes confirmed events with verified container numbers to Prosper Smart Yard API.
"""
import os
import sys
import time
from pathlib import Path
from sqlalchemy import select

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config.settings import Settings
from app.database.database import Database
from app.database.models import GateEvent
from app.database.repositories.gate import GateRepository
from app.snapshots.snapshot_manager import SnapshotManager
from app.events.cloud_dispatcher import CloudOutboxDispatcher


def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    env_file = os.path.join(base_dir, '.env')
    cameras_file = Path(os.path.join(base_dir, 'config', 'cameras.yaml'))
    settings = Settings(_env_file=env_file, cameras_file=cameras_file, snapshot_directory=Path(os.path.join(base_dir, 'snapshots')))
    db = Database(settings.database_url.get_secret_value())
    repo = GateRepository(db)
    snapshots = SnapshotManager(settings.snapshot_directory)
    dispatcher = CloudOutboxDispatcher(settings, repository=repo, snapshots=snapshots)

    with db.sessions() as session:
        events = session.execute(
            select(GateEvent)
            .where(
                GateEvent.status != 'UNREADABLE',
                GateEvent.container_number.isnot(None),
                GateEvent.container_number != '',
                (GateEvent.cloud_synced == False) | (GateEvent.cloud_synced.is_(None))
            )
            .order_by(GateEvent.timestamp.asc())
        ).scalars().all()
        target_ids = [e.id for e in events if dispatcher.is_dispatchable_event(repo.event(e.id))]

    print(f"Starting cloud backlog sync. Total eligible events (OCR accuracy >= 50%) to send: {len(target_ids)}")
    synced_total = 0
    failed_total = 0

    for i, eid in enumerate(target_ids, 1):
        ev = repo.event(eid)
        cntr = ev.get('container_number') if ev else None
        etype = ev.get('event_type') if ev else None
        print(f"[{i}/{len(target_ids)}] Syncing event {eid} ({etype} {cntr})...", end=' ', flush=True)

        try:
            res = dispatcher.dispatch_event(eid)
            if res.get('success'):
                synced_total += 1
                print(f"SUCCESS -> Cloud Event: {res.get('cloud_event_id')}")
            else:
                failed_total += 1
                print(f"FAILED -> Status: {res.get('status')}, Error: {res.get('error')}")
        except Exception as e:
            failed_total += 1
            print(f"EXCEPTION -> {e}")

        time.sleep(0.3)

    print("\n==========================================")
    print(f"Backlog Sync Finished!")
    print(f"Total Synced Successfully: {synced_total}")
    print(f"Total Failed/Deferred   : {failed_total}")
    print("==========================================")


if __name__ == '__main__':
    main()
