"""
BusSense Local Offline Queue Client
===================================
Maintains local disk persistence for edge events when mobile network connectivity
is interrupted (Section 16). Handles retry loops and server ACK confirmations.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import httpx


class BusSenseOfflineQueueClient:
    """
    Edge-side local persistent queue and upload dispatcher.
    Prevents loss of critical safety evidence during cellular dead zones.
    """

    def __init__(self, queue_dir: Optional[str] = None):
        if queue_dir is None:
            root_dir = Path(__file__).resolve().parent.parent
            self.queue_dir = root_dir / "outputs" / "edge_queue"
        else:
            self.queue_dir = Path(queue_dir)

        self.queue_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, event_data: Dict[str, Any], endpoint_path: str) -> str:
        """
        Persists an event to the local disk queue.
        Returns the absolute filepath of the queued record.
        """
        event_id = event_data.get("event_id", f"EVT_QUEUED_{int(time.time()*1000)}")
        item = {
            "event_id": event_id,
            "endpoint_path": endpoint_path,
            "enqueued_at": time.time(),
            "retry_count": 0,
            "data": event_data
        }
        file_path = self.queue_dir / f"{event_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)
        return str(file_path)

    def get_pending_events(self) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Returns all pending event records on disk: [(file_path, item_dict)]
        """
        pending = []
        if not self.queue_dir.exists():
            return pending

        for p in sorted(self.queue_dir.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    item = json.load(f)
                pending.append((str(p), item))
            except Exception:
                pass
        return pending

    def mark_uploaded(self, file_path: str):
        """
        Removes queued record upon confirmed HTTP 200/201 ACK from server.
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass

    def dispatch_event(
        self,
        event_data: Dict[str, Any],
        endpoint_path: str,
        client: httpx.Client,
        base_url: str = "http://localhost:8000"
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Attempts direct HTTP upload.
        If network fails or server returns 5xx: automatically enqueues to local disk.
        """
        url = f"{base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
        try:
            resp = client.post(url, json=event_data, timeout=5.0)
            if resp.status_code in [200, 201]:
                return True, resp.json()
            elif resp.status_code == 422:
                # Validation error: reject without infinite retry
                return False, {"error": "validation_error", "detail": resp.json()}
            else:
                # Server 5xx or transient error -> queue
                self.enqueue(event_data, endpoint_path)
                return False, {"error": "server_error", "status_code": resp.status_code}
        except (httpx.ConnectError, httpx.TimeoutException, Exception) as e:
            # Network outage -> queue offline
            self.enqueue(event_data, endpoint_path)
            return False, {"error": "network_down", "exception": str(e)}

    def flush_pending(self, client: httpx.Client, base_url: str = "http://localhost:8000") -> int:
        """
        Flushes and retries pending queued events.
        Returns count of successfully acknowledged uploads.
        """
        pending = self.get_pending_events()
        ack_count = 0
        for fpath, item in pending:
            endpoint = item["endpoint_path"]
            data = item["data"]
            url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"
            try:
                resp = client.post(url, json=data, timeout=5.0)
                if resp.status_code in [200, 201]:
                    self.mark_uploaded(fpath)
                    ack_count += 1
                elif resp.status_code == 422:
                    # Drop corrupted / invalid record
                    self.mark_uploaded(fpath)
            except Exception:
                # Still offline, retain in queue for next cycle
                break
        return ack_count
