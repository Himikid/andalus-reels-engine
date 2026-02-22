from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Callable

from app.core.config import settings


class SingleWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max(1, settings.max_workers))
        self._jobs: dict[str, Future] = {}
        self._guard = Lock()

    def submit(self, job_id: str, fn: Callable[[], None]) -> None:
        with self._guard:
            if job_id in self._jobs and not self._jobs[job_id].done():
                return
            self._jobs[job_id] = self._executor.submit(fn)

    def is_running(self, job_id: str) -> bool:
        with self._guard:
            fut = self._jobs.get(job_id)
            return bool(fut and not fut.done())


worker = SingleWorker()
