"""Bounded reusable connections for the Apache Doris storage provider."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import pymysql


class DorisConnectionPoolTimeout(TimeoutError):
    """Raised when bounded Doris admission cannot make progress in time."""


def _int_setting(name: str, default: int, *, minimum: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def _float_setting(name: str, default: float, *, minimum: float) -> float:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a number") from error
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True)
class DorisPoolConfig:
    """Validated process-local Doris connection limits."""

    max_size: int
    control_reserve: int
    checkout_timeout: float
    recycle_seconds: float

    @classmethod
    def from_env(cls) -> DorisPoolConfig:
        max_size = _int_setting("TRACKIO_DORIS_POOL_SIZE", 16, minimum=2)
        control_reserve = _int_setting(
            "TRACKIO_DORIS_CONTROL_RESERVE", 2, minimum=1
        )
        if control_reserve >= max_size:
            raise RuntimeError(
                "TRACKIO_DORIS_CONTROL_RESERVE must be smaller than "
                "TRACKIO_DORIS_POOL_SIZE"
            )
        return cls(
            max_size=max_size,
            control_reserve=control_reserve,
            checkout_timeout=_float_setting(
                "TRACKIO_DORIS_POOL_TIMEOUT", 10.0, minimum=0.001
            ),
            recycle_seconds=_float_setting(
                "TRACKIO_DORIS_POOL_RECYCLE_SECONDS", 300.0, minimum=0.0
            ),
        )


@dataclass
class _IdleConnection:
    connection: Any
    created_at: float


@dataclass(frozen=True)
class DorisPoolStats:
    max_size: int
    control_reserve: int
    total: int
    checked_out: int
    idle: int
    high_watermark: int


def _disconnect_error(error: BaseException) -> bool:
    if isinstance(error, pymysql.err.InterfaceError):
        return True
    if not isinstance(error, pymysql.err.OperationalError):
        return False
    code = error.args[0] if error.args else None
    return code in {2006, 2013, 2055}


class DorisConnectionPool:
    """Reuse a small connection set and reserve admission for control writes.

    Ordinary reads and bulk ingestion stop at ``max_size - control_reserve``.
    Control operations may use the entire pool, ensuring that artifact
    finalization can still make progress while general work is saturated.
    """

    def __init__(
        self,
        settings: dict[str, Any],
        config: DorisPoolConfig,
        *,
        connect: Callable[..., Any] = pymysql.connect,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._settings = settings.copy()
        self._config = config
        self._connect = connect
        self._clock = clock
        self._condition = threading.Condition()
        self._idle: deque[_IdleConnection] = deque()
        self._created_at: dict[int, float] = {}
        self._total = 0
        self._checked_out = 0
        self._high_watermark = 0
        self._closed = False

    def _admitted(self, *, control: bool) -> bool:
        limit = (
            self._config.max_size
            if control
            else self._config.max_size - self._config.control_reserve
        )
        return self._checked_out < limit

    @staticmethod
    def _close_connection(connection: Any) -> None:
        try:
            connection.close()
        except Exception:
            pass

    def _discard_checked_out(self, connection: Any) -> None:
        self._close_connection(connection)
        with self._condition:
            self._created_at.pop(id(connection), None)
            self._checked_out -= 1
            self._total -= 1
            self._condition.notify_all()

    @staticmethod
    def _healthy(connection: Any) -> bool:
        try:
            connection.ping(reconnect=False)
        except Exception:
            return False
        return True

    def acquire(self, *, control: bool = False) -> Any:
        deadline = self._clock() + self._config.checkout_timeout
        while True:
            create = False
            idle: _IdleConnection | None = None
            with self._condition:
                while True:
                    if self._closed:
                        raise RuntimeError("Doris connection pool is closed")
                    if self._admitted(control=control):
                        if self._idle:
                            idle = self._idle.popleft()
                            self._checked_out += 1
                            break
                        if self._total < self._config.max_size:
                            self._total += 1
                            self._checked_out += 1
                            create = True
                            break
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        lane = "control" if control else "ordinary"
                        raise DorisConnectionPoolTimeout(
                            "Doris connection capacity is temporarily unavailable "
                            f"for the {lane} lane"
                        )
                    self._condition.wait(remaining)
                self._high_watermark = max(
                    self._high_watermark, self._checked_out
                )

            if create:
                try:
                    connection = self._connect(**self._settings)
                except BaseException:
                    with self._condition:
                        self._checked_out -= 1
                        self._total -= 1
                        self._condition.notify_all()
                    raise
                with self._condition:
                    if self._closed:
                        self._checked_out -= 1
                        self._total -= 1
                        self._condition.notify_all()
                        self._close_connection(connection)
                        raise RuntimeError("Doris connection pool is closed")
                    self._created_at[id(connection)] = self._clock()
                return connection

            assert idle is not None
            expired = (
                self._config.recycle_seconds > 0
                and self._clock() - idle.created_at
                >= self._config.recycle_seconds
            )
            if expired or not self._healthy(idle.connection):
                self._discard_checked_out(idle.connection)
                continue
            return idle.connection

    def release(self, connection: Any, *, discard: bool = False) -> None:
        if discard:
            self._discard_checked_out(connection)
            return
        with self._condition:
            self._checked_out -= 1
            if self._closed:
                self._total -= 1
                self._created_at.pop(id(connection), None)
                close = True
            else:
                self._idle.append(
                    _IdleConnection(
                        connection,
                        self._created_at[id(connection)],
                    )
                )
                close = False
            self._condition.notify_all()
        if close:
            self._close_connection(connection)

    @contextmanager
    def connection(self, *, control: bool = False) -> Iterator[Any]:
        connection = self.acquire(control=control)
        discard = False
        try:
            yield connection
        except BaseException as error:
            discard = _disconnect_error(error)
            raise
        finally:
            self.release(connection, discard=discard)

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            idle = [entry.connection for entry in self._idle]
            self._idle.clear()
            for connection in idle:
                self._created_at.pop(id(connection), None)
            self._total -= len(idle)
            self._condition.notify_all()
        for connection in idle:
            self._close_connection(connection)

    def stats(self) -> DorisPoolStats:
        with self._condition:
            return DorisPoolStats(
                max_size=self._config.max_size,
                control_reserve=self._config.control_reserve,
                total=self._total,
                checked_out=self._checked_out,
                idle=len(self._idle),
                high_watermark=self._high_watermark,
            )
