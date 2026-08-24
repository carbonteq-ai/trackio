import threading
import time

import pymysql
import pytest

from trackio.doris_pool import (
    DorisConnectionPool,
    DorisConnectionPoolTimeout,
    DorisPoolConfig,
)
from trackio.storage import is_retryable_storage_error


class FakeConnection:
    def __init__(self, identity):
        self.identity = identity
        self.closed = False
        self.healthy = True

    def ping(self, reconnect=False):
        assert reconnect is False
        if not self.healthy:
            raise pymysql.err.OperationalError(2006, "gone")

    def close(self):
        self.closed = True


class ConnectionFactory:
    def __init__(self):
        self.connections = []
        self.lock = threading.Lock()

    def __call__(self, **settings):
        assert settings == {"host": "doris"}
        with self.lock:
            connection = FakeConnection(len(self.connections))
            self.connections.append(connection)
            return connection


def config(*, size=4, reserve=1, timeout=0.05, recycle=300):
    return DorisPoolConfig(
        max_size=size,
        control_reserve=reserve,
        checkout_timeout=timeout,
        recycle_seconds=recycle,
    )


def test_pool_configuration_rejects_an_unreserved_pool(monkeypatch):
    monkeypatch.setenv("TRACKIO_DORIS_POOL_SIZE", "4")
    monkeypatch.setenv("TRACKIO_DORIS_CONTROL_RESERVE", "4")

    with pytest.raises(RuntimeError, match="must be smaller"):
        DorisPoolConfig.from_env()


def test_returned_connection_is_reused():
    factory = ConnectionFactory()
    pool = DorisConnectionPool({"host": "doris"}, config(), connect=factory)

    with pool.connection() as first:
        pass
    with pool.connection() as second:
        pass

    assert first is second
    assert len(factory.connections) == 1
    assert pool.stats().idle == 1
    pool.close()


def test_control_reserve_remains_available_when_ordinary_lane_is_full():
    factory = ConnectionFactory()
    pool = DorisConnectionPool(
        {"host": "doris"}, config(size=3, reserve=1), connect=factory
    )
    ordinary = [pool.acquire(), pool.acquire()]

    with pytest.raises(DorisConnectionPoolTimeout, match="ordinary lane"):
        pool.acquire()

    control = pool.acquire(control=True)
    assert pool.stats().checked_out == 3
    assert pool.stats().high_watermark == 3

    pool.release(control)
    for connection in ordinary:
        pool.release(connection)
    pool.close()


def test_dead_idle_connection_is_discarded_and_replaced():
    factory = ConnectionFactory()
    pool = DorisConnectionPool({"host": "doris"}, config(), connect=factory)
    first = pool.acquire()
    pool.release(first)
    first.healthy = False

    second = pool.acquire()

    assert second is not first
    assert first.closed is True
    assert len(factory.connections) == 2
    pool.release(second)
    pool.close()


def test_connection_is_recycled_by_physical_age_not_idle_age():
    factory = ConnectionFactory()
    now = [10.0]
    pool = DorisConnectionPool(
        {"host": "doris"},
        config(recycle=5),
        connect=factory,
        clock=lambda: now[0],
    )
    first = pool.acquire()
    now[0] = 14.0
    pool.release(first)
    now[0] = 16.0

    second = pool.acquire()

    assert second is not first
    assert first.closed is True
    pool.release(second)
    pool.close()


def test_disconnect_during_operation_is_not_returned_to_pool():
    factory = ConnectionFactory()
    pool = DorisConnectionPool({"host": "doris"}, config(), connect=factory)

    with pytest.raises(pymysql.err.OperationalError):
        with pool.connection() as connection:
            raise pymysql.err.OperationalError(2013, "timed out")

    assert connection.closed is True
    assert pool.stats().total == 0
    pool.close()


def test_pool_timeout_is_a_retryable_storage_error():
    error = DorisConnectionPoolTimeout(
        "Doris connection capacity is temporarily unavailable"
    )

    assert is_retryable_storage_error(error)


def test_concurrent_control_callers_never_exceed_physical_maximum():
    factory = ConnectionFactory()
    pool = DorisConnectionPool(
        {"host": "doris"},
        config(size=4, reserve=1, timeout=1),
        connect=factory,
    )
    start = threading.Barrier(13)
    active = 0
    observed_max = 0
    lock = threading.Lock()
    errors = []

    def worker():
        nonlocal active, observed_max
        start.wait()
        try:
            with pool.connection(control=True):
                with lock:
                    active += 1
                    observed_max = max(observed_max, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert errors == []
    assert observed_max <= 4
    assert len(factory.connections) == 4
    assert pool.stats().high_watermark == 4
    pool.close()


def test_close_closes_idle_and_checked_out_connections_on_return():
    factory = ConnectionFactory()
    pool = DorisConnectionPool({"host": "doris"}, config(), connect=factory)
    checked_out = pool.acquire()
    idle = pool.acquire(control=True)
    pool.release(idle)

    pool.close()
    pool.release(checked_out)

    assert idle.closed is True
    assert checked_out.closed is True
    assert pool.stats().total == 0
