import threading
import time
from types import SimpleNamespace

import pytest

from simba.Framework_objects import frameworkLattice
from simba.Modules.parallel import (
    DEFAULT_MAX_WORKERS,
    ENV_MAX_WORKERS,
    get_max_workers,
    thread_map,
)


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV_MAX_WORKERS, raising=False)


def test_get_max_workers_default(clean_env):
    assert 1 <= get_max_workers() <= DEFAULT_MAX_WORKERS


def test_get_max_workers_capped_by_njobs(clean_env):
    assert get_max_workers(8, 3) == 3
    assert get_max_workers(2, 10) == 2
    assert get_max_workers(0, 10) == 1


def test_get_max_workers_from_environment(monkeypatch):
    monkeypatch.setenv(ENV_MAX_WORKERS, "3")
    assert get_max_workers(njobs=10) == 3
    assert get_max_workers(5, 10) == 5


def test_get_max_workers_invalid_environment(monkeypatch):
    monkeypatch.setenv(ENV_MAX_WORKERS, "not-a-number")
    with pytest.warns(UserWarning):
        assert 1 <= get_max_workers(njobs=10) <= DEFAULT_MAX_WORKERS


def test_thread_map_argument_types(clean_env):
    def func(a, b=0):
        return a + b

    assert thread_map(func, []) == []
    assert thread_map(func, [{"a": 1, "b": 2}, (3, 4), [5, 6], 7]) == [3, 7, 11, 7]


def test_thread_map_preserves_order(clean_env):
    def func(i):
        # Reverse the order in which the jobs finish
        time.sleep(0.01 * (10 - i))
        return i

    assert thread_map(func, range(10), max_workers=10) == list(range(10))


def test_thread_map_runs_concurrently(clean_env):
    threads = set()

    def func(i):
        threads.add(threading.current_thread().name)
        time.sleep(0.1)
        return i

    start = time.time()
    thread_map(func, range(8), max_workers=4)
    assert len(threads) == 4
    assert time.time() - start < 0.8


def test_thread_map_serial(clean_env):
    def func(i):
        return threading.current_thread().name

    names = thread_map(func, range(4), max_workers=1)
    assert set(names) == {threading.current_thread().name}


def test_thread_map_raises_first_error(clean_env):
    def func(i):
        return 1 / i

    with pytest.raises(RuntimeError) as excinfo:
        thread_map(func, [1, 0, 2], label="divide")
    assert "divide" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ZeroDivisionError)


def test_thread_map_warns_instead_of_raising(clean_env):
    def func(i):
        return 1 / i

    with pytest.warns(UserWarning):
        results = thread_map(func, [1, 0, 2], raise_errors=False)
    assert results == [1.0, None, 0.5]


def test_lattice_parallel_map_uses_global_parameters(clean_env):
    lattice = SimpleNamespace(
        objectname="TEST", global_parameters={"max_workers": 1}
    )

    def func(i):
        return threading.current_thread().name

    names = frameworkLattice.parallel_map(lattice, func, range(4))
    assert set(names) == {threading.current_thread().name}

    lattice.global_parameters["max_workers"] = None
    assert frameworkLattice.parallel_map(lattice, lambda i: i * 2, range(4)) == [
        0,
        2,
        4,
        6,
    ]


def test_lattice_parallel_map_labels_errors(clean_env):
    lattice = SimpleNamespace(objectname="TEST", global_parameters={})

    def failing_function(i):
        raise ValueError(i)

    with pytest.raises(RuntimeError) as excinfo:
        frameworkLattice.parallel_map(lattice, failing_function, [1])
    assert "TEST: failing_function" in str(excinfo.value)
