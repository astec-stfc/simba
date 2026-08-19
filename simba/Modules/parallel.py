"""
Helpers for running independent (post-)processing tasks concurrently.

The number of worker threads can be set globally with the ``SIMBA_MAX_WORKERS``
environment variable, or with the ``global_parameters["max_workers"]``
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, List, Mapping, Sequence
from warnings import warn

__all__ = ["get_max_workers", "thread_map", "DEFAULT_MAX_WORKERS", "ENV_MAX_WORKERS"]

DEFAULT_MAX_WORKERS = 32
"""Upper limit on the number of worker threads."""

ENV_MAX_WORKERS = "SIMBA_MAX_WORKERS"
"""Environment variable used to override the number of worker threads."""


def get_max_workers(max_workers: int | None = None, njobs: int | None = None) -> int:
    """
    Determine how many worker threads should be used.

    Parameters
    ----------
    max_workers: int or None
        Explicitly requested number of threads; if `None`, fall back to the
        environment variable or the default.
    njobs: int or None
        Number of jobs to be executed.

    Returns
    -------
    int
        The number of worker threads to use, always at least 1.
    """
    if max_workers is None:
        env = os.environ.get(ENV_MAX_WORKERS)
        if env is not None:
            try:
                max_workers = int(env)
            except ValueError:
                warn(f"Ignoring invalid {ENV_MAX_WORKERS}={env!r}")
    if max_workers is None:
        max_workers = min(DEFAULT_MAX_WORKERS, (os.cpu_count() or 1) + 4)
    max_workers = max(1, int(max_workers))
    if njobs is not None:
        max_workers = max(1, min(max_workers, njobs))
    return max_workers


def _split_job(job: Any) -> tuple[Sequence, Mapping]:
    """Convert a job specification into positional and keyword arguments."""
    if isinstance(job, Mapping):
        return (), job
    if isinstance(job, tuple):
        if len(job) == 2 and isinstance(job[1], Mapping) and isinstance(job[0], (tuple, list)):
            return job[0], job[1]
        return job, {}
    if isinstance(job, list):
        return job, {}
    return (job,), {}


def thread_map(
    func: Callable,
    jobs: Iterable,
    max_workers: int | None = None,
    label: str | None = None,
    raise_errors: bool = True,
) -> List:
    """
    Call `func` once for every entry in `jobs`, using a thread pool.

    Parameters
    ----------
    func: callable
        The function to be called for each job.
    jobs: iterable
        The arguments for each call, see above.
    max_workers: int or None
        Maximum number of worker threads, see
        :func:`~simba.Modules.parallel.get_max_workers`.
    label: str or None
        Name used in warnings, to indicate which step failed.
    raise_errors: bool
        If True, re-raise the first exception raised by any of the jobs once all of
        them have finished; if False, only emit a warning for each failure and
        return `None` in place of the corresponding result.

    Returns
    -------
    list
        The return value of `func` for each job, in the order given by `jobs`.
        Jobs that raised an exception (with `raise_errors` set to False) give `None`.
    """
    jobs = list(jobs)
    if len(jobs) == 0:
        return []
    name = label if label is not None else getattr(func, "__name__", str(func))
    nworkers = get_max_workers(max_workers, len(jobs))
    results: List = [None] * len(jobs)
    errors: List = []
    if nworkers == 1:
        for i, job in enumerate(jobs):
            args, kwargs = _split_job(job)
            try:
                results[i] = func(*args, **kwargs)
            except Exception as e:
                errors.append((i, e))
    else:
        with ThreadPoolExecutor(max_workers=nworkers) as executor:
            futures = {}
            for i, job in enumerate(jobs):
                args, kwargs = _split_job(job)
                futures[executor.submit(func, *args, **kwargs)] = i
            for future, i in futures.items():
                try:
                    results[i] = future.result()
                except Exception as e:
                    errors.append((i, e))
    if errors:
        if raise_errors:
            index, error = errors[0]
            raise RuntimeError(
                f"{name}: {len(errors)} of {len(jobs)} job(s) failed; "
                f"first failure was job {index}"
            ) from error
        for index, error in errors:
            warn(f"{name}: job {index} failed with {error!r}")
    return results
