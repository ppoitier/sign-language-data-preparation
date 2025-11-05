from joblib import Parallel, delayed
from typing import Callable, List, Dict, Any


def run_parallel(func: Callable, kwargs_list: List[Dict[str, Any]], n_jobs: int) -> List[Any]:
    """
    Launches a function in parallel with different sets of keyword arguments using processes (with joblib).

    Args:
        func: The function to execute in parallel.
        kwargs_list: A list of dictionaries, where each dictionary contains
                       the keyword arguments for a single call to `func`.
        n_jobs: The number of parallel processes to use.

    Returns:
        A list containing the return values from each function call,
        in the same order as the input `kwargs_list`.
    """
    print(f"Starting parallel execution of '{func.__name__}' with {len(kwargs_list)} tasks on {n_jobs} processes...")
    # Ensures process-based parallelism, which is truly parallel and avoids GIL issues.
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(func)(**kwargs) for kwargs in kwargs_list
    )
    print("Parallel execution finished.")
    return results