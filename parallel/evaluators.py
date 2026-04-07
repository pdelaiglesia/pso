import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from joblib import Parallel, delayed

class SequentialEvaluator:
    def evaluate(self, func, positions):
        return [func(pos) for pos in positions]

class ThreadEvaluator:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def evaluate(self, func, positions):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(func, positions))

class ProcessEvaluator:
    def __init__(self, max_workers=4, chunksize=1):
        self.max_workers = max_workers
        self.chunksize = chunksize

    def evaluate(self, func, positions):
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(func, positions, chunksize=self.chunksize))

class AsyncEvaluator:
    def evaluate(self, func, positions):
        loop = asyncio.get_event_loop()
        tasks = [func(pos) for pos in positions]
        return loop.run_until_complete(asyncio.gather(*tasks))

class VectorizedEvaluator:
    def evaluate(self, func, positions):
        return func(np.array(positions))
class JoblibEvaluator:
    def __init__(self, n_jobs=4):
        self.n_jobs = n_jobs

    def evaluate(self, func, positions):
        return Parallel(n_jobs=self.n_jobs)(delayed(func)(pos) for pos in positions)