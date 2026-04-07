import asyncio
import numpy as np

async def async_sphere(x):
    await asyncio.sleep(0.01)
    return np.sum(x**2)