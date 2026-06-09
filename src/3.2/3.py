from fealpy.utils import timer
from fealpy.backend import backend_manager as bm

N = 2000000  
repeats = 20

tmr = timer()
next(tmr)

# NumPy 后端（默认）
a = bm.random.rand(N)
b = bm.random.rand(N)
for _ in range(repeats):
    c = a * b
tmr.send('numpy')

bm.set_backend('pytorch')
tmr.send('pytorch(set)')

# PyTorch 后端(CPU)
a = bm.random.rand(N)
b = bm.random.rand(N)
for _ in range(repeats):
    c = a * b
tmr.send('pytorch(CPU)')

next(tmr)


from fealpy import logger

logger.debug("...")
logger.info("...")
logger.warning("something warning")
logger.error("something error")
