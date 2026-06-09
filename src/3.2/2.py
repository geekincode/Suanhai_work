from fealpy.backend import backend_manager as bm
from fealpy.sparse import coo_matrix, csr_matrix

def fd_derivative_bm(f, x0, h=1e-6):
    x1 = bm.array([x0 + h], dtype=bm.float64)
    x2 = bm.array([x0 - h], dtype=bm.float64)
    return ((f(x1) - f(x2)) / (2*h)).item()

# 使用 NumPy 后端
bm.set_backend('numpy')
print('backend numpy:', fd_derivative_bm(bm.sin, 1.23))

# 使用 PyTorch 后端
bm.set_backend('pytorch')
print('backend pytorch:', fd_derivative_bm(bm.sin, 1.23))


from fealpy.utils import timer

tmr = timer()

next(tmr) # 或： timer.send(None)
# ... some operations
tmr.send('tag1')
# ... some operations
tmr.send('tag2')

next(tmr)
# ... some operations
tmr.send('tag3')
# ... some operations
tmr.send('tag4')

next(tmr)