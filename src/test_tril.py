from fealpy.backend import backend_manager as bm
import numpy as np
import pytest

bm.set_backend('numpy')

def test_tril_basic():
    # 输入
    a = bm.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    # 计算结果
    a_tril = bm.tril(a)
    # 预期输出
    exp = bm.array([[1, 0, 0],
                    [4, 5, 0],
                    [7, 8, 9]])
    # 比较结果和预期输出
    assert bm.all(a_tril == exp)
    
    b = bm.array([[1, 2, 3],
                [3, 4, 5]])
    b_tril = bm.tril(b)
    exp_b = bm.array([[1, 0, 0],
                    [3, 4, 0]])
    assert bm.all(b_tril == exp_b)
    
def test_tril_k():
    a = bm.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
    a_tril_k1 = bm.tril(a, k=1)
    exp_k1 = bm.array([[1, 2, 0],
                    [4, 5, 6],
                    [7, 8, 9]])
    assert bm.all(a_tril_k1 == exp_k1)

    a_tril_k_1 = bm.tril(a, k=-1)
    exp_k_1 = bm.array([[0, 0, 0],
                        [4, 0, 0],
                        [7, 8, 0]])
    assert bm.all(a_tril_k_1 == exp_k_1)
    
def test_tril_empty():
    a = bm.array([[]])
    a_tril = bm.tril(a)
    exp = bm.array([[]])
    assert bm.all(a_tril == exp)
    
def test_tril_abnomal():
    # 测试异常情况
    a = bm.array([[1, 2, 3],
                [4, 5, 6]])
    with pytest.raises(TypeError):
        bm.tril(a, k=None)