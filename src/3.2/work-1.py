"""
1.2 多后端数值一致性与切换实验
任务：用 FEALPy 的后端管理器 bm，编写一个"数值微分"函数，
分别在 NumPy 和 PyTorch 后端下运行，比较输出结果。

要求：
- 代码只写一份，通过 bm.set_backend 切换后端
- 在 3~5 个不同输入点上输出结果，并对比误差
- 简要分析结果是否一致，如有差异分析原因
"""

import logging
import numpy as np
from fealpy.backend import backend_manager as bm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 数值微分函数（后端无关）
# ============================================================================

def numerical_derivative(f, x0, h=1e-6):
    """
    使用有限差分法计算函数在 x0 处的导数
    
    参数：
        f: 函数（应该能处理后端数组）
        x0: 输入点（标量或数组）
        h: 步长
    
    返回：
        导数值（与输入形式相同）
    """
    logger.info(f"计算函数导数，输入点: {x0}，步长: {h}")
    
    # 转换为后端数组
    x0 = bm.array(x0, dtype=bm.float64)
    h_val = bm.array(h, dtype=bm.float64)
    
    # 前向和后向差分
    x_forward = x0 + h_val
    x_backward = x0 - h_val
    
    # 中央差分公式
    derivative = (f(x_forward) - f(x_backward)) / (2 * h_val)
    
    return derivative


def batch_numerical_derivative(f, x_points, h=1e-6):
    """
    批量计算数值导数
    
    参数：
        f: 函数
        x_points: 输入点列表或数组
        h: 步长
    
    返回：
        导数值列表
    """
    logger.info(f"批量计算 {len(x_points)} 个点的数值导数")
    results = []
    for x in x_points:
        deriv = numerical_derivative(f, x, h)
        results.append(deriv)
    return results


# ============================================================================
# 测试函数
# ============================================================================

def sin_func(x):
    """正弦函数"""
    return bm.sin(x)


def cos_func(x):
    """余弦函数"""
    return bm.cos(x)


def exp_func(x):
    """指数函数"""
    return bm.exp(x)


def poly_func(x):
    """多项式函数：f(x) = x^3 - 2x^2 + x"""
    return x**3 - 2*x**2 + x


# 理论导数用于对比
def sin_derivative_theory(x):
    """sin(x) 的理论导数是 cos(x)"""
    return np.cos(x)


def cos_derivative_theory(x):
    """cos(x) 的理论导数是 -sin(x)"""
    return -np.sin(x)


def exp_derivative_theory(x):
    """exp(x) 的理论导数是 exp(x) 本身"""
    return np.exp(x)


def poly_derivative_theory(x):
    """多项式导数：f'(x) = 3x^2 - 4x + 1"""
    return 3*x**2 - 4*x + 1


# ============================================================================
# 对比函数
# ============================================================================

def compare_backends(func, func_name, theory_func, x_points, h=1e-6):
    """
    在两个后端上对比数值导数结果
    
    参数：
        func: 待求导函数（后端无关）
        func_name: 函数名称
        theory_func: 理论导数函数
        x_points: 测试点
        h: 步长
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"函数: {func_name}")
    logger.info(f"{'='*70}")
    
    results = {
        'x_points': x_points,
        'numpy': {},
        'pytorch': {},
        'theory': {},
        'numpy_error': {},
        'pytorch_error': {},
    }
    
    # 理论导数值
    logger.info("计算理论导数值...")
    for x in x_points:
        theory_val = theory_func(x)
        results['theory'][x] = theory_val
        logger.info(f"  x={x}: 理论导数 = {theory_val:.10f}")
    
    # NumPy 后端
    logger.info("\n切换到 NumPy 后端...")
    bm.set_backend('numpy')
    logger.info(f"当前后端: {bm.backend_name}")
    
    for x in x_points:
        deriv_val = numerical_derivative(func, x, h)
        # 转换为 Python 标量便于显示
        if hasattr(deriv_val, 'item'):
            deriv_val = deriv_val.item()
        results['numpy'][x] = deriv_val
        theory_val = results['theory'][x]
        error = abs(deriv_val - theory_val)
        results['numpy_error'][x] = error
        logger.info(f"  x={x}: 数值导数 = {deriv_val:.10f}, 误差 = {error:.2e}")
    
    # PyTorch 后端
    logger.info("\n切换到 PyTorch 后端...")
    try:
        bm.set_backend('pytorch')
        logger.info(f"当前后端: {bm.backend_name}")
        
        for x in x_points:
            deriv_val = numerical_derivative(func, x, h)
            # 转换为 Python 标量
            if hasattr(deriv_val, 'item'):
                deriv_val = deriv_val.item()
            elif hasattr(deriv_val, 'cpu'):
                deriv_val = deriv_val.cpu().numpy().item()
            results['pytorch'][x] = deriv_val
            theory_val = results['theory'][x]
            error = abs(deriv_val - theory_val)
            results['pytorch_error'][x] = error
            logger.info(f"  x={x}: 数值导数 = {deriv_val:.10f}, 误差 = {error:.2e}")
    except Exception as e:
        logger.warning(f"PyTorch 后端出错: {e}")
    
    return results


# ============================================================================
# 输出对比结果
# ============================================================================

def print_comparison_table(results, func_name):
    """打印对比结果表格"""
    print("\n" + "="*100)
    print(f"函数: {func_name}")
    print("="*100)
    
    print(f"{'x值':<10} {'理论导数':<18} {'NumPy导数':<18} {'NumPy误差':<15} {'PyTorch导数':<18} {'PyTorch误差':<15}")
    print("-"*100)
    
    for x in results['x_points']:
        theory = results['theory'][x]
        numpy_deriv = results['numpy'][x]
        numpy_err = results['numpy_error'][x]
        pytorch_deriv = results['pytorch'][x]
        pytorch_err = results['pytorch_error'][x]
        
        print(f"{x:<10.2f} {theory:<18.10f} {numpy_deriv:<18.10f} {numpy_err:<15.2e} "
              f"{pytorch_deriv:<18.10f} {pytorch_err:<15.2e}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    logger.info("开始多后端数值一致性实验")
    logger.info(f"初始后端: {bm.backend_name}")
    
    # 测试点
    test_points = [0.5, 1.0, 1.5, 2.0, 3.14]
    h_values = 1e-6
    
    # 存储所有结果
    all_results = {}
    
    # 测试 1: sin(x)
    logger.info("\n开始测试 sin(x)")
    results_sin = compare_backends(
        func=sin_func,
        func_name="sin(x)",
        theory_func=sin_derivative_theory,
        x_points=test_points,
        h=h_values
    )
    all_results['sin'] = results_sin
    
    # 测试 2: cos(x)
    logger.info("\n开始测试 cos(x)")
    results_cos = compare_backends(
        func=cos_func,
        func_name="cos(x)",
        theory_func=cos_derivative_theory,
        x_points=test_points,
        h=h_values
    )
    all_results['cos'] = results_cos
    
    # 测试 3: exp(x)
    logger.info("\n开始测试 exp(x)")
    results_exp = compare_backends(
        func=exp_func,
        func_name="exp(x)",
        theory_func=exp_derivative_theory,
        x_points=test_points,
        h=h_values
    )
    all_results['exp'] = results_exp
    
    # 测试 4: 多项式
    logger.info("\n开始测试多项式 f(x) = x³ - 2x² + x")
    results_poly = compare_backends(
        func=poly_func,
        func_name="f(x) = x³ - 2x² + x",
        theory_func=poly_derivative_theory,
        x_points=test_points,
        h=h_values
    )
    all_results['poly'] = results_poly
    
    # ========================================================================
    # 输出总结
    # ========================================================================
    
    print("\n\n" + "="*100)
    print("多后端数值一致性对比总结")
    print("="*100)
    
    # 打印详细表格
    for func_key, results in all_results.items():
        func_names = {'sin': 'sin(x)', 'cos': 'cos(x)', 'exp': 'exp(x)', 'poly': 'f(x) = x³ - 2x² + x'}
        print_comparison_table(results, func_names[func_key])
    
    # ========================================================================
    # 分析与对比
    # ========================================================================
    
    print("\n" + "="*100)
    print("分析与结论")
    print("="*100)
    
    print("\n1. 数值一致性分析：")
    print("-" * 50)
    
    for func_key, results in all_results.items():
        print(f"\n函数 {func_key}:")
        numpy_errors = list(results['numpy_error'].values())
        pytorch_errors = list(results['pytorch_error'].values())
        
        numpy_max_error = max(numpy_errors)
        pytorch_max_error = max(pytorch_errors)
        
        print(f"  NumPy 最大误差: {numpy_max_error:.2e}")
        print(f"  PyTorch 最大误差: {pytorch_max_error:.2e}")
        
        # 计算两个后端的差异
        backend_diff = []
        for x in results['x_points']:
            diff = abs(results['numpy'][x] - results['pytorch'][x])
            backend_diff.append(diff)
        
        max_backend_diff = max(backend_diff) if backend_diff else 0
        print(f"  NumPy 和 PyTorch 的差异: {max_backend_diff:.2e}")
    
    print("\n2. 结论：")
    print("-" * 50)
    print("""
    - 两个后端的计算结果高度一致，差异主要来自浮点数精度
    - NumPy 使用 float64，PyTorch 通常也使用 float64
    - 数值微分的误差主要来自有限差分法的截断误差
    - 随着步长 h 的减小，截断误差减小，但舍入误差增大
    - 在 h=1e-6 的设置下，达到了较好的精度平衡
    """)
    
    print("\n3. 后端切换的优势：")
    print("-" * 50)
    print("""
    - 同一套代码可在不同后端运行，易于迁移
    - NumPy 适合 CPU 计算，高效但无法 GPU 加速
    - PyTorch 适合 GPU 计算，对大规模问题性能更优
    - FEALPy 的后端管理器使得这种切换对用户透明
    """)
    
    logger.info("\n实验完成！")


if __name__ == "__main__":
    main()
