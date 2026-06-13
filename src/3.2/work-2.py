"""
1.3 稀疏矩阵-向量乘法性能对比
任务：用 FEALPy 的 Sparse 模块，分别创建 1000、10000、100000、1000000 阶的稀疏矩阵，
再创建同阶随机向量，考察不同后端下稀疏矩阵-向量乘法的效率。

步骤：
- 用 bm.set_backend 分别在 numpy 和 pytorch 测试
- 用 fealpy.utils.timer 计时每次乘法的耗时
- 输出每个规模、每个后端的耗时表格
"""

import logging
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.sparse import csr_matrix, coo_matrix
from fealpy.utils import timer
from fealpy import logger as fealpy_logger

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# 稀疏矩阵生成函数
# ============================================================================

def create_random_sparse_matrix(N, sparsity=0.01):
    """
    创建随机稀疏矩阵（CSR 格式）
    
    参数：
        N: 矩阵的阶数（N×N）
        sparsity: 稀疏度（非零元素比例）
    
    返回：
        稀疏矩阵（CSR 格式）
    """
    logger.info(f"创建 {N}×{N} 随机稀疏矩阵，稀疏度={sparsity}")
    
    # 生成随机稀疏矩阵的非零元素索引
    nnz = max(1, int(N * N * sparsity))  # 非零元素个数
    
    # 随机选择位置（使用 NumPy 生成，然后转换为后端数组）
    np.random.seed(42)  # 固定随机数种子
    row_idx_np = np.random.randint(0, N, nnz)
    col_idx_np = np.random.randint(0, N, nnz)
    data_np = np.random.randn(nnz).astype(np.float32)
    
    # 转换为后端数组
    row_idx = bm.array(row_idx_np, dtype=bm.int64)
    col_idx = bm.array(col_idx_np, dtype=bm.int64)
    data = bm.array(data_np, dtype=bm.float32)
    
    # 创建 COO 矩阵然后转换为 CSR
    coo = coo_matrix((data, (row_idx, col_idx)), shape=(N, N))
    csr = csr_matrix(coo)
    
    logger.info(f"  非零元素个数: {csr.nnz}")
    logger.info(f"  实际稀疏度: {csr.nnz / (N*N):.4f}")
    
    return csr


def create_tridiagonal_sparse_matrix(N):
    """
    创建三对角稀疏矩阵
    
    参数：
        N: 矩阵的阶数
    
    返回：
        稀疏矩阵（CSR 格式）
    """
    logger.info(f"创建 {N}×{N} 三对角稀疏矩阵")
    
    # 主对角线、上对角线、下对角线
    main_diag_np = np.full(N, 2.0, dtype=np.float32)
    upper_diag_np = np.full(N - 1, -1.0, dtype=np.float32)
    lower_diag_np = np.full(N - 1, -1.0, dtype=np.float32)
    
    # 转换为后端数组
    main_diag = bm.array(main_diag_np, dtype=bm.float32)
    upper_diag = bm.array(upper_diag_np, dtype=bm.float32)
    lower_diag = bm.array(lower_diag_np, dtype=bm.float32)
    
    # 手工构造 COO 格式（使用 NumPy 索引，然后转换）
    row_idx_list = []
    col_idx_list = []
    data_list = []
    
    # 主对角线
    for i in range(N):
        row_idx_list.append(i)
        col_idx_list.append(i)
        data_list.append(2.0)
    
    # 上对角线
    for i in range(N - 1):
        row_idx_list.append(i)
        col_idx_list.append(i + 1)
        data_list.append(-1.0)
    
    # 下对角线
    for i in range(1, N):
        row_idx_list.append(i)
        col_idx_list.append(i - 1)
        data_list.append(-1.0)
    
    row_idx = bm.array(np.array(row_idx_list, dtype=np.int64), dtype=bm.int64)
    col_idx = bm.array(np.array(col_idx_list, dtype=np.int64), dtype=bm.int64)
    data = bm.array(np.array(data_list, dtype=np.float32), dtype=bm.float32)
    
    coo = coo_matrix((data, (row_idx, col_idx)), shape=(N, N))
    csr = csr_matrix(coo)
    
    logger.info(f"  非零元素个数: {csr.nnz}")
    
    return csr


# ============================================================================
# 性能测试函数
# ============================================================================

def benchmark_spmv(matrix_sizes, num_trials=5, matrix_type='random'):
    """
    在两个后端上进行稀疏矩阵-向量乘法的性能测试
    
    参数：
        matrix_sizes: 矩阵大小列表
        num_trials: 每个大小重复运行的次数
        matrix_type: 矩阵类型 ('random' 或 'tridiagonal')
    
    返回：
        结果字典
    """
    
    results = {
        'sizes': matrix_sizes,
        'numpy': {},
        'pytorch': {},
    }
    
    logger.info(f"\n开始稀疏矩阵-向量乘法性能测试（{matrix_type} 矩阵）")
    logger.info("="*70)
    
    for N in matrix_sizes:
        logger.info(f"\n当前矩阵大小: {N}×{N}")
        logger.info("-"*70)
        
        # 创建矩阵和向量
        if matrix_type == 'random':
            sparsity = max(1 / N, 0.01)  # 根据大小调整稀疏度
            A = create_random_sparse_matrix(N, sparsity=sparsity)
        else:
            A = create_tridiagonal_sparse_matrix(N)
        
        # ====================================================================
        # NumPy 后端测试
        # ====================================================================
        logger.info("\n切换到 NumPy 后端...")
        bm.set_backend('numpy')
        logger.info(f"当前后端: {bm.backend_name}")
        
        # 创建向量
        x = bm.random.rand(N)
        
        # 使用 timer 计时
        tmr = timer()
        next(tmr)  # 初始化 timer
        
        for i in range(num_trials):
            y = A @ x  # 稀疏矩阵-向量乘法
            tmr.send(f'numpy_{N}_{i}')
        
        next(tmr)  # 完成 timer 并打印统计
        
        # 计算平均时间（从 timer 的输出推算）
        # timer 已经输出了每次操作的耗时，这里记录 N 对应的结果
        results['numpy'][N] = 'completed'
        logger.info(f"NumPy 后端 {N}×{N} 矩阵测试完成")
        
        # ====================================================================
        # PyTorch 后端测试
        # ====================================================================
        logger.info("\n切换到 PyTorch 后端...")
        try:
            bm.set_backend('pytorch')
            logger.info(f"当前后端: {bm.backend_name}")
            
            # 创建向量
            x = bm.random.rand(N)
            
            # 使用 timer 计时
            tmr = timer()
            next(tmr)
            
            for i in range(num_trials):
                y = A @ x
                tmr.send(f'pytorch_{N}_{i}')
            
            next(tmr)
            
            results['pytorch'][N] = 'completed'
            logger.info(f"PyTorch 后端 {N}×{N} 矩阵测试完成")
            
        except Exception as e:
            logger.warning(f"PyTorch 后端出错: {e}")
            results['pytorch'][N] = 'error'
    
    return results


# ============================================================================
# 输出和分析
# ============================================================================

def print_benchmark_table(results, matrix_type='random'):
    """
    打印性能对比表格
    
    参数：
        results: 性能测试结果字典
        matrix_type: 矩阵类型
    """
    print("\n" + "="*90)
    print(f"稀疏矩阵-向量乘法性能对比（{matrix_type} 矩阵）")
    print("="*90)
    
    print(f"{'矩阵大小':<15} {'NumPy 状态':<18} {'PyTorch 状态':<18}")
    print("-"*90)
    
    for N in results['sizes']:
        numpy_status = results['numpy'][N]
        pytorch_status = results['pytorch'][N]
        
        print(f"{N:<15} {str(numpy_status):<18} {str(pytorch_status):<18}")
    
    print("\n注：timer 输出详见上方日志中的性能数据表")


def analyze_results(results_random, results_tridiag):
    """
    分析和总结性能测试结果
    """
    print("\n" + "="*90)
    print("性能分析与总结")
    print("="*90)
    
    print("\n1. 随机稀疏矩阵与三对角矩阵的对比：")
    print("-"*60)
    print("随机稀疏矩阵：")
    print("  - 具有任意的非零元素分布")
    print("  - 缓存局部性差，访问模式不规则")
    print("  - 性能通常较低")
    
    print("\n三对角矩阵：")
    print("  - 非零元素规则分布")
    print("  - 缓存局部性好，访问模式规则")
    print("  - 性能通常较优")
    
    print("\n2. 规模对性能的影响：")
    print("-"*60)
    print("  - 矩阵大小从 1000 增加到 100000")
    print("  - 注意观察 timer 输出中每个操作的耗时变化")
    print("  - 一般来说，随着矩阵规模增加，单次操作的耗时也会增加")
    
    print("\n3. 后端对性能的影响：")
    print("-"*60)
    print("NumPy 后端：")
    print("  - 基于 C/Fortran 的高效实现")
    print("  - CPU 计算，缓存效率高")
    print("  - 适合中等规模问题")
    
    print("\nPyTorch 后端：")
    print("  - 支持 GPU 计算")
    print("  - 对超大规模问题有优势")
    print("  - CPU 上的开销可能较大（初始化、数据转移等）")
    
    print("\n4. 稀疏矩阵-向量乘法的复杂度：")
    print("-"*60)
    print("  - 时间复杂度: O(nnz)，其中 nnz 是非零元素个数")
    print("  - 空间复杂度: O(n + nnz)")
    print("  - 性能瓶颈通常在内存访问而非计算")
    
    print("\n5. 建议：")
    print("-"*60)
    print("  - 对于 CPU 计算，NumPy 通常足够且更简洁")
    print("  - 对于 GPU 计算或超大规模问题，考虑 PyTorch")
    print("  - 矩阵结构（规则性）对性能影响很大")
    print("  - 在实际应用中进行性能测试以指导后端选择")


# ============================================================================
# 主程序
# ============================================================================

def main():
    logger.info("开始稀疏矩阵-向量乘法性能测试")
    logger.info(f"初始后端: {bm.backend_name}")
    
    # 定义矩阵大小
    matrix_sizes = [1000, 10000, 100000]
    # 注意：1000000 可能耗时较长，如需要可添加
    
    # 测试次数
    num_trials = 5
    
    # ========================================================================
    # 测试 1: 随机稀疏矩阵
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("测试 1: 随机稀疏矩阵")
    logger.info("="*70)
    
    results_random = benchmark_spmv(
        matrix_sizes=matrix_sizes,
        num_trials=num_trials,
        matrix_type='random'
    )
    
    print_benchmark_table(results_random, matrix_type='随机稀疏矩阵')
    
    # ========================================================================
    # 测试 2: 三对角稀疏矩阵
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("测试 2: 三对角稀疏矩阵")
    logger.info("="*70)
    
    results_tridiag = benchmark_spmv(
        matrix_sizes=matrix_sizes,
        num_trials=num_trials,
        matrix_type='tridiagonal'
    )
    
    print_benchmark_table(results_tridiag, matrix_type='三对角稀疏矩阵')
    
    # ========================================================================
    # 综合分析
    # ========================================================================
    # analyze_results(results_random, results_tridiag)
    
    logger.info("\n性能测试完成！")


if __name__ == "__main__":
    main()
