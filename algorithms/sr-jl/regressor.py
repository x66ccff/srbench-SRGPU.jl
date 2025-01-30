import subprocess
import numpy as np
import sympy as sp
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import tempfile
import os
import re
import sys
import threading



class OutputCapture:
    def __init__(self):
        self.best_model = None      # 存储最佳方程
        self.best_complexity = None # 存储最佳复杂度 
        self.best_loss = None       # 存储最佳损失值
        self.pf_entries = []        # 存储帕累托前沿
        self.capture_pf = False
        self.capture_best_next_line = False
        self.error_message = ''
        
def process_stdout(line, oc):
    stripped_line = line.strip()
    
    if oc.capture_best_next_line:
        # 解析格式: "Complexity: 5  Loss: 0.2  Equation: x1 + sin(x2)"
        parts = stripped_line.split('\t')
        for part in parts:
            if part.startswith('Complexity: '):
                oc.best_complexity = int(part[len('Complexity: '):].strip())
            elif part.startswith('Loss: '):
                oc.best_loss = float(part[len('Loss: '):].strip())
            elif part.startswith('Equation: '):
                oc.best_model = part[len('Equation: '):].strip()
        oc.capture_best_next_line = False
    
    elif stripped_line.startswith("Best Expression:"):
        oc.capture_best_next_line = True
    
    # 帕累托前沿捕获逻辑
    elif stripped_line == "=== PF START ===":
        oc.capture_pf = True
    elif stripped_line == "=== PF END ===":
        oc.capture_pf = False
    elif oc.capture_pf and stripped_line:
        parts = stripped_line.split()
        if len(parts) >= 3:
            oc.pf_entries.append({
                'complexity': parts[0],
                'loss': parts[1],
                'equation': ' '.join(parts[2:])
            })

def read_stream(stream, oc, stream_type):
    for line in iter(stream.readline, ''):
        if stream_type == 'stdout':
            sys.stdout.write(line)
            sys.stdout.flush()
            process_stdout(line, oc)
        else:
            sys.stderr.write(line)
            sys.stderr.flush()
            oc.error_message += line


class SymbolicRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        binary_operators=None,
        unary_operators=None,
        n_threads=4,
        population_size=100,
        generations=50
    ):
        self.binary_operators = binary_operators or ['+', '*', '/', '-']
        self.unary_operators = unary_operators or ['sin','cos', 'exp', 'log', 'sqrt']
        self.n_threads = n_threads
        self.population_size = population_size
        self.generations = generations
        self.n_features = None
        self.feature_names = []

    def fit(self, X, y):
        # 将X转换为二维数组并保存特征数
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # 转换为 (n_samples, 1) 形状
        self.n_features = X.shape[1]  # 保存特征数量
        self.feature_names = ['x{}'.format(i+1) for i in range(self.n_features)]

        # 保存数据到临时文件
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f_X:
            np.savetxt(f_X.name, X)
            self.X_path = f_X.name

        # 处理y，确保是一维数组
        y = np.asarray(y).ravel()
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f_y:
            np.savetxt(f_y.name, y)
            self.y_path = f_y.name

        # Julia脚本
        julia_script = f"""
using SymbolicRegression
using DelimitedFiles
using LoopVectorization

function choose_best(members, options)
    losses = [member.loss for member in members]
    scores = [member.score for member in members]
    complexities = [compute_complexity(member.tree, options) for member in members]
    
    threshold = 1.5 * minimum(losses)
    best_idx = argmax([
        (loss <= threshold) ? score : typemin(typeof(score)) 
        for (loss, score) in zip(losses, scores)
    ])
    return members[best_idx]
end

X = Float32.(transpose(readdlm("{self.X_path}")))
y = Float32.(vec(readdlm("{self.y_path}")))

options = SymbolicRegression.Options(;
    binary_operators={self.binary_operators},
    unary_operators={self.unary_operators},
    # population_size=100,
    # populations=15,
    batching=true,
    batch_size=100,
    adaptive_parsimony_scaling=1_000.0,
    parsimony=0.0,
    maxsize=30,
    maxdepth=20,
    turbo=true,
    should_optimize_constants=false,
    optimizer_iterations=4,
    optimizer_f_calls_limit=1000,
    optimizer_probability=0.02,
    early_stop_condition=(l, c) -> l < 1e-6 && c == 5,
    constraints = [
        sin => 9,
        cos => 9,
        exp => 9,
        log => 9,
        sqrt => 9
    ],
    nested_constraints = [
        sin => [
            sin => 0,
            cos => 0,
            exp => 1,
            log => 1,
            sqrt => 1
        ],
        exp => [
            exp => 0,
            log => 0
        ],
        log => [
            exp => 0,
            log => 0
        ],
        sqrt => [
            sqrt => 0
        ]
    ]
)

hall_of_fame = equation_search(X, y; 
    options=options, 
    parallelism=:multithreading
)

dominating = calculate_pareto_frontier(hall_of_fame)

# 打印帕累托前沿
println("=== PF START ===")
for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    equation = string_tree(member.tree, options)
    println("$(complexity)\t$(loss)\t$(equation)")
end
println("=== PF END ===")

# 选择最优成员
best_member = choose_best(dominating, options)

# 输出最优结果
complexity = compute_complexity(best_member.tree, options)
loss = best_member.loss
equation = string_tree(best_member.tree, options)
println("Best Expression:")
println("Complexity: $(complexity)\tLoss: $(loss)\tEquation: $(equation)")
println("")
        """.replace("'",'')
        

        # print('The script is: ')
        # print('='*40)
        # print(julia_script)
        # print('='*40)


        # 保存脚本到临时文件
        with tempfile.NamedTemporaryFile(suffix='.jl', delete=False) as f_script:
            f_script.write(julia_script.encode())
            script_path = f_script.name

        # 设置环境变量
        env = os.environ.copy()
        env["JULIA_NUM_THREADS"] = str(self.n_threads)

        oc = OutputCapture()

        try:
            print("Executing Julia script:", "julia", script_path)
            
            # 运行Julia脚本
            proc = subprocess.Popen(
                ["julia", script_path],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # 启动线程读取输出
            t_stdout = threading.Thread(target=read_stream, args=(proc.stdout, oc, 'stdout'))
            t_stderr = threading.Thread(target=read_stream, args=(proc.stderr, oc, 'stderr'))
            t_stdout.start()
            t_stderr.start()

            # 等待进程结束
            proc.wait()
            t_stdout.join()
            t_stderr.join()

            # 检查返回码
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(
                    proc.returncode, 
                    proc.args, 
                    output=oc.error_message, 
                    stderr=oc.error_message
                )

            # 解析结果
            self.results = oc.pf_entries

            
            self.best_model = oc.best_model
            self.best_complexity = oc.best_complexity
            self.best_loss = oc.best_loss
            
            from datetime import datetime

            # 定义日志文件路径
            log_file = '/home/kent/_Project/PTSjl/SRbench-GPU-prime/SRbench-GPU-prime/full_logs_SR.txt'

            # 将内容同时打印并写入文件
            def log_print(*args, **kwargs):
                # 获取当前时间，格式: 2025-01-29 14:30:05.123
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                
                # 组合时间和内容
                content = f"[{current_time}] " + " ".join(map(str, args))
                
                # 正常打印到控制台
                print(content, **kwargs)
                
                # 将内容写入文件
                with open(log_file, 'a', encoding='utf-8') as f:
                    print(content, **kwargs, file=f)

            # 使用示例
            log_print('✨self.results✨', self.results)
            log_print(f'✨ 最佳方程: {self.best_model}')
            log_print(f'✨ 模型复杂度: {self.best_complexity}')
            log_print(f'✨ 验证损失: {self.best_loss}')

        except subprocess.CalledProcessError as e:
            print("\nJulia script failed with error:")
            print(e.stderr if e.stderr else e.output)
            # 可以根据需要重新抛出异常或处理错误
            raise
        finally:
            # 清理临时文件
            os.remove(script_path)

        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        best_expr_str = self.best_model
        print('best_expr_str')
        print(best_expr_str)
        print('self.feature_names')
        print(self.feature_names)
        print('X')
        print(X.shape)

        def expr_to_Y_pred(expr_sympy, X, variables):
            functions = {
                'sin': np.sin,
                'cos': np.cos,
                'tan': np.tan,
                'exp': np.exp,
                'log': np.log,
                'sqrt': np.sqrt,
                'sinh': np.sinh,
                'cosh': np.cosh,
                'tanh': np.tanh,
                'arcsin': np.arcsin,
                'arccos': np.arccos,
                'arctan': np.arctan,
                'sign': np.sign,
                'e': np.exp(1),
                'pi': np.pi,
            }
            try:
                expr_str = str(expr_sympy)
                values = {variables[j]: X[:, j:j+1] for j in range(X.shape[1])}
                pred = eval(expr_str.lower(), functions, values) * np.ones((X.shape[0], 1))
                return pred
            except Exception as e:
                print('Exception in expr_to_Y_pred',e)
                return np.nan * np.ones((X.shape[0], 1))

        Y_pred = expr_to_Y_pred(best_expr_str, X, self.feature_names)
        return Y_pred
    

est = SymbolicRegression()


def complexity(est):
    cplx = est.best_complexity
    return cplx


def model(est):
    expr = est.best_model

    def replace_variables(expr, feature_names):
        import re
        mapping = {f'x{i}': k for i, k in enumerate(feature_names)}
        sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
        pattern = '|'.join(r'\b' + re.escape(k) + r'\b' for k in sorted_keys)
        def replace_func(match):
            return mapping[match.group(0)]
        
        new_model = re.sub(pattern, replace_func, expr)
        
        return new_model

    print('要准备化简了:')
    print('expr')
    print(expr)
    print('sympify之后:')
    expr_sympified = sp.sympify(expr)
    print(expr_sympified)
    print('化简之后:')
    expr_symplified = sp.simplify(expr_sympified)
    print(expr_symplified)
    print('求值之后：')
    expr_evalf = expr_symplified.evalf(9)
    print(expr_evalf)
    expr = str(expr_evalf)
    print('返回：')
    return expr


drmask_dir = '~/_Project/PTSjl/SRbench-GPU-prime/SRbench-GPU-prime/srbench-master/experiment/methods/PTS/dr_mask'


hyper_params = [{}]

eval_kwargs = {
    "test_params": dict(
        variables=['x'],
        operators=['Add', 'Mul', 'Sub', 'Div', 'Identity',
                'Sin', 'Cos', 'Exp', 'Log'],
        n_symbol_layers=1,
        n_inputs=1,
        use_dr_mask=True,
        dr_mask_dir=drmask_dir,
        use_const=False,
        trying_const_num=1,
        trying_const_range=[0,3],
        trying_const_n_try=3,
        
        device='cuda',
    )
}
