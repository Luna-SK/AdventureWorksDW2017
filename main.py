import argparse
import subprocess
from pathlib import Path

def run_analysis(script_name, task_name):
    """运行指定的分析脚本并捕获输出"""
    print(f"\n开始{task_name}...")
    try:
        result = subprocess.run(['python', f'src/{script_name}'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print(result.stdout)
        if result.stderr:
            print("警告/错误信息：")
            print(result.stderr)
        print(f"{task_name}完成！")
        if task_name == '季节性产品关联规则挖掘':
            print("分析报告已生成在 reports/季节性产品关联规则/ 目录下")
        elif task_name == '客户流失预测分析':
            print("分析报告已生成在 reports/客户流失预测/ 目录下")
        elif task_name == '澳元客户RFM分析':
            print("分析报告已生成在 reports/澳元客户RFM/ 目录下")
    except subprocess.CalledProcessError as e:
        print(f"运行{task_name}时出错：")
        print(e.stderr)
        raise

def main():
    parser = argparse.ArgumentParser(description='数据仓库与数据挖掘综合实践项目')
    parser.add_argument('--task', 
                       choices=['churn', 'association', 'rfm', 'all'],
                       default='all',
                       help='选择要运行的分析任务：churn(客户流失预测), association(关联规则挖掘), rfm(RFM分析), all(全部)')
    
    args = parser.parse_args()
    
    # 确保reports目录存在
    Path("reports").mkdir(exist_ok=True)
    
    # 根据参数运行相应的分析任务
    if args.task == 'all' or args.task == 'churn':
        run_analysis('churn_predict.py', '客户流失预测分析')
    
    if args.task == 'all' or args.task == 'association':
        run_analysis('seasonal_association.py', '季节性产品关联规则挖掘')
    
    if args.task == 'all' or args.task == 'rfm':
        run_analysis('rfm_analysis.py', '澳元客户RFM分析')
    
    print("\n所有选定的分析任务已完成！")

if __name__ == "__main__":
    main()
