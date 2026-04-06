import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

# 创建输出目录
def create_output_dir():
    """创建输出目录"""
    output_dir = "output/metrics"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def plot_multi_loss_curve(log_file):
    """绘制多重 Loss 曲线"""
    # 检查文件是否存在
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"找不到真实日志 {log_file}，绝不使用模拟数据！")
    
    # 读取数据
    df = pd.read_csv(log_file)
    
    plt.figure(figsize=(12, 8), dpi=300)
    
    # 绘制总损失
    plt.plot(df['Epoch'], df['Loss_Total'], label='Total Loss', linewidth=2, marker='o', markersize=4, color='#1f77b4')
    # 绘制分类损失
    plt.plot(df['Epoch'], df['Loss_Cls'], label='Classification Loss', linewidth=2, marker='s', markersize=4, color='#ff7f0e')
    # 绘制回归损失
    plt.plot(df['Epoch'], df['Loss_Reg'], label='Regression Loss', linewidth=2, marker='^', markersize=4, color='#2ca02c')
    
    plt.title('Training Loss Curves', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_dir = create_output_dir()
    output_path = os.path.join(output_dir, 'fig_1_loss.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Multi-loss curve saved to: {output_path}')

def plot_real_metrics(log_file):
    """绘制真实指标曲线"""
    # 检查文件是否存在
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"找不到真实日志 {log_file}，绝不使用模拟数据！")
    
    # 读取数据
    df = pd.read_csv(log_file)
    
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=300)
    
    # 左 Y 轴：Matched_Q 和 Pos_Acc
    color = '#1f77b4'
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Matched_Q / Pos_Acc', color=color, fontsize=14)
    ax1.plot(df['Epoch'], df['Matched_Q'], label='Matched Queries', linewidth=2, marker='o', markersize=4, color=color)
    ax1.plot(df['Epoch'], df['Pos_Acc'], label='Positive Accuracy', linewidth=2, marker='s', markersize=4, color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left', fontsize=12)
    
    # 右 Y 轴：XYZ_Err_m
    ax2 = ax1.twinx()
    color = '#d62728'
    ax2.set_ylabel('XYZ Error (m)', color=color, fontsize=14)
    ax2.plot(df['Epoch'], df['XYZ_Err_m'], label='XYZ Error', linewidth=2, marker='^', markersize=4, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right', fontsize=12)
    
    plt.title('Training Metrics', fontsize=16, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_dir = create_output_dir()
    output_path = os.path.join(output_dir, 'fig_2_metrics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Real metrics plot saved to: {output_path}')

def plot_lr_curve(log_file):
    """绘制学习率衰减曲线"""
    # 检查文件是否存在
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"找不到真实日志 {log_file}，绝不使用模拟数据！")
    
    # 读取数据
    df = pd.read_csv(log_file)
    
    plt.figure(figsize=(12, 8), dpi=300)
    
    # 绘制学习率
    plt.plot(df['Epoch'], df['lr'], label='Learning Rate', linewidth=2, marker='o', markersize=4, color='#1f77b4')
    
    plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    output_dir = create_output_dir()
    output_path = os.path.join(output_dir, 'fig_3_lr.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'LR curve saved to: {output_path}')

def find_latest_log_file():
    """找到最新时间戳文件夹中的 train_log.csv"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        raise FileNotFoundError(f"找不到 logs 目录: {logs_dir}")
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, d))]
    if not subdirs:
        raise FileNotFoundError(f"logs 目录中没有子文件夹")
    
    # 按时间戳排序，最新的在前
    subdirs.sort(reverse=True)
    
    # 查找最新文件夹中的 train_log.csv
    for subdir in subdirs:
        log_file = os.path.join(logs_dir, subdir, 'train_log.csv')
        if os.path.exists(log_file):
            return log_file
    
    raise FileNotFoundError(f"在 logs 目录中找不到 train_log.csv 文件")

if __name__ == '__main__':
    """主函数"""
    try:
        # 自动找到最新的日志文件
        log_file = find_latest_log_file()
        print(f"使用最新的训练日志: {log_file}")
        
        # 调用绘图函数
        plot_multi_loss_curve(log_file)
        plot_real_metrics(log_file)
        plot_lr_curve(log_file)
        
        print('\nAll plots generated successfully!')
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
