import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_season_sales(df, save_dir):
    if 'Season' not in df.columns:
        return
    season_sales = df.groupby('Season')['Sales'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=season_sales, x='Season', y='Sales')
    plt.title('季节平均销量')
    plt.xlabel('季节')
    plt.ylabel('销量')
    plt.savefig(os.path.join(save_dir, 'season_sales.png'))
    plt.close()

def plot_day_sales(df, save_dir):
    if 'DayOfWeek' not in df.columns:
        return
    day_sales = df.groupby('DayOfWeek')['Sales'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=day_sales, x='DayOfWeek', y='Sales')
    plt.title('星期平均销量')
    plt.xlabel('星期')
    plt.ylabel('销量')
    plt.savefig(os.path.join(save_dir, 'day_sales.png'))
    plt.close()

def plot_correlation(df, save_dir):
    cols = ['Sales', 'Promo', 'SchoolHoliday', 'DayOfWeek']
    # 确保列存在
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return
    
    corr = df[cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='RdBu')
    plt.title('相关性矩阵')
    plt.savefig(os.path.join(save_dir, 'correlation.png'))
    plt.close()


def plot_metrics_comparison(metrics_df, save_dir):
    """模型指标对比可视化"""
    metrics_to_plot = ['MAE', 'RMSE', 'R2']
    
    for metric in metrics_to_plot:
        if metric not in metrics_df.columns:
            continue
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette('husl', len(metrics_df))
        bars = plt.bar(metrics_df['Model'], metrics_df[metric], color=colors)
        
        # 在柱子上显示数值
        for bar, val in zip(bars, metrics_df[metric]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                     f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.title(f'模型对比 - {metric}')
        plt.xlabel('模型')
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'comparison_{metric}.png'))
        plt.close()
    
    print(f"指标对比图已保存至 {save_dir}")


def plot_pred_vs_true(models, X_test, y_test, save_dir):
    """预测值 vs 真实值散点图"""
    y_test_arr = np.array(y_test)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test_arr, y_pred, alpha=0.3, s=10)
        
        # 对角线
        min_val = min(y_test_arr.min(), y_pred.min())
        max_val = max(y_test_arr.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        plt.title(f'真实值 vs 预测值 - {name}')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'pred_vs_true_{name}.png'))
        plt.close()


def plot_residuals(models, X_test, y_test, save_dir):
    """残差分布图"""
    y_test_arr = np.array(y_test)
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        residuals = y_test_arr - y_pred
        
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=50)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'残差分布 - {name}')
        plt.xlabel('残差 (真实值 - 预测值)')
        plt.ylabel('频数')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'residuals_{name}.png'))
        plt.close()


def plot_feature_importance(models, feature_names, save_dir):
    """特征重要性图"""
    for name, model in models.items():
        if name == 'Stacking':
            continue
        
        if not hasattr(model, 'feature_importances_'):
            continue
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        top_n = min(15, len(feature_names))
        
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_importances, y=top_features)
        plt.title(f'特征重要性 - {name}')
        plt.xlabel('重要性')
        plt.ylabel('特征')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'feature_importance_{name}.png'))
        plt.close()
