import sys
import os
import joblib
import pandas as pd
from src import config
from src.preprocessing import load_data, preprocess_data, encode_and_split
from src.visualization import (plot_season_sales, plot_day_sales, plot_correlation,
                               plot_metrics_comparison, plot_pred_vs_true, 
                               plot_residuals, plot_feature_importance)
from src.models import train_all_models
from src.evaluation import evaluate_all_models

def main():

    print("开始运行 Store Predict 项目...")
    # 1. 加载数据
    try:
        df_train, df_store = load_data()
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件。请确保数据在 {config.DATA_DIR} 目录下。")
        print(e)
        return
    
    # 2. 预处理
    df_processed = preprocess_data(df_train, df_store)
    
    # 3. 可视化
    print(f"正在生成可视化图表，保存至 {config.PLOT_DIR}...")
    plot_season_sales(df_processed, config.PLOT_DIR)
    plot_day_sales(df_processed, config.PLOT_DIR)
    plot_correlation(df_processed, config.PLOT_DIR)
    
    # 4. 数据集划分
    X_train, X_test, y_train, y_test = encode_and_split(df_processed)
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    # 5. 模型训练
    print("开始训练模型 (随机森林, XGBoost, LightGBM, Stacking)...")
    models = train_all_models(X_train, y_train)
    # 6. 模型评估
    metrics_df = evaluate_all_models(models, X_test, y_test)
    print("\n模型评估结果:")
    print(metrics_df)
    
    # 保存评估结果
    metrics_path = os.path.join(config.OUTPUT_DIR, 'metrics_report.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"评估结果已保存至 {metrics_path}")
    
    # 7. 指标对比可视化
    print("正在生成模型评估可视化图表...")
    plot_metrics_comparison(metrics_df, config.PLOT_DIR)
    plot_pred_vs_true(models, X_test, y_test, config.PLOT_DIR)
    plot_residuals(models, X_test, y_test, config.PLOT_DIR)
    plot_feature_importance(models, X_train.columns.tolist(), config.PLOT_DIR)
    
    # 8. 保存模型
    print(f"正在保存模型至 {config.MODEL_DIR}...")
    for name, model in models.items():
        joblib.dump(model, os.path.join(config.MODEL_DIR, f'{name}.pkl'))
    
    print("运行完成！")

if __name__ == "__main__":
    main()
