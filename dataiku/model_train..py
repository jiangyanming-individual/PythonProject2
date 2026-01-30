import xgboost as xgb
import pandas as pd
import joblib

# --------------------------
# 配置项
# --------------------------
TRAIN_DATA_PATH = "train_data.csv"  # 训练数据路径
MODEL_SAVE_PATH = "xgboost_local_model.pkl"  # 模型保存路径
LABEL_COLUMN = "出料流量"  # 标签列名

# 模型参数
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "logloss"
}


def train_and_save_model():
    """加载训练数据，训练XGBoost分类模型，保存为.pkl文件"""
    
    # 1. 加载训练数据
    print(f"正在加载训练数据：{TRAIN_DATA_PATH}")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"训练数据加载成功，数据形状：{train_df.shape}")
    print(f"训练数据前5行预览：\n{train_df.head()}")
    
    # 2. 分离特征和标签
    X_train = train_df.drop(columns=[LABEL_COLUMN])
    y_train = train_df[LABEL_COLUMN]
    print(f"\n特征数据形状：{X_train.shape}")
    print(f"标签分布：\n{y_train.value_counts()}")
    
    # 3. 构建并训练模型
    print(f"\n正在构建XGBoost分类模型...")
    print(f"模型参数：{MODEL_PARAMS}")
    
    model = xgb.XGBClassifier(**MODEL_PARAMS)
    
    print("\n正在训练模型...")
    model.fit(X_train, y_train)
    print("模型训练完成！")
    
    # 4. 使用joblib保存模型
    print(f"\n正在保存模型到：{MODEL_SAVE_PATH}")
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"模型已保存为: {MODEL_SAVE_PATH}")
    
    # 5. 输出特征重要性（可选）
    print(f"\n========== 特征重要性 ==========")
    feature_importance = model.feature_importances_
    for name, importance in zip(X_train.columns, feature_importance):
        print(f"{name}: {importance:.4f}")
    print("===================================")
    
    return model


if __name__ == "__main__":
    train_and_save_model()