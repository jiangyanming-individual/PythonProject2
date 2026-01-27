import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb


def get_rf_model():
    """RandomForest 固定参数"""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )


def get_xgb_model():
    """XGBoost 固定参数 (参考 demo.py)"""
    return xgb.XGBRegressor(
        n_estimators=500,
        max_depth=9,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.3,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        n_jobs=-1
    )


def get_gb_model():
    """GradientBoosting 固定参数 (参考 demo.py)"""
    return GradientBoostingRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )


def build_stacking_model(estimators):
    """Stacking 模型"""
    return StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=3,
        n_jobs=-1
    )


def train_all_models(X_train, y_train):
    """训练所有模型（使用固定参数，不进行超参数搜索）"""
    
    # RandomForest
    print("正在训练 RandomForest...")
    rf_model = get_rf_model()
    rf_model.fit(X_train, y_train)
    
    # XGBoost
    print("正在训练 XGBoost...")
    xgb_model = get_xgb_model()
    xgb_model.fit(X_train, y_train)
    
    # GradientBoosting
    print("正在训练 GradientBoosting...")
    gb_model = get_gb_model()
    gb_model.fit(X_train, y_train)
    
    # 构建 Stacking 模型
    estimators = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('gb', gb_model)
    ]
    print("正在训练 Stacking 模型...")
    stacking_model = build_stacking_model(estimators)
    stacking_model.fit(X_train, y_train)
    
    models = {
        'RandomForest': rf_model,
        'XGBoost': xgb_model,
        'GradientBoosting': gb_model,
        'Stacking': stacking_model
    }
    
    return models
