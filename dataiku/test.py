import pandas as pd
import pickle
import joblib
import xgboost as xgb

# 配置你的模型文件路径（.pkl 或 .joblib 均可）
MODEL_PATH1 = "xgboost_local_model.pkl"  # 替换为你的模型文件路径
MODEL_PATH2 = "xgboost_native_model.pkl"  # 替换为你的模型文件路径

def print_xgb_full_params(model_path1,model_path2):
    # 1. 加载模型
    model = joblib.load(model_path1)
    print("=== XGBClassifier dataiku===")

    # 2. 用 get_params() 获取所有参数（返回字典，无截断）
    full_params = model.get_params()

    # 3. 遍历打印所有参数（包括 None 值参数）
    for param_name, param_value in sorted(full_params.items()):
        print(f"{param_name}: {param_value}")

    model = joblib.load(model_path2)
    print("============================ XGBClassifier  local==============================")

    # 2. 用 get_params() 获取所有参数（返回字典，无截断）
    full_params = model.get_params()

    # 3. 遍历打印所有参数（包括 None 值参数）
    for param_name, param_value in sorted(full_params.items()):
        print(f"{param_name}: {param_value}")

    return model, full_params





if __name__ == "__main__":
    xgb_model, full_params = print_xgb_full_params(MODEL_PATH1,MODEL_PATH2)