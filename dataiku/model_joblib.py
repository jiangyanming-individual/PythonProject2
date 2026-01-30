# 导入所需依赖包
import pandas as pd
import joblib
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

warnings.filterwarnings("ignore")

MODEL_PATH = "xgboost_native_model.joblib"  # 关键修改：.pkl → .joblib
TEST_CSV_PATH = "test_data.csv"  # CSV测试集路径
PREDICTION_SAVE_PATH = "prediction_result.csv"  # 预测结果保存路径
LABEL_COLUMN = "出料流量"


def predict_with_xgboost_joblib():
    try:
        # 1. 加载 .joblib 原生XGBoost模型（joblib.load()
        print(f"正在加载模型：{MODEL_PATH}")
        model = joblib.load(MODEL_PATH)  # 核心：仍用joblib.load()
        print("模型加载成功！")
        # 2. 读取 CSV 测试集（无修改）
        print(f"\n正在读取测试集：{TEST_CSV_PATH}")  # 修复原代码的换行符缺失（\ → \n）
        test_df = pd.read_csv(TEST_CSV_PATH)
        print(f"测试集读取成功，数据形状：{test_df.shape}")
        print(f"测试集前5行预览：\n{test_df.head()}")

        # 3. 数据预处理：分离特征和标签（若有标签列）（无修改）
        if LABEL_COLUMN and LABEL_COLUMN in test_df.columns:
            X_test = test_df.drop(columns=[LABEL_COLUMN])  # 特征数据
            y_test = test_df[LABEL_COLUMN]  # 真实标签
        else:
            X_test = test_df  # 无标签列，全部为特征
            y_test = None
        print(f"\n特征数据预处理完成，特征形状：{X_test.shape}")

        # 4. 执行预测（无修改）
        print("\n正在执行模型预测...")
        # 4.1 预测类别
        y_pred = model.predict(X_test)
        # 4.2 预测类别概率（可选，用于评估模型置信度）
        try:
            y_pred_proba = model.predict_proba(X_test)
        except AttributeError:
            y_pred_proba = None
            print("模型不支持预测概率，跳过概率输出")
        print("预测完成！")

        # 5. 整理预测结果（无修改）
        result_df = test_df.copy()
        result_df["predicted_label"] = y_pred  # 添加预测类别列
        if y_pred_proba is not None:
            # 为每个类别添加概率列
            for i in range(y_pred_proba.shape[1]):
                result_df[f"predicted_proba_class_{i}"] = y_pred_proba[:, i]
        print(
            f"\n预测结果前5行预览：\n{result_df[['predicted_label'] + ([LABEL_COLUMN] if LABEL_COLUMN else [])].head()}")

        # 6. （可选）保存预测结果到新CSV（无修改）
        result_df.to_csv(PREDICTION_SAVE_PATH, index=False)
        print(f"\n预测结果已保存到：{PREDICTION_SAVE_PATH}")

        # 7. 若有真实标签，输出分类评估指标（无修改）
        if y_test is not None:
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n========== 模型评估指标 ==========")
            print(f"准确率 (Accuracy): {accuracy:.4f}")

            # 处理多分类/二分类
            unique_labels = np.unique(np.concatenate([y_test, y_pred]))
            if len(unique_labels) == 2:
                avg = 'binary'
            else:
                avg = 'weighted'
            precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
            recall = recall_score(y_test, y_pred, average=avg, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

            print(f"精确率 (Precision): {precision:.4f}")
            print(f"召回率 (Recall):    {recall:.4f}")
            print(f"F1分数 (F1-Score): {f1:.4f}")
            print("===================================")

            print(f"\n混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
            print(f"\n详细分类报告:\n{classification_report(y_test, y_pred, zero_division=0)}")

        return result_df

    except FileNotFoundError as e:
        print(f"错误：文件不存在 - {e}")
    except Exception as e:
        print(f"错误：预测过程中出现异常 - {e}")
        raise

if __name__ == "__main__":

    predict_with_xgboost_joblib()