# 导入所需依赖包
import pandas as pd
import joblib
import numpy as np
import warnings
import json  # 新增：用于加载模型元数据
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

warnings.filterwarnings("ignore")

# --------------------------
# 配置项（新增元数据文件路径，其余保留）
# --------------------------
MODEL_PATH = "xgboost_native_model.joblib"
METADATA_PATH = "model_metadata.json"  # 新增：模型元数据文件
TEST_CSV_PATH = "test_data.csv"
PREDICTION_SAVE_PATH = "prediction_result.csv"
LABEL_COLUMN = "出料流量"
FILLNA_VALUE = 0.0  # 新增：缺失值填充值（和训练时保持一致）


def load_model_metadata():
    """新增：加载模型元数据，用于溯源和参数验证"""
    if not METADATA_PATH or not os.path.exists(METADATA_PATH):
        print(f"⚠️  元数据文件 {METADATA_PATH} 不存在，跳过元数据加载")
        return None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print("=" * 60)
        print("✅ 模型元数据加载成功！")
        print(f"模型版本：{metadata.get('model_version', '未知')} | 训练时间：{metadata.get('train_time', '未知')}")
        print(
            f"核心参数：n_estimators={metadata.get('n_estimators', '未知')}, max_depth={metadata.get('max_depth', '未知')}")
        print("=" * 60)
        return metadata
    except Exception as e:
        print(f"⚠️  加载元数据失败：{e}，跳过元数据加载")
        return None


def predict_with_xgboost_joblib():
    try:
        # 新增：先加载模型元数据（可选，用于溯源）
        metadata = load_model_metadata()

        # 1. 加载 .joblib 原生XGBoost模型
        print(f"\n正在加载模型：{MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        print("✅ 模型加载成功！")

        # 2. 读取 CSV 测试集（优化：添加编码兼容、缺失文件友好提示）
        print(f"\n正在读取测试集：{TEST_CSV_PATH}")
        try:
            # 优化：兼容utf-8和gbk编码（解决中文/特殊字符读取失败）
            test_df = pd.read_csv(TEST_CSV_PATH, encoding="utf-8-sig")
        except:
            test_df = pd.read_csv(TEST_CSV_PATH, encoding="gbk")
        print(f"✅ 测试集读取成功，数据形状：{test_df.shape}")
        print(f"测试集前5行预览：\n{test_df.head()}")

        # 3. 数据预处理（大幅优化：增加缺失值填充、数据类型校验、特征列清洗）
        print(f"\n正在进行数据预处理...")
        if LABEL_COLUMN and LABEL_COLUMN in test_df.columns:
            X_test = test_df.drop(columns=[LABEL_COLUMN])  # 分离特征
            y_test = test_df[LABEL_COLUMN]  # 分离真实标签
        else:
            X_test = test_df
            y_test = None

        # 优化1：填充缺失值（和训练时保持一致，避免预测报错）
        X_test = X_test.fillna(FILLNA_VALUE)

        # 优化2：强制转换为数值型特征（排除非数值列，避免XGBoost预测报错）
        non_numeric_cols = []
        for col in X_test.columns:
            try:
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(FILLNA_VALUE)
            except:
                non_numeric_cols.append(col)

        # 优化3：删除非数值型特征列（避免预测异常）
        if non_numeric_cols:
            X_test = X_test.drop(columns=non_numeric_cols)
            print(f"⚠️  检测到非数值型特征列，已自动删除：{non_numeric_cols}")

        print(f"✅ 特征数据预处理完成，特征形状：{X_test.shape}（特征列数：{len(X_test.columns)}）")

        # 4. 执行预测（保留原有逻辑，增加预测异常捕获）
        print("\n正在执行模型预测...")
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            raise Exception(f"预测失败：请检查特征列是否与训练集一致，错误详情：{e}")

        # 4.2 预测类别概率（可选）
        y_pred_proba = None
        try:
            y_pred_proba = model.predict_proba(X_test)
        except AttributeError:
            print("⚠️  模型不支持预测概率，跳过概率输出")
        print("✅ 预测完成！")

        # 5. 整理预测结果（保留原有逻辑，优化预览格式）
        result_df = test_df.copy()
        result_df["predicted_label"] = y_pred  # 添加预测类别列

        if y_pred_proba is not None:
            # 为每个类别添加概率列
            for i in range(y_pred_proba.shape[1]):
                result_df[f"predicted_proba_class_{i}"] = y_pred_proba[:, i]

        # 优化：预览结果格式更清晰
        preview_cols = ["predicted_label"] + ([LABEL_COLUMN] if LABEL_COLUMN in test_df.columns else [])
        print(f"\n预测结果前5行预览：\n{result_df[preview_cols].head()}")

        # 6. 保存预测结果（优化：添加编码兼容，避免中文乱码）
        result_df.to_csv(PREDICTION_SAVE_PATH, index=False, encoding="utf-8-sig")
        print(f"\n✅ 预测结果已保存到：{PREDICTION_SAVE_PATH}")

        # 7. 若有真实标签，输出分类评估指标（保留原有逻辑，优化格式）
        if y_test is not None:
            accuracy = accuracy_score(y_test, y_pred)
            print(f"\n========== 模型评估指标 ==========")
            print(f"准确率 (Accuracy): {accuracy:.4f}")

            # 处理多分类/二分类
            unique_labels = np.unique(np.concatenate([y_test, y_pred]))
            avg = 'binary' if len(unique_labels) == 2 else 'weighted'
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
        print(f"❌ 错误：文件不存在 - {e}")
    except Exception as e:
        print(f"❌ 错误：预测过程中出现异常 - {e}")
        raise

if __name__ == "__main__":
    # 新增：导入os模块（用于元数据文件存在性判断）
    import os
    # 执行预测
    predict_with_xgboost_joblib()