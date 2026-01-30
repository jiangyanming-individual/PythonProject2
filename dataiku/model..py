from sklearn.base import BaseEstimator
import xgboost as xgb
import numpy as np
import mlflow
import joblib
import os

MLFLOW_TRACKING_URI = "http://mlflow.ml.svc.cluster.local:5000"
MLFLOW_EXPERIMENT_NAME = "test_mlflow_kubeflow"
MODEL_REGISTER_NAME = "mlflow_kubeflow_xgboost_test"

# 初始化MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

class MyXGBoostClassifier(BaseEstimator):
    """基于XGBoost的分类器（符合Dataiku自定义模型规范，序列化原生XGBoost模型）"""

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.xgb_clf = None
        self.classes_ = None
        self.run_id = None
        self.model_version = None

    def fit(self, X, y):
        """训练模型 + 序列化原生XGBoost模型为.pkl（避免自定义类路径问题）"""

        # 1. 基础训练逻辑
        self.classes_ = np.unique(y)
        self.xgb_clf = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.xgb_clf.fit(X, y)

        # 2. MLflow注册逻辑（核心：序列化 self.xgb_clf 而非 self）
        with mlflow.start_run() as run:
            # 记录训练参数
            mlflow.log_params({
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "eval_metric": "logloss"
            })

            # 3. 核心修改：序列化 XGBoost 原生模型（self.xgb_clf）
            model_filename = "xgboost_native_model.pkl"
            joblib.dump(self.xgb_clf, model_filename)  # 传入 self.xgb_clf 而非 self

            # 4. 上传到 MLflow
            mlflow.log_artifact(
                local_path=model_filename,
                artifact_path="model"
            )
            # 5. 手动注册模型
            model_uri = f"runs:/{run.info.run_id}/model/{model_filename}"
            mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_REGISTER_NAME
            )
            # 6. 获取 run_id 和版本号
            self.run_id = run.info.run_id
            client = mlflow.MlflowClient()
            latest_versions = client.get_latest_versions(MODEL_REGISTER_NAME)
            self.model_version = latest_versions[0].version

            # 7. 打印日志
            print(f"模型训练+joblib序列化完成（原生XGBoost模型，.pkl格式）！")
            print(f"MLflow Run ID: {self.run_id}")
            print(f"模型名称: {MODEL_REGISTER_NAME}, 版本: {self.model_version}")

            # 8. 清理本地临时文件（不变）
            if os.path.exists(model_filename):
                os.remove(model_filename)
        return self

    def predict(self, X):
        if self.xgb_clf is None:
            raise ValueError("模型未训练，请先调用fit(X, y)方法！")
        return self.xgb_clf.predict(X)

    def predict_proba(self, X):
        if self.xgb_clf is None:
            raise ValueError("模型未训练，请先调用fit(X, y)方法！")
        return self.xgb_clf.predict_proba(X)

# 初始化模型
clf = MyXGBoostClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)