import sys
import dataiku
import numpy as np
import pandas as pd
import sklearn as sk
import dataiku.core.pandasutils as pdu
from dataiku.doctor.preprocessing import PCA
from collections import defaultdict, Counter


pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# We apply the preparation that you defined. You should not modify this.
preparation_steps = []
preparation_output_schema = {'columns': [{'name': '三效液相温度', 'type': 'double'}, {'name': '三效气相温度', 'type': 'double'}, {'name': '三效蒸发室气相压力', 'type': 'double'}, {'name': '出料泵电流1', 'type': 'double'}, {'name': '出料泵电流2', 'type': 'double'}, {'name': '出料流量', 'type': 'double'}], 'userModified': False}

ml_dataset_handle = dataiku.Dataset('train_data')
ml_dataset_handle.set_preparation_steps(preparation_steps, preparation_output_schema)
ml_dataset = ml_dataset_handle.get_dataframe(limit = 100000)

print ('Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
# Five first records",
ml_dataset.head(5)

ml_dataset = ml_dataset[['三效液相温度', '出料流量', '出料泵电流2', '三效气相温度', '三效蒸发室气相压力', '出料泵电流1']]

# astype('unicode') does not work as expected

def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x,'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)


categorical_features = []
numerical_features = ['三效液相温度', '出料泵电流2', '三效气相温度', '三效蒸发室气相压力', '出料泵电流1']
text_features = []
from dataiku.doctor.utils import datetime_to_epoch
for feature in categorical_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in text_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in numerical_features:
    if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
        ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
    else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')

target_map = {'0.0': 0, '1.0': 1}
ml_dataset['__target__'] = ml_dataset['出料流量'].map(str).map(target_map)
del ml_dataset['出料流量']


# Remove rows for which the target is unknown.
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]

ml_dataset['__target__'] = ml_dataset['__target__'].astype(np.int64)


train, test = pdu.split_train_valid(ml_dataset, prop=0.8)
print ('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
print ('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))

drop_rows_when_missing = []
impute_when_missing = [{'feature': '三效液相温度', 'impute_with': 'MEAN'}, {'feature': '出料泵电流2', 'impute_with': 'MEAN'}, {'feature': '三效气相温度', 'impute_with': 'MEAN'}, {'feature': '三效蒸发室气相压力', 'impute_with': 'MEAN'}, {'feature': '出料泵电流1', 'impute_with': 'MEAN'}]

# Features for which we drop rows with missing values"
for feature in drop_rows_when_missing:
    train = train[train[feature].notnull()]
    test = test[test[feature].notnull()]
    print ('Dropped missing records in %s' % feature)

# Features for which we impute missing values"
for feature in impute_when_missing:
    if feature['impute_with'] == 'MEAN':
        v = train[feature['feature']].mean()
    elif feature['impute_with'] == 'MEDIAN':
        v = train[feature['feature']].median()
    elif feature['impute_with'] == 'CREATE_CATEGORY':
        v = 'NULL_CATEGORY'
    elif feature['impute_with'] == 'MODE':
        v = train[feature['feature']].value_counts().index[0]
    elif feature['impute_with'] == 'CONSTANT':
        v = feature['value']
    train[feature['feature']] = train[feature['feature']].fillna(v)
    test[feature['feature']] = test[feature['feature']].fillna(v)
    print ('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))
rescale_features = {'三效液相温度': 'AVGSTD', '出料泵电流2': 'AVGSTD', '三效气相温度': 'AVGSTD', '三效蒸发室气相压力': 'AVGSTD', '出料泵电流1': 'AVGSTD'}
for (feature_name, rescale_method) in rescale_features.items():
    if rescale_method == 'MINMAX':
        _min = train[feature_name].min()
        _max = train[feature_name].max()
        scale = _max - _min
        shift = _min
    else:
        shift = train[feature_name].mean()
        scale = train[feature_name].std()
    if scale == 0.:
        del train[feature_name]
        del test[feature_name]
        print ('Feature %s was dropped because it has no variance' % feature_name)
    else:
        print ('Rescaled %s' % feature_name)
        train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
        test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale

X_train = train.drop('__target__', axis=1)
X_test = test.drop('__target__', axis=1)

y_train = np.array(train['__target__'])
y_test = np.array(test['__target__'])



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
    """基于XGBoost的分类器（符合Dataiku自定义模型规范，明确用joblib格式保存）"""

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
        """训练模型 + 明确用joblib格式序列化原生XGBoost模型（.joblib后缀）"""
        # 1. 基础训练逻辑（不变）
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

        # 2. MLflow注册逻辑（核心：明确joblib格式，用.joblib后缀标识）
        with mlflow.start_run() as run:
            # 记录训练参数（不变）
            mlflow.log_params({
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "random_state": self.random_state,
                "eval_metric": "logloss"
            })

            # 3.
            model_filename = "xgboost_native_model.joblib"  # 关键修改：.joblib
            # joblib.dump() 本身就是joblib格式的核心序列化方法，此处明确保存为.joblib文件
            joblib.dump(self.xgb_clf, model_filename)

            # 4. 上传到 MLflow（不变，自动识别.joblib文件）
            mlflow.log_artifact(
                local_path=model_filename,
                artifact_path="model"
            )

            # 5. 手动注册模型（更新模型URI，对应.joblib文件）
            model_uri = f"runs:/{run.info.run_id}/model/{model_filename}"
            mlflow.register_model(
                model_uri=model_uri,
                name=MODEL_REGISTER_NAME
            )

            # 6. 获取 run_id 和版本号（不变）
            self.run_id = run.info.run_id
            client = mlflow.MlflowClient()
            latest_versions = client.get_latest_versions(MODEL_REGISTER_NAME)
            self.model_version = latest_versions[0].version

            # 7. 打印日志（标注joblib格式和.joblib后缀文件）
            print(f"模型训练+joblib序列化完成（原生XGBoost模型，.joblib格式）！")
            print(f"MLflow Run ID: {self.run_id}")
            print(f"模型名称: {MODEL_REGISTER_NAME}, 版本: {self.model_version}")
            print(f"生成的joblib格式模型文件: {model_filename}")

            # 8. 清理本地临时.joblib文件（不变）
            if os.path.exists(model_filename):
                os.remove(model_filename)
                print(f"本地临时.joblib模型文件已删除")

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

clf.class_weight = "balanced"

from dataiku.doctor.utils.skcompat import dku_fit  # Make the linear models regressors normalization compatible with all sklearn versions. Simply call clf.fit for non-linear models.

dku_fit(clf, X_train, y_train)

time_predictions = clf.predict(X_test)
time_probas = clf.predict_proba(X_test)
predictions = pd.Series(data=_predictions, index=X_test.index, name='predicted_value')
cols = [
    u'probability_of_value_%s' % label
    for (_, label) in sorted([(int(target_map[label]), label) for label in target_map])
]
probabilities = pd.DataFrame(data=_probas, index=X_test.index, columns=cols)

# Build scored dataset
results_test = X_test.join(predictions, how='left')
results_test = results_test.join(probabilities, how='left')
results_test = results_test.join(test['__target__'], how='left')
results_test = results_test.rename(columns= {'__target__': '出料流量'})

from dataiku.doctor.utils.metrics import mroc_auc_score

y_test_ser = pd.Series(y_test)

print('AUC value:', mroc_auc_score(y_test_ser, _probas))

inv_map = { target_map[label] : label for label in target_map}
predictions.map(inv_map)

