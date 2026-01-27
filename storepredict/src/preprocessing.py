import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from . import config

def load_data():
    print("正在加载数据...")
    df_train = pd.read_csv(config.TRAIN_PATH)
    df_store = pd.read_csv(config.STORE_PATH)
    return df_train, df_store

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def preprocess_data(df_train, df_store):
    print("正在进行数据预处理...")
    # 合并数据
    df = pd.merge(df_train, df_store, on='Store', how='left')
    # 删除列
    drop_cols = ["CompetitionOpenSinceMonth","CompetitionOpenSinceYear","Promo2SinceWeek",
                 "Promo2SinceYear","PromoInterval", "Customers", "Open"]
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    
    # 填充缺失值
    df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)
    
    # 日期特征
    df['Date'] = pd.to_datetime(df['Date'])
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df.drop(columns=['Date'], inplace=True)
    
    # 过滤销量 > 0 的数据
    df = df[df["Sales"] > 0]
    
    # 节假日处理
    df["StateHoliday"] = df["StateHoliday"].replace(0, '0')
    
    # 是否周末
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)
    
    # 季节
    df['Season'] = df['Month'].apply(get_season)
    
    # 排序以便计算滞后特征
    df = df.sort_values(by=['Store', 'Month', 'Day'])
    
    # 滞后特征 (Lags)
    df['Sales_Lag_1'] = df.groupby('Store')['Sales'].shift(1)
    df['Sales_Lag_7'] = df.groupby('Store')['Sales'].shift(7)
    df = df.dropna() # 删除包含 NaN 的行 (前 7 天)
    
    # CompetitionDistance 分箱
    df['CompetitionDistance'] = pd.cut(df['CompetitionDistance'],
                               bins=[0, 500, 2000, np.inf],
                               labels=['Near', 'Medium', 'Far'],
                               right=False,
                               include_lowest=True)
    
    # 选择列
    cols = [
        'Store', 'StoreType', 'Assortment',
        'DayOfWeek', 'Day', 'Month', 'Is_Weekend', 'Season',
        'CompetitionDistance',
        'Promo', 'Promo2',
        'StateHoliday', 'SchoolHoliday',
        'Sales_Lag_1', 'Sales_Lag_7', 'Sales'
    ]
    df = df[cols]
    
    return df

def encode_and_split(df):
    print("正在进行特征编码和数据集划分...")
    # 独热编码 (One-Hot Encoding)
    nominal_cols = ['StoreType', 'Assortment', 'StateHoliday', 'Season']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # 序数编码 (Ordinal Encoding)
    comp_dist_map = {'Near': 0, 'Medium': 1, 'Far': 2}
    df['CompetitionDistance_encoded'] = df['CompetitionDistance'].map(comp_dist_map).astype(float)
    df.drop(['CompetitionDistance'], axis=1, inplace=True)
    
    # 数据集划分
    X = df.drop(['Sales'], axis=1)
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 目标编码 (Store) - 处理数据泄露
    # 在训练集上计算均值
    store_means = X_train.join(y_train).groupby('Store')['Sales'].mean()
    global_mean = y_train.mean()
    
    # 映射到训练集和测试集
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    X_train['Store_encoded'] = X_train['Store'].map(store_means).fillna(global_mean)
    X_test['Store_encoded'] = X_test['Store'].map(store_means).fillna(global_mean)
    
    X_train.drop(['Store'], axis=1, inplace=True)
    X_test.drop(['Store'], axis=1, inplace=True)
    
    return X_train, X_test, y_train, y_test
