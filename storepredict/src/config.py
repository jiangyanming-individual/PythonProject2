import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'data')

# 确保能找到 data，如果找不到则尝试直接指定绝对路径
if not os.path.exists(DATA_DIR):
    DATA_DIR = r'../../data'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
STORE_PATH = os.path.join(DATA_DIR, 'store.csv')

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

PLOT_DIR = os.path.join(OUTPUT_DIR, 'plots')
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
