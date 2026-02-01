

import joblib
import pickle

# 配置文件路径
FILE_PATH = "xgboost_native_model.pkl"  # 替换为你的 .joblib 或 .pkl 文件路径

def view_serialized_file(file_path):
    try:
        # 步骤 1：加载序列化文件（joblib 兼容 .joblib 和 .pkl 格式）
        print(f"正在加载文件：{file_path}")
        obj = joblib.load(file_path)
        print(f"文件加载成功，对象类型：{type(obj)}\n")

        # 步骤 2：方式 1 - 直接打印对象（查看基本信息）
        print("=== 方式 1：直接打印对象（基本信息）===")
        print(obj)
        print("\n")

        # 步骤 3：方式 2 - 用 dir() 查看对象的所有属性和方法（找到可访问的内容）
        print("=== 方式 2：查看对象所有属性和方法（前20个，避免输出过长）===")
        all_attributes = dir(obj)
        # 过滤掉内置魔法方法（以 __ 开头结尾），更易读
        useful_attributes = [attr for attr in all_attributes if not (attr.startswith('__') and attr.endswith('__'))]
        print(f"对象共有 {len(useful_attributes)} 个有效属性/方法，前20个如下：")
        print(useful_attributes[:20])  # 打印前20个，可去掉 [:20] 查看全部
        print("\n")

        # 步骤 4：方式 3 - 针对性打印关键属性（根据对象类型调整，这里以通用场景为例）
        print("=== 方式 3：针对性打印关键属性（示例）===")
        # 示例1：如果是字典对象（序列化的字典）
        if isinstance(obj, dict):
            print("该文件是字典对象，内容如下：")
            for key, value in obj.items():
                print(f"  {key}: {value}")
        # 示例2：如果是 sklearn/XGBoost 模型对象（你的场景）
        elif hasattr(obj, "get_params"):
            print("该文件是机器学习模型对象，模型参数如下：")
            model_params = obj.get_params()
            for param, value in model_params.items():
                print(f"  {param}: {value}")
        # 其他对象可按需扩展
        else:
            print("暂不支持该对象类型的针对性打印，可通过 dir() 查看属性后手动打印")

        return obj

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
    except Exception as e:
        print(f"错误：加载/查看文件失败 - {e}")

if __name__ == "__main__":
    # 调用函数查看文件内容
    loaded_obj = view_serialized_file(FILE_PATH)