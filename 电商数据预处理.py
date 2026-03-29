import pandas as pd
import numpy as np
from typing import Dict,Tuple,Optional

# 列名映射：英文 -> 中文
COLUMN_MAPPING: Dict[str, str] = {
    "CustomerID": "顾客ID",
    "Churn": "流失标志",
    "Tenure": "使用平台时长（月）",
    "PreferredLoginDevice": "首选登录设备",
    "CityTier": "城市等级",
    "WarehouseToHome": "仓库到顾客家的距离（公里）",
    "MaritalStatus": "婚姻状况",
    "AgeGroup": "年龄分组",
    "Gender": "性别",
    "HourSpendOnApp": "APP使用时长（小时）",
    "OrderCount": "上月订单数量",
    "OrderAmountHikeFromlastYear": "订单金额较去年增长百分比",
    "DaySinceLastOrder": "距上次下单天数",
    "PreferedOrderCat": "上月首选订单类别",
    "NumberOfStreamerFollowed": "关注的主播数量",
    "SatisfactionScore": "服务满意度评分",
    "Complain": "上月投诉情况",
    "CouponUsed": "上月使用优惠券数量",
    "DiscountAmount": "上月平均返现金额",
}

# 年龄分组映射：数字/字符串 -> 中文描述
AGE_GROUP_MAPPING: Dict[object, str] = {
    "1": "20岁以下",
    1: "20岁以下",
    "2": "20-29岁",
    2: "20-29岁",
    "3": "30-39岁",
    3: "30-39岁",
    "4": "40-49岁",
    4: "40-49岁",
    "5": "50-59岁",
    5: "50-59岁",
    "6": "60岁以上",
    6: "60岁以上",
}

# 订单类别映射：英文 -> 中文
ORDER_CATEGORY_MAPPING = {
    "Fashion": "时尚类",
    "Grocery": "食品杂货类",
    "Household": "家居类",
    "Mobile Phone": "手机类",
    "Laptop & Accessory": "笔记本电脑及配件类",
    "Others": "其他类",
}

# 性别映射：英文 -> 中文
GENDER_MAPPING = {"Male": "男性", "Female": "女性"}

# 设备类型映射：英文 -> 中文
DEVICE_MAPPING = {
    "Mobile Phone": "手机",
    "Phone": "手机（可能为功能机）",
    "Pad": "平板电脑",
}

# 城市等级映射：数字 -> 中文
CITY_TIER_MAPPING = {
    "1": "一线城市",
    1: "一线城市",
    "2": "二线城市",
    2: "二线城市",
    "3": "三线城市",
    3: "三线城市",
}

# 婚姻状况映射：英文 -> 中文
MARITAL_STATUS_MAPPING = {"Single": "单身", "Married": "已婚", "Divorced": "离异"}

# 流失标志映射：数字 -> 中文
CHURN_MAPPING = {0: "未流失", 1: "流失", "0": "未流失", "1": "流失"}

# 投诉情况映射：数字 -> 中文
COMPLAIN_MAPPING = {0: "无", 1: "有", "0": "无", "1": "有"}

# 目标列的顺序（按照中文数据文件的顺序）
TARGET_COLUMNS = [
    "顾客ID",
    "流失标志",
    "使用平台时长（月）",
    "首选登录设备",
    "城市等级",
    "仓库到顾客家的距离（公里）",
    "婚姻状况",
    "年龄分组",
    "性别",
    "APP使用时长（小时）",
    "上月订单数量",
    "订单金额较去年增长百分比",
    "距上次下单天数",
    "上月首选订单类别",
    "关注的主播数量",
    "服务满意度评分",
    "上月投诉情况",
    "上月使用优惠券数量",
    "上月平均返现金额",
]

def ensure_column_order(dataframe: pd.DataFrame) -> pd.DataFrame:
    """确保所有列都存在，对于不存在的列创建空列"""
    for column in TARGET_COLUMNS:
        if column not in dataframe.columns:
            print(f"检验列函数:'{column}'不存在，创建空列")
            dataframe[column] = np.nan
    return dataframe[TARGET_COLUMNS]

def test_ensure_column_order():
    try:
        print("-------检验列函数测试---")
        test_data={
            "使用平台时长（月）": [5,12,8],
            "顾客ID": [1001, 1002, 1003],
            "性别":["男","女","男"],
            "额外性别列": ["A","B","C"],
        }
        test_dt=pd.DataFrame(test_data)
        print(f"原始列顺序：{list(test_dt.columns)}")

        print("\n 调用检验列函数: ")
        result_df= ensure_column_order(test_dt)
        print(f"处理后列顺序：{list(result_df.columns)}")
        
        print("\n多方验证测试结果:")
        #验证确实列是否都被创建:
        assert "流失标志"in result_df.columns, "缺失的流失标志应当被自动填充"
        assert result_df["流失标志"].isna().all(), "新创建的流失标志列应当全部填充为空值"
        print("缺失列创建完毕")

        #原始数据是否保留
        assert result_df["顾客ID"].iloc[0]==1001, "顾客ID的第一列应当为1001"
        assert result_df["性别"].iloc[1]=="女", "性别的第二列应当为女"
        assert result_df["使用平台时长（月）"].iloc[2]==8,"使用平台时长的第三列应当为8"
        print("原始数据保留完毕")

        #验证额外列有无被移除
        assert "额外测试列" not in result_df.columns, "不在目标列的额外测试列应当被移除"
        print("额外列移除完毕")

        
    except AssertionError as e:
        print(f"\n-------断言失败:{e}")
    except Exception as e:
        print(f"\n-------发生意外错误:{type(e).__name__}:{e}")
        
def convert_all_values(dataframe:pd.DataFrame) -> pd.DataFrame:
    #调用映射常量
    conversion_pairs=[
        ("流失标志", CHURN_MAPPING),
        ("城市等级", CITY_TIER_MAPPING),
        ("首选登录设备", DEVICE_MAPPING),
        ("婚姻状况", MARITAL_STATUS_MAPPING),
        ("年龄分组", AGE_GROUP_MAPPING),
        ("上月首选订单类别", ORDER_CATEGORY_MAPPING),
        ("上月投诉情况", COMPLAIN_MAPPING),
        ("性别",GENDER_MAPPING)
    ]

    #逐列转换
    for column_name,mapping_dict in conversion_pairs:
        if column_name in dataframe.columns and mapping_dict:
            dataframe[column_name]=dataframe[column_name].map(
                lambda x:mapping_dict.get(x,x)
            )

    return dataframe

def test():
    try:
        #创建测试数据
        test_data={
            "顾客ID": [60001, 60002, 60003],
            "流失标志": [0,1,0],
            "城市等级": [1,2,3],
            "性别":["Male","Female","Male"],
            "年龄分组": [2,4,5],
            "上月首选订单类别": ["Mobile Phone","Laptop & Accessory","Household"],
            "上月投诉情况": [0,1,0],
            "无需转换列":["A","B","C"],#没有映射的列
        }        
        test_df=pd.DataFrame(test_data)

        print("原始数据")
        print(test_df[["流失标志","城市等级","性别","年龄分组"]])

        #调用转换函数
        result_df=convert_all_values(test_df)
        print("\n转换后数据")
        print(result_df[["流失标志","城市等级","性别","年龄分组"]])

        #验证无需转换列
        print("无需转换列原始数据")
        print(test_df[["无需转换列"]])
        print("\n转换后结果")
        print(result_df[["无需转换列"]])

        #测试未知值处理
        test_data2=pd.DataFrame({"性别":["Unknown","Male"]})
        result_df2=convert_all_values(test_data2)
        print("\n转换后结果")
        print(result_df2[["性别"]])

    except AssertionError as e:
        print(f"\n-------断言失败:{e}")
    except Exception as e:
        print(f"\n-------发生意外错误:{type(e).__name__}:{e}")


def load_project_dataset(file_path: str,sample_size:Optional[int]=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        #验证文件类型
        if not file_path.endswith((".xlsx",".xls")):
            print("警告：文件格式不符------")

        if sample_size:
            print(f"加载前{sample_size}行数据" )
            ecomm_df=pd.read_excel(file_path, sheet_name="E Comm", nrows=sample_size)
            data_dict_df=pd.read_excel(file_path, sheet_name="Data Dict")
        else:
            print("完整加载")
            data_dict_df=pd.read_excel(file_path, sheet_name="Data Dict")
            ecomm_df=pd.read_excel(file_path, sheet_name="E Comm")

        print(f"成功加载文件:{file_path}")
        print(f"数据字典行数:{len(data_dict_df)}")
        print(f"实际数据行数:{len(ecomm_df)}")
        return data_dict_df,ecomm_df

    except FileNotFoundError:
        print(f"错误：找不到文件'{file_path}'")
        print("检查当前目录下是否存在指定文件")
        raise
    except PermissionError:
        error_msg=f"错误：没有权限访问文件'{file_path}'"
        print(error_msg)
        print("请确保文件没有被其他程序打开，并且你有读写权限")
    except ValueError as e:
        if "Worksheet" in str(e):
            error_msg=f"错误：Excel文件中缺少对应的表格"
            try:
                xls=pd.ExcelFile(file_path)
                print(f"文件中实际存在的工作表列表: {xls.sheet_names}")
                print(error_msg)
            except:
                pass
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        raise

def test_load_project_dataset():
    try:
        load_project_dataset("Project Dataset.xlsx",15)
    except AssertionError as e:
        print(f"\n断言失败")
    except Exception as e:
        print(f"\n发生意外错误:{type(e).__name__}:{e}")

def transform_project_dataset_to_ecommerce(
        project_file_path:str,
        output_file_path:str
)-> pd.DataFrame:
    """协调整个转换过程"""
    print("-------开始数据转换-------")
    print(f"输入文件路径: {project_file_path}")
    print(f"输出文件路径: {output_file_path}")

    dict_df,data_df=load_project_dataset(project_file_path)
    

    #重命名列，转换列值
    transform_dt=data_df.rename(columns=COLUMN_MAPPING).pipe(convert_all_values)
    # rename_dt=data_df.rename(columns=COLUMN_MAPPING)
    # transform_dt=convert_all_values(rename_dt)
    print("\n重命名完毕")

    #补全空列，调整顺序
    final_df=ensure_column_order(transform_dt)
    print("顺序调整完毕")

    #保存
    print("\n开始保存")
    with pd.ExcelWriter(output_file_path, engine="openpyxl") as writer:
        final_df.to_excel(writer, sheet_name="客户行为数据", index=False,header=True)

    print("\n-------数据转换完成-------")
    return final_df

def run_example()-> Optional[pd.DataFrame]:
    """完整使用示例"""
    print("开始处理")

    #输入，输出文件路径
    input_file="Project Dataset.xlsx"
    out_file="电商行为数据.xlsx"
    
    try:
        result_df=transform_project_dataset_to_ecommerce(input_file,out_file)
        return result_df

    except FileNotFoundError:
        print(f"\n错误：·找不到指定文件'{input_file}'")
        print("请检查当前目录下是否存在指定文件")
        return None
    except Exception as e:
        print(f"\n转换过程中发生错误:{type(e).__name__}:{e}")
        return None
    
def main():
   """主函数"""
   run_example()


if __name__ == "__main__":
    main()
    
