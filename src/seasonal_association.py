import pandas as pd
from data_utils import get_engine, read_sql
from mlxtend.frequent_patterns import apriori, association_rules

# 1. 数据加载
# 2. 按季节分组
# 3. 关联规则挖掘
# 4. 结果输出

def load_data():
    engine = get_engine()
    # query = "SELECT ... FROM ..."
    # df = read_sql(query, engine)
    # return df
    pass

def preprocess_season(df):
    # 按季节分组
    pass

def mine_association_rules(df_season):
    # 关联规则挖掘
    pass

def main():
    # 主流程
    pass

if __name__ == "__main__":
    main()
