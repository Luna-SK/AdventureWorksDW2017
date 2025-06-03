import pandas as pd
import sqlalchemy
import os

# 假设DSN名称为 adventureworks
DB_CONFIG = {
    'dsn': 'adventureworks',
    'user': 'sa',
    'password': 'Alaska2017',
    'database': 'AdventureWorksDW2017',
}

def get_engine():
    """
    使用DSN方式创建SQLAlchemy数据库引擎
    """
    conn_str = (
        f"mssql+pyodbc://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['dsn']}?charset=utf8mb4"
    )
    return sqlalchemy.create_engine(conn_str)

def read_sql(query, engine=None):
    """
    用于读取SQL查询结果为DataFrame
    """
    if engine is None:
        engine = get_engine()
    return pd.read_sql(query, engine)
