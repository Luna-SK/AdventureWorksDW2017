import pandas as pd
import sqlalchemy
import os

# 数据库连接信息
DB_CONFIG = {
    'host': 'localhost',
    'port': 8433,
    'user': 'sa',
    'password': 'Alaska2017',
    'database': 'AdventureWorksDW2017',
}

def get_engine():
    """
    创建SQLAlchemy数据库引擎
    """
    conn_str = (
        f"mssql+pyodbc://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        f"?driver=ODBC+Driver+17+for+SQL+Server"
    )
    return sqlalchemy.create_engine(conn_str)

def read_sql(query, engine=None):
    """
    用于读取SQL查询结果为DataFrame
    """
    if engine is None:
        engine = get_engine()
    return pd.read_sql(query, engine)
