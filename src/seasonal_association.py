import pandas as pd
from data_utils import get_engine, read_sql
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import os
import networkx as nx

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

REPORT_DIR = './reports/季节性产品关联规则/'
os.makedirs(REPORT_DIR, exist_ok=True)

# 季节映射函数
SEASON_MAP = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
}

def load_data():
    engine = get_engine()
    query = '''
    SELECT 
        f.SalesOrderNumber,
        f.OrderDate,
        f.OrderQuantity,
        p.EnglishProductName,
        d.MonthNumberOfYear
    FROM FactInternetSales f
    JOIN DimProduct p ON f.ProductKey = p.ProductKey
    JOIN DimDate d ON f.OrderDateKey = d.DateKey
    '''
    df = read_sql(query, engine)
    return df

def add_season(df):
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['Season'] = df['OrderDate'].dt.month.map(SEASON_MAP)
    return df

def build_baskets(df, season):
    # 过滤指定季节
    season_df = df[df['Season'] == season]
    # 构建订单-产品二元表
    basket = season_df.groupby(['SalesOrderNumber', 'EnglishProductName'])['OrderQuantity'].sum().unstack().fillna(0)
    basket = (basket > 0).astype(bool)
    return basket

def mine_rules(basket, min_support=0.02, min_confidence=0.3):
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    rules = rules.sort_values(by=['lift', 'confidence'], ascending=False)
    return rules

def plot_top_rules(rules, season, topn=10):
    if rules.empty:
        return
    plt.figure(figsize=(8,5))
    top_rules = rules.head(topn)
    labels = [f"{', '.join(list(x))}→{', '.join(list(y))}" for x, y in zip(top_rules['antecedents'], top_rules['consequents'])]
    sns.barplot(x=top_rules['lift'], y=labels, color='#4C72B0')
    plt.xlabel('Lift')
    plt.title(f'Top {topn} Association Rules ({season})')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f'{season}_top_rules.png'), dpi=120)
    plt.close()

def plot_freq_itemsets_heatmap(basket, season):
    # 只画前20个商品的热力图
    if basket.shape[1] < 2:
        return
    top_products = basket.sum().sort_values(ascending=False).head(20).index
    corr = basket[top_products].corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=False, cmap='Blues', square=True)
    plt.title(f'Product Co-occurrence Heatmap ({season})')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f'{season}_heatmap.png'), dpi=120)
    plt.close()

def save_top_rules_text(rules, season, topn=3):
    if rules.empty:
        return
    lines = [f"{season}季节Top{topn}关联规则业务解读：\n"]
    for i, row in rules.head(topn).iterrows():
        ant = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        lines.append(f"规则{i+1}: 如果购买[{ant}]，则很可能购买[{cons}]。\n"
                     f"  - 支持度: {row['support']:.3f}，置信度: {row['confidence']:.3f}，提升度: {row['lift']:.3f}\n")
        # 业务建议举例
        lines.append(f"  - 建议：可将[{ant}]与[{cons}]进行捆绑促销或货架邻近陈列，提升联动销售。\n")
    with open(os.path.join(REPORT_DIR, f'{season}_top_rules.txt'), 'w', encoding='utf-8') as f:
        f.writelines(lines)

def plot_rules_network(rules, season, min_conf=0.5):
    # 只画高置信度规则
    rules_net = rules[rules['confidence'] >= min_conf].head(20)
    if rules_net.empty:
        return
    G = nx.DiGraph()
    for _, row in rules_net.iterrows():
        ant = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        G.add_edge(ant, cons, weight=row['lift'])
    plt.figure(figsize=(8,6))
    pos = nx.spring_layout(G, k=0.5, seed=42)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw(G, pos, with_labels=True, node_color='#4C72B0', edge_color=weights, edge_cmap=plt.cm.Blues,
            node_size=1200, font_size=10, arrowsize=20, width=2)
    plt.title(f'High Confidence Rules Network ({season})')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, f'{season}_rules_network.png'), dpi=120)
    plt.close()

def main():
    print("加载数据...")
    df = load_data()
    print(f"原始订单数: {df['SalesOrderNumber'].nunique()} 条记录: {len(df)}")

    print("添加季节信息...")
    df = add_season(df)
    print(df[['OrderDate', 'Season']].head())

    for season in ['Spring', 'Summer', 'Autumn', 'Winter']:
        print(f"\n=== {season} 购物篮构建与关联规则挖掘 ===")
        basket = build_baskets(df, season)
        print(f"订单数: {basket.shape[0]}, 商品数: {basket.shape[1]}")
        rules = mine_rules(basket)
        rules.to_csv(os.path.join(REPORT_DIR, f'{season}_rules.csv'), index=False)
        print(f"{season} 关联规则已保存: {REPORT_DIR}{season}_rules.csv")
        # 可视化
        plot_top_rules(rules, season)
        plot_freq_itemsets_heatmap(basket, season)
        plot_rules_network(rules, season, min_conf=0.5)
        save_top_rules_text(rules, season, topn=3)
        print(f"{season} 可视化图和业务解读已保存到 {REPORT_DIR}")

if __name__ == "__main__":
    main()
