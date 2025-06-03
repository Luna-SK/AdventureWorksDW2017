import pandas as pd
from data_utils import get_engine, read_sql
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import base64
from io import BytesIO
import os
import plotly.express as px
import plotly.graph_objects as go

# 1. 数据加载
# 2. RFM特征计算
# 3. 聚类分析
# 4. 结果输出

def load_data(engine):
    query = '''
    SELECT CustomerKey, OrderDate, SalesAmount, ProductKey
    FROM FactInternetSales
    WHERE CurrencyKey = 6
    '''
    df = read_sql(query, engine)
    return df

def calculate_rfm(df):
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    snapshot_date = df['OrderDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerKey').agg({
        'OrderDate': lambda x: (snapshot_date - x.max()).days,
        'SalesAmount': ['count', 'sum']
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    rfm = rfm.reset_index()
    return rfm

def calculate_rfm_scores(rfm_df):
    """计算RFM得分（1-5分）"""
    r_labels = range(5, 0, -1)
    r_quartiles = pd.qcut(rfm_df['Recency'], q=5, labels=r_labels)
    f_labels = range(1, 6)
    f_quartiles = pd.qcut(rfm_df['Frequency'], q=5, labels=f_labels)
    m_labels = range(1, 6)
    m_quartiles = pd.qcut(rfm_df['Monetary'], q=5, labels=m_labels)
    
    rfm_df['R_Score'] = r_quartiles
    rfm_df['F_Score'] = f_quartiles
    rfm_df['M_Score'] = m_quartiles
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
    return rfm_df

def cluster_rfm(rfm_df, n_clusters=4):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    return rfm_df

def plot_rfm_clusters(rfm_clustered):
    out_dir = './reports/澳元客户RFM/'
    palette = sns.color_palette('Set2', n_colors=rfm_clustered['Cluster'].nunique())
    # Recency vs Monetary
    plt.figure(figsize=(9,7))
    ax = sns.scatterplot(data=rfm_clustered, x='Recency', y='Monetary', hue='Cluster', palette=palette, s=70, alpha=0.8, edgecolor='k', linewidth=0.5)
    centers = rfm_clustered.groupby('Cluster')[['Recency','Monetary']].mean().reset_index()
    for i, row in centers.iterrows():
        plt.scatter(row['Recency'], row['Monetary'], c=[palette[int(row['Cluster'])]], marker='X', s=180, edgecolor='black', linewidths=2, zorder=10)
        plt.text(row['Recency'], row['Monetary'], f"{int(row['Cluster'])}", fontsize=14, fontweight='bold', color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    plt.title('RFM Cluster Analysis (Recency vs Monetary)', fontsize=16, fontweight='bold')
    plt.xlabel('Recency (days)', fontsize=13)
    plt.ylabel('Monetary Value (AUD)', fontsize=13)
    plt.legend(title='Cluster', fontsize=11, title_fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir + 'rfm_cluster_scatter.png', dpi=200)
    plt.close()
    # Frequency vs Monetary
    plt.figure(figsize=(9,7))
    ax = sns.scatterplot(data=rfm_clustered, x='Frequency', y='Monetary', hue='Cluster', palette=palette, s=70, alpha=0.8, edgecolor='k', linewidth=0.5)
    centers2 = rfm_clustered.groupby('Cluster')[['Frequency','Monetary']].mean().reset_index()
    for i, row in centers2.iterrows():
        plt.scatter(row['Frequency'], row['Monetary'], c=[palette[int(row['Cluster'])]], marker='X', s=180, edgecolor='black', linewidths=2, zorder=10)
        plt.text(row['Frequency'], row['Monetary'], f"{int(row['Cluster'])}", fontsize=14, fontweight='bold', color='black', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
    plt.title('RFM Cluster Analysis (Frequency vs Monetary)', fontsize=16, fontweight='bold')
    plt.xlabel('Frequency (purchase count)', fontsize=13)
    plt.ylabel('Monetary Value (AUD)', fontsize=13)
    plt.legend(title='Cluster', fontsize=11, title_fontsize=12)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir + 'rfm_cluster_freq_monetary.png', dpi=200)
    plt.close()

def cluster_naming_and_advice(rfm_clustered):
    # 统计各群体RFM均值
    summary = rfm_clustered.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().reset_index()
    # 按RFM均值特征命名
    names = {}
    for _, row in summary.iterrows():
        c = int(row['Cluster'])
        if row['Recency'] < summary['Recency'].median() and row['Frequency'] > summary['Frequency'].median() and row['Monetary'] > summary['Monetary'].median():
            names[c] = '重要价值客户'
        elif row['Recency'] < summary['Recency'].median() and row['Frequency'] > summary['Frequency'].median():
            names[c] = '高潜力客户'
        elif row['Recency'] > summary['Recency'].median() and row['Monetary'] < summary['Monetary'].median():
            names[c] = '流失预警客户'
        else:
            names[c] = '一般客户'
    # 输出建议
    advice = {
        '重要价值客户': '重点维护，定制高端服务和专属优惠，提升忠诚度。',
        '高潜力客户': '加强营销互动，激励复购，转化为高价值客户。',
        '流失预警客户': '及时关怀，发送唤回优惠或回访，降低流失风险。',
        '一般客户': '常规营销触达，关注其成长潜力。'
    }
    # 保存命名和建议
    out_dir = './reports/澳元客户RFM/'
    with open(out_dir + 'rfm_cluster_naming_advice.txt', 'w', encoding='utf-8') as f:
        for c, name in names.items():
            f.write(f"群体{c}：{name}\n")
            f.write(f"RFM均值：Recency={summary.loc[summary['Cluster']==c,'Recency'].values[0]:.1f}，Frequency={summary.loc[summary['Cluster']==c,'Frequency'].values[0]:.1f}，Monetary={summary.loc[summary['Cluster']==c,'Monetary'].values[0]:.1f}\n")
            f.write(f"服务建议：{advice[name]}\n\n")
    print(f"群体命名与服务建议已保存到 {out_dir}rfm_cluster_naming_advice.txt")

def plot_rfm_radar(rfm_clustered):
    """绘制RFM雷达图"""
    out_dir = './reports/澳元客户RFM/'
    
    # 计算每个群体的RFM均值
    cluster_means = rfm_clustered.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    
    # 标准化数据用于雷达图
    scaler = StandardScaler()
    cluster_means_scaled = pd.DataFrame(
        scaler.fit_transform(cluster_means),
        columns=cluster_means.columns,
        index=cluster_means.index
    )
    
    # 设置雷达图的角度
    categories = ['Recency', 'Frequency', 'Monetary']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合图形
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 绘制每个群体的雷达图
    for cluster in cluster_means_scaled.index:
        values = cluster_means_scaled.loc[cluster].values.tolist()
        values += values[:1]  # 闭合图形
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    # 设置雷达图的刻度和标签
    plt.xticks(angles[:-1], ['Recency', 'Frequency', 'Monetary'])
    ax.set_ylim(-2, 2)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('RFM Cluster Characteristics', size=15, y=1.1)
    
    plt.tight_layout()
    plt.savefig(out_dir + 'rfm_radar.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_customer_value_distribution(rfm_clustered):
    """绘制客户价值分布图"""
    out_dir = './reports/澳元客户RFM/'
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    
    # 价值分布直方图
    sns.histplot(data=rfm_clustered, x='Monetary', bins=50, ax=ax1)
    ax1.set_title('Customer Value Distribution')
    ax1.set_xlabel('Monetary Value (AUD)')
    ax1.set_ylabel('Number of Customers')
    
    # 群体价值箱线图
    sns.boxplot(data=rfm_clustered, x='Cluster', y='Monetary', ax=ax2)
    ax2.set_title('Customer Value Distribution by Cluster')
    ax2.set_xlabel('Customer Cluster')
    ax2.set_ylabel('Monetary Value (AUD)')
    
    plt.tight_layout()
    plt.savefig(out_dir + 'customer_value_distribution.png', dpi=200, bbox_inches='tight')
    plt.close()

def get_plot_base64(fig):
    """将matplotlib图形转换为base64编码"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def generate_html_report(rfm_clustered):
    """生成静态HTML分析报告"""
    out_dir = './reports/澳元客户RFM/'
    
    # 计算基本统计信息
    cluster_stats = rfm_clustered.groupby('Cluster').agg({
        'Recency': ['mean', 'std', 'min', 'max'],
        'Frequency': ['mean', 'std', 'min', 'max'],
        'Monetary': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # 生成可视化并转换为base64
    # RFM散点图
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    palette = sns.color_palette('Set2', n_colors=rfm_clustered['Cluster'].nunique())
    
    # Recency vs Monetary
    sns.scatterplot(data=rfm_clustered, x='Recency', y='Monetary', hue='Cluster', 
                   palette=palette, s=70, alpha=0.8, ax=ax1)
    ax1.set_title('RFM Cluster (Recency vs Monetary)')
    ax1.set_xlabel('Recency (days)')
    ax1.set_ylabel('Monetary Value')
    
    # Frequency vs Monetary
    sns.scatterplot(data=rfm_clustered, x='Frequency', y='Monetary', hue='Cluster', 
                   palette=palette, s=70, alpha=0.8, ax=ax2)
    ax2.set_title('RFM Cluster (Frequency vs Monetary)')
    ax2.set_xlabel('Frequency (count)')
    ax2.set_ylabel('Monetary Value')
    
    plt.tight_layout()
    scatter_plot = get_plot_base64(fig1)
    
    # 生成HTML报告（保持中文，因为这是HTML内容）
    html_content = f"""
    <html>
    <head>
        <title>澳元客户RFM分析报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
            .section {{ margin: 20px 0; padding: 20px; background: #fff; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            .section h2 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f8f9fa; color: #2c3e50; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            .plot-container {{ margin: 20px 0; text-align: center; }}
            .plot-container img {{ max-width: 100%; height: auto; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .timestamp {{ color: #666; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 style="color: #2c3e50; text-align: center;">澳元客户RFM分析报告</h1>
            <p class="timestamp">生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                <h2>1. 分析概述</h2>
                <p>总客户数：{len(rfm_clustered):,}</p>
                <p>分析维度：</p>
                <ul>
                    <li>Recency（最近购买时间）：距离最近一次购买的天数</li>
                    <li>Frequency（购买频率）：购买次数</li>
                    <li>Monetary（消费金额）：总消费金额</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>2. 群体统计</h2>
                {cluster_stats.to_html(classes='table table-striped')}
            </div>
            
            <div class="section">
                <h2>3. 可视化分析</h2>
                
                <div class="plot-container">
                    <h3>3.1 RFM散点图</h3>
                    <img src="data:image/png;base64,{scatter_plot}" alt="RFM散点图">
                </div>
                
                <div class="plot-container">
                    <h3>3.2 RFM雷达图</h3>
                    <img src="rfm_radar.png" alt="RFM雷达图">
                </div>
                
                <div class="plot-container">
                    <h3>3.3 客户价值分布</h3>
                    <img src="customer_value_distribution.png" alt="客户价值分布">
                </div>
            </div>
            
            <div class="section">
                <h2>4. 群体特征与服务建议</h2>
                <p>详细的服务建议请参考：<a href="rfm_cluster_naming_advice.txt">群体命名与服务建议</a></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(out_dir + 'rfm_analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

def analyze_product_preferences(df, clusters, engine):
    """分析各群体的产品偏好"""
    # 关联产品维度表获取产品类别信息
    product_query = """
    SELECT 
        p.ProductKey,
        p.EnglishProductName,
        pc.EnglishProductCategoryName,
        psc.EnglishProductSubcategoryName
    FROM DimProduct p
    JOIN DimProductSubcategory psc ON p.ProductSubcategoryKey = psc.ProductSubcategoryKey
    JOIN DimProductCategory pc ON psc.ProductCategoryKey = pc.ProductCategoryKey
    """
    products = pd.read_sql(product_query, engine)
    
    # 合并销售数据和产品信息
    df_with_products = df.merge(products, on='ProductKey')
    
    # 添加群体信息
    df_with_products = df_with_products.merge(clusters[['CustomerKey', 'Cluster']], on='CustomerKey')
    
    # 按群体和产品类别统计
    preferences = df_with_products.groupby(['Cluster', 'EnglishProductCategoryName']).agg({
        'SalesAmount': ['sum', 'mean', 'count'],
        'ProductKey': 'nunique'
    }).reset_index()
    
    # 生成产品偏好分析报告
    preferences.to_csv('reports/澳元客户RFM/product_preferences.csv', index=False)
    
    # 生成产品偏好可视化
    plt.figure(figsize=(12, 8))
    sns.barplot(data=preferences, x='EnglishProductCategoryName', y=('SalesAmount', 'sum'), hue='Cluster')
    plt.xticks(rotation=45)
    plt.title('Product Category Preferences by Cluster')
    plt.xlabel('Product Category')
    plt.ylabel('Total Sales Amount')
    plt.tight_layout()
    plt.savefig('reports/澳元客户RFM/product_preferences.png')
    plt.close()
    
    return preferences

def analyze_seasonal_patterns(df, clusters):
    """分析季节性购买特征"""
    # 添加月份信息
    df['Month'] = pd.to_datetime(df['OrderDate']).dt.month
    
    # 按群体和月份统计
    seasonal = df.merge(clusters[['CustomerKey', 'Cluster']], on='CustomerKey').groupby(['Cluster', 'Month']).agg({
        'SalesAmount': ['sum', 'mean', 'count']
    }).reset_index()
    
    # 生成季节性分析报告
    seasonal.to_csv('reports/澳元客户RFM/seasonal_patterns.csv', index=False)
    
    # 生成季节性趋势图
    plt.figure(figsize=(12, 8))
    for cluster in seasonal['Cluster'].unique():
        cluster_data = seasonal[seasonal['Cluster'] == cluster]
        plt.plot(cluster_data['Month'], cluster_data[('SalesAmount', 'sum')], 
                label=f'Cluster {cluster}', marker='o')
    
    plt.title('Monthly Purchase Trends by Cluster')
    plt.xlabel('Month')
    plt.ylabel('Sales Amount')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/澳元客户RFM/seasonal_purchase_trend.png')
    plt.close()
    
    return seasonal

def analyze_customer_lifecycle(df, clusters):
    """分析客户生命周期"""
    # 计算每个客户的生命周期指标
    lifecycle = df.groupby('CustomerKey').agg({
        'OrderDate': ['min', 'max', 'count'],
        'SalesAmount': ['sum', 'mean']
    }).reset_index()
    
    lifecycle.columns = ['CustomerKey', 'FirstPurchase', 'LastPurchase', 
                        'PurchaseCount', 'TotalAmount', 'AverageAmount']
    
    # 计算生命周期长度（天）
    lifecycle['LifecycleDays'] = (lifecycle['LastPurchase'] - lifecycle['FirstPurchase']).dt.days
    
    # 添加群体信息
    lifecycle = lifecycle.merge(clusters[['CustomerKey', 'Cluster']], on='CustomerKey')
    
    # 生成生命周期分析报告
    lifecycle.to_csv('reports/澳元客户RFM/customer_lifecycle.csv', index=False)
    
    # 生成生命周期分析图
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=lifecycle, x='Cluster', y='LifecycleDays')
    plt.title('Customer Lifecycle Distribution by Cluster')
    plt.xlabel('Customer Cluster')
    plt.ylabel('Lifecycle Days')
    plt.savefig('reports/澳元客户RFM/customer_lifecycle.png')
    plt.close()
    
    return lifecycle

def main():
    # 创建报告目录
    os.makedirs('reports/澳元客户RFM', exist_ok=True)
    
    # 创建数据库连接
    engine = get_engine()
    
    # 加载数据
    df = load_data(engine)
    
    # 计算RFM特征
    rfm = calculate_rfm(df)
    
    # 聚类分析
    rfm_clustered = cluster_rfm(rfm, n_clusters=4)

    print("各群体客户数：")
    print(rfm_clustered['Cluster'].value_counts().sort_index())
    print("各群体RFM均值：")
    print(rfm_clustered.groupby('Cluster')[['Recency','Frequency','Monetary']].mean())

    # 生成所有可视化
    plot_rfm_clusters(rfm_clustered)
    plot_rfm_radar(rfm_clustered)
    plot_customer_value_distribution(rfm_clustered)
    cluster_naming_and_advice(rfm_clustered)
    generate_html_report(rfm_clustered)
    
    # 分析产品偏好
    preferences = analyze_product_preferences(df, rfm_clustered, engine)
    
    # 分析季节性特征
    seasonal = analyze_seasonal_patterns(df, rfm_clustered)
    
    # 分析客户生命周期
    lifecycle = analyze_customer_lifecycle(df, rfm_clustered)
    
    print("分析完成，报告已生成在 reports/澳元客户RFM/ 目录下")

if __name__ == "__main__":
    main()
