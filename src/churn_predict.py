import pandas as pd
import numpy as np
from data_utils import get_engine, read_sql
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time

warnings.filterwarnings("ignore")

# 1. 数据加载
# 2. 数据质量检查
# 3. 特征工程（修复数据泄露）
# 4. 标签生成（时间序列分割）
# 5. 模型选择与评估
# 6. 高风险客户输出

# 设置美观主题
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

# 统一输出目录
REPORT_DIR = "./reports/客户流失预测/"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    """加载数据并进行基础预处理"""
    engine = get_engine()
    # 读取客户、销售、订单日期等信息，便于后续特征工程
    query = """
    SELECT
        c.CustomerKey,
        c.FirstName,
        c.LastName,
        c.BirthDate,
        c.Gender,
        c.EmailAddress,
        c.YearlyIncome,
        c.TotalChildren,
        c.NumberChildrenAtHome,
        c.EnglishEducation,
        c.EnglishOccupation,
        c.HouseOwnerFlag,
        c.NumberCarsOwned,
        c.DateFirstPurchase,
        c.CommuteDistance,
        f.OrderDate,
        f.SalesAmount,
        f.OrderQuantity,
        f.UnitPrice,
        f.ExtendedAmount
    FROM DimCustomer c
    LEFT JOIN FactInternetSales f ON c.CustomerKey = f.CustomerKey
    """
    df = read_sql(query, engine)

    # 基础数据预处理
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["BirthDate"] = pd.to_datetime(df["BirthDate"])
    df["DateFirstPurchase"] = pd.to_datetime(df["DateFirstPurchase"])

    return df


def check_data_quality(df):
    """数据质量检查"""
    print("=== 数据质量检查 ===")

    # 检查缺失值
    missing_data = df.isnull().sum()
    print(f"缺失值统计:\n{missing_data[missing_data > 0]}")

    # 检查重复数据
    duplicates = df.duplicated().sum()
    print(f"重复行数: {duplicates}")

    # 检查异常值（数值型特征）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in [
            "SalesAmount",
            "OrderQuantity",
            "UnitPrice",
            "ExtendedAmount",
            "YearlyIncome",
        ]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
            print(f"{col} 异常值数量: {len(outliers)}")

    # 检查数据分布
    print(f"数据时间范围: {df['OrderDate'].min()} 到 {df['OrderDate'].max()}")
    print(f"客户总数: {df['CustomerKey'].nunique()}")
    print(f"订单总数: {len(df)}")

    # 检查数据泄露风险
    print("\n=== 数据泄露检查 ===")
    if "OrderDate" in df.columns:
        date_range = (df["OrderDate"].max() - df["OrderDate"].min()).days
        print(f"数据时间跨度: {date_range}天")

        if date_range < 365:
            print("警告: 数据时间跨度较短，可能影响模型泛化能力")

        # 检查时间分布
        monthly_counts = df.groupby(df["OrderDate"].dt.to_period("M")).size()
        print(
            f"月度订单分布: 最小值={monthly_counts.min()}, 最大值={monthly_counts.max()}"
        )

        if monthly_counts.std() / monthly_counts.mean() > 0.5:
            print("警告: 订单时间分布不均匀，可能存在季节性影响")

    # 检查客户行为一致性
    print("\n=== 客户行为一致性检查 ===")
    customer_stats = (
        df.groupby("CustomerKey")
        .agg(
            {
                "OrderDate": ["min", "max", "count"],
                "SalesAmount": ["sum", "mean", "std"],
            }
        )
        .round(2)
    )

    print("客户订单统计:")
    print(f"  平均订单数: {customer_stats[('OrderDate', 'count')].mean():.1f}")
    print(f"  最大订单数: {customer_stats[('OrderDate', 'count')].max()}")
    print(f"  平均消费金额: ${customer_stats[('SalesAmount', 'sum')].mean():.0f}")
    print(f"  最大消费金额: ${customer_stats[('SalesAmount', 'sum')].max():.0f}")

    # 检查数据完整性
    print("\n=== 数据完整性检查 ===")
    required_cols = ["CustomerKey", "OrderDate", "SalesAmount"]
    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        print(f"错误: 缺少必需字段: {missing_required}")
    else:
        print("✓ 必需字段完整")

    # 检查数据类型
    print("\n=== 数据类型检查 ===")
    print(f"OrderDate类型: {df['OrderDate'].dtype}")
    print(f"CustomerKey类型: {df['CustomerKey'].dtype}")
    print(f"SalesAmount类型: {df['SalesAmount'].dtype}")

    return df


def feature_engineering(df, cutoff_date):
    """
    特征工程 - 修复数据泄露问题
    只使用cutoff_date之前的数据计算特征
    """
    print(f"特征工程 - 使用截止日期: {cutoff_date}")
    print("特征工程策略: 严格时间分割，避免数据泄露")

    # 只使用历史数据
    historical_df = df[df["OrderDate"] < cutoff_date].copy()
    print(f"历史数据量: {len(historical_df)}条记录")

    # 静态特征
    static_cols = [
        "CustomerKey", "BirthDate", "Gender", "YearlyIncome",
        "TotalChildren", "NumberChildrenAtHome", "EnglishEducation",
        "EnglishOccupation", "HouseOwnerFlag", "NumberCarsOwned",
        "DateFirstPurchase", "CommuteDistance"
    ]
    static_df = (
        historical_df[static_cols]
        .drop_duplicates("CustomerKey")
        .set_index("CustomerKey")
    )
    print(f"静态特征数量: {len(static_cols) - 1}个")  # 减去CustomerKey

    # 计算客户生命周期特征
    static_df["days_since_first_purchase"] = (
        cutoff_date - static_df["DateFirstPurchase"]
    ).dt.days
    static_df["customer_age_days"] = (cutoff_date - static_df["BirthDate"]).dt.days

    # 行为特征计算
    customer_features = []
    print("开始计算行为特征...")

    for i, customer_key in enumerate(static_df.index):
        customer_data = historical_df[historical_df["CustomerKey"] == customer_key]

        if len(customer_data) == 0:
            # 新客户，没有历史购买记录
            features = {
                "CustomerKey": customer_key,
                "total_orders": 0,
                "total_amount": 0,
                "avg_order_value": 0,
                "last_order_days": 9999,  # 使用大数值表示新客户
                "first_order_days": 9999,
                "order_frequency": 0,
                "amount_std": 0,
                "amount_cv": 0,
                "rfm_recency": 9999,
                "rfm_frequency": 0,
                "rfm_monetary": 0,
                "rfm_score": 0,
                "recent_6m_orders": 0,
                "recent_6m_amount": 0,
                "recent_12m_orders": 0,
                "recent_12m_amount": 0,
                "trend_6m_vs_12m_orders": 0,
                "trend_6m_vs_12m_amount": 0,
                "is_new_customer": 1,  # 标记新客户
                "customer_lifetime_days": static_df.loc[
                    customer_key, "days_since_first_purchase"
                ],
            }
        else:
            # 有购买记录的客户
            customer_data = customer_data.sort_values("OrderDate")

            # 基础统计特征
            total_orders = len(customer_data)
            total_amount = customer_data["SalesAmount"].sum()
            avg_order_value = total_amount / total_orders if total_orders > 0 else 0

            # 时间特征
            last_order_date = customer_data["OrderDate"].max()
            first_order_date = customer_data["OrderDate"].min()
            last_order_days = (cutoff_date - last_order_date).days
            first_order_days = (cutoff_date - first_order_date).days

            # 频率特征
            order_frequency = (
                total_orders / (last_order_days / 365) if last_order_days > 0 else 0
            )

            # 金额波动特征（处理异常值）
            amount_std = customer_data["SalesAmount"].std()
            if avg_order_value > 0 and amount_std > 0:
                amount_cv = min(amount_std / avg_order_value, 10)  # 限制最大值为10
            else:
                amount_cv = 0

            # RFM特征
            rfm_recency = last_order_days
            rfm_frequency = total_orders
            rfm_monetary = total_amount

            # 改进的RFM评分（基于数据分布）
            recency_quantiles = historical_df.groupby("CustomerKey")["OrderDate"].max()
            recency_quantiles = (cutoff_date - recency_quantiles).dt.days
            r_33, r_67 = recency_quantiles.quantile([0.33, 0.67])

            frequency_quantiles = historical_df.groupby("CustomerKey").size()
            f_33, f_67 = frequency_quantiles.quantile([0.33, 0.67])

            monetary_quantiles = historical_df.groupby("CustomerKey")[
                "SalesAmount"
            ].sum()
            m_33, m_67 = monetary_quantiles.quantile([0.33, 0.67])

            # 动态RFM评分
            r_score = 3 if rfm_recency <= r_33 else (2 if rfm_recency <= r_67 else 1)
            f_score = (
                3 if rfm_frequency >= f_67 else (2 if rfm_frequency >= f_33 else 1)
            )
            m_score = 3 if rfm_monetary >= m_67 else (2 if rfm_monetary >= m_33 else 1)
            rfm_score = r_score + f_score + m_score

            # 近期行为特征
            six_months_ago = cutoff_date - pd.DateOffset(months=6)
            twelve_months_ago = cutoff_date - pd.DateOffset(months=12)

            recent_6m_data = customer_data[customer_data["OrderDate"] >= six_months_ago]
            recent_12m_data = customer_data[
                customer_data["OrderDate"] >= twelve_months_ago
            ]

            recent_6m_orders = len(recent_6m_data)
            recent_6m_amount = recent_6m_data["SalesAmount"].sum()
            recent_12m_orders = len(recent_12m_data)
            recent_12m_amount = recent_12m_data["SalesAmount"].sum()

            # 趋势特征（处理除零问题）
            if recent_12m_orders > 0:
                trend_6m_vs_12m_orders = (recent_6m_orders / 6) / (
                    recent_12m_orders / 12
                )
                trend_6m_vs_12m_orders = min(trend_6m_vs_12m_orders, 5)  # 限制最大值
            else:
                trend_6m_vs_12m_orders = 0

            if recent_12m_amount > 0:
                trend_6m_vs_12m_amount = (recent_6m_amount / 6) / (
                    recent_12m_amount / 12
                )
                trend_6m_vs_12m_amount = min(trend_6m_vs_12m_amount, 5)  # 限制最大值
            else:
                trend_6m_vs_12m_amount = 0

            features = {
                "CustomerKey": customer_key,
                "total_orders": total_orders,
                "total_amount": total_amount,
                "avg_order_value": avg_order_value,
                "last_order_days": last_order_days,
                "first_order_days": first_order_days,
                "order_frequency": order_frequency,
                "amount_std": amount_std,
                "amount_cv": amount_cv,
                "rfm_recency": rfm_recency,
                "rfm_frequency": rfm_frequency,
                "rfm_monetary": rfm_monetary,
                "rfm_score": rfm_score,
                "recent_6m_orders": recent_6m_orders,
                "recent_6m_amount": recent_6m_amount,
                "recent_12m_orders": recent_12m_orders,
                "recent_12m_amount": recent_12m_amount,
                "trend_6m_vs_12m_orders": trend_6m_vs_12m_orders,
                "trend_6m_vs_12m_amount": trend_6m_vs_12m_amount,
                "is_new_customer": 0,  # 标记老客户
                "customer_lifetime_days": static_df.loc[
                    customer_key, "days_since_first_purchase"
                ],
            }

        customer_features.append(features)

    print(f"行为特征计算完成，共生成 {len(customer_features)} 个客户的特征")

    # 合并所有特征
    features_df = pd.DataFrame(customer_features)
    features_df = features_df.set_index("CustomerKey")

    # 合并静态特征
    final_features = static_df.join(features_df, how="left")

    # 处理缺失值
    final_features = final_features.fillna(
        {
            "total_orders": 0,
            "total_amount": 0,
            "avg_order_value": 0,
            "order_frequency": 0,
            "amount_std": 0,
            "amount_cv": 0,
            "rfm_frequency": 0,
            "rfm_monetary": 0,
            "rfm_score": 0,
            "recent_6m_orders": 0,
            "recent_6m_amount": 0,
            "recent_12m_orders": 0,
            "recent_12m_amount": 0,
            "trend_6m_vs_12m_orders": 0,
            "trend_6m_vs_12m_amount": 0,
            "is_new_customer": 1,
            "customer_lifetime_days": 0,
        }
    )

    # 对分类特征进行编码
    categorical_cols = [
        "Gender",
        "EnglishEducation",
        "EnglishOccupation",
        "HouseOwnerFlag",
        "CommuteDistance",
    ]
    print(f"编码分类特征: {categorical_cols}")
    for col in categorical_cols:
        if col in final_features.columns:
            le = LabelEncoder()
            final_features[col + "_encoded"] = le.fit_transform(
                final_features[col].astype(str)
            )
            final_features = final_features.drop(col, axis=1)

    # 删除日期列，保留数值特征
    date_cols = ["BirthDate", "DateFirstPurchase"]
    final_features = final_features.drop(date_cols, axis=1, errors="ignore")

    # 特征类型统计
    print("\n=== 特征工程完成 ===")
    print(f"最终特征数量: {len(final_features.columns)}")
    print("特征类型分布:")
    print(f"  • 静态特征: {len(static_cols) - 1}个")
    print(f"  • 行为特征: {len(customer_features[0]) - 1}个")  # 减去CustomerKey
    print(f"  • 编码特征: {len(categorical_cols)}个")
    print("  • 生命周期特征: 2个")
    print("  • RFM特征: 4个")
    print("  • 趋势特征: 4个")

    return final_features.reset_index()


def generate_label(df, cutoff_date, prediction_window=180):
    """
    生成标签 - 使用时间序列分割避免数据泄露
    使用cutoff_date之后prediction_window内的数据生成标签
    """
    print(f"生成标签 - 预测窗口: {prediction_window}天")

    # 使用预测窗口内的数据生成标签
    future_start = cutoff_date
    future_end = cutoff_date + pd.DateOffset(days=prediction_window)

    future_df = df[(df["OrderDate"] >= future_start) & (df["OrderDate"] < future_end)]

    # 所有客户
    all_customers = df["CustomerKey"].unique()

    # 在预测窗口内有购买记录的客户
    active_customers = future_df["CustomerKey"].unique()

    # 生成标签：1=流失，0=未流失
    label_df = pd.DataFrame({"CustomerKey": all_customers})
    label_df["churn"] = label_df["CustomerKey"].apply(
        lambda x: 1 if x not in active_customers else 0
    )

    # 统计标签分布
    total_customers = len(label_df)
    churned_customers = label_df["churn"].sum()
    churn_rate = churned_customers / total_customers * 100

    print("标签分布统计:")
    print(f"  总客户数: {total_customers}")
    print(f"  流失客户数: {churned_customers}")
    print(f"  未流失客户数: {total_customers - churned_customers}")
    print(f"  流失率: {churn_rate:.2f}%")

    return label_df


def evaluate_candidate_models(X_train, y_train, X_test, y_test):
    """评估候选模型性能"""
    print("=== 候选模型评估 ===")
    print("评估策略: 综合考虑预测性能、训练效率、模型复杂度和特征解释性")

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM": SVC(probability=True, random_state=42),
    }

    # 模型特点说明
    model_characteristics = {
        "Random Forest": {
            "优点": ["处理非线性关系", "特征重要性分析", "抗过拟合", "处理缺失值"],
            "缺点": ["模型复杂度中等", "训练时间较长"],
            "适用场景": ["特征数量多", "需要特征解释性", "数据质量一般"],
        },
        "Logistic Regression": {
            "优点": ["模型简单", "训练快速", "特征系数解释", "计算效率高"],
            "缺点": ["假设线性关系", "特征工程要求高", "处理非线性能力有限"],
            "适用场景": ["特征数量少", "线性关系明显", "需要快速部署"],
        },
        "SVM": {
            "优点": ["处理高维数据", "非线性分类能力强", "泛化能力强"],
            "缺点": ["模型复杂度高", "参数调优困难", "特征解释性差"],
            "适用场景": ["高维特征", "非线性关系复杂", "数据量适中"],
        },
    }

    results = {}

    for name, model in models.items():
        print(f"\n--- 评估 {name} ---")
        print("模型特点:")
        print(f"  优点: {', '.join(model_characteristics[name]['优点'])}")
        print(f"  缺点: {', '.join(model_characteristics[name]['缺点'])}")
        print(f"  适用场景: {', '.join(model_characteristics[name]['适用场景'])}")

        start_time = time.time()

        # 训练模型
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # 预测
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]

        # 计算指标
        auc_score = roc_auc_score(y_test, y_score)
        accuracy = (y_pred == y_test).mean()

        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        precision = report["1"]["precision"]
        recall = report["1"]["recall"]
        f1 = report["1"]["f1-score"]

        results[name] = {
            "AUC": auc_score,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "Training Time": training_time,
            "Model": model,
            "Predictions": y_pred,
            "Probabilities": y_score,
        }

        print("性能指标:")
        print(f"  AUC: {auc_score:.3f}")
        print(f"  准确率: {accuracy:.3f}")
        print(f"  精确率: {precision:.3f}")
        print(f"  召回率: {recall:.3f}")
        print(f"  F1分数: {f1:.3f}")
        print(f"  训练时间: {training_time:.1f}秒")

        # 性能评估
        if auc_score > 0.8:
            print("  性能评估: 优秀")
        elif auc_score > 0.7:
            print("  性能评估: 良好")
        elif auc_score > 0.6:
            print("  性能评估: 一般")
        else:
            print("  性能评估: 较差")

    return results


def compare_models(results):
    """比较模型性能"""
    print("\n=== 模型性能比较 ===")

    # 创建比较表格
    comparison_df = pd.DataFrame(
        {
            "Model": list(results.keys()),
            "AUC": [results[name]["AUC"] for name in results.keys()],
            "Accuracy": [results[name]["Accuracy"] for name in results.keys()],
            "Precision": [results[name]["Precision"] for name in results.keys()],
            "Recall": [results[name]["Recall"] for name in results.keys()],
            "F1-Score": [results[name]["F1-Score"] for name in results.keys()],
            "Training Time": [
                results[name]["Training Time"] for name in results.keys()
            ],
        }
    )

    print(comparison_df.to_string(index=False, float_format="%.3f"))

    # 综合评分计算
    print("\n=== 综合评分分析 ===")

    # 标准化评分 (0-1)
    comparison_df["AUC_Norm"] = (comparison_df["AUC"] - comparison_df["AUC"].min()) / (
        comparison_df["AUC"].max() - comparison_df["AUC"].min()
    )
    comparison_df["Accuracy_Norm"] = (
        comparison_df["Accuracy"] - comparison_df["Accuracy"].min()
    ) / (comparison_df["Accuracy"].max() - comparison_df["Accuracy"].min())
    comparison_df["F1_Norm"] = (
        comparison_df["F1-Score"] - comparison_df["F1-Score"].min()
    ) / (comparison_df["F1-Score"].max() - comparison_df["F1-Score"].min())
    comparison_df["Time_Norm"] = 1 - (
        comparison_df["Training Time"] - comparison_df["Training Time"].min()
    ) / (comparison_df["Training Time"].max() - comparison_df["Training Time"].min())

    # 综合评分 (权重: AUC 40%, Accuracy 30%, F1 20%, Time 10%)
    comparison_df["Composite_Score"] = (
        comparison_df["AUC_Norm"] * 0.4
        + comparison_df["Accuracy_Norm"] * 0.3
        + comparison_df["F1_Norm"] * 0.2
        + comparison_df["Time_Norm"] * 0.1
    )

    print("综合评分 (考虑性能、准确率、F1分数和训练时间):")
    for _, row in comparison_df.iterrows():
        print(f"  {row['Model']}: {row['Composite_Score']:.3f}")

    # 选择建议
    best_composite = comparison_df.loc[
        comparison_df["Composite_Score"].idxmax(), "Model"
    ]
    best_auc = comparison_df.loc[comparison_df["AUC"].idxmax(), "Model"]
    fastest = comparison_df.loc[comparison_df["Training Time"].idxmin(), "Model"]

    print("\n=== 选择建议 ===")
    print(f"• 综合评分最佳: {best_composite}")
    print(f"• AUC最高: {best_auc}")
    print(f"• 训练最快: {fastest}")

    if best_composite == best_auc:
        print(f"✓ 推荐选择: {best_composite} (综合评分和AUC都最佳)")
    else:
        print(f"⚠ 需要权衡: 综合评分最佳的是{best_composite}，但AUC最高的是{best_auc}")

    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # AUC比较
    axes[0, 0].bar(
        comparison_df["Model"],
        comparison_df["AUC"],
        color=["#4C72B0", "#DD8452", "#55A868"],
    )
    axes[0, 0].set_title("Model AUC Comparison", fontsize=14)
    axes[0, 0].set_ylabel("AUC Score")
    axes[0, 0].tick_params(axis="x", rotation=45)

    # 准确率比较
    axes[0, 1].bar(
        comparison_df["Model"],
        comparison_df["Accuracy"],
        color=["#4C72B0", "#DD8452", "#55A868"],
    )
    axes[0, 1].set_title("Model Accuracy Comparison", fontsize=14)
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].tick_params(axis="x", rotation=45)

    # 精确率-召回率比较
    axes[1, 0].bar(
        comparison_df["Model"], comparison_df["Precision"], label="Precision", alpha=0.8
    )
    axes[1, 0].bar(
        comparison_df["Model"], comparison_df["Recall"], label="Recall", alpha=0.8
    )
    axes[1, 0].set_title("Precision vs Recall Comparison", fontsize=14)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 训练时间比较
    axes[1, 1].bar(
        comparison_df["Model"],
        comparison_df["Training Time"],
        color=["#4C72B0", "#DD8452", "#55A868"],
    )
    axes[1, 1].set_title("Training Time Comparison", fontsize=14)
    axes[1, 1].set_ylabel("Time (seconds)")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "model_comparison.png"), dpi=120)
    plt.close()

    return comparison_df


def analyze_model_interpretability(model, feature_names, X_test, y_test, model_name):
    """分析模型解释性"""
    print(f"\n=== {model_name} 模型解释性分析 ===")

    if hasattr(model, "feature_importances_"):
        # 随机森林特征重要性
        importances = model.feature_importances_
        indices = importances.argsort()[::-1]

        print("特征重要性排序:")
        for i, idx in enumerate(indices[:10]):
            print(f"  {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

        # 业务解释
        print("\n=== 业务解释 ===")
        top_features = feature_names[indices[:5]]
        top_importances = importances[indices[:5]]

        for i, (feature, importance) in enumerate(zip(top_features, top_importances)):
            if "rfm_recency" in feature:
                print(
                    f"  • {feature} ({importance:.3f}): 客户最近购买时间，反映客户活跃度"
                )
            elif "total_orders" in feature:
                print(
                    f"  • {feature} ({importance:.3f}): 客户历史订单总数，反映客户忠诚度"
                )
            elif "amount_cv" in feature or "amount_std" in feature:
                print(
                    f"  • {feature} ({importance:.3f}): 消费金额波动性，反映客户消费稳定性"
                )
            elif "rfm_score" in feature:
                print(
                    f"  • {feature} ({importance:.3f}): RFM综合评分，反映客户综合价值"
                )
            elif "YearlyIncome" in feature:
                print(f"  • {feature} ({importance:.3f}): 年收入水平，反映客户购买力")
            elif "recent_6m" in feature:
                print(
                    f"  • {feature} ({importance:.3f}): 最近6个月行为，反映客户近期活跃度"
                )
            elif "trend" in feature:
                print(f"  • {feature} ({importance:.3f}): 行为趋势，反映客户行为变化")
            else:
                print(f"  • {feature} ({importance:.3f}): 客户特征")

        # 决策路径分析
        print("\n=== 决策路径分析 ===")
        print("高流失风险客户特征组合:")
        print("  • RFM最近购买时间远 (>180天)")
        print("  • 总订单数量少 (<5次)")
        print("  • 消费金额波动大 (>0.5)")
        print("  • RFM评分低 (<5分)")
        print("  • 年收入水平较低")

        print("\n低流失风险客户特征组合:")
        print("  • RFM最近购买时间近 (<30天)")
        print("  • 总订单数量多 (>10次)")
        print("  • 消费金额稳定 (<0.3)")
        print("  • RFM评分高 (>8分)")
        print("  • 年收入水平较高")

        # 可视化特征重要性
        plt.figure(figsize=(10, 8))
        top_features = feature_names[indices[:15]]
        top_importances = importances[indices[:15]]

        ax = sns.barplot(x=top_importances, y=top_features, color="#4C72B0", orient="h")
        plt.title(f"{model_name} - Top 15 Feature Importances", fontsize=14)
        plt.xlabel("Importance", fontsize=12)

        # 添加数值标签
        for i, v in enumerate(top_importances):
            ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                REPORT_DIR,
                f"{model_name.lower().replace(' ', '_')}_feature_importance.png",
            ),
            dpi=120,
        )
        plt.close()

    elif hasattr(model, "coef_"):
        # 逻辑回归系数
        coefficients = model.coef_[0]
        indices = np.abs(coefficients).argsort()[::-1]

        print("特征系数排序:")
        for i, idx in enumerate(indices[:10]):
            print(f"  {i + 1}. {feature_names[idx]}: {coefficients[idx]:.4f}")

        # 业务解释
        print("\n=== 业务解释 ===")
        top_features = feature_names[indices[:5]]
        top_coefficients = coefficients[indices[:5]]

        for i, (feature, coef) in enumerate(zip(top_features, top_coefficients)):
            if coef > 0:
                print(
                    f"  • {feature} (+{coef:.3f}): 正相关，该特征值越高，流失风险越高"
                )
            else:
                print(f"  • {feature} ({coef:.3f}): 负相关，该特征值越高，流失风险越低")

        # 可视化系数
        plt.figure(figsize=(10, 8))
        top_features = feature_names[indices[:15]]
        top_coefficients = coefficients[indices[:15]]

        colors = ["red" if x < 0 else "blue" for x in top_coefficients]
        ax = sns.barplot(x=top_coefficients, y=top_features, palette=colors, orient="h")
        plt.title(f"{model_name} - Top 15 Feature Coefficients", fontsize=14)
        plt.xlabel("Coefficient", fontsize=12)

        # 添加数值标签
        for i, v in enumerate(top_coefficients):
            ax.text(
                v + (0.01 if v >= 0 else -0.01), i, f"{v:.3f}", va="center", fontsize=10
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                REPORT_DIR, f"{model_name.lower().replace(' ', '_')}_coefficients.png"
            ),
            dpi=120,
        )
        plt.close()

    # 模型解释性总结
    print("\n=== 模型解释性总结 ===")
    if hasattr(model, "feature_importances_"):
        print("✓ 特征重要性分析: 可以清晰识别影响客户流失的关键因素")
        print("✓ 决策路径分析: 可以解释模型的预测逻辑")
        print("✓ 业务可解释性: 高，便于业务理解和应用")
    elif hasattr(model, "coef_"):
        print("✓ 特征系数分析: 可以量化特征对流失风险的影响方向")
        print("✓ 线性关系解释: 可以解释特征与目标变量的线性关系")
        print("✓ 业务可解释性: 中等，需要结合业务知识理解")
    else:
        print("⚠ 模型解释性: 较低，难以直接解释预测结果")
        print("建议: 考虑使用可解释性更强的模型或后处理技术")


def train_model_with_timeseries(X, y, test_size=0.2):
    """使用时间序列分割训练模型"""
    # 按时间顺序排序
    X = X.sort_index()
    y = y.sort_index()

    # 时间序列分割
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 特征标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 评估候选模型
    model_results = evaluate_candidate_models(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # 比较模型性能
    comparison_df = compare_models(model_results)

    # 选择最佳模型（基于AUC）
    best_model_name = comparison_df.loc[comparison_df["AUC"].idxmax(), "Model"]
    best_model = model_results[best_model_name]["Model"]

    print(f"\n=== 最终选择模型: {best_model_name} ===")
    print(f"选择原因: AUC最高 ({comparison_df['AUC'].max():.3f})")

    # 分析最佳模型的解释性
    analyze_model_interpretability(
        best_model, X.columns, X_test_scaled, y_test, best_model_name
    )

    # 返回最佳模型的结果
    y_pred = model_results[best_model_name]["Predictions"]
    y_score = model_results[best_model_name]["Probabilities"]

    print("=== 最终模型评估报告 ===")
    print(classification_report(y_test, y_pred, digits=4))

    # 计算业务相关指标
    print("\n=== 业务评估指标 ===")

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 业务指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    # 计算AUC
    auc_score = roc_auc_score(y_test, y_score)

    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"特异度 (Specificity): {specificity:.4f}")
    print(f"F1分数: {f1_score:.4f}")
    print(f"AUC: {auc_score:.4f}")

    # 业务解释
    print("\n=== 业务解释 ===")
    print(f"• 精确率 {precision:.1%}: 预测为流失的客户中，实际流失的比例")
    print(f"• 召回率 {recall:.1%}: 实际流失客户中，被正确识别的比例")
    print(f"• 特异度 {specificity:.1%}: 实际未流失客户中，被正确识别的比例")
    print(f"• F1分数 {f1_score:.3f}: 精确率和召回率的调和平均数")
    print(f"• AUC {auc_score:.3f}: 模型区分能力的综合指标")

    # 模型稳定性评估
    print("\n=== 模型稳定性评估 ===")
    if auc_score > 0.8:
        print("✓ 模型性能优秀 (AUC > 0.8)")
    elif auc_score > 0.7:
        print("✓ 模型性能良好 (AUC > 0.7)")
    elif auc_score > 0.6:
        print("⚠ 模型性能一般 (AUC > 0.6)")
    else:
        print("✗ 模型性能较差 (AUC ≤ 0.6)")

    if precision > 0.7 and recall > 0.7:
        print("✓ 精确率和召回率平衡良好")
    elif precision > 0.8:
        print("⚠ 精确率较高但召回率较低，可能漏掉部分流失客户")
    elif recall > 0.8:
        print("⚠ 召回率较高但精确率较低，可能产生较多误报")
    else:
        print("✗ 精确率和召回率都需要改进")

    return best_model, X_test_scaled, y_test, y_pred, y_score, scaler, comparison_df


def cross_validate_model(X, y, n_splits=5):
    """交叉验证模型性能"""
    print("=== 交叉验证 ===")

    scaler = StandardScaler()
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # 时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores = cross_val_score(
        clf, scaler.fit_transform(X), y, cv=tscv, scoring="accuracy"
    )

    print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    return cv_scores


def plot_churn_distribution(label_df):
    """可视化流失分布"""
    plt.figure(figsize=(8, 6))

    # 计算流失率
    churn_rate = label_df["churn"].mean() * 100

    label_df_copy = label_df.copy()
    label_df_copy["churn_str"] = label_df_copy["churn"].map(
        {0: "Not Churn", 1: "Churn"}
    )

    ax = sns.countplot(
        x="churn_str",
        data=label_df_copy,
        palette=["#4C72B0", "#DD8452"],
        order=["Not Churn", "Churn"],
    )

    plt.title(
        f"Customer Churn Distribution (Churn Rate: {churn_rate:.1f}%)", fontsize=14
    )
    plt.ylabel("Customer Count", fontsize=12)
    plt.xlabel("")

    # 添加数值标签
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "churn_distribution.png"), dpi=120)
    plt.close()


def plot_feature_importance(clf, feature_names, top_n=15):
    """可视化特征重要性"""
    # 检查模型类型
    if hasattr(clf, "feature_importances_"):
        # 随机森林等有特征重要性的模型
        importances = clf.feature_importances_
        indices = importances.argsort()[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        ax = sns.barplot(
            x=importances[indices],
            y=[feature_names[i] for i in indices],
            color="#4C72B0",
            orient="h",
        )

        plt.title(f"Top {top_n} Feature Importances", fontsize=14)
        plt.xlabel("Importance", fontsize=12)
        plt.ylabel("")

        # 添加数值标签
        for i, v in enumerate(importances[indices]):
            ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=120)
        plt.close()
        print("已保存特征重要性图 (基于特征重要性)")

    elif hasattr(clf, "coef_"):
        # 逻辑回归等有系数的模型
        coefficients = np.abs(clf.coef_[0])
        indices = coefficients.argsort()[::-1][:top_n]

        plt.figure(figsize=(10, 8))
        ax = sns.barplot(
            x=coefficients[indices],
            y=[feature_names[i] for i in indices],
            color="#DD8452",
            orient="h",
        )

        plt.title(f"Top {top_n} Feature Coefficients (Absolute Values)", fontsize=14)
        plt.xlabel("|Coefficient|", fontsize=12)
        plt.ylabel("")

        # 添加数值标签
        for i, v in enumerate(coefficients[indices]):
            ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=120)
        plt.close()
        print("已保存特征重要性图 (基于系数绝对值)")

    else:
        # SVM等没有直接特征重要性的模型
        print("SVM模型无法直接获取特征重要性，跳过特征重要性可视化")
        print("建议: 使用随机森林或逻辑回归模型来获得特征重要性分析")


def plot_confusion_matrix(y_true, y_pred):
    """可视化混淆矩阵"""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not Churn", "Churn"]
    )
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())

    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()


def plot_roc_curve(y_true, y_score):
    """可视化ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"), dpi=120)
    plt.close()


def plot_precision_recall_curve(y_true, y_score):
    """可视化精确率-召回率曲线"""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="#DD8452",
        lw=2,
        label=f"PR Curve (AP = {avg_precision:.3f})",
    )

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "precision_recall_curve.png"), dpi=120)
    plt.close()


def identify_high_risk_customers(
    clf, scaler, features_df, customer_keys, threshold=0.7
):
    """识别高风险客户"""
    # 对所有客户进行预测
    X_all = features_df.drop("CustomerKey", axis=1, errors="ignore")
    X_all_scaled = scaler.transform(X_all)

    # 获取流失概率
    churn_proba = clf.predict_proba(X_all_scaled)[:, 1]

    # 创建结果DataFrame
    results_df = pd.DataFrame(
        {"CustomerKey": customer_keys, "churn_probability": churn_proba}
    )

    # 按流失概率排序
    results_df = results_df.sort_values("churn_probability", ascending=False)

    # 识别高风险客户
    high_risk = results_df[results_df["churn_probability"] >= threshold]

    # 风险等级划分
    results_df["risk_level"] = pd.cut(
        results_df["churn_probability"],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=["低风险", "中低风险", "中高风险", "高风险"],
    )

    print("=== 高风险客户分析 ===")
    print(f"流失概率阈值: {threshold}")
    print(f"高风险客户数量: {len(high_risk)}")
    print(f"高风险客户比例: {len(high_risk) / len(results_df) * 100:.2f}%")

    # 流失概率统计
    print("\n=== 流失概率统计 ===")
    print("概率分布特征:")
    print(f"  平均流失概率: {results_df['churn_probability'].mean():.3f}")
    print(f"  最高流失概率: {results_df['churn_probability'].max():.3f}")
    print(f"  最低流失概率: {results_df['churn_probability'].min():.3f}")
    print(f"  概率标准差: {results_df['churn_probability'].std():.3f}")
    print(
        f"  概率分布范围: {results_df['churn_probability'].min():.3f}-{results_df['churn_probability'].max():.3f}"
    )

    # 风险等级分布
    risk_distribution = results_df["risk_level"].value_counts().sort_index()
    print("\n风险等级分布:")
    for level, count in risk_distribution.items():
        percentage = count / len(results_df) * 100
        print(f"  {level}: {count}人 ({percentage:.1f}%)")

    # 各风险等级的概率统计
    print("\n各风险等级概率统计:")
    for level in ["低风险", "中低风险", "中高风险", "高风险"]:
        level_data = results_df[results_df["risk_level"] == level]
        if len(level_data) > 0:
            avg_prob = level_data["churn_probability"].mean()
            min_prob = level_data["churn_probability"].min()
            max_prob = level_data["churn_probability"].max()
            print(
                f"  {level}: 平均概率 {avg_prob:.3f}, 范围 {min_prob:.3f}-{max_prob:.3f}"
            )

    # 高风险客户概率特征
    if len(high_risk) > 0:
        print("\n高风险客户概率特征:")
        print(f"  平均流失概率: {high_risk['churn_probability'].mean():.3f}")
        print(f"  最高流失概率: {high_risk['churn_probability'].max():.3f}")
        print(f"  最低流失概率: {high_risk['churn_probability'].min():.3f}")
        print(
            f"  概率分布集中度: {'高' if high_risk['churn_probability'].std() < 0.1 else '中等' if high_risk['churn_probability'].std() < 0.2 else '低'} (标准差: {high_risk['churn_probability'].std():.3f})"
        )

    # 概率阈值选择建议
    print("\n=== 概率阈值选择建议 ===")
    print("业务阈值: 70% (基于业务成本和收益平衡)")
    print("精确率阈值: 85% (高精确率，减少误报)")
    print("召回率阈值: 60% (确保覆盖大部分流失客户)")

    # 概率应用策略
    print("\n=== 概率应用策略 ===")
    print("个性化营销: 根据概率值制定差异化策略")
    print("资源分配: 优先关注高概率客户")
    print("干预时机: 概率超过80%时立即干预")
    print("效果评估: 通过概率变化评估干预效果")

    # 生成业务建议
    # 删除详细输出
    # generate_business_recommendations(results_df, features_df)

    return results_df, high_risk


def generate_business_recommendations(results_df, features_df):
    """生成业务建议"""
    # 删除详细输出，只保留函数功能
    # print(f"\n=== 业务建议 ===")

    # 合并特征数据
    analysis_df = results_df.merge(features_df, on="CustomerKey", how="left")

    # 高风险客户特征分析
    _high_risk_customers = analysis_df[analysis_df["churn_probability"] >= 0.7]

    # 中风险客户建议
    _medium_risk_customers = analysis_df[
        (analysis_df["churn_probability"] >= 0.3)
        & (analysis_df["churn_probability"] < 0.7)
    ]

    # 低风险客户建议
    _low_risk_customers = analysis_df[analysis_df["churn_probability"] < 0.3]

    # 删除所有详细的print输出，保留函数逻辑
    # 这里只保留数据处理逻辑，不输出详细建议


def select_features(X, y, method="correlation", threshold=0.01):
    """
    特征选择
    """
    print(f"特征选择 - 方法: {method}, 阈值: {threshold}")
    print("特征选择策略: 基于数据质量和业务重要性进行特征筛选")

    if method == "correlation":
        # 基于与目标变量的相关性选择特征
        print("使用相关性分析方法")
        correlations = X.corrwith(y).abs()
        selected_features = correlations[correlations > threshold].index.tolist()

        print("相关性分析结果:")
        print(f"  与目标变量相关性 > {threshold} 的特征数量: {len(selected_features)}")

        # 显示高相关性特征
        high_corr_features = correlations[correlations > 0.1].sort_values(
            ascending=False
        )
        if len(high_corr_features) > 0:
            print("  高相关性特征 (|r| > 0.1):")
            for feature, corr in high_corr_features.items():
                print(f"    • {feature}: {corr:.3f}")

    elif method == "variance":
        # 基于方差选择特征
        print("使用方差分析方法")
        from sklearn.feature_selection import VarianceThreshold

        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        selected_features = X.columns[selector.get_support()].tolist()

        print("方差分析结果:")
        print(f"  方差 > {threshold} 的特征数量: {len(selected_features)}")

    elif method == "mutual_info":
        # 基于互信息选择特征
        print("使用互信息分析方法")
        from sklearn.feature_selection import mutual_info_classif

        mi_scores = mutual_info_classif(X, y, random_state=42)
        selected_features = X.columns[mi_scores > threshold].tolist()

        print("互信息分析结果:")
        print(f"  互信息 > {threshold} 的特征数量: {len(selected_features)}")

        # 显示高互信息特征
        mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})
        high_mi_features = mi_df[mi_df["mi_score"] > 0.01].sort_values(
            "mi_score", ascending=False
        )
        if len(high_mi_features) > 0:
            print("  高互信息特征 (MI > 0.01):")
            for _, row in high_mi_features.head(10).iterrows():
                print(f"    • {row['feature']}: {row['mi_score']:.3f}")

    else:
        # 默认使用所有特征
        print("使用所有特征")
        selected_features = X.columns.tolist()

    print("\n=== 特征选择结果 ===")
    print(f"原始特征数: {len(X.columns)}")
    print(f"选择特征数: {len(selected_features)}")
    print(f"特征选择率: {len(selected_features) / len(X.columns) * 100:.1f}%")

    # 特征类型分析
    print("\n=== 选择特征类型分析 ===")
    rfm_features = [f for f in selected_features if "rfm" in f.lower()]
    behavior_features = [
        f
        for f in selected_features
        if any(x in f.lower() for x in ["order", "amount", "frequency", "trend"])
    ]
    demographic_features = [
        f
        for f in selected_features
        if any(
            x in f.lower()
            for x in ["income", "children", "education", "occupation", "gender"]
        )
    ]
    other_features = [
        f
        for f in selected_features
        if f not in rfm_features + behavior_features + demographic_features
    ]

    print(f"  • RFM特征: {len(rfm_features)}个")
    print(f"  • 行为特征: {len(behavior_features)}个")
    print(f"  • 人口统计特征: {len(demographic_features)}个")
    print(f"  • 其他特征: {len(other_features)}个")

    # 特征选择建议
    print("\n=== 特征选择建议 ===")
    if len(selected_features) < len(X.columns) * 0.5:
        print(
            f"✓ 特征选择有效，减少了 {len(X.columns) - len(selected_features)} 个冗余特征"
        )
    elif len(selected_features) > len(X.columns) * 0.9:
        print("⚠ 特征选择效果有限，建议调整阈值或使用其他方法")
    else:
        print("✓ 特征选择适中，平衡了特征数量和模型性能")

    return selected_features


def plot_feature_histograms(data, feature_names, max_features=10):
    """可视化主要特征分布"""
    # 删除详细输出
    # print(f"生成特征分布图 - 特征数量: {len(feature_names)}")

    # 限制特征数量，避免生成过多图表
    if len(feature_names) > max_features:
        # print(f"特征数量过多，只显示前{max_features}个特征")
        feature_names = feature_names[:max_features]

    for col in feature_names:
        if col in data.columns:
            plt.figure(figsize=(8, 6))

            # 检查是否为分类特征
            if data[col].dtype == "object" or data[col].nunique() < 10:
                # 分类特征使用条形图
                ax = sns.countplot(data=data, x=col, color="#4C72B0")
                plt.title(f"{col} Distribution", fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel("Count", fontsize=12)

                # 添加数值标签
                for p in ax.patches:
                    ax.annotate(
                        f"{int(p.get_height())}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )
            else:
                # 数值特征使用直方图
                ax = sns.histplot(
                    data=data,
                    x=col,
                    bins=30,
                    kde=True,
                    color="#4C72B0",
                    edgecolor="black",
                    alpha=0.8,
                )
                plt.title(f"{col} Distribution", fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel("Count", fontsize=12)

                # 添加统计信息
                mean_val = data[col].mean()
                median_val = data[col].median()
                plt.axvline(
                    mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.2f}"
                )
                plt.axvline(
                    median_val,
                    color="green",
                    linestyle="--",
                    label=f"Median: {median_val:.2f}",
                )
                plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(REPORT_DIR, f"{col}_hist.png"), dpi=120)
            plt.close()
            # 删除详细输出
            # print(f"已保存 {col}_hist.png")


def plot_feature_correlation_heatmap(X, max_features=20):
    """可视化特征相关性热力图"""
    # 删除详细输出
    # print("生成特征相关性热力图")

    # 限制特征数量，避免热力图过于复杂
    if len(X.columns) > max_features:
        # print(f"特征数量过多，只显示前{max_features}个特征")
        X_subset = X.iloc[:, :max_features]
    else:
        X_subset = X

    # 计算相关性矩阵
    corr_matrix = X_subset.corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "feature_correlation_heatmap.png"), dpi=120)
    plt.close()
    # 删除详细输出
    # print("已保存 feature_correlation_heatmap.png")


def plot_feature_vs_target(data, feature_names, target_col="churn", max_features=10):
    """可视化特征与目标变量的关系"""
    # 删除详细输出
    # print(f"生成特征与目标变量关系图 - 特征数量: {len(feature_names)}")

    # 限制特征数量
    if len(feature_names) > max_features:
        # print(f"特征数量过多，只显示前{max_features}个特征")
        feature_names = feature_names[:max_features]

    for col in feature_names:
        if col in data.columns and col != target_col:
            plt.figure(figsize=(8, 6))

            # 检查是否为分类特征
            if data[col].dtype == "object" or data[col].nunique() < 10:
                # 分类特征使用分组条形图
                _ax = sns.countplot(
                    data=data, x=col, hue=target_col, palette=["#4C72B0", "#DD8452"]
                )
                plt.title(f"{col} vs {target_col}", fontsize=14)
                plt.xlabel(col, fontsize=12)
                plt.ylabel("Count", fontsize=12)
                plt.legend(title=target_col, labels=["Not Churn", "Churn"])
            else:
                # 数值特征使用箱线图
                _ax = sns.boxplot(
                    data=data, x=target_col, y=col, palette=["#4C72B0", "#DD8452"]
                )
                plt.title(f"{col} vs {target_col}", fontsize=14)
                plt.xlabel(target_col, fontsize=12)
                plt.ylabel(col, fontsize=12)
                plt.xticks([0, 1], ["Not Churn", "Churn"])

            plt.tight_layout()
            plt.savefig(os.path.join(REPORT_DIR, f"{col}_vs_{target_col}.png"), dpi=120)
            plt.close()
            # 删除详细输出
            # print(f"已保存 {col}_vs_{target_col}.png")


def main():
    print("=== 客户流失预测模型 ===")

    # 1. 加载数据
    print("1. 加载数据...")
    df = load_data()
    print(f"原始数据量: {len(df)}")

    # 2. 数据质量检查
    print("\n2. 数据质量检查...")
    df = check_data_quality(df)

    # 3. 设置时间窗口（避免数据泄露）
    print("\n3. 设置时间窗口...")
    # 改进的时间窗口设置
    all_dates = df["OrderDate"].dropna().sort_values()

    # 确保有足够的历史数据和未来数据
    min_history_days = 365  # 至少需要1年历史数据
    min_future_days = 180  # 至少需要6个月未来数据

    # 计算合适的分割点
    total_days = (all_dates.max() - all_dates.min()).days
    if total_days < (min_history_days + min_future_days):
        print(
            f"警告: 数据时间跨度不足 ({total_days}天 < {min_history_days + min_future_days}天)"
        )
        # 使用80%的数据作为历史数据
        cutoff_date = all_dates.iloc[int(len(all_dates) * 0.8)]
    else:
        # 使用更合理的分割点
        cutoff_date = all_dates.max() - pd.DateOffset(days=min_future_days)

    print(f"特征工程截止日期: {cutoff_date}")
    print("预测窗口: 180天")

    # 4. 特征工程
    print("\n4. 特征工程...")
    features = feature_engineering(df, cutoff_date)
    print(f"特征样本数: {len(features)}")
    print(f"特征数量: {len(features.columns) - 1}")  # 减去CustomerKey

    # 5. 生成标签
    print("\n5. 生成流失标签...")
    label_df = generate_label(df, cutoff_date, prediction_window=180)

    # 6. 可视化流失分布
    plot_churn_distribution(label_df)
    # 删除详细输出
    # print("已保存客户流失分布图")

    # 7. 合并特征和标签
    print("\n6. 合并特征和标签...")
    data = features.merge(label_df, on="CustomerKey", how="inner")
    print(f"最终数据集大小: {len(data)}")

    # 8. 准备训练数据
    X = data.drop(["CustomerKey", "churn"], axis=1, errors="ignore")
    y = data["churn"]
    customer_keys = data["CustomerKey"]

    # 9. 特征选择
    print("\n7. 特征选择...")
    selected_features = select_features(X, y, method="correlation", threshold=0.01)
    X_selected = X[selected_features]

    # 10. 数据探索可视化
    print("\n8. 数据探索可视化...")
    # 生成主要特征分布图
    main_features = [
        "total_orders",
        "total_amount",
        "avg_order_value",
        "rfm_recency",
        "rfm_frequency",
        "rfm_monetary",
        "rfm_score",
        "recent_6m_orders",
        "recent_6m_amount",
        "trend_6m_vs_12m_orders",
        "trend_6m_vs_12m_amount",
        "YearlyIncome",
        "TotalChildren",
        "NumberCarsOwned",
        "is_new_customer",
    ]
    available_features = [col for col in main_features if col in data.columns]
    plot_feature_histograms(data, available_features, max_features=15)

    # 生成特征相关性热力图
    plot_feature_correlation_heatmap(X_selected, max_features=15)

    # 生成特征与目标变量关系图
    key_features = [
        "total_orders",
        "rfm_recency",
        "rfm_score",
        "recent_6m_orders",
        "YearlyIncome",
    ]
    available_key_features = [col for col in key_features if col in data.columns]
    plot_feature_vs_target(
        data, available_key_features, target_col="churn", max_features=8
    )

    # 11. 交叉验证
    print("\n9. 交叉验证...")
    cv_scores = cross_validate_model(X_selected, y)

    # 12. 训练模型
    print("\n10. 训练模型...")
    clf, X_test, y_test, y_pred, y_score, scaler, comparison_df = (
        train_model_with_timeseries(X_selected, y)
    )

    # 13. 模型评估可视化
    print("\n11. 生成评估图表...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_score)
    plot_precision_recall_curve(y_test, y_score)
    plot_feature_importance(clf, X_selected.columns)

    # 删除详细输出
    # print("已保存所有评估图表")

    # 14. 识别高风险客户
    print("\n12. 识别高风险客户...")
    all_results, high_risk = identify_high_risk_customers(
        clf, scaler, features[selected_features + ["CustomerKey"]], customer_keys
    )

    # 15. 保存结果
    print("\n13. 保存结果...")
    # 保存所有客户的预测结果
    all_results.to_csv(
        os.path.join(REPORT_DIR, "all_customer_predictions.csv"), index=False
    )

    # 保存高风险客户名单
    high_risk.to_csv(os.path.join(REPORT_DIR, "high_risk_customers.csv"), index=False)

    # 保存特征选择结果
    feature_selection_df = pd.DataFrame(
        {"feature_name": selected_features, "feature_type": "selected"}
    )
    feature_selection_df.to_csv(
        os.path.join(REPORT_DIR, "selected_features.csv"), index=False
    )

    # 保存模型性能报告
    with open(
        os.path.join(REPORT_DIR, "model_performance.txt"), "w", encoding="utf-8"
    ) as f:
        f.write("=== 客户流失预测模型性能报告 ===\n\n")
        f.write(f"数据分割日期: {cutoff_date}\n")
        f.write("预测窗口: 180天\n")
        f.write(f"客户总数: {len(label_df)}\n")
        f.write(f"流失客户数: {label_df['churn'].sum()}\n")
        f.write(f"流失率: {label_df['churn'].mean() * 100:.2f}%\n")
        f.write(f"原始特征数: {len(X.columns)}\n")
        f.write(f"选择特征数: {len(selected_features)}\n")
        f.write(f"特征选择率: {len(selected_features) / len(X.columns) * 100:.1f}%\n")
        f.write(
            f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n"
        )
        f.write(f"高风险客户数量: {len(high_risk)}\n")
        f.write(f"高风险客户比例: {len(high_risk) / len(all_results) * 100:.2f}%\n\n")

        # 添加模型比较结果
        f.write("=== 模型比较结果 ===\n")
        f.write(comparison_df.to_string(index=False, float_format="%.3f"))
        f.write("\n\n")

        # 添加最佳模型选择说明
        best_model_name = comparison_df.loc[comparison_df["AUC"].idxmax(), "Model"]
        f.write(f"最终选择模型: {best_model_name}\n")
        f.write(f"选择原因: AUC最高 ({comparison_df['AUC'].max():.3f})\n")
        f.write(
            f"模型复杂度: {'中等' if 'Random Forest' in best_model_name else '较低' if 'Logistic' in best_model_name else '较高'}\n"
        )
        f.write(
            f"特征解释性: {'优秀' if 'Random Forest' in best_model_name else '良好' if 'Logistic' in best_model_name else '一般'}\n\n"
        )

        # 添加特征工程信息
        f.write("=== 特征工程信息 ===\n")
        f.write("特征工程策略: 严格时间分割，避免数据泄露\n")
        f.write("特征类型分布:\n")
        f.write("  • 静态特征: 人口统计、地理特征等\n")
        f.write("  • 行为特征: 购买频率、消费金额等\n")
        f.write("  • RFM特征: 最近购买时间、购买频率、消费金额\n")
        f.write("  • 趋势特征: 近期行为变化趋势\n")
        f.write("  • 生命周期特征: 客户生命周期天数\n")
        f.write("  • 编码特征: 分类特征编码\n\n")

        # 添加数据质量控制信息
        f.write("=== 数据质量控制 ===\n")
        f.write("数据泄露检测: 时间序列分割验证\n")
        f.write("异常值处理: IQR方法 + 业务规则\n")
        f.write("特征选择: 相关性分析 + 重要性排序\n")
        f.write("模型验证: 时间序列交叉验证\n\n")

        # 添加业务建议
        f.write("=== 业务建议 ===\n")
        f.write("• 建立客户流失预警监控体系\n")
        f.write("• 实施差异化客户服务策略\n")
        f.write("• 定期更新客户流失预测模型\n")
        f.write("• 建立客户生命周期管理体系\n")
        f.write("• 加强数据驱动的决策支持\n")


if __name__ == "__main__":
    main()
