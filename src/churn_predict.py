import pandas as pd
from data_utils import get_engine, read_sql
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 数据加载
# 2. 特征工程
# 3. 标签生成
# 4. 建模与评估
# 5. 高风险客户输出

# 设置美观主题
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

# 统一输出目录
REPORT_DIR = "./reports/客户流失预测/"
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
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
    return df


def feature_engineering(df):
    # 静态特征直接保留
    static_cols = [
        "CustomerKey",
        "BirthDate",
        "Gender",
        "YearlyIncome",
        "TotalChildren",
        "NumberChildrenAtHome",
        "EnglishEducation",
        "EnglishOccupation",
        "HouseOwnerFlag",
        "NumberCarsOwned",
        "DateFirstPurchase",
        "CommuteDistance",
    ]
    static_df = df[static_cols].drop_duplicates("CustomerKey").set_index("CustomerKey")

    # 行为特征：近6个月购买频率、消费金额波动率等
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    recent_date = df["OrderDate"].max()
    six_months_ago = recent_date - pd.DateOffset(months=6)
    recent_df = df[df["OrderDate"] >= six_months_ago]

    # 购买频率
    freq = recent_df.groupby("CustomerKey")["OrderDate"].count().rename("freq_6m")
    # 消费金额波动率
    amount_std = (
        recent_df.groupby("CustomerKey")["SalesAmount"].std().rename("amount_std_6m")
    )
    amount_mean = (
        recent_df.groupby("CustomerKey")["SalesAmount"].mean().rename("amount_mean_6m")
    )
    amount_cv = (amount_std / amount_mean).rename("amount_cv_6m")

    # 合并所有特征
    features = static_df.join([freq, amount_cv]).fillna(0)
    features = features.reset_index()

    # 对所有非数值型特征进行独热编码
    features = pd.get_dummies(features, drop_first=True)

    return features


def generate_label(df):
    # 近6个月未购买的客户
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    recent_date = df["OrderDate"].max()
    six_months_ago = recent_date - pd.DateOffset(months=6)
    # 有购买记录的客户
    active_customers = df[df["OrderDate"] >= six_months_ago]["CustomerKey"].unique()
    # 所有客户
    all_customers = df["CustomerKey"].unique()
    # 未购买的客户
    inactive_customers = set(all_customers) - set(active_customers)
    # 标签：1=流失，0=未流失
    label_df = pd.DataFrame({"CustomerKey": all_customers})
    label_df["churn"] = label_df["CustomerKey"].apply(
        lambda x: 1 if x in inactive_customers else 0
    )
    return label_df


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))
    return clf, X_test, y_test, y_pred


def plot_churn_distribution(label_df):
    plt.figure(figsize=(5, 4))
    # 为了消除palette警告，先构造churn为字符串类别
    label_df = label_df.copy()
    label_df["churn_str"] = label_df["churn"].map({0: "Not Churn", 1: "Churn"})
    ax = sns.countplot(
        x="churn_str",
        data=label_df,
        hue="churn_str",
        palette=["#4C72B0", "#DD8452"],
        legend=False,
        order=["Not Churn", "Churn"],
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not Churn", "Churn"])
    plt.title("Churn Distribution", fontsize=14)
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("")
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(REPORT_DIR, "churn_distribution.png"), dpi=120)
    plt.close()


def plot_feature_hist(features, feature_names):
    for col in feature_names:
        plt.figure(figsize=(5, 4))
        ax = sns.histplot(
            features[col],
            bins=30,
            kde=True,
            color="#4C72B0",
            edgecolor="black",
            alpha=0.8,
        )
        plt.title(f"{col} Distribution", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.tight_layout(pad=1.0)
        plt.savefig(os.path.join(REPORT_DIR, f"{col}_hist.png"), dpi=120)
        plt.close()


def plot_feature_importance(clf, feature_names):
    importances = clf.feature_importances_
    indices = importances.argsort()[::-1][:10]
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        color="#4C72B0",
        orient="h",
    )
    plt.title("Top 10 Feature Importances", fontsize=14)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("")
    for i, v in enumerate(importances[indices]):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=120)
    plt.close()


def plot_confusion_matrix(y_true, y_pred):
    plt.figure(figsize=(5, 4))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Not Churn", "Churn"]
    )
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.title("Confusion Matrix", fontsize=14)
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()


def plot_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic", fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout(pad=1.0)
    plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"), dpi=120)
    plt.close()


def main():
    print("加载数据...")
    df = load_data()
    print(f"原始数据量: {len(df)}")

    print("特征工程...")
    features = feature_engineering(df)
    print(f"特征样本数: {len(features)}")

    print("生成流失标签...")
    label_df = generate_label(df)
    print(f"客户总数: {len(label_df)}，流失客户数: {label_df['churn'].sum()}")

    # 可视化流失分布
    plot_churn_distribution(label_df)
    print("已保存客户流失分布图到 ./reports/客户流失预测/churn_distribution.png")

    # 合并特征和标签
    data = features.merge(label_df, on="CustomerKey")
    X = data.drop(
        ["CustomerKey", "churn", "BirthDate", "DateFirstPurchase"],
        axis=1,
        errors="ignore",
    )
    y = data["churn"]

    # 可视化主要特征分布
    main_features = [
        col
        for col in [
            "freq_6m",
            "amount_cv_6m",
            "YearlyIncome",
            "TotalChildren",
            "NumberCarsOwned",
        ]
        if col in X.columns
    ]
    plot_feature_hist(data, main_features)
    print("已保存主要特征分布图到 ./reports/客户流失预测/")

    print("训练模型并评估...")
    clf, X_test, y_test, y_pred = train_model(X, y)

    # 混淆矩阵可视化
    plot_confusion_matrix(y_test, y_pred)
    print("已保存混淆矩阵图到 ./reports/客户流失预测/confusion_matrix.png")

    # ROC曲线可视化
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test, y_score)
        print("已保存ROC曲线图到 ./reports/客户流失预测/roc_curve.png")

    # 特征重要性可视化
    plot_feature_importance(clf, X.columns)
    print("已保存特征重要性图到 ./reports/客户流失预测/feature_importance.png")

    # 输出高风险客户名单
    high_risk = data.loc[X_test.index][y_pred == 1]
    print(f"高风险客户数: {len(high_risk)}")
    high_risk[["CustomerKey"]].to_csv(
        os.path.join(REPORT_DIR, "high_risk_customers.csv"), index=False
    )
    print("高风险客户名单已保存到 ./reports/客户流失预测/high_risk_customers.csv")


if __name__ == "__main__":
    main()
