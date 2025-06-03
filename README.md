# 数据仓库与数据挖掘综合实践

本项目基于 AdventureWorksDW 数据仓库，按照 CRISP-DM 流程，完成客户流失预测、季节性产品关联规则挖掘、澳元客户RFM分析等数据挖掘任务。

## 目录结构

```
project/
├── data/        # 数据下载、缓存、处理文件
├── src/         # 主要Python脚本
├── reports/     # 分析报告、图表
├── doc/         # 项目文档
├── README.md    # 项目说明
└── pyproject.toml # 依赖管理
```

## 主要任务
1. 客户流失预测与留存策略
2. 季节性产品关联规则挖掘与促销建议
3. 澳元客户RFM模型群体划分与服务方案

## 环境说明
- Python 3.11（推荐使用uv安装）
- uv包管理器
- SQL Server 2019 (docker)

## 数据库连接信息
- host: localhost
- port: 8433
- username: sa
- password: Alaska2017
- database: AdventureWorksDW2017

## 项目运行指导

### 1. 环境准备

#### 1.1 安装uv包管理器

**macOS/Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
(Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -UseBasicParsing).Content | pwsh -Command -
```

安装完成后，将uv添加到PATH：
```powershell
$env:PATH = "$env:LOCALAPPDATA\uv\bin;$env:PATH"
```

#### 1.2 使用uv安装Python

**macOS/Linux**
```bash
# 安装Python 3.11
uv venv --python=3.11 .venv

# 激活虚拟环境
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
# 安装Python 3.11
uv venv --python=3.11 .venv

# 激活虚拟环境
.venv\Scripts\Activate.ps1
```

**Windows (CMD)**
```cmd
# 安装Python 3.11
uv venv --python=3.11 .venv

# 激活虚拟环境
.venv\Scripts\activate.bat
```

> 注意：如果系统中没有Python 3.11，uv会自动下载并安装。如果希望手动安装Python，可以参考以下备选方案：

**备选方案：手动安装Python 3.11**

**macOS**
```bash
# 使用Homebrew安装
brew install python@3.11

# 配置PATH（如果使用Homebrew）
echo 'export PATH="/opt/homebrew/opt/python@3.11/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**Windows**
```powershell
# 使用winget安装
winget install Python.Python.3.11

# 或从Python官网下载安装包
# https://www.python.org/downloads/release/python-3110/
```

**Linux (Ubuntu/Debian)**
```bash
# 添加deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# 安装Python 3.11
sudo apt install python3.11 python3.11-venv
```

#### 1.3 安装项目依赖

所有系统通用：
```bash
# 使用uv安装依赖
uv sync
```

### 2. 数据库设置

#### 2.1 安装Docker

**macOS**
```bash
# 使用Homebrew安装
brew install --cask docker
```

**Windows**
```powershell
# 使用winget安装
winget install Docker.DockerDesktop
```

**Linux (Ubuntu)**
```bash
# 安装Docker
sudo apt update
sudo apt install docker.io

# 启动Docker服务
sudo systemctl start docker
sudo systemctl enable docker

# 将当前用户添加到docker组
sudo usermod -aG docker $USER
```

#### 2.2 启动SQL Server容器

所有系统通用：
```bash
# 拉取SQL Server镜像
docker pull mcr.microsoft.com/mssql/server:2019-latest

# 启动容器
docker run -d \
 -e "ACCEPT_EULA=Y" \
 -e "SA_PASSWORD=Alaska2017" \
 -p 8433:1433 \
 -v vlm_0001_mssql:/var/opt/mssql \
 -v vlm_000_sqlserver:/var/opt/sqlserver \
 -v $(pwd)/volumes/mssql/backup:/mssql_backups \
 --name mssql19 \
  mcr.microsoft.com/mssql/server:2019-latest
```

#### 2.3 安装SQL Server命令行工具

**macOS**
```bash
# 使用Homebrew安装
brew tap microsoft/mssql-release https://github.com/Microsoft/homebrew-mssql-release
brew update
brew install msodbcsql18
```

**Windows**
```powershell
# 使用winget安装
winget install Microsoft.SQLServer.CLI
```

**Linux (Ubuntu)**
```bash
# 添加Microsoft仓库
curl https://packages.microsoft.com/keys/microsoft.asc | sudo apt-key add -
curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | sudo tee /etc/apt/sources.list.d/mssql-release.list

# 安装工具
sudo apt update
sudo apt install -y mssql-tools18 unixodbc-dev
```

#### 2.4 验证数据库连接

所有系统通用：
```bash
# 等待数据库启动（约30秒）
sleep 30

# 测试连接
sqlcmd -S localhost,8433 -U sa -P Alaska2017 -Q "SELECT @@VERSION"
```

#### 2.5 恢复数据库

所有系统通用：
```bash
# 下载数据库备份文件
curl -L -o AdventureWorksDW2017.bak https://github.com/Microsoft/sql-server-samples/releases/download/adventureworks/AdventureWorksDW2017.bak

# 恢复数据库
sqlcmd -S localhost,8433 -U sa -P Alaska2017 -Q "RESTORE DATABASE AdventureWorksDW2017 FROM DISK = '$(pwd)/AdventureWorksDW2017.bak' WITH MOVE 'AdventureWorksDW2017' TO '/var/opt/mssql/data/AdventureWorksDW2017.mdf', MOVE 'AdventureWorksDW2017_log' TO '/var/opt/mssql/data/AdventureWorksDW2017_log.ldf'"
```

### 3. 运行分析任务

所有系统通用：

1. 运行分析任务（支持命令行参数）
```bash
# 运行所有分析任务（默认行为）
uv run python main.py

# 运行指定分析任务
uv run python main.py --task churn        # 仅运行客户流失预测
uv run python main.py --task association  # 仅运行季节性关联规则挖掘
uv run python main.py --task rfm          # 仅运行澳元客户RFM分析
uv run python main.py --task all          # 运行所有分析任务（与默认行为相同）

# 查看帮助信息
uv run python main.py --help
```

2. 直接运行单个分析脚本（不推荐，建议使用main.py）
```bash
# 客户流失预测
uv run python src/churn_predict.py

# 季节性关联规则挖掘
uv run python src/seasonal_association.py

# 澳元客户RFM分析
uv run python src/rfm_analysis.py
```

3. 查看分析结果
- 客户流失预测报告：`reports/客户流失预测/`
  - 特征重要性图：`feature_importance.png`
  - 混淆矩阵：`confusion_matrix.png`
  - ROC曲线：`roc_curve.png`
  - 高风险客户名单：`high_risk_customers.csv`
- 季节性关联规则报告：`reports/季节性产品关联规则/`
  - 各季节关联规则：`{Season}_rules.csv`
  - 规则网络图：`{Season}_rules_network.png`
  - 热力图：`{Season}_heatmap.png`
- 澳元客户RFM报告：`reports/澳元客户RFM/`
  - 客户分群结果：`rfm_cluster_scatter.png`
  - 群体特征雷达图：`rfm_radar.png`
  - 产品偏好分析：`product_preferences.png`
  - 客户生命周期分析：`customer_lifecycle.png`

### 4. 常见问题

#### 4.1 数据库连接问题

所有系统通用：
```bash
# 检查Docker容器状态
docker ps

# 检查容器日志
docker logs adventureworks

# 验证端口映射
docker port adventureworks

# 检查数据库状态
sqlcmd -S localhost,8433 -U sa -P Alaska2017 -Q "SELECT name, state_desc FROM sys.databases"
```

#### 4.2 依赖问题

所有系统通用：
```bash
# 检查uv版本
uv --version

# 更新uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 重新同步依赖
uv sync --upgrade
```

#### 4.3 分析任务问题

所有系统通用：
```bash
# 检查Python版本
uv run python --version

# 验证虚拟环境
uv run which python

# 检查依赖状态
uv pip list
```

### 5. 开发说明

1. 代码规范
   - 遵循PEP 8编码规范
   - 使用类型注解
   - 编写单元测试

2. 文档维护
   - 更新README.md
   - 及时更新依赖配置

3. 版本控制
   - 使用Git进行版本管理
   - 遵循语义化版本规范
   - 保持提交信息清晰

## 使用说明
所有核心分析代码位于 src/ 目录下，运行前请确保已安装依赖并正确配置数据库环境。详细的分析报告和可视化结果保存在 reports/ 目录下。
