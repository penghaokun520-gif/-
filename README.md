# 皮肤识别服务

Valorant 皮肤识别 HTTP 服务，部署在本地电脑（Mac M1/M2/M3），通过 HTTP 接口接收截图并返回识别结果。

## 文件说明

| 文件 | 说明 |
|------|------|
| `server.py` | FastAPI 服务主文件 |
| `skin_recognizer.py` | 识别核心逻辑（EasyOCR + YOLO） |
| `update_skins.py` | 皮肤数据库更新工具 |
| `updater.py` | 服务自动更新工具 |
| `best.pt` | YOLO 模型权重（不上传 Git，手动放置） |
| `data/` | 皮肤数据库（不上传 Git，手动放置） |

## 安装（Mac Apple Silicon）

```bash
# 1. 克隆仓库
git clone https://github.com/penghaokun520-gif/-.git
cd -

# 2. 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 3. 安装依赖（torch 支持 MPS 加速）
pip install torch torchvision
pip install -r requirements.txt

# 4. 手动放置模型和数据库
# 将 best.pt 放到项目根目录
# 将 data/ 目录放到项目根目录
```

## 启动

```bash
# 设置 API Key 并启动
API_KEY=你设置的密钥 ./start.sh

# 或者指定端口
API_KEY=xxx PORT=8765 ./start.sh
```

## 接口

### GET /health
检查服务状态
```json
{"status": "ok", "model_loaded": true}
```

### POST /recognize
上传截图进行识别，Header 需带 `X-Api-Key`

```bash
curl -X POST http://localhost:8765/recognize \
  -H "X-Api-Key: 你的密钥" \
  -F "file=@截图.jpg"
```

返回：
```json
{
  "ok": true,
  "elapsed": 8.32,
  "count": 3,
  "results": [
    {
      "index": 1,
      "skin_name": "冠军赛幻象",
      "weapon": "幻象",
      "skin_tier": "至臻",
      "match_score": 0.95,
      ...
    }
  ]
}
```

## 更新

```bash
# 检查是否有新版本
python updater.py --check

# 执行更新（不覆盖 best.pt 和 data/）
python updater.py
```

发布新版本：在 GitHub 创建新的 Release，打上版本 tag（如 `v1.1.0`），updater 会自动识别。
