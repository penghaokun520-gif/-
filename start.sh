#!/bin/bash
# 启动皮肤识别服务（Mac / Linux）
# 用法：API_KEY=你的密钥 ./start.sh

set -e
cd "$(dirname "$0")"

PORT=${PORT:-8765}

# 可选：启动前自动检查更新
# python updater.py

echo "启动皮肤识别服务，端口 $PORT"
API_KEY="$API_KEY" PORT="$PORT" python server.py
