"""
Valorant 皮肤识别 HTTP 服务
部署在本地电脑（Mac/Windows），接收图片 → 跑识别 → 返回 JSON

启动：
  API_KEY=你的密钥 uvicorn server:app --host 0.0.0.0 --port 8765
  或直接：python server.py
"""
import os
import sys
import tempfile
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import JSONResponse
import uvicorn

from skin_recognizer import SkinRecognizer

API_KEY = os.environ.get("API_KEY", "")

app = FastAPI(title="皮肤识别服务", docs_url=None, redoc_url=None)

recognizer: SkinRecognizer | None = None


@app.on_event("startup")
def load_model():
    global recognizer
    print("正在加载识别模型...")
    recognizer = SkinRecognizer()
    print("模型加载完成，服务就绪")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": recognizer is not None}


@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    x_api_key: str = Header(default=""),
):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if recognizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted")

    suffix = os.path.splitext(file.filename or "img.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        t0 = time.time()
        results = recognizer.recognize_screenshot(tmp_path)
        elapsed = round(time.time() - t0, 2)
        return JSONResponse({
            "ok": True,
            "elapsed": elapsed,
            "count": len(results),
            "results": results,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
