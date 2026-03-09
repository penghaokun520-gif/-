"""
自动更新工具
从 GitHub Releases 拉取最新版本，更新本地代码文件（不覆盖 best.pt 和 data/）

用法：
  python updater.py          # 检查并更新
  python updater.py --check  # 只检查，不更新
"""
import argparse
import hashlib
import json
import os
import shutil
import ssl
import sys
import tempfile
import urllib.request
import zipfile

REPO = "penghaokun520-gif/skin-recognition-service"
GITHUB_API = f"https://api.github.com/repos/{REPO}/releases/latest"

# 这些文件/目录不会被更新覆盖（模型和数据库是本地的）
SKIP_OVERWRITE = {"best.pt", "data", ".env"}

HERE = os.path.dirname(os.path.abspath(__file__))


def _fetch(url: str, headers: dict = None) -> bytes:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req_headers = {"User-Agent": "skin-recognizer-updater/1.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return resp.read()


def get_latest_release() -> dict:
    """获取 GitHub 最新 Release 信息"""
    data = _fetch(GITHUB_API)
    return json.loads(data.decode("utf-8"))


def get_current_version() -> str:
    version_file = os.path.join(HERE, "VERSION")
    if os.path.exists(version_file):
        return open(version_file).read().strip()
    return "unknown"


def download_and_extract(download_url: str, target_dir: str):
    """下载 zip 并解压，跳过不应覆盖的文件"""
    print(f"正在下载...")
    zip_bytes = _fetch(download_url)

    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "release.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.namelist()
            # GitHub zip 的顶层目录是 repo-tag/，需要去掉
            prefix = members[0].split("/")[0] + "/" if members else ""

            for member in members:
                # 去掉顶层目录前缀
                rel_path = member[len(prefix):] if member.startswith(prefix) else member
                if not rel_path:
                    continue

                # 跳过不覆盖的文件/目录
                top = rel_path.split("/")[0]
                if top in SKIP_OVERWRITE:
                    continue

                dest = os.path.join(target_dir, rel_path)
                if member.endswith("/"):
                    os.makedirs(dest, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        dst.write(src.read())

    print("解压完成")


def update(check_only: bool = False):
    print(f"当前版本: {get_current_version()}")
    print("正在检查更新...")

    try:
        release = get_latest_release()
    except Exception as e:
        print(f"检查更新失败: {e}")
        return False

    latest_tag = release.get("tag_name", "")
    latest_name = release.get("name", latest_tag)
    current = get_current_version()

    print(f"最新版本: {latest_tag} ({latest_name})")

    if latest_tag == current:
        print("已是最新版本，无需更新")
        return False

    if check_only:
        print(f"有新版本可用: {latest_tag}")
        return True

    # 找到 source code zip 下载链接
    zipball_url = release.get("zipball_url", "")
    if not zipball_url:
        print("未找到下载链接")
        return False

    print(f"准备更新到 {latest_tag}...")
    download_and_extract(zipball_url, HERE)

    # 写入新版本号
    with open(os.path.join(HERE, "VERSION"), "w") as f:
        f.write(latest_tag)

    print(f"更新完成！当前版本: {latest_tag}")
    print("请重启服务以应用更新")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="皮肤识别服务自动更新工具")
    parser.add_argument("--check", action="store_true", help="只检查是否有更新，不执行更新")
    args = parser.parse_args()
    update(check_only=args.check)
