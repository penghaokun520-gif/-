"""
Valorant 皮肤数据库更新工具
从 valorant-api.com 拉取最新皮肤数据，同步到本地数据库。

用法：python update_skins.py
"""
import json
import os
import sys
import urllib.request
import ssl
import io

try:
    import numpy as np
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

API_BASE = "https://valorant-api.com/v1"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "valorant_skins.json")

TIER_MAP = {
    "Exclusive": "至臻",
    "Ultra": "至尊",
    "Premium": "尊享",
    "Select": "精选",
    "Deluxe": "豪华",
    "Battle Pass": "精选",
}


def fetch_bytes(url):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return resp.read()


def compute_phash(img_bytes, hash_size=8):
    """计算图片的感知哈希（pHash），返回十六进制字符串"""
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    # 缩放到 (hash_size*4) x (hash_size*4)，转灰度，做 DCT
    size = hash_size * 4
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(gray)
    dct_low = dct[:hash_size, :hash_size]
    median = np.median(dct_low)
    bits = (dct_low > median).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return format(val, f'0{hash_size * hash_size // 4}x')


def fetch_json(url):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def update():
    print("正在从 valorant-api.com 拉取最新数据...")

    # 1. 拉取品质等级
    tiers_data = fetch_json(f"{API_BASE}/contenttiers?language=zh-CN")
    tier_uuid_map = {}
    for t in tiers_data["data"]:
        en_name = t.get("devName", "")
        cn_name = TIER_MAP.get(en_name, t["displayName"])
        tier_uuid_map[t["uuid"]] = {"name_cn": cn_name, "name_en": en_name}

    # 2. 拉取武器（简体+繁体）
    weapons_cn = fetch_json(f"{API_BASE}/weapons?language=zh-CN")
    weapons_tw = fetch_json(f"{API_BASE}/weapons?language=zh-TW")

    # 构建繁体皮肤名映射 uuid -> tw_name
    tw_skin_map = {}
    tw_weapon_map = {}
    for w in weapons_tw["data"]:
        tw_weapon_map[w["uuid"]] = w["displayName"]
        for s in w.get("skins", []):
            tw_skin_map[s["uuid"]] = s["displayName"]

    # 3. 构建皮肤和武器列表
    weapons_list = []
    skins_list = []
    tier_stats = {}

    for w in weapons_cn["data"]:
        weapons_list.append({
            "uuid": w["uuid"],
            "name": w["displayName"],
            "category": w.get("shopData", {}).get("category", "") if w.get("shopData") else "",
            "icon": f"https://media.valorant-api.com/weapons/{w['uuid']}/displayicon.png",
            "name_tw": tw_weapon_map.get(w["uuid"], w["displayName"]),
        })

        for s in w.get("skins", []):
            tier_uuid = s.get("contentTierUuid")
            if tier_uuid and tier_uuid in tier_uuid_map:
                tier_cn = tier_uuid_map[tier_uuid]["name_cn"]
                tier_en = tier_uuid_map[tier_uuid]["name_en"]
            else:
                tier_cn = "默认"
                tier_en = "Default"

            skins_list.append({
                "uuid": s["uuid"],
                "name": s["displayName"],
                "weapon": w["displayName"],
                "weapon_uuid": w["uuid"],
                "tier": tier_cn,
                "tier_en": tier_en,
                "icon": s.get("displayIcon", ""),
                "chromas": len(s.get("chromas", [])),
                "levels": len(s.get("levels", [])),
                "name_tw": tw_skin_map.get(s["uuid"], s["displayName"]),
            })

            tier_stats[tier_cn] = tier_stats.get(tier_cn, 0) + 1

    # 3b. 为年份歧义系列计算 pHash（OCR 读不完整年份时用视觉区分）
    # 策略：找出系列名去掉年份数字后相同的皮肤组（如 "2021/2023全球冠军赛"）
    import re
    from collections import defaultdict
    HIGH_TIERS = {"至臻", "至尊", "尊享"}

    def _series_name(s):
        if s["weapon"] == "近战武器":
            parts = s["name"].rsplit(" ", 1)
            return parts[0] if len(parts) > 1 else s["name"]
        return s["name"].replace(f" {s['weapon']}", "").replace(s["weapon"], "").strip()

    def _series_no_year(series):
        return re.sub(r'\d{4}', '', series).strip()

    if _HAS_CV2:
        # 按"去年份系列名 + 武器"分组
        no_year_groups = defaultdict(list)
        for s in skins_list:
            if s.get("tier") not in HIGH_TIERS or not s.get("icon"):
                continue
            key = (_series_no_year(_series_name(s)), s["weapon"])
            no_year_groups[key].append(s)

        # 只处理有多个年份的组（真正歧义）
        ambiguous = {k: v for k, v in no_year_groups.items() if len(v) > 1}
        if ambiguous:
            targets = [s for group in ambiguous.values() for s in group]
            print(f"\n发现 {len(ambiguous)} 个年份歧义系列（{len(targets)} 条），正在计算 pHash...")
            for s in targets:
                try:
                    img_bytes = fetch_bytes(s["icon"])
                    phash = compute_phash(img_bytes)
                    if phash:
                        s["phash"] = phash
                except Exception as e:
                    print(f"  警告: {s['name']} pHash 计算失败: {e}")
            print(f"pHash 计算完成")
        else:
            print("\n未发现年份歧义系列，跳过 pHash 计算")
    else:
        print("警告: 未安装 opencv-python，跳过 pHash 计算（年份歧义皮肤将无法视觉区分）")

    # 4. 加载本地数据库，保留手动修正的繁体名
    overrides = {}
    if os.path.exists(DB_PATH):
        old_db = json.load(open(DB_PATH, "r", encoding="utf-8"))
        old_skins = {s["uuid"]: s for s in old_db.get("skins", []) if "uuid" in s}
        # 检测手动修正：本地繁体名与API不同的条目
        for s in skins_list:
            old = old_skins.get(s["uuid"])
            if old and old.get("name_tw") and old["name_tw"] != s["name_tw"]:
                overrides[s["uuid"]] = old["name_tw"]
            # 保留旧数据库中已有的 pHash（避免每次更新都重新下载）
            if old and old.get("phash") and not s.get("phash"):
                s["phash"] = old["phash"]

    # 应用手动修正
    if overrides:
        print(f"保留 {len(overrides)} 个手动修正的繁体名：")
        for uid, tw in overrides.items():
            for s in skins_list:
                if s["uuid"] == uid:
                    print(f"  {s['name']}: API={s['name_tw']} -> 保留={tw}")
                    s["name_tw"] = tw
                    break

    # 5. 保存
    db = {
        "weapons": weapons_list,
        "skins": skins_list,
        "stats": {
            "total_weapons": len(weapons_list),
            "total_skins": len(skins_list),
            "by_tier": tier_stats,
        },
    }

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"\n更新完成！")
    print(f"  武器: {len(weapons_list)}")
    print(f"  皮肤: {len(skins_list)}")
    print(f"  品质分布:")
    for t, c in sorted(tier_stats.items(), key=lambda x: -x[1]):
        print(f"    {t}: {c}")
    print(f"\n已保存到: {DB_PATH}")


if __name__ == "__main__":
    try:
        update()
    except Exception as e:
        print(f"更新失败: {e}", file=sys.stderr)
        sys.exit(1)
