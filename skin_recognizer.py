import cv2
import numpy as np
import json
import sys
import easyocr
import torch
from difflib import SequenceMatcher
from collections import Counter
from ultralytics import YOLO
import opencc

WEAPON_CN_NAMES = [
    "奥丁", "战神", "狂徒", "獠犬", "幻影", "判官", "雄鹿", "狂怒",
    "标配", "追猎", "鬼魅", "正义", "短炮", "冥驹", "戍卫", "莽侠",
    "飞将", "骇灵", "蜂刺", "近战武器",
]

YOLO_EN_TO_CN = {
    "Odin": "奥丁", "Ares": "战神", "Vandal": "狂徒", "Bulldog": "獠犬",
    "Phantom": "幻影", "Judge": "判官", "Bucky": "雄鹿", "Frenzy": "狂怒",
    "Classic": "标配", "Bandit": "追猎", "Ghost": "鬼魅", "Sheriff": "正义",
    "Shorty": "短炮", "Operator": "冥驹", "Guardian": "戍卫", "Outlaw": "莽侠",
    "Marshal": "飞将", "Spectre": "骇灵", "Stinger": "蜂刺", "Melee": "近战武器",
}

MELEE_KEYWORDS = ["短剑", "蝴蝶刀", "爪刀", "战刀", "军刀", "撬棍", "战锤", "战斧",
                  "匕首", "歧途", "链枷", "镰刃", "簇刃", "遗器"]

WEAPON_TW_TO_CN = {
    "奧丁": "奥丁", "戰神": "战神", "暴徒": "狂徒", "鬥牛犬": "獠犬",
    "幻象": "幻影", "判官": "判官", "重砲": "雄鹿", "狂弒": "狂怒",
    "制式手槍": "标配", "盜賊": "追猎", "鬼魅": "鬼魅", "神射": "正义",
    "短管": "短炮", "間諜": "冥驹", "捍衛者": "戍卫", "逃犯": "莽侠",
    "警長": "飞将", "惡靈": "骇灵", "刺針": "蜂刺", "近戰武器": "近战武器",
    "近戰": "近战武器",
    # t2s转换后的繁体武器名（OCR文本经过t2s，需要同时匹配简化后的形式）
    "奥丁": "奥丁", "战神": "战神", "斗牛犬": "獠犬",
    "重炮": "雄鹿", "狂弑": "狂怒",
    "制式手枪": "标配", "盗贼": "追猎", "神射": "正义",
    "间谍": "冥驹", "捍卫者": "戍卫", "逃犯": "莽侠",
    "警长": "飞将", "恶灵": "骇灵", "刺针": "蜂刺", "近战武器": "近战武器",
    "近战": "近战武器",
}

TARGET_TIERS = {"至尊", "至臻", "尊享"}


class SkinRecognizer:
    def __init__(self, db_path="data/valorant_skins.json", model_path="best.pt"):
        self._use_gpu = torch.cuda.is_available()
        self.ocr = easyocr.Reader(['ch_sim', 'en'], gpu=self._use_gpu, verbose=False)
        self._ocr_tra = None
        self.yolo = YOLO(model_path, verbose=False)
        self.t2s = opencc.OpenCC('t2s')
        with open(db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.skins = data["skins"]
        self._weapons_data = data.get("weapons", [])
        self._build_index()

    def _extract_series(self, skin):
        if skin["weapon"] == "近战武器":
            parts = skin["name"].rsplit(" ", 1)
            return parts[0] if len(parts) > 1 else skin["name"]
        return skin["name"].replace(f" {skin['weapon']}", "").replace(skin["weapon"], "").strip()

    def _extract_series_tw(self, skin):
        """提取繁体系列名（t2s转换后）"""
        name_tw = skin.get("name_tw", "")
        if not name_tw:
            return None
        name_tw = self.t2s.convert(name_tw)
        weapon_tw = skin.get("_weapon_tw_s", skin["weapon"])
        if skin["weapon"] == "近战武器":
            parts = name_tw.rsplit(" ", 1)
            return parts[0] if len(parts) > 1 else name_tw
        return name_tw.replace(f" {weapon_tw}", "").replace(weapon_tw, "").strip()

    def _build_index(self):
        self.skins_by_weapon = {}
        self.skins_by_series = {}
        self.skins_by_series_tw = {}
        self.melee_by_series = {}
        # 预计算繁体武器名(t2s)
        weapon_tw_map = {}
        for w in getattr(self, '_weapons_data', []):
            if w.get("name_tw"):
                weapon_tw_map[w["uuid"]] = self.t2s.convert(w["name_tw"])
        for s in self.skins:
            s["_weapon_tw_s"] = weapon_tw_map.get(s.get("weapon_uuid", ""), s["weapon"])
            self.skins_by_weapon.setdefault(s["weapon"], []).append(s)
            series = self._extract_series(s)
            if series:
                self.skins_by_series.setdefault(series, []).append(s)
                if s["weapon"] == "近战武器":
                    self.melee_by_series.setdefault(series, []).append(s)
            # 繁体系列索引
            series_tw = self._extract_series_tw(s)
            if series_tw and series_tw != series:
                self.skins_by_series_tw.setdefault(series_tw, []).append(s)
        self.skin_name_map = {s["name"]: s for s in self.skins}

    def detect_cards(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        row_std = np.std(gray, axis=1)
        row_maxs = np.max(gray, axis=1)
        dark_rows = np.where((row_std < 8) & (row_maxs < 40))[0]
        all_row_seps = self._find_separators(dark_rows)
        thin_seps = [s for s in all_row_seps if s[1] - s[0] <= 20]
        # 基于行间距规律外推缺失的分隔线（自适应任意分辨率）
        if len(thin_seps) >= 3:
            gaps = [thin_seps[i+1][0] - thin_seps[i][1] for i in range(len(thin_seps)-1)]
            # 取众数间距（最常见的行高）作为标准间距
            gap_rounded = [round(g / 10) * 10 for g in gaps]
            common_gap = Counter(gap_rounded).most_common(1)[0][0]
            tolerance = common_gap * 0.15
            std_gaps = [g for g in gaps if abs(g - common_gap) < tolerance]
            if std_gaps:
                avg_gap = int(np.mean(std_gaps))
                avg_thick = int(np.mean([s[1]-s[0] for s in thin_seps]))
                # 仅向下外推：在最后一条thin_sep之后补缺失行（不向上外推，避免侵入header区）
                y = thin_seps[-1][1] + avg_gap
                while y + avg_thick < h:
                    virtual = (y, y + avg_thick)
                    thin_seps.append(virtual)
                    all_row_seps.append(virtual)
                    y += avg_gap + avg_thick
                all_row_seps.sort(key=lambda s: s[0])
        # 检测繁体版交替间距模式(大间距=武器区,小间距=文字区)
        if len(thin_seps) >= 3:
            gaps = [thin_seps[i+1][0] - thin_seps[i][1] for i in range(len(thin_seps)-1)]
            small_gaps = sum(1 for g in gaps if g < 50)
            if small_gaps > len(gaps) * 0.3:
                # 繁体模式：保留后面跟着大间距的分隔线（卡片起始位置）
                card_row_seps = []
                for i in range(len(thin_seps) - 1):
                    if thin_seps[i+1][0] - thin_seps[i][1] > 80:
                        card_row_seps.append(thin_seps[i])
                card_row_seps.append(thin_seps[-1])
            else:
                card_row_seps = thin_seps
        else:
            card_row_seps = thin_seps

        if len(card_row_seps) >= 2:
            sy1, sy2 = card_row_seps[0][1] + 5, card_row_seps[1][0] - 5
        elif len(card_row_seps) == 1:
            sy1, sy2 = card_row_seps[0][1] + 5, min(card_row_seps[0][1] + 205, h)
        else:
            sy1, sy2 = h // 4, h // 2

        col_std = np.std(gray[sy1:sy2, :], axis=0)
        col_maxs = np.max(gray[sy1:sy2, :], axis=0)
        dark_cols = np.where((col_std < 8) & (col_maxs < 40))[0]
        col_seps = [s for s in self._find_separators(dark_cols) if s[1] - s[0] >= 2]

        row_ranges = []
        header_seps = [s for s in all_row_seps
                       if s[1] - s[0] > 20 and s[1] < (card_row_seps[0][0] if card_row_seps else h)]
        if header_seps and card_row_seps:
            fs, fe = header_seps[-1][1] + 1, card_row_seps[0][0]
            if fe - fs > 50:
                row_ranges.append((fs, fe))
        for i in range(len(card_row_seps) - 1):
            s, e = card_row_seps[i][1] + 1, card_row_seps[i + 1][0]
            if e - s > 50:
                row_ranges.append((s, e))
        if card_row_seps:
            ls = card_row_seps[-1][1] + 1
            # 用已检测行的平均高度估算尾部行范围
            avg_row_h = int(np.mean([e - s for s, e in row_ranges])) if row_ranges else 200
            tail = [s for s in all_row_seps if s[0] > card_row_seps[-1][1] and s[1] - s[0] > 20]
            le = tail[0][0] if tail else min(ls + avg_row_h, h)
            if le - ls > 50:
                row_ranges.append((ls, le))

        col_ranges = self._sep_to_ranges(col_seps, w)

        # 后处理：用THICK分隔线修正异常行范围（繁体版分类边界处）
        thick_seps = [s for s in all_row_seps if s[1] - s[0] > 20]
        filtered_rows = []
        for y1, y2 in row_ranges:
            blockers = [s for s in thick_seps if y1 < s[0] < y2]
            if blockers:
                new_y2 = blockers[0][0]
                if new_y2 - y1 > 50:
                    filtered_rows.append((y1, new_y2))
                last_blocker_end = blockers[-1][1] + 1
                if y2 - last_blocker_end > 50:
                    filtered_rows.append((last_blocker_end, y2))
            else:
                filtered_rows.append((y1, y2))
        # 扩展过短范围（缺失文字区的卡片，向下补齐到下一个分隔线后）
        extended_rows = []
        for y1, y2 in filtered_rows:
            if y2 - y1 < 160:
                next_seps = [s for s in all_row_seps if s[0] >= y2]
                if len(next_seps) >= 2:
                    y2 = next_seps[1][0]
                elif next_seps:
                    y2 = min(next_seps[0][1] + 40, h)
            if y2 - y1 > 50:
                extended_rows.append((y1, y2))
        row_ranges = extended_rows

        cards = []
        for y1, y2 in row_ranges:
            for x1, x2 in col_ranges:
                if x2 - x1 < 50:
                    continue
                if np.std(gray[y1:y2, x1:x2]) > 5:
                    cards.append((x1, y1, x2 - x1, y2 - y1))
        cards.sort(key=lambda c: (c[1], c[0]))
        return cards

    @property
    def ocr_tra(self):
        if self._ocr_tra is None:
            self._ocr_tra = easyocr.Reader(['ch_tra', 'en'], gpu=self._use_gpu, verbose=False)
        return self._ocr_tra

    def _find_text_region(self, card_img):
        """智能检测卡片内文字区域。返回 (text_region, is_traditional)"""
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY) if len(card_img.shape) == 3 else card_img
        h = gray.shape[0]
        row_maxs = np.max(gray, axis=1)
        row_stds = np.std(gray, axis=1)
        # 在卡片下半部找内部暗色分隔线(std<3, max<30)
        for y in range(h // 2, h):
            if row_maxs[y] < 30 and row_stds[y] < 3:
                sep_end = y
                while sep_end < h and row_maxs[sep_end] < 40:
                    sep_end += 1
                if sep_end < h:
                    return card_img[sep_end:, :], True
                break
        return card_img[int(h * 0.45):, :], False

    def _find_separators(self, dark_indices):
        if len(dark_indices) == 0:
            return []
        segments, start = [], dark_indices[0]
        for i in range(1, len(dark_indices)):
            if dark_indices[i] - dark_indices[i-1] > 5:
                segments.append((start, dark_indices[i-1]))
                start = dark_indices[i]
        segments.append((start, dark_indices[-1]))
        return [s for s in segments if s[1] - s[0] >= 2]

    def _sep_to_ranges(self, seps, total_size):
        if not seps:
            return [(0, total_size)]
        ranges = []
        for i in range(len(seps) - 1):
            s, e = seps[i][1] + 1, seps[i+1][0]
            if e - s > 30:
                ranges.append((s, e))
        return ranges

    def extract_weapon_from_ocr(self, ocr_text):
        """从OCR文字中提取武器名。简体格式'系列 武器名'，繁体格式'武器|系列名'"""
        text = ocr_text.strip()
        # 处理pipe分隔格式（繁体版: "武器 | 系列"）
        if "|" in text:
            parts = text.split("|", 1)
            weapon_part = parts[0].strip()
            # 在pipe前部分查找武器名
            for wn in WEAPON_CN_NAMES:
                if wn == "近战武器":
                    continue
                if wn in weapon_part:
                    return wn
            for tw_name, cn_name in WEAPON_TW_TO_CN.items():
                if tw_name in weapon_part:
                    return cn_name
            for kw in MELEE_KEYWORDS:
                if kw in weapon_part:
                    return "近战武器"
            if "近战" in weapon_part or "近戰" in weapon_part:
                return "近战武器"
        # 直接匹配已知武器名（从末尾开始找）
        for wn in WEAPON_CN_NAMES:
            if wn == "近战武器":
                continue
            if text.endswith(wn):
                return wn
        # 匹配繁体武器名
        for tw_name, cn_name in WEAPON_TW_TO_CN.items():
            if tw_name in text:
                return cn_name
        # 检查近战武器关键词
        for kw in MELEE_KEYWORDS:
            if kw in text:
                return "近战武器"
        # 空格分割后检查最后一个词
        parts = text.replace("  ", " ").split(" ")
        if len(parts) >= 2:
            last = parts[-1]
            for wn in WEAPON_CN_NAMES:
                if wn == "近战武器":
                    continue
                if SequenceMatcher(None, last, wn).ratio() > 0.7:
                    return wn
        return None

    def match_skin(self, weapon_cn, ocr_text, exclude_names=None, card_img=None):
        """匹配皮肤。返回 (skin_dict, score, is_series_only)"""
        if not ocr_text.strip():
            return None, 0, False

        # 有武器名 → 在该武器的皮肤中精确匹配
        if weapon_cn:
            candidates = self.skins_by_weapon.get(weapon_cn, [])
            if exclude_names:
                candidates = [s for s in candidates if s["name"] not in exclude_names]
            best, score = self._best_match(candidates, ocr_text)
            if best and score > 0.45:
                return best, score, False

        # 系列匹配：同时搜索简体和繁体系列索引
        series_scores = []  # [(score, series_name, skins_list)]
        all_series = list(self.skins_by_series.items()) + list(self.skins_by_series_tw.items())
        text = ocr_text.strip()
        for series, skins in all_series:
            ratio = SequenceMatcher(None, text, series).ratio()
            # 子串匹配提升：要求长度差不超过50%，避免短系列名误匹配长文本
            if text == series:
                ratio = max(ratio, 0.95)
            elif (text in series or series in text) and min(len(text), len(series)) > max(len(text), len(series)) * 0.5:
                ratio = max(ratio, 0.85)
            if ratio > 0.7:
                series_scores.append((ratio, series, skins))

        if series_scores:
            series_scores.sort(key=lambda x: -x[0])
            best_score = series_scores[0][0]
            # 收集得分与最高分相差 ≤0.05 的所有候选皮肤（年份歧义时用 pHash 区分）
            tied_skins = []
            for score, _, skins in series_scores:
                if best_score - score <= 0.05:
                    tied_skins.extend([s for s in skins if s["tier"] in TARGET_TIERS])
            if tied_skins:
                # 去重（同一皮肤可能出现在简体+繁体两个索引中）
                seen = set()
                unique_tied = []
                for s in tied_skins:
                    if s["uuid"] not in seen:
                        seen.add(s["uuid"])
                        unique_tied.append(s)
                # 返回第一个候选，并把完整候选列表附在 skin 上供识别循环做年份消歧
                chosen = unique_tied[0]
                chosen["_year_candidates"] = unique_tied
                return chosen, best_score, True

        # 兜底：全量搜索（含繁体名），要求更高分数避免误匹配
        best, score = self._best_match(self.skins, ocr_text)
        if score < 0.7:
            return None, 0, False
        return best, score, False

    def _best_match(self, candidates, ocr_text):
        best_match, best_score = None, 0
        for skin in candidates:
            name = skin["name"]
            series = self._extract_series(skin)
            # 也比较繁体名(t2s后)
            name_tw = self.t2s.convert(skin.get("name_tw", "")) if skin.get("name_tw") else ""
            series_tw = self._extract_series_tw(skin) or ""
            scores = [
                SequenceMatcher(None, ocr_text, name).ratio(),
                SequenceMatcher(None, ocr_text, series).ratio() if series else 0,
                SequenceMatcher(None, ocr_text, name_tw).ratio() if name_tw else 0,
                SequenceMatcher(None, ocr_text, series_tw).ratio() if series_tw else 0,
            ]
            score = max(scores)
            all_names = [name, series, name_tw, series_tw]
            if any(n and (n.startswith(ocr_text) or n.endswith(ocr_text)) for n in all_names):
                score = max(score, 0.8 + len(ocr_text) / max(len(name), 1) * 0.2)
            if len(ocr_text) > 3:
                for n in all_names:
                    if n and (ocr_text in n or n in ocr_text) and min(len(ocr_text), len(n)) > max(len(ocr_text), len(n)) * 0.5:
                        score = max(score, 0.85)
                        break
            if score > best_score:
                best_score = score
                best_match = skin
        return best_match, best_score

    def recognize_screenshot(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图片: {img_path}")
            return []

        cards = self.detect_cards(img)
        print(f"检测到 {len(cards)} 个卡片区域")

        results = []
        seen_skins = {}  # skin_name → count, 用于去重统计
        series_counts = {}  # series_name → count, 系列匹配计数

        # 用第一张卡片检测布局类型（繁体/简体）
        use_tra = False
        if cards:
            x0, y0, w0, h0 = cards[0]
            _, use_tra = self._find_text_region(img[y0:y0+h0, x0:x0+w0])
            if use_tra:
                print("检测到繁体版布局，使用繁体OCR")

        for i, (x, y, w, h) in enumerate(cards):
            card_img = img[y:y+h, x:x+w]
            text_region, _ = self._find_text_region(card_img)
            ocr_result = self.ocr_text(text_region, use_tra=use_tra)
            # OCR为空时用另一种语言重试（处理简繁混合布局）
            if not ocr_result.strip():
                ocr_result = self.ocr_text(text_region, use_tra=not use_tra)

            weapon_cn = self.extract_weapon_from_ocr(ocr_result)
            # 繁体pipe格式 "武器|系列名" → 提取系列名用于匹配
            match_text = ocr_result
            if "|" in ocr_result:
                match_text = ocr_result.split("|", 1)[1].strip()

            # 匹配皮肤，如果最佳匹配已被使用则排除后重试
            skin, match_score, is_series_only = self.match_skin(weapon_cn, match_text, card_img=card_img)
            if skin and skin["name"] in seen_skins:
                skin, match_score, is_series_only = self.match_skin(
                    weapon_cn, match_text, exclude_names=set(seen_skins.keys()), card_img=card_img)

            if not skin or skin["tier"] not in TARGET_TIERS:
                continue
            if match_score < 0.4:
                continue

            yolo_topk = []
            yolo_weapon, yolo_conf = (None, 0)
            final_weapon = weapon_cn

            skin_name = skin["name"]
            series = self._extract_series(skin)

            if is_series_only:
                # 若有年份歧义候选列表（多年份同系列），直接使用
                year_candidates = skin.pop("_year_candidates", None)
                series_skins = [s for s in self.skins_by_series.get(series, [])
                                if s["tier"] in TARGET_TIERS]
                # 也查繁体系列索引
                if not series_skins:
                    series_skins = [s for s in self.skins_by_series_tw.get(series, [])
                                    if s["tier"] in TARGET_TIERS]
                # 有年份歧义时：series_skins 扩展为所有年份候选
                if year_candidates and len(year_candidates) > len(series_skins):
                    series_skins = year_candidates
                if len(series_skins) == 1:
                    skin = series_skins[0]
                    skin_name = skin["name"]
                    final_weapon = skin["weapon"]
                    display_name = skin_name
                    is_series_only = False  # 单武器系列视为精确匹配
                else:
                    # 先查近战数据库：如果系列有对应的刀，直接用YOLO判断是不是刀
                    # 搜索基础系列及其子系列的所有近战皮肤
                    melee_list = []
                    for s_key, s_melees in self.melee_by_series.items():
                        if s_key == series or s_key.startswith(series):
                            melee_list.extend(s_melees)
                    if melee_list and not weapon_cn:
                        yolo_topk = self.classify_weapon_topk(card_img)
                        if yolo_topk and yolo_topk[0][0] == "近战武器" and yolo_topk[0][1] > 0.5:
                            # 从未使用的近战皮肤中选一个
                            chosen = next((m for m in melee_list if m["name"] not in seen_skins), None)
                            if chosen:
                                skin = chosen
                                skin_name = skin["name"]
                                final_weapon = "近战武器"
                                yolo_weapon, yolo_conf = yolo_topk[0]
                                display_name = skin_name
                                seen_skins[skin_name] = 1
                                results.append({
                                    "index": len(results) + 1,
                                    "weapon": final_weapon,
                                    "ocr_text": ocr_result,
                                    "skin_name": display_name,
                                    "skin_tier": skin["tier"],
                                    "name_tw": skin.get("name_tw", ""),
                                    "match_score": match_score,
                                    "is_series_only": is_series_only,
                                    "yolo_weapon": yolo_weapon,
                                    "yolo_conf": yolo_conf,
                                })
                                continue
                    # 非近战：用YOLO识别枪械类型
                    if not weapon_cn:
                        yolo_topk = self.classify_weapon_topk(card_img)
                        if yolo_topk:
                            yolo_weapon, yolo_conf = yolo_topk[0]
                    # top-k 与系列候选武器交叉匹配
                    candidate_weapons = {s["weapon"] for s in series_skins}
                    matched = None
                    matched_conf = 0
                    MIN_CROSSMATCH_CONF = 0.01
                    for yw, yc in yolo_topk:
                        if yc < MIN_CROSSMATCH_CONF:
                            break
                        hit = next((s for s in series_skins if s["weapon"] == yw), None)
                        if hit:
                            matched, matched_conf = hit, yc
                            break
                    if matched:
                        skin = matched
                        skin_name = skin["name"]
                        final_weapon = skin["weapon"]
                        yolo_weapon, yolo_conf = final_weapon, matched_conf
                        # 若同武器有多个年份候选，用 pHash 进一步区分
                        same_weapon_skins = [s for s in series_skins if s["weapon"] == final_weapon]
                        if len(same_weapon_skins) > 1:
                            phash_winner = self._phash_match(card_img, same_weapon_skins)
                            if phash_winner:
                                skin = phash_winner
                                skin_name = skin["name"]
                                display_name = skin_name
                            else:
                                display_name = skin_name
                        else:
                            display_name = skin_name
                    else:
                        # 尝试用 pHash 从候选中选最相似的
                        phash_matched = self._phash_match(card_img, series_skins)
                        if phash_matched:
                            skin = phash_matched
                            skin_name = skin["name"]
                            final_weapon = skin["weapon"]
                            display_name = skin_name
                        else:
                            idx = series_counts.get(series, 0)
                            series_counts[series] = idx + 1
                            if idx < len(series_skins):
                                skin = series_skins[idx]
                                skin_name = skin["name"]
                            display_name = f"{series} ({final_weapon or skin['weapon']}?)"
            else:
                display_name = skin_name

            # YOLO验证：匹配分不够高时，用YOLO确认/修正武器类型
            if match_score < 0.9 and not yolo_topk:
                yolo_topk = self.classify_weapon_topk(card_img)
                if yolo_topk:
                    yolo_weapon, yolo_conf = yolo_topk[0]
            # 如果YOLO识别的武器与当前匹配不同，尝试在同系列中找到正确的皮肤
            if yolo_topk and yolo_weapon and yolo_weapon != skin["weapon"]:
                series_key = self._extract_series(skin)
                all_series_skins = self.skins_by_series.get(series_key, [])
                if not all_series_skins:
                    # 查繁体系列索引
                    stw = self._extract_series_tw(skin)
                    if stw:
                        all_series_skins = self.skins_by_series_tw.get(stw, [])
                for yw, yc in yolo_topk:
                    hit = next((s for s in all_series_skins
                                if s["weapon"] == yw and s["tier"] in TARGET_TIERS
                                and s["name"] not in seen_skins), None)
                    if hit:
                        skin = hit
                        skin_name = skin["name"]
                        display_name = skin_name
                        final_weapon = skin["weapon"]
                        yolo_weapon, yolo_conf = yw, yc
                        break

            # 统一去重：所有路径解析出的皮肤名都查重
            resolved_name = skin["name"]
            if resolved_name in seen_skins:
                continue
            seen_skins[resolved_name] = 1

            results.append({
                "index": len(results) + 1,
                "weapon": final_weapon or "未知",
                "ocr_text": ocr_result,
                "skin_name": display_name,
                "skin_tier": skin["tier"],
                "name_tw": skin.get("name_tw", ""),
                "match_score": match_score,
                "is_series_only": is_series_only,
                "yolo_weapon": yolo_weapon,
                "yolo_conf": yolo_conf,
            })
        return results

    @staticmethod
    def _compute_phash(img, hash_size=8):
        """对 BGR 图像计算 pHash，返回整数"""
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
        return val

    @staticmethod
    def _hamming(a, b):
        return bin(a ^ b).count('1')

    @staticmethod
    def _hu_vec(img):
        """提取图像的 Hu Moments 特征向量（对数尺度，用于形状比对）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        m = cv2.moments(thresh)
        hu = cv2.HuMoments(m).flatten()
        return -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    def _phash_match(self, card_img, candidates):
        """从候选皮肤中用 pHash 选汉明距离最小的（用于年份歧义消歧）。
        无论枪类还是近战，都取距离最小的候选（总比随机选好）。
        """
        phash_candidates = [s for s in candidates if s.get("phash")]
        if not phash_candidates:
            return None
        h = card_img.shape[0]
        weapon_region = card_img[:int(h * 0.60), :]
        card_hash = self._compute_phash(weapon_region)
        best, best_dist = None, 999
        for s in phash_candidates:
            try:
                db_hash = int(s["phash"], 16)
                dist = self._hamming(card_hash, db_hash)
                if dist < best_dist:
                    best_dist = dist
                    best = s
            except (ValueError, TypeError):
                continue
        return best

    def classify_weapon(self, card_img):
        """用YOLO模型对卡片武器图片区域分类，返回 (武器中文名, 置信度)"""
        h = card_img.shape[0]
        weapon_region = card_img[:int(h * 0.45), :]
        results = self.yolo(weapon_region, verbose=False)
        if not results or not results[0].probs:
            return None, 0
        probs = results[0].probs
        pred_en = results[0].names[probs.top5[0]]
        conf = probs.top5conf.tolist()[0]
        return YOLO_EN_TO_CN.get(pred_en, pred_en), conf

    def classify_weapon_topk(self, card_img, k=5):
        """返回YOLO top-k分类结果列表 [(武器中文名, 置信度), ...]"""
        h = card_img.shape[0]
        weapon_region = card_img[:int(h * 0.45), :]
        results = self.yolo(weapon_region, verbose=False)
        if not results or not results[0].probs:
            return []
        probs = results[0].probs
        names = results[0].names
        return [(YOLO_EN_TO_CN.get(names[idx], names[idx]), conf)
                for idx, conf in zip(probs.top5[:k], probs.top5conf.tolist()[:k])]

    def ocr_text(self, card_img, use_tra=False):
        h, w = card_img.shape[:2]
        if h < 40:
            scale = 3
            card_img = cv2.resize(card_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
        elif h < 60 or w < 120:
            card_img = cv2.resize(card_img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        reader = self.ocr_tra if use_tra else self.ocr
        min_conf = 0.10
        result = reader.readtext(card_img)
        texts = [txt for _, txt, conf in result if conf > min_conf]
        raw = " ".join(texts)
        return self.t2s.convert(raw)

    def print_results(self, results):
        TIER_COLORS = {"至尊": "🔴", "至臻": "🟡", "尊享": "🟣"}
        exact = [r for r in results if not r.get("is_series_only")]
        series = [r for r in results if r.get("is_series_only")]
        print(f"\n{'='*70}")
        print(f"识别结果 (共{len(results)}个高品质皮肤: {len(exact)}个精确匹配, {len(series)}个系列匹配)")
        print(f"{'='*70}")
        for r in results:
            icon = TIER_COLORS.get(r['skin_tier'], "")
            flag = " ⚠️" if r.get("is_series_only") else ""
            yolo_info = f" [YOLO:{r['yolo_weapon']}({r['yolo_conf']:.0%})]" if r.get("is_series_only") and r.get("yolo_weapon") else ""
            tw_info = f" ({r['name_tw']})" if r.get('name_tw') and r['name_tw'] != r['skin_name'] else ""
            print(f"  #{r['index']:02d} {icon}[{r['skin_tier']}] {r['skin_name']}{tw_info}"
                  f" | {r['weapon']} | {r['match_score']:.0%}{flag}{yolo_info}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python skin_recognizer.py <截图路径>")
        sys.exit(1)
    recognizer = SkinRecognizer()
    results = recognizer.recognize_screenshot(sys.argv[1])
    recognizer.print_results(results)
