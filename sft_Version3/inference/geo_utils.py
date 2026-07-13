#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
geo_utils: 收藏坐标 -> 邻域合法 geohash 集合（供 generate.py 的缓存增广用）。

用 pygeohash 保证与 item map 建表时同一套编码（prepare_rqvae_data/.../step1_get_item.py
的 compute_geohash 用的正是它，precision=6），字符串必然与词表里的 geo token 对齐。

算法（网格扫描 + 真实距离过滤，等价于环形扩展但不依赖 pygeohash 版本相关的
neighbor/get_adjacent API，只用最稳定的 encode/decode_exactly）：
  以收藏坐标为中心，按该 precision 的格子边长（decode_exactly 给出的半宽/半高）
  在经纬度网格上扫描，每个格点编码回 geohash，用 haversine 实际距离 <= radius_km
  才保留——比单纯"8 邻居"更准确覆盖圆形邻域，且在高纬度地区不失真。
"""

import math

try:
    import pygeohash as pgh
except ImportError:
    pgh = None

EARTH_RADIUS_KM = 6371.0088


def _require_pgh():
    if pgh is None:
        raise ImportError("需要 pygeohash（pip install pygeohash），"
                          "用于收藏坐标 -> geohash 编码，须与建 item map 时同一套算法")


def geohash_precision_from_vocab(tokenizer) -> int:
    """从词表里任意一个真实 geo token 反推 geohash 位数（去尖括号后的字符串长度）。"""
    sample = tokenizer.level_token_ids[0][0]
    tok_str = tokenizer.id2token[sample]
    return len(tok_str) - 2                     # 去掉 '<' '>'


def encode(lat: float, lng: float, precision: int) -> str:
    _require_pgh()
    return pgh.encode(lat, lng, precision=precision)


def parse_favor_coords(raw, topk: int) -> list:
    """'lng@lat^lng@lat^...' -> [(lng, lat), ...]（最多 topk 条，跳过非法/空值）。"""
    if not raw:
        return []
    out = []
    for part in str(raw).split("^"):
        part = part.strip()
        if not part or "@" not in part:
            continue
        lng_s, _, lat_s = part.partition("@")
        try:
            out.append((float(lng_s), float(lat_s)))
        except ValueError:
            continue
        if len(out) >= topk:
            break
    return out


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def neighbor_geohash_set(lat: float, lng: float, radius_km: float, precision: int) -> set:
    """以 (lat,lng) 为圆心、radius_km 内、precision 位的 geohash 集合（未与词表求交，
       调用方需再与真实存在的 geo token 集合取交集，得到"合法"邻域）。"""
    _require_pgh()
    center_hash = pgh.encode(lat, lng, precision=precision)
    _, _, lat_err, lng_err = pgh.decode_exactly(center_hash)
    step_lat, step_lng = lat_err * 2, lng_err * 2   # 一个格子的高/宽（度）

    km_per_deg_lat = 111.32
    km_per_deg_lng = 111.32 * max(math.cos(math.radians(lat)), 1e-6)
    n_lat = int(math.ceil(radius_km / (step_lat * km_per_deg_lat))) + 1
    n_lng = int(math.ceil(radius_km / (step_lng * km_per_deg_lng))) + 1

    out = set()
    for i in range(-n_lat, n_lat + 1):
        for j in range(-n_lng, n_lng + 1):
            plat = lat + i * step_lat
            plng = lng + j * step_lng
            if haversine_km(lat, lng, plat, plng) <= radius_km:
                out.add(pgh.encode(plat, plng, precision=precision))
    return out
