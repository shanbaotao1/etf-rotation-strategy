#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ETF动量轮动策略 · 数据计算脚本
用途：拉取数据、计算信号、输出 results.json
可在 GitHub Actions / 本地 / 服务器 运行
"""

import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np

try:
    import akshare as ak
except ImportError:
    print("请安装 akshare: pip install akshare")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, 'results.json')
CACHE_FILE = os.path.join(SCRIPT_DIR, 'kline_cache.json')

# ================== 策略参数 ==================
MOMENTUM_DAYS = 25
ALPHA = 0.08
MIN_SCORE = 0

SINGLE_HOLDING = True
USE_DEFENSIVE_ETF = True
DEFENSIVE_ETF = "511880.XSHG"
USE_WEAK_POOL_SWITCH = True
GLOBAL_POOL = ["513100.XSHG", "513520.XSHG", "513030.XSHG", "518880.XSHG",
               "159985.XSHE", "159980.XSHE", "511090.XSHG", "511260.XSHG", "563020.XSHG"]
WEAK_R2_THRESHOLD = 0.0
USE_DYNAMIC_WINDOW = True
DYNAMIC_WIN_SHORT = 23
DYNAMIC_WIN_LONG = MOMENTUM_DAYS
DYNAMIC_R2_QUALITY = 0.65
DYNAMIC_HYST = 2
R2_FILTER_THRESHOLD = 0.0

ETF_POOL = [
    "513100.XSHG", "513520.XSHG", "513030.XSHG", "513130.XSHG",
    "510180.XSHG", "159915.XSHE", "510500.XSHG",
    "588120.XSHG", "515070.XSHG", "512480.XSHG", "516510.XSHG", "159667.XSHE",
    "159755.XSHE", "516160.XSHG",
    "512710.XSHG", "159227.XSHE",
    "510410.XSHG", "159980.XSHE", "159985.XSHE",
    "512290.XSHG", "159992.XSHE",
    "159928.XSHE", "159865.XSHE",
    "512000.XSHG",
    "511090.XSHG", "511260.XSHG",
    "518880.XSHG",
    "563020.XSHG", "520810.XSHG",
]

ETF_NAMES = {
    "513100.XSHG": "纳指ETF", "513520.XSHG": "日经ETF", "513030.XSHG": "德国ETF",
    "513130.XSHG": "恒生科技ETF", "510180.XSHG": "上证180ETF", "159915.XSHE": "创业板ETF",
    "510500.XSHG": "中证500ETF", "588120.XSHG": "科创100ETF", "515070.XSHG": "人工智能ETF",
    "512480.XSHG": "半导体ETF", "516510.XSHG": "云计算ETF", "159667.XSHE": "工业母机ETF",
    "159755.XSHE": "电池ETF", "516160.XSHG": "新能源ETF", "512710.XSHG": "军工龙头ETF",
    "159227.XSHE": "航空航天ETF", "510410.XSHG": "资源ETF", "159980.XSHE": "有色ETF",
    "159985.XSHE": "豆粕ETF", "512290.XSHG": "生物医药ETF", "159992.XSHE": "创新药ETF",
    "159928.XSHE": "消费ETF", "159865.XSHE": "养殖ETF", "512000.XSHG": "证券ETF",
    "511090.XSHG": "30年国债ETF", "511260.XSHG": "10年国债ETF", "518880.XSHG": "黄金ETF",
    "563020.XSHG": "红利低波ETF", "520810.XSHG": "港股红利ETF", "511880.XSHG": "银华日利",
}

SINA_CODE_MAP = {}
for code in ETF_POOL + ([DEFENSIVE_ETF] if DEFENSIVE_ETF not in ETF_POOL else []):
    num = code.split('.')[0]
    if code.endswith('XSHG'):
        SINA_CODE_MAP[code] = f"sh{num}"
    else:
        SINA_CODE_MAP[code] = f"sz{num}"


def calculate_score(closes, current_price, m_days=25):
    if len(closes) < m_days:
        return 0, 0, 0
    prices = closes[-m_days:] + [current_price]
    y = np.log([max(p, 0.0001) for p in prices])
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))
    try:
        slope, intercept = np.polyfit(x, y, 1, w=weights)
    except:
        return 0, 0, 0
    ann = math.exp(slope * 250) - 1
    y_pred = slope * x + intercept
    ss_res = np.sum(weights * (y - y_pred) ** 2)
    ss_tot = np.sum(weights * (y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return ann, r2, ann * r2


def check_recent_drop(closes, days=3, threshold=0.95):
    if len(closes) < days + 1:
        return False
    for i in range(1, days + 1):
        if i >= len(closes) or closes[-i - 1] <= 0:
            continue
        if closes[-i] / closes[-i - 1] < threshold:
            return True
    return False


def fetch_klines():
    """获取所有ETF的K线数据"""
    today = datetime.now().strftime('%Y-%m-%d')

    # 读缓存
    cache = {"date": "", "data": {}}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
        except:
            pass

    if cache.get('date') == today and cache.get('data'):
        print(f"缓存可用 ({today})")
        return cache['data'], today

    print(f"获取 {len(ETF_POOL)} 只ETF K线...")
    results = {}
    all_codes = ETF_POOL + ([DEFENSIVE_ETF] if DEFENSIVE_ETF not in ETF_POOL else [])

    for i, code in enumerate(all_codes, 1):
        sina_code = SINA_CODE_MAP.get(code)
        if not sina_code:
            continue
        try:
            df = ak.fund_etf_hist_sina(symbol=sina_code)
            if df is None or df.empty:
                print(f"  [{i}/{len(all_codes)}] {code} ❌ 无数据")
                continue
            df = df.sort_values('date')
            klines = []
            for _, row in df.iterrows():
                close = float(row['close'])
                if close <= 0:
                    continue
                klines.append({'close': close, 'volume': float(row.get('volume', 0))})
            if len(klines) >= MOMENTUM_DAYS:
                results[code] = klines
                print(f"  [{i}/{len(all_codes)}] {code} ✅ {len(klines)}条")
            else:
                print(f"  [{i}/{len(all_codes)}] {code} ⚠️ 仅{len(klines)}条")
        except Exception as e:
            print(f"  [{i}/{len(all_codes)}] {code} ❌ {e}")
        time.sleep(0.2)

    cache = {'date': today, 'data': results}
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False)
    except:
        pass

    print(f"完成: {len(results)}/{len(all_codes)} 数据充足")
    return results, today


def get_regime(kline_data):
    """判断市场状态"""
    PROXIES = ['510300.XSHG', '159915.XSHE', '510500.XSHG']
    abv = 0; up = 0; blw = 0
    for pc in PROXIES:
        recs = kline_data.get(pc, [])
        closes = [r['close'] for r in recs if r.get('close', 0) > 0]
        if len(closes) >= 30:
            ma20 = sum(closes[-20:]) / 20
            ma20_p = sum(closes[-21:-1]) / 20
            if closes[-1] > ma20:
                abv += 1
                if ma20 > ma20_p * 1.001:
                    up += 1
            else:
                blw += 1
    if abv >= 2 and up >= 2:
        return 'bull', 7
    elif blw >= 2:
        return 'bear', 3
    else:
        return 'sideways', 5


def get_effective_window(kline_data):
    """动态动量窗口"""
    if not USE_DYNAMIC_WINDOW:
        return MOMENTUM_DAYS, 0

    r2s = []
    for code in ETF_POOL:
        recs = kline_data.get(code, [])
        if len(recs) < 26:
            continue
        closes = [float(k['close']) for k in recs if k.get('close', 0) > 0]
        if len(closes) < 26:
            continue
        _, r2, _ = calculate_score(closes, closes[-1], 25)
        if r2 > 0:
            r2s.append(r2)

    med = sorted(r2s)[len(r2s) // 2] if r2s else 0.0
    win = DYNAMIC_WIN_SHORT if med > DYNAMIC_R2_QUALITY else DYNAMIC_WIN_LONG
    return win, med


def filter_etfs(kline_data, prices, max_score, m_days, candidate_codes=None, r2_threshold=0.0):
    eligible = []
    excluded = []

    for etf in ETF_POOL:
        if candidate_codes is not None and etf not in candidate_codes:
            continue

        name = ETF_NAMES.get(etf, etf)
        price = prices.get(etf, 0)
        if price <= 0:
            recs = kline_data.get(etf)
            if recs:
                price = recs[-1].get('close', 0)

        klines = kline_data.get(etf)
        if not klines:
            excluded.append({'code': etf, 'name': name, 'reason': '无数据', 'score': 0})
            continue
        if len(klines) < m_days:
            excluded.append({'code': etf, 'name': name, 'reason': f'K线不足({len(klines)}/{m_days})', 'score': 0})
            continue
        if price <= 0:
            excluded.append({'code': etf, 'name': name, 'reason': '无价格', 'score': 0})
            continue

        closes = [float(k['close']) for k in klines if k.get('close', 0) > 0]
        if len(closes) < m_days:
            excluded.append({'code': etf, 'name': name, 'reason': f'收盘价不足', 'score': 0})
            continue

        ann, r2, score = calculate_score(closes, price, m_days)

        if score == 0:
            excluded.append({'code': etf, 'name': name, 'reason': '得分=0', 'score': 0, 'r2': r2, 'ann': ann})
        elif r2 <= 0:
            excluded.append({'code': etf, 'name': name, 'reason': f'R²≤0({r2:.4f})', 'score': score, 'r2': r2, 'ann': ann})
        elif r2_threshold > 0 and r2 < r2_threshold:
            excluded.append({'code': etf, 'name': name, 'reason': f'R²<{r2_threshold}', 'score': score, 'r2': r2, 'ann': ann})
        elif check_recent_drop(closes):
            excluded.append({'code': etf, 'name': name, 'reason': '近3日跌>5%', 'score': score, 'r2': r2, 'ann': ann})
        elif score <= MIN_SCORE:
            excluded.append({'code': etf, 'name': name, 'reason': '得分≤0', 'score': score, 'r2': r2, 'ann': ann})
        elif score >= max_score:
            excluded.append({'code': etf, 'name': name, 'reason': f'得分≥{max_score}', 'score': score, 'r2': r2, 'ann': ann})
        else:
            eligible.append({'code': etf, 'name': name, 'score': score, 'ann': ann, 'r2': r2, 'price': price})

    eligible.sort(key=lambda x: x['score'], reverse=True)
    return eligible, excluded


def main():
    print("=" * 50)
    print("ETF动量轮动 · 数据计算")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    t0 = time.time()

    # 1. 获取数据
    kline_data, data_date = fetch_klines()
    if not kline_data:
        print("错误: 无法获取任何ETF数据")
        sys.exit(1)

    # 2. 获取实时价格 (用收盘价近似)
    prices = {}
    for code, klines in kline_data.items():
        if klines:
            prices[code] = klines[-1].get('close', 0)

    # 3. 市场状态
    regime, dynamic_max = get_regime(kline_data)
    regime_map = {'bull': '🟢 牛市', 'bear': '🔴 熊市', 'sideways': '🟡 震荡市'}
    print(f"市场状态: {regime_map.get(regime, regime)} (上限{dynamic_max}分)")

    # 4. 动态窗口
    m_eff, r2_med = get_effective_window(kline_data)
    print(f"动量窗口: {m_eff}天 | R²中位数: {r2_med:.4f}")

    # 5. 候选池
    cand_codes = None
    r2_th = R2_FILTER_THRESHOLD
    use_weak = False
    if USE_WEAK_POOL_SWITCH and regime == 'bear':
        cand_codes = set(GLOBAL_POOL)
        r2_th = WEAK_R2_THRESHOLD
        use_weak = True
        print(f"熊市避险池: {len(GLOBAL_POOL)}只")

    # 6. 筛选
    eligible, excluded = filter_etfs(kline_data, prices, dynamic_max, m_eff,
                                     candidate_codes=cand_codes, r2_threshold=r2_th)
    print(f"入选: {len(eligible)}只 | 排除: {len(excluded)}只")

    if eligible:
        print(f"\n🏆 TOP 5:")
        for i, e in enumerate(eligible[:5]):
            print(f"  {i+1}. {e['name']}({e['code']}) 得分={e['score']:.4f} 年化={e['ann']*100:.1f}% R²={e['r2']:.3f}")

    # 7. 输出 JSON
    def fmt(etf):
        return {
            'code': etf['code'],
            'name': etf['name'],
            'score': round(etf['score'], 6),
            'ann': round(etf.get('ann', 0), 6),
            'r2': round(etf.get('r2', 0), 6),
            'price': round(etf.get('price', 0), 3),
        } if etf else None

    result = {
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_date': data_date,
        'regime': regime,
        'regime_label': regime_map.get(regime, regime),
        'max_score': dynamic_max,
        'momentum_days': m_eff,
        'r2_median': round(r2_med, 6),
        'use_weak_pool': use_weak,
        'global_pool_count': len(GLOBAL_POOL),
        'pool_count': len(ETF_POOL),
        'use_defensive': USE_DEFENSIVE_ETF,
        'price_type': '收盘价',
        'top_pick': fmt(eligible[0]) if eligible else None,
        'eligible': [fmt(e) for e in eligible],
        'excluded': [{'code': e['code'], 'name': e['name'],
                       'score': round(e.get('score', 0), 6), 'reason': e['reason']}
                      for e in excluded],
        'elapsed_sec': round(time.time() - t0, 1),
    }

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 结果已保存: {OUTPUT_FILE} ({time.time()-t0:.1f}秒)")


if __name__ == '__main__':
    main()
