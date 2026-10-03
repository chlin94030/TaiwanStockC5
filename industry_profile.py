"""Plain-language fine-industry labels for Taiwan Alpha Radar.

These labels are model-facing, human-readable business positioning tags. They are
not official TWSE/TPEx industry classifications. Curated labels take precedence;
unknown names fall back to a conservative broad-industry label.
"""
from __future__ import annotations


def _code(ticker: str) -> str:
    return str(ticker or "").split(".")[0].strip()


# Curated labels for frequently followed Taiwan names.  Keep labels concise and
# understandable to non-specialists; avoid claims that need live verification.
FINE_INDUSTRY_BY_CODE = {
    # Semiconductor / IC
    "2330": "半導體－晶圓代工龍頭",
    "2303": "半導體－成熟製程晶圓代工",
    "2454": "半導體－手機/邊緣AI晶片設計",
    "3034": "半導體－顯示驅動IC",
    "3661": "半導體－ASIC客製化晶片",
    "2379": "半導體－高速傳輸IC",
    "3443": "半導體－高速傳輸/介面IC",
    "6531": "半導體－高速傳輸IC",
    "3711": "半導體－封裝測試龍頭",
    "2449": "半導體－IC測試",
    "6239": "半導體－先進封測設備",
    "3131": "半導體設備－濕製程/先進製程",
    "3583": "半導體設備－設備整合/再生晶圓",
    "6187": "半導體設備－封裝自動化",
    "1560": "半導體設備－切割/研磨設備",
    "6643": "半導體設備－廠務工程",
    "2408": "半導體－記憶體",
    "2344": "半導體－記憶體",

    # AI server / EMS / PC
    "2317": "電子中游－EMS/垂直整合",
    "2382": "電子中游－ODM/AI伺服器",
    "3231": "電子中游－ODM/AI伺服器",
    "6669": "電子中游－AI伺服器ODM",
    "2357": "電子中游－PC/伺服器品牌",
    "2376": "電子中游－主機板/AI伺服器",
    "4938": "電子中游－EMS/系統組裝",
    "2324": "電子中游－ODM/系統組裝",
    "2356": "電子中游－ODM/系統組裝",

    # Cooling / power / mechanical
    "2308": "電子中游－電源與冷卻系統",
    "3017": "電子中游－水冷散熱",
    "3324": "電子中游－水冷散熱",
    "3653": "電子中游－散熱模組",
    "2421": "電子中游－風扇/散熱",
    "2059": "電子零組件－伺服器滑軌",
    "3533": "電子零組件－CPU插槽/連接器",
    "2327": "電子零組件－被動元件",
    "2492": "電子零組件－連接器",

    # PCB / CCL
    "2383": "電子材料－高速CCL",
    "6274": "電子材料－高速CCL",
    "2368": "電子中游－伺服器PCB",
    "3037": "電子中游－PCB/載板",
    "8046": "電子中游－PCB",
    "3189": "電子材料－銅箔/高階材料",

    # Optics / networking
    "3008": "光學－高階鏡頭",
    "2345": "網通－高速交換器",
    "6285": "網通－企業/電信設備",

    # Heavy electric / green energy
    "1519": "重電綠能－變壓器外銷",
    "1503": "重電綠能－變壓器/配電設備",
    "1513": "重電綠能－重電設備",
    "1609": "重電綠能－電線電纜/能源工程",
    "1504": "重電綠能－機電系統",

    # Finance
    "2881": "金融－金控/壽險銀行",
    "2882": "金融－金控/銀行保險",
    "2891": "金融－金控/銀行",
    "2886": "金融－金控/銀行",
    "2884": "金融－金控/銀行",
    "5871": "金融－租賃/企業金融",

    # Traditional / others
    "1301": "塑化－石化上游",
    "1303": "塑化－石化/電子材料",
    "2002": "鋼鐵－一貫鋼廠",
    "2603": "航運－貨櫃航運",
    "2609": "航運－貨櫃航運",
    "2615": "航運－貨櫃航運",
}


BROAD_FALLBACKS = {
    "水泥工業": "原物料－水泥", "食品工業": "民生消費－食品", "食品": "民生消費－食品",
    "塑膠工業": "原物料－塑化", "塑膠": "原物料－塑化", "紡織纖維": "傳產－紡織",
    "電機機械": "重電綠能－電機設備", "電機機械業": "重電綠能－電機設備",
    "電器電纜": "重電綠能－電線電纜", "玻璃陶瓷": "原物料－玻璃陶瓷", "造紙工業": "原物料－造紙",
    "鋼鐵工業": "原物料－鋼鐵", "鋼鐵": "原物料－鋼鐵", "橡膠工業": "原物料－橡膠",
    "汽車工業": "汽車－整車/零組件", "建材營造": "資產營建－建材營造", "航運業": "運輸－航運", "航運": "運輸－航運",
    "觀光餐旅": "民生消費－觀光餐旅", "金融保險": "金融－金控/銀行/保險", "金融保險業": "金融－金控/銀行/保險",
    "貿易百貨": "民生消費－通路百貨", "綜合": "綜合產業", "其他": "其他產業",
    "化學工業": "原物料－化工", "化學": "原物料－化工", "生技醫療業": "生技醫療－產品/服務", "生技醫療": "生技醫療－產品/服務",
    "油電燃氣業": "公用能源－油電燃氣", "半導體": "半導體－產業鏈", "半導體業": "半導體－產業鏈",
    "電腦及週邊": "電子中游－電腦/伺服器", "電腦及週邊設備業": "電子中游－電腦/伺服器",
    "光電": "電子中游－光電", "光電業": "電子中游－光電", "通信網路": "電子中游－網通設備", "通信網路業": "電子中游－網通設備",
    "電子零組件": "電子中游－零組件", "電子零組件業": "電子中游－零組件", "電子通路業": "電子中游－電子通路",
    "資訊服務業": "數位服務－資訊服務", "其他電子": "電子中游－電子製造/設備", "其他電子業": "電子中游－電子製造/設備",
    "文化創意業": "消費服務－文化創意", "農業科技業": "民生產業－農業科技", "綠能環保": "綠能環保－設備/服務",
    "數位雲端": "數位服務－雲端/平台", "運動休閒": "民生消費－運動休閒", "居家生活": "民生消費－居家生活",
    "管理股票": "其他－管理股票",
}


def fine_industry(ticker: str, name: str = "", broad_industry: str = "") -> str:
    """Return a concise, plain-language business positioning label."""
    code = _code(ticker)
    if code in FINE_INDUSTRY_BY_CODE:
        return FINE_INDUSTRY_BY_CODE[code]

    broad = str(broad_industry or "").strip()
    if broad in BROAD_FALLBACKS:
        return BROAD_FALLBACKS[broad]

    # Conservative keyword fallbacks.  These are intentionally broad.
    if "半導體" in broad:
        return "半導體－產業鏈"
    if "電子" in broad:
        return "電子產業－供應鏈"
    if "金融" in broad or "銀行" in broad or "保險" in broad:
        return "金融－金控/銀行/保險"
    if "電機" in broad or "電纜" in broad:
        return "重電綠能－電機設備"
    if "航運" in broad:
        return "運輸－航運"
    if "食品" in broad:
        return "民生消費－食品"
    return broad or "產業定位待補"


def profile_note() -> str:
    return "細產業為模型用通俗定位，非交易所正式產業分類。"
