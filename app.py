import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ===============================
# 페이지 설정
# ===============================
st.set_page_config(
    page_title="특판 모양지 오더 검토",
    layout="wide"
)

st.title("📊 특판 모양지 오더 검토")
st.caption("수주잔량 기준 · 재고 · PO · 포캐스트 통합 판단")

# ===============================
# 공통 유틸
# ===============================
def read_csv(file):
    df = pd.read_csv(file, encoding="cp949")
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df.columns = df.columns.str.strip()
    return df

def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(",", "").str.strip(),
        errors="coerce"
    ).fillna(0)

# ===============================
# 업로드
# ===============================
st.subheader("① 소스 파일 업로드 (CSV)")

files = st.file_uploader(
    "현재고 / PO / 수주예정등록 / 시판스펙관리 / 품목정보",
    type="csv",
    accept_multiple_files=True
)

if not files:
    st.stop()

data = {}

for f in files:
    df = read_csv(f)
    cols = " ".join(df.columns)

    if "재고수량" in cols:
        data["stock"] = df
    elif "PO" in cols or "잔량" in cols:
        data["po"] = df
    elif "세대당예상수량" in cols or "수주잔량" in cols:
        data["order"] = df
    elif "시판" in cols:
        data["market"] = df
    elif "상품명" in cols and "상품코드" in cols:
        data["item"] = df

required = ["stock", "po", "order", "item"]
if any(k not in data for k in required):
    st.error("❌ 필수 파일이 모두 인식되지 않았습니다.")
    st.stop()

# ===============================
# 기준 설정
# ===============================
st.subheader("② 기준 설정")

base_date = st.date_input("조회 기준일", value=datetime.today())

period_type = st.selectbox(
    "포캐스트 기간 단위",
    ["주 단위", "월 단위", "분기 단위", "연 단위"]
)

period_count = st.number_input(
    "포캐스트 기간 개수",
    min_value=1,
    max_value=12,
    value=4
)

# ===============================
# 데이터 정제
# ===============================
item = data["item"].rename(columns={"상품코드": "품번"})
item["품번"] = item["품번"].astype(str).str.strip()
item["평량"] = to_num(item["평량"])

stock = data["stock"]
stock["품번"] = stock["품번"].astype(str).str.strip()
stock["재고수량"] = to_num(stock["재고수량"])

po = data["po"]
po["품번"] = po["품번"].astype(str).str.strip()
po_qty_col = next(c for c in po.columns if "잔량" in c or "수량" in c)
po["PO잔량"] = to_num(po[po_qty_col])

order = data["order"]

# 수주예정 헤더 보정
if "Unnamed" in order.columns[0]:
    order.columns = order.iloc[0]
    order = order.iloc[1:]

order.rename(columns={"상품코드": "품번"}, inplace=True)
order["품번"] = order["품번"].astype(str).str.strip()

# 수주잔량 기준 필터
if "수주잔량" in order.columns:
    order["수주잔량"] = to_num(order["수주잔량"])
    order = order[order["수주잔량"] > 0]
else:
    order["세대수"] = to_num(order["세대수"])
    order["세대당예상수량"] = to_num(order["세대당예상수량"])
    order["수주잔량"] = order["세대수"] * order["세대당예상수량"]
    order = order[order["수주잔량"] > 0]

# ===============================
# 오더 대상 품번
# ===============================
target_items = order["품번"].unique()
result = item[item["품번"].isin(target_items)].copy()

# ===============================
# 재고 / PO 계산
# ===============================
result["현재고(m)"] = result["품번"].map(
    stock.groupby("품번")["재고수량"].sum()
).fillna(0)

result["PO잔량(kg)"] = result["품번"].map(
    po.groupby("품번")["PO잔량"].sum()
).fillna(0)

result["PO환산(m)"] = np.where(
    result["평량"] > 0,
    result["PO잔량(kg)"] / (result["평량"] * 1.26 / 1000),
    0
)

result["가용재고(m)"] = result["현재고(m)"] + result["PO환산(m)"]

# ===============================
# 포캐스트 횡 전개
# ===============================
def next_date(d, step):
    if period_type == "주 단위":
        return d + relativedelta(weeks=step)
    if period_type == "월 단위":
        return d + relativedelta(months=step)
    if period_type == "분기 단위":
        return d + relativedelta(months=3 * step)
    if period_type == "연 단위":
        return d + relativedelta(years=step)

forecast = order.groupby("품번")["수주잔량"].sum()

remaining = result["가용재고(m)"].copy()

for i in range(1, period_count + 1):
    col = next_date(base_date, i).strftime("%Y-%m-%d")
    result[col] = remaining - forecast
    remaining = result[col]

# ===============================
# 발주 판단
# ===============================
result["발주판단"] = np.where(
    remaining < 0, "발주필요",
    np.where(remaining < 1000, "주의", "OK")
)

# ===============================
# 결과 표시
# ===============================
st.subheader("③ 오더 검토 결과")

st.dataframe(
    result,
    use_container_width=True
)

# ===============================
# 다운로드
# ===============================
st.download_button(
    "📥 결과 다운로드 (CSV)",
    data=result.to_csv(index=False, encoding="cp949"),
    file_name="특판_모양지_오더검토_결과.csv"
)
