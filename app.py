"""Ứng dụng: tải ảnh CCCD lên → đọc thông tin → xuất "Sổ Khai báo lưu trú.xlsx".

Chạy:  streamlit run app.py
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from cccd.excel import build_workbook
from cccd.pipeline import process

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp"}
COLS = {
    "anh": "Ảnh", "trang_thai": "Kết quả ảnh", "ho_ten": "Họ tên", "ngay_sinh": "Ngày sinh",
    "gioi_tinh": "GT", "loai_giay_to": "Loại giấy tờ", "so_giay_to": "Số giấy tờ",
    "quoc_tich": "Quốc tịch", "noi_thuong_tru": "Nơi thường trú", "so_phong": "Số phòng",
    "ngay_den": "Ngày đến", "ngay_di_du_kien": "Ngày đi dự kiến", "ngay_di_thuc_te": "Ngày đi thực tế",
    "ly_do": "Lý do cư trú", "co_so": "Cơ sở lưu trú", "ghi_chu": "Cần kiểm tra", "bo_qua": "Bỏ qua",
}

st.set_page_config(page_title="CCCD → Sổ lưu trú", layout="wide")
st.title("Đọc CCCD → Sổ khai báo lưu trú")

if "results" not in st.session_state:
    st.session_state.results = {}  # mã dòng ("tên file" hoặc "tên file #2") -> dict
    st.session_state.done = set()  # tên file đã đọc

with st.sidebar:
    st.header("Thông tin chung")
    co_so = st.text_input("Cơ sở lưu trú", placeholder="Tên khách sạn")
    dia_chi = st.text_input("Dòng địa chỉ (ô A8)", placeholder="Tên khách sạn - số nhà, đường, xã, tỉnh")
    ngay_den = st.date_input("Ngày đến", date.today(), format="DD/MM/YYYY")
    ngay_di = st.date_input("Ngày đi dự kiến", date.today(), format="DD/MM/YYYY")
    ly_do = st.text_input("Lý do cư trú", "Du lịch")
    nguoi_thuc_hien = st.text_input("Người thực hiện", "")
    st.divider()
    if st.button("Xoá kết quả, làm lại"):
        st.session_state.results = {}
        st.session_state.done = set()
        st.rerun()

tab_upload, tab_folder = st.tabs(["Tải ảnh lên", "Đọc cả thư mục ảnh"])
with tab_upload:
    uploads = st.file_uploader("Chọn ảnh CCCD (chọn được nhiều ảnh)", accept_multiple_files=True,
                               type=[e[1:] for e in IMAGE_EXT])
with tab_folder:
    folder = st.text_input("Đường dẫn thư mục chứa ảnh", placeholder="/Users/ten/Desktop/anh_cccd")

files: dict = {}
for f in uploads or []:
    files[f.name] = f.getvalue
if folder and Path(folder).is_dir():
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in IMAGE_EXT:
            files[str(p.relative_to(folder))] = p.read_bytes
elif folder:
    st.warning("Không tìm thấy thư mục này.")

todo = [name for name in files if name not in st.session_state.done]
st.write(f"**{len(files)}** ảnh — {len(todo)} ảnh chưa đọc.")

if todo and st.button(f"Đọc {len(todo)} ảnh", type="primary"):
    bar = st.progress(0.0, "Đang đọc…")

    def work(name):
        status, recs = process(files[name]())
        if not recs:
            return name, [{"anh": name, "trang_thai": status, "bo_qua": True, "ghi_chu": status}]
        rows = []
        for i, rec in enumerate(recs, 1):
            label = name if len(recs) == 1 else f"{name} #{i}"
            rows.append({**rec.__dict__, "anh": label, "trang_thai": status, "bo_qua": False,
                         "ghi_chu": "; ".join(rec.ghi_chu)})
        return name, rows

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        for i, (name, rows) in enumerate(pool.map(work, todo), 1):
            for row in rows:
                st.session_state.results[row["anh"]] = row
            st.session_state.done.add(name)
            bar.progress(i / len(todo), f"Đã đọc {i}/{len(todo)} ảnh")
    st.rerun()

results = [r for r in st.session_state.results.values() if r["anh"].split(" #")[0] in files]
if not results:
    st.stop()

df = pd.DataFrame(results)
for key in list(COLS) + ["nguon"]:
    if key not in df:
        df[key] = ""
# Giá trị chung điền cho các dòng còn trống
defaults = {"ngay_den": ngay_den.strftime("%d/%m/%Y"), "ngay_di_du_kien": ngay_di.strftime("%d/%m/%Y"),
            "ly_do": ly_do, "co_so": co_so}
for key, val in defaults.items():
    df[key] = df[key].fillna("").replace("", val)
df = df.fillna("")

# Cùng một người chụp nhiều ảnh (mặt trước + mặt sau...): ưu tiên bản QR, rồi chữ in (có dấu), cuối cùng MRZ
UU_TIEN = {"QR": 0, "Chữ": 1, "MRZ": 2}
cand = df[(df.so_giay_to != "") & ~df.bo_qua].sort_values(
    "nguon", key=lambda s: s.map(UU_TIEN).fillna(3), kind="stable")
dup = cand.duplicated("so_giay_to", keep="first")
df.loc[dup[dup].index, "bo_qua"] = True
df.loc[dup[dup].index, "ghi_chu"] = "Trùng số giấy tờ với ảnh khác"

n_ok = int((~df.bo_qua & (df.ghi_chu == "")).sum())
n_warn = int((~df.bo_qua & (df.ghi_chu != "")).sum())
c1, c2, c3 = st.columns(3)
c1.metric("Đọc tốt", n_ok)
c2.metric("Cần kiểm tra", n_warn)
c3.metric("Bỏ qua / lỗi", int(df.bo_qua.sum()))

only_issues = st.toggle("Chỉ hiện dòng cần kiểm tra / lỗi")
view = df[df.ghi_chu != ""] if only_issues else df
view = view[list(COLS)].rename(columns=COLS)

edited = st.data_editor(
    view, use_container_width=True, hide_index=True, height=520,
    disabled=["Ảnh", "Kết quả ảnh"],
    column_config={"Bỏ qua": st.column_config.CheckboxColumn(), "Cần kiểm tra": st.column_config.TextColumn(width="large")},
    key=f"editor_{only_issues}",
)
# Lưu chỉnh sửa tay lại vào kết quả
for _, r in edited.rename(columns={v: k for k, v in COLS.items()}).iterrows():
    st.session_state.results[r["anh"]].update({k: r[k] for k in COLS if k not in ("anh", "trang_thai")})

final = [r for r in st.session_state.results.values() if r["anh"].split(" #")[0] in files]
final = [{**defaults, **{k: v for k, v in r.items() if v not in ("", None)}} for r in final if not r.get("bo_qua")]
vn = [r for r in final if (r.get("quoc_tich") or "Việt Nam").strip().lower() in ("việt nam", "viet nam", "vn")]
nn = [r for r in final if r not in vn]

st.divider()
st.subheader("Xuất Excel")
base_file = st.file_uploader(
    "File Excel đang dùng của khách (không bắt buộc)", type=["xlsx"],
    help="Có file: ghi tiếp vào cuối mỗi sheet, STT đánh tiếp, bỏ qua người đã có trong file. "
         "Không có: tạo file mới từ mẫu Sổ khai báo lưu trú.",
)
try:
    data, kq = build_workbook(vn, nn, nguoi_thuc_hien, date.today(), dia_chi,
                              base=base_file.getvalue() if base_file else None)
except Exception as e:
    st.error(f"Không ghi được vào file Excel này: {e}")
    st.stop()

for sheet, r in kq.items():
    msg = f"Sheet **{sheet}**: thêm {r.them} người"
    if r.bo_qua_trung:
        msg += f", bỏ qua {r.bo_qua_trung} người đã có trong file"
    st.write(msg)

name = Path(base_file.name).stem if base_file else "So_khai_bao_luu_tru"
st.download_button(
    f"⬇️ Tải file Excel ({len(vn)} khách Việt Nam, {len(nn)} khách nước ngoài)",
    data,
    file_name=f"{name}_{date.today():%d-%m-%Y}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary",
)
