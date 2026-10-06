"""Ghi danh sách khách vào file "Sổ Khai báo lưu trú.xlsx", giữ nguyên định dạng.

- Khách Việt Nam vào sheet "Việt Nam", khách nước ngoài vào sheet "Người nước ngoài".
- Cột được tìm theo tên tiêu đề (dòng có ô "STT"), nên file của khách có đổi thứ tự cột vẫn điền đúng.
- Có thể ghi tiếp vào file đang dùng: nối sau dòng cuối, STT đánh tiếp, bỏ qua người đã có trong file.
"""
from __future__ import annotations

import io
import re
import unicodedata
from copy import copy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import openpyxl
from openpyxl.worksheet.worksheet import Worksheet

TEMPLATE = Path(__file__).resolve().parent.parent / "template.xlsx"
FOOTER = "Người thực hiện"
FOOTER_GAP = 5  # số dòng trống giữa dữ liệu và "Người thực hiện" (giống file mẫu)

# tiêu đề cột (đã chuẩn hoá) -> khoá dữ liệu
HEADERS = {
    "stt": "stt",
    "ho ten": "ho_ten", "ho va ten": "ho_ten",
    "ngay sinh": "ngay_sinh",
    "gt": "gioi_tinh", "gioi tinh": "gioi_tinh",
    "loai giay to": "loai_giay_to",
    "so giay to": "so_giay_to", "so ho chieu": "so_giay_to", "so cccd": "so_giay_to",
    "quoc tich": "quoc_tich",
    "noi thuong tru": "noi_thuong_tru", "dia chi": "noi_thuong_tru", "noi o": "noi_thuong_tru",
    "ngay den": "ngay_den",
    "ngay di du kien": "ngay_di_du_kien",
    "ngay di thuc te": "ngay_di_thuc_te",
    "so phong": "so_phong",
    "ly do cu tru": "ly_do", "ly do": "ly_do",
    "co so luu tru": "co_so",
}


def _norm(s) -> str:
    s = unicodedata.normalize("NFD", str(s or "")).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip().lower()


@dataclass
class SheetResult:
    them: int = 0
    bo_qua_trung: int = 0


def _sheet(wb, prefix: str, index: int) -> Worksheet:
    for ws in wb.worksheets:
        if _norm(ws.title).startswith(prefix):
            return ws
    return wb.worksheets[index]


def _header(ws: Worksheet):
    for r in range(1, 40):
        if _norm(ws.cell(r, 1).value) == "stt":
            cols = {}
            for c in range(1, ws.max_column + 1):
                key = HEADERS.get(_norm(ws.cell(r, c).value))
                if key and key not in cols:
                    cols[key] = c
            return r, cols
    raise ValueError(f'Sheet "{ws.title}" không có dòng tiêu đề bắt đầu bằng "STT"')


def _fill(ws: Worksheet, rows: List[Dict], nguoi_thuc_hien: str, ngay_lap: date, dia_chi: str,
          append: bool) -> SheetResult:
    header_row, cols = _header(ws)
    first = header_row + 1
    ncols = max(cols.values())

    # Dòng "Người thực hiện" và các dòng dữ liệu đã có
    footer_pos, footer_style = (None, ncols), None
    last, existing = header_row, set()
    for r in range(first, ws.max_row + 1):
        for c in range(1, ws.max_column + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str) and v.strip().startswith(FOOTER):
                footer_pos, footer_style = (r, c), copy(ws.cell(r, c)._style)
        if footer_pos[0] is None and any(ws.cell(r, c).value not in (None, "") for c in cols.values()):
            last = r
            so = ws.cell(r, cols["so_giay_to"]).value if "so_giay_to" in cols else None
            if so:
                existing.add(str(so).strip())

    style_row = first if last >= first else first
    row_style = {c: copy(ws.cell(style_row, c)._style) for c in range(1, ncols + 1)}
    row_height = ws.row_dimensions[style_row].height

    if footer_pos[0] is not None:
        ws.cell(*footer_pos).value = None
    if not append:  # dùng file mẫu: xoá dữ liệu mẫu cũ
        for r in range(first, ws.max_row + 1):
            for c in range(1, ncols + 1):
                ws.cell(r, c).value = None
        last, existing = header_row, set()

    ws["A3"] = f"Ngày {ngay_lap.day} tháng {ngay_lap.month} năm {ngay_lap.year}"
    if dia_chi:
        ws["A8"] = dia_chi

    res = SheetResult()
    stt = last - header_row
    r = last
    for row in rows:
        so = str(row.get("so_giay_to", "") or "").strip()
        if so and so in existing:
            res.bo_qua_trung += 1
            continue
        existing.add(so)
        r += 1
        stt += 1
        for c in range(1, ncols + 1):
            ws.cell(r, c)._style = copy(row_style[c])
        for key, c in cols.items():
            val = stt if key == "stt" else str(row.get(key, "") or "")
            cell = ws.cell(r, c, val)
            if key != "stt":
                cell.number_format = "@"  # giữ số 0 đầu của số CCCD
        if row_height:
            ws.row_dimensions[r].height = row_height
        res.them += 1

    footer_row = max(r, header_row + 1) + FOOTER_GAP
    cell = ws.cell(footer_row, footer_pos[1], f"{FOOTER}: {nguoi_thuc_hien}".strip())
    if footer_style:
        cell._style = footer_style
    return res


def build_workbook(viet_nam: List[Dict], nuoc_ngoai: List[Dict], nguoi_thuc_hien: str = "",
                   ngay_lap: date = None, dia_chi: str = "", base: Optional[bytes] = None):
    """base = file Excel đang dùng của khách (ghi tiếp); None = dùng file mẫu trống.

    Trả về (nội dung file, {"Việt Nam": SheetResult, "Người nước ngoài": SheetResult}).
    """
    wb = openpyxl.load_workbook(io.BytesIO(base) if base else TEMPLATE)
    ngay_lap = ngay_lap or date.today()
    append = base is not None
    kq = {
        "Việt Nam": _fill(_sheet(wb, "viet nam", 0), viet_nam, nguoi_thuc_hien, ngay_lap, dia_chi, append),
        "Người nước ngoài": _fill(_sheet(wb, "nguoi nuoc ngoai", 1), nuoc_ngoai, nguoi_thuc_hien,
                                  ngay_lap, dia_chi, append),
    }
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue(), kq
