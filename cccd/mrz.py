"""Đọc vùng MRZ (các dòng chữ có dấu <<<) bằng OCR offline.

- Hộ chiếu (TD3): 2 dòng x 44 ký tự.
- Mặt sau CCCD gắn chip (TD1): 3 dòng x 30 ký tự, số CCCD 12 số nằm ở dòng 1.
Các trường số đều có chữ số kiểm tra (check digit) nên phát hiện được OCR đọc sai.
"""
from __future__ import annotations

import re
import threading
from datetime import date
from typing import List, Optional

import cv2
import numpy as np

from .qr import IdRecord

_ocr = None
_lock = threading.Lock()


def _engine():
    global _ocr
    with _lock:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR

            _ocr = RapidOCR()
    return _ocr


QUOC_TICH = {
    "VNM": "Việt Nam", "KOR": "Hàn Quốc", "PRK": "Triều Tiên", "CHN": "Trung Quốc", "TWN": "Đài Loan",
    "HKG": "Hồng Kông", "JPN": "Nhật Bản", "USA": "Hoa Kỳ", "GBR": "Anh", "GBD": "Anh", "FRA": "Pháp",
    "D": "Đức", "DEU": "Đức", "RUS": "Nga", "UKR": "Ukraina", "AUS": "Úc", "NZL": "New Zealand",
    "CAN": "Canada", "THA": "Thái Lan", "LAO": "Lào", "KHM": "Campuchia", "MYS": "Malaysia",
    "SGP": "Singapore", "IDN": "Indonesia", "PHL": "Philippines", "MMR": "Myanmar", "IND": "Ấn Độ",
    "ITA": "Ý", "ESP": "Tây Ban Nha", "NLD": "Hà Lan", "BEL": "Bỉ", "CHE": "Thụy Sĩ", "SWE": "Thụy Điển",
    "NOR": "Na Uy", "DNK": "Đan Mạch", "FIN": "Phần Lan", "POL": "Ba Lan", "CZE": "Séc", "AUT": "Áo",
    "ISR": "Israel", "TUR": "Thổ Nhĩ Kỳ", "BRA": "Brazil", "MNG": "Mông Cổ", "KAZ": "Kazakhstan",
}

_TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5",
                           "B": "8", "G": "6", "T": "7"})
_TO_ALPHA = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})


def _check(field: str) -> int:
    total = 0
    for i, ch in enumerate(field):
        if ch.isdigit():
            v = int(ch)
        elif ch.isalpha():
            v = ord(ch) - 55
        else:
            v = 0
        total += v * (7, 3, 1)[i % 3]
    return total % 10


def _ok(field: str, digit: str) -> bool:
    return digit.isdigit() and _check(field) == int(digit)


def _clean(line: str) -> str:
    line = line.upper().replace(" ", "").replace("«", "<").replace("‹", "<")
    line = re.sub(r"(?<=<<)[KC](?=<<)", "<", line)  # "<" giữa chuỗi <<< hay bị đọc thành K/C
    return re.sub(r"[^A-Z0-9<]", "", line)


def _date(yymmdd: str, future: bool = False) -> str:
    yy, mm, dd = int(yymmdd[:2]), yymmdd[2:4], yymmdd[4:6]
    cur = date.today().year % 100
    century = 2000 if (yy <= cur or future) else 1900
    return f"{dd}/{mm}/{century + yy}"


def _name(field: str) -> str:
    field = re.split(r"<{3,}", field.strip("<"))[0]  # bỏ phần đệm <<<< (và ký tự rác OCR đọc ở cuối)
    parts = field.split("<<", 1)
    parts = [p.replace("<", " ").strip() for p in parts]
    return " ".join(p for p in parts if p)


def _gioi_tinh(c: str) -> str:
    return {"M": "Nam", "F": "Nữ"}.get(c, "")


def _parse_td3(l1: str, l2: str) -> Optional[IdRecord]:
    """Hộ chiếu."""
    m = re.search(r"([A-Z0-9<]{9})([0-9OIL])([A-Z0-9<]{3})([0-9OIZSBQDG]{6})([0-9OI])([MF<])", l2)
    n = re.search(r"P[A-Z<]([A-Z0-9<]{3})([A-Z<]+)", l1)
    if not m or not n:
        return None
    so, so_ck = m.group(1), m.group(2).translate(_TO_DIGIT)
    dob, dob_ck = m.group(4).translate(_TO_DIGIT), m.group(5).translate(_TO_DIGIT)
    nat = (m.group(3) if m.group(3).strip("<") else n.group(1)).translate(_TO_ALPHA).replace("<", "")
    rec = IdRecord(
        so_giay_to=so.replace("<", ""),
        ho_ten=_name(n.group(2)),
        ngay_sinh=_date(dob),
        gioi_tinh=_gioi_tinh(m.group(6)),
        loai_giay_to="Hộ chiếu",
        quoc_tich=QUOC_TICH.get(nat, nat),
        nguon="MRZ",
    )
    if not _ok(so, so_ck):
        # OCR hay nhầm O/0, I/1 trong số hộ chiếu: thử sửa phần đuôi thành số
        fixed = so[:2] + so[2:].translate(_TO_DIGIT)
        if _ok(fixed, so_ck):
            rec.so_giay_to = fixed.replace("<", "")
        else:
            rec.ghi_chu.append("Số hộ chiếu có thể đọc sai")
    if not _ok(dob, dob_ck):
        rec.ghi_chu.append("Ngày sinh có thể đọc sai")
    return rec


def _parse_td1(l1: str, l2: str, l3: str) -> Optional[IdRecord]:
    """Mặt sau CCCD gắn chip / Căn cước."""
    a = re.search(r"I[A-Z<]VNM([0-9OI]{9})([0-9OI])([0-9OIZSBQDG]{12})", l1)
    b = re.search(r"([0-9OIZSBQDG]{6})([0-9OI])([MF<])([0-9OIZSBQDG]{6})([0-9OI])VNM", l2)
    if not a or not b:
        return None
    so = a.group(3).translate(_TO_DIGIT)
    dob, dob_ck = b.group(1).translate(_TO_DIGIT), b.group(2).translate(_TO_DIGIT)
    rec = IdRecord(
        so_giay_to=so,
        ho_ten=_name(l3.translate(_TO_ALPHA)),
        ngay_sinh=_date(dob),
        gioi_tinh=_gioi_tinh(b.group(3)),
        nguon="MRZ",
    )
    rec.ghi_chu.append("Đọc từ mặt sau: họ tên không dấu, không có nơi thường trú")
    if not _ok(dob, dob_ck):
        rec.ghi_chu.append("Ngày sinh có thể đọc sai")
    if not _ok(a.group(1).translate(_TO_DIGIT), a.group(2).translate(_TO_DIGIT)):
        rec.ghi_chu.append("Số CCCD có thể đọc sai")
    return rec


def lines_from_result(result):
    """[(text đã làm sạch, khung chữ)] sắp theo toạ độ y để giữ thứ tự dòng."""
    result = sorted(result or [], key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)
    return [(_clean(r[1]), np.array(r[0], np.float32)) for r in result]


def _is_mrz(text: str) -> bool:
    # Dòng 2 hộ chiếu có thể không có dấu < nào (khi số cá nhân lấp đầy dòng)
    return len(text) >= 15 and (text.count("<") >= 2 or len(text) >= 40)


def _line_above(img: np.ndarray, box: np.ndarray) -> List[str]:
    """Đọc lại dòng MRZ nằm ngay trên một dòng đã đọc được (OCR hay bỏ sót dòng tên).

    Chỉ đọc nửa đầu dòng: chuỗi <<<<< dài ở cuối làm bộ nhận dạng đọc sai cả dòng.
    """
    x0, y0 = box.min(0)
    x1, y1 = box.max(0)
    h = y1 - y0
    if h <= 0 or (x1 - x0) < 3 * h:  # chỉ xử lý khi dòng nằm ngang trong ảnh này
        return []
    out = []
    for k in (1.2, 1.35, 1.1):
        ya, yb = int(y0 - k * h - 0.15 * h), int(y1 - k * h + 0.15 * h)
        strip = img[max(ya, 0):max(yb, 0), max(int(x0 - h / 2), 0):int(x0 + (x1 - x0) * 0.6)]
        if strip.size == 0:
            continue
        for v in (strip, cv2.resize(strip, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)):
            res, _ = _engine()(v, use_det=False, use_cls=False, use_rec=True)
            if res:
                out.append(_clean(res[0][0]))
    return out


def find_mrz(img: np.ndarray, lines) -> Optional[IdRecord]:
    mrz = [(t, b) for t, b in lines if _is_mrz(t)]
    texts = [t for t, _ in mrz]
    for i in range(len(texts) - 1):
        if texts[i].startswith("P"):
            rec = _parse_td3(texts[i], texts[i + 1])
            if rec:
                return rec
    for i in range(len(texts) - 2):
        if texts[i].startswith("I"):
            rec = _parse_td1(texts[i], texts[i + 1], texts[i + 2])
            if rec:
                return rec
    # Chỉ đọc được dòng 2 của hộ chiếu: tìm lại dòng tên ngay phía trên
    for t, b in mrz:
        if t.startswith("P"):
            continue
        for l1 in _line_above(img, b):
            if l1.startswith("P"):
                rec = _parse_td3(l1, t)
                if rec:
                    return rec
    return None
