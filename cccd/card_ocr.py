"""Đọc chữ in trên mặt trước CCCD/CMND khi không đọc được QR (ảnh nhỏ, nén mạnh, QR bị lóa...).

- Phát hiện vùng chữ + đọc số bằng RapidOCR (nhanh, chính xác với chữ số).
- Đọc họ tên / địa chỉ tiếng Việt có dấu bằng VietOCR (chạy offline, trọng số trong models/).
- Mỗi thẻ được định vị bằng số CCCD (12 số) / CMND (9 số) làm mốc, nên ảnh chụp chung nhiều thẻ vẫn tách được.
- Năm sinh và giới tính được đối chiếu/sửa theo số CCCD (chữ số 4-6 mã hoá thế kỷ, giới tính, năm sinh).
"""
from __future__ import annotations

import re
import threading
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .qr import IdRecord

WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "vgg_transformer.pth"

_viet = None
_viet_lock = threading.RLock()


def _vietocr():
    global _viet
    with _viet_lock:
        if _viet is None:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor

            cfg = Cfg.load_config_from_name("vgg_transformer")
            cfg["device"] = "cpu"
            cfg["cnn"]["pretrained"] = False
            cfg["predictor"]["beamsearch"] = False
            if WEIGHTS.exists():
                cfg["weights"] = str(WEIGHTS)
            _viet = Predictor(cfg)
    return _viet


def read_vietnamese(img: np.ndarray, boxes: List["Box"]) -> List[str]:
    from PIL import Image

    crops = []
    for b in boxes:
        pad = b.h * 0.15
        c = img[int(max(b.y0 - pad, 0)):int(b.y1 + pad), int(max(b.x0 - pad, 0)):int(b.x1 + pad)]
        crops.append(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
    if not crops:
        return []
    with _viet_lock:  # Predictor không an toàn khi gọi song song
        return _vietocr().predict_batch(crops)


@dataclass
class Box:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def h(self):
        return self.y1 - self.y0

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def cy(self):
        return (self.y0 + self.y1) / 2

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2


def to_boxes(result) -> List[Box]:
    out = []
    for quad, text, _ in result or []:
        q = np.array(quad, np.float32)
        (x0, y0), (x1, y1) = q.min(0), q.max(0)
        out.append(Box(text, float(x0), float(y0), float(x1), float(y1)))
    return out


def _plain(s: str) -> str:
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


_ID_RE = re.compile(r"(?<!\d)(\d{12}|\d{9})(?!\d)")
_DATE_RE = re.compile(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{4})")
LABEL_WORDS = ("place", "origin", "residence", "full name", "date", "birth", "expiry", "sex", "nationality",
               "citizen", "identity", "card", "socialist", "republic", "independence", "freedom", "happiness",
               "que quan", "thuong tru", "gioi tinh", "quoc tich", "ho va ten", "ngay sinh", "gia tri",
               "can cuoc", "cong hoa", "doc lap", "so /", "no.", "personal")


def _id_anchor(b: Box) -> Optional[str]:
    t = b.text.replace(" ", "").replace(".", "")
    # bỏ qua dòng MRZ và ngày tháng; giữ trường hợp OCR đọc dính nhãn "Số/No:" vào trước số
    if "<" in t or len(t) > 24 or _DATE_RE.search(t):
        return None
    m = _ID_RE.search(t)
    if not m:
        return None
    so = m.group(1)
    if len(so) == 12 and not (1 <= int(so[:3]) <= 96 and so[3] in "0123"):
        return None  # không phải mã tỉnh / mã thế kỷ hợp lệ
    return so


def _is_label(text: str) -> bool:
    p = _plain(text)
    return any(w in p for w in LABEL_WORDS)


HO = {  # họ phổ biến: bản không dấu -> có dấu (sửa lỗi dấu của OCR ở chữ đầu tiên)
    "NGUYEN": "NGUYỄN", "TRAN": "TRẦN", "LE": "LÊ", "PHAM": "PHẠM", "HOANG": "HOÀNG", "HUYNH": "HUỲNH",
    "PHAN": "PHAN", "VU": "VŨ", "VO": "VÕ", "DANG": "ĐẶNG", "BUI": "BÙI", "DO": "ĐỖ", "HO": "HỒ",
    "NGO": "NGÔ", "DUONG": "DƯƠNG", "LY": "LÝ", "DINH": "ĐINH", "DOAN": "ĐOÀN", "TRINH": "TRỊNH",
    "TRUONG": "TRƯƠNG", "LAM": "LÂM", "LUONG": "LƯƠNG", "LUU": "LƯU", "TA": "TẠ", "KIEU": "KIỀU",
    "THAI": "THÁI", "QUACH": "QUÁCH", "TONG": "TỐNG", "VUONG": "VƯƠNG", "PHUNG": "PHÙNG", "CAO": "CAO",
    "MAI": "MAI", "HA": "HÀ", "TO": "TÔ", "CHU": "CHU", "TRIEU": "TRIỆU", "KHUAT": "KHUẤT", "NONG": "NÔNG",
}
DEM = {"THI": "THỊ", "VAN": "VĂN"}


def _fix_name(name: str) -> str:
    words = re.sub(r"[^\w\s]", " ", name).upper().split()
    if words:
        words[0] = HO.get(_plain(words[0]).upper(), words[0])
    if len(words) > 2:
        words[1] = DEM.get(_plain(words[1]).upper(), words[1])
    return " ".join(words)


def _same_line(boxes: List[Box]) -> List[List[Box]]:
    lines: List[List[Box]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if lines and abs(lines[-1][-1].cy - b.cy) < 0.5 * min(b.h, lines[-1][-1].h):
            lines[-1].append(b)
        else:
            lines.append([b])
    return [sorted(l, key=lambda b: b.x0) for l in lines]


def _parse_card(img: np.ndarray, anchor: Box, so: str, boxes: List[Box]) -> IdRecord:
    h, W = anchor.h, anchor.w
    rec = IdRecord(so_giay_to=so, loai_giay_to="Thẻ CCCD" if len(so) == 12 else "CMND", nguon="Chữ")
    below = sorted([b for b in boxes if b.y0 > anchor.y1 - 0.3 * h], key=lambda b: (b.cy, b.x0))

    # Ngày sinh: ngày đầu tiên bên dưới số thẻ, nằm trong cột chữ (ngày hết hạn nằm dưới ảnh chân dung)
    dob_box = None
    for b in below:
        m = _DATE_RE.search(b.text)
        if m and b.x1 > anchor.x0 - 0.2 * W:
            dob_box = b
            rec.ngay_sinh = f"{int(m.group(1)):02d}/{int(m.group(2)):02d}/{m.group(3)}"
            break

    # Họ tên: dòng chữ in hoa đầu tiên dưới số thẻ (trước ngày sinh)
    limit = dob_box.y0 if dob_box else anchor.y1 + 3.5 * h
    cand = [b for b in below if b.cy < limit and b.x1 > anchor.x0 - 0.3 * W
            and not _ID_RE.search(b.text.replace(" ", "")) and not _is_label(b.text)]
    name_boxes = []
    if cand:
        texts = read_vietnamese(img, cand)
        for b, t in zip(cand, texts):
            letters = re.sub(r"[^\w]", "", t)
            if len(t.split()) >= 2 and letters.isalpha() and t.upper() == t and not _is_label(t):
                name_boxes.append((b, t))
                break
    if name_boxes:
        rec.ho_ten = _fix_name(name_boxes[0][1])

    # Nơi thường trú: các dòng sau nhãn "Nơi thường trú / Place of residence"
    label = next((b for b in below if "residen" in _plain(b.text) or "thuong tru" in _plain(b.text)), None)
    if label:
        addr = [b for b in boxes if b.y0 > label.y0 - 0.3 * label.h and b is not label
                and b.cy < label.y1 + 3 * h and b.x1 > anchor.x0 - 0.3 * W and not _is_label(b.text)
                and not _DATE_RE.search(b.text)]
        if addr:
            texts = read_vietnamese(img, addr)
            ordered = sorted(zip(addr, texts), key=lambda z: (round(z[0].cy / h), z[0].x0))
            rec.noi_thuong_tru = ", ".join(t.strip(" ,") for _, t in ordered if t.strip())

    rec.ghi_chu.append("Đọc từ chữ in (không có QR): nên kiểm tra lại họ tên")
    _fix_from_id(rec)
    return rec


def _fix_from_id(rec: IdRecord) -> None:
    """Giới tính + năm sinh suy ra từ số CCCD (đáng tin hơn chữ OCR)."""
    so = rec.so_giay_to
    if len(so) != 12:
        return
    code = int(so[3])
    rec.gioi_tinh = "Nữ" if code % 2 else "Nam"
    year = str(1900 + 100 * (code // 2) + int(so[4:6]))
    m = re.fullmatch(r"(\d{2}/\d{2}/)(\d{4})", rec.ngay_sinh)
    if m and m.group(2) != year:
        rec.ngay_sinh = m.group(1) + year
        rec.ghi_chu.append("Đã sửa năm sinh theo số CCCD")
    elif not m:
        rec.ghi_chu.append(f"Không đọc được ngày sinh (năm sinh theo số CCCD: {year})")


def extract_cards(img: np.ndarray, boxes: List[Box], skip=frozenset(), qr_near=None) -> List[IdRecord]:
    """Đọc mọi thẻ trong ảnh, dùng số thẻ làm mốc.

    skip: các số thẻ đã đọc được bằng QR.
    qr_near(anchor) -> IdRecord | None: thử đọc QR ngay góc trên-phải của thẻ trước khi đọc chữ.
    """
    anchors = [(b, so) for b in boxes if (so := _id_anchor(b))]
    if not anchors:
        return []
    # Gán mỗi vùng chữ cho thẻ gần nhất (số thẻ nằm phía trên-trái các trường còn lại)
    groups = {i: [] for i in range(len(anchors))}
    for b in boxes:
        best, dist = None, None
        for i, (a, _) in enumerate(anchors):
            if not (a.x0 - 1.0 * a.w < b.cx < a.x1 + 1.2 * a.w and a.y0 - 4 * a.h < b.cy < a.y1 + 6 * a.h):
                continue
            d = abs(b.y0 - a.y0) + 0.5 * abs(b.x0 - a.x0)
            if b.cy < a.y0:
                d += 10 * a.h  # ưu tiên thẻ có số thẻ nằm phía trên vùng chữ
            if dist is None or d < dist:
                best, dist = i, d
        if best is not None:
            groups[best].append(b)
    seen, out = set(), []
    for i, (a, so) in enumerate(anchors):
        if so in seen:
            continue
        seen.add(so)
        if so in skip:
            continue
        rec = qr_near(a) if qr_near else None
        if rec is not None and rec.so_giay_to not in seen | set(skip):
            seen.add(rec.so_giay_to)
            out.append(rec)
            if rec.so_giay_to == so:
                continue
        out.append(_parse_card(img, a, so, [b for b in groups[i] if b is not a]))
    return out
