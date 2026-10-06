"""Đọc mã QR trên CCCD gắn chip / thẻ Căn cước.

Nội dung QR có dạng:
    Số CCCD|Số CMND cũ|Họ tên|Ngày sinh(ddmmyyyy)|Giới tính|Nơi thường trú|Ngày cấp(ddmmyyyy)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

_wechat = cv2.wechat_qrcode_WeChatQRCode()
_aruco = cv2.QRCodeDetectorAruco()


@dataclass
class IdRecord:
    so_giay_to: str = ""
    ho_ten: str = ""
    ngay_sinh: str = ""  # dd/mm/yyyy
    gioi_tinh: str = ""  # Nam / Nữ
    noi_thuong_tru: str = ""
    loai_giay_to: str = "Thẻ CCCD"
    quoc_tich: str = "Việt Nam"
    nguon: str = "QR"  # QR / MRZ
    ghi_chu: List[str] = field(default_factory=list)


def _fix_mojibake(s: str) -> str:
    try:
        fixed = s.encode("latin-1").decode("utf-8")
        return fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _fmt_date(raw: str) -> str:
    raw = re.sub(r"\D", "", raw or "")
    if len(raw) == 8:
        return f"{raw[:2]}/{raw[2:4]}/{raw[4:]}"
    return raw


def parse_qr_text(text: str) -> Optional[IdRecord]:
    text = _fix_mojibake(text.strip())
    parts = [p.strip() for p in text.split("|")]
    if len(parts) < 6 or not re.fullmatch(r"\d{12}", parts[0]):
        return None
    gioi_tinh = parts[4]
    if gioi_tinh.lower() in ("nu", "nữ", "female", "f"):
        gioi_tinh = "Nữ"
    elif gioi_tinh.lower() in ("nam", "male", "m"):
        gioi_tinh = "Nam"
    return IdRecord(
        so_giay_to=parts[0],
        ho_ten=parts[2].upper(),
        ngay_sinh=_fmt_date(parts[3]),
        gioi_tinh=gioi_tinh,
        noi_thuong_tru=parts[5],
    )


def _resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def _candidates(img: np.ndarray):
    """Các biến thể của cả ảnh để thử decode."""
    yield img
    for side in (2000, 1400, 1000):
        if max(img.shape[:2]) > side:
            yield _resize_max(img, side)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    yield cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    # Phóng to khi ảnh nhỏ (QR chiếm ít pixel)
    if max(img.shape[:2]) < 1600:
        yield cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


def _tiles(img: np.ndarray, n: int):
    """Cắt ảnh thành lưới n x n ô chồng lấn 50%, phóng to ô nhỏ.

    Dùng cho ảnh chụp chung nhiều thẻ: mỗi QR chỉ chiếm ít pixel nên đọc cả ảnh dễ sót.
    """
    h, w = img.shape[:2]
    th, tw = int(h * 2 / (n + 1)), int(w * 2 / (n + 1))
    for i in range(n):
        for j in range(n):
            y, x = int(i * th / 2), int(j * tw / 2)
            tile = img[y:y + th, x:x + tw]
            if max(tile.shape[:2]) < 1200:
                tile = cv2.resize(tile, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            yield tile


def _decode(img: np.ndarray, found: dict) -> None:
    texts, _ = _wechat.detectAndDecode(img)
    ok, more, _, _ = _aruco.detectAndDecodeMulti(img)
    for t in list(texts) + (list(more) if ok else []):
        rec = parse_qr_text(t) if t and "|" in t else None
        if rec and rec.so_giay_to not in found:
            found[rec.so_giay_to] = rec


def extract_all(img: np.ndarray) -> List[IdRecord]:
    """Đọc tất cả CCCD có trong ảnh (1 hoặc nhiều thẻ chụp chung)."""
    found: dict = {}
    for cand in _candidates(img):
        _decode(cand, found)
    # Cắt nhỏ dần (lưới 2x2 và 3x3 lệch nhau nên luôn thử cả hai);
    # chỉ cắt 4x4 khi 3x3 còn tìm thêm được thẻ
    for n in (2, 3, 4):
        before = len(found)
        for tile in _tiles(img, n):
            _decode(tile, found)
        if n >= 3 and len(found) == before:
            break
    return list(found.values())


def read_image(data: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    # HEIC từ iPhone
    try:
        import io

        import pillow_heif
        from PIL import Image

        pillow_heif.register_heif_opener()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def qr_in_region(img: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> List[IdRecord]:
    """Đọc QR trong một vùng nhỏ (góc thẻ), phóng to để QR nhỏ vẫn đọc được."""
    h, w = img.shape[:2]
    crop = img[max(int(y0), 0):min(int(y1), h), max(int(x0), 0):min(int(x1), w)]
    if crop.size == 0:
        return []
    found: dict = {}
    scale = max(1.0, 800 / max(crop.shape[:2]))
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale > 1 else crop
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    for v in (big, cv2.cvtColor(cv2.createCLAHE(2.0, (8, 8)).apply(gray), cv2.COLOR_GRAY2BGR),
              cv2.addWeighted(big, 1.8, cv2.GaussianBlur(big, (0, 0), 3), -0.8, 0)):
        _decode(v, found)
        if found:
            break
    return list(found.values())


def quick_qr(img: np.ndarray) -> List[IdRecord]:
    """Chỉ đọc QR trên toàn ảnh (nhanh), không cắt ô."""
    found: dict = {}
    for cand in _candidates(img):
        _decode(cand, found)
    return list(found.values())
