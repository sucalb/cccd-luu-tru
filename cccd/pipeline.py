from __future__ import annotations

import re
from typing import List, Tuple

import cv2

from .card_ocr import extract_cards, to_boxes
from .mrz import _engine, find_mrz, lines_from_result
from .qr import IdRecord, extract_all, qr_in_region, quick_qr, read_image


def check_cccd(rec: IdRecord) -> List[str]:
    """Đối chiếu số CCCD với ngày sinh / giới tính.

    Số CCCD 12 số: 3 số mã tỉnh, 1 số (thế kỷ + giới tính), 2 số cuối năm sinh, 6 số ngẫu nhiên.
    Mã giới tính: 0/1 = Nam/Nữ sinh 1900-1999, 2/3 = Nam/Nữ sinh 2000-2099.
    """
    issues = []
    if rec.loai_giay_to != "Thẻ CCCD":
        return issues
    so = rec.so_giay_to
    if not re.fullmatch(r"\d{12}", so):
        return [f"Số CCCD không đủ 12 chữ số ({so or 'trống'})"]
    m = re.fullmatch(r"\d{2}/\d{2}/(\d{4})", rec.ngay_sinh)
    if not m:
        return ["Ngày sinh không đúng dạng dd/mm/yyyy"]
    year = int(m.group(1))
    code = int(so[3])
    if so[4:6] != str(year)[2:] or code // 2 != (year // 100) - 19:
        issues.append("Năm sinh không khớp với số CCCD")
    if rec.gioi_tinh in ("Nam", "Nữ") and (code % 2 == 1) != (rec.gioi_tinh == "Nữ"):
        issues.append("Giới tính không khớp với số CCCD")
    return issues


ROTATIONS = (None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180)
OCR_SIDE = 2400  # cạnh dài tối đa khi chạy OCR


def _read(img, qr_recs: List[IdRecord]) -> Tuple[List[IdRecord], bool]:
    """Tìm vị trí từng thẻ bằng OCR; với mỗi thẻ chưa có QR: thử QR ở góc thẻ, không được thì đọc chữ.

    Không thấy thẻ nào -> thử MRZ (hộ chiếu / mặt sau CCCD). Thử lần lượt các hướng xoay;
    tắt bộ tự xoay chữ của OCR để chỉ hướng đúng mới đọc ra chữ.
    Trả về (danh sách, có tìm thấy vị trí giấy tờ nào không).
    """
    known = frozenset(r.so_giay_to for r in qr_recs)
    for rot in ROTATIONS:
        full = img if rot is None else cv2.rotate(img, rot)
        s = min(1.0, OCR_SIDE / max(full.shape[:2]))
        view = full if s == 1 else cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        result, _ = _engine()(view, use_cls=False)
        boxes = to_boxes(result)

        def qr_near(a, full=full, s=s):
            # QR nằm phía trên-phải số thẻ; cắt từ ảnh gốc (độ phân giải đầy đủ)
            x0, x1 = a.x0 - 0.5 * a.w, a.x1 + 1.4 * a.w
            y0, y1 = a.y0 - 7 * a.h, a.y1 + a.h
            recs = qr_in_region(full, x0 / s, y0 / s, x1 / s, y1 / s)
            return recs[0] if recs else None

        cards = extract_cards(view, boxes, known, qr_near)
        if cards or any(_digits(b.text) in known for b in boxes):
            return cards, True
        rec = find_mrz(view, lines_from_result(result))
        if rec:
            return ([] if qr_recs else [rec]), True
    return [], False


def _digits(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


def process(data: bytes) -> Tuple[str, List[IdRecord]]:
    """Trả về (trạng thái, danh sách giấy tờ đọc được trong ảnh)."""
    img = read_image(data)
    if img is None:
        return "Lỗi: không mở được ảnh", []
    recs = quick_qr(img)
    more, located = _read(img, recs)
    recs += [r for r in more if r.so_giay_to not in {q.so_giay_to for q in recs}]
    if not located:
        # Không định vị được thẻ nào bằng chữ: cắt ô dò QR khắp ảnh
        recs += [r for r in extract_all(img) if r.so_giay_to not in {q.so_giay_to for q in recs}]
    if not recs:
        return "Không đọc được (ảnh mờ / không phải giấy tờ)", []
    for rec in recs:
        rec.ghi_chu.extend(check_cccd(rec))
    kind = "hộ chiếu" if recs[0].loai_giay_to == "Hộ chiếu" else "thẻ"
    src = "+".join(sorted({r.nguon for r in recs}))
    return f"{len(recs)} {kind} ({src})", recs
