"""Kiểm tra nhanh khi build: tạo thẻ giả có QR + hộ chiếu giả có MRZ, chạy qua pipeline.

Dữ liệu hoàn toàn bịa, không dùng ảnh giấy tờ thật.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cccd.excel import build_workbook  # noqa: E402
from cccd.mrz import _check  # noqa: E402
from cccd.pipeline import process  # noqa: E402


def fake_card() -> bytes:
    text = "001095012345||Nguyễn Văn Thử|12071995|Nam|Số 1 Ngõ Test, Hà Nội|01012022"
    qr = cv2.QRCodeEncoder_create().encode(text.encode("utf-8").decode("latin-1"))
    qr = cv2.resize(qr, (200, 200), interpolation=cv2.INTER_NEAREST)
    img = np.full((800, 1200, 3), 210, np.uint8)
    img[100:300, 900:1100] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
    return cv2.imencode(".jpg", img)[1].tobytes()


def fake_passport() -> bytes:
    l1 = "P<KORTESTER<<JOHN".ljust(44, "<")
    n, dob, exp = "M12345678", "650916", "290410"
    body = n + str(_check(n)) + "KOR" + dob + str(_check(dob)) + "M" + exp + str(_check(exp)) + "<" * 14 + "0"
    l2 = body + str(_check(body[0:10] + body[13:20] + body[21:43]))
    img = np.full((700, 1500, 3), 235, np.uint8)
    for i, line in enumerate((l1, l2)):
        cv2.putText(img, line, (40, 520 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (20, 20, 20), 3)
    return cv2.imencode(".jpg", img)[1].tobytes()


def main():
    assert hasattr(cv2, "wechat_qrcode_WeChatQRCode"), "thiếu opencv-contrib (bộ đọc QR WeChat)"
    status, recs = process(fake_card())
    print("card:", status, [r.__dict__ for r in recs])
    assert recs and recs[0].ho_ten == "NGUYỄN VĂN THỬ" and recs[0].so_giay_to == "001095012345"

    status, recs = process(fake_passport())
    print("passport:", status, [r.__dict__ for r in recs])
    assert recs and recs[0].so_giay_to == "M12345678" and recs[0].quoc_tich == "Hàn Quốc"

    data, kq = build_workbook([], [r.__dict__ for r in recs])
    assert kq["Người nước ngoài"].them == 1 and len(data) > 1000

    from cccd.card_ocr import _vietocr
    _vietocr()  # nạp mô hình tiếng Việt offline (không cần mạng)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
