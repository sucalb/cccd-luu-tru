# Đọc CCCD → Sổ khai báo lưu trú

Ứng dụng chạy trên máy (offline) đọc ảnh CCCD / hộ chiếu và điền vào file Excel "Sổ khai báo lưu trú".

- **QR trên CCCD gắn chip**: đọc họ tên, ngày sinh, giới tính, số CCCD, nơi thường trú.
- **Không có QR** (ảnh nhỏ, nén, lóa): đọc chữ in trên thẻ bằng OCR tiếng Việt (VietOCR); năm sinh và giới tính được đối chiếu với số CCCD.
- **Hộ chiếu / mặt sau CCCD**: đọc dòng MRZ (có chữ số kiểm tra).
- Một ảnh chụp chung nhiều thẻ vẫn tách được từng thẻ.
- Khách Việt Nam → sheet "Việt Nam", khách nước ngoài → sheet "Người nước ngoài"; có thể ghi tiếp vào file Excel đang dùng.

Ảnh không được gửi lên bất kỳ máy chủ nào.

## Chạy

Mac: bấm đúp `Chay ung dung.command` (lần đầu tự cài thư viện và tải mô hình OCR ~150MB).

Thủ công:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip uninstall -y opencv-python opencv-python-headless
.venv/bin/pip install --force-reinstall --no-deps opencv-contrib-python-headless "pillow>=11.1"
.venv/bin/streamlit run app.py
```
