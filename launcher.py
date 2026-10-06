"""Điểm khởi động cho bản đóng gói (DocCCCD.exe): chạy app Streamlit và mở trình duyệt."""
import os
import socket
import sys
import threading
import webbrowser


def _free_port(start: int = 8501) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return start


def _bundle_imports():  # không bao giờ gọi: chỉ để PyInstaller gom đủ thư viện mà app.py dùng
    import cccd.card_ocr  # noqa: F401
    import cccd.excel  # noqa: F401
    import cccd.mrz  # noqa: F401
    import cccd.pipeline  # noqa: F401
    import cccd.qr  # noqa: F401
    import pandas  # noqa: F401
    import pillow_heif  # noqa: F401


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định không in được tiếng Việt
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    port = _free_port()
    url = f"http://localhost:{port}"
    print("=" * 60)
    print(" Đọc CCCD -> Sổ khai báo lưu trú")
    print(f" Đang mở {url} ... (lần đầu có thể mất 10-20 giây)")
    print(" Giữ cửa sổ này mở trong lúc dùng. Đóng cửa sổ này để tắt ứng dụng.")
    print("=" * 60)
    threading.Timer(4, lambda: webbrowser.open(url)).start()

    from streamlit.web import cli as stcli

    sys.argv = [
        "streamlit", "run", os.path.join(base, "app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
        f"--server.port={port}",
        "--server.address=127.0.0.1",  # chỉ máy này truy cập được, không mở ra mạng LAN
        "--server.maxUploadSize=2000",
        "--server.fileWatcherType=none",
        "--browser.gatherUsageStats=false",
    ]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
