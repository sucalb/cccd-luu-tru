#!/bin/bash
# Bấm đúp để chạy trên Mac
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  # rapidocr / vietocr kéo theo bản opencv thường, ghi đè bản contrib (cần cho bộ đọc QR WeChat)
  .venv/bin/pip uninstall -y opencv-python opencv-python-headless
  .venv/bin/pip install --force-reinstall --no-deps opencv-contrib-python-headless "pillow>=11.1"
fi
.venv/bin/streamlit run app.py --server.maxUploadSize 2000 --browser.gatherUsageStats false
