"""Telegram alert — gửi ảnh + mô tả vi phạm. Mock mode in ra console."""
import requests


class TelegramAlerter:
    def __init__(self, token="", chat_id="", mock=True):
        self.token = token
        self.chat_id = chat_id
        self.mock = mock or not (token and chat_id)

    def send(self, text, image_path=None):
        if self.mock:
            print(f"[TELEGRAM MOCK] {text}" + (f" | image: {image_path}" if image_path else ""))
            return True
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.chat_id, "text": text}, timeout=10)
            if image_path:
                url2 = f"https://api.telegram.org/bot{self.token}/sendPhoto"
                with open(image_path, "rb") as f:
                    requests.post(url2, data={"chat_id": self.chat_id},
                                  files={"photo": f}, timeout=15)
            return True
        except Exception as e:
            print(f"[TELEGRAM FAIL] {e}")
            return False
