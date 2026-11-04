import re
import unicodedata


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[@#]\S+', '', text)
    # 顔文字・記号の除去（日本語文字・基本句読点・英数字は保持）
    text = re.sub(r'[^\w　-鿿゠-ヿ぀-ゟ\s、。！？]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
