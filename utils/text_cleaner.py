# PDF는 반드시 텍스트 정제가 필요하다.
# 추가 확장 가능:
# 페이지 번호 제거
# 머리글/바닥글 제거
# 특수문자 정리
# 표 텍스트 보정

import re

def clean_text(text: str) -> str:

    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\u200b", "", text)

    return text.strip()
