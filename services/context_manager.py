# 1️⃣ Context Window Manager 설계
# 역할:
# 토큰 수 계산
# 초과 시 문서 제거
# 혹은 compression 호출

# ✔ 상위 문서부터 채운다
# ✔ 답변 공간은 reserved_tokens로 남긴다

from transformers import AutoTokenizer
from config import LLM_MODEL


class ContextWindowManager:

    def __init__(self, max_tokens=4000, reserved_tokens=1000):
        self.max_tokens = max_tokens
        self.reserved_tokens = reserved_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def trim(self, docs):

        total = 0
        allowed = self.max_tokens - self.reserved_tokens
        selected = []

        for doc in docs:
            tokens = self.count_tokens(doc.page_content)

            if total + tokens > allowed:
                break

            selected.append(doc)
            total += tokens

        return selected

