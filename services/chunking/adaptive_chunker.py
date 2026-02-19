# PDF → Clean
#      → Paragraph Split
#      → Document Profiling
#      → Adaptive Size 결정
#      → Semantic Merge
#      → Overlap 적용
#      → Metadata Attach

# 🔥 이 코드가 하는 일
# 1️⃣ 문단 기준 분리
# 2️⃣ 문서 평균 길이 분석
# 3️⃣ 문서 특성에 맞는 target_size 계산
# 4️⃣ embedding 기반 semantic similarity 검사
# 5️⃣ 의미적으로 가까운 문단만 병합
# 6️⃣ overlap 자동 적용
# 🎯 이 설계의 장점
# ✔ 고정 chunk_size 제거
# ✔ 의미 단위 유지
# ✔ 긴 문서 대응
# ✔ 매뉴얼/법률 대응
# ✔ Reranker 효율 상승
# ✔ Compression 비용 감소

# ⚠️ 성능 주의
# Semantic 단계는 embedding을 문단 단위로 전부 계산하므로:
# 문서가 매우 길면 느릴 수 있음
# Worker 비동기 처리 필수
# 엔터프라이즈에서는:
# 1000 paragraph 이상이면 semantic merge skip
# 또는 batch embedding 처리

import numpy as np
from typing import List
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity


class AdaptiveSemanticChunker:

    def __init__(
        self,
        embedding_model,
        base_chunk_size: int = 800,
        min_chunk_size: int = 500,
        max_chunk_size: int = 1400,
        similarity_threshold: float = 0.80,
        overlap_ratio: float = 0.15
    ):
        self.embedding_model = embedding_model
        self.base_chunk_size = base_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.overlap_ratio = overlap_ratio

    # -------------------------
    # Public API
    # -------------------------

    def chunk(self, documents: List[Document]) -> List[Document]:

        paragraphs = self._split_into_paragraphs(documents)

        profile = self._analyze(paragraphs)

        adaptive_size = self._choose_chunk_size(profile)

        merged = self._semantic_merge(paragraphs, adaptive_size)

        final_chunks = self._apply_overlap(merged)

        return final_chunks

    # -------------------------
    # Step 1: Paragraph Split
    # -------------------------

    def _split_into_paragraphs(self, documents):

        paragraphs = []

        for doc in documents:
            splits = doc.page_content.split("\n\n")
            for text in splits:
                text = text.strip()
                if len(text) > 50:
                    paragraphs.append(
                        Document(
                            page_content=text,
                            metadata=doc.metadata
                        )
                    )

        return paragraphs

    # -------------------------
    # Step 2: Profiling
    # -------------------------

    def _analyze(self, paragraphs):

        lengths = [len(p.page_content) for p in paragraphs]

        avg_len = np.mean(lengths)
        total_len = np.sum(lengths)

        return {
            "avg_len": avg_len,
            "total_len": total_len
        }

    # -------------------------
    # Step 3: Adaptive Size
    # -------------------------

    def _choose_chunk_size(self, profile):

        avg_len = profile["avg_len"]

        if avg_len > 1200:
            return min(self.max_chunk_size, int(avg_len * 1.2))
        elif avg_len > 700:
            return int(self.base_chunk_size)
        else:
            return max(self.min_chunk_size, int(avg_len * 2))

    # -------------------------
    # Step 4: Semantic Merge
    # -------------------------

    def _semantic_merge(self, paragraphs, target_size):

        if not paragraphs:
            return []

        embeddings = self.embedding_model.embed_documents(
            [p.page_content for p in paragraphs]
        )

        merged_chunks = []
        current_chunk = paragraphs[0].page_content
        current_embedding = embeddings[0]

        for i in range(1, len(paragraphs)):

            next_text = paragraphs[i].page_content
            next_embedding = embeddings[i]

            similarity = cosine_similarity(
                [current_embedding],
                [next_embedding]
            )[0][0]

            combined_length = len(current_chunk) + len(next_text)

            if (
                similarity > self.similarity_threshold
                and combined_length < target_size
            ):
                current_chunk += "\n\n" + next_text
                current_embedding = (
                    current_embedding + next_embedding
                ) / 2

            else:
                merged_chunks.append(current_chunk)
                current_chunk = next_text
                current_embedding = next_embedding

        merged_chunks.append(current_chunk)

        return [
            Document(page_content=chunk, metadata={})
            for chunk in merged_chunks
        ]

    # -------------------------
    # Step 5: Overlap
    # -------------------------

    def _apply_overlap(self, chunks):

        if not chunks:
            return []

        final_chunks = []

        overlap_size = int(self.overlap_ratio * self.base_chunk_size)

        for i, chunk in enumerate(chunks):

            text = chunk.page_content

            if i > 0:
                prev_text = chunks[i - 1].page_content
                overlap_text = prev_text[-overlap_size:]
                text = overlap_text + "\n\n" + text

            final_chunks.append(
                Document(
                    page_content=text,
                    metadata=chunk.metadata
                )
            )

        return final_chunks
