# 
# 파일명 정규화
# 확장자 검사
# 안전한 경로 생성
# 파일 존재 체크
# 해시 생성
# 
import os
import hashlib

def normalize_filename(filename: str) -> str:
    return filename.replace(" ", "_")

def is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")

def generate_file_hash(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()
