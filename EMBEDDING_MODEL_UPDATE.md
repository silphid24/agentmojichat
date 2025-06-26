# OpenAI 임베딩 모델 업데이트

## 변경 사항
OpenAI 임베딩 모델을 `text-embedding-ada-002`에서 `text-embedding-3-small`로 변경했습니다.

## 변경된 파일
1. `app/rag/enhanced_rag.py` - RAG 파이프라인의 임베딩 설정
2. `app/rag/embeddings.py` - 전역 임베딩 모델 설정

## text-embedding-3-small 특징
- **성능**: ada-002보다 향상된 성능
- **비용**: 더 저렴한 가격 (ada-002 대비 약 5배 저렴)
- **크기**: 512차원 (ada-002는 1536차원)
- **속도**: 더 빠른 처리 속도

## 장점
1. **비용 절감**: 임베딩 생성 비용이 크게 감소
2. **속도 향상**: 더 작은 차원으로 인한 빠른 처리
3. **저장 공간 절약**: 벡터 크기가 작아 DB 용량 절약
4. **성능 유지**: 대부분의 사용 사례에서 충분한 성능

## 주의사항
- 기존 ada-002로 생성된 임베딩과 호환되지 않음
- 새 모델 사용 시 기존 벡터 DB를 재생성해야 함

## 사용 방법
```bash
# 기존 벡터 DB 삭제 (필요시)
rm -rf data/vectordb/*

# 문서 재업로드
python3 upload_docs.py
```