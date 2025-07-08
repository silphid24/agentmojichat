# RAG_PARAMETER_GUIDE.md

## 📋 개요

이 문서는 MOJI AI Agent의 RAG(Retrieval-Augmented Generation) 시스템에서 사용할 수 있는 모든 중요한 파라미터들의 위치와 기능을 상세히 설명합니다. 각 파라미터를 조정하여 성능과 품질의 최적 균형점을 찾을 수 있습니다.

## 🎯 파라미터 카테고리

### 1. 청킹(Chunking) 파라미터
### 2. 검색(Search) 파라미터  
### 3. 하이브리드 검색 가중치
### 4. 성능 최적화 파라미터
### 5. 품질 향상 파라미터

---

## 📍 **1. 청킹(Chunking) 파라미터**

### 🔧 **기본 청킹 설정**

**파일 위치**: `app/rag/enhanced_rag.py` (43-47행)

```python
def __init__(
    self,
    chunk_size: int = 1500,              # 청크 크기 (문자 수)
    chunk_overlap: int = 500,            # 청크 중복 (문자 수)
    use_semantic_chunking: bool = True,  # 의미론적 청킹 사용
):
```

#### **파라미터 설명**

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| `chunk_size` | 1500 | 300-3000 | 각 텍스트 청크의 최대 문자 수 |
| `chunk_overlap` | 500 | 50-1000 | 인접 청크 간 중복되는 문자 수 |
| `use_semantic_chunking` | True | True/False | 의미론적 청킹 활성화 여부 |

#### **조정 가이드**

**`chunk_size` (청크 크기)**
- **작게 설정 (500-1000자)**:
  - ✅ 정확한 검색 결과
  - ✅ 빠른 처리 속도
  - ❌ 컨텍스트 부족 위험
  
- **크게 설정 (1500-2500자)**:
  - ✅ 풍부한 컨텍스트
  - ✅ 완전한 정보 포함
  - ❌ 검색 노이즈 증가
  - ❌ 느린 처리 속도

**`chunk_overlap` (청크 중복)**
- **권장값**: chunk_size의 20-50%
- **낮은 값 (100-200자)**: 중복 최소화, 경계 정보 손실 위험
- **높은 값 (400-800자)**: 경계 정보 보존, 저장 공간 증가

---

### 🧩 **의미론적 청킹 세부 설정**

**파일 위치**: `app/rag/semantic_chunker.py` (48-54행)

```python
def __init__(
    self,
    similarity_threshold: float = 0.5,    # 유사도 임계값
    use_structure_hints: bool = True,     # 구조 힌트 사용
):
```

#### **조정 가이드**

**`similarity_threshold` (유사도 임계값)**
- **낮은 값 (0.3-0.5)**: 더 많은 문장을 하나의 청크로 그룹화
- **높은 값 (0.6-0.8)**: 더 엄격한 의미 단위로 분할
- **권장값**: 0.4-0.7

---

## 📍 **2. 검색(Search) 파라미터**

### 🔍 **기본 검색 설정**

**파일 위치**: `app/rag/enhanced_rag.py`

```python
# 쿼리 재작성과 함께 검색 (522행)
async def search_with_rewriting(
    self,
    query: str,
    k: int = 5,                      # 검색할 문서 수
    score_threshold: float = 2.0,    # 점수 임계값
):

# 신뢰도와 함께 답변 생성 (795행)
async def answer_with_confidence(
    self,
    query: str,
    k: int = 5,                      # 검색할 문서 수
    score_threshold: float = 1.5,    # 점수 임계값
):
```

#### **파라미터 설명**

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| `k` | 5 | 1-20 | 검색에서 가져올 문서 개수 |
| `score_threshold` | 1.5-2.0 | 0.1-5.0 | 유사도 점수 임계값 |

#### **조정 가이드**

**`k` (검색 문서 수)**
- **작은 값 (1-3개)**: 빠른 처리, 높은 정확도, 정보 부족 위험
- **큰 값 (7-15개)**: 풍부한 정보, 노이즈 증가, 느린 처리

**`score_threshold` (점수 임계값)**
- **낮은 값 (0.5-1.5)**: 더 많은 후보 문서 포함 (재현율 ↑)
- **높은 값 (2.0-4.0)**: 고품질 문서만 선택 (정밀도 ↑)
- **언어별 권장값**: 한국어 1.0-3.0, 영어 0.5-2.0

---

## 📍 **3. 하이브리드 검색 가중치**

### ⚖️ **검색 방식별 가중치**

**파일 위치**: `app/rag/hybrid_search.py` (83-87행)

```python
def __init__(
    self,
    vector_weight: float = 0.4,    # 벡터 검색 가중치
    keyword_weight: float = 0.4,   # 키워드 검색 가중치
    bm25_weight: float = 0.2,      # BM25 가중치
):
```

#### **가중치 설명**

| 검색 방식 | 기본값 | 범위 | 특징 |
|----------|--------|------|------|
| `vector_weight` | 0.4 | 0.1-0.8 | 의미적 유사성 중심 |
| `keyword_weight` | 0.4 | 0.1-0.7 | 정확한 단어 일치 중심 |
| `bm25_weight` | 0.2 | 0.1-0.5 | 전통적 정보검색 방식 |

> **⚠️ 중요**: 모든 가중치의 합은 1.0이 되어야 합니다.

#### **시나리오별 권장 설정**

**의미 검색 중심** (복잡한 질문, 개념적 질의):
```python
vector_weight = 0.6
keyword_weight = 0.25
bm25_weight = 0.15
```

**키워드 검색 중심** (정확한 용어, 기술 문서):
```python
vector_weight = 0.3
keyword_weight = 0.5
bm25_weight = 0.2
```

**균형 검색** (일반적 사용):
```python
vector_weight = 0.4
keyword_weight = 0.4
bm25_weight = 0.2
```

---

## 📍 **4. BM25 알고리즘 파라미터**

**파일 위치**: `app/rag/hybrid_search.py` (26-27행)

```python
def __init__(self, corpus: List[str], k1: float = 1.2, b: float = 0.75):
    self.k1 = k1  # 용어 빈도 포화점
    self.b = b    # 문서 길이 정규화 강도
```

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| `k1` | 1.2 | 1.0-3.0 | 용어 빈도 포화점 |
| `b` | 0.75 | 0.0-1.0 | 문서 길이 정규화 강도 |

---

## 📍 **5. 성능 최적화 파라미터**

**파일 위치**: `app/rag/enhanced_rag.py` (36-39행)

```python
# 병렬 처리 관리자들
self.parallel_search_manager = ParallelSearchManager(max_concurrent=4)
self.batch_processor = AsyncBatchProcessor(batch_size=10, max_concurrent=3)
```

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|--------|------|------|
| `max_concurrent` | 4 | 1-16 | 동시 실행 가능한 검색 수 |
| `batch_size` | 10 | 5-50 | 배치 처리 크기 |

---

## 🎯 **시나리오별 최적 설정**

### 🚀 **성능 우선 설정** (빠른 응답)

```python
# 청킹 설정
chunk_size = 800
chunk_overlap = 200
use_semantic_chunking = False

# 검색 설정
k = 3
score_threshold = 0.5

# 하이브리드 가중치
vector_weight = 0.6
keyword_weight = 0.4
bm25_weight = 0.0
```

**예상 성능**: 응답시간 1-2초, 정확도 70-80%

### 🎯 **품질 우선 설정** (정확한 답변)

```python
# 청킹 설정
chunk_size = 1500
chunk_overlap = 500
use_semantic_chunking = True
similarity_threshold = 0.6

# 검색 설정
k = 7
score_threshold = 0.1

# 하이브리드 가중치
vector_weight = 0.4
keyword_weight = 0.4
bm25_weight = 0.2
```

**예상 성능**: 응답시간 3-5초, 정확도 90-95%

### ⚖️ **균형 설정** (성능과 품질 균형)

```python
# 청킹 설정
chunk_size = 1200
chunk_overlap = 300
use_semantic_chunking = True

# 검색 설정
k = 5
score_threshold = 0.2

# 하이브리드 가중치
vector_weight = 0.5
keyword_weight = 0.3
bm25_weight = 0.2
```

**예상 성능**: 응답시간 2-3초, 정확도 85-90%

---

## 🛠️ **파라미터 수정 방법**

### 1. **코드 직접 수정**
각 파일의 해당 위치에서 파라미터 값을 직접 수정

### 2. **환경 변수 사용** (권장)
`app/core/config.py`에 RAG 설정 추가 후 환경 변수로 제어

### 3. **런타임 동적 설정**
테스트용으로 런타임에 파라미터 변경

---

## 📊 **성능 모니터링**

### **모니터링 도구**

```bash
# RAG 시스템 상태 확인
python3 rag_health_check.py

# 벡터 DB 통계 확인
python3 vector_db_manager.py stats

# 대화형 테스트
python3 tools/interactive_chat.py
```

---

## 🔧 **튜닝 가이드라인**

### **단계별 최적화 접근법**

1. **기본 설정으로 시작**: 기본 균형 설정 사용
2. **성능 요구사항 확인**: 응답시간 vs 정확도 우선순위 결정
3. **점진적 조정**: 한 번에 하나의 파라미터만 변경
4. **테스트 및 검증**: 각 변경사항의 효과 측정
5. **최적값 도출**: 여러 테스트를 통한 최적 조합 찾기

### **주의사항**

- **메모리 사용량**: 큰 chunk_size와 높은 k 값은 메모리 사용량 증가
- **일관성 유지**: 관련 파라미터들 간의 균형 고려
- **도메인 특성**: 문서 유형과 질문 패턴에 맞는 조정 필요
- **지속적 모니터링**: 실제 사용 패턴에 따른 지속적 최적화

---

**📌 마지막 업데이트**: 2025-07-07  
**📌 버전**: 1.0.0  
**📌 작성자**: MOJI AI Agent Development Team
