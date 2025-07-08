# RAG 평가 시스템 사용 가이드 (v2.0)

MOJI AI Agent의 RAG 시스템 성능을 평가하고 개선하기 위한 종합 가이드입니다.

**최신 업데이트 (2025-01-07)**: 품질 중심 최적화 반영, 의미론적 청킹, 하이브리드 검색 가중치 조정

## 📋 목차

1. [개요](#개요)
2. [설치 및 설정](#설치-및-설정)
3. [주요 구성 요소](#주요-구성-요소)
4. [사용법](#사용법)
5. [평가 메트릭](#평가-메트릭)
6. [결과 해석](#결과-해석)
7. [개선 방안](#개선-방안)
8. [최신 RAG 구성](#최신-rag-구성)
9. [고급 사용법](#고급-사용법)
10. [문제 해결](#문제-해결)

## 🎯 개요

RAG 평가 시스템은 다음과 같은 기능을 제공합니다:

- **자동화된 성능 평가**: RAGAS 메트릭을 사용한 객관적 성능 측정
- **시각적 분석**: 차트와 그래프를 통한 직관적 결과 확인
- **개선 추천**: AI 기반 성능 개선 방안 제시
- **배치 처리**: 대량 쿼리 동시 평가
- **캐싱 지원**: 효율적인 재평가
- **품질 중심 평가**: 응답 시간보다 답변 품질을 우선시하는 평가 기준
- **의미론적 청킹**: 문맥을 고려한 고급 문서 분할 평가

## 🔧 설치 및 설정

### 1. 의존성 설치

```bash
# 기본 의존성 설치
pip install -r requirements.txt

# RAGAS 추가 설치 (선택사항)
pip install ragas>=0.1.0

# 의미론적 청킹용 추가 패키지
pip install nltk sentence-transformers

# 가상환경 사용 권장
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. 벡터 스토어 관리

```bash
# 벡터 스토어만 초기화 (문서 재인덱싱 없이)
python3 clear_and_reload_docs.py --clear-only -y

# 전체 초기화 및 재인덱싱
python3 clear_and_reload_docs.py -y

# ChromaDB만 초기화 (FAISS는 유지)
python3 clear_and_reload_docs.py --clear-only --no-faiss -y
```

### 3. 디렉토리 구조 확인

```
agentmojichat/
├── app/
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── ragas_evaluator.py
│   │   └── metrics_dashboard.py
│   └── rag/
│       ├── semantic_chunker.py    # NEW: 의미론적 청킹
│       ├── enhanced_rag.py        # 업데이트됨
│       └── hybrid_search.py       # 업데이트됨
├── data/
│   ├── documents/          # RAG 문서들
│   ├── evaluation/         # 평가 결과 저장
│   └── demo_documents/     # 데모용 문서
├── scripts/
│   └── evaluation_demo.py  # 데모 스크립트 (확장됨)
├── rag_health_check.py     # 업데이트됨
├── upload_docs.py          # 업데이트됨
├── clear_and_reload_docs.py # 업데이트됨
└── vector_db_manager.py    # 업데이트됨
```

### 4. 환경 변수 설정

```bash
# .env 파일에 추가
LLM_PROVIDER=deepseek
LLM_API_KEY=your_api_key_here
OPENAI_API_KEY=your_openai_key_here  # 임베딩용
```

## 🧩 주요 구성 요소

### 1. RAGASEvaluator (`app/evaluation/ragas_evaluator.py`)

RAG 시스템의 핵심 평가 엔진입니다.

**주요 기능:**
- RAGAS 메트릭 계산
- 폴백 메트릭 제공
- 배치 평가 지원
- 결과 자동 저장

### 2. MetricsDashboard (`app/evaluation/metrics_dashboard.py`)

평가 결과를 시각화하고 분석하는 대시보드입니다.

**주요 기능:**
- 6가지 시각화 차트
- HTML 리포트 생성
- 성능 분석
- 개선 추천 생성

### 3. 평가 데모 스크립트 (`scripts/evaluation_demo.py`)

완전 자동화된 평가 데모를 제공합니다.

## 🚀 사용법

### 1. 빠른 시작 - 데모 실행

```bash
# 전체 평가 데모 실행
cd /home/smhaccp/dev/agentmojichat
python scripts/evaluation_demo.py

# 단일 쿼리 빠른 테스트
python scripts/evaluation_demo.py quick
```

### 2. 프로그래밍 방식 사용

#### 기본 사용법

```python
import asyncio
from app.rag.enhanced_rag import rag_pipeline
from app.evaluation.ragas_evaluator import RAGASEvaluator
from app.evaluation.metrics_dashboard import MetricsDashboard

async def basic_evaluation():
    # 1. 평가기 초기화
    evaluator = RAGASEvaluator(
        rag_pipeline=rag_pipeline,
        use_ragas=True
    )
    
    # 2. 단일 쿼리 평가
    result = await evaluator.evaluate_single_query(
        query="회사의 주요 기술 스택은 무엇인가요?",
        ground_truth="Python, FastAPI, LangChain"  # 선택사항
    )
    
    print(f"신뢰도: {result.faithfulness:.3f}")
    print(f"답변 관련성: {result.answer_relevancy:.3f}")
    
    return result

# 실행
asyncio.run(basic_evaluation())
```

#### 배치 평가

```python
async def batch_evaluation():
    evaluator = RAGASEvaluator(rag_pipeline=rag_pipeline)
    
    # 테스트 쿼리들
    queries = [
        "회사 이름이 무엇인가요?",
        "주요 기술 스택은 무엇인가요?",
        "직원 복리후생은 어떻게 되나요?"
    ]
    
    # 배치 평가 실행
    results, summary = await evaluator.evaluate_dataset(
        queries=queries,
        save_results=True
    )
    
    # 대시보드 생성
    dashboard = MetricsDashboard()
    report = dashboard.generate_report(results, summary, save_plots=True)
    html_path = dashboard.create_html_report(results, summary)
    
    print(f"HTML 리포트: {html_path}")
    return results, summary, report

# 실행
asyncio.run(batch_evaluation())
```

### 3. 커스텀 테스트 쿼리 사용

```python
# 자신만의 테스트 쿼리 정의
custom_queries = [
    "시스템 요구사항은 무엇인가요?",
    "설치 방법을 알려주세요",
    "API 사용법은 어떻게 되나요?",
    "문제 해결 방법은?"
]

# Ground truth 답변 (선택사항)
ground_truths = [
    "Python 3.11+, FastAPI, PostgreSQL 필요",
    "pip install 후 docker-compose up",
    "REST API 엔드포인트 /api/v1/ 사용",
    "로그 확인 후 이슈 트래커 등록"
]

# 평가 실행
results, summary = await evaluator.evaluate_dataset(
    queries=custom_queries,
    ground_truths=ground_truths
)
```

## 📊 평가 메트릭

### RAGAS 메트릭

| 메트릭 | 설명 | 범위 | 해석 |
|--------|------|------|------|
| **Faithfulness** | 답변이 제공된 컨텍스트에 얼마나 충실한가 | 0.0-1.0 | 높을수록 좋음 |
| **Answer Relevancy** | 답변이 질문과 얼마나 관련있는가 | 0.0-1.0 | 높을수록 좋음 |
| **Context Precision** | 검색된 컨텍스트의 정밀도 | 0.0-1.0 | 높을수록 좋음 |
| **Context Recall** | 관련 정보를 얼마나 잘 검색했는가 | 0.0-1.0 | 높을수록 좋음 |
| **Context Relevancy** | 컨텍스트가 질문과 얼마나 관련있는가 | 0.0-1.0 | 높을수록 좋음 |

### 폴백 메트릭

RAGAS 설치가 안 된 경우 사용되는 대체 메트릭:
- 키워드 기반 관련성 계산
- 답변 길이 기반 품질 평가
- 컨텍스트 존재 여부 확인

### 성능 메트릭

- **Response Time**: 응답 시간 (초)
- **Retrieval Time**: 문서 검색 시간
- **Generation Time**: 답변 생성 시간
- **Total Tokens**: 사용된 토큰 수

## 📈 결과 해석

### 품질 등급

| 평균 점수 | 등급 | 상태 |
|-----------|------|------|
| 0.8+ | Excellent | 🟢 매우 좋음 |
| 0.6-0.8 | Good | 🟡 좋음 |
| 0.4-0.6 | Fair | 🟠 보통 |
| <0.4 | Poor | 🔴 개선 필요 |

### 주요 지표 해석

1. **Faithfulness < 0.7**: 답변이 문서 내용과 일치하지 않음
2. **Answer Relevancy < 0.6**: 답변이 질문과 관련성이 낮음
3. **Context Precision < 0.5**: 검색된 문서들이 부정확함
4. **Response Time > 3초**: 응답 속도가 느림 (품질 우선 설정에서는 허용 범위)

### 생성되는 파일들

```
data/evaluation/
├── evaluation_results_YYYYMMDD_HHMMSS.json    # 상세 결과
├── evaluation_results_YYYYMMDD_HHMMSS.csv     # CSV 형태 결과
├── evaluation_summary_YYYYMMDD_HHMMSS.json    # 요약 통계
├── evaluation_report_YYYYMMDD_HHMMSS.html     # HTML 리포트
├── metrics_overview_YYYYMMDD_HHMMSS.png       # 개요 차트
├── detailed_analysis_YYYYMMDD_HHMMSS.png      # 상세 분석 차트
└── rag_report_YYYYMMDD_HHMMSS.json           # 종합 리포트
```

## 🔧 개선 방안

### 자동 추천 시스템

평가 결과에 따라 다음과 같은 개선 방안이 자동으로 제안됩니다:

1. **신뢰도 개선**
   - 더 정확한 문서 청킹
   - 컨텍스트 필터링 강화
   - 검색 임계값 조정

2. **관련성 개선**
   - 쿼리 재작성 활용
   - 더 나은 검색 알고리즘 사용
   - 의미론적 검색 강화

3. **성능 개선**
   - 캐싱 시스템 활용
   - 더 빠른 임베딩 모델 사용
   - 병렬 처리 최적화

4. **컨텍스트 품질 개선**
   - 리랭킹 시스템 도입
   - 하이브리드 검색 사용
   - 문서 품질 개선

### 수동 개선 작업

1. **문서 개선**
   ```bash
   # 문서 품질 확인
   python scripts/document_analyzer.py
   
   # 중복 문서 제거
   python scripts/deduplicate_docs.py
   ```

2. **검색 설정 조정**
   ```python
   # hybrid_search.py에서 설정 수정
   vector_weight=0.4,      # 벡터 검색 가중치
   keyword_weight=0.4,     # 키워드 검색 가중치 (증가)
   score_threshold=0.05,   # 더 낮은 임계값으로 recall 향상
   ```

3. **청킹 전략 변경 (최신 권장 설정)**
   ```python
   # enhanced_rag.py에서 최신 설정 사용
   use_semantic_chunking=True,  # 의미론적 청킹 활성화
   chunk_size=1500,            # 품질 향상을 위한 큰 청크
   chunk_overlap=500           # 경계 정보 보존 강화
   ```

## 🆕 최신 RAG 구성

### 품질 중심 최적화 (2025-07-07 업데이트)

최신 RAG 시스템은 응답 속도보다 답변 품질을 우선시하도록 최적화되었습니다.

#### 주요 변경사항

1. **청킹 파라미터 향상**
   ```python
   # 기존 설정
   chunk_size=1000
   chunk_overlap=200
   
   # 새로운 설정 (품질 향상)
   chunk_size=1500      # 더 많은 컨텍스트 포함
   chunk_overlap=500    # 경계 정보 보존 강화
   ```

2. **하이브리드 검색 가중치 조정**
   ```python
   # 기존 가중치
   vector_weight=0.5
   keyword_weight=0.3
   bm25_weight=0.2
   
   # 새로운 가중치 (키워드 검색 강화)
   vector_weight=0.4
   keyword_weight=0.4   # 키워드 검색 가중치 증가
   bm25_weight=0.2
   ```

3. **의미론적 청킹 활성화**
   ```python
   use_semantic_chunking=True  # 기본값으로 활성화
   ```

4. **향상된 쿼리 처리**
   - 도메인별 키워드 변형 사전 추가
   - Chain-of-Thought 한국어 추론 체인 도입
   - 더 정교한 쿼리 재작성 로직

#### 성능 임계값 조정

```python
# 품질 우선 평가 기준
PERFORMANCE_THRESHOLDS = {
    "response_time": 3.0,        # 기존 2.0초 → 3.0초
    "faithfulness": 0.8,         # 신뢰도 임계값 상향
    "answer_relevancy": 0.7,     # 관련성 임계값 상향
    "context_precision": 0.6     # 정밀도 임계값 상향
}
```

### 업데이트된 유틸리티 도구들

#### 1. RAG 상태 점검 (`rag_health_check.py`)
```bash
# 새로운 점검 항목들
python rag_health_check.py

# 확인 사항:
# - 의미론적 청킹 상태
# - 업데이트된 청크 파라미터
# - 새로운 성능 임계값 (3초)
# - 하이브리드 검색 가중치
```

#### 2. 문서 업로드 도구 (`upload_docs.py`)
```bash
python upload_docs.py

# 새로운 통계 표시:
# - 청크 중복 정보
# - 의미론적 청킹 상태
# - 향상된 RAG 명령어 예시
```

#### 3. 벡터 DB 관리자 (`vector_db_manager.py`)
```bash
python vector_db_manager.py stats

# 한글화된 통계 표시:
# - 총 문서 수: XX개
# - 청크 크기: 1500
# - 청크 중복: 500
# - 의미론적 청킹: 활성화
```

### 평가 시 고려사항

1. **응답 시간 평가**
   - 3초 이내: 우수
   - 3-5초: 양호 (품질 우선)
   - 5초 이상: 개선 필요

2. **품질 지표 우선순위**
   ```
   1순위: Faithfulness (신뢰도)
   2순위: Answer Relevancy (답변 관련성)
   3순위: Context Precision (컨텍스트 정밀도)
   4순위: Response Time (응답 시간)
   ```

3. **의미론적 청킹 평가**
   ```python
   # 의미론적 청킹 활성화 확인
   if stats.get('use_semantic_chunking'):
       # 더 높은 품질 기준 적용
       quality_threshold = 0.8
   else:
       # 표준 품질 기준
       quality_threshold = 0.7
   ```

### 설정 검증 명령어

```bash
# 현재 RAG 설정 확인
python rag_health_check.py

# 의미론적 청킹 상태 확인
python upload_docs.py --no-test

# 하이브리드 검색 테스트
python scripts/evaluation_demo.py quick
```

## 🔬 고급 사용법

### 1. 벡터 스토어 관리

```bash
# 벡터 스토어 상태 확인
python3 clear_and_reload_docs.py --clear-only --help

# 선택적 초기화 옵션들
python3 clear_and_reload_docs.py --clear-only              # 확인 후 초기화
python3 clear_and_reload_docs.py --clear-only -y           # 즉시 초기화
python3 clear_and_reload_docs.py --clear-only --no-faiss   # ChromaDB만 초기화
python3 clear_and_reload_docs.py --clear-only --no-faiss -y # ChromaDB만 즉시 초기화
```

**주요 옵션들:**
- `--clear-only`: 벡터 스토어만 삭제하고 문서 재인덱싱은 건너뛰기
- `-y, --yes`: 확인 프롬프트 없이 자동으로 진행
- `--no-faiss`: FAISS 인덱스는 유지하고 ChromaDB만 초기화

### 2. 커스텀 메트릭 추가

```python
def custom_metric(result: EvaluationResult) -> float:
    """커스텀 평가 메트릭"""
    # 답변 길이와 품질의 균형 점수
    length_score = min(len(result.answer) / 200, 1.0)
    quality_score = (result.faithfulness + result.answer_relevancy) / 2
    return (length_score + quality_score) / 2

# 평가 결과에 추가
for result in results:
    result.custom_score = custom_metric(result)
```

### 3. A/B 테스트

```python
async def ab_test_chunking_strategies():
    """청킹 전략 A/B 테스트"""
    
    # 전략 A: 기존 설정
    pipeline_a = EnhancedRAGPipeline(
        use_semantic_chunking=False,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 전략 B: 최신 품질 중심 설정
    pipeline_b = EnhancedRAGPipeline(
        use_semantic_chunking=True,
        chunk_size=1500,
        chunk_overlap=500
    )
    
    test_queries = ["테스트 쿼리1", "테스트 쿼리2"]
    
    # 각 전략 평가
    evaluator_a = RAGASEvaluator(pipeline_a)
    evaluator_b = RAGASEvaluator(pipeline_b)
    
    results_a, summary_a = await evaluator_a.evaluate_dataset(test_queries)
    results_b, summary_b = await evaluator_b.evaluate_dataset(test_queries)
    
    # 결과 비교
    print(f"전략 A 평균 점수: {summary_a.avg_faithfulness:.3f}")
    print(f"전략 B 평균 점수: {summary_b.avg_faithfulness:.3f}")
```

### 4. 연속 모니터링

```python
import schedule
import time

def scheduled_evaluation():
    """정기적 평가 실행"""
    asyncio.run(batch_evaluation())
    
# 매일 자정에 평가 실행
schedule.every().day.at("00:00").do(scheduled_evaluation)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## 🔍 문제 해결

### 자주 발생하는 문제들

1. **RAGAS 설치 실패**
   ```bash
   # 해결: 폴백 메트릭 자동 사용
   # 또는 수동 설치
   pip install ragas datasets evaluate
   ```

2. **메모리 부족**
   ```python
   # 배치 크기 줄이기
   evaluator = RAGASEvaluator(batch_size=5)
   ```

3. **느린 평가 속도**
   ```python
   # 병렬 처리 비활성화
   use_parallel_search=False
   
   # 캐싱 활용
   use_ragas=False  # 폴백 메트릭 사용
   ```

4. **권한 오류**
   ```bash
   # 디렉토리 권한 확인
   chmod 755 data/evaluation/
   ```

### 로그 확인

```bash
# 평가 로그 확인
tail -f logs/evaluation.log

# 오류 로그만 확인
grep ERROR logs/evaluation.log
```

### 성능 최적화

1. **캐싱 활용**: 동일 쿼리 재평가 방지
2. **배치 크기 조정**: 메모리와 속도의 균형
3. **병렬 처리**: CPU 코어 수에 맞게 조정
4. **임베딩 캐싱**: 동일 문서 재처리 방지

## 📚 추가 자료

- [RAGAS 공식 문서](https://docs.ragas.io/)
- [LangChain 평가 가이드](https://python.langchain.com/docs/guides/evaluation/)
- [RAG 성능 최적화 가이드](./RAG_OPTIMIZATION_GUIDE.md)

## 🤝 기여하기

평가 시스템 개선에 기여하고 싶다면:

1. 새로운 메트릭 제안
2. 시각화 개선
3. 성능 최적화
4. 문서 개선

문의사항이나 버그 리포트는 이슈 트래커에 등록해 주세요.

---

*이 가이드는 MOJI AI Agent RAG 평가 시스템 v2.0을 기준으로 작성되었습니다. (품질 중심 최적화 반영)*