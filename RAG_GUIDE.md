# MOJI RAG (검색 증강 생성) 가이드 (2024 최신)

MOJI의 문서 지식 베이스를 활용한 RAG 시스템 사용 및 테스트 완전 가이드입니다.

## 주요 특징

- 하이브리드 검색(벡터+키워드)
- 리랭킹 시스템, 신뢰도 점수, 출처 인용
- 적응형 쿼리 처리, 실시간 성능 모니터링
- WebChat v2와의 통합

## 빠른 시작

### 1. 문서 준비

프로젝트 루트에 `data/documents/` 폴더 생성:
```bash
mkdir -p documents
```

문서 추가:
- 지원 형식: `.txt`, `.md`, `.pdf`, `.docx`
- 권장사항: 텍스트 또는 마크다운 파일로 시작
- `documents/` 폴더에 파일 배치

예시 문서 구조:
```
documents/
├── company_policies.md
├── product_guide.txt
├── faq.md
└── technical_specs.pdf
```

### 2. 문서 업로드

```bash
# 모든 문서 업로드
python upload_docs.py

# 특정 파일 업로드
python upload_docs.py --file product_guide.txt

# 특정 폴더 업로드
python upload_docs.py --folder policies/

# 증분 업데이트 (변경된 파일만)
python upload_docs.py --incremental

# 배치 크기 설정
python upload_docs.py --batch-size 10
```

### 3. 서버 실행 및 RAG 활성화

.env 파일에 아래 항목 포함:
```env
RAG_ENABLED=true
VECTOR_STORE_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

서버 실행:
```bash
./run_server.sh
```

### 4. WebChat v2에서 RAG 사용

- http://localhost:8000/static/moji-webchat-v2.html 접속 후 `/rag` 명령어 사용

## RAG 명령어

### 기본 쿼리
```
/rag 반품 정책은 무엇인가요?
```

### 출처와 함께 쿼리
```
/rag-sources 제품 보증은 어떻게 작동하나요?
```

### 시스템 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `/rag` | 지식 베이스 검색 | `/rag 어떤 기능이 포함되어 있나요?` |
| `/rag-help` | RAG 도움말 표시 | `/rag-help` |
| `/rag-stats` | 통계 표시 | `/rag-stats` |
| `/rag-sources` | 출처 인용과 함께 검색 | `/rag-sources 가격은 얼마인가요?` |
| `/rag-clear` | 벡터 스토어 초기화 | `/rag-clear` |

## 고급 기능

### 1. 쿼리 재작성

MOJI는 더 나은 결과를 위해 쿼리를 자동으로 개선합니다:
- 원본: "얼마야?"
- 재작성: "제품의 가격은 얼마입니까?"

### 2. 신뢰도 점수

각 응답에는 신뢰도 지표가 포함됩니다:
- 🟢 높은 신뢰도 (>0.8): 문서에서 직접 답변
- 🟡 중간 신뢰도 (0.5-0.8): 부분 일치, 추론 포함 가능
- 🔴 낮은 신뢰도 (<0.5): 제한된 정보

### 3. 출처 표시

응답에는 출처 참조가 포함됩니다:
```
[product_guide.txt:15]에 따르면, 보증 기간은 2년입니다...
```

### 4. 다중 문서 통합

MOJI는 여러 문서의 정보를 결합할 수 있습니다:
```
[faq.md:23]과 [policies.md:45]를 기반으로, 반품 절차는...
```

### 5. 하이브리드 검색

벡터 검색과 키워드 검색 결합:
```python
# app/rag/enhanced_rag.py에서
enable_hybrid_search=True
keyword_weight=0.3
vector_weight=0.7
```

### 6. 리랭킹

리랭킹 시스템을 통해 최적의 결과를 찾습니다:
```python
# app/rag/reranking.py에서
rerank_enabled=True
```

### 7. 적응형 기능

쿼리를 실시간으로 처리하고 성능을 모니터링합니다:
```python
# app/rag/adaptive_query.py에서
adaptive_enabled=True
```

## 성능 최적화

### 1. 문서 준비
- 문서를 집중적이고 잘 구조화된 상태로 유지
- 명확한 제목과 섹션 사용
- 매우 큰 파일은 분할 (10MB 초과 시)

### 2. 인덱싱 전략

#### 기본 업로드 옵션
```bash
# 모든 문서 업로드
python upload_docs.py

# 특정 폴더만 업로드
python upload_docs.py --folder policies/

# 특정 파일만 업로드  
python upload_docs.py --file guide.txt

# 증분 업데이트 (변경된 파일만)
python upload_docs.py --incremental

# 배치 크기 조정
python upload_docs.py --batch-size 5

# 강제 전체 재처리
python upload_docs.py --incremental --force
```

#### 완전 재인덱싱
```bash
# 벡터 스토어 완전 초기화 및 재인덱싱
python clear_and_reload_docs.py
```

#### 문서 관리
```bash
# 문서 목록 조회
python manage_docs.py list

# 특정 폴더 문서 조회 (메타데이터 포함)
python manage_docs.py list --folder policies/ --metadata

# 문서 검색
python manage_docs.py search "시스템 사양"

# 시스템 통계
python manage_docs.py stats

# 고아 파일 정리
python manage_docs.py cleanup

# 백업 생성
python manage_docs.py backup ./backup/2024-01-01/

# 백업 복원
python manage_docs.py restore ./backup/2024-01-01/
```

#### 시스템 상태 점검
```bash
# 전체 상태 점검
python rag_health_check.py

# JSON 형식 출력
python rag_health_check.py --json

# 결과 파일로 저장
python rag_health_check.py --save health_report.json

# 조용한 모드
python rag_health_check.py --quiet
```

### 3. 쿼리 최적화
- 구체적인 키워드 사용
- 한 번에 한 가지 질문
- 필요시 컨텍스트 포함

### 4. 성능 최적화 팁
- 문서 분할(청크) 크기 조정
- 캐싱, 병렬 검색, 모델 예열
- 실시간 모니터링 활용

## 테스트 시나리오

### 1. 기본 검색 테스트
```bash
# 테스트 문서 업로드
echo "MOJI는 다양한 작업을 돕는 AI 어시스턴트입니다." > documents/test.txt
python upload_docs.py

# 내용 쿼리
/rag MOJI가 무엇인가요?
```

### 2. 정확도 테스트
```bash
# 특정 테스트 내용 생성
cat > documents/test_accuracy.md << EOF
# 제품 사양
- 모델: MOJI-2024
- 버전: 1.0.0
- 메모리: 16GB
- 저장 공간: 512GB
EOF

python upload_docs.py

# 특정 쿼리 테스트
/rag 모델 번호는 무엇인가요?
/rag 메모리는 얼마나 되나요?
```

### 3. 컨텍스트 윈도우 테스트
```bash
# 긴 컨텍스트로 테스트
/rag 문서에 언급된 모든 기능을 설명해주세요
```

### 4. 부정 테스트
```bash
# 존재하지 않는 정보 쿼리
/rag 제품의 색상은 무엇인가요?
# 낮은 신뢰도 또는 정보 없음 표시되어야 함
```

## 설정

### 벡터 스토어 설정

`.env` 파일에서:
```bash
# Chroma DB 설정
VECTOR_STORE_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# RAG 설정
RAG_ENABLED=true
RAG_MAX_RESULTS=5
RAG_SIMILARITY_THRESHOLD=0.7
```

### 문서 처리

```python
# app/config.py에서
RAG_CONFIG = {
    "chunk_size": 1000,          # 청크당 문자 수
    "chunk_overlap": 200,        # 청크 간 중복
    "max_chunks_per_query": 5,   # 검색할 최대 청크 수
    "min_similarity_score": 0.7  # 최소 유사도 임계값
}
```

## 성능 최적화

### 1. 문서 준비
- 문서를 집중적이고 잘 구조화된 상태로 유지
- 명확한 제목과 섹션 사용
- 매우 큰 파일은 분할 (10MB 초과 시)

### 2. 인덱싱 전략

#### 기본 업로드 옵션
```bash
# 모든 문서 업로드
python upload_docs.py

# 특정 폴더만 업로드
python upload_docs.py --folder policies/

# 특정 파일만 업로드  
python upload_docs.py --file guide.txt

# 증분 업데이트 (변경된 파일만)
python upload_docs.py --incremental

# 배치 크기 조정
python upload_docs.py --batch-size 5

# 강제 전체 재처리
python upload_docs.py --incremental --force
```

#### 완전 재인덱싱
```bash
# 벡터 스토어 완전 초기화 및 재인덱싱
python clear_and_reload_docs.py
```

#### 문서 관리
```bash
# 문서 목록 조회
python manage_docs.py list

# 특정 폴더 문서 조회 (메타데이터 포함)
python manage_docs.py list --folder policies/ --metadata

# 문서 검색
python manage_docs.py search "시스템 사양"

# 시스템 통계
python manage_docs.py stats

# 고아 파일 정리
python manage_docs.py cleanup

# 백업 생성
python manage_docs.py backup ./backup/2024-01-01/

# 백업 복원
python manage_docs.py restore ./backup/2024-01-01/
```

#### 시스템 상태 점검
```bash
# 전체 상태 점검
python rag_health_check.py

# JSON 형식 출력
python rag_health_check.py --json

# 결과 파일로 저장
python rag_health_check.py --save health_report.json

# 조용한 모드
python rag_health_check.py --quiet
```

### 3. 쿼리 최적화
- 구체적인 키워드 사용
- 한 번에 한 가지 질문
- 필요시 컨텍스트 포함

### 4. 성능 최적화 팁
- 문서 분할(청크) 크기 조정
- 캐싱, 병렬 검색, 모델 예열
- 실시간 모니터링 활용

## 문제 해결

### 일반적인 문제

1. **결과를 찾을 수 없음**
   - 문서 업로드 확인: `/rag-stats`
   - 검색어가 문서 내용과 일치하는지 확인
   - 설정에서 유사도 임계값 낮추기

2. **느린 응답**
   - 더 빠른 처리를 위해 청크 크기 줄이기
   - 반환되는 최대 결과 수 제한
   - 벡터 스토어 성능 확인

3. **부정확한 결과**
   - 문서 품질 검토
   - 상충되는 정보 확인
   - 임베딩 모델 조정

4. **메모리 문제**
   ```bash
   # 벡터 스토어 초기화
   python clear_and_reload_docs.py
   
   # 또는 명령어 사용
   /rag-clear
   ```

### 디버그 모드

RAG 디버깅 활성화:
```python
# .env에서
RAG_DEBUG=true
LOG_LEVEL=debug
```

로그 확인:
```bash
tail -f logs/rag.log
```

### 시스템 진단

종합적인 RAG 시스템 상태 점검:
```bash
# 전체 건강 상태 점검
python rag_health_check.py

# 문제 해결을 위한 상세 정보
python rag_health_check.py --json | jq '.recommendations'
```

진단 항목:
- ✅ 환경 설정 (API 키, 라이브러리)
- ✅ 디렉토리 구조 (문서/벡터DB 폴더)
- ✅ 벡터 스토어 상태 (인덱싱, 설정)
- ✅ 검색 성능 (응답시간, 품질)
- ✅ RAG 파이프라인 (전체 워크플로우)
- ✅ LLM 통합 (프로바이더 연결)

## 모범 사례

### 문서 가이드라인

1. **구조**
   - 명확한 제목 사용
   - 관련 정보를 함께 유지
   - 일관된 포맷 사용

2. **내용**
   - 구체적이고 상세하게 작성
   - 모호한 언어 피하기
   - 도움이 될 때 예시 포함

3. **업데이트**
   - 문서 버전 관리
   - 주요 변경 후 재인덱싱
   - 오래된 내용 제거

### 쿼리 가이드라인

1. **구체적으로 질문**
   - ❌ "그것에 대해 알려줘"
   - ✅ "MOJI의 주요 기능은 무엇인가요?"

2. **한 번에 한 주제**
   - ❌ "가격이 얼마고 어떻게 설치하고 보증은 뭐야?"
   - ✅ "MOJI의 가격은 얼마인가요?"

3. **컨텍스트 제공**
   - ❌ "얼마나 오래?"
   - ✅ "보증 기간은 얼마나 되나요?"

## API 사용법

### API를 통한 문서 업로드
```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@documents/guide.pdf"
```

### API를 통한 쿼리
```bash
curl -X POST http://localhost:8000/api/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MOJI가 무엇인가요?", "max_results": 3}'
```

### 통계 조회
```bash
curl http://localhost:8000/api/rag/stats
```

## 고급 주제

### 커스텀 임베딩
```python
# OpenAI 임베딩 사용
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_PROVIDER=openai
```

### 다국어 지원
```bash
# 비영어 문서용
EMBEDDING_MODEL=multilingual-e5-large
```

## 관리 도구 요약

### 주요 스크립트
| 스크립트 | 용도 | 주요 기능 |
|---------|------|----------|
| `upload_docs.py` | 문서 업로드 | 증분/배치/폴더별 업로드 |
| `clear_and_reload_docs.py` | 완전 재인덱싱 | 벡터 스토어 초기화 후 전체 재로드 |
| `manage_docs.py` | 문서 관리 | 검색/통계/백업/복원/정리 |
| `rag_health_check.py` | 시스템 진단 | 종합 상태 점검 및 문제 진단 |

### 일반적인 워크플로우
1. **초기 설정**: `python upload_docs.py` (전체 업로드)
2. **일상 업데이트**: `python upload_docs.py --incremental` (변경 파일만)
3. **정기 점검**: `python rag_health_check.py` (시스템 상태)
4. **문제 해결**: `python clear_and_reload_docs.py` (완전 재인덱싱)

### 웹챗에서 RAG 사용
1. 서버 시작: `uvicorn app.main:app --reload`
2. 웹챗 접속: http://localhost:8000/static/webchat-test.html
3. RAG 토글 ON으로 설정
4. 질문 입력 및 출처 확인

## 지원

- RAG 로그: `logs/rag.log`
- 벡터 스토어 데이터: `data/vectordb/`
- 문서 메타데이터: `data/.doc_metadata.json`
- 시스템 진단: `python rag_health_check.py`

---

일반적인 웹챗 사용법은 [WEBCHAT_GUIDE.md](WEBCHAT_GUIDE.md)를 참조하세요.