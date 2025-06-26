# Docker Development Environment

MOJI AI Agent의 Docker 개발 환경 설정 가이드입니다.

## 빠른 시작

```bash
# 환경 시작
make start

# 헬스 체크
make health

# 로그 확인
make logs

# 환경 중지
make stop
```

## 서비스 구성

- **FastAPI App**: http://localhost:8000
- **PostgreSQL**: localhost:5432 (user: moji, password: moji123, db: moji_db)
- **Redis**: localhost:6379

## 주요 명령어

### 개발 환경 관리
```bash
make build    # Docker 이미지 빌드
make start    # 개발 환경 시작
make stop     # 개발 환경 중지
make clean    # 전체 정리 (볼륨 포함)
make health   # 서비스 상태 확인
```

### 개발 작업
```bash
make logs     # 애플리케이션 로그 확인
make shell    # 앱 컨테이너 쉘 접속
make db-shell # PostgreSQL 쉘 접속
make test     # 테스트 실행
make lint     # 코드 검사
```

## 파일 구조

```
docker/
├── scripts/
│   ├── start.sh      # 시작 스크립트
│   ├── stop.sh       # 중지 스크립트
│   ├── clean.sh      # 정리 스크립트
│   └── health-check.sh # 헬스 체크
└── README.md

Dockerfile            # 프로덕션 이미지
Dockerfile.dev        # 개발 이미지
docker-compose.yml    # 프로덕션 설정
docker-compose.dev.yml # 개발 설정
```

## 환경 변수

`.env` 파일에서 설정 가능:
- `DATABASE_URL`: PostgreSQL 연결 문자열
- `REDIS_URL`: Redis 연결 문자열
- `LLM_PROVIDER`: LLM 제공자 (기본: deepseek)
- `LLM_MODEL`: LLM 모델 (기본: deepseek-r1)
- `DEBUG`: 디버그 모드 (개발: true)

## 볼륨 마운트

개발 환경에서는 다음 디렉토리가 자동 마운트됩니다:
- `./app` → `/app/app` (핫 리로드 지원)
- `./tests` → `/app/tests`
- `./.env` → `/app/.env`

## 문제 해결

### 포트 충돌
```bash
# 사용 중인 포트 확인
lsof -i :8000
lsof -i :5432
lsof -i :6379
```

### 컨테이너 로그 확인
```bash
docker-compose -f docker-compose.dev.yml logs [service_name]
```

### 데이터베이스 초기화
```bash
make clean
make start
```