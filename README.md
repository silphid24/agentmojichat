# MOJI AI Agent & SMHACCP 프로젝트 관리 플랫폼

## 🚀 프로젝트 개요

MOJI는 다양한 플랫폼에서 활용 가능한 지능형 AI 어시스턴트이며, SMHACCP는 MOJI를 통합한 프로젝트 관리 시스템입니다.

### 주요 특징

- **다중 플랫폼 지원**: Slack, Microsoft Teams, KakaoTalk, Discord, Web Chat
- **지능형 대화 엔진**: LangChain 기반 대화 처리
- **RAG 시스템**: 문서 기반 지식 검색 및 활용
- **확장 가능한 LLM**: DeepSeek, OpenAI, Anthropic 등 다양한 모델 지원
- **플러그인 시스템**: 동적 기능 확장

## 📋 프로젝트 구조

```
agentmoji/
├── script/          # 개발 작업 설명서 (task_01-15.txt)
├── cursor/          # 개발 가이드라인
│   └── rules/
│       └── global.mdc
├── moji-prd-v3.md  # 제품 요구사항 문서
├── CLAUDE.md       # Claude AI 개발 가이드
└── README.md       # 프로젝트 설명서
```

## 🛠 기술 스택

### Backend (Phase 1)
- **Language**: Python 3.11+
- **Framework**: FastAPI 0.111
- **AI/ML**: LangChain 0.2.x, LangGraph 0.1.x
- **Database**: PostgreSQL 15, Redis 7, Chroma DB
- **LLM**: DeepSeek R1 (기본), 환경 변수로 변경 가능

### Frontend (Phase 2)
- **Framework**: Next.js 14
- **Language**: TypeScript 5
- **UI**: Tailwind CSS 3, Shadcn/ui
- **State**: Zustand, React Query v5

## 🚦 개발 로드맵

### Phase 1: MOJI AI Agent (8주)
1. 아키텍처 설계 및 API 스펙 정의
2. Docker 환경 구성
3. FastAPI 서버 구축
4. Multi-Agent 시스템 구현
5. LLM Router 개발
6. RAG 파이프라인 구축
7. Vector Store 설정
8. 플랫폼 어댑터 개발

### Phase 2: SMHACCP Platform (6주)
1. Frontend 개발
2. MOJI 통합
3. 외부 서비스 연동
4. 모니터링 시스템 구축

## 🏃‍♂️ 시작하기

현재 프로젝트는 계획 단계입니다. 개발을 시작하려면:

1. `/script/task_01_architecture_design.txt`부터 순차적으로 진행
2. 각 task 파일의 MVP Details 섹션을 우선 구현
3. `/cursor/rules/global.mdc`의 개발 지침 준수

## 🔧 환경 설정

### LLM 설정 (환경 변수)
```bash
export LLM_PROVIDER=deepseek      # deepseek, openai, anthropic, custom
export LLM_MODEL=deepseek-r1      # 모델명
export LLM_API_BASE=https://...   # API 엔드포인트
export LLM_API_KEY=your-api-key   # API 키
```

## 📝 개발 원칙

- **테스트 우선**: 코어 로직은 TDD로 구현
- **SOLID 원칙**: 5대 원칙 준수
- **클린 아키텍처**: 계층 분리 유지
- **MVP 우선**: 최소 기능부터 구현

자세한 개발 가이드라인은 [`CLAUDE.md`](./CLAUDE.md)와 [`/cursor/rules/global.mdc`](./cursor/rules/global.mdc)를 참조하세요.

## 📊 성공 지표

### MOJI AI Agent
- 응답 정확도: 90% 이상
- 응답 시간: 2초 이내
- 동시 사용자: 1,000명 이상
- 가용성: 99.9% SLA

### SMHACCP Platform
- 사용자 채택률: 80% 이상
- 업무 효율성: 50% 향상
- 일일 MOJI 활용: 200건 이상

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

프로젝트 라이선스는 추후 결정 예정입니다.

## 📞 문의

프로젝트 관련 문의사항은 이슈를 통해 남겨주세요.