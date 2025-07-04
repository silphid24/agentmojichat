# CLAUDE.md (2024 최신)

이 문서는 MOJI 프로젝트의 개발 가이드, 코드 품질, 테스트, 성능, 문서화, 최신 기능(WebChat v2, 하이브리드 검색 등) 기준을 안내합니다.

## 개발 지침
- `/cursor/rules/global.mdc`의 원칙을 반드시 준수
- 테스트 우선(TDD), SOLID, 클린 아키텍처, 코드 리뷰, 성능 기준, 보안, 문서화
- PR 전 셀프 리뷰, 커밋 메시지 명확화, 브랜치 전략 준수

## 기술 스택
- Python 3.11+, FastAPI 0.111, LangChain 0.2.x, PostgreSQL 15, Redis 7, Chroma DB
- DeepSeek, OpenAI, Anthropic, 커스텀 LLM, WebChat v2, 하이브리드 검색, 적응형 기능

## 구현/테스트 원칙
- 테스트 커버리지 80% 이상, 단위/통합/성능 테스트
- WebChat v2, 하이브리드 검색, 실시간 모니터링 등 실제 시나리오 기반 테스트
- 성능 기준: API 응답 200ms 이내, 페이지 로드 3초 이내

## 코드 품질/리팩토링/디버깅/리뷰/에러 처리/성능/문서화 등은 기존 원칙을 유지하되, 최신 기능과 실제 구현 상태를 반영해 보강

## 프로젝트 구조/공통 작업/중요 리마인더 등은 기존 예시를 최신화하여 유지

## Important: Development Guidelines
이 프로젝트는 `/cursor/rules/global.mdc`에 정의된 개발 지침을 따릅니다. 코드 작성 전 반드시 해당 파일을 참고하세요.

## Project Overview

This is the MOJI AI Agent - a multi-platform AI assistant supporting Slack, Teams, KakaoTalk, Discord, and Web Chat.

Note: The SMHACCP Project Management Platform (previously Phase 2) has been moved to a separate project. 
See SMHACCP_PROJECT_GUIDE.md for migration guidance.

## Development Approach

### MVP-First Strategy
The project follows an MVP-first approach. When implementing features:
- Start with `# MVP Details` sections in task files
- Defer `# Future Development` items for later phases
- Begin with monolithic architecture (single FastAPI server), plan for microservices later

### Task Execution Order
Follow the numbered tasks in `/script/task_*.txt`:
1. Architecture Design → 2. Docker Setup → 3. FastAPI Server → 4. Multi-Agent System → 5. LLM Router → 6. RAG Pipeline → 7. Vector Store → 8. Platform Adapters → 9. Plugin System → 10. Security/Auth → 11. Testing/Deployment

## Technology Stack

### Backend (Python 3.11+)
- **Framework**: FastAPI 0.111
- **AI/ML**: LangChain 0.2.x, LangGraph 0.1.x
- **Databases**: PostgreSQL 15, Redis 7, Chroma DB
- **MVP LLM**: DeepSeek API (deepseek-r1)
- **LLM Configuration**: 환경 변수를 통한 동적 모델 변경 지원
  - `LLM_PROVIDER`: deepseek, openai, anthropic, custom 등
  - `LLM_MODEL`: deepseek-r1, gpt-3.5-turbo, claude-3, llama-3 등
  - `LLM_API_BASE`: 커스텀 API 엔드포인트 URL
  - `LLM_API_KEY`: API 인증 키

### Frontend Support
- Web Chat widget (static HTML/JS)
- Platform-specific integrations
- API client examples

## Development Principles (from `/cursor/rules/global.mdc`)

### 1. 구현 작업 원칙
- **테스트 우선**: 비즈니스 로직 구현 전, 반드시 테스트를 먼저 작성 - Test코드는 Tests폴더에 생성해줘. 
- **SOLID 준수**: 5대 원칙에 따라 설계·구현
- **클린 아키텍처**: 계층·의존성 규칙을 지켜 구조를 설계
- **UI 작업**: 구현을 다 끝낸 다음에 테스트 코드 진행
- **코어 로직**: TDD로 구현

### 2. 코드 품질 원칙
- **단순성**: 복잡한 해결책보다 *가장 단순한 솔루션* 우선
- **DRY**: 중복을 피하고, 기존 기능을 재사용
- **가드레일**: 테스트 외 환경에서는 *모의 데이터* 사용 금지
- **효율성**: 명확성을 해치지 않는 범위에서 토큰 사용 최소화

### 3. 리팩터링
- **승인 프로세스**: 리팩터링 계획 설명 후 허가를 받아 진행
- **목표 명확화**: *기능 변경 없이* 코드 구조 개선에 집중
- **테스트 통과**: 리팩터링 후 모든 테스트가 통과하는지 확인


### 4. 디버깅
- **원인·해결책 설명**: 디버깅 시 원인과 해결 방안을 문서화하고 허락을 받은 뒤 진행
- **정상 동작이 목표**: 단순 에러 제거가 아닌 *올바른 기능 수행* 최우선
- **상세 로그**: 원인이 모호할 경우 추가 로그로 분석 가능성 향상

### 5. 코드 리뷰 원칙
- **PR 크기**: 한 번에 검토 가능한 200-400줄 이내로 제한
- **셀프 리뷰**: PR 생성 전 반드시 본인이 먼저 검토
- **체크리스트**:
  - [ ] 테스트 커버리지 80% 이상
  - [ ] 린트 에러 없음
  - [ ] 성능 영향도 검토
  - [ ] 보안 취약점 스캔

### 6. 에러 처리 전략
- **예외 계층화**: 도메인별 커스텀 예외 클래스 정의
- **에러 경계**: UI 레벨에서 Error Boundary 구현
- **복구 전략**: 재시도 로직과 폴백 메커니즘 구현
- **모니터링**: 에러 발생 시 자동 알림 설정

### 7. 성능 최적화
- **측정 우선**: 최적화 전 반드시 성능 측정
- **임계값 설정**:
  - API 응답: 200ms 이내
  - 페이지 로드: 3초 이내
  - 메모리 사용: 증가율 모니터링
- **캐싱 전략**: 적절한 캐싱 레이어 구현

### 8. 언어 및 문서화
- **리소스 설명**: 관련 설명은 한글로 작성
- **용어 보존**: 기술 용어·라이브러리 이름은 원문 표기 유지
- **다이어그램**:
  - 간단한 흐름도: **mermaid** 사용
  - 복잡한 아키텍처: 별도 **SVG** 생성 후 문서 삽입

## Project Structure

```
agentmoji/
├── script/          # Task descriptions (task_01-15.txt)
├── cursor/          # Development rules
│   └── rules/
│       └── global.mdc
├── moji-prd-v3.md  # Product Requirements Document
└── CLAUDE.md       # This file
```

## Key Implementation Notes

### MOJI Agent Development
- Start with single LangChain agent (not LangGraph multi-agent initially)
- Use DeepSeek API as default, but support model switching via environment variables
- LLM provider abstraction layer for easy API switching (DeepSeek, OpenAI, Anthropic, custom endpoints)
- Basic RAG with text files only (TXT, Markdown)
- Simple JWT authentication (no OAuth flow initially)
- Docker deployment (defer Kubernetes)
- Platform adapters for Slack, Teams, KakaoTalk, Discord, WebChat

## Common Development Tasks

Since this is a new project without implemented code yet, the first tasks will be:

1. **Setup Development Environment**:
   ```bash
   # Create project structure based on task_01
   # Initialize Docker environment per task_02
   # Create basic FastAPI server per task_03
   ```

2. **When Starting Implementation**:
   - Read the corresponding task file
   - Create directory structure as needed
   - Write tests first for core logic
   - Implement MVP features only

3. **Testing Strategy**:
   - Unit tests with pytest
   - Integration tests for API endpoints
   - Use GitHub Actions for CI (when implemented)

## Important Reminders

- MOJI is now the sole focus of this repository
- SMHACCP platform has been moved to a separate project
- Follow test-first approach for core logic
- Maintain clean architecture with proper layer separation
- Target 80% test coverage minimum
- Keep API responses under 200ms