# AGENTS.md / README.md / docs/README.md 뼈대 작성

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-20 |
| 적용 범위 | 루트 `README.md`, `CLAUDE.md`, `docs/README.md` 본문 작성 |
| 관련 문서 | 없음 |

## 1. 목적과 배경

이 프로젝트는 초기 단계에서 문서 우선 운영 방식이 아직 부트스트랩되지 않았다. 이 plan은
`README.md`, `CLAUDE.md`, `docs/README.md`, `docs/plans/`를 먼저 갖추어 이후 작업을 개별 plan으로
진행할 수 있는 기반을 마련한다.

## 2. 범위

포함:
- `README.md`(루트) 신규 작성
- `CLAUDE.md`(루트) 신규 작성 — 공통 규칙은 상위 워크스페이스 `/mnt/d/projects/nampluskr/CLAUDE.md`
  및 이 문서의 운영 규칙을 참조하는 최소 진입점 문서로 작성
- `docs/README.md` 본문 작성 — 문서 색인 및 plan 워크플로우 canonical 정의
- `docs/plans/` 폴더 생성 (이 문서 자체가 첫 산출물)

제외 (후속 plan에서 수행):
- 실제 구현 파일의 이동, 리네임 또는 삭제
- model assembly 세부 설계 수정
- 후속 구조 개편 방안의 구체적 설계

이 plan 자체는 내부 문서만 생성한다.

## 3. 문서 간 책임 분리 원칙

| 문서 | 책임 | 대상 독자 |
| --- | --- | --- |
| `README.md` | 제품 소개와 현재 상태 | 프로젝트를 처음 접하는 독자 |
| `CLAUDE.md` | Claude Code 전용 진입점, 운영 규칙은 참조로 위임 | Claude Code |
| `docs/README.md` | 문서 색인 + plan 워크플로우 canonical 정의 | 상세 기준을 찾는 작업 수행자 |
| `docs/plans/` | 계획 문서와 이력 보관 | 작업 수행자 |

동일한 규칙을 여러 문서에 중복 정의하지 않는다. plan 파일 명명 규칙은 `docs/README.md`에서만
canonical하게 정의하고 `CLAUDE.md`는 해당 절을 참조만 한다.

## 4. README.md 목차안

1. 제목과 1줄 소개
2. 현재 상태
3. 주요 구성
4. 제약 사항과 문서 기준
5. 문서 안내

## 5. CLAUDE.md 목차안

1. 진입점 안내 — 운영 규칙은 `docs/README.md`를 따름을 명시
2. Claude 전용 규칙 — 구조 변경은 반드시 승인된 plan을 통해서만 수행

## 6. docs/README.md 본문 채우기 방향

- **문서 요약**: 이 문서가 plan 워크플로우의 canonical 정의임을 서술
- **문서 색인 표**: `README.md`, `CLAUDE.md`, `docs/plans/` 각 항목의 상태 표기
- **plan 문서 규칙**: 경로 `docs/plans/NNNN-topic-plan.md`, 4자리 0-padding, 순증가, 상태
  `Draft`/`Approved`/`Done`, 완료 후에도 파일 보존.
- **첫 세션 진행 순서**: 이 plan(0001) 승인 후 루트 문서 작성, 이후 구조 확립 plan으로 이어짐을 명시

## 7. 완료 기준

- **README.md**: 프로젝트 목적과 현재 상태가 명시될 것
- **CLAUDE.md**: 구조 변경은 plan을 통해 수행한다는 규칙이 명시될 것
- **docs/README.md**: plan 규칙이 이 plan(0001) 자신을 예시로 설명하고, 문서 색인 표의 파일명이
  실제 파일명과 일치할 것
- 공통: 이모지 없음, UTF-8, 한글 깨짐(U+FFFD) 없음, 문서 간 동일 규칙 중복 서술 없음

## 참고 파일

- 현재 repository의 root README와 docs README를 기준으로 삼는다.

## 검증

이 plan 문서 자체는 실행(코드/타 문서 수정) 없이 순수 문서 생성 1건이므로 별도 빌드/테스트는
불필요하다. 생성 후 사용자에게 내용 검토를 요청한다.
