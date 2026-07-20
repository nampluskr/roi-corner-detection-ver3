# AGENTS.md / README.md / docs/README.md 뼈대 작성

| 항목 | 값 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-20 |
| 적용 범위 | 루트 `README.md`, `CLAUDE.md`, `docs/README.md` 본문 작성 |
| 관련 문서 | [project-start-guide.md](../../project-start-guide.md) |

## 1. 목적과 배경

`260720_roi-corner-detection-ver3`는 `260712_roi-corner-detection-ver2`의 폴더/파일 구성을
재검토하고 재구성하기 위한 작업 공간이다. 현재 이 프로젝트 루트에는 상위 워크스페이스에서 복사한
`project-start-guide.md` 외에 문서가 없고, 문서 우선(plan 기반) 운영 방식이 아직 부트스트랩되지
않았다.

ver2는 `docs/architecture/model-assembly.md`를 canonical SSOT로 하는 이미 성숙한 구조를 갖고
있다. Method(`reg`, `seg`, `det`, `heatmap`, `line`) 및 model/variant 조립 체계, `docs/plans/`에
23건(0001~0023)의 완료·진행 이력, `outputs/<dataset>/<method>/<model>/<exp_name>/` 형식의 실험
산출물 경로 규칙이 SSOT 문서와 `AGENTS.md`/`CLAUDE.md`에 명시되어 있다. 이 때문에 폴더 이름, 경로
계약, 문서 상태 표기(canonical/reference/deprecated)를 바꾸려면 먼저 SSOT 문서와의 정합성을
검토해야 하며, 임의 재구성은 기존 23개 plan의 참조 경로와 실험 산출물 경로를 깨뜨릴 위험이 있다.

이번 plan은 ver3에 문서 워크플로우(README.md/CLAUDE.md/docs/README.md/docs/plans/)를 먼저
갖추어, 이후 ver2 재구성 작업을 개별 plan으로 진행할 수 있는 기반을 마련한다.

## 2. 범위

포함:
- `README.md`(루트) 신규 작성 — ver3가 ver2 재구성 작업 공간임을 명시
- `CLAUDE.md`(루트) 신규 작성 — 공통 규칙은 상위 워크스페이스 `/mnt/d/projects/nampluskr/CLAUDE.md`
  및 이 문서의 운영 규칙을 참조하는 최소 진입점 문서로 작성
- `docs/README.md` 본문 작성 — 문서 색인 및 plan 워크플로우 canonical 정의
- `docs/plans/` 폴더 생성 (이 문서 자체가 첫 산출물)

제외 (후속 plan에서 수행):
- ver2 내부 파일/폴더의 실제 이동, 리네임 또는 삭제
- ver2 SSOT 문서(`model-assembly.md`) 수정
- ver2 재구성 방안의 구체적 설계 (대상 구조, 이동 매핑 등)

이 plan 자체는 ver3 내부 문서만 생성하며, ver2에는 어떤 파일도 수정하지 않는다.

## 3. 문서 간 책임 분리 원칙

| 문서 | 책임 | 대상 독자 |
| --- | --- | --- |
| `README.md` | 제품 소개 — ver3의 목적(ver2 재구성 작업 공간)과 현재 상태 | 프로젝트를 처음 접하는 독자 |
| `CLAUDE.md` | Claude Code 전용 진입점, 운영 규칙은 참조로 위임 | Claude Code |
| `docs/README.md` | 문서 색인 + plan 워크플로우 canonical 정의 | 상세 기준을 찾는 작업 수행자 |
| `docs/plans/` | 계획 문서와 이력 보관 | 작업 수행자 |

동일한 규칙을 여러 문서에 중복 정의하지 않는다. plan 파일 명명 규칙은 `docs/README.md`에서만
canonical하게 정의하고 `CLAUDE.md`는 해당 절을 참조만 한다.

## 4. README.md 목차안

1. 제목 + 1줄 소개 — ver2의 폴더/파일 재구성을 위한 작업 공간
2. 현재 상태 — ver2는 `docs/architecture/model-assembly.md`를 SSOT로 하는 성숙한 구조이며 재구성
   범위와 방안은 아직 미정임을 명시
3. 재구성 대상 — `260712_roi-corner-detection-ver2` 경로 링크와 그 SSOT 문서 링크
4. 제약 사항 — SSOT 문서 구속력, 기존 23개 plan 이력, 실험 산출물 경로 규칙과의 정합성 필요
5. 문서 안내 — `docs/README.md` 링크

## 5. CLAUDE.md 목차안

1. 진입점 안내 — 운영 규칙은 `docs/README.md`를 따름을 명시
2. Claude 전용 규칙 — 하위 프로젝트(ver2) 내부 파일은 요청 없이 직접 수정하지 않고, 재구성 작업은
   반드시 승인된 plan을 통해서만 수행

## 6. docs/README.md 본문 채우기 방향

- **문서 요약**: ver3의 목적(ver2 재구성)과 이 문서가 plan 워크플로우의 canonical 정의임을 서술
- **문서 색인 표**: `README.md`, `CLAUDE.md`, `docs/plans/` 각 항목의 상태 표기
- **plan 문서 규칙**: 경로 `docs/plans/NNNN-topic-plan.md`, 4자리 0-padding, 순증가, 상태
  `Draft`/`Approved`/`Done`, 완료 후에도 파일 보존. `project-start-guide.md` 6장 규칙을 그대로 계승
- **첫 세션 진행 순서**: 이 plan(0001) 승인 후 루트 문서 작성, 이후 ver2 현황 조사 plan(0002)과
  재구성 설계 plan으로 이어짐을 명시

## 7. 완료 기준

- **README.md**: ver3의 목적과 현재 상태(재구성 방안 미정)가 명시되고 ver2·SSOT 문서 링크가 유효할 것
- **CLAUDE.md**: 하위 프로젝트(ver2) 파일을 직접 수정하지 않는다는 규칙이 명시될 것
- **docs/README.md**: plan 규칙이 이 plan(0001) 자신을 예시로 설명하고, 문서 색인 표의 파일명이
  실제 파일명과 일치할 것
- 공통: 이모지 없음, UTF-8, 한글 깨짐(U+FFFD) 없음, 문서 간 동일 규칙 중복 서술 없음

## 참고 파일

- [project-start-guide.md](../../project-start-guide.md) — 이 부트스트랩 절차의 근거 가이드
- [260712_roi-corner-detection-ver2/README.md](../../../260712_roi-corner-detection-ver2/README.md) — 재구성 대상 개요
- [260712_roi-corner-detection-ver2/docs/architecture/model-assembly.md](../../../260712_roi-corner-detection-ver2/docs/architecture/model-assembly.md) — ver2 SSOT 문서
- [260712_roi-corner-detection-ver2/AGENTS.md](../../../260712_roi-corner-detection-ver2/AGENTS.md) — ver2 운영 규칙 (plan 번호 체계, 경로 규칙 참고)

## 검증

이 plan 문서 자체는 실행(코드/타 문서 수정) 없이 순수 문서 생성 1건이므로 별도 빌드/테스트는
불필요하다. 생성 후 사용자에게 내용 검토를 요청한다.
