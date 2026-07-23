# Independent Project Cleanup Plan

| 항목 | 내용 |
| --- | --- |
| 상태 | Done |
| 작성일 | 2026-07-23 |
| 적용 범위 | `AGENTS.md`, `CLAUDE.md`, root and docs README files, `docs/plans/`, `slides/`, public dataset slide asset script |
| 관련 문서 | [../README.md](../README.md), [../../README.md](../../README.md) |

## 목적과 배경

현재 workspace는 필요한 구현과 문서를 내부에 갖춘 독립 프로젝트로 관리한다. 과거 sibling project를
전제로 한 경로, 출처 설명, 읽기 전용 참조 규칙이 남아 있으면 프로젝트의 현재 기준이 외부에 있는 것처럼
보일 수 있다. 이 작업은 문서와 생성 스크립트에서 그런 연결을 제거하고 현재 repository 내부 기준만
남긴다.

## 범위

포함 항목은 다음과 같다.

| 항목 | 내용 |
| --- | --- |
| 루트 지침 | `AGENTS.md`와 `CLAUDE.md`를 현재 workspace 기준으로 갱신하고 byte-level 일치를 검증 |
| 문서 정리 | root README, docs README, model README, slide README, architecture와 model 문서에서 외부 출처 표현 제거 |
| history plan 정리 | 완료 plan 안의 legacy sibling project 경로와 직접 참조 표현 제거 |
| asset script | public dataset 예시 생성 script가 현재 repo의 `data/public` CSV를 읽도록 수정 |

제외 항목은 다음과 같다.

| 항목 | 제외 이유 |
| --- | --- |
| 완료 plan 삭제 | plan 파일은 history로 보존한다 |
| public dataset CSV 이동 | 현재 repo 내부 경로가 이미 존재하므로 이동하지 않는다 |

## 완료 기준

다음 조건을 만족하면 이 plan은 완료된 것으로 본다.

| 기준 | 확인 방법 |
| --- | --- |
| 외부 sibling project 직접 참조 제거 | 지정 검색어에 대한 `rg` 결과가 없음 |
| 지침 파일 동기화 | `AGENTS.md`와 `CLAUDE.md`의 SHA-256 값이 같음 |
| public asset script 독립화 | script가 `data/public` CSV로 PNG를 재생성함 |
| 삭제 요청 파일 유지 | 오래된 root guide 두 파일이 삭제 상태로 남음 |

## 검증

검증은 conda 환경 `pytorch_env`를 활성화한 뒤 project root에서 수행한다.

```bash
rtk rg -n "legacy sibling project markers" AGENTS.md CLAUDE.md README.md docs slides
sha256sum AGENTS.md CLAUDE.md
python slides/assets/make_public_dataset_figs.py
python -m py_compile slides/assets/make_public_dataset_figs.py
rtk git status --short
```
