# 문서 색인

이 문서는 `260712_roi-corner-detection-ver2`의 폴더/파일을 재구성한 결과를 확립하는 문서
워크플로우의 canonical 정의다.

ver2의 SSOT/canonical 문서(`docs/architecture/model-assembly.md` 등)는 참고 자료로만 사용한다.
사용자의 재구성 요청은 ver3에서 실제로 실행하고, 확정된 결과를 이 프로젝트의 canonical 문서로
새로 기록한다.

## 문서 색인 표

| 문서 | 상태 | 책임 |
| --- | --- | --- |
| [AGENTS.md](../AGENTS.md) | 현재 기준 | Codex 전용 진입점 |
| [README.md](../README.md) | 현재 기준 | 프로젝트 목적, 재구성 대상 |
| [CLAUDE.md](../CLAUDE.md) | 현재 기준 | Claude Code 전용 진입점 |
| [plans/](plans/) | 이력 | 작업 계획과 완료 이력 |

## 첫 세션 진행 순서

1. `docs/plans/0001-root-and-docs-index-plan.md` 승인 후 루트 `README.md`, `CLAUDE.md`, `AGENTS.md`,
   `docs/README.md`를 작성한다.
2. 이후 사용자의 재구성 요청을 반영하는 실행과 canonical 문서 확립은 `0002`부터 순증가하는 plan으로
   이어간다.

## 플랜 규칙

- 경로는 `docs/plans/NNNN-topic-plan.md`이다.
- 번호는 4자리 0-padding의 순증가 번호이며 재사용하거나 삭제하지 않는다.
- 상태는 `Draft`, `Approved`, `Done` 중 하나를 사용한다.
- 완료된 plan은 이력으로 보존한다.

## 폴더 구조

```text
docs/
├── README.md
└── plans/       # 계획과 이력
```

초기 범위에서는 `architecture/`, `design/`, `specs/` 등의 폴더를 만들지 않는다. 재구성 결과를 담을
canonical 문서 폴더는 실제로 필요해질 때 승인된 plan으로 추가한다.
