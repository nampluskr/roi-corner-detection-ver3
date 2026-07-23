---
상태: Done
작성일: 2026-07-22
적용 범위: `slides/assets/` transform 예시 이미지와 synthetic 합성 변형 변수 예시 이미지 신설, `docs/architecture/05-data-strategy.md`에 synthetic 합성 변형 변수 절 추가, `slides/outline.md`에 변형 변수 페이지 추가, `slides/README.md` asset 목록 갱신
관련 문서: [../architecture/05-data-strategy.md](../architecture/05-data-strategy.md), [../../slides/outline.md](../../slides/outline.md), [../../slides/README.md](../../slides/README.md), [0017-data-strategy-doc-plan.md](0017-data-strategy-doc-plan.md)
---

## 목적과 배경

plan 0017은 data 3단계 전략과 offline pre-augmentation, online transform 구분을 문서와 slide에
서술했으나 이미지 없이 mermaid 흐름도와 표로만 구성했다. 이후 사용자는 두 가지를 추가로 요청했다.

첫째, 데이터 변형과 transform을 실제 이미지 사례로 보이기를 원했다. 개념 도식이 아니라 project의 실제
transform class를 합성 panel에 적용한 before, after 예시가 필요하다.

둘째, synthetic 원본을 합성할 때 포함되는 여러 변형 변수, 즉 위치, corner 라운딩, 외부 지그의 위치와
크기와 개수, 카메라 위치, 배경 밝기, fringe 왜곡 변수를 변수 리스트 표와 예시 이미지로 보이기를
원했다. 이 내용은 사용자가 필요한 변형의 개수를 변수별로 차별 적용하기 위한 근거가 된다.

이 두 요청은 plan 0017 범위의 "새 slide 이미지나 asset은 만들지 않는다" 항목을 넘어서므로 별도 plan으로
기록한다. 변형 변수 범위는 현재 synthetic data 전략에 맞춰 정리한다.

## 범위

포함 항목은 다음과 같다.

- `slides/assets/make_transform_figs.py` 신설. `src/data/transforms.py`의 실제 transform class를 합성
  fringe panel에 적용해 offline distortion 계열과 online 단순 계열의 before, after 예시 두 장을
  만든다. corner가 image와 함께 변환됨을 보인다.
- `slides/assets/make_synthetic_variation_figs.py` 신설. 위치와 자세, corner 라운딩, 외부 지그, 카메라
  hole, 배경 밝기, fringe 왜곡 여섯 변수 계열별 예시 이미지 여섯 장을 만든다. 각 예시에 정답 corner를
  표시한다.
- `docs/architecture/05-data-strategy.md`에 synthetic 합성 변형 변수 절을 별도 절로 추가한다. 변수
  계열별 옵션과 범위 표, 예시 이미지 참조, 변수별 변형 개수 차별 적용 목적을 서술한다.
- `slides/outline.md`에 변형 변수 페이지를 추가하고 이후 페이지 번호를 순증가로 재조정한다. offline과
  online transform 페이지에 transform 예시 이미지를 삽입한다.
- `slides/README.md`의 asset tree, 파일 역할 표, 재생성 명령에 새 script와 이미지를 반영한다.

제외 항목은 다음과 같다.

- 새 dataset loader나 합성 data 생성 pipeline은 만들지 않는다. 예시 이미지는 slide 자산 렌더링 script
  이며 학습 data 생성기가 아니다.
- 변수 범위 값은 개념 baseline으로만 기록하고 실제 생성 script의 최종 parameter를 확정하지 않는다.

## 완료 기준

다음을 모두 충족하면 이 plan을 `Done`으로 본다.

- transform 예시 이미지 두 장과 synthetic 변형 변수 예시 이미지 여섯 장이 `slides/assets/`에 생성되고
  각 예시에 정답 corner가 함께 표시된다.
- `docs/architecture/05-data-strategy.md`에 synthetic 합성 변형 변수 절이 변수 표와 이미지 참조와 함께
  추가되고 절 번호가 연속한다.
- `slides/outline.md`에 변형 변수 페이지가 추가되고 페이지 번호가 1부터 연속하며 내부 페이지 참조가
  갱신된다.
- `slides/README.md`가 새 script와 이미지, 재생성 명령을 반영한다.

## 검증

이번 작업은 이미지 생성 script 신설과 문서 갱신이다. 다음 항목을 확인한다.

- conda 환경 `pytorch_env`에서 두 script를 실행해 여덟 장의 PNG가 생성되고 각 이미지의 corner 표시가
  올바른지 육안으로 확인한다.
- 갱신 문서가 em dash, 유니코드 화살표, 이모지, 수평 구분선을 사용하지 않고 본문 화살표는 `$\to$`,
  code 안 화살표는 ASCII `->`를 사용한다.
- `slides/outline.md`의 페이지 번호와 `docs/architecture/05-data-strategy.md`의 절 번호가 연속하고
  내부 상호 참조가 실제 번호와 일치한다.
- 문서와 slide의 이미지 참조 경로가 `slides/assets/`의 실제 파일과 일치한다.
