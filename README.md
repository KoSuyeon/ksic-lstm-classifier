# 자연어 기반 산업분류 자동화 모델

> 통계청 「2022 통계데이터 인공지능 활용대회」 참가 프로젝트  
> 사업체 텍스트 설명을 입력받아 한국표준산업분류(KSIC) 대·중·소분류를 자동 예측하는 딥러닝 모델

---
## 📌 프로젝트 진행기간: '21.04~'21.07
## 📌 프로젝트 개요

전국사업체조사 데이터에서 각 사업체의 업종 설명 텍스트(무엇을 가지고, 어떤 방법으로, 무엇을 생산·제공하는지)를 바탕으로 산업 대분류(19개)·중분류(74개)·소분류(225개)를 순차적으로 예측하는 자연어 분류 모델입니다.

- **학습 데이터**: 2019년 기준 전국사업체조사 샘플 데이터 약 100만 건
- **예측 데이터**: 모델개발용 자료 100,000건
- **팀원**: 고수연, 김혜인, 이은지 (성신여자대학교 통계학과 19학번)

---

## 🗂 데이터 구조

| 컬럼 | 설명 | 예시 |
|------|------|------|
| `AI_id` | 일련번호 | id_0000001 |
| `digit_1` | 산업 대분류 (알파벳) | G |
| `digit_2` | 산업 중분류 (2자리 숫자) | 46 |
| `digit_3` | 산업 소분류 (3자리 숫자) | 464 |
| `text_obj` | 무엇을 가지고 (원재료, 영업장소 등) | 상점내에서 |
| `text_mthd` | 어떤 방법으로 (주요 영업·생산 활동) | 일반인을 대상으로 |
| `text_deal` | 무엇을 생산·제공하였는가 (최종 재화·용역) | 채소·과일판매 |

---

## 모델 아키텍처

상위 분류 결과를 하위 분류 예측에 활용하는 **계층적 순차 예측** 방식을 사용합니다.

```
[학습]
text → Model 1 → pred_digit_1 (대분류, 19개 클래스)
text + digit_1_name → Model 2 → pred_digit_2 (중분류, 74개 클래스)
text + digit_1_name + digit_2_name → Model 3 → pred_digit_3 (소분류, 225개 클래스)

[예측]
text → Model 1 → pred_digit_1
text + pred_digit_1 → Model 2 → pred_digit_2
text + pred_digit_1 + pred_digit_2 → Model 3 → pred_digit_3
```

각 모델은 동일한 구조의 CNN + LSTM 네트워크를 사용합니다.

```
Embedding(vocab=25000, dim=128)
→ Conv1D(filters=32, kernel_size=3, activation='tanh')
→ MaxPooling1D(pool_size=2)
→ LSTM(128)
→ Dropout(0.2)
→ Dense(N, activation='softmax')   # N: 각 분류 단계의 클래스 수
```

---

## 전처리 파이프라인

1. **텍스트 결합**: `text_obj`, `text_mthd`, `text_deal` 세 컬럼을 하나의 텍스트로 합침
2. **정규표현식 정제**: 비문자 데이터 공백 처리
3. **형태소 분석 (Mecab)**: 2글자 이상의 명사만 추출하여 불필요한 조사·어미 제거
4. **토큰화 및 시퀀싱**: Keras `Tokenizer`로 정수 인코딩 후 `pad_sequences`로 길이 통일
5. **레이블 인코딩**: `pd.factorize`로 카테고리 정수 인코딩 후 원핫벡터 변환

---

## 📊 성능

EarlyStopping(patience=0)을 적용하여 최적 epoch에서 조기 종료한 결과입니다.

| 분류 단계 | Validation Accuracy |
|-----------|-------------------|
| 대분류 | **97.2%** |
| 중분류 | **95.9%** |
| 소분류 | **91.1%** |

---

## 파일 구조

```
├── code.ipynb                  # 전체 학습 및 예측 코드
├── 산업분류_예측값.csv          # 최종 제출 결과 (100,000건)
└── README.md
```

> 학습에 사용된 원본 데이터(`실습용자료.txt`, `모델개발용자료.txt`, `한국표준산업분류.xlsx`)는 통계청 저작권 관계로 미포함

---

## 실행 방법

### 환경 요구사항

```bash
pip install tensorflow keras konlpy pandas numpy scikit-learn matplotlib seaborn
```

Mecab 한국어 형태소 분석기 설치 필요 ([설치 가이드](https://konlpy.org/en/latest/install/))

### 순서

1. 통계청에서 학습 데이터(`실습용자료.txt`)와 분류 코드표(`한국표준산업분류.xlsx`) 다운로드
2. `code.ipynb` 순차 실행
   - 셀 1: 라이브러리 및 함수 정의
   - 셀 2: 데이터 불러오기
   - 셀 3: 전처리 (카테고리 코드 결합)
   - 셀 4: 대·중·소분류 순서로 모델 학습
   - 셀 5: 예측 및 결과 CSV 출력

---

## 주요 의사결정 및 고민 과정

- **형태소 분석기 선택**: Komoran, 한나눔, 꼬꼬마, Okt 등을 비교 검토 후 Mecab 채택 (오타 처리 및 분석 품질 우수)
- **계층적 예측 방식 도입**: 처음에는 대·중·소 각각 독립 학습 → 상위 분류 결과를 하위 모델 입력에 포함하는 방식으로 개선하여 소분류 정확도 향상
- **활성화함수 비교**: Conv1D에서 `relu`, `tanh`, `sigmoid` 비교 실험 후 `tanh` 채택
- **과적합 방지**: Dropout(0.2) 및 EarlyStopping 적용

---

## 참고

- [한국표준산업분류 10차]([https://kssc.kostat.go.kr](https://kssc.mods.go.kr:8443/ksscNew_web/kssc/common/ClassificationContent.do?gubun=1&strCategoryNameCode=001))
- [통계데이터 인공지능 활용대회](https://data.kostat.go.kr/sbchome/contents/cntPage.do?cntntsId=CNTS_000000000000575&curMenuNo=OPT_09_03_00_0)
