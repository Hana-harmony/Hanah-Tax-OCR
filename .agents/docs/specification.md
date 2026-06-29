
## OCR 적용 서류
1. 거주자증명서 (미국 현지에서 발급)
- 추출 정보: 명칭(성명), 납세자 번호, 거주지국, 거주지국 코드(US)
- 샘플 경로: sample_data/거주자증명서/*

### 미국 거주증명서 영문
어느정도 지역별로 양식이 상이하나, 아래 내용은 공통 표준을 따르고있음
```text
DEPARTMENT OF THE TREASURY
INTERNAL REVENUE SERVICE
PHILADELPHIA, PA 19255

Date: [발급일자, 예: Month DD, YYYY]
Taxpayer: [신청자 영문 이름 또는 회사명]
TIN: [미국 납세자 번호, 예: SSN 또는 EIN]
Tax Year: [증명하려는 과세 연도]

CERTIFICATION

I certify that, to the best of our knowledge, the above-named taxpayer is a resident of the United States of America for purposes of U.S. taxation.

[서명]
[담당자 이름(Printed Name)]
Director, Accounts Management
Internal Revenue Service
```

2. 아포스티유
- 샘플 경로: sample_data/아포스티유/
### 미국 거주증명서 영문
어느정도 지역별로 양식이 상이하나, 아래 내용은 공통 표준을 따르고있음
```text
발행 국가: United States of America (미국)
서명자 정보: 해당 문서를 공증한 공증인(Notary Public) 또는 서명한 주정부 관계자의 성함
서명자의 자격: 서명자가 어떤 자격으로 서명했는지 명시 (예: Notary Public)
기관 인장: 해당 주 국무장관실(Secretary of State)의 관인 또는 스탬프
발행 장소 및 날짜: 아포스티유가 발급된 도시 및 날짜
발행 번호 및 인증 기관: 증명서 고유 번호 및 발행 부서명
```

3. 국내원천소득 제한세율 적용신청서(비거주자용)
- 추출 정보: 주소
- 샘플경로: sample_data/국내원천소득 제한세율/*

## 검증
1. 거주자 증명서 OCR 체크 항목
- Taxpayer Name
- TIN(납세 번호): (형식 000-00-0000)
- Tax Year (형식 2026)
- 서명 (우측 하단)
- Date: (우측상단 January 12, 2026)
- 좌측 상단 인장 검출 (certification program 상단)

2. 아포스티유 OCR 체크 항목
- 10개 문항 작성 내용
- 좌측 하단 인장
- 우측 하단 서명 여부

3. 제한세율신청서 OCR 체크 항목
- 성명 (성/이름 분리 인식, First Name, Last Name)
- 주소
- 납세자 번호 (xxx-xx-xxxx 형식이 맞는지)
- 거주지국 및 거주지국 코드
- 배당소득 세율란: 한·미 조세조약에 따른 법정 제한세율인 15%
- 아니오에 모두 체크가 되어 있는지
- 서명 및 일자 (서명 누락 시 무조건 반려)

서류 간 교차 검증 항목
- 서류 1, 3: 성명, 납세자번호, 거주지국 

## 구현사항

1. 서류에서 정보 추출
2. 추출 정보 활용 
   - 양식 검사 
   - 경정청구서 작성 

## 최종 목표
자료가 잘 쓰여져 있는지, 누락된 빈칸은 없는지, 흐릿하진 않은지를 OCR로 검토해서 통과 or 반려되도록 함