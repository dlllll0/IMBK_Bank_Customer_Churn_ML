# 🏦 IMBK_Bank_Customer_Churn_ML

고객 이탈 분류 ML 모델링 및 데이터 기반 인사이트 도출 분석 > 본 프로젝트는 Kaggle의 은행 고객 데이터를 활용하여 이탈 여부를 예측하고, SHAP 및 사후분석을 통해 비즈니스 인사이트를 제안하는 머신러닝 풀 프로세스 프로젝트입니다.

## 1. 프로젝트 개요
- 목적 :  고객 이탈(Churn) 예측 모델 구현 및 핵심 이탈 원인 파악을 통한 비즈니스 전략 제안
- 기간 : 2026.04.10
- 주요 성과 : Stacking Ensemble을 통한 F1-Score 0.635 달성 및 성별/국가별 타겟 마케팅 인사이트 도출



## 2. 기술 스택

* **Language**
    <br><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=whitehttps://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">

* **Library**
    * **Data & Preprocessing**
        <br><img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=whitehttps://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"> <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=whitehttps://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"> <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=whitehttps://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
    * **Visualization**
        <br><img src="https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=blackhttps://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black"> <img src="https://img.shields.io/badge/Seaborn-4479A1?style=for-the-badge&logo=python&logoColor=whitehttps://img.shields.io/badge/Seaborn-4479A1?style=for-the-badge&logo=python&logoColor=white">
    * **AutoML & Tuning**
        <br><img src="https://img.shields.io/badge/PyCaret-6311C2?style=for-the-badge&logo=python&logoColor=whitehttps://img.shields.io/badge/PyCaret-6311C2?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Optuna-405BFF?style=for-the-badge&logo=python&logoColor=whitehttps://img.shields.io/badge/Optuna-405BFF?style=for-the-badge&logo=python&logoColor=white">
    * **Modeling Stacking Pipeline**
        <br>**[Layer 1: Base]**
        <br><img src="https://img.shields.io/badge/LGBM-0080FF?style=for-the-badge&logo=LGBM&logoColor=whitehttps://img.shields.io/badge/LGBM-0080FF?style=for-the-badge&logo=LGBM&logoColor=white"> <img src="https://img.shields.io/badge/GBC-3776AB?style=for-the-badge&logo=python&logoColor=whitehttps://img.shields.io/badge/GBC-3776AB?style=for-the-badge&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/RandomForest-228B22?style=for-the-badge&logo=scikit-learn&logoColor=whitehttps://img.shields.io/badge/RandomForest-228B22?style=for-the-badge&logo=scikit-learn&logoColor=white"> <img src="https://img.shields.io/badge/DecisionTree-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=whitehttps://img.shields.io/badge/DecisionTree-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
        <br>**[Layer 2: Meta]**
        <br><img src="https://img.shields.io/badge/Logistic--Regression-000000?style=for-the-badge&logo=scikit-learn&logoColor=whitehttps://img.shields.io/badge/Logistic--Regression-000000?style=for-the-badge&logo=scikit-learn&logoColor=white">
    * **XAI 사후 분석**
        <br><img src="https://img.shields.io/badge/SHAP-000000?style=for-the-badge&logo=SHAP&logoColor=whitehttps://img.shields.io/badge/SHAP-000000?style=for-the-badge&logo=SHAP&logoColor=white">



## 3. Data 정보
###  데이터 구성
- Source : https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data
- 규모: 10,000 Rows, 12 Columns
- Target: churn (0: 유지, 1: 이탈)

####  변수 설명
| 변수명 | 설명 | 타입 | 형태(예시) |
|---|---|---|---|
| customer_id | 고객 고유 식별 번호 | integer | 15634602 |
| credit_score | 신용점수 | integer | 300 - 850 |
| country | 거주 국가 | String | France, Spain, Germany |
| gender | 성별 | String | Male, Female |
| age | 나이 | integer | 18 - 92 |
| tenure | 은행 거래 기간 (년) | integer | 0 - 10 |
| balance | 잔고 (계좌 잔액) | Float | 0 - 250,898 |
| products_number | 이용중인 상품 수 | integer | 1, 2, 3, 4 |
| credit_card | 신용카드 보유 여부 | Binary | 1 (보유), 0 (미보유) |
| active_member | 활성 고객 여부 | Binary | 1 (활성), 0 (비활성) |
| estimated_salary | 추정 연봉 | Float | 11.58 - 199,992 |
| Churn | 이탈 여부(Target) | Binary | 1 (이탈), 0 (유지) |


---


## TASK
### 1. 데이터 전처리
> 데이터 가치 극대화 및 논리적 인코딩
- Feature Drop : 고유번호인 customer_id 제거(오버피팅 방지)  
- Encoding :
    - gender : Label Encoding (이진특성 유지)
    - Country : One-Hot Encoding (범주 간 순서 관계 제거 및 모델 해석력 향상)
- Scaling : 각 변수의 단위차이를 조정한 후 모델 학습의 안정성 확보하기 위해 Standard Scaling 진행
- Imbalance Control : 타겟 데이터의 불균형을 고려하여 stratify=y 적용 및 최종 모델 가중치 (class_weight = 'balance')조절.

### 2. EDA
  
- AGE : 30-40대는 유지율이 높으나, 40-50대에서 이탈률이 급격히 증가하는 양상을 포착함.
  
<img width="1002" height="548" alt="image" src="https://github.com/user-attachments/assets/f5e51942-a223-47f6-afcd-2daba6ed1d1e" />
  

  
- Balance : 이탈 고객 그룹의 잔고 중앙값이 유지고객보다 높음 → 고액 자산가의 이탈 위험 확인  
  
<img width="1484" height="984" alt="image" src="https://github.com/user-attachments/assets/695755da-35a9-43c7-bd3a-47fc77cc841b" />

  
  
- 상관계수  
상관계수는 전체적으로 높지않았다.  
churn과 다른 변수들을 봤을 때, gender(-0.1), age(0.28), balance(0.11), active_member(-0.15)임을 확인하였다.  
원핫인코딩 진행 이후 해당 변수에 nation_germany(0.17)가 추가됨을 확인할 수 있었고, nation_germany의 경우 balance와 0.4의 양의상관계수를 가짐을 확인할 수 있었다.  
  
- VIF  
VIF를 확인해 본 결과, 모든 변수가 1에 가까운 값을 가져 모델의 왜곡위험은 없다고 판단하였고, 모든 데이터를 다 넣어 데이터 가치를 극대화하는 전략을 취했음


### 3. 모델링 및 성능 평가
#### 모델링

- automl
<img width="811" height="298" alt="image" src="https://github.com/user-attachments/assets/d18ac560-20e0-4f10-88c1-4a659155ae2c" />

- ML선정
- 1단계 : 부스팅모델 중심의 스태킹으로 시작  
    Stacking F1-Score: 0.5997

- 2단계 : 클래스 불균형 보정(증강대신 class_weight='balanced', 후방모델에 가중치 부여)  
    Stacking F1-Score: 0.6310

- 3단계 : 원핫인코딩 진행 + 구조개선(원핫인코딩 적용 및 전/후방 중복 가중치 실험) 전후방으로 다 주면 너무 예민해져서 최적의 가중치 밸런스 도출  
    Stacking F1-Score: 0.6166

    - 최종 앙상블 : lgbm, gbc(쭉 성능 좋았던 2), rf(배깅 :: 다양성, 예측안정성), dt(recall 제일 높음) + 후방 = logistic  
        **Stacking F1-Score: 0.6350**  
<img width="1046" height="237" alt="image" src="https://github.com/user-attachments/assets/0cd6fae8-1f38-4d16-8a8a-ac6b62ffdc85" />

    - 모든 모델은 optuna로 하이퍼파라미터 튜닝을 거쳤고, 최종적으로 f1score은 0.6350을 도출  
    - 이탈 고객을 더 잘 잡아내는 모델을 만들기 위해 정밀도와 재현율을 모두 고려하는 f1score을 높이는 방향으로 모델링을 진행하였다.  


##### 최종 검증 결과

| metric | Score |
|---|---|
|F1-Score | 0.6350 |
|Accuracy | 0.8320 |
    
  

### 4. SHAP 분석

- feature importance
<img width="914" height="526" alt="image" src="https://github.com/user-attachments/assets/7e20fafe-429b-45a4-8fbf-9803e8e03e62" />

- shap
<img width="758" height="573" alt="image" src="https://github.com/user-attachments/assets/b535b483-b4b9-42e7-8080-ec22abe9726f" />


- 결정적 요인: age, products_number, active_member가 모델 의사결정에 가장 큰 기여를 하는 것을 확인했습니다.

- 영향력 방향:  
Age: 연령대가 높을수록 이탈 확률이 가파르게 상승하는 양의 영향력을 보임.  
Nation_Germany: 독일 거주 고객일수록 이탈 위험군에 속할 확률이 높음.  
Active_member / Gender: 활동적일수록, 남성일수록 이탈 확률이 낮아지는 경향 확인.

  
- 사후분석

    - shap valud에서 gender가 영향이 뚜렷함을 확인하여 추후분석 진행

      |gender|churn|count|
      |---|---|---|
      |0|0|3404|
      | |1|1139|
      |1|0|4559|
      | |1|898|

      > 여자의 이탈률 25%, 남자의 이탈률 16.5%
  
    - shap value에서 발견된 이용상품수가 높고, 이탈률을 높이는 군집 확인 products_number
    <img width="844" height="550" alt="image" src="https://github.com/user-attachments/assets/97fa748a-e46d-4b22-997d-5b761396ad3c" />
    
    | products_number | churn | count |
  |---|---|---|
  | 1 | 0 | 3675 |
  |  | 1 | 1409 |
  | 2 | 0 | 4242 |
  |  | 1 | 348 |
  | 3 | 0 | 46 |
  |  | 1 | 220 |
  | 4 | 1 | 60 |

      
    - country - balance의 상관관계가 0.4로 높았기에, 확인 진행
    <img width="1025" height="548" alt="image" src="https://github.com/user-attachments/assets/a87d0efc-ae77-45e1-86e3-f142425fe311" />
    
    --------- [국가별 잔액 통계] ---------
   
    | country | mean | median | std | count |
    |---|---|---|---|---|
    | France | 62092.63 | 62153.50 | 64133.56 | 5014 | 
    | Germany | 119730.11 | 119703.10 | 27022.00 | 2509 | 
    | Spain | 61818.14 | 61710.44 | 64235.55 | 2477 |

  
  
### 5. 데이터 기반 비즈니스 전략 제안
  
1) 상품 가입의 역설 (Product Paradox)  
현황: 보유 상품 수가 4개인 고객은 60명 전원이 이탈(100%) 하는 극단적인 현상 발견.
  
    전략: 무조건적인 상품 가입 권유는 오히려 고객 이탈을 초래할 수 있음. 2개 이하 보유 고객을 핵심 단골로 전환하기 위한 '서비스 만족도 제고'에 집중하고, 3개 이상 보유자에게는 '이탈 방지 전담 케어'가 필요함.
  
3) 독일 시장 및 고액 자산가 정밀 타겟팅  
현황: 독일 고객의 이탈률은 타 국가 대비 약 2배 높으며, 이들은 평균적으로 고액의 잔액(Balance)을 보유하고 있음.
  
    전략: 독일 시장 내 금리 경쟁력을 전면 재검토하고, 고액 잔액 유지 시 혜택을 주는 '프리미엄 멤버십' 도입 등 독일 거주 자산가 맞춤형 락인(Lock-in) 전략이 시급함.  
   
5) 4050 세대 및 여성 고객 특화 마케팅  
현황: 이탈 밀도가 가장 높은 40~50대 고령층과 남성(16.5%) 대비 높은 이탈률을 보이는 여성(25%) 고객군 확인.
  
    전략:  
    4050: 자산 관리 서비스 및 은퇴 설계 관련 금융 상담 서비스 강화.  
    여성: 여성 고객의 페인 포인트(Pain Point)를 파악하기 위한 심층 분석 및 여성 선호 혜택(쇼핑, 교육 등) 연계 캠페인 진행.  
