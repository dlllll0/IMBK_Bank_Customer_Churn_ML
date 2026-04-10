# 🏦 IMBK_Bank_Customer_Churn_ML

> 고객 이탈 분류 ML 모델링 및 데이터 기반 인사이트 도출 분석 > 본 프로젝트는 Kaggle의 은행 고객 데이터를 활용하여 이탈 여부를 예측하고, SHAP 및 사후분석을 통해 비즈니스 인사이트를 제안하는 머신러닝 풀 프로세스 프로젝트입니다.

## 1. 프로젝트 개요
- 목적 :  고객 이탈(Churn) 예측 모델 구현 및 핵심 이탈 원인 파악을 통한 비즈니스 전략 제안
- 기간 : 2026.04.10
- 주요 성과 : Stacking Ensemble을 통한 F1-Score 0.635 달성 및 성별/국가별 타겟 마케팅 인사이트 도출


## 2. 기술 스택

* **Language**
    <br><img src="[https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)">

* **Library**
    * **Data**
        <br><img src="[https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)"> <img src="[https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)">
    * **Visualization**
        <br><img src="[https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=matplotlib&logoColor=black)"> <img src="[https://img.shields.io/badge/Seaborn-4479A1?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/Seaborn-4479A1?style=for-the-badge&logo=python&logoColor=white)">
    * **ML / AutoML**
        <br><img src="[https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)"> <img src="[https://img.shields.io/badge/PyCaret-6311C2?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/PyCaret-6311C2?style=for-the-badge&logo=python&logoColor=white)"> <img src="[https://img.shields.io/badge/Optuna-405BFF?style=for-the-badge&logo=python&logoColor=white](https://img.shields.io/badge/Optuna-405BFF?style=for-the-badge&logo=python&logoColor=white)">
    * **Modeling**
        <br><img src="[https://img.shields.io/badge/LGBM-0080FF?style=for-the-badge&logo=LGBM&logoColor=white](https://img.shields.io/badge/LGBM-0080FF?style=for-the-badge&logo=LGBM&logoColor=white)"> <img src="[https://img.shields.io/badge/XGBoost-2C3E50?style=for-the-badge&logo=XGBoost&logoColor=white](https://img.shields.io/badge/XGBoost-2C3E50?style=for-the-badge&logo=XGBoost&logoColor=white)"> <img src="[https://img.shields.io/badge/CatBoost-FFD200?style=for-the-badge&logo=CatBoost&logoColor=black](https://img.shields.io/badge/CatBoost-FFD200?style=for-the-badge&logo=CatBoost&logoColor=black)"> <img src="[https://img.shields.io/badge/RandomForest-228B22?style=for-the-badge&logo=scikit-learn&logoColor=white](https://img.shields.io/badge/RandomForest-228B22?style=for-the-badge&logo=scikit-learn&logoColor=white)">
    * **XAI**
        <br><img src="[https://img.shields.io/badge/SHAP-000000?style=for-the-badge&logo=SHAP&logoColor=white](https://img.shields.io/badge/SHAP-000000?style=for-the-badge&logo=SHAP&logoColor=white)">


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
