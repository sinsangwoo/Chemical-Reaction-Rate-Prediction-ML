
# --- [공통] 필요 라이브러리 불러오기 ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import platform

# 머신러닝 관련 라이브러리
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

# Matplotlib 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin': # Mac OS
    plt.rc('font', family='AppleGothic')
else: # Linux
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False


# --- [단계 1] 고도화된 가상 실험 데이터 생성 ---
# 아레니우스 방정식 형태를 모사하여 온도의 비선형 효과 반영
print("--- [단계 1] 고도화된 가상 실험 데이터 생성 시작 ---")
num_samples = 300
np.random.seed(42)

# 독립 변수 생성
temperatures_c = np.random.uniform(10, 90, num_samples) # 섭씨 온도
temperatures_k = temperatures_c + 273.15 # 절대 온도 (K)
concentrations = np.random.uniform(0.1, 2.5, num_samples)
catalysts = np.random.randint(0, 2, num_samples)

# 아레니우스 방정식 기반 반응 속도 상수(k) 모사
A = 1e5  # 빈도 인자 (임의의 값)
Ea = 40000 # 활성화 에너지 (J/mol)
Ea_cat = 25000 # 촉매 사용 시 활성화 에너지
R = 8.314 # 기체 상수

k = A * np.exp(-Ea / (R * temperatures_k))
k_cat = A * np.exp(-Ea_cat / (R * temperatures_k))

# 속도 법칙 (rate = k * [C]^1) 모사
# 촉매 유무에 따라 다른 활성화 에너지(반응 속도 상수) 적용
base_rate = np.where(catalysts == 1, k_cat, k) * concentrations

# 노이즈 추가 (측정 오차 등)
noise = np.random.normal(0, np.mean(base_rate) * 0.05, num_samples)
reaction_rates = base_rate + noise
reaction_rates[reaction_rates < 0] = 0

data = pd.DataFrame({
    '온도(C)': temperatures_c,
    '농도(mol/L)': concentrations,
    '촉매사용여부': catalysts,
    '반응속도(mol/L·s)': reaction_rates
})

data.to_csv('chemical_reaction_data_advanced.csv', index=False)
print("고도화된 가상 데이터 생성 완료.")
print(data.head())
print("-" * 50)


# --- [단계 2] 데이터 전처리 및 분리 ---
print("--- [단계 2] 데이터 전처리 및 분리 시작 ---")
X = data[['온도(C)', '농도(mol/L)', '촉매사용여부']]
y = data['반응속도(mol/L·s)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"훈련 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
print("-" * 50)


# --- [단계 3 & 4] 다양한 모델 훈련 및 교차 검증 평가 ---
print("--- [단계 3 & 4] 모델 훈련 및 교차 검증 평가 시작 ---")

# 비교할 모델 정의
# 1. 선형 회귀
lr_model = LinearRegression()

# 2. 다항 회귀 (2차) - Pipeline으로 특성 변환과 모델을 묶음
poly_model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('linear_regression', LinearRegression())
])

# 3. 서포트 벡터 회귀 (SVR)
svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)

# 4. 랜덤 포레스트 회귀
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

models = {
    '선형 회귀': lr_model,
    '다항 회귀 (2차)': poly_model,
    'SVR': svr_model,
    '랜덤 포레스트': rf_model
}

# K-겹 교차 검증 설정 (K=5)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    # 교차 검증 수행 (평가 지표: R^2 Score)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    results[name] = cv_scores
    print(f"[{name}] 교차 검증 R² 점수: {cv_scores.mean():.4f} (표준편차: {cv_scores.std():.4f})")

print("-" * 50)

# --- [단계 5] 최종 모델 선택 및 예측 ---
print("--- [단계 5] 최종 모델 선택 및 새로운 조건 예측 ---")
# 교차 검증 결과, 랜덤 포레스트가 가장 우수하므로 최종 모델로 선택
final_model = RandomForestRegressor(n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
print("최종 모델(랜덤 포레스트) 훈련 완료.")

# 성능 평가 (테스트 데이터)
y_pred = final_model.predict(X_test)
final_mae = mean_absolute_error(y_test, y_pred)
final_r2 = r2_score(y_test, y_pred)
print(f"\n최종 모델의 테스트 데이터 성능:")
print(f"평균 절대 오차 (MAE): {final_mae:.4f}")
print(f"결정 계수 (R²): {final_r2:.4f}")

# 새로운 조건 예측
new_experiment = pd.DataFrame([[80, 1.5, 1]], columns=['온도(C)', '농도(mol/L)', '촉매사용여부'])
predicted_rate = final_model.predict(new_experiment)
print(f"\n입력된 실험 조건:\n{new_experiment}\n")
print(f"예측된 반응 속도: {predicted_rate[0]:.4f} mol/L·s")
print("-" * 50)


# --- [단계 6] 심화된 시각화 및 결과 분석 ---
print("--- [단계 6] 시각화 및 결과 분석 시작 ---")

# 시각화 1: 교차 검증 결과 비교
plt.figure(figsize=(12, 7))
sns.boxplot(data=pd.DataFrame(results))
plt.title('모델별 K-겹 교차 검증 R² 점수 분포', fontsize=16)
plt.ylabel('R² Score', fontsize=12)
plt.xlabel('머신러닝 모델', fontsize=12)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()
print("그래프 1: 모델별 교차 검증 성능 비교 그래프 생성 완료.")

# 시각화 2: 실제 값 vs. 최종 모델 예측 값 비교
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', label='예측 값')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='이상적인 예측 (y=x)')
plt.title('최종 모델의 실제-예측 비교 (랜덤 포레스트)', fontsize=16)
plt.xlabel('실제 반응 속도 (mol/L·s)', fontsize=12)
plt.ylabel('예측된 반응 속도 (mol/L·s)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
print("그래프 2: 최종 모델 실제-예측 비교 그래프 생성 완료.")

# 시각화 3: 최종 모델의 변수(특성) 중요도 분석
feature_importances = pd.Series(final_model.feature_importances_, index=X.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_sorted, y=feature_importances_sorted.index)
plt.title('최종 모델의 변수 중요도', fontsize=16)
plt.xlabel('중요도', fontsize=12)
plt.ylabel('변수 (입력 특성)', fontsize=12)
plt.show()
print("그래프 3: 변수 중요도 그래프 생성 완료.")
print("-" * 50)