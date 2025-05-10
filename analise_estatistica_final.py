import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Função para gerar resumo estatístico completo
def regression_metrics_summary(y_true, y_pred, alpha=0.05):
    """
    Gera um resumo completo das métricas de regressão incluindo p-values e standard errors
    """
    # Números de observações
    n = len(y_true)
    
    # Calcular métricas básicas
    r2 = metrics.r2_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Média dos valores reais (para baseline)
    y_mean = np.mean(y_true)
    
    # Calcular residuais e soma dos quadrados
    residuals = y_true - y_pred
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum(residuals ** 2)
    
    # Graus de liberdade
    df_total = n - 1
    df_residual = n - 2  # Assumindo um modelo com intercepto
    
    # Standard errors para as métricas
    # Para R², utilizamos fórmula aproximada baseada na teoria assintótica
    se_r2 = np.sqrt(4 * r2 * (1 - r2)**2 / (n - 2)) if n > 2 else np.nan
    
    # Para MAE, usamos o erro padrão da média dos valores absolutos de resíduos
    abs_residuals = np.abs(residuals)
    se_mae = np.std(abs_residuals) / np.sqrt(n)
    
    # Para RMSE, utilizamos propagação de erro aproximada
    se_rmse = rmse / np.sqrt(2 * df_residual) if df_residual > 0 else np.nan
    
    # Cálculo de p-values
    # Para R², testamos contra 0 (nenhum poder preditivo)
    t_r2 = r2 / se_r2 if se_r2 > 0 else np.nan
    p_r2 = 2 * (1 - stats.t.cdf(abs(t_r2), df=df_residual)) if not np.isnan(t_r2) else np.nan
    
    # Para MAE e RMSE, comparamos com modelo nulo (predição da média)
    mae_null = np.mean(np.abs(y_true - y_mean))
    rmse_null = np.sqrt(np.mean((y_true - y_mean) ** 2))
    
    t_mae = (mae_null - mae) / se_mae if se_mae > 0 else np.nan
    p_mae = 1 - stats.t.cdf(t_mae, df=df_residual) if not np.isnan(t_mae) else np.nan
    
    t_rmse = (rmse_null - rmse) / se_rmse if se_rmse > 0 else np.nan
    p_rmse = 1 - stats.t.cdf(t_rmse, df=df_residual) if not np.isnan(t_rmse) else np.nan
    
    # Intervalo de confiança
    t_critical = stats.t.ppf(1 - alpha/2, df=df_residual)
    
    ci_lower_r2 = r2 - t_critical * se_r2 if not np.isnan(se_r2) else np.nan
    ci_upper_r2 = r2 + t_critical * se_r2 if not np.isnan(se_r2) else np.nan
    
    ci_lower_mae = mae - t_critical * se_mae if not np.isnan(se_mae) else np.nan
    ci_upper_mae = mae + t_critical * se_mae if not np.isnan(se_mae) else np.nan
    
    ci_lower_rmse = rmse - t_critical * se_rmse if not np.isnan(se_rmse) else np.nan
    ci_upper_rmse = rmse + t_critical * se_rmse if not np.isnan(se_rmse) else np.nan
    
    # Criar DataFrame com resultados
    results = pd.DataFrame({
        'Metric': ['R²', 'MAE', 'RMSE'],
        'Value': [r2, mae, rmse],
        'Std Error': [se_r2, se_mae, se_rmse],
        'Lower CI (95%)': [ci_lower_r2, ci_lower_mae, ci_lower_rmse],
        'Upper CI (95%)': [ci_upper_r2, ci_upper_mae, ci_upper_rmse],
        'p-value': [p_r2, p_mae, p_rmse],
        'Significant': [p_r2 < alpha if not np.isnan(p_r2) else np.nan, 
                        p_mae < alpha if not np.isnan(p_mae) else np.nan, 
                        p_rmse < alpha if not np.isnan(p_rmse) else np.nan]
    })
    
    return results

# Função auxiliar para obter as métricas básicas (do seu código original)
def get_metrics(Y_actual, Y_pred):
    dict_metrics = {
        'r2': metrics.r2_score(Y_actual, Y_pred),
        'mae': metrics.mean_absolute_error(Y_actual, Y_pred),
        'mse': np.mean((Y_actual - Y_pred)**2),
        'rmse': np.sqrt(np.mean((Y_actual - Y_pred)**2))
    }
    return dict_metrics

# Load the new dataset with the engineered features
file_path = 'Dataset3.xlsx'  # Update with your file path
data = pd.read_excel(file_path)

# Create a column called Inverse interatomic distance
data['Inverse interatomic distance'] = 1 / data['Interatomic distance']

# Create a column called Lennard_Jones interatomic distance
# Lennard-Jones potential parameters
epsilon = 1
sigma = 1

# Calculate Lennard-Jones potential
data['Lennard_Jones interatomic distance'] = 4 * epsilon * (
    (sigma * data['Inverse interatomic distance']) ** 12 - 
    (sigma * data['Inverse interatomic distance']) ** 6
)

# Drop the index column, categorical variable, and the original interatomic distance column
data_cleaned = data.drop(columns=['Index', 'Interaction', 'Interatomic distance', 'Inverse interatomic distance'])

# Data analysis 
pd.set_option('display.max_columns', None)
data_cleaned_describe = data_cleaned.describe()
print(data_cleaned_describe)

# Separate target (Y) and features (X)
X = data_cleaned.drop(columns=['LPED'])
Y = data_cleaned['LPED']

# Split the dataset into train and test sets with a 0.20 proportion with the best random_state value
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=4)

# Define the Regularized Gradient Boosting model with optimal parameters
best_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

# Create a pipeline with standardization
model_pipeline = make_pipeline(StandardScaler(), best_model)

# Train the model
model_pipeline.fit(X_train, Y_train)

# Make predictions
Y_pred_train = model_pipeline.predict(X_train)
Y_pred_test = model_pipeline.predict(X_test)

# Get basic metrics (do seu código original)
train_metrics = get_metrics(Y_train, Y_pred_train)
test_metrics = get_metrics(Y_test, Y_pred_test)

print("\nTraining Set Metrics:")
print(train_metrics)

print("\nTest Set Metrics:")
print(test_metrics)

# Calculate overfitting indicators
mse_ratio = test_metrics['mse'] / train_metrics['mse'] if train_metrics['mse'] > 0 else float('inf')
r2_gap = train_metrics['r2'] - test_metrics['r2']

print("\nOverfitting Indicators:")
print(f"MSE Ratio (Test/Train): {mse_ratio:.2f}x")
print(f"R² Gap (Train - Test): {r2_gap:.4f}")

# Extract feature importances from the model
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model_pipeline.named_steps['gradientboostingregressor'].feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feature_importances)

# NOVA PARTE: Análise estatística avançada com p-values e standard errors
print("\n" + "="*50)
print("ANÁLISE ESTATÍSTICA AVANÇADA")
print("="*50)

# Calcular métricas estatísticas para conjunto de treino
print("\nMétricas estatísticas para conjunto de TREINO:")
train_stats = regression_metrics_summary(Y_train.values, Y_pred_train)
print(train_stats.to_string(index=False))

# Calcular métricas estatísticas para conjunto de teste
print("\nMétricas estatísticas para conjunto de TESTE:")
test_stats = regression_metrics_summary(Y_test.values, Y_pred_test)
print(test_stats.to_string(index=False))

# Criar tabela comparativa com ambas métricas
comparison_df = pd.DataFrame({
    'Metric': train_stats['Metric'],
    'Train Value': train_stats['Value'],
    'Train SE': train_stats['Std Error'],
    'Train p-value': train_stats['p-value'],
    'Test Value': test_stats['Value'],
    'Test SE': test_stats['Std Error'],
    'Test p-value': test_stats['p-value']
})

print("\nComparação entre métricas de treino e teste:")
print(comparison_df.to_string(index=False))

# Visualize feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance in Regularized Gradient Boosting Model', fontsize=16)
plt.tight_layout()
plt.savefig('feature_importances.png', dpi=300)
plt.show()

# Visualize actual vs. predicted values
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred_test, alpha=0.7)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r--')
plt.xlabel('Actual Values', fontsize=14)
plt.ylabel('Predicted Values', fontsize=14)
plt.title(f'Actual vs. Predicted Values (Test Set, R² = {test_metrics["r2"]:.4f})', fontsize=16)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()

# Calculate and plot residuals
residuals = Y_test - Y_pred_test

plt.figure(figsize=(12, 5))

# Residuals vs predicted values
plt.subplot(1, 2, 1)
plt.scatter(Y_pred_test, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.title('Residuals vs Predicted Values', fontsize=14)
plt.grid(True, alpha=0.3)

# Histogram of residuals
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=10, alpha=0.7, edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Residual Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Histogram of Residuals', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residual_analysis.png', dpi=300)
plt.show()