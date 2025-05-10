import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def regression_metrics_summary(y_true, y_pred, alpha=0.05):
    """
    Gera um resumo completo das métricas de regressão incluindo p-values e standard errors
    """
    # Números de observações
    n = len(y_true)
    
    # Calcular métricas básicas
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = np.mean((y_true - y_pred)**2)
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

# Dados do conjunto de validação externa
predicted_values = np.array([-4.00, -7.74, -3.94, -5.11, -2.53, -5.48, -2.28, -6.01, -5.07, -1.20])
actual_values = np.array([-4.68, -9.77, -4.11, -5.13, -1.55, -1.81, -1.96, -7.20, -5.08, -1.05])

# Criar DataFrame para visualização
df = pd.DataFrame({
    'Predicted_LPED': predicted_values,
    'Real_LPED': actual_values
})

# Calcular R² original (versão simples)
r2_original = r2_score(df['Real_LPED'], df['Predicted_LPED'])
print(f"R² (versão simples): {r2_original:.4f}")

# Calcular R² manualmente
real = np.array(df['Real_LPED'])
predicted = np.array(df['Predicted_LPED'])
mean_real = np.mean(real)
ss_res = np.sum((real - predicted) ** 2)
ss_tot = np.sum((real - mean_real) ** 2)
r2_manual = 1 - (ss_res / ss_tot)
print(f"R² (cálculo manual): {r2_manual:.4f}")

# Análise estatística avançada
print("\n" + "="*50)
print("ANÁLISE ESTATÍSTICA AVANÇADA - VALIDAÇÃO EXTERNA")
print("="*50)

# Obter métricas com p-values e standard errors
external_metrics = regression_metrics_summary(np.array(df['Real_LPED']), np.array(df['Predicted_LPED']))
print("\nMétricas detalhadas:")
print(external_metrics.to_string(index=False))

# Gerar relatório de desempenho do modelo
print("\n" + "="*50)
print("RELATÓRIO DE DESEMPENHO - VALIDAÇÃO EXTERNA")
print("="*50)

print(f"\nTamanho da amostra: {len(df)} observações")
print(f"\nR²: {external_metrics.loc[0, 'Value']:.4f} ± {external_metrics.loc[0, 'Std Error']:.4f}")
print(f"IC 95%: [{external_metrics.loc[0, 'Lower CI (95%)']:.4f}, {external_metrics.loc[0, 'Upper CI (95%)']:.4f}]")
print(f"p-value: {external_metrics.loc[0, 'p-value']:.6f}")

print(f"\nMAE: {external_metrics.loc[1, 'Value']:.4f} ± {external_metrics.loc[1, 'Std Error']:.4f}")
print(f"IC 95%: [{external_metrics.loc[1, 'Lower CI (95%)']:.4f}, {external_metrics.loc[1, 'Upper CI (95%)']:.4f}]")
print(f"p-value: {external_metrics.loc[1, 'p-value']:.6f}")

print(f"\nRMSE: {external_metrics.loc[2, 'Value']:.4f} ± {external_metrics.loc[2, 'Std Error']:.4f}")
print(f"IC 95%: [{external_metrics.loc[2, 'Lower CI (95%)']:.4f}, {external_metrics.loc[2, 'Upper CI (95%)']:.4f}]")
print(f"p-value: {external_metrics.loc[2, 'p-value']:.6f}")

# Visualizar resultados do modelo externo
plt.figure(figsize=(10, 8))

# Scatter plot com linha de identidade
plt.subplot(2, 1, 1)
plt.scatter(df['Real_LPED'], df['Predicted_LPED'], alpha=0.7)
plt.plot([df['Real_LPED'].min(), df['Real_LPED'].max()], 
         [df['Real_LPED'].min(), df['Real_LPED'].max()], 'r--')

# Adicionar linha de regressão com intervalo de confiança
slope, intercept, r_value, p_value, std_err = stats.linregress(df['Real_LPED'], df['Predicted_LPED'])
x_line = np.linspace(df['Real_LPED'].min(), df['Real_LPED'].max(), 100)
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, 'b-', label=f'Regressão (slope={slope:.2f})')

# Intervalo de confiança aproximado
plt.fill_between(x_line, 
                y_line - 1.96 * external_metrics.loc[2, 'Std Error'], 
                y_line + 1.96 * external_metrics.loc[2, 'Std Error'], 
                alpha=0.2, color='blue', label='IC 95%')

plt.title(f'Validação Externa: Valores Reais vs. Preditos\nR² = {external_metrics.loc[0, "Value"]:.4f} ± {external_metrics.loc[0, "Std Error"]:.4f} (p={external_metrics.loc[0, "p-value"]:.4f})', fontsize=14)
plt.xlabel('Valores Reais', fontsize=12)
plt.ylabel('Valores Preditos', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Residuais
plt.subplot(2, 1, 2)
residuals = df['Real_LPED'] - df['Predicted_LPED']
plt.scatter(df['Predicted_LPED'], residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')

# Adicionar intervalo de confiança para residuais
plt.axhline(y=np.mean(residuals) + 1.96 * np.std(residuals), color='g', linestyle=':', label='+1.96σ')
plt.axhline(y=np.mean(residuals) - 1.96 * np.std(residuals), color='g', linestyle=':', label='-1.96σ')

plt.xlabel('Valores Preditos', fontsize=12)
plt.ylabel('Residuais', fontsize=12)
plt.title('Análise de Residuais', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('external_validation_analysis.png', dpi=300)
plt.show()

# Visualizar as métricas com barras de erro
plt.figure(figsize=(10, 6))

x_pos = np.arange(len(external_metrics['Metric']))
plt.bar(x_pos, external_metrics['Value'], yerr=external_metrics['Std Error'], 
        capsize=10, alpha=0.7, color=['blue', 'green', 'red'])

for i, v in enumerate(external_metrics['Value']):
    plt.text(i, v + external_metrics['Std Error'][i] + 0.03, 
             f"{v:.3f} ± {external_metrics['Std Error'][i]:.3f}", 
             ha='center', va='bottom')

plt.xticks(x_pos, external_metrics['Metric'])
plt.xlabel('Métrica', fontsize=14)
plt.ylabel('Valor', fontsize=14)
plt.title('Métricas de Validação Externa com Erro Padrão', fontsize=16)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('external_metrics_bars.png', dpi=300)
plt.show()