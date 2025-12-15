import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Estado aleatório consistente
RANDOM_STATE = 4

# Carregar conjunto de dados
df = pd.read_excel('Dataset3.xlsx')

# Seleção de características
features = ['Interatomic distance', 'Atomic charge A (MK)', 'Atomic charge B (MK)']
target = 'LPED'

# Preparar dados
X = df[features]
y = df[target]

# Validar tamanho do conjunto de dados
num_samples = X.shape[0]
min_required_samples = 50  # Ajustar conforme necessário

if num_samples < min_required_samples:
    raise ValueError(f"Conjunto de dados muito pequeno ({num_samples} amostras). São necessárias pelo menos {min_required_samples}.")

# Dividir dados para avaliação final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# Função para criar curvas de aprendizado e avaliar modelos
def analyze_model(model, model_name, X, y, X_train, X_test, y_train, y_test):
    # Configurar validação cruzada
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    
    # Ajustar tamanhos de treinamento baseado no tamanho do conjunto de dados - menos pontos para espaçamento mais claro
    train_sizes = np.linspace(0.1, 1.0, 5)
    
    # Computar curva de aprendizado MSE
    train_sizes_abs, train_scores_mse, test_scores_mse = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring='neg_mean_squared_error'
    )
    
    # Converter MSE negativo para positivo
    train_mse = -train_scores_mse
    test_mse = -test_scores_mse
    
    # Computar curva de aprendizado R²
    _, train_scores_r2, test_scores_r2 = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes,
        scoring='r2'
    )
    
    # Calcular médias e desvios padrão
    train_mse_mean = np.mean(train_mse, axis=1)
    train_mse_std = np.std(train_mse, axis=1)
    test_mse_mean = np.mean(test_mse, axis=1)
    test_mse_std = np.std(test_mse, axis=1)
    
    train_r2_mean = np.mean(train_scores_r2, axis=1)
    train_r2_std = np.std(train_scores_r2, axis=1)
    test_r2_mean = np.mean(test_scores_r2, axis=1)
    test_r2_std = np.std(test_scores_r2, axis=1)
    
    # Criar gráfico de curva de aprendizado MSE
    plt.figure(figsize=(12, 6))
    plt.title(f"Curvas de Aprendizado (MSE) - {model_name}", fontsize=18)
    plt.xlabel("Exemplos de treinamento", fontsize=16)
    plt.ylabel("Erro Quadrático Médio", fontsize=16)
    
    plt.fill_between(train_sizes_abs, train_mse_mean - train_mse_std,
                     train_mse_mean + train_mse_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, test_mse_mean - test_mse_std,
                     test_mse_mean + test_mse_std, alpha=0.1, color='g')
    
    plt.plot(train_sizes_abs, train_mse_mean, 'o-', color='r', label='Pontuação de treinamento', markersize=8)
    plt.plot(train_sizes_abs, test_mse_mean, 'o-', color='g', label='Pontuação de validação cruzada', markersize=8)
    
    plt.xticks(train_sizes_abs, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'{model_name.replace(" ", "_")}_curva_aprendizado_MSE.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Criar gráfico de curva de aprendizado R²
    plt.figure(figsize=(12, 6))
    plt.title(f"Curvas de Aprendizado (R²) - {model_name}", fontsize=18)
    plt.xlabel("Exemplos de treinamento", fontsize=16)
    plt.ylabel("Pontuação R²", fontsize=16)
    
    plt.fill_between(train_sizes_abs, train_r2_mean - train_r2_std,
                     train_r2_mean + train_r2_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes_abs, test_r2_mean - test_r2_std,
                     test_r2_mean + test_r2_std, alpha=0.1, color='g')
    
    plt.plot(train_sizes_abs, train_r2_mean, 'o-', color='r', label='Pontuação de treinamento', markersize=8)
    plt.plot(train_sizes_abs, test_r2_mean, 'o-', color='g', label='Pontuação de validação cruzada', markersize=8)
    
    plt.xticks(train_sizes_abs, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(loc='best', fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(f'{model_name.replace(" ", "_")}_curva_aprendizado_R2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Ajustar modelo e obter predições
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calcular métricas
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Criar tabela de métricas
    metrics = pd.DataFrame({
        'Métrica': ['MSE', 'RMSE', 'MAE', 'R²'],
        'Conjunto de Treinamento': [train_mse, train_rmse, train_mae, train_r2],
        'Conjunto de Teste': [test_mse, test_rmse, test_mae, test_r2]
    })
    
    metrics['Diferença (%)'] = ((metrics['Conjunto de Teste'] - metrics['Conjunto de Treinamento']) / 
                               metrics['Conjunto de Treinamento'] * 100)
    
    # Formatar as métricas
    formatted_metrics = metrics.copy()
    formatted_metrics['Conjunto de Treinamento'] = formatted_metrics['Conjunto de Treinamento'].apply(lambda x: f"{x:.4f}")
    formatted_metrics['Conjunto de Teste'] = formatted_metrics['Conjunto de Teste'].apply(lambda x: f"{x:.4f}")
    formatted_metrics['Diferença (%)'] = formatted_metrics['Diferença (%)'].apply(lambda x: f"{x:.2f}%")
    
    # Calcular métricas de sobreajuste
    mse_ratio = test_mse / train_mse if train_mse > 0 else float('inf')
    r2_gap = train_r2 - test_r2
    
    # Imprimir resultados
    print("\n" + "="*80)
    print(f"ANÁLISE DO MODELO - {model_name}")
    print("="*80)
    
    print("\nMétricas de Desempenho do Modelo:")
    print(formatted_metrics.to_string(index=False))
    
    print("\nIndicadores de Sobreajuste:")
    print(f"Razão MSE (Teste/Treinamento): {mse_ratio:.2f}x")
    print(f"Lacuna R² (Treinamento - Teste): {r2_gap:.4f}")
    
    # Analisar convergência das curvas de aprendizado
    final_train_cv_mse_gap = test_mse_mean[-1] - train_mse_mean[-1]
    final_train_cv_r2_gap = train_r2_mean[-1] - test_r2_mean[-1]
    
    is_converging_mse = abs(test_mse_mean[-1] - train_mse_mean[-1]) < abs(test_mse_mean[0] - train_mse_mean[0])
    is_converging_r2 = abs(train_r2_mean[-1] - test_r2_mean[-1]) < abs(train_r2_mean[0] - test_r2_mean[0])
    
    print("\nAnálise de Curva de Aprendizado:")
    print(f"Lacuna Final MSE VC-Treinamento: {final_train_cv_mse_gap:.4f}")
    print(f"Lacuna Final R² VC-Treinamento: {final_train_cv_r2_gap:.4f}")
    print(f"Curvas MSE estão {'convergindo' if is_converging_mse else 'não convergindo'}")
    print(f"Curvas R² estão {'convergindo' if is_converging_r2 else 'não convergindo'}")
    
    # Visualizar real vs predito
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Valores Reais', fontsize=14)
    plt.ylabel('Valores Preditos', fontsize=14)
    plt.title(f'{model_name}: Real vs Predito (R² = {test_r2:.4f})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_")}_real_vs_predito.png', dpi=300)
    plt.show()
    
    return {
        'model': model,
        'train_sizes': train_sizes_abs,
        'train_mse_mean': train_mse_mean,
        'test_mse_mean': test_mse_mean,
        'train_r2_mean': train_r2_mean,
        'test_r2_mean': test_r2_mean,
        'metrics': metrics,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'mse_ratio': mse_ratio,
        'r2_gap': r2_gap
    }

# Definir vários modelos Gradient Boosting refinados
models = {
    "GB Regularizado Original": make_pipeline(
        StandardScaler(), 
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
    ),
    
    "GB Mais Regularizado": make_pipeline(
        StandardScaler(), 
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.02,     # Reduzido ainda mais
            max_depth=1,            # Árvores ainda menores
            min_samples_split=5,
            min_samples_leaf=3,     # Aumentado
            subsample=0.7,          # Reduzido
            random_state=42
        )
    ),
    
    "GB com Parada Antecipada": make_pipeline(
        StandardScaler(), 
        GradientBoostingRegressor(
            n_estimators=500,           # Mais árvores, mas serão limitadas pela parada antecipada
            learning_rate=0.05,
            max_depth=2,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            validation_fraction=0.2,    # Usar 20% dos dados de treinamento para validação
            n_iter_no_change=10,        # Parar se não houver melhoria por 10 iterações
            tol=0.001,                  # Tolerância para melhoria
            random_state=42
        )
    ),
    
    "GB Balanceado": make_pipeline(
        StandardScaler(), 
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.03,         # Meio termo
            max_depth=2,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.85,             # Leve aumento
            max_features=0.8,           # Usar subconjunto de características por árvore
            random_state=42
        )
    )
}

# Analisar cada modelo
results = {}
for name, model in models.items():
    print(f"\nAnalisando {name}...")
    results[name] = analyze_model(model, name, X, y, X_train, X_test, y_train, y_test)

# Criar tabela de comparação
comparison = pd.DataFrame({
    'Modelo': list(results.keys()),
    'R² Teste': [results[m]['test_r2'] for m in results],
    'MSE Teste': [results[m]['test_mse'] for m in results],
    'Razão MSE': [results[m]['mse_ratio'] for m in results],
    'Lacuna R²': [results[m]['r2_gap'] for m in results]
}).sort_values(by='R² Teste', ascending=False)

print("\n" + "="*80)
print("RESUMO DE COMPARAÇÃO DE MODELOS")
print("="*80)
pd.set_option('display.float_format', '{:.4f}'.format)
print(comparison)

# Encontrar melhor modelo
best_model_name = comparison.iloc[0]['Modelo']
best_model = results[best_model_name]['model']

print(f"\nMelhor modelo: {best_model_name}")
print(f"R² Teste: {comparison.iloc[0]['R² Teste']}")
print(f"MSE Teste: {comparison.iloc[0]['MSE Teste']}")

# Plotar comparação de curvas de aprendizado (R²)
plt.figure(figsize=(12, 8))
plt.title("Comparação de Curvas de Aprendizado (R²)", fontsize=18)
plt.xlabel("Exemplos de treinamento", fontsize=16)
plt.ylabel("Pontuação R² de Validação Cruzada", fontsize=16)

for name in results:
    plt.plot(results[name]['train_sizes'], results[name]['test_r2_mean'], 'o-', label=name)

plt.legend(loc='best', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('comparacao_curva_aprendizado_r2.png', dpi=300)
plt.show()

# Plotar comparação de curvas de aprendizado (MSE)
plt.figure(figsize=(12, 8))
plt.title("Comparação de Curvas de Aprendizado (MSE)", fontsize=18)
plt.xlabel("Exemplos de treinamento", fontsize=16)
plt.ylabel("MSE de Validação Cruzada", fontsize=16)

for name in results:
    plt.plot(results[name]['train_sizes'], results[name]['test_mse_mean'], 'o-', label=name)

plt.legend(loc='best', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('comparacao_curva_aprendizado_mse.png', dpi=300)
plt.show()

# Extrair e visualizar importâncias de características do melhor modelo
if hasattr(best_model.named_steps['gradientboostingregressor'], 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Característica': features,
        'Importância': best_model.named_steps['gradientboostingregressor'].feature_importances_
    }).sort_values(by='Importância', ascending=False)
    
    print("\nImportâncias de Características (Melhor Modelo):")
    print("="*80)
    print(feature_importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Característica'], feature_importances['Importância'])
    plt.xlabel('Importância', fontsize=14)
    plt.title(f'Importâncias de Características - {best_model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig('importancias_caracteristicas_melhor_modelo.png', dpi=300)
    plt.show()
