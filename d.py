from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from Custom import StabilityAdaptiveKNN, ConfidenceAdaptiveKNN, DensityConfidenceKNN
from sklearn.neighbors import KNeighborsClassifier as KN
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Numero di prove per i test statistici
NUM_TRIALS = 5

# Leggi il file CSV
df = pd.read_csv("Parkinsson disease.csv")

# Rimuovi colonne non numeriche (es: 'name')
df = df.drop(columns=["name"])

# Se ci fossero colonne categoriche, convertila in numerico
for col in df.select_dtypes(include='object').columns:
    df[col] = pd.factorize(df[col])[0]

# Separazione feature (X) e target (y)
X = df.drop(columns=['status'])
y = df['status']

# Divisione in training e test set (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisci griglie di parametri per ogni classe
param_grids = [
    {
        'name': 'StabilityAdaptiveKNN',
        'estimator': StabilityAdaptiveKNN(),
        'params': {
            'stability_patience': [i for i in range(1, 6)],  # Ridotto per velocizzare
            'k_step_stability': [i for i in range(1, 6)],    # Ridotto per velocizzare
            'max_k_adaptive': [i for i in range(1,int(np.floor(np.sqrt(len(X_train)))),10)],  # Limitato a 20 per velocizzare
            'weights': ['uniform', 'distance']
        }
    },
    {
        'name': 'ConfidenceAdaptiveKNN',
        'estimator': ConfidenceAdaptiveKNN(),
        'params': {
            'confidence_threshold': [0.6, 0.7, 0.8, 0.9],
            'max_k_adaptive': [i for i in range(1,int(np.floor(np.sqrt(len(X_train)))),10)],,
            'weights': ['uniform', 'distance']
        }
    },
    {
        'name': 'DensityConfidenceKNN',
        'estimator': DensityConfidenceKNN(),
        'params': {
            'confidence_threshold': [0.6, 0.7, 0.8, 0.9],
            'density_quantile': [0.7, 0.8, 0.9],
            'max_k_adaptive': [i for i in range(1,int(np.floor(np.sqrt(len(X_train)))),10)],,
            'weights': ['uniform', 'distance']
        }
    },
    {
        'name': 'KNeighborsClassifier',
        'estimator': KN(),
        'params': {
            'n_neighbors': [i for i in range(1, min(20, int(np.floor(np.sqrt(len(X_train))))))],  # Limitato a 20 per velocizzare
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto'],  # Ridotto per velocizzare
            'metric': ['euclidean', 'manhattan']  # Ridotto per velocizzare
        }
    }
]

# Dizionario per memorizzare tutti i risultati
results = {}
cv_results_all = {}  # Per memorizzare i risultati di cross validation per ogni modello

# Configura la cross-validation stratificata
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Esegui GridSearchCV per ogni modello separatamente
for model in param_grids:
    estimator_name = model['name']
    estimator = model['estimator']
    param_grid = model['params']
    
    print(f"\n{'='*50}")
    print(f"Avvio GridSearchCV per {estimator_name}...")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # Esegui Grid Search per trovare i migliori parametri
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=10,  # CV ridotto per test più rapidi
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Recupera il miglior modello
        best_model = grid_search.best_estimator_
        
        # Valuta sul set di test
        y_pred_test = best_model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        
        # Calcola intervalli di confidenza usando bootstrapping
        n_bootstraps = 1000
        bootstrap_scores = []
        
        rng = np.random.RandomState(seed=42)
        for i in range(n_bootstraps):
            # Campiona con sostituzione
            indices = rng.randint(0, len(y_test), len(y_test))
            y_pred_bootstrap = y_pred_test[indices]
            y_true_bootstrap = y_test.values[indices]
            
            # Calcola accuratezza sul campione bootstrap
            bootstrap_score = accuracy_score(y_true_bootstrap, y_pred_bootstrap)
            bootstrap_scores.append(bootstrap_score)
        
        # Calcola intervallo di confidenza al 95%
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        
        elapsed_time = time.time() - start_time
        
        # Esegui cross-validation ripetuta per test statistici
        cv_scores = []
        for trial in range(NUM_TRIALS):
            cv_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=trial)
            scores = cross_val_score(best_model, X, y, cv=cv_fold, scoring='accuracy')
            cv_scores.extend(scores)
        
        cv_results_all[estimator_name] = cv_scores
        
        # Salva i risultati nel dizionario
        results[estimator_name] = {
            'Best Parameters': grid_search.best_params_,
            'CV Score': grid_search.best_score_,
            'Test Accuracy': accuracy_test,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            'Training Time': elapsed_time,
            'Model Object': best_model
        }
        
    except Exception as e:
        print(f"Errore durante GridSearchCV per {estimator_name}: {e}")

# Crea una lista ordinata di modelli per accuratezza
sorted_models = sorted(results.keys(), key=lambda x: results[x]['Test Accuracy'], reverse=True)

# Stampa tabella comparativa di tutti i modelli
print("\n\n" + "="*100)
print("CONFRONTO COMPLESSIVO DEI MODELLI (ordinati per accuratezza sul test set)")
print("="*100)

table_data = []
for i, model_name in enumerate(sorted_models):
    res = results[model_name]
    table_data.append([
        i+1,
        model_name,
        f"{res['Test Accuracy']:.4f}",
        f"[{res['CI Lower']:.4f}, {res['CI Upper']:.4f}]",
        f"{res['CV Score']:.4f}",
        f"{res['Training Time']:.2f} sec",
        str({k: v for k, v in res['Best Parameters'].items() if k not in ['algorithm', 'leaf_size', 'metric']})
    ])

headers = ["Rank", "Modello", "Test Accuracy", "95% CI", "CV Score", "Tempo", "Principali Parametri"]
print(tabulate(table_data, headers=headers, tablefmt="grid"))

# Esegui t-test accoppiato tra tutti i modelli
print("\n\n" + "="*80)
print("CONFRONTO STATISTICO TRA MODELLI (p-values da t-test accoppiato)")
print("="*80)

p_value_matrix = np.zeros((len(sorted_models), len(sorted_models)))
for i, model_i in enumerate(sorted_models):
    for j, model_j in enumerate(sorted_models):
        if i != j:
            # Esegui t-test accoppiato
            t_stat, p_val = ttest_rel(cv_results_all[model_i], cv_results_all[model_j])
            p_value_matrix[i, j] = p_val
        else:
            p_value_matrix[i, j] = 1.0  # Diagonale principale

# Crea tabella di p-values
p_value_table = []
for i, model_i in enumerate(sorted_models):
    row = [model_i]
    for j, model_j in enumerate(sorted_models):
        if i == j:
            row.append("-")
        else:
            p_val = p_value_matrix[i, j]
            significant = "**" if p_val < 0.05 else ""
            row.append(f"{p_val:.4f}{significant}")
    p_value_table.append(row)

p_headers = ["Model"] + sorted_models
print(tabulate(p_value_table, headers=p_headers, tablefmt="grid"))
print("** indica differenza statisticamente significativa (p < 0.05)")

# Visualizza i risultati di cross-validation con boxplot
plt.figure(figsize=(12, 6))
data_to_plot = [cv_results_all[model] for model in sorted_models]
ax = sns.boxplot(data=data_to_plot)
# Corretto l'errore: imposta prima i ticks e poi i ticklabels
ax.set_xticks(range(len(sorted_models)))
ax.set_xticklabels(sorted_models, rotation=45, ha='right')
plt.title('Confronto delle prestazioni dei modelli (Cross-Validation)')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.savefig('model_comparison_boxplot.png')
plt.close()

# Identifica il miglior modello
best_model_name = sorted_models[0]
best_model_result = results[best_model_name]
print("\n" + "="*80)
print(f"IL MIGLIOR MODELLO È: {best_model_name}")
print(f"Accuratezza sul Test Set: {best_model_result['Test Accuracy']:.4f}")
print(f"Intervallo di Confidenza al 95%: [{best_model_result['CI Lower']:.4f}, {best_model_result['CI Upper']:.4f}]")
print("Parametri Completi:", best_model_result['Best Parameters'])
print("="*80)

# Stampa report di classificazione dettagliato per il miglior modello
best_model = best_model_result['Model Object']
y_pred_best = best_model.predict(X_test)
print("\nReport di Classificazione Dettagliato per il Miglior Modello:")
print(classification_report(y_test, y_pred_best))

# Analisi statistica aggiuntiva del miglior modello
print("\n" + "="*80)
print(f"ANALISI STATISTICA AGGIUNTIVA PER {best_model_name}")
print("="*80)

mean_cv_score = np.mean(cv_results_all[best_model_name])
std_cv_score = np.std(cv_results_all[best_model_name])
se_cv_score = std_cv_score / np.sqrt(len(cv_results_all[best_model_name]))
t_critical = stats.t.ppf(0.975, len(cv_results_all[best_model_name])-1)
margin_error = t_critical * se_cv_score

print(f"Media CV Score: {mean_cv_score:.4f}")
print(f"Deviazione standard: {std_cv_score:.4f}")
print(f"Errore standard: {se_cv_score:.4f}")
print(f"Intervallo di confidenza t-test: {mean_cv_score:.4f} ± {margin_error:.4f}")
print(f"Intervallo di confidenza t-test: [{mean_cv_score-margin_error:.4f}, {mean_cv_score+margin_error:.4f}]")