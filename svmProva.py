# -*- coding: utf-8 -*-
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split # train_test_split non più per tuning globale
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KN
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as t_dist
from sklearn.base import clone # Importato per clonare l'estimatore base
from Custom import StabilityAdaptiveKNN, ConfidenceAdaptiveKNN, DensityConfidenceKNN
import scikit_posthocs as sp

# --- CONFIGURAZIONI ---
DATASET_CONFIGS = [
    {
        "name": "Parkinson", "file_path": "Parkinsson disease.csv", "target_column": "status",
        "columns_to_drop": ["name"], "separator": ",", "categorical_cols_to_factorize": []
    },
    {
        "name": "Diabetes", "file_path": "diabetes_dataset.csv", "target_column": "Outcome",
        "columns_to_drop": [], "separator": ",", "categorical_cols_to_factorize": []
    },
    {
        "name": "Marketing Campaign", "file_path": "marketing_campaign.csv", "target_column": "Response",
        "columns_to_drop": ["ID"], "separator": "\t", "categorical_cols_to_factorize": ["Education", "Marital_Status"]
    }
]

NUM_ITER = 10 # Outer trials
N_SPLITS_OUTER_CV = 10 # Outer K-folds
N_SPLITS_INNER_CV = 5 # Inner K-folds for GridSearchCV
CONFIDENCE_LEVEL_CI = 0.95

# --- FUNZIONI STATISTICHE ---
def corrected_resampled_ttest(scores1, scores2, n_train, n_test):
    differences = np.array(scores1) - np.array(scores2)
    k = len(differences)
    if k < 2: return 1.0
    mean_diff = np.mean(differences)
    var_diff = np.var(differences, ddof=1)
    if var_diff < 1e-10: return 1.0 if np.isclose(mean_diff, 0) else 0.0
    if n_train <= 0:
        print("AVVISO: n_train <= 0 in corrected_resampled_ttest.")
        return 1.0
    variance_correction_factor = (1/k + n_test/n_train)
    if variance_correction_factor <= 0:
        print("AVVISO: Fattore di correzione non positivo in corrected_resampled_ttest.")
        return 1.0
    denominator_corrected = np.sqrt(variance_correction_factor * var_diff)
    if denominator_corrected < 1e-10: return 1.0 if np.isclose(mean_diff, 0) else 0.0
    t_stat = mean_diff / denominator_corrected
    df_degrees_of_freedom = k - 1
    p_val = t_dist.sf(np.abs(t_stat), df_degrees_of_freedom) * 2
    return p_val

def calculate_confidence_interval_t(data, confidence=0.95):
    a = np.array(data)[~np.isnan(data)]
    n = len(a)
    if n < 2: return (np.nan, np.nan)
    mean = np.mean(a)
    std_err = np.std(a, ddof=1) / np.sqrt(n)
    alpha = 1 - confidence
    t_critical = t_dist.ppf(1 - alpha / 2, n - 1)
    ci_lower = mean - t_critical * std_err
    ci_upper = mean + t_critical * std_err
    return (ci_lower, ci_upper)

# --- FUNZIONE PER GRIGLIE DI PARAMETRI DINAMICHE ---
def get_param_grids(X_train_len_ds):
    if X_train_len_ds <= 0: X_train_len_ds = 1
    upper_bound_adaptive_k = int(np.floor(np.sqrt(X_train_len_ds)))
    if upper_bound_adaptive_k < 1: upper_bound_adaptive_k = 1
    step_adaptive = max(1, upper_bound_adaptive_k // 10 if upper_bound_adaptive_k > 10 else 1)
    adaptive_k_values = list(range(1, upper_bound_adaptive_k + 1, step_adaptive))
    if not adaptive_k_values: adaptive_k_values = [min(1, X_train_len_ds)]

    upper_bound_knn = int(np.floor(np.sqrt(X_train_len_ds)))
    if upper_bound_knn < 1: upper_bound_knn = 1
    n_neighbors_knn_max = min(20, upper_bound_knn)
    n_neighbors_knn = list(range(1, n_neighbors_knn_max + 1))
    if not n_neighbors_knn: n_neighbors_knn = [min(1, X_train_len_ds)]

    grids = []
    if CUSTOM_CLASSES_LOADED:
        grids.extend([
            {'name': 'StabilityAdaptiveKNN', 'estimator': StabilityAdaptiveKNN(),
             'params': {'stability_patience': [1, 2, 3], 'k_step_stability': [1, 2],
                        'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}},
            {'name': 'ConfidenceAdaptiveKNN', 'estimator': ConfidenceAdaptiveKNN(),
             'params': {'confidence_threshold': [0.6, 0.7, 0.8], 'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}},
            {'name': 'DensityConfidenceKNN', 'estimator': DensityConfidenceKNN(),
             'params': {'confidence_threshold': [0.6, 0.7, 0.8], 'density_quantile': [0.7, 0.8, 0.9],
                        'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}}
        ])
    grids.append(
        {'name': 'KNeighborsClassifier', 'estimator': KN(),
         'params': {'n_neighbors': n_neighbors_knn, 'weights': ['uniform', 'distance'],
                    'algorithm': ['auto'], 'metric': ['euclidean', 'manhattan']}})
    if not grids: raise ValueError("ERRORE CRITICO: Nessuna griglia definita.")
    return grids

ALL_POTENTIAL_MODEL_NAMES = [model_config['name'] for model_config in get_param_grids(X_train_len_ds=100)] # Placeholder, aggiornato dopo caricamento X

# --- PREPROCESSING ---
def preprocess_dataset(df, config, target_col_name):
    df_processed = df.copy()
    df_processed = df_processed.drop(columns=config.get("columns_to_drop", []), errors='ignore')
    for col_loop in config.get("categorical_cols_to_factorize", []):
        if col_loop in df_processed.columns and df_processed[col_loop].dtype == 'object':
            df_processed[col_loop] = pd.factorize(df_processed[col_loop])[0]
    for col_loop in df_processed.select_dtypes(include='object').columns:
        if col_loop != target_col_name:
            df_processed[col_loop] = pd.factorize(df_processed[col_loop])[0]
    if target_col_name not in df_processed.columns:
        raise ValueError(f"Target column '{target_col_name}' non trovata.")
    X = df_processed.drop(columns=[target_col_name])
    y = df_processed[target_col_name]
    if y.dtype == 'object':
        y = pd.factorize(y)[0]
        y = pd.Series(y, index=df_processed.index)
    if not X.empty:
        X_numeric = X.select_dtypes(include=np.number)
        X_numeric = X_numeric.dropna() # Drop rows with NaN in numeric features
        X = X.loc[X_numeric.index]
        y = y.loc[X_numeric.index] # Align y with X
    if X.empty and not df_processed.empty :
        raise ValueError(f"Nessun dato rimasto in X dopo il preprocessing per {config['name']}.")
    return X, y

# --- NESTED REPEATED CROSS-VALIDATION ---
def nested_repeated_cross_validation(X, y, base_estimator_proto, param_grid,
                                     n_trials=NUM_ITER,
                                     n_splits_outer=N_SPLITS_OUTER_CV,
                                     n_splits_inner=N_SPLITS_INNER_CV,
                                     random_state_base=42,
                                     dataset_idx_for_inner_cv_rand=0): # Per variare random_state dell'inner CV
    outer_scores = []
    
    print(f"    Inizio Nested CV: {n_trials} trials, {n_splits_outer}-fold outer CV, {n_splits_inner}-fold inner CV for tuning.")
    
    for trial in range(n_trials):
        outer_skf = StratifiedKFold(n_splits=n_splits_outer, shuffle=True, random_state=random_state_base + trial)
        fold_num = 0
        for train_outer_idx, test_outer_idx in outer_skf.split(X, y):
            X_train_outer, X_test_outer = X.iloc[train_outer_idx], X.iloc[test_outer_idx]
            y_train_outer, y_test_outer = y.iloc[train_outer_idx], y.iloc[test_outer_idx]

            inner_cv_random_state = random_state_base + trial * n_splits_outer + fold_num + dataset_idx_for_inner_cv_rand
            inner_cv_gridsearch = StratifiedKFold(n_splits=n_splits_inner, shuffle=True, random_state=inner_cv_random_state)
            current_base_estimator = clone(base_estimator_proto)
            gs = GridSearchCV(current_base_estimator, param_grid, cv=inner_cv_gridsearch,
                              scoring='accuracy', error_score=np.nan, n_jobs=-1) 
            best_estimator_for_fold = None
            try:
                gs.fit(X_train_outer, y_train_outer)
                if gs.best_estimator_ is not None:
                    best_estimator_for_fold = gs.best_estimator_
                    score = best_estimator_for_fold.score(X_test_outer, y_test_outer)
                    outer_scores.append(score)
                else:
                    print(f"    ATTENZIONE (Trial {trial+1}, OuterFold {fold_num+1}): GridSearchCV non ha trovato un best_estimator_. Aggiungo NaN.")
                    outer_scores.append(np.nan)
            except ValueError as ve: # Spesso per problemi di stratificazione con pochi campioni/classi
                 print(f"    ERRORE ValueError in GridSearchCV/fit (Trial {trial+1}, OuterFold {fold_num+1}): {ve}. Aggiungo NaN.")
                 outer_scores.append(np.nan)
            except Exception as e:
                print(f"    ERRORE generico in GridSearchCV/fit/score (Trial {trial+1}, OuterFold {fold_num+1}): {e}. Aggiungo NaN.")
                outer_scores.append(np.nan)
            fold_num += 1
            
    return outer_scores

# --- CICLO SUI DATASET ---
all_repeated_cv_means_global = {name: [] for name in ALL_POTENTIAL_MODEL_NAMES} # Inizializzato prima, poi aggiornato
processed_datasets_count = 0
TOTAL_EXPECTED_SCORES_PER_MODEL_DS = NUM_ITER * N_SPLITS_OUTER_CV

for dataset_idx, config in enumerate(DATASET_CONFIGS):
    dataset_name = config["name"]
    dataset_file = config["file_path"]
    target_col = config["target_column"]

    print(f"\n{'='*50}\n ELABORAZIONE DATASET: {dataset_name} ({dataset_file})\n{'='*50}")

    try:
        df_loaded = pd.read_csv(dataset_file, sep=config.get("separator", ","))
    except Exception as e:
        print(f"ERRORE caricamento {dataset_file}: {e}. Salto dataset.")
        continue

    try:
        X, y = preprocess_dataset(df_loaded, config, target_col)
        if X.empty or y.empty:
            print(f"Dataset {dataset_name} è vuoto dopo il preprocessing. Salto.")
            continue
        
        # Controllo per StratifiedKFold (sia outer che inner)
        min_samples_for_cv = max(N_SPLITS_OUTER_CV, N_SPLITS_INNER_CV) # Deve essere almeno N_SPLITS_INNER_CV per il training set di un outer fold
        n_classes = len(y.unique())
        
        class_counts = y.value_counts()
        if any(class_counts < min_samples_for_cv): # O n_splits_inner, n_splits_outer
             print(f"Dataset {dataset_name} ha classi con meno campioni ({class_counts.min()}) di n_splits ({min_samples_for_cv}) richiesti per StratifiedKFold. Salto.")
             continue
        if len(X) < min_samples_for_cv or n_classes < 2 :
             print(f"Dataset {dataset_name} non ha abbastanza campioni o classi ({len(X)} campioni, {n_classes} classi) per CV con {min_samples_for_cv} splits. Salto.")
             continue
    except ValueError as e:
        print(f"ERRORE nel preprocessing di {dataset_name}: {e}. Salto dataset.")
        continue
    except Exception as e:
        print(f"ERRORE sconosciuto nel preprocessing di {dataset_name}: {e}. Salto dataset.")
        continue

    # Aggiorna ALL_POTENTIAL_MODEL_NAMES se non ancora fatto con la lunghezza di X corrente
    # (O meglio, fallo all'inizio del ciclo dataset se le griglie dipendono da len(X) in modo significativo)
    current_param_grids = get_param_grids(len(X)) # X qui è l'intero dataset caricato per 'dataset_name'
    current_model_names = [model_cfg['name'] for model_cfg in current_param_grids]
    
    # Assicurati che all_repeated_cv_means_global contenga tutti i nomi dei modelli correnti
    for model_name_iter in current_model_names:
        if model_name_iter not in all_repeated_cv_means_global:
            all_repeated_cv_means_global[model_name_iter] = []

    scores_per_model_dataset = {}
    print(f"\nInizio Nested Repeated Cross-Validation per {dataset_name} (X shape: {X.shape})...")

    for model_config in current_param_grids:
        estimator_name = model_config['name']
        base_estimator_proto = model_config['estimator'] # Prototipo, sarà clonato
        param_grid = model_config['params']

        if not CUSTOM_CLASSES_LOADED and estimator_name != 'KNeighborsClassifier':
            print(f"  Salto {estimator_name} perché le classi custom non sono caricate.")
            # Aggiungiamo NaN per questo modello e dataset per mantenere la coerenza nelle analisi globali
            scores_per_model_dataset[estimator_name] = [np.nan] * TOTAL_EXPECTED_SCORES_PER_MODEL_DS
            if estimator_name in all_repeated_cv_means_global: # Dovrebbe esserci
                 all_repeated_cv_means_global[estimator_name].append(np.nan)
            else: # Fallback se non inizializzato correttamente prima
                 all_repeated_cv_means_global[estimator_name] = [np.nan]
            continue

        print(f"  Eseguo Nested CV per {estimator_name}...")
        
        # La random_state_base per nested_repeated_cross_validation dovrebbe essere consistente per lo stesso dataset
        # ma dataset_idx può essere usato per variare gli inner random_state se si vuole più diversità
        current_scores = nested_repeated_cross_validation(
            X, y, base_estimator_proto, param_grid,
            n_trials=NUM_ITER,
            n_splits_outer=N_SPLITS_OUTER_CV,
            n_splits_inner=N_SPLITS_INNER_CV, # Usiamo la nuova costante
            random_state_base=dataset_idx * 1000, # Base per trial e fold esterni
            dataset_idx_for_inner_cv_rand=dataset_idx # Per diversificare ulteriormente gli inner CV random states
        )
        scores_per_model_dataset[estimator_name] = current_scores
        
        valid_scores_current_ds = np.array(current_scores)[~np.isnan(current_scores)]
        if len(valid_scores_current_ds) > 0:
            all_repeated_cv_means_global[estimator_name].append(np.mean(valid_scores_current_ds))
        else:
            # Se tutti i tentativi di nested CV falliscono (es. tutti NaN), aggiungiamo un NaN alla media globale
            all_repeated_cv_means_global[estimator_name].append(np.nan)


    if not scores_per_model_dataset:
        print(f"Nessun modello valutato con successo per {dataset_name}. Salto summary e test locali.")
        processed_datasets_count +=1
        continue

    # Il resto del codice per summary locale, test t locali, etc. può rimanere simile
    # Assicurati che 'ALL_POTENTIAL_MODEL_NAMES' sia aggiornato o usi 'current_model_names' per le tabelle locali
    summary_local = []
    model_names_for_local_analysis = list(scores_per_model_dataset.keys())

    for model_name in model_names_for_local_analysis:
        scores_list = scores_per_model_dataset.get(model_name, []) # Usa .get per sicurezza
        valid_scores = np.array(scores_list)[~np.isnan(scores_list)]
        n_valid = len(valid_scores)
        mean_acc = np.mean(valid_scores) if n_valid > 0 else np.nan
        std_dev = np.std(valid_scores, ddof=1) if n_valid > 1 else np.nan
        ci_lower, ci_upper = calculate_confidence_interval_t(valid_scores, confidence=CONFIDENCE_LEVEL_CI)
        summary_local.append({
            "Modello": model_name, "Mean Acc": mean_acc, "Std Dev": std_dev,
            "CI Lower": ci_lower, "CI Upper": ci_upper, "Valid Scores Num": n_valid
        })
    summary_local = sorted(summary_local, key=lambda x: x["Mean Acc"] if not pd.isna(x["Mean Acc"]) else -1, reverse=True)

    table_data_local = [[rank+1, item["Modello"],
                         f"{item['Mean Acc']:.4f}" if not pd.isna(item['Mean Acc']) else "N/A",
                         f"[{item['CI Lower']:.4f}, {item['CI Upper']:.4f}]" if not pd.isna(item['CI Lower']) else "N/A",
                         f"{item['Std Dev']:.4f}" if not pd.isna(item['Std Dev']) else "N/A",
                         f"{item['Valid Scores Num']}/{TOTAL_EXPECTED_SCORES_PER_MODEL_DS}"]
                        for rank, item in enumerate(summary_local)]
    headers_local = ["Rank", "Modello", "Mean Acc", f"{int(CONFIDENCE_LEVEL_CI*100)}% CI (t-dist)", "Std Dev (s)", "Valid Scores"]
    print(f"\nRisultati per {dataset_name} (Nested CV):")
    print(tabulate(table_data_local, headers=headers_local, tablefmt="grid"))

    print("\n--- Test Statistici LOCALI (Corrected Resampled t-test) ---")
    n_total_samples_dataset = len(X)
    if N_SPLITS_OUTER_CV <= 0: # N_SPLITS_OUTER_CV è n_splits_outer
        print("ATTENZIONE: N_SPLITS_OUTER_CV (outer) non valido. Impossibile calcolare n_train/n_test per t-test.")
        can_run_local_ttest = False
    else:
        n_test_for_ttest = n_total_samples_dataset / N_SPLITS_OUTER_CV
        n_train_for_ttest = n_total_samples_dataset - n_test_for_ttest
        can_run_local_ttest = n_train_for_ttest > 0 and n_test_for_ttest > 0

    if len(model_names_for_local_analysis) >=2 and can_run_local_ttest:
        p_value_matrix_local = pd.DataFrame(index=model_names_for_local_analysis, columns=model_names_for_local_analysis, dtype=float)
        for i, model_i_name in enumerate(model_names_for_local_analysis):
            for j, model_j_name in enumerate(model_names_for_local_analysis):
                if i == j:
                    p_value_matrix_local.loc[model_i_name, model_j_name] = 1.0
                    continue
                scores_i = np.array(scores_per_model_dataset.get(model_i_name, []))
                scores_j = np.array(scores_per_model_dataset.get(model_j_name, []))
                valid_mask = ~np.isnan(scores_i) & ~np.isnan(scores_j)
                scores_i_valid = scores_i[valid_mask]
                scores_j_valid = scores_j[valid_mask]
                if len(scores_i_valid) < 2: # Necessari almeno 2 punti per var_diff > 0
                    p_value_matrix_local.loc[model_i_name, model_j_name] = np.nan
                    continue
                p_value = corrected_resampled_ttest(scores_i_valid, scores_j_valid, n_train_for_ttest, n_test_for_ttest)
                p_value_matrix_local.loc[model_i_name, model_j_name] = p_value
        
        p_table_local_display = []
        for m_i in model_names_for_local_analysis:
            row = [m_i]
            for m_j in model_names_for_local_analysis:
                pval = p_value_matrix_local.loc[m_i, m_j]
                cell_str = "---" # Per i == j
                if m_i != m_j:
                    cell_str = f"{pval:.4f}" if not pd.isna(pval) else "N/A"
                    if not pd.isna(pval) and pval < 0.05:
                        cell_str += "**"
                row.append(cell_str)
            p_table_local_display.append(row)

        print(tabulate(p_table_local_display, headers=["Modello ↓ vs →"] + model_names_for_local_analysis, tablefmt="grid"))
        print("  ** indica p < 0.05 (differenza significativa)")
    elif not can_run_local_ttest:
         print("Impossibile eseguire test t locali a causa di n_train/n_test non validi.")
    else:
        print("Non abbastanza modelli (o dati validi) per eseguire test t locali.")

    processed_datasets_count += 1


# --- ANALISI GLOBALE ---
if processed_datasets_count > 0:
    print("\n" + "="*70 + "\n ANALISI GLOBALE SUI RISULTATI MEDI PER DATASET (Nested CV)\n" + "="*70)
    
    # ALL_POTENTIAL_MODEL_NAMES ora dovrebbe essere l'unione di tutti i model_names da current_param_grids
    # o semplicemente usare le chiavi di all_repeated_cv_means_global che hanno dati.
    all_model_names_from_runs = list(all_repeated_cv_means_global.keys())
    
    global_summary_data = []
    active_model_names_global = [name for name in all_model_names_from_runs if any(not pd.isna(m) for m in all_repeated_cv_means_global.get(name,[]))]

    model_names_sorted_global = sorted(
        active_model_names_global,
        key=lambda mn: np.nanmean(all_repeated_cv_means_global.get(mn, [np.nan])),
        reverse=True
    )

    for model_name in model_names_sorted_global:
        means_list_for_model = all_repeated_cv_means_global.get(model_name, [])
        valid_means_for_model = np.array([m for m in means_list_for_model if not pd.isna(m)])
        num_valid_datasets = len(valid_means_for_model)
        if num_valid_datasets > 0:
            mean_global_acc = np.mean(valid_means_for_model)
            std_dev_global_acc = np.std(valid_means_for_model, ddof=1) if num_valid_datasets > 1 else np.nan
            ci_lower_g, ci_upper_g = calculate_confidence_interval_t(valid_means_for_model, confidence=CONFIDENCE_LEVEL_CI)
        else:
            mean_global_acc, std_dev_global_acc, ci_lower_g, ci_upper_g = np.nan, np.nan, np.nan, np.nan
        global_summary_data.append([
            model_name,
            f"{mean_global_acc:.4f}" if not pd.isna(mean_global_acc) else "N/A",
            f"[{ci_lower_g:.4f}, {ci_upper_g:.4f}]" if not pd.isna(ci_lower_g) else "N/A",
            f"{std_dev_global_acc:.4f}" if not pd.isna(std_dev_global_acc) else "N/A",
            num_valid_datasets
        ])
    headers_global_summary = ["Modello", "Mean Global Acc", f"{int(CONFIDENCE_LEVEL_CI*100)}% CI", "Std Dev (s)", "Num Datasets Validi"]
    print(tabulate(global_summary_data, headers=headers_global_summary, tablefmt="grid"))

    df_global_comparison_input = pd.DataFrame(all_repeated_cv_means_global)
    df_global_comparison_input = df_global_comparison_input[active_model_names_global] # Considera solo modelli attivi
    df_global_comparison_input.dropna(axis=0, how='all', inplace=True) # Rimuovi dataset dove nessun modello ha score
    df_global_comparison_input.dropna(axis=1, how='all', inplace=True) # Rimuovi modelli che non hanno score su nessun dataset (dovrebbe essere già gestito da active_model_names_global)
    
    df_nemenyi_data_complete_cases = df_global_comparison_input.dropna(axis=0, how='any') # Casi completi per Nemenyi

    if df_nemenyi_data_complete_cases.shape[0] >= 2 and df_nemenyi_data_complete_cases.shape[1] >= 2:
        print("\n--- Confronto Globale con Nemenyi Post-Hoc (su dataset con punteggi per tutti i modelli attivi) ---")
        print(f"Numero di dataset considerati per Nemenyi (casi completi): {df_nemenyi_data_complete_cases.shape[0]}")
        print(f"Modelli considerati: {list(df_nemenyi_data_complete_cases.columns)}")
        try:
            print("\nNemenyi Post-Hoc (p-value):")
            nemenyi_result = sp.posthoc_nemenyi_friedman(df_nemenyi_data_complete_cases)
            print(nemenyi_result)
        except Exception as e_nemenyi:
            print(f"Errore durante il test di Nemenyi: {e_nemenyi}")
    else:
        print(f"\nNon abbastanza dati ({df_nemenyi_data_complete_cases.shape[0]} dataset completi, {df_nemenyi_data_complete_cases.shape[1]} modelli) per il test di Nemenyi.")

    data_to_plot_global_boxplot = []
    labels_global_boxplot = []
    for model_name_bp in model_names_sorted_global: # Usa gli stessi modelli ordinati della tabella globale
        valid_means_bp = np.array([m for m in all_repeated_cv_means_global.get(model_name_bp, []) if not pd.isna(m)])
        if len(valid_means_bp) > 0:
            data_to_plot_global_boxplot.append(valid_means_bp)
            labels_global_boxplot.append(f"{model_name_bp} (n={len(valid_means_bp)})")

    if data_to_plot_global_boxplot:
        plt.figure(figsize=(max(10, len(labels_global_boxplot) * 1.5), 7))
        sns.boxplot(data=data_to_plot_global_boxplot)
        plt.xticks(range(len(labels_global_boxplot)), labels_global_boxplot, rotation=45, ha='right')
        plt.title('Distribuzione delle Accuratezze Medie dei Modelli (Nested CV) sui Diversi Dataset')
        plt.ylabel('Mean Accuracy per Dataset (Outer Folds)')
        plt.tight_layout()
        try:
            plt.savefig('boxplot_global_nested_cv_means.png')
            print("\nBoxplot globale salvato come 'boxplot_global_nested_cv_means.png'")
        except Exception as e_save:
            print(f"Errore nel salvataggio del boxplot: {e_save}")
        plt.close()
    else:
        print("\nNessun dato valido per generare il boxplot globale.")
else:
    print("\nNessun dataset è stato processato con successo. Analisi globale e visualizzazione saltate.")

print("\nEsecuzione completata.")