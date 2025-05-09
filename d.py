from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KN
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as t_dist
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

NUM_TRIALS_WEKA = 10
N_SPLITS_WEKA_CV = 10
CONFIDENCE_LEVEL_CI = 0.95 # Usato per CI locali e globali



# --- FUNZIONI STATISTICHE ---
def corrected_resampled_ttest(scores1, scores2, n_train, n_test):
    """Calcola il Corrected Resampled t-test tra due modelli."""
    differences = np.array(scores1) - np.array(scores2)
    k = len(differences)

    if k < 2:
        return 1.0

    mean_diff = np.mean(differences)
    var_diff = np.var(differences, ddof=1)

    if var_diff < 1e-10:
        return 1.0 if np.isclose(mean_diff, 0) else 0.0

    if n_train <= 0:
        print("AVVISO: n_train <= 0 in corrected_resampled_ttest. Impossibile calcolare il test.")
        return 1.0

    variance_correction_factor = (1/k + n_test/n_train)
    if variance_correction_factor <= 0:
        print("AVVISO: Fattore di correzione non positivo in corrected_resampled_ttest.")
        return 1.0

    denominator_corrected = np.sqrt(variance_correction_factor * var_diff)

    if denominator_corrected < 1e-10:
        return 1.0 if np.isclose(mean_diff, 0) else 0.0

    t_stat = mean_diff / denominator_corrected
    df_degrees_of_freedom = k - 1
    p_val = t_dist.sf(np.abs(t_stat), df_degrees_of_freedom) * 2
    return p_val

def calculate_confidence_interval_t(data, confidence=0.95):
    """Calcola l'Intervallo di Confidenza basato sulla distribuzione t."""
    a = np.array(data)[~np.isnan(data)]
    n = len(a)
    if n < 2:
        return (np.nan, np.nan)
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

ALL_POTENTIAL_MODEL_NAMES = [model_config['name'] for model_config in get_param_grids(X_train_len_ds=100)]

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
        raise ValueError(f"Target column '{target_col_name}' non trovata nel dataset dopo il drop.")

    X = df_processed.drop(columns=[target_col_name])
    y = df_processed[target_col_name]

    if y.dtype == 'object':
        y = pd.factorize(y)[0]
        y = pd.Series(y, index=df_processed.index)

    if not X.empty:
        X_numeric = X.select_dtypes(include=np.number)
        X_numeric = X_numeric.dropna()
        X = X.loc[X_numeric.index]
        y = y.loc[X_numeric.index]

    if X.empty and not df_processed.empty :
        raise ValueError(f"Nessun dato rimasto in X dopo il preprocessing per {config['name']}.")

    return X, y

# --- WEKA-STYLE REPEATED CROSS-VALIDATION ---
def repeated_cross_validation(X, y, model, n_trials=NUM_TRIALS_WEKA, n_splits=N_SPLITS_WEKA_CV, random_state_base=42):
    scores = []
    from sklearn.base import clone # Importazione locale di clone se necessario qui
    for trial in range(n_trials):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state_base + trial)
        for train_idx, test_idx in skf.split(X, y):
            current_model = clone(model) # Clona per ogni fold/trial
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            try:
                current_model.fit(X_train, y_train)
                scores.append(current_model.score(X_test, y_test))
            except Exception as e:
                print(f"Errore durante fit/score nel CV: {e}. Aggiungo NaN.")
                scores.append(np.nan)
    return scores

# --- CICLO SUI DATASET ---
all_repeated_cv_means_global = {name: [] for name in ALL_POTENTIAL_MODEL_NAMES}
processed_datasets_count = 0
TOTAL_EXPECTED_SCORES_PER_MODEL_DS = NUM_TRIALS_WEKA * N_SPLITS_WEKA_CV

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
        min_samples_for_cv = N_SPLITS_WEKA_CV
        if len(X) < min_samples_for_cv or len(y.unique()) < 2 :
             print(f"Dataset {dataset_name} non ha abbastanza campioni o classi ({len(X)} campioni, {len(y.unique())} classi) per CV con {N_SPLITS_WEKA_CV} splits. Salto.")
             continue
    except ValueError as e:
        print(f"ERRORE nel preprocessing di {dataset_name}: {e}. Salto dataset.")
        continue
    except Exception as e:
        print(f"ERRORE sconosciuto nel preprocessing di {dataset_name}: {e}. Salto dataset.")
        continue


    param_grids_for_dataset = get_param_grids(len(X))
    optimized_estimators = {}

    try:
        X_train_tune, _, y_train_tune, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=dataset_idx
        )
    except ValueError as e:
        print(f"Errore nello split per tuning su {dataset_name} (possibile problema di stratificazione): {e}. Tento split non stratificato.")
        try:
            X_train_tune, _, y_train_tune, _ = train_test_split(
                X, y, test_size=0.2, random_state=dataset_idx
            )
        except Exception as e_nostrat:
            print(f"Errore anche nello split non stratificato per tuning su {dataset_name}: {e_nostrat}. Salto ottimizzazione parametri per questo dataset.")
            continue


    print(f"Inizio ottimizzazione parametri per {dataset_name} (X_train_tune shape: {X_train_tune.shape})...")
    for model_config in param_grids_for_dataset:
        estimator_name = model_config['name']
        estimator = model_config['estimator']
        param_grid = model_config['params']

        if not CUSTOM_CLASSES_LOADED and estimator_name != 'KNeighborsClassifier':
            print(f"Salto {estimator_name} perché le classi custom non sono caricate.")
            continue

        print(f"  Ottimizzazione per {estimator_name}...")
        try:
            cv_gridsearch = StratifiedKFold(n_splits=5, shuffle=True, random_state=dataset_idx)
            gs = GridSearchCV(estimator, param_grid, cv=cv_gridsearch, scoring='accuracy', error_score=np.nan, n_jobs=-1)
            gs.fit(X_train_tune, y_train_tune)
            if gs.best_estimator_ is not None:
                 optimized_estimators[estimator_name] = gs.best_estimator_
                 print(f"    Miglior punteggio per {estimator_name}: {gs.best_score_:.4f} con parametri: {gs.best_params_}")
            else:
                 print(f"    ATTENZIONE: GridSearchCV non ha trovato un best_estimator_ per {estimator_name} (tutti i fit falliti?).")
        except Exception as e:
            print(f"    ERRORE durante GridSearchCV per {estimator_name}: {e}")
            optimized_estimators[estimator_name] = None

    scores_per_model_dataset = {}
    print(f"\nInizio Repeated Cross-Validation per {dataset_name} (X shape: {X.shape})...")
    for model_name, est in optimized_estimators.items():
        if est is None:
            print(f"  Salto Repeated CV per {model_name} perché l'ottimizzazione è fallita.")
            all_repeated_cv_means_global[model_name].append(np.nan)
            scores_per_model_dataset[model_name] = [np.nan] * TOTAL_EXPECTED_SCORES_PER_MODEL_DS
            continue
        print(f"  Eseguo Repeated CV per {model_name}...")
        current_scores = repeated_cross_validation(X, y, est, n_trials=NUM_TRIALS_WEKA, n_splits=N_SPLITS_WEKA_CV, random_state_base=dataset_idx*100)
        scores_per_model_dataset[model_name] = current_scores
        
        valid_scores_current_ds = np.array(current_scores)[~np.isnan(current_scores)]
        if len(valid_scores_current_ds) > 0:
            all_repeated_cv_means_global[model_name].append(np.mean(valid_scores_current_ds))
        else:
            all_repeated_cv_means_global[model_name].append(np.nan)

    if not scores_per_model_dataset:
        print(f"Nessun modello valutato con successo per {dataset_name}. Salto summary e test locali.")
        processed_datasets_count +=1
        continue


    summary_local = []
    for model_name, scores_list in scores_per_model_dataset.items():
        valid_scores = np.array(scores_list)[~np.isnan(scores_list)]
        n_valid = len(valid_scores)
        mean_acc = np.mean(valid_scores) if n_valid > 0 else np.nan
        std_dev = np.std(valid_scores, ddof=1) if n_valid > 1 else np.nan

        ci_lower, ci_upper = calculate_confidence_interval_t(valid_scores, confidence=CONFIDENCE_LEVEL_CI)

        summary_local.append({
            "Modello": model_name,
            "Mean Acc": mean_acc,
            "Std Dev": std_dev,
            "CI Lower": ci_lower,
            "CI Upper": ci_upper,
            "Valid Scores Num": n_valid
        })
    summary_local = sorted(summary_local, key=lambda x: x["Mean Acc"] if not pd.isna(x["Mean Acc"]) else -1, reverse=True)


    table_data_local = [[rank+1, item["Modello"],
                         f"{item['Mean Acc']:.4f}" if not pd.isna(item['Mean Acc']) else "N/A",
                         f"[{item['CI Lower']:.4f}, {item['CI Upper']:.4f}]" if not pd.isna(item['CI Lower']) else "N/A",
                         f"{item['Std Dev']:.4f}" if not pd.isna(item['Std Dev']) else "N/A",
                         f"{item['Valid Scores Num']}/{TOTAL_EXPECTED_SCORES_PER_MODEL_DS}"]
                        for rank, item in enumerate(summary_local)]
    headers_local = ["Rank", "Modello", "Mean Acc", f"{int(CONFIDENCE_LEVEL_CI*100)}% CI (t-dist)", "Std Dev (s)", "Valid Scores"]
    print(f"\nRisultati per {dataset_name}:")
    print(tabulate(table_data_local, headers=headers_local, tablefmt="grid"))

    print("\n--- Test Statistici LOCALI (Corrected Resampled t-test) ---")
    n_total_samples_dataset = len(X)
    if N_SPLITS_WEKA_CV <= 0:
        print("ATTENZIONE: N_SPLITS_WEKA_CV non valido. Impossibile calcolare n_train/n_test per t-test.")
        can_run_local_ttest = False
    else:
        n_test_for_ttest = n_total_samples_dataset / N_SPLITS_WEKA_CV
        n_train_for_ttest = n_total_samples_dataset - n_test_for_ttest
        can_run_local_ttest = n_train_for_ttest > 0 and n_test_for_ttest > 0

    model_names_for_local_tests = list(scores_per_model_dataset.keys())
    if len(model_names_for_local_tests) >=2 and can_run_local_ttest:
        p_value_matrix_local = pd.DataFrame(index=model_names_for_local_tests, columns=model_names_for_local_tests, dtype=float)

        for i, model_i_name in enumerate(model_names_for_local_tests):
            for j, model_j_name in enumerate(model_names_for_local_tests):
                if i == j:
                    p_value_matrix_local.loc[model_i_name, model_j_name] = 1.0
                    continue

                scores_i = np.array(scores_per_model_dataset[model_i_name])
                scores_j = np.array(scores_per_model_dataset[model_j_name])

                valid_mask = ~np.isnan(scores_i) & ~np.isnan(scores_j)
                scores_i_valid = scores_i[valid_mask]
                scores_j_valid = scores_j[valid_mask]

                if len(scores_i_valid) < 2:
                    p_value_matrix_local.loc[model_i_name, model_j_name] = np.nan
                    continue

                p_value = corrected_resampled_ttest(scores_i_valid, scores_j_valid, n_train_for_ttest, n_test_for_ttest)
                p_value_matrix_local.loc[model_i_name, model_j_name] = p_value

        p_table_local = [[m_i] + [f"{p_value_matrix_local.loc[m_i, m_j]:.4f}" +
                                  ("**" if not pd.isna(p_value_matrix_local.loc[m_i, m_j]) and m_i != m_j and p_value_matrix_local.loc[m_i, m_j] < 0.05 else "")
                                  for m_j in model_names_for_local_tests]
                         for m_i in model_names_for_local_tests]
        print(tabulate(p_table_local, headers=["Modello ↓ vs →"] + model_names_for_local_tests, tablefmt="grid"))
        print("  ** indica p < 0.05 (differenza significativa)")
    elif not can_run_local_ttest:
         print("Impossibile eseguire test t locali a causa di n_train/n_test non validi.")
    else:
        print("Non abbastanza modelli (o dati) per eseguire test t locali.")

    processed_datasets_count += 1

# --- ANALISI GLOBALE ---
if processed_datasets_count > 0:
    print("\n" + "="*70 + "\n ANALISI GLOBALE SUI RISULTATI MEDI PER DATASET\n" + "="*70)
    global_summary_data = []

    active_model_names = [name for name, means in all_repeated_cv_means_global.items() if any(not pd.isna(m) for m in means)]

    model_names_sorted_global = sorted(
        active_model_names,
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
            mean_global_acc = np.nan
            std_dev_global_acc = np.nan
            ci_lower_g, ci_upper_g = (np.nan, np.nan)

        global_summary_data.append([
            model_name,
            f"{mean_global_acc:.4f}" if not pd.isna(mean_global_acc) else "N/A",
            f"[{ci_lower_g:.4f}, {ci_upper_g:.4f}]" if not pd.isna(ci_lower_g) else "N/A",
            f"{std_dev_global_acc:.4f}" if not pd.isna(std_dev_global_acc) else "N/A",
            num_valid_datasets
        ])

    headers_global_summary = ["Modello", "Mean Global Acc (across datasets)", f"{int(CONFIDENCE_LEVEL_CI*100)}% CI", "Std Dev (s)", "Num Datasets Validi"]
    print(tabulate(global_summary_data, headers=headers_global_summary, tablefmt="grid"))

    # --- NEMENYI POST-HOC (sulle medie per dataset) ---
    df_global_comparison_input = pd.DataFrame(all_repeated_cv_means_global)
    df_global_comparison_input = df_global_comparison_input[active_model_names]

    df_global_comparison_input.dropna(axis=0, how='all', inplace=True)
    df_global_comparison_input.dropna(axis=1, how='all', inplace=True)

    df_nemenyi_data_complete_cases = df_global_comparison_input.dropna(axis=0, how='any')

    if df_nemenyi_data_complete_cases.shape[0] >= 2 and df_nemenyi_data_complete_cases.shape[1] >= 2:
        print("\n--- Confronto Globale con Nemenyi Post-Hoc (su dataset con punteggi per tutti i modelli selezionati) ---")
        print(f"Numero di dataset considerati per Nemenyi (casi completi): {df_nemenyi_data_complete_cases.shape[0]}")
        print(f"Modelli considerati: {list(df_nemenyi_data_complete_cases.columns)}")

        if df_nemenyi_data_complete_cases.shape[1] < 2:
             print("Non abbastanza modelli dopo la pulizia dei NaN per il test di Nemenyi.")
        else:
            try:
                print("\nNemenyi Post-Hoc (p-value):")
                # scikit-posthocs.posthoc_nemenyi_friedman può prendere direttamente il DataFrame (campioni x gruppi)
                # dove i campioni sono i dataset e i gruppi sono i modelli.
                nemenyi_result = sp.posthoc_nemenyi_friedman(df_nemenyi_data_complete_cases)
                print(nemenyi_result)
            except Exception as e_nemenyi:
                print(f"Errore durante il test di Nemenyi: {e_nemenyi}")
    else:
        print("\nNon abbastanza dati (dataset/modelli con punteggi completi) per il test di Nemenyi.")

    # --- VISUALIZZAZIONE ---
    data_to_plot_global_boxplot = []
    labels_global_boxplot = []
    for model_name_bp in model_names_sorted_global:
        valid_means_bp = np.array([m for m in all_repeated_cv_means_global.get(model_name_bp, []) if not pd.isna(m)])
        if len(valid_means_bp) > 0:
            data_to_plot_global_boxplot.append(valid_means_bp)
            labels_global_boxplot.append(f"{model_name_bp} (n={len(valid_means_bp)})")

    if data_to_plot_global_boxplot:
        plt.figure(figsize=(max(10, len(labels_global_boxplot) * 1.5), 7))
        sns.boxplot(data=data_to_plot_global_boxplot)
        plt.xticks(range(len(labels_global_boxplot)), labels_global_boxplot, rotation=45, ha='right')
        plt.title('Distribuzione delle Accuratezze Medie dei Modelli sui Diversi Dataset')
        plt.ylabel('Mean Accuracy per Dataset')
        plt.tight_layout()
        try:
            plt.savefig('boxplot_global_repeatedcv_means.png')
            print("\nBoxplot globale salvato come 'boxplot_global_repeatedcv_means.png'")
        except Exception as e_save:
            print(f"Errore nel salvataggio del boxplot: {e_save}")
        plt.close()
    else:
        print("\nNessun dato valido per generare il boxplot globale.")

else:
    print("\nNessun dataset è stato processato con successo. Analisi globale e visualizzazione saltate.")

# --- FINE ---
print("\nEsecuzione completata.")