# -*- coding: utf-8 -*-
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
try:
    # Assicurati che Custom.py sia nella stessa directory o nel PYTHONPATH
    from Custom import StabilityAdaptiveKNN, ConfidenceAdaptiveKNN, DensityConfidenceKNN
    CUSTOM_CLASSES_LOADED = True
    print("Classi personalizzate da Custom.py caricate con successo.")
except ImportError as e:
    print(f"ATTENZIONE: Impossibile importare classi da Custom.py: {e}")
    print(">>> Verranno usate classi fittizie. Solo KNeighborsClassifier sarà testato. <<<")
    CUSTOM_CLASSES_LOADED = False
    # Definiamo classi fittizie per permettere allo script di girare almeno con KNeighborsClassifier
    class StabilityAdaptiveKNN: pass
    class ConfidenceAdaptiveKNN: pass
    class DensityConfidenceKNN: pass
    # KN viene importato successivamente

from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
from scipy.stats import ttest_rel, t as t_dist, norm as normal_dist
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil

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

# Parametri per le analisi statistiche
NUM_TRIALS_CV_REPEATED = 3 # Numero di ripetizioni della CV per i test statistici per-dataset
N_SPLITS_CV_REPEATED = 5   # Numero di fold per la CV ripetuta
CONFIDENCE_LEVEL_CI = 0.95 # Livello di confidenza per gli intervalli

# --- FUNZIONE PER GRIGLIE DI PARAMETRI DINAMICHE ---
def get_param_grids(X_train_len_ds):
    """Genera le griglie di parametri dinamicamente."""
    if X_train_len_ds <= 0: X_train_len_ds = 1 # Evita errori

    # Calcolo per max_k_adaptive
    upper_bound_adaptive_k = int(np.floor(np.sqrt(X_train_len_ds)))
    if upper_bound_adaptive_k < 1: upper_bound_adaptive_k = 1
    adaptive_k_values = [i for i in range(1, upper_bound_adaptive_k + 1, 10)]
    if not adaptive_k_values: adaptive_k_values = [min(1, X_train_len_ds if X_train_len_ds > 0 else 1)]

    # Per KNeighborsClassifier
    upper_bound_knn = int(np.floor(np.sqrt(X_train_len_ds)))
    if upper_bound_knn < 1: upper_bound_knn = 1
    n_neighbors_knn = [i for i in range(1, min(20, upper_bound_knn) + 1)]
    if not n_neighbors_knn: n_neighbors_knn = [min(1, X_train_len_ds if X_train_len_ds > 0 else 1)]

    grids = []
    if CUSTOM_CLASSES_LOADED:
        grids.extend([
            {'name': 'StabilityAdaptiveKNN', 'estimator': StabilityAdaptiveKNN(),
             'params': {'stability_patience': [i for i in range(1, 4)], 'k_step_stability': [i for i in range(1, 4)],
                        'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}},
            {'name': 'ConfidenceAdaptiveKNN', 'estimator': ConfidenceAdaptiveKNN(),
             'params': {'confidence_threshold': [0.6, 0.8], 'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}},
            {'name': 'DensityConfidenceKNN', 'estimator': DensityConfidenceKNN(),
             'params': {'confidence_threshold': [0.6, 0.8], 'density_quantile': [0.7, 0.85],
                        'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}},
        ])
    # Aggiungi sempre KNeighborsClassifier
    grids.append(
        {'name': 'KNeighborsClassifier', 'estimator': KN(),
         'params': {'n_neighbors': n_neighbors_knn, 'weights': ['uniform', 'distance'], 'algorithm': ['auto'], 'metric': ['euclidean', 'manhattan']}}
    )

    if not grids:
        print("ERRORE CRITICO: Nessuna griglia di parametri valida definita. Uscita.")
        exit()
    return grids

# Ottieni i nomi di tutti i modelli che verranno testati
# Usiamo una lunghezza fittizia solo per ottenere tutti i nomi possibili
ALL_POTENTIAL_MODEL_NAMES = [model_config['name'] for model_config in get_param_grids(X_train_len_ds=100)]

# --- PREPARAZIONE FILE DI ESEMPIO ---
source_parkinson = "Parkinsson disease.csv"
if not os.path.exists(source_parkinson):
    print(f"ERRORE: File sorgente '{source_parkinson}' per gli esempi non trovato. Lo script potrebbe fallire se i dataset non esistono.")

diabetes_example_path = "diabetes_dataset.csv"
if not os.path.exists(diabetes_example_path) and os.path.exists(source_parkinson):
    print(f"ATTENZIONE: '{diabetes_example_path}' non trovato. Creo un esempio da '{source_parkinson}'.")
    try:
        shutil.copy(source_parkinson, diabetes_example_path)
        df_temp = pd.read_csv(diabetes_example_path);
        if 'status' in df_temp.columns: df_temp.rename(columns={'status': 'Outcome'}, inplace=True)
        df_temp.to_csv(diabetes_example_path, index=False)
        print(f"  Esempio '{diabetes_example_path}' creato.")
    except Exception as e: print(f"  Errore modifica esempio {diabetes_example_path}: {e}")

marketing_example_path = "marketing_campaign.csv"
if not os.path.exists(marketing_example_path) and os.path.exists(source_parkinson):
    print(f"ATTENZIONE: '{marketing_example_path}' non trovato. Creo un esempio da '{source_parkinson}'.")
    try:
        shutil.copy(source_parkinson, marketing_example_path); df_temp = pd.read_csv(marketing_example_path)
        num_rows = len(df_temp)
        df_temp.rename(columns={'status': 'Response', 'name': 'ID'}, inplace=True, errors='ignore')
        if 'Response' not in df_temp.columns: df_temp['Response'] = np.random.randint(0,2, size=num_rows)
        if 'ID' not in df_temp.columns: df_temp['ID'] = range(num_rows)
        df_temp['Education'] = np.random.choice(['Graduation', 'PhD', 'Master', '2n Cycle', 'Basic'], size=num_rows)
        df_temp['Marital_Status'] = np.random.choice(['Single', 'Married', 'Divorced', 'Together', 'Widow'], size=num_rows)
        start_date = pd.to_datetime('2012-01-01'); df_temp['Dt_Customer'] = [(start_date + pd.Timedelta(days=np.random.randint(0, 730))).strftime('%d-%m-%Y') for _ in range(num_rows)]
        numeric_cols = df_temp.select_dtypes(include=np.number).columns.tolist()
        if 'Response' in numeric_cols: numeric_cols.remove('Response')
        if 'ID' in numeric_cols: numeric_cols.remove('ID')
        if len(numeric_cols) > 0:
            col_to_nan = np.random.choice(numeric_cols); sample_size = max(1, int(0.1 * num_rows))
            if num_rows >= sample_size : nan_indices = np.random.choice(df_temp.index, size=sample_size, replace=False); df_temp.loc[nan_indices, col_to_nan] = np.nan
        else: df_temp['Numeric_Example_For_NaN'] = np.random.rand(num_rows); nan_indices = np.random.choice(df_temp.index, size=max(1, int(0.1 * num_rows)), replace=False); df_temp.loc[nan_indices, 'Numeric_Example_For_NaN'] = np.nan
        df_temp.to_csv(marketing_example_path, sep='\t', index=False); print(f"  Esempio '{marketing_example_path}' creato/aggiornato (TSV).")
    except Exception as e: print(f"  Errore creazione/modifica esempio {marketing_example_path}: {e}")


# Strutture dati per i risultati GLOBALI
all_test_accuracies_per_model_global = {name: [] for name in ALL_POTENTIAL_MODEL_NAMES}

# --- CICLO SUI DATASET ---
processed_datasets_count = 0
for dataset_idx, config in enumerate(DATASET_CONFIGS):
    # Estrai configurazione
    dataset_name = config["name"]; dataset_file = config["file_path"]; target_col = config["target_column"]
    cols_to_drop_config = config["columns_to_drop"]; separator = config["separator"]; categorical_cols_config = config["categorical_cols_to_factorize"]
    
    print(f"\n\n{'='*50}\n ELABORAZIONE DATASET: {dataset_name} ({dataset_file}) ({dataset_idx+1}/{len(DATASET_CONFIGS)})\n{'='*50}")

    # Dizionari per risultati specifici di questo dataset
    results_per_dataset = {} # Info principali (Acc, CI, Tempo, Param)
    cv_scores_all_models_per_dataset = {} # Score CV ripetute per t-test interni
    optimized_model_objects_per_dataset = {} # Oggetti modello ottimizzati

    # --- CARICAMENTO DATI ---
    if not os.path.exists(dataset_file):
        print(f"ERRORE: File {dataset_file} non trovato. Salto dataset.");
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES: all_test_accuracies_per_model_global[mn_loop].append(np.nan)
        continue
    try:
        df = pd.read_csv(dataset_file, sep=separator)
        print(f"  Dataset '{dataset_name}' caricato. Shape: {df.shape}")
    except Exception as e:
        print(f"ERRORE caricamento {dataset_file}: {e}. Salto dataset.");
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES: all_test_accuracies_per_model_global[mn_loop].append(np.nan)
        continue

    # --- PREPROCESSING ---
    print("  Avvio Preprocessing...")
    # Gestione Dt_Customer
    if dataset_name == "Marketing Campaign" and 'Dt_Customer' in df.columns:
        try:
            df['Dt_Customer_Parsed'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True, errors='coerce')
            valid_dates_mask = df['Dt_Customer_Parsed'].notna()
            if not valid_dates_mask.all(): print(f"    ATTENZIONE: 'Dt_Customer' contiene {sum(~valid_dates_mask)} date non valide (NaT).")
            if valid_dates_mask.any(): # Prosegui solo se ci sono date valide
                df['Year_Customer'] = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].dt.year
                df['Month_Customer'] = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].dt.month
                latest_date_in_data = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].max()
                df['Customer_Age_Days'] = (latest_date_in_data - df.loc[valid_dates_mask, 'Dt_Customer_Parsed']).dt.days
                # Riempi NaN nelle nuove colonne (per le date non valide) se necessario, es. con la media o mediana
                for col_fe in ['Year_Customer', 'Month_Customer', 'Customer_Age_Days']:
                    if df[col_fe].isnull().any():
                        fill_val = df[col_fe].median() # O media, o valore specifico
                        df[col_fe].fillna(fill_val, inplace=True)
                        print(f"    NaN in '{col_fe}' riempiti con mediana ({fill_val}).")
            df = df.drop(columns=['Dt_Customer', 'Dt_Customer_Parsed'], errors='ignore')
            print("    FE su 'Dt_Customer' completata.")
        except Exception as e_date:
            print(f"    ATTENZIONE: Errore FE 'Dt_Customer' ({e_date}). Droppo colonna."); df = df.drop(columns=['Dt_Customer'], errors='ignore')

    # Drop colonne
    df = df.drop(columns=cols_to_drop_config, errors='ignore')

    # Factorize categoriche
    for col_loop in categorical_cols_config:
        if col_loop in df.columns and (df[col_loop].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col_loop])):
            df[col_loop] = pd.factorize(df[col_loop])[0]; print(f"    Colonna '{col_loop}' factorizzata (specificata).")
    for col_loop in df.select_dtypes(include='object').columns:
        if col_loop != target_col and col_loop not in categorical_cols_config:
            df[col_loop] = pd.factorize(df[col_loop])[0]; print(f"    Colonna '{col_loop}' (object) factorizzata.")

    # Target check
    if target_col not in df.columns:
        print(f"ERRORE: Target '{target_col}' non in {dataset_file}. Salto dataset.");
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES: all_test_accuracies_per_model_global[mn_loop].append(np.nan)
        continue
        
    # Definizione X e y
    X_processed_dataset = df.drop(columns=[target_col]); y_dataset = df[target_col]
    if y_dataset.dtype == 'object': y_dataset = pd.factorize(y_dataset)[0]; print(f"    Target '{target_col}' factorizzato.")

    # Imputazione NaN
    numeric_cols_in_X = X_processed_dataset.select_dtypes(include=np.number).columns
    if not X_processed_dataset.empty and len(numeric_cols_in_X) > 0:
        # Tenta conversione robusta
        for col_num_check in numeric_cols_in_X[:]: # Itera su una copia
             if not pd.api.types.is_numeric_dtype(X_processed_dataset[col_num_check]):
                print(f"    Tentativo conversione colonna '{col_num_check}' a numerico...")
                try: X_processed_dataset[col_num_check] = pd.to_numeric(X_processed_dataset[col_num_check], errors='coerce')
                except Exception as e_conv: print(f"      Conversione fallita: {e_conv}")
        # Imputa solo sulle colonne effettivamente numeriche dopo la conversione
        numeric_cols_in_X_after_conv = X_processed_dataset.select_dtypes(include=np.number).columns
        if not numeric_cols_in_X_after_conv.empty:
            print(f"    Imputazione NaN su {len(numeric_cols_in_X_after_conv)} colonne numeriche.")
            imputer = SimpleImputer(strategy='mean')
            X_processed_dataset[numeric_cols_in_X_after_conv] = imputer.fit_transform(X_processed_dataset[numeric_cols_in_X_after_conv])
            if X_processed_dataset[numeric_cols_in_X_after_conv].isnull().sum().sum() > 0: print(f"    ATTENZIONE: NaN ancora in X DOPO imputazione!")
        else: print("    Nessuna colonna numerica trovata per l'imputazione.")
    elif X_processed_dataset.empty:
        print(f"ATTENZIONE: X vuoto prima dell'imputazione. Salto dataset.");
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES: all_test_accuracies_per_model_global[mn_loop].append(np.nan)
        continue

    # Controllo finale non numeriche
    final_non_numeric_cols = X_processed_dataset.select_dtypes(exclude=np.number).columns
    if not final_non_numeric_cols.empty:
        print(f"ERRORE CRITICO: Colonne non numeriche rimaste in X: {list(final_non_numeric_cols)}. Salto dataset.");
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES: all_test_accuracies_per_model_global[mn_loop].append(np.nan)
        continue

    # Controllo dati sufficienti per CV
    min_samples_cv_fold = 5
    if len(X_processed_dataset) < min_samples_cv_fold * 2 or y_dataset.value_counts().min() < min_samples_cv_fold :
        print(f"ERRORE: Dati insuff. per CV ({len(X_processed_dataset)} campioni, min classe {y_dataset.value_counts().min()}). Salto dataset.");
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES: all_test_accuracies_per_model_global[mn_loop].append(np.nan)
        continue
    print("  Preprocessing completato.")

    # --- TRAIN-TEST SPLIT ---
    try:
        n_splits_tt_cv = min(min_samples_cv_fold, y_dataset.value_counts().min());
        if n_splits_tt_cv < 2: n_splits_tt_cv = 2
        X_train_ds, X_test_ds, y_train_ds, y_test_ds = train_test_split(X_processed_dataset, y_dataset, test_size=0.2, random_state=42, stratify=y_dataset)
        print(f"  Train/Test split eseguito. Train shape: {X_train_ds.shape}, Test shape: {X_test_ds.shape}")
    except ValueError as e:
        print(f"ERRORE train_test_split: {e}. Salto dataset.");
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES: all_test_accuracies_per_model_global[mn_loop].append(np.nan)
        continue

    # --- CICLO SUI MODELLI PER QUESTO DATASET ---
    processed_datasets_count += 1
    current_param_grids = get_param_grids(len(X_train_ds))

    for model_config in current_param_grids:
        estimator_name = model_config['name']; estimator = model_config['estimator']; param_grid = model_config['params']
        print(f"\n--- GridSearchCV per {estimator_name} su {dataset_name} ---")
        start_time = time.time(); gs_best_score, test_acc, ci_low, ci_upp, elapsed_t, best_p, best_est = np.nan, np.nan, np.nan, np.nan, np.nan, {}, None
        try:
            current_n_splits_gs_cv = min(n_splits_tt_cv, y_train_ds.value_counts().min())
            if current_n_splits_gs_cv < 2: current_n_splits_gs_cv = 2
            # Param grid check (fallback già in get_param_grids)
            # if 'n_neighbors' in param_grid and not param_grid['n_neighbors']: param_grid['n_neighbors'] = [1]
            # if 'max_k_adaptive' in param_grid and not param_grid['max_k_adaptive']: param_grid['max_k_adaptive'] = [1]

            gs = GridSearchCV(estimator, param_grid, cv=StratifiedKFold(n_splits=current_n_splits_gs_cv, shuffle=True, random_state=dataset_idx),
                              scoring='accuracy', n_jobs=-1, verbose=0, error_score=np.nan)
            gs.fit(X_train_ds, y_train_ds)

            if not pd.isna(gs.best_score_):
                best_est = gs.best_estimator_; gs_best_score = gs.best_score_; best_p = gs.best_params_
                if len(X_test_ds) > 0 and len(y_test_ds) > 0:
                    y_pred_test_ds = best_est.predict(X_test_ds); test_acc = accuracy_score(y_test_ds, y_pred_test_ds)
                    print(f"    Test Acc: {test_acc:.4f}, CV (Train) Acc: {gs_best_score:.4f}")
                    N_test = len(y_test_ds)
                    if N_test > 0 and not pd.isna(test_acc):
                        alpha = 1 - CONFIDENCE_LEVEL_CI; z_crit_norm = normal_dist.ppf(1 - alpha/2)
                        term1_num = test_acc + (z_crit_norm**2) / (2 * N_test)
                        term_sqrt_content = (test_acc * (1 - test_acc) / N_test) + (z_crit_norm**2) / (4 * N_test**2)
                        if term_sqrt_content < 0: term_sqrt_content = 0
                        term_sqrt = np.sqrt(term_sqrt_content)
                        denominator = 1 + (z_crit_norm**2) / N_test
                        ci_low_wilson = (term1_num - z_crit_norm * term_sqrt) / denominator
                        ci_upp_wilson = (term1_num + z_crit_norm * term_sqrt) / denominator
                        ci_low = max(0, ci_low_wilson); ci_upp = min(1, ci_upp_wilson)
                        print(f"    {int(CONFIDENCE_LEVEL_CI*100)}% CI (Wilson): [{ci_low:.4f}, {ci_upp:.4f}]")
                    else: ci_low, ci_upp = np.nan, np.nan
                else: print("    Test set vuoto, impossibile calcolare Test Acc e CI."); test_acc, ci_low, ci_upp = np.nan, np.nan, np.nan
                optimized_model_objects_per_dataset[estimator_name] = best_est
            else: print(f"    GridSearchCV fallito."); ci_low, ci_upp = np.nan, np.nan
            elapsed_t = time.time() - start_time; print(f"    Tempo GridSearchCV: {elapsed_t:.2f}s")
        except Exception as e: print(f"ERRORE GridSearchCV {estimator_name}: {e}"); elapsed_t = time.time() - start_time

        results_per_dataset[estimator_name] = {'Best Parameters': best_p, 'CV Score (Train)': gs_best_score, 'Test Accuracy': test_acc,
                                               'CI Lower': ci_low, 'CI Upper': ci_upp, 'Training Time': elapsed_t}
        # Aggiungi risultato (anche NaN) ai risultati globali
        all_test_accuracies_per_model_global[estimator_name].append(test_acc)

        # --- CV Ripetuta per Test Statistici Interni ---
        if best_est is not None:
            print(f"    Esecuzione CV ripetuta ({NUM_TRIALS_CV_REPEATED} trials)...")
            start_cv_rep_time = time.time(); temp_cv_scores = []
            n_splits_stat_test_ds_cv = min(N_SPLITS_CV_REPEATED, y_dataset.value_counts().min())
            if n_splits_stat_test_ds_cv < 2: n_splits_stat_test_ds_cv = 2
            for trial in range(NUM_TRIALS_CV_REPEATED):
                skf_rep = StratifiedKFold(n_splits=n_splits_stat_test_ds_cv, shuffle=True, random_state=(420 + trial + dataset_idx))
                try:
                    scores = cross_val_score(best_est, X_processed_dataset, y_dataset, cv=skf_rep, scoring='accuracy', n_jobs=-1, error_score=np.nan)
                    temp_cv_scores.extend(scores[~np.isnan(scores)])
                except Exception as e_cv_rep: print(f"      Errore CV ripetuta trial {trial}: {e_cv_rep}")
            cv_scores_all_models_per_dataset[estimator_name] = temp_cv_scores
            cv_rep_time = time.time() - start_cv_rep_time
            if temp_cv_scores: print(f"      Media CV Ripetuta: {np.mean(temp_cv_scores):.4f} ± {np.std(temp_cv_scores):.4f} (Tempo: {cv_rep_time:.2f}s)")
            else: print(f"      CV Ripetuta non ha prodotto score (Tempo: {cv_rep_time:.2f}s).")

    # --- FINE CICLO MODELLI: ANALISI PER QUESTO DATASET ---
    print(f"\n{'-'*20} RIEPILOGO E ANALISI PER DATASET: {dataset_name} {'-'*20}")
    
    # Tabella Riassuntiva Per-Dataset
    sorted_models_ds = sorted(results_per_dataset.keys(), key=lambda x: results_per_dataset[x].get('Test Accuracy', -1), reverse=True)
    table_data_ds = []
    for rank_ds, mn_ds in enumerate(sorted_models_ds):
         res_ds = results_per_dataset[mn_ds]
         # Verifica che tutti i valori siano disponibili prima di formattare
         test_acc_str = f"{res_ds.get('Test Accuracy', np.nan):.4f}" if not pd.isna(res_ds.get('Test Accuracy')) else "N/A"
         ci_str = f"[{res_ds.get('CI Lower', np.nan):.4f}, {res_ds.get('CI Upper', np.nan):.4f}]" if not pd.isna(res_ds.get('CI Lower')) else "N/A"
         cv_score_str = f"{res_ds.get('CV Score (Train)', np.nan):.4f}" if not pd.isna(res_ds.get('CV Score (Train)')) else "N/A"
         time_str = f"{res_ds.get('Training Time', np.nan):.2f}s" if not pd.isna(res_ds.get('Training Time')) else "N/A"
         params_str = str({k: v for k, v in res_ds.get('Best Parameters', {}).items() if k not in ['algorithm', 'leaf_size', 'metric']}) if res_ds.get('Best Parameters') else "{}"
         
         table_data_ds.append([rank_ds + 1, mn_ds, test_acc_str, ci_str, cv_score_str, time_str, params_str])
                     
    headers_ds = ["Rank", "Modello", "Test Acc", f"{int(CONFIDENCE_LEVEL_CI*100)}% CI (Wilson)", "CV (Train) Acc", "Tempo GridS.", "Parametri Ottimali"]
    if table_data_ds: print(tabulate(table_data_ds, headers=headers_ds, tablefmt="grid"))
    else: print("Nessun risultato valido da tabulare per questo dataset.")

    # Test Statistici Interni al Dataset
    model_names_for_stat_ds = [name for name, scores in cv_scores_all_models_per_dataset.items() if scores and len(scores) >= N_SPLITS_CV_REPEATED]
    if len(model_names_for_stat_ds) >= 2:
        print(f"\n--- Test Statistici INTERNI (t-test su CV Ripetute) per {dataset_name} ---")
        p_value_matrix_ds = pd.DataFrame(index=model_names_for_stat_ds, columns=model_names_for_stat_ds, dtype=float)
        for i, model_i_ds in enumerate(model_names_for_stat_ds):
            for j, model_j_ds in enumerate(model_names_for_stat_ds):
                if i == j: p_value_matrix_ds.loc[model_i_ds, model_j_ds] = 1.0; continue
                s_i_ds, s_j_ds = np.array(cv_scores_all_models_per_dataset[model_i_ds]), np.array(cv_scores_all_models_per_dataset[model_j_ds])
                min_len_ds = min(len(s_i_ds), len(s_j_ds))
                if min_len_ds < N_SPLITS_CV_REPEATED: p_value_matrix_ds.loc[model_i_ds, model_j_ds] = np.nan
                elif np.allclose(s_i_ds[:min_len_ds], s_j_ds[:min_len_ds]): p_value_matrix_ds.loc[model_i_ds, model_j_ds] = 1.0
                else:
                    try: p_value_matrix_ds.loc[model_i_ds, model_j_ds] = ttest_rel(s_i_ds[:min_len_ds], s_j_ds[:min_len_ds])[1]
                    except ValueError: p_value_matrix_ds.loc[model_i_ds, model_j_ds] = np.nan
        p_table_disp_ds = [[m_i] + [f"{p_value_matrix_ds.loc[m_i, m_j]:.4f}{'** (Diff. Sign.)' if not pd.isna(p_value_matrix_ds.loc[m_i, m_j]) and p_value_matrix_ds.loc[m_i, m_j] < 0.05 else (' (No Sign. Diff.)' if not pd.isna(p_value_matrix_ds.loc[m_i, m_j]) else '')}" if m_i != m_j and not pd.isna(p_value_matrix_ds.loc[m_i, m_j]) else ('-' if m_i == m_j else "N/A") for m_j in model_names_for_stat_ds] for m_i in model_names_for_stat_ds]
        print(tabulate(p_table_disp_ds, headers=["Modello ↓ vs →"] + model_names_for_stat_ds, tablefmt="grid")); print("  ** (Diff. Sign.) indica p < 0.05 (differenza statisticamente significativa). (No Sign. Diff.) indica p >= 0.05.")
    else: print(f"\n--- Non abbastanza modelli con CV ripetute valide per test statistici interni su {dataset_name} ---")

    # Boxplot per-dataset
    data_to_plot_ds = [cv_scores_all_models_per_dataset[model] for model in model_names_for_stat_ds if model in cv_scores_all_models_per_dataset and cv_scores_all_models_per_dataset[model]]
    if data_to_plot_ds:
        try:
            plt.figure(figsize=(10, 5)); ax_ds = sns.boxplot(data=data_to_plot_ds)
            ax_ds.set_xticks(range(len(model_names_for_stat_ds))); ax_ds.set_xticklabels(model_names_for_stat_ds, rotation=45, ha='right')
            plt.title(f'Prestazioni Modelli (CV Ripetute) su Dataset: {dataset_name}'); plt.ylabel('Accuracy (CV Ripetuta)')
            plt.tight_layout(); plt.savefig(f'boxplot_{dataset_name.replace(" ", "_")}.png'); plt.close(); print(f"  Boxplot per {dataset_name} salvato.")
        except Exception as e_plot: print(f"  Errore durante la creazione del boxplot per {dataset_name}: {e_plot}"); plt.close() # Chiudi figura in caso di errore

    # Dettagli Modelli per-dataset
    print(f"\n--- Dettagli Modelli su {dataset_name} ---")
    for model_name_detail in sorted_models_ds:
        if model_name_detail in optimized_model_objects_per_dataset:
            print(f"\n  -- Dettagli per {model_name_detail} su {dataset_name} --")
            best_model_obj_ds = optimized_model_objects_per_dataset[model_name_detail]
            # Report Classificazione
            if len(X_test_ds) > 0 and len(y_test_ds) > 0:
                 try:
                     y_pred_detail = best_model_obj_ds.predict(X_test_ds)
                     print("  Report di Classificazione:"); print(classification_report(y_test_ds, y_pred_detail, zero_division=0))
                 except Exception as e_report: print(f"  Errore durante la generazione del report di classificazione: {e_report}")
            else: print("  Test set vuoto o y_test_ds non valida, impossibile generare report.")
            # Analisi Statistica CV Ripetuta
            if model_name_detail in cv_scores_all_models_per_dataset and cv_scores_all_models_per_dataset[model_name_detail]:
                sc_cv_rep = np.array(cv_scores_all_models_per_dataset[model_name_detail]); m_cv_rep, s_cv_rep = np.mean(sc_cv_rep), np.std(sc_cv_rep)
                print(f"  Analisi Statistica CV Ripetuta ({len(sc_cv_rep)} scores): Media: {m_cv_rep:.4f}, Std Dev: {s_cv_rep:.4f}")
                if len(sc_cv_rep) > 1:
                    se_cv = s_cv_rep / np.sqrt(len(sc_cv_rep)); t_crit_val = t_dist.ppf(0.975, len(sc_cv_rep) - 1); marg_err = t_crit_val * se_cv
                    print(f"    Errore Standard: {se_cv:.4f}, {int(CONFIDENCE_LEVEL_CI*100)}% IC (t-test): [{m_cv_rep - marg_err:.4f}, {m_cv_rep + marg_err:.4f}]")
            else: print(f"  Nessun score CV ripetuto valido per {model_name_detail}.")
        else:
             print(f"\n  -- Modello {model_name_detail} non ottimizzato con successo su {dataset_name} --")


# --- FINE CICLO DATASET: ANALISI GLOBALE ---
if processed_datasets_count == 0: print("\n\nERRORE CRITICO: Nessun dataset processato con successo."); exit()

print("\n\n" + "="*70 + "\n ANALISI GLOBALE INTER-DATASET \n" + "="*70)
global_summary_data = []
model_names_sorted_global = sorted(ALL_POTENTIAL_MODEL_NAMES, key=lambda mn: np.nanmean(all_test_accuracies_per_model_global.get(mn, [np.nan])), reverse=True)
for rank_g, model_name_g in enumerate(model_names_sorted_global):
    accs_g_list = all_test_accuracies_per_model_global.get(model_name_g, [])
    accs_g = np.array(accs_g_list); valid_accs_g = accs_g[~np.isnan(accs_g)]
    num_valid_ds_g = len(valid_accs_g); m_acc_g, s_acc_g, ci_lower_g, ci_upper_g = np.nan, np.nan, np.nan, np.nan
    if num_valid_ds_g > 0:
        m_acc_g = np.mean(valid_accs_g); s_acc_g = np.std(valid_accs_g)
        if num_valid_ds_g > 1:
            se_g = s_acc_g / np.sqrt(num_valid_ds_g); alpha_g = 1 - CONFIDENCE_LEVEL_CI
            try:
                 # Usiamo t-distribuzione perché n (numero di dataset) è spesso piccolo
                 t_crit_g = t_dist.ppf(1 - alpha_g/2, num_valid_ds_g - 1)
                 margin_err_g = t_crit_g * se_g
                 ci_lower_g = m_acc_g - margin_err_g; ci_upper_g = m_acc_g + margin_err_g
            except Exception as e_t_crit: print(f"  Attenzione: Errore calcolo t-critico per IC globale di {model_name_g}: {e_t_crit}")

    global_summary_data.append([
        rank_g + 1, model_name_g,
        f"{m_acc_g:.4f}" if not pd.isna(m_acc_g) else "N/A",
        f"[{ci_lower_g:.4f}, {ci_upper_g:.4f}]" if not pd.isna(ci_lower_g) else "N/A",
        f"{s_acc_g:.4f}" if not pd.isna(s_acc_g) else "N/A", num_valid_ds_g ])
headers_global_summary = ["Rank", "Modello", "Mean Global Test Acc", f"{int(CONFIDENCE_LEVEL_CI*100)}% CI (t-dist)", "Std Global Test Acc", "Num Datasets Validi"]
if global_summary_data: print(tabulate(global_summary_data, headers=headers_global_summary, tablefmt="grid"))

print("\n--- Test Statistici GLOBALI (t-test su Test Acc per dataset) ---")
final_model_names_for_global_stats = []
temp_scores_for_global_test = {mn: np.array(all_test_accuracies_per_model_global.get(mn, [])) for mn in ALL_POTENTIAL_MODEL_NAMES}
for mn in ALL_POTENTIAL_MODEL_NAMES:
    if np.sum(~np.isnan(temp_scores_for_global_test.get(mn, np.array([])))) >= 2: final_model_names_for_global_stats.append(mn)
if len(final_model_names_for_global_stats) >= 2:
    p_value_matrix_global = pd.DataFrame(index=final_model_names_for_global_stats, columns=final_model_names_for_global_stats, dtype=float)
    for i, model_i_g in enumerate(final_model_names_for_global_stats):
        for j, model_j_g in enumerate(final_model_names_for_global_stats):
            if i == j: p_value_matrix_global.loc[model_i_g, model_j_g] = 1.0; continue
            s_i_g, s_j_g = temp_scores_for_global_test[model_i_g], temp_scores_for_global_test[model_j_g]
            valid_mask_g = ~np.isnan(s_i_g) & ~np.isnan(s_j_g); p_i_g, p_j_g = s_i_g[valid_mask_g], s_j_g[valid_mask_g]
            # Assicurati che le lunghezze corrispondano dopo aver rimosso i NaN per coppie
            if len(p_i_g) != len(p_j_g): # Questo non dovrebbe succedere con la maschera, ma per sicurezza
                 print(f"Attenzione: Lunghezze non corrispondenti per t-test globale tra {model_i_g} e {model_j_g} dopo rimozione NaN.")
                 p_value_matrix_global.loc[model_i_g, model_j_g] = np.nan
                 continue
            if len(p_i_g) < 2: p_value_matrix_global.loc[model_i_g, model_j_g] = np.nan
            elif np.allclose(p_i_g, p_j_g): p_value_matrix_global.loc[model_i_g, model_j_g] = 1.0
            else:
                try: p_value_matrix_global.loc[model_i_g, model_j_g] = ttest_rel(p_i_g, p_j_g)[1]
                except ValueError: p_value_matrix_global.loc[model_i_g, model_j_g] = np.nan
    p_table_disp_g = [[m_i] + [f"{p_value_matrix_global.loc[m_i, m_j]:.4f}{'** (Diff. Sign.)' if not pd.isna(p_value_matrix_global.loc[m_i, m_j]) and p_value_matrix_global.loc[m_i, m_j] < 0.05 else (' (No Sign. Diff.)' if not pd.isna(p_value_matrix_global.loc[m_i, m_j]) else '')}" if m_i != m_j and not pd.isna(p_value_matrix_global.loc[m_i, m_j]) else ('-' if m_i == m_j else "N/A") for m_j in final_model_names_for_global_stats] for m_i in final_model_names_for_global_stats]
    print(tabulate(p_table_disp_g, headers=["Modello ↓ vs →"] + final_model_names_for_global_stats, tablefmt="grid")); print("  ** (Diff. Sign.) indica p < 0.05 (differenza statisticamente significativa). (No Sign. Diff.) indica p >= 0.05.")
else: print("--- Non abbastanza modelli con dati accoppiabili validi per test statistici globali ---")

print("\n--- Boxplot Globale delle Accuratezze Test per Dataset ---")
data_to_plot_g, plot_labels_g = [], []
for model_name_g in model_names_sorted_global:
    accs_g = np.array(all_test_accuracies_per_model_global.get(model_name_g, [])); valid_accs_g = accs_g[~np.isnan(accs_g)]
    if len(valid_accs_g) > 0: data_to_plot_g.append(valid_accs_g); plot_labels_g.append(f"{model_name_g} (n={len(valid_accs_g)})")
if data_to_plot_g:
    try:
        plt.figure(figsize=(12, 6)); ax_g = sns.boxplot(data=data_to_plot_g)
        ax_g.set_xticks(range(len(plot_labels_g))); ax_g.set_xticklabels(plot_labels_g, rotation=45, ha='right')
        plt.title('Prestazioni Globali Modelli (Accuratezza Test per Dataset)'); plt.ylabel('Accuratezza Test')
        plt.tight_layout(); plt.savefig('boxplot_global_performance.png'); plt.close(); print("  Boxplot globale salvato.")
    except Exception as e_plot_g: print(f"  Errore durante la creazione del boxplot globale: {e_plot_g}"); plt.close()
else: print("  Nessun dato per generare il boxplot globale.")

print("\nEsecuzione completata.")