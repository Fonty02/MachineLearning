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

from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
from scipy.stats import t as t_dist, norm as normal_dist, ttest_rel
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

# Parametri per WEKA-style Corrected Resampled T-Test
NUM_TRIALS_WEKA = 10      # Numero di ripetizioni (trials) della CV
N_SPLITS_WEKA_CV = 10     # Numero di fold nella CV
N_SPLITS_INNER_CV = 5     # Numero di fold per la CV interna di tuning (AGGIUNTO)
CONFIDENCE_LEVEL_CI = 0.95 # Livello di confidenza per gli intervalli

# --- FUNZIONE PER GRIGLIE DI PARAMETRI DINAMICHE ---
def get_param_grids(X_train_len_ds):
    """Genera le griglie di parametri dinamicamente."""
    if X_train_len_ds <= 0: X_train_len_ds = 1
    upper_bound_adaptive_k = int(np.floor(np.sqrt(X_train_len_ds)))
    if upper_bound_adaptive_k < 1: upper_bound_adaptive_k = 1
    # Ensure step is not 0, and at least one value is generated if upper_bound_adaptive_k is small
    step_adaptive = max(1, upper_bound_adaptive_k // 10 if upper_bound_adaptive_k > 10 else 1) 
    adaptive_k_values = [i for i in range(1, upper_bound_adaptive_k + 1, step_adaptive)]
    if not adaptive_k_values: adaptive_k_values = [min(1, X_train_len_ds if X_train_len_ds > 0 else 1)]
    
    upper_bound_knn = int(np.floor(np.sqrt(X_train_len_ds)))
    if upper_bound_knn < 1: upper_bound_knn = 1
    # Ensure n_neighbors_knn is not empty and contains reasonable values
    n_neighbors_knn_max = min(20, upper_bound_knn) 
    n_neighbors_knn = [i for i in range(1, n_neighbors_knn_max + 1)]
    if not n_neighbors_knn: n_neighbors_knn = [min(1, X_train_len_ds if X_train_len_ds > 0 else 1)]

    grids = []
    if CUSTOM_CLASSES_LOADED:
        grids.extend([
            {'name': 'StabilityAdaptiveKNN', 'estimator': StabilityAdaptiveKNN(),
             'params': {'stability_patience': [1, 3], 'k_step_stability': [1, 3],
                        'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}},
            {'name': 'ConfidenceAdaptiveKNN', 'estimator': ConfidenceAdaptiveKNN(),
             'params': {'confidence_threshold': [0.6, 0.8], 'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}},
            {'name': 'DensityConfidenceKNN', 'estimator': DensityConfidenceKNN(),
             'params': {'confidence_threshold': [0.6, 0.8], 'density_quantile': [0.7, 0.85],
                        'max_k_adaptive': adaptive_k_values, 'weights': ['uniform', 'distance']}},
        ])
    grids.append(
        {'name': 'KNeighborsClassifier', 'estimator': KN(),
         'params': {'n_neighbors': n_neighbors_knn, 'weights': ['uniform', 'distance'], 'algorithm': ['auto'], 'metric': ['euclidean', 'manhattan']}}
    )
    if not grids: print("ERRORE CRITICO: Nessuna griglia definita."); exit()
    return grids

ALL_POTENTIAL_MODEL_NAMES = [model_config['name'] for model_config in get_param_grids(X_train_len_ds=100)] # Pass a default length for initialization

# --- PREPARAZIONE FILE DI ESEMPIO ---
source_parkinson = "Parkinsson disease.csv"
if not os.path.exists(source_parkinson): print(f"ERRORE: File sorgente '{source_parkinson}' non trovato.")
diabetes_example_path = "diabetes_dataset.csv"
if not os.path.exists(diabetes_example_path) and os.path.exists(source_parkinson):
    print(f"ATTENZIONE: '{diabetes_example_path}' non trovato. Creo un esempio da '{source_parkinson}'.")
    try:
        shutil.copy(source_parkinson, diabetes_example_path); df_temp = pd.read_csv(diabetes_example_path);
        if 'status' in df_temp.columns: df_temp.rename(columns={'status': 'Outcome'}, inplace=True)
        df_temp.to_csv(diabetes_example_path, index=False); print(f"  Esempio '{diabetes_example_path}' creato.")
    except Exception as e: print(f"  Errore modifica esempio {diabetes_example_path}: {e}")
marketing_example_path = "marketing_campaign.csv"
if not os.path.exists(marketing_example_path) and os.path.exists(source_parkinson):
    print(f"ATTENZIONE: '{marketing_example_path}' non trovato. Creo un esempio da '{source_parkinson}'.")
    try:
        shutil.copy(source_parkinson, marketing_example_path); df_temp = pd.read_csv(marketing_example_path) # Assume source is CSV for copy
        num_rows = len(df_temp); df_temp.rename(columns={'status': 'Response', 'name': 'ID'}, inplace=True, errors='ignore')
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
all_repeated_cv_means_global = {name: [] for name in ALL_POTENTIAL_MODEL_NAMES}

# --- CICLO SUI DATASET ---
processed_datasets_count = 0
for dataset_idx, config in enumerate(DATASET_CONFIGS):
    dataset_name = config["name"]; dataset_file = config["file_path"]; target_col = config["target_column"]
    cols_to_drop_config = config["columns_to_drop"]; separator = config["separator"]; categorical_cols_config = config["categorical_cols_to_factorize"]
    print(f"\n\n{'='*50}\n ELABORAZIONE DATASET: {dataset_name} ({dataset_file}) ({dataset_idx+1}/{len(DATASET_CONFIGS)})\n{'='*50}")

    results_per_dataset = {}
    all_scores_10x10_cv_dataset = {name: [] for name in ALL_POTENTIAL_MODEL_NAMES}
    optimized_estimators_dataset = {}

    # --- CARICAMENTO DATI ---
    if not os.path.exists(dataset_file):
        print(f"ERRORE: File {dataset_file} non trovato. Salto dataset.")
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES:
            all_repeated_cv_means_global[mn_loop].append(np.nan)
        continue # Salta al prossimo dataset nel loop esterno
    try:
        df = pd.read_csv(dataset_file, sep=separator)
        print(f"  Dataset '{dataset_name}' caricato. Shape: {df.shape}")
    except Exception as e:
        print(f"ERRORE caricamento {dataset_file}: {e}. Salto dataset.")
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES:
            all_repeated_cv_means_global[mn_loop].append(np.nan)
        continue # Salta al prossimo dataset nel loop esterno

    # --- PREPROCESSING ---
    print("  Avvio Preprocessing...")
    if dataset_name == "Marketing Campaign" and 'Dt_Customer' in df.columns:
        try:
            df['Dt_Customer_Parsed'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce') # Explicit format
            valid_dates_mask = df['Dt_Customer_Parsed'].notna()
            if not valid_dates_mask.all(): print(f"    ATTENZIONE: 'Dt_Customer' con {sum(~valid_dates_mask)} date non valide (NaT).")
            if valid_dates_mask.any():
                df['Year_Customer'] = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].dt.year
                df['Month_Customer'] = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].dt.month
                latest_date_in_data = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].max()
                df['Customer_Age_Days'] = (latest_date_in_data - df.loc[valid_dates_mask, 'Dt_Customer_Parsed']).dt.days
                for col_fe in ['Year_Customer', 'Month_Customer', 'Customer_Age_Days']:
                    if df[col_fe].isnull().any():
                        fill_val = df[col_fe].median()
                        df[col_fe].fillna(fill_val, inplace=True)
            df = df.drop(columns=['Dt_Customer', 'Dt_Customer_Parsed'], errors='ignore'); print("    FE su 'Dt_Customer' completata.")
        except Exception as e_date: print(f"    ATTENZIONE: Errore FE 'Dt_Customer' ({e_date}). Droppo."); df = df.drop(columns=['Dt_Customer'], errors='ignore')

    df = df.drop(columns=cols_to_drop_config, errors='ignore')
    for col_loop in categorical_cols_config:
        if col_loop in df.columns and (df[col_loop].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col_loop])):
            df[col_loop] = pd.factorize(df[col_loop])[0]; print(f"    Colonna '{col_loop}' factorizzata.")
    for col_loop in df.select_dtypes(include='object').columns:
        if col_loop != target_col: # Non fattorizzare la colonna target qui se è object
            df[col_loop] = pd.factorize(df[col_loop])[0]; print(f"    Colonna '{col_loop}' (object non target) factorizzata.")
    
    if target_col not in df.columns:
        print(f"ERRORE: Target '{target_col}' non in {dataset_file}. Salto.")
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES:
            all_repeated_cv_means_global[mn_loop].append(np.nan)
        continue

    X_processed_dataset = df.drop(columns=[target_col])
    y_dataset = df[target_col]
    if y_dataset.dtype == 'object' or pd.api.types.is_categorical_dtype(y_dataset): # Factorize target if object/categorical
        y_dataset = pd.factorize(y_dataset)[0]; print(f"    Target '{target_col}' factorizzato.")
    
    numeric_cols_in_X = X_processed_dataset.select_dtypes(include=np.number).columns
    if not X_processed_dataset.empty and len(numeric_cols_in_X) > 0:
        for col_num_check in list(numeric_cols_in_X): # Iterate over a copy for potential modification
             if not pd.api.types.is_numeric_dtype(X_processed_dataset[col_num_check]):
                try: X_processed_dataset[col_num_check] = pd.to_numeric(X_processed_dataset[col_num_check], errors='coerce')
                except Exception: pass # Keep original if conversion fails
        # Re-select numeric columns after potential conversions
        numeric_cols_in_X_after_conv = X_processed_dataset.select_dtypes(include=np.number).columns
        if not numeric_cols_in_X_after_conv.empty:
            print(f"    Imputazione NaN su {len(numeric_cols_in_X_after_conv)} colonne numeriche.")
            imputer = SimpleImputer(strategy='mean')
            X_processed_dataset[numeric_cols_in_X_after_conv] = imputer.fit_transform(X_processed_dataset[numeric_cols_in_X_after_conv])
            if X_processed_dataset[numeric_cols_in_X_after_conv].isnull().sum().sum() > 0: print(f"    ATTENZIONE: NaN ancora in X DOPO imputazione!")
        else: print("    Nessuna colonna numerica trovata per l'imputazione dopo tentativi di conversione.")
    elif X_processed_dataset.empty:
        print(f"ATTENZIONE: X vuoto. Salto.")
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES:
            all_repeated_cv_means_global[mn_loop].append(np.nan)
        continue
    
    final_non_numeric_cols = X_processed_dataset.select_dtypes(exclude=np.number).columns
    if not final_non_numeric_cols.empty:
        print(f"ERRORE CRITICO: Colonne non numeriche in X: {list(final_non_numeric_cols)}. Salto.")
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES:
            all_repeated_cv_means_global[mn_loop].append(np.nan)
        continue
        
    n_splits_weka_adjusted = min(N_SPLITS_WEKA_CV, y_dataset.value_counts().min()) if y_dataset.nunique() > 1 else N_SPLITS_WEKA_CV
    if len(X_processed_dataset) < n_splits_weka_adjusted * 2 or n_splits_weka_adjusted < 2 :
        print(f"ERRORE: Dati insuff. per {n_splits_weka_adjusted}-Fold CV (n_samples={len(X_processed_dataset)}, min_class_count={y_dataset.value_counts().min()}). Salto.")
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES:
            all_repeated_cv_means_global[mn_loop].append(np.nan)
        continue
    print("  Preprocessing completato.")

    # --- OTTIMIZZAZIONE IPERPARAMETRI (su split temporaneo) ---
    print("\n  Ottimizzazione iperparametri preliminare...")
    tuning_successful = False
    try:
        # Ensure y_dataset is suitable for stratify
        if y_dataset.nunique() < 2:
             print(f"    Target con una sola classe ({y_dataset.unique()}), impossibile stratificare per tuning. Uso split non stratificato.")
             X_train_tune, _, y_train_tune, _ = train_test_split(X_processed_dataset, y_dataset, test_size=0.2, random_state=dataset_idx)
        else:
             X_train_tune, _, y_train_tune, _ = train_test_split(X_processed_dataset, y_dataset, test_size=0.2, random_state=dataset_idx, stratify=y_dataset)
        
        current_param_grids_tune = get_param_grids(len(X_train_tune))
        n_splits_tune_cv = min(N_SPLITS_INNER_CV, y_train_tune.value_counts().min() if y_train_tune.nunique() > 1 else N_SPLITS_INNER_CV)
        if n_splits_tune_cv < 2: n_splits_tune_cv = 2 # Ensure at least 2 splits if possible

        for model_config in current_param_grids_tune:
            estimator_name = model_config['name']; estimator = model_config['estimator']; param_grid = model_config['params']
            print(f"    Tuning {estimator_name}...")
            start_tune_time = time.time()
            
            current_n_splits_tune_cv = n_splits_tune_cv
            if y_train_tune.nunique() < 2: # Cannot use StratifiedKFold if only one class
                cv_tune_object = current_n_splits_tune_cv 
            else:
                # Adjust n_splits if y_train_tune has fewer samples in a class than n_splits
                min_class_count_tune = y_train_tune.value_counts().min()
                if min_class_count_tune < current_n_splits_tune_cv:
                    print(f"      Adattamento n_splits per tuning da {current_n_splits_tune_cv} a {max(2, min_class_count_tune)} per {estimator_name} a causa di poche istanze di classe.")
                    current_n_splits_tune_cv = max(2, min_class_count_tune)
                if current_n_splits_tune_cv < 2: # Fallback if still not enough
                     cv_tune_object = 2 
                else:
                     cv_tune_object = StratifiedKFold(n_splits=current_n_splits_tune_cv, shuffle=True, random_state=dataset_idx+1)

            gs_tune = GridSearchCV(estimator, param_grid, cv=cv_tune_object,
                                  scoring='accuracy', n_jobs=-1, verbose=0, error_score=np.nan)
            gs_tune.fit(X_train_tune, y_train_tune)
            elapsed_tune_time = time.time() - start_tune_time
            if not pd.isna(gs_tune.best_score_):
                optimized_estimators_dataset[estimator_name] = gs_tune.best_estimator_
                print(f"      Best Params found: {gs_tune.best_params_} (Score: {gs_tune.best_score_:.4f}, Tempo: {elapsed_tune_time:.2f}s)")
                tuning_successful = True 
            else:
                print(f"      GridSearchCV di tuning fallito per {estimator_name}. Uso istanza base. (Tempo: {elapsed_tune_time:.2f}s)")
                optimized_estimators_dataset[estimator_name] = clone(estimator) 
                tuning_successful = True 
    except Exception as e_tune:
        print(f"ERRORE durante l'ottimizzazione preliminare: {e_tune}. Salto dataset.")
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES:
            all_repeated_cv_means_global[mn_loop].append(np.nan)
        continue

    if not tuning_successful or not optimized_estimators_dataset:
        print(f"ERRORE: Nessun modello disponibile dopo l'ottimizzazione per {dataset_name}. Salto dataset.")
        for mn_loop in ALL_POTENTIAL_MODEL_NAMES:
            all_repeated_cv_means_global[mn_loop].append(np.nan)
        continue

    # --- WEKA-STYLE REPEATED CROSS-VALIDATION (10x10) ---
    print(f"\n  Avvio Repeated Cross-Validation ({NUM_TRIALS_WEKA} trials x {n_splits_weka_adjusted} folds)...")
    model_names_to_run_cv = list(optimized_estimators_dataset.keys())
    total_repeated_cv_time = 0
    scores_per_fold_trial = {mn: [[] for _ in range(NUM_TRIALS_WEKA)] for mn in model_names_to_run_cv}

    for trial in range(NUM_TRIALS_WEKA):
        print(f"    Trial {trial + 1}/{NUM_TRIALS_WEKA}...")
        start_trial_time = time.time()
        
        # Stratification check for main CV
        current_n_splits_weka = n_splits_weka_adjusted
        if y_dataset.nunique() < 2:
            skf_weka = StratifiedKFold(n_splits=current_n_splits_weka, shuffle=True, random_state=(dataset_idx * 100 + trial)) # Will likely fail if KFold is needed
            print(f"    ATTENZIONE: Target con una sola classe per CV principale. StratifiedKFold potrebbe fallire o comportarsi come KFold.")
        else:
            min_class_count_main_cv = y_dataset.value_counts().min()
            if min_class_count_main_cv < current_n_splits_weka:
                 print(f"    Adattamento n_splits per CV principale da {current_n_splits_weka} a {max(2,min_class_count_main_cv)} a causa di poche istanze di classe.")
                 current_n_splits_weka = max(2, min_class_count_main_cv)
            if current_n_splits_weka < 2: # Should have been caught by earlier check, but as a safeguard
                print(f"    ERRORE: Non abbastanza campioni per CV principale ({current_n_splits_weka} splits). Salto CV per questo dataset.")
                for model_name in model_names_to_run_cv: scores_per_fold_trial[model_name][trial] = [np.nan] * n_splits_weka_adjusted # Fill with NaNs
                continue # Skip to next trial or end if this was the only way
            skf_weka = StratifiedKFold(n_splits=current_n_splits_weka, shuffle=True, random_state=(dataset_idx * 100 + trial))
        
        fold_idx = 0
        try:
            for train_idx, test_idx in skf_weka.split(X_processed_dataset, y_dataset):
                X_train_cv, X_test_cv = X_processed_dataset.iloc[train_idx], X_processed_dataset.iloc[test_idx]
                y_train_cv, y_test_cv = y_dataset.iloc[train_idx], y_dataset.iloc[test_idx]
                
                if len(X_train_cv) == 0 or len(X_test_cv) == 0:
                    print(f"      Fold {fold_idx} vuoto (train o test). Inserimento NaN.")
                    for model_name in model_names_to_run_cv:
                        scores_per_fold_trial[model_name][trial].append(np.nan)
                    fold_idx += 1
                    continue

                for model_name in model_names_to_run_cv:
                    estimator_instance = clone(optimized_estimators_dataset[model_name])
                    try:
                        estimator_instance.fit(X_train_cv, y_train_cv)
                        score = estimator_instance.score(X_test_cv, y_test_cv)
                        scores_per_fold_trial[model_name][trial].append(score)
                    except Exception as e_fit_score:
                        print(f"      Errore fit/score {model_name} fold {fold_idx}: {e_fit_score}")
                        scores_per_fold_trial[model_name][trial].append(np.nan)
                fold_idx += 1
        except ValueError as ve_skf: # Catch errors from skf.split itself (e.g. not enough members in a class for stratify)
            print(f"    ERRORE durante lo split CV (Trial {trial+1}): {ve_skf}. Riempio gli score del trial con NaN.")
            for model_name in model_names_to_run_cv:
                # Fill remaining scores for this trial with NaNs, ensure list length is consistent
                num_expected_folds_this_trial = current_n_splits_weka # Use the adjusted n_splits for this trial
                scores_per_fold_trial[model_name][trial] = [np.nan] * num_expected_folds_this_trial
            # We might want to 'continue' to the next trial here if the error is specific to this trial's random_state
            # or break if it's a fundamental issue with the data for this n_splits setting.
            # For now, we fill with NaNs and let it proceed.

        elapsed_trial_time = time.time() - start_trial_time
        total_repeated_cv_time += elapsed_trial_time
        # Ensure consistent number of scores per trial (even if NaNs from errors)
        for model_name in model_names_to_run_cv:
             if len(scores_per_fold_trial[model_name][trial]) < current_n_splits_weka:
                  scores_per_fold_trial[model_name][trial].extend([np.nan] * (current_n_splits_weka - len(scores_per_fold_trial[model_name][trial])))


    print(f"  Tempo totale Repeated CV: {total_repeated_cv_time:.2f}s")

    # Raccogli tutti gli score per modello
    for model_name in model_names_to_run_cv:
        all_scores_10x10_cv_dataset[model_name] = [score for trial_scores in scores_per_fold_trial[model_name] for score in trial_scores]

    # --- FINE CV RIPETUTA: Analisi per questo dataset ---
    processed_datasets_count += 1
    print(f"\n{'-'*20} RIEPILOGO E ANALISI PER DATASET: {dataset_name} {'-'*20}")

    summary_per_dataset_list = []
    any_model_succeeded_ds = False
    # Use n_splits_weka_adjusted (original target) or dynamically track actual splits used if it varies
    # For k_total_folds, it should be consistent if all trials used the same number of folds.
    # If current_n_splits_weka changed per trial, this needs more careful handling.
    # Assuming n_splits_weka_adjusted is the intended number for calculating total folds.
    k_total_folds = NUM_TRIALS_WEKA * n_splits_weka_adjusted # Ideal number of scores
    
    for model_name in model_names_to_run_cv: 
        scores = np.array(all_scores_10x10_cv_dataset.get(model_name, []))
        # Ensure scores array has expected length, padding with NaN if shorter (e.g. due to early CV failure)
        if len(scores) < k_total_folds:
            scores = np.pad(scores, (0, k_total_folds - len(scores)), 'constant', constant_values=np.nan)

        valid_scores = scores[~np.isnan(scores)]; num_valid_scores = len(valid_scores)
        
        mean_score, std_score, ci_lower_ds, ci_upper_ds = np.nan, np.nan, np.nan, np.nan
        if num_valid_scores > 0:
            mean_score = np.mean(valid_scores); std_score = np.std(valid_scores); any_model_succeeded_ds = True
            if num_valid_scores > 1:
                 se_ds = std_score / np.sqrt(num_valid_scores); alpha_ds = 1 - CONFIDENCE_LEVEL_CI
                 try:
                     t_crit_ds = t_dist.ppf(1 - alpha_ds/2, num_valid_scores - 1)
                     margin_err_ds = t_crit_ds * se_ds
                     ci_lower_ds = mean_score - margin_err_ds; ci_upper_ds = mean_score + margin_err_ds
                 except Exception as e_t_crit_ds: print(f"  Attenzione: Errore calcolo IC locale {model_name}: {e_t_crit_ds}")
        summary_per_dataset_list.append({"Modello": model_name, "Mean Acc": mean_score, "Std Dev": std_score,
                                         "CI Lower": ci_lower_ds, "CI Upper": ci_upper_ds, "Valid Scores": num_valid_scores})
        all_repeated_cv_means_global[model_name].append(mean_score)
    
    for model_name in ALL_POTENTIAL_MODEL_NAMES:
        if model_name not in model_names_to_run_cv: all_repeated_cv_means_global[model_name].append(np.nan)

    if not any_model_succeeded_ds: print("ATTENZIONE: Nessun modello ha prodotto score validi dalla Repeated CV.")

    summary_per_dataset_list.sort(key=lambda x: x["Mean Acc"] if not pd.isna(x["Mean Acc"]) else -1, reverse=True)
    table_data_ds = [[rank + 1, item["Modello"], f"{item['Mean Acc']:.4f}" if not pd.isna(item['Mean Acc']) else "N/A",
                      f"[{item['CI Lower']:.4f}, {item['CI Upper']:.4f}]" if not pd.isna(item['CI Lower']) else "N/A",
                      f"{item['Std Dev']:.4f}" if not pd.isna(item['Std Dev']) else "N/A", f"{item['Valid Scores']}/{k_total_folds}"]
                     for rank, item in enumerate(summary_per_dataset_list)]
    headers_ds = ["Rank", "Modello", f"Mean Acc ({NUM_TRIALS_WEKA}x{n_splits_weka_adjusted} CV)", f"{int(CONFIDENCE_LEVEL_CI*100)}% CI (t-dist)", "Std Dev", "Valid Scores"]
    if table_data_ds: print(tabulate(table_data_ds, headers=headers_ds, tablefmt="grid"))

    model_names_for_stat_ds = [name for name in model_names_to_run_cv 
                               if all_scores_10x10_cv_dataset.get(name) and 
                                  len(all_scores_10x10_cv_dataset[name]) == k_total_folds and 
                                  not np.all(np.isnan(all_scores_10x10_cv_dataset[name]))] # Ensure not all NaNs for a model

    if len(model_names_for_stat_ds) >= 2:
        print(f"\n--- Test Statistici LOCALI (Corrected Resampled t-test) per {dataset_name} ---")
        # Use the actual number of splits from the main CV if it was adjusted.
        # For simplicity, using n_splits_weka_adjusted, assuming it was mostly stable.
        # If current_n_splits_weka was used and varied, this calculation of n1, n2 might need adjustment.
        n1 = len(X_processed_dataset) * (1 - 1/n_splits_weka_adjusted) if n_splits_weka_adjusted > 0 else 0
        n2 = len(X_processed_dataset) / n_splits_weka_adjusted if n_splits_weka_adjusted > 0 else 0
        test_train_ratio = n2 / n1 if n1 > 0 else 0
        k_corrected_test = k_total_folds # Total number of observations (scores)

        p_value_matrix_corrected_ds = pd.DataFrame(index=model_names_for_stat_ds, columns=model_names_for_stat_ds, dtype=float)
        for i, model_i_ds in enumerate(model_names_for_stat_ds):
            for j, model_j_ds in enumerate(model_names_for_stat_ds):
                if i == j: p_value_matrix_corrected_ds.loc[model_i_ds, model_j_ds] = 1.0; continue
                scores_i_raw = np.array(all_scores_10x10_cv_dataset[model_i_ds])
                scores_j_raw = np.array(all_scores_10x10_cv_dataset[model_j_ds])
                
                # Handle NaNs by pairwise removal for differences
                valid_pair_mask = ~np.isnan(scores_i_raw) & ~np.isnan(scores_j_raw)
                scores_i = scores_i_raw[valid_pair_mask]
                scores_j = scores_j_raw[valid_pair_mask]

                if len(scores_i) < 2: # Not enough valid pairs for t-test
                    p_value_matrix_corrected_ds.loc[model_i_ds, model_j_ds] = np.nan
                    continue

                differences = scores_i - scores_j
                mean_diff = np.mean(differences); var_diff = np.var(differences, ddof=1)
                
                current_k_corrected_test = len(differences) # Use actual number of valid differences

                if var_diff < 1e-10: p_val_corrected = 1.0 if np.isclose(mean_diff, 0) else 0.0
                else:
                    denominator_corrected = np.sqrt( (1/current_k_corrected_test + test_train_ratio) * var_diff )
                    if denominator_corrected < 1e-10 or np.isclose(denominator_corrected, 0):
                        p_val_corrected = 1.0 if np.isclose(mean_diff, 0) else 0.0
                    else:
                        t_stat_corrected = mean_diff / denominator_corrected
                        p_val_corrected = t_dist.sf(np.abs(t_stat_corrected), current_k_corrected_test - 1) * 2
                p_value_matrix_corrected_ds.loc[model_i_ds, model_j_ds] = p_val_corrected
        p_table_disp_corr_ds = [[m_i] + [f"{p_value_matrix_corrected_ds.loc[m_i, m_j]:.4f}{'** (Diff. Sign.)' if not pd.isna(p_value_matrix_corrected_ds.loc[m_i, m_j]) and p_value_matrix_corrected_ds.loc[m_i, m_j] < 0.05 else (' (No Sign. Diff.)' if not pd.isna(p_value_matrix_corrected_ds.loc[m_i, m_j]) else '')}" if m_i != m_j and not pd.isna(p_value_matrix_corrected_ds.loc[m_i, m_j]) else ('-' if m_i == m_j else "N/A") for m_j in model_names_for_stat_ds] for m_i in model_names_for_stat_ds]
        print(tabulate(p_table_disp_corr_ds, headers=["Modello ↓ vs →"] + model_names_for_stat_ds, tablefmt="grid")); print("  ** (Diff. Sign.) indica p < 0.05 (Corrected Resampled t-test). (No Sign. Diff.) indica p >= 0.05.")
    else: print(f"\n--- Non abbastanza modelli ({len(model_names_for_stat_ds)}) con score CV 10x10 completi e validi per Corrected t-test su {dataset_name} ---")

    data_to_plot_ds_valid = []
    labels_plot_ds_valid = []
    for item in summary_per_dataset_list:
        if item["Valid Scores"] > 0:
            scores_for_plot = np.array(all_scores_10x10_cv_dataset[item["Modello"]])
            valid_scores_for_plot = scores_for_plot[~np.isnan(scores_for_plot)]
            if len(valid_scores_for_plot) > 0:
                 data_to_plot_ds_valid.append(valid_scores_for_plot)
                 labels_plot_ds_valid.append(item["Modello"])

    if data_to_plot_ds_valid:
        try:
            plt.figure(figsize=(max(10, len(labels_plot_ds_valid) * 0.8), 5)) # Adjust width
            ax_ds = sns.boxplot(data=data_to_plot_ds_valid)
            ax_ds.set_xticks(range(len(labels_plot_ds_valid)))
            ax_ds.set_xticklabels(labels_plot_ds_valid, rotation=45, ha='right')
            plt.title(f'Prestazioni Modelli ({NUM_TRIALS_WEKA}x{n_splits_weka_adjusted}-Fold CV Scores) su: {dataset_name}'); plt.ylabel('Accuracy')
            plt.tight_layout(); plt.savefig(f'boxplot_repeatedcv_{dataset_name.replace(" ", "_")}.png'); plt.close(); print(f"  Boxplot CV ripetuta per {dataset_name} salvato.")
        except Exception as e_plot: print(f"  Errore boxplot CV ripetuta: {e_plot}"); plt.close()
    else: print("  Nessun dato valido per il boxplot CV ripetuta.")


# --- FINE CICLO DATASET: ANALISI GLOBALE ---
if processed_datasets_count == 0:
    print("\n\nERRORE CRITICO: Nessun dataset processato con successo."); exit()

print("\n\n" + "="*70 + f"\n ANALISI GLOBALE INTER-DATASET (basata su medie {NUM_TRIALS_WEKA}x{N_SPLITS_WEKA_CV}-Fold CV) \n" + "="*70)
global_summary_data = []
model_names_sorted_global = sorted(ALL_POTENTIAL_MODEL_NAMES, key=lambda mn: np.nanmean(all_repeated_cv_means_global.get(mn, [np.nan])), reverse=True)
for rank_g, model_name_g in enumerate(model_names_sorted_global):
    means_g_list = all_repeated_cv_means_global.get(model_name_g, [])
    means_g = np.array(means_g_list); valid_means_g = means_g[~np.isnan(means_g)]
    num_valid_ds_g = len(valid_means_g); m_mean_g, s_mean_g, ci_lower_g, ci_upper_g = np.nan, np.nan, np.nan, np.nan
    if num_valid_ds_g > 0:
        m_mean_g = np.mean(valid_means_g); s_mean_g = np.std(valid_means_g)
        if num_valid_ds_g > 1:
            se_g = s_mean_g / np.sqrt(num_valid_ds_g); alpha_g = 1 - CONFIDENCE_LEVEL_CI
            try:
                 t_crit_g = t_dist.ppf(1 - alpha_g/2, num_valid_ds_g - 1)
                 margin_err_g = t_crit_g * se_g
                 ci_lower_g = m_mean_g - margin_err_g; ci_upper_g = m_mean_g + margin_err_g
            except Exception as e_t_crit: print(f"  Attenzione: Errore calcolo IC globale {model_name_g}: {e_t_crit}")
    global_summary_data.append([ rank_g + 1, model_name_g, f"{m_mean_g:.4f}" if not pd.isna(m_mean_g) else "N/A",
                                f"[{ci_lower_g:.4f}, {ci_upper_g:.4f}]" if not pd.isna(ci_lower_g) else "N/A",
                                f"{s_mean_g:.4f}" if not pd.isna(s_mean_g) else "N/A", num_valid_ds_g ])
headers_global_summary = ["Rank", "Modello", f"Mean Global Acc ({NUM_TRIALS_WEKA}x{N_SPLITS_WEKA_CV} CV)", f"{int(CONFIDENCE_LEVEL_CI*100)}% CI (t-dist)", "Std Dev Global Acc", "Num Datasets Validi"]
if global_summary_data: print(tabulate(global_summary_data, headers=headers_global_summary, tablefmt="grid"))

print(f"\n--- Test Statistici GLOBALI (t-test su medie {NUM_TRIALS_WEKA}x{N_SPLITS_WEKA_CV}-Fold CV per dataset) ---")
final_model_names_for_global_stats = []
temp_scores_for_global_test = {mn: np.array(all_repeated_cv_means_global.get(mn, [])) for mn in ALL_POTENTIAL_MODEL_NAMES}
for mn in ALL_POTENTIAL_MODEL_NAMES:
    if np.sum(~np.isnan(temp_scores_for_global_test.get(mn, np.array([])))) >= 2: final_model_names_for_global_stats.append(mn)
unique_final_model_names_global = []
seen_global = set()
for name in model_names_sorted_global: # Iterate in sorted order to maintain consistency in tables
    if name in final_model_names_for_global_stats and name not in seen_global:
        unique_final_model_names_global.append(name)
        seen_global.add(name)

if len(unique_final_model_names_global) >= 2:
    p_value_matrix_global = pd.DataFrame(index=unique_final_model_names_global, columns=unique_final_model_names_global, dtype=float)
    for i, model_i_g in enumerate(unique_final_model_names_global):
        for j, model_j_g in enumerate(unique_final_model_names_global):
            if i == j: p_value_matrix_global.loc[model_i_g, model_j_g] = 1.0; continue
            s_i_g_raw, s_j_g_raw = temp_scores_for_global_test[model_i_g], temp_scores_for_global_test[model_j_g]
            valid_mask_g = ~np.isnan(s_i_g_raw) & ~np.isnan(s_j_g_raw)
            p_i_g, p_j_g = s_i_g_raw[valid_mask_g], s_j_g_raw[valid_mask_g]
            if len(p_i_g) < 2: p_value_matrix_global.loc[model_i_g, model_j_g] = np.nan
            elif np.allclose(p_i_g, p_j_g): p_value_matrix_global.loc[model_i_g, model_j_g] = 1.0
            else:
                try:
                    stat_result = ttest_rel(p_i_g, p_j_g)
                    p_value_matrix_global.loc[model_i_g, model_j_g] = stat_result.pvalue
                except ValueError: p_value_matrix_global.loc[model_i_g, model_j_g] = np.nan # e.g. if one array is all same values
    p_table_disp_g = [[m_i] + [f"{p_value_matrix_global.loc[m_i, m_j]:.4f}{'** (Diff. Sign.)' if not pd.isna(p_value_matrix_global.loc[m_i, m_j]) and p_value_matrix_global.loc[m_i, m_j] < 0.05 else (' (No Sign. Diff.)' if not pd.isna(p_value_matrix_global.loc[m_i, m_j]) else '')}" if m_i != m_j and not pd.isna(p_value_matrix_global.loc[m_i, m_j]) else ('-' if m_i == m_j else "N/A") for m_j in unique_final_model_names_global] for m_i in unique_final_model_names_global]
    print(tabulate(p_table_disp_g, headers=["Modello ↓ vs →"] + unique_final_model_names_global, tablefmt="grid")); print("  ** (Diff. Sign.) indica p < 0.05 (t-test standard su medie CV). (No Sign. Diff.) indica p >= 0.05.")
else: print(f"--- Non abbastanza modelli ({len(unique_final_model_names_global)}) per test statistici globali ---")

print(f"\n--- Boxplot Globale delle Medie Accuratezze ({NUM_TRIALS_WEKA}x{N_SPLITS_WEKA_CV}-Fold CV) per Dataset ---")
data_to_plot_g, plot_labels_g = [], []
for model_name_g in model_names_sorted_global: # Use sorted names for consistent plot order
    means_g = np.array(all_repeated_cv_means_global.get(model_name_g, []))
    valid_means_g = means_g[~np.isnan(means_g)]
    if len(valid_means_g) > 0:
        data_to_plot_g.append(valid_means_g)
        plot_labels_g.append(f"{model_name_g} (n={len(valid_means_g)})")
if data_to_plot_g:
    try:
        plt.figure(figsize=(max(12, len(plot_labels_g)*0.8), 6)) # Adjust width
        ax_g = sns.boxplot(data=data_to_plot_g)
        ax_g.set_xticks(range(len(plot_labels_g)))
        ax_g.set_xticklabels(plot_labels_g, rotation=45, ha='right')
        plt.title(f'Prestazioni Globali Modelli (Media {NUM_TRIALS_WEKA}x{N_SPLITS_WEKA_CV}-Fold CV per Dataset)'); plt.ylabel(f'Mean Accuracy ({NUM_TRIALS_WEKA}x{N_SPLITS_WEKA_CV} CV)')
        plt.tight_layout(); plt.savefig('boxplot_global_repeatedcv_means.png'); plt.close(); print("  Boxplot globale salvato.")
    except Exception as e_plot_g: print(f"  Errore boxplot globale: {e_plot_g}"); plt.close()
else: print("  Nessun dato per generare il boxplot globale.")

print("\nEsecuzione completata.")