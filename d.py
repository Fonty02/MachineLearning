# -*- coding: utf-8 -*-
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score # classification_report non serve più qui
try:
    from Custom import StabilityAdaptiveKNN, ConfidenceAdaptiveKNN, DensityConfidenceKNN
    CUSTOM_CLASSES_LOADED = True
    print("Classi personalizzate da Custom.py caricate con successo.")
except ImportError as e:
    print(f"ATTENZIONE: Impossibile importare classi da Custom.py: {e}")
    print(">>> Verranno usate classi fittizie. Solo KNeighborsClassifier sarà testato. <<<")
    CUSTOM_CLASSES_LOADED = False
    class StabilityAdaptiveKNN: pass
    class ConfidenceAdaptiveKNN: pass
    class DensityConfidenceKNN: pass

from sklearn.neighbors import KNeighborsClassifier as KN
from sklearn.impute import SimpleImputer
from sklearn.base import clone
import pandas as pd
import numpy as np
import time
# from tabulate import tabulate # Meno necessario se l'output principale è per WEKA
# from scipy.stats import t as t_dist, norm as normal_dist, ttest_rel # I test saranno in WEKA
# import matplotlib.pyplot as plt # I plot possono essere fatti da WEKA o da Python separatamente
# import seaborn as sns
import os
import shutil

# Import necessari per la conversione in ARFF e gestione JVM
import weka.core.jvm as jvm
from weka.core.converters import Loader, Saver
# from weka.core.dataset import Instances # Non strettamente necessario se salviamo via CSV -> ARFF

# --- AVVIO JVM (NECESSARIO PER python-weka-wrapper3) ---
try:
    jvm.start(max_heap_size="2g")
    print("JVM avviata con successo.")
except Exception as e:
    print(f"ATTENZIONE: Impossibile avviare JVM. Dettagli: {e}. La conversione in ARFF potrebbe fallire.")
    # exit()

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
N_SPLITS_INNER_CV = 5
# CONFIDENCE_LEVEL_CI = 0.95 # Meno rilevante qui se l'analisi è in WEKA

def get_param_grids(X_train_len_ds):
    if X_train_len_ds <= 0: X_train_len_ds = 1
    upper_bound_adaptive_k = int(np.floor(np.sqrt(X_train_len_ds)))
    if upper_bound_adaptive_k < 1: upper_bound_adaptive_k = 1
    step_adaptive = max(1, upper_bound_adaptive_k // 10 if upper_bound_adaptive_k > 10 else 1)
    adaptive_k_values = [i for i in range(1, upper_bound_adaptive_k + 1, step_adaptive)]
    if not adaptive_k_values: adaptive_k_values = [min(1, X_train_len_ds if X_train_len_ds > 0 else 1)]

    upper_bound_knn = int(np.floor(np.sqrt(X_train_len_ds)))
    if upper_bound_knn < 1: upper_bound_knn = 1
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

ALL_POTENTIAL_MODEL_NAMES = [model_config['name'] for model_config in get_param_grids(X_train_len_ds=100)]

# --- PREPARAZIONE FILE DI ESEMPIO (come prima) ---
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


# --- FUNZIONE PER CONVERTIRE DATAFRAME PANDAS IN FILE ARFF ---
# Questa funzione è per i *dati* di input, non per i *risultati*. La modificheremo per i risultati.
def pandas_df_to_arff_file(df, filename, is_result_file=False, target_col_name=None):
    """
    Converte un DataFrame pandas in un file ARFF.
    Se is_result_file è True, non imposta un class_index specifico.
    Altrimenti, cerca di impostare target_col_name come class_index.
    """
    # WEKA non gestisce bene i nomi delle colonne con certi caratteri speciali.
    # Sostituiamo spazi e altri caratteri problematici con underscore.
    original_columns = list(df.columns) # Salva una copia
    df.columns = [str(col).replace(' ', '_').replace('[', '').replace(']', '').replace('<', '').replace('(', '').replace(')', '') for col in df.columns]
    if target_col_name:
        target_col_name_clean = target_col_name.replace(' ', '_').replace('[', '').replace(']', '').replace('<', '').replace('(', '').replace(')', '')

    temp_csv_path = f"temp_for_weka_{os.path.basename(filename).replace('.arff','')}.csv"
    df.to_csv(temp_csv_path, index=False)

    try:
        loader = Loader(classname="weka.core.converters.CSVLoader")
        data = loader.load_file(temp_csv_path)

        if not is_result_file and target_col_name:
            target_idx = -1
            for i in range(data.num_attributes):
                if data.attribute(i).name == target_col_name_clean:
                    target_idx = i
                    break
            if target_idx != -1:
                data.class_index = target_idx
                print(f"    Colonna target '{target_col_name_clean}' impostata per ARFF dati.")
            elif data.num_attributes > 0: # Se non specificata o non trovata, imposta l'ultima
                print(f"    ATTENZIONE: Target '{target_col_name_clean}' non trovato o non specificato. Impostata l'ultima colonna come target per ARFF dati.")
                data.class_index = data.num_attributes - 1
        elif not is_result_file and data.num_attributes > 0: # Se no target col specificato, l'ultima
             data.class_index = data.num_attributes - 1
        # Per i file di risultati, non impostiamo un class_index specifico qui.
        # WEKA Analyse lo gestirà in base alle selezioni dell'utente.

        saver = Saver(classname="weka.core.converters.ArffSaver")
        saver.save_file(data, filename)
        print(f"  DataFrame convertito e salvato in {filename}")
    except Exception as e:
        print(f"ERRORE durante la conversione in ARFF per {filename}: {e}")
    finally:
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        df.columns = original_columns # Ripristina i nomi originali delle colonne sul DataFrame in memoria


# Lista per raccogliere tutti i risultati dei fold da tutti i dataset e algoritmi
all_fold_results_for_weka = []

# --- CICLO SUI DATASET ---
processed_datasets_count = 0
for dataset_idx, config in enumerate(DATASET_CONFIGS):
    dataset_name = config["name"]; dataset_file = config["file_path"]; target_col = config["target_column"]
    cols_to_drop_config = config["columns_to_drop"]; separator = config["separator"]; categorical_cols_config = config["categorical_cols_to_factorize"]
    print(f"\n\n{'='*50}\n ELABORAZIONE DATASET: {dataset_name} ({dataset_file}) ({dataset_idx+1}/{len(DATASET_CONFIGS)})\n{'='*50}")

    # --- CARICAMENTO DATI ---
    if not os.path.exists(dataset_file):
        print(f"ERRORE: File {dataset_file} non trovato. Salto dataset.")
        continue
    try:
        df = pd.read_csv(dataset_file, sep=separator)
        print(f"  Dataset '{dataset_name}' caricato. Shape: {df.shape}")
    except Exception as e:
        print(f"ERRORE caricamento {dataset_file}: {e}. Salto dataset.")
        continue

    # --- PREPROCESSING (identico al tuo script originale, con piccole correzioni di robustezza) ---
    print("  Avvio Preprocessing...")
    if dataset_name == "Marketing Campaign" and 'Dt_Customer' in df.columns:
        try:
            df['Dt_Customer_Parsed'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y', errors='coerce')
            valid_dates_mask = df['Dt_Customer_Parsed'].notna()
            if not valid_dates_mask.all(): print(f"    ATTENZIONE: 'Dt_Customer' con {sum(~valid_dates_mask)} date non valide (NaT).")
            if valid_dates_mask.any(): # Solo se ci sono date valide
                df['Year_Customer'] = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].dt.year
                df['Month_Customer'] = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].dt.month
                latest_date_in_data = df.loc[valid_dates_mask, 'Dt_Customer_Parsed'].max() # Basato solo su date valide
                df['Customer_Age_Days'] = (latest_date_in_data - df.loc[valid_dates_mask, 'Dt_Customer_Parsed']).dt.days
                for col_fe in ['Year_Customer', 'Month_Customer', 'Customer_Age_Days']:
                    if col_fe in df.columns and df[col_fe].isnull().any(): # Verifica che la colonna esista
                        fill_val = df[col_fe].median()
                        df[col_fe].fillna(fill_val, inplace=True)
            df = df.drop(columns=['Dt_Customer', 'Dt_Customer_Parsed'], errors='ignore'); print("    FE su 'Dt_Customer' completata.")
        except Exception as e_date: print(f"    ATTENZIONE: Errore FE 'Dt_Customer' ({e_date}). Droppo."); df = df.drop(columns=['Dt_Customer'], errors='ignore')

    df = df.drop(columns=cols_to_drop_config, errors='ignore')
    for col_loop in categorical_cols_config:
        if col_loop in df.columns and (df[col_loop].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col_loop])):
            df[col_loop] = pd.factorize(df[col_loop])[0]; print(f"    Colonna '{col_loop}' factorizzata.")
    for col_loop in df.select_dtypes(include='object').columns:
        if col_loop != target_col:
            df[col_loop] = pd.factorize(df[col_loop])[0]; print(f"    Colonna '{col_loop}' (object non target) factorizzata.")

    if target_col not in df.columns:
        print(f"ERRORE: Target '{target_col}' non in {dataset_file}. Salto.")
        continue

    X_processed_dataset = df.drop(columns=[target_col])
    y_dataset = df[target_col]
    if y_dataset.dtype == 'object' or pd.api.types.is_categorical_dtype(y_dataset):
        y_dataset, _ = pd.factorize(y_dataset); print(f"    Target '{target_col}' factorizzato.")
    else: # Assicurati che sia numerico, potrebbe essere bool
        y_dataset = y_dataset.astype(int)


    numeric_cols_in_X = X_processed_dataset.select_dtypes(include=np.number).columns
    if not X_processed_dataset.empty and len(numeric_cols_in_X) > 0:
        print(f"    Imputazione NaN su {len(numeric_cols_in_X)} colonne numeriche.")
        imputer = SimpleImputer(strategy='mean')
        X_processed_dataset[numeric_cols_in_X] = imputer.fit_transform(X_processed_dataset[numeric_cols_in_X])
        if X_processed_dataset[numeric_cols_in_X].isnull().sum().sum() > 0: print(f"    ATTENZIONE: NaN ancora in X DOPO imputazione!")
    elif X_processed_dataset.empty:
        print(f"ATTENZIONE: X vuoto. Salto."); continue

    final_non_numeric_cols = X_processed_dataset.select_dtypes(exclude=np.number).columns
    if not final_non_numeric_cols.empty:
        print(f"ERRORE CRITICO: Colonne non numeriche in X: {list(final_non_numeric_cols)}. Salto."); continue

    n_splits_weka_adjusted = min(N_SPLITS_WEKA_CV, y_dataset.value_counts().min()) if y_dataset.nunique() > 1 else N_SPLITS_WEKA_CV
    if len(X_processed_dataset) < n_splits_weka_adjusted * 2 or n_splits_weka_adjusted < 2 :
        print(f"ERRORE: Dati insuff. per {n_splits_weka_adjusted}-Fold CV. Salto."); continue
    print("  Preprocessing completato.")

    # --- OTTIMIZZAZIONE IPERPARAMETRI (come nel tuo script originale) ---
    print("\n  Ottimizzazione iperparametri preliminare...")
    optimized_estimators_dataset = {}
    tuning_successful = False
    try:
        if y_dataset.nunique() < 2:
             X_train_tune, _, y_train_tune, _ = train_test_split(X_processed_dataset, y_dataset, test_size=0.2, random_state=dataset_idx)
        else:
             X_train_tune, _, y_train_tune, _ = train_test_split(X_processed_dataset, y_dataset, test_size=0.2, random_state=dataset_idx, stratify=y_dataset)
        
        current_param_grids_tune = get_param_grids(len(X_train_tune))
        n_splits_tune_cv = min(N_SPLITS_INNER_CV, y_train_tune.value_counts().min() if y_train_tune.nunique() > 1 else N_SPLITS_INNER_CV)
        if n_splits_tune_cv < 2: n_splits_tune_cv = 2

        for model_config in current_param_grids_tune:
            estimator_name = model_config['name']; estimator = model_config['estimator']; param_grid = model_config['params']
            print(f"    Tuning {estimator_name}...")
            
            current_n_splits_tune_cv = n_splits_tune_cv
            if y_train_tune.nunique() < 2:
                cv_tune_object = current_n_splits_tune_cv
            else:
                min_class_count_tune = y_train_tune.value_counts().min()
                if min_class_count_tune < current_n_splits_tune_cv:
                    current_n_splits_tune_cv = max(2, min_class_count_tune)
                cv_tune_object = StratifiedKFold(n_splits=current_n_splits_tune_cv, shuffle=True, random_state=dataset_idx+1) if current_n_splits_tune_cv >=2 else 2


            gs_tune = GridSearchCV(estimator, param_grid, cv=cv_tune_object, scoring='accuracy', n_jobs=-1, error_score=np.nan)
            gs_tune.fit(X_train_tune, y_train_tune)
            if not pd.isna(gs_tune.best_score_):
                optimized_estimators_dataset[estimator_name] = gs_tune.best_estimator_
                print(f"      Best Params: {gs_tune.best_params_} (Score: {gs_tune.best_score_:.4f})")
                tuning_successful = True
            else:
                print(f"      GridSearchCV di tuning fallito per {estimator_name}. Uso istanza base.")
                optimized_estimators_dataset[estimator_name] = clone(estimator)
                tuning_successful = True # Consideriamo successo anche se usa il default
    except Exception as e_tune:
        print(f"ERRORE durante l'ottimizzazione preliminare: {e_tune}. Salto dataset."); continue

    if not tuning_successful or not optimized_estimators_dataset:
        print(f"ERRORE: Nessun modello disponibile dopo ottimizzazione per {dataset_name}. Salto."); continue

    # --- WEKA-STYLE REPEATED CROSS-VALIDATION (10x10) in Python ---
    print(f"\n  Avvio Repeated Cross-Validation ({NUM_TRIALS_WEKA} trials x {n_splits_weka_adjusted} folds)...")
    model_names_to_run_cv = list(optimized_estimators_dataset.keys())

    for trial in range(NUM_TRIALS_WEKA):
        print(f"    Trial {trial + 1}/{NUM_TRIALS_WEKA}...")
        
        current_n_splits_weka = n_splits_weka_adjusted
        skf_weka_cv_obj = None
        if y_dataset.nunique() < 2:
            # Impossibile usare StratifiedKFold. KFold potrebbe non essere appropriato per la valutazione.
            # Per il Corrected Resampled T-test, è cruciale avere fold validi.
            # Se il dataset non è stratificabile, potremmo dover saltare o gestire diversamente.
            print(f"    ATTENZIONE: Target con una sola classe ({y_dataset.unique()}). Impossibile stratificare per la CV principale. Potrebbe portare a problemi. Tento KFold.")
            from sklearn.model_selection import KFold # Importa solo se necessario
            skf_weka_cv_obj = KFold(n_splits=current_n_splits_weka, shuffle=True, random_state=(dataset_idx * 100 + trial))
        else:
            min_class_count_main_cv = y_dataset.value_counts().min()
            if min_class_count_main_cv < current_n_splits_weka:
                 current_n_splits_weka = max(2, min_class_count_main_cv)
            if current_n_splits_weka < 2:
                print(f"    ERRORE: Non abbastanza campioni per CV principale ({current_n_splits_weka} splits) anche dopo aggiustamento. Salto trial.")
                for model_name in model_names_to_run_cv:
                    for fold_num_fill in range(n_splits_weka_adjusted): # Usa n_splits_weka_adjusted per coerenza
                         all_fold_results_for_weka.append({
                            'DatasetName': dataset_name,
                            'AlgorithmName': model_name,
                            'AlgorithmParameters': str(optimized_estimators_dataset[model_name].get_params() if model_name in optimized_estimators_dataset else "N/A"),
                            'Trial': trial + 1,
                            'Fold': fold_num_fill + 1,
                            'Accuracy': np.nan
                        })
                continue # Salta al prossimo trial
            skf_weka_cv_obj = StratifiedKFold(n_splits=current_n_splits_weka, shuffle=True, random_state=(dataset_idx * 100 + trial))
        
        fold_idx = 0
        try:
            for train_idx, test_idx in skf_weka_cv_obj.split(X_processed_dataset, y_dataset):
                X_train_cv, X_test_cv = X_processed_dataset.iloc[train_idx], X_processed_dataset.iloc[test_idx]
                y_train_cv, y_test_cv = y_dataset.iloc[train_idx], y_dataset.iloc[test_idx]

                if len(X_train_cv) == 0 or len(X_test_cv) == 0:
                    print(f"      Fold {fold_idx+1} vuoto. Inserimento NaN.")
                    for model_name in model_names_to_run_cv:
                        all_fold_results_for_weka.append({
                            'DatasetName': dataset_name,
                            'AlgorithmName': model_name,
                            'AlgorithmParameters': str(optimized_estimators_dataset[model_name].get_params() if model_name in optimized_estimators_dataset else "N/A"),
                            'Trial': trial + 1,
                            'Fold': fold_idx + 1,
                            'Accuracy': np.nan
                        })
                    fold_idx += 1
                    continue

                for model_name in model_names_to_run_cv:
                    estimator_instance = clone(optimized_estimators_dataset[model_name])
                    score_val = np.nan
                    try:
                        estimator_instance.fit(X_train_cv, y_train_cv)
                        score_val = accuracy_score(y_test_cv, estimator_instance.predict(X_test_cv))
                    except Exception as e_fit_score:
                        print(f"      Errore fit/score {model_name} fold {fold_idx+1}: {e_fit_score}")
                    
                    all_fold_results_for_weka.append({
                        'DatasetName': dataset_name,
                        'AlgorithmName': model_name,
                        'AlgorithmParameters': str(optimized_estimators_dataset[model_name].get_params()), # Salva i parametri
                        'Trial': trial + 1,
                        'Fold': fold_idx + 1, # Fold da 1 a N
                        'Accuracy': score_val
                    })
                fold_idx += 1
            
            # Se current_n_splits_weka < n_splits_weka_adjusted, riempi i fold mancanti con NaN per questo trial
            if fold_idx < n_splits_weka_adjusted:
                for model_name in model_names_to_run_cv:
                    for missing_fold_num in range(fold_idx, n_splits_weka_adjusted):
                        all_fold_results_for_weka.append({
                            'DatasetName': dataset_name,
                            'AlgorithmName': model_name,
                            'AlgorithmParameters': str(optimized_estimators_dataset[model_name].get_params() if model_name in optimized_estimators_dataset else "N/A"),
                            'Trial': trial + 1,
                            'Fold': missing_fold_num + 1,
                            'Accuracy': np.nan
                        })

        except ValueError as ve_skf: # Errore da split
            print(f"    ERRORE durante lo split CV (Trial {trial+1}): {ve_skf}. Riempio gli score del trial con NaN.")
            for model_name in model_names_to_run_cv:
                for fold_num_fill_err in range(n_splits_weka_adjusted): # Usa n_splits_weka_adjusted per coerenza
                    all_fold_results_for_weka.append({
                        'DatasetName': dataset_name,
                        'AlgorithmName': model_name,
                        'AlgorithmParameters': str(optimized_estimators_dataset[model_name].get_params() if model_name in optimized_estimators_dataset else "N/A"),
                        'Trial': trial + 1,
                        'Fold': fold_num_fill_err + 1,
                        'Accuracy': np.nan
                    })
    processed_datasets_count +=1


# --- FINE CICLO DATASET: SALVATAGGIO RISULTATI GLOBALI PER WEKA ---
if not all_fold_results_for_weka:
    print("\n\nERRORE CRITICO: Nessun risultato di fold raccolto. Impossibile generare file per WEKA.")
else:
    print("\n\n" + "="*70 + "\n PREPARAZIONE FILE RISULTATI PER WEKA ANALYSER \n" + "="*70)
    results_df_for_weka = pd.DataFrame(all_fold_results_for_weka)
    
    # Assicurati che la colonna Accuracy sia numerica, convertendo errori in NaN
    results_df_for_weka['Accuracy'] = pd.to_numeric(results_df_for_weka['Accuracy'], errors='coerce')

    output_arff_filename = "python_cv_results_for_weka_analysis.arff"
    
    # Usiamo la funzione pandas_df_to_arff_file, specificando che è un file di risultati
    # Non c'è una singola "target_col_name" per il training qui, WEKA userà "Accuracy" per il confronto.
    pandas_df_to_arff_file(results_df_for_weka.copy(), output_arff_filename, is_result_file=True)

    print(f"\nFile ARFF '{output_arff_filename}' generato con i risultati di tutti i fold.")
    print("Questo file contiene le seguenti colonne:")
    for col in results_df_for_weka.columns:
        print(f"  - {col}")
    
    print("\n--- ISTRUZIONI PER WEKA EXPERIMENTER (Scheda 'Analyse') ---")
    print(f"1. Apri WEKA Experimenter e vai alla scheda 'Analyse'.")
    print(f"2. Clicca sul pulsante 'File...' e carica il file '{output_arff_filename}'.")
    print(f"3. Configura il test (ad esempio, 'Paired T-Tester (corrected)'):")
    print(f"   - Comparison field: Seleziona 'Accuracy' (o la tua metrica).")
    print(f"   - Row field (Base): Potresti non averne bisogno o selezionare 'DatasetName' se vuoi confrontare i risultati *per dataset*.")
    print(f"   - Column field (Comparison): Seleziona 'AlgorithmName'.")
    print(f"   - Run field: Seleziona 'Trial'.")
    print(f"   - Fold field: Seleziona 'Fold'.")
    print(f"   - Dataset field (per raggruppare i test): Seleziona 'DatasetName'. Questo farà sì che WEKA esegua i test separatamente per ogni dataset.")
    print(f"4. Imposta 'Significance' (es. 0.05).")
    print(f"5. Clicca 'Perform test'.")
    print(f"\n   Nota: La configurazione esatta dei campi 'Row', 'Column', 'Dataset field' etc.")
    print(f"   dipende da come vuoi visualizzare e raggruppare i test. L'esempio sopra è comune.")

if processed_datasets_count == 0:
    print("\n\nNessun dataset processato con successo.")

# --- ARRESTO JVM ---
try:
    if jvm.started:
        jvm.stop()
        print("\nJVM arrestata.")
except Exception as e:
    print(f"ATTENZIONE: Problema durante l'arresto della JVM: {e}")

print("\nEsecuzione completata.")

