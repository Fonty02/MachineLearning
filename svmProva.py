from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Custom import StabilityAdaptiveKNN, EntropyAdaptiveKNN, LocalLOOCVAdaptiveKNN
from sklearn.neighbors import KNeighborsClassifier as KN
import pandas as pd

# Genera dati
dataset= pd.read_csv('marketing_campaign.csv', sep="\t")
print(dataset.head())

#remoove column with name="ID" 
dataset = dataset.drop(['ID','Dt_Customer'], axis=1)

#replace Education and Marital_Status with numbers usign pd.factorize
dataset['Education'] = pd.factorize(dataset['Education'])[0]
dataset['Marital_Status'] = pd.factorize(dataset['Marital_Status'])[0]
dataset= dataset.dropna()
dataset= dataset.drop_duplicates()     

X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Definisci griglie di parametri per ogni classe
param_grids = [
    {
        'estimator': [StabilityAdaptiveKNN()],
        'estimator__stability_patience': [i for i in range(1, 10)], # Ridotto per velocità
        # --- MODIFICA QUI ---
        'estimator__k_step_stability': [i for i in range(1, 10)], # Devono essere interi!
        # --- FINE MODIFICA ---
        'estimator__max_k_adaptive': [10, 20, 30], # Aggiunto None per più flessibilità
        'estimator__weights': ['uniform', 'distance']
    },
    {
        'estimator': [EntropyAdaptiveKNN()],
        'estimator__entropy_threshold': [0.05, 0.15, 0.5, 0.75, 1.0],
        'estimator__entropy_patience': [1, 2],
        'estimator__min_entropy_decrease': [0.005, 0.015, 0.05, 0.075, 0.1],
        'estimator__k_step_entropy': [i for i in range(1, 5)], # Devono essere interi!
        # --- FINE MODIFICA ---
        'estimator__max_k_adaptive': [10, 20, 30],
        'estimator__weights': ['uniform', 'distance']
    },
    {
        'estimator':[KN()],
        'estimator__n_neighbors': [i for i in range(1, 21,2)],
        'estimator__weights': ['uniform', 'distance'],
        'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'estimator__leaf_size': [30, 50, 70],
        'estimator__metric': ['euclidean', 'manhattan', 'minkowski']
    }
]

# Per GridSearchCV con diversi estimator, è più semplice fare loop
best_score = -1
best_params_overall = None
best_estimator_overall = None

for i, grid_config in enumerate(param_grids):
    estimator_type = grid_config['estimator'][0].__class__.__name__
    print(f"\nAvvio GridSearchCV per {estimator_type}...")
    
    # Rimuovi 'estimator' dalla griglia passata a GridSearchCV
    current_param_grid = {key.replace('estimator__', ''): val 
                          for key, val in grid_config.items() if key != 'estimator'}

    grid_search = GridSearchCV(estimator=grid_config['estimator'][0],
                               param_grid=current_param_grid,
                               cv=2, # CV più piccolo per test rapidi
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=1)
    try:
        grid_search.fit(X_train, y_train)
        print(f"Migliori parametri per {estimator_type}:")
        print(grid_search.best_params_)
        print(f"Migliore score CV ({estimator_type}): {grid_search.best_score_:.4f}")

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_params_overall = grid_search.best_params_
            best_estimator_overall = grid_search.best_estimator_ # Questo è già fittato

    except Exception as e:
        print(f"Errore durante GridSearchCV per {estimator_type}: {e}")


if best_estimator_overall:
    print("\n--- Risultato Complessivo Migliore ---")
    print(f"Miglior Estimator: {best_estimator_overall.__class__.__name__}")
    print(f"Migliori Parametri: {best_params_overall}") # Questi sono i parametri senza 'estimator__'
    print(f"Migliore Score CV Globale: {best_score:.4f}")
    
    y_pred_test = best_estimator_overall.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Accuratezza Test del miglior modello globale: {accuracy_test:.4f}")
else:
    print("Nessun estimator è stato testato con successo.")