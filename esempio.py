import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_rel

# Numero di prove casuali
NUM_TRIALS = 3

# Caricamento del dataset (assumiamo che la colonna target si chiami "Outcome")
df = pd.read_csv("diabetes_dataset.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Definizione dei parametri da ottimizzare per ciascun modello
param_grid_dt = {
    "max_depth": [None, 3, 5, 7, 10],
    "min_samples_split": [2, 5, 10]
}

param_grid_rf = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 3, 5, 7, 10],
    "min_samples_split": [2, 5, 10]
}

# Array per salvare i punteggi ottenuti con la Nested CV
nested_scores_dt = np.zeros(NUM_TRIALS)
nested_scores_rf = np.zeros(NUM_TRIALS)

# Ciclo sui NUM_TRIALS per avere una stima robusta delle performance
for i in range(NUM_TRIALS):
    # Definizione di inner e outer CV con stratificazione
    inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
    
    # ----- Decision Tree -----
    clf_dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                          param_grid=param_grid_dt,
                          cv=inner_cv,
                          verbose=2)
    # Nested CV: ottimizzazione con inner CV e valutazione con outer CV
    nested_score_dt = cross_val_score(clf_dt, X, y, cv=outer_cv)
    nested_scores_dt[i] = nested_score_dt.mean()
    
    # ----- Random Forest -----
    clf_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                          param_grid=param_grid_rf,
                          cv=inner_cv,
                          verbose=2)
    nested_score_rf = cross_val_score(clf_rf, X, y, cv=outer_cv)
    nested_scores_rf[i] = nested_score_rf.mean()

# Visualizzazione dei punteggi medi (Nested CV) per entrambi i modelli
print("Decision Tree - Nested CV average score: {:.6f}".format(nested_scores_dt.mean()))
print("Random Forest - Nested CV average score:   {:.6f}".format(nested_scores_rf.mean()))

# Confronto statistico: t-test appaiato sui punteggi di Nested CV dei due modelli
t_stat, p_value = ttest_rel(nested_scores_dt, nested_scores_rf)
print("\nPaired t-test between Decision Tree and Random Forest (Nested CV scores):")
print("t-statistic: {:.6f}, p-value: {:.6f}".format(t_stat, p_value))
if p_value < 0.05:
    print("La differenza di performance è statisticamente significativa (p < 0.05).")
else:
    print("La differenza di performance NON è statisticamente significativa (p >= 0.05).")

# Plot dei punteggi Nested CV per ogni t
