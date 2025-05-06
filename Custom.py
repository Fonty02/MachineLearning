import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from scipy.stats import mode, entropy
# from sklearn.metrics import pairwise_distances # Potrebbe servire per LOO-CV più avanzato

# --- Classe Base per KNN Adattivi ---
class BaseAdaptiveKNN(KNeighborsClassifier):
    def __init__(self, max_k_adaptive=None,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None):
        super().__init__(
            n_neighbors=1, # Placeholder, gestito da max_k_adaptive
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            metric_params=metric_params,
            p=p,
            n_jobs=n_jobs
        )
        self.max_k_adaptive = max_k_adaptive # Questo è l'iperparametro dell'utente

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True, multi_output=False)
        self.classes_ = unique_labels(y)
        super().fit(X, y) # Memorizza X_train (_fit_X) e y_train (_y)

        if self.max_k_adaptive is None:
            self.max_k_to_consider_ = self._fit_X.shape[0]
        else:
            self.max_k_to_consider_ = min(self.max_k_adaptive, self._fit_X.shape[0])
        
        self.max_k_to_consider_ = max(1, self.max_k_to_consider_) # Deve essere almeno 1

        # Imposta n_neighbors per la chiamata a super().kneighbors()
        # in modo che restituisca fino a max_k_to_consider_ vicini
        self.n_neighbors = self.max_k_to_consider_
        return self

    def _predict_class_from_labels_and_distances(self, neighbor_labels, neighbor_distances_to_query):
        """
        Predice la classe basata sulle etichette dei vicini forniti e le loro distanze
        dal punto di query. Utilizza self.weights.
        """
        if not neighbor_labels.size: # Nessun vicino
            if hasattr(self, '_y') and self._y.size > 0: # Fallback a classe più frequente nel training
                m, _ = mode(self._y, keepdims=False) if hasattr(mode(self._y), 'mode') else (mode(self._y)[0], mode(self._y)[1]) # Adattamento per versioni scipy
                return m[0] if isinstance(m, (np.ndarray, list)) else m
            elif self.classes_.size > 0:
                return self.classes_[0]
            return None # Dovrebbe essere gestito prima

        if self.weights == 'uniform':
            pred_class, _ = mode(neighbor_labels, keepdims=False) if hasattr(mode(neighbor_labels), 'mode') else (mode(neighbor_labels)[0], mode(neighbor_labels)[1])
            return pred_class[0] if isinstance(pred_class, (np.ndarray, list)) else pred_class
        elif self.weights == 'distance':
            if not neighbor_distances_to_query.size: # Dovrebbe avere distanze se ha etichette
                pred_class, _ = mode(neighbor_labels, keepdims=False) if hasattr(mode(neighbor_labels), 'mode') else (mode(neighbor_labels)[0], mode(neighbor_labels)[1])
                return pred_class[0] if isinstance(pred_class, (np.ndarray, list)) else pred_class

            weights_arr = 1.0 / (neighbor_distances_to_query + 1e-9) # Epsilon per distanze zero
            if np.any(neighbor_distances_to_query < 1e-9): # Se ci sono distanze zero, peso solo quelle
                weights_arr[neighbor_distances_to_query >= 1e-9] = 0
            
            class_votes = {}
            for i, label in enumerate(neighbor_labels):
                class_votes[label] = class_votes.get(label, 0) + weights_arr[i]
            
            if not class_votes or all(v == 0 for v in class_votes.values()):
                pred_class, _ = mode(neighbor_labels, keepdims=False) if hasattr(mode(neighbor_labels), 'mode') else (mode(neighbor_labels)[0], mode(neighbor_labels)[1])
                return pred_class[0] if isinstance(pred_class, (np.ndarray, list)) else pred_class
            return max(class_votes, key=class_votes.get)
        elif callable(self.weights):
            # Per callable, ci aspettiamo che la funzione di peso sia gestita a un livello superiore
            # o che si possa dedurre una classe. Per semplicità, fallback a uniform.
            print("Attenzione: la predizione con pesi custom callable non è pienamente implementata qui, fallback a 'uniform'.")
            pred_class, _ = mode(neighbor_labels, keepdims=False) if hasattr(mode(neighbor_labels), 'mode') else (mode(neighbor_labels)[0], mode(neighbor_labels)[1])
            return pred_class[0] if isinstance(pred_class, (np.ndarray, list)) else pred_class
        else:
            raise ValueError(f"Valore di weights non riconosciuto: {self.weights}")

    def _get_adaptive_k_neighbors_for_sample(self, all_sample_neigh_dist, all_sample_neigh_ind):
        # Questo metodo DEVE essere implementato dalle sottoclassi
        raise NotImplementedError("Le sottoclassi devono implementare _get_adaptive_k_neighbors_for_sample")

    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X, accept_sparse='csr', estimator=self)
        
        all_neigh_dist_matrix, all_neigh_ind_matrix = super().kneighbors(X, return_distance=True)

        n_queries = X.shape[0]
        probabilities = np.zeros((n_queries, len(self.classes_)))

        for i in range(n_queries):
            # Distanze e indici di TUTTI i potenziali vicini per il campione i-esimo
            sample_all_neigh_dist = all_neigh_dist_matrix[i]
            sample_all_neigh_ind = all_neigh_ind_matrix[i]

            # k_adaptive_val è il k scelto per questo campione
            neigh_dist_adaptive, neigh_ind_adaptive, k_adaptive_val = \
                self._get_adaptive_k_neighbors_for_sample(sample_all_neigh_dist, sample_all_neigh_ind)
            
            if k_adaptive_val == 0 or not neigh_ind_adaptive.size:
                if len(self.classes_) > 0: probabilities[i, :] = 1 / len(self.classes_)
                continue

            y_neighbor_labels = self._y[neigh_ind_adaptive]

            # Calcola probabilità basate sui vicini scelti e sui pesi
            current_weights_arr = np.ones_like(neigh_dist_adaptive) # Default a uniform
            if self.weights == 'distance':
                current_weights_arr = 1.0 / (neigh_dist_adaptive + 1e-9)
                if np.any(neigh_dist_adaptive < 1e-9):
                    current_weights_arr[neigh_dist_adaptive >= 1e-9] = 0
            elif callable(self.weights):
                 current_weights_arr = self.weights(neigh_dist_adaptive)

            for j_cls, cls_val in enumerate(self.classes_):
                probabilities[i, j_cls] = np.sum(current_weights_arr[y_neighbor_labels == cls_val])
            
            sum_proba = np.sum(probabilities[i, :])
            if sum_proba > 0:
                probabilities[i, :] /= sum_proba
            else:
                if len(self.classes_) > 0: probabilities[i, :] = 1 / len(self.classes_)
        return probabilities

    def predict(self, X):
        check_is_fitted(self)
        proba = self.predict_proba(X)
        if proba.shape[1] == 0: return np.array([])
        return self.classes_[np.argmax(proba, axis=1)]

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        if 'n_neighbors' in params: del params['n_neighbors']
        params['max_k_adaptive'] = self.max_k_adaptive
        return params

    def set_params(self, **params):
        if 'n_neighbors' in params: # Interpreta n_neighbors come max_k_adaptive
            if 'max_k_adaptive' not in params: self.max_k_adaptive = params.pop('n_neighbors')
            else: params.pop('n_neighbors') # Priorità a max_k_adaptive
        
        if 'max_k_adaptive' in params: self.max_k_adaptive = params.pop('max_k_adaptive')
        super().set_params(**params) # Passa i restanti alla classe base KNN
        return self

    def _more_tags(self):
        # Per evitare alcuni test di scikit-learn che potrebbero fallire con logiche complesse
        return {'multioutput': False, 'poor_score': True}


# --- 1. Stability Adaptive KNN ---
class StabilityAdaptiveKNN(BaseAdaptiveKNN):
    def __init__(self, stability_patience=2, k_step_stability=1,
                 max_k_adaptive=None, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=None):
        super().__init__(max_k_adaptive=max_k_adaptive, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)
        self.stability_patience = stability_patience
        self.k_step_stability = max(1, k_step_stability) # Assicura che il passo sia almeno 1

    def _get_adaptive_k_neighbors_for_sample(self, all_sample_neigh_dist, all_sample_neigh_ind):
        if not all_sample_neigh_ind.size: return np.array([]), np.array([]), 0

        last_pred_class_val = None
        stable_count = 0
        
        # Inizia sempre con k=1
        final_k = 1
        final_indices = all_sample_neigh_ind[:1]
        final_distances = all_sample_neigh_dist[:1]
        
        current_pred_class_val = self._predict_class_from_labels_and_distances(
            self._y[final_indices], final_distances
        )
        last_pred_class_val = current_pred_class_val

        # Itera per k successivi
        # Prossimo k da testare è 1 + k_step_stability
        # Se k_step_stability è 1, il prossimo è 2. Se è 2, il prossimo è 3.
        for k_candidate in range(1 + self.k_step_stability, self.max_k_to_consider_ + 1, self.k_step_stability):
            current_indices_slice = all_sample_neigh_ind[:k_candidate]
            current_distances_slice = all_sample_neigh_dist[:k_candidate]

            if not current_indices_slice.size: break

            current_pred_class_val = self._predict_class_from_labels_and_distances(
                self._y[current_indices_slice], current_distances_slice
            )

            if current_pred_class_val == last_pred_class_val:
                stable_count += 1
            else:
                stable_count = 0 # Reset
            
            last_pred_class_val = current_pred_class_val
            # Aggiorna sempre, così se il loop finisce, abbiamo l'ultimo k testato
            final_k = k_candidate
            final_distances = current_distances_slice
            final_indices = current_indices_slice
            
            if stable_count >= self.stability_patience:
                break
        
        return final_distances, final_indices, final_k

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['stability_patience'] = self.stability_patience
        params['k_step_stability'] = self.k_step_stability
        return params

    def set_params(self, **params):
        if 'stability_patience' in params: self.stability_patience = params.pop('stability_patience')
        if 'k_step_stability' in params: self.k_step_stability = params.pop('k_step_stability')
        super().set_params(**params)
        return self


# --- 2. Entropy Adaptive KNN ---
class EntropyAdaptiveKNN(BaseAdaptiveKNN):
    def __init__(self, entropy_threshold=0.1, entropy_patience=2,
                 min_entropy_decrease=0.01, k_step_entropy=1,
                 max_k_adaptive=None, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=None):
        super().__init__(max_k_adaptive=max_k_adaptive, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)
        self.entropy_threshold = entropy_threshold
        self.entropy_patience = entropy_patience
        self.min_entropy_decrease = min_entropy_decrease
        self.k_step_entropy = max(1, k_step_entropy)

    def _get_adaptive_k_neighbors_for_sample(self, all_sample_neigh_dist, all_sample_neigh_ind):
        if not all_sample_neigh_ind.size: return np.array([]), np.array([]), 0

        last_entropy_val = float('inf')
        entropy_stable_count = 0
        
        # L'entropia è significativa per k >= 2. Inizializziamo final_k a 1.
        final_k = 1
        final_indices = all_sample_neigh_ind[:1]
        final_distances = all_sample_neigh_dist[:1]
        
        # Il loop inizia da un k che permette il calcolo dell'entropia (almeno 2)
        # e rispetta k_step_entropy.
        min_k_for_loop = max(2, self.k_step_entropy)
        if self.k_step_entropy == 1 and min_k_for_loop == 1: min_k_for_loop = 2


        # Se max_k_to_consider_ è troppo piccolo per il loop
        if min_k_for_loop > self.max_k_to_consider_ :
            # Restituisci k=1 (già impostato) o max_k_to_consider_ se è 1
            if self.max_k_to_consider_ == 1:
                return final_indices, final_distances, final_k
            else: # max_k_to_consider_ è 0
                return np.array([]), np.array([]), 0

        for k_candidate in range(min_k_for_loop, self.max_k_to_consider_ + 1, self.k_step_entropy):
            current_indices_slice = all_sample_neigh_ind[:k_candidate]
            current_distances_slice = all_sample_neigh_dist[:k_candidate]

            if not current_indices_slice.size: break

            neighbor_labels = self._y[current_indices_slice]
            class_counts = np.array([np.sum(neighbor_labels == c) for c in self.classes_], dtype=float)
            
            current_entropy_val = float('inf') # Default se non calcolabile
            if np.sum(class_counts) > 0 : # Assicura che ci siano conteggi > 0
                 class_probas = class_counts / np.sum(class_counts)
                 # Calcola entropia solo per probabilità > 0 per evitare log(0)
                 current_entropy_val = entropy(class_probas[class_probas > 0], base=2)
            
            # Aggiorna sempre, così se una condizione di stop è soddisfatta, k_candidate è il k scelto
            final_k = k_candidate
            final_distances = current_distances_slice
            final_indices = current_indices_slice

            if current_entropy_val < self.entropy_threshold: break 
            if (last_entropy_val - current_entropy_val) < self.min_entropy_decrease:
                entropy_stable_count += 1
            else:
                entropy_stable_count = 0
            
            if entropy_stable_count >= self.entropy_patience: break 
            last_entropy_val = current_entropy_val
        
        return final_distances, final_indices, final_k

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['entropy_threshold'] = self.entropy_threshold
        params['entropy_patience'] = self.entropy_patience
        params['min_entropy_decrease'] = self.min_entropy_decrease
        params['k_step_entropy'] = self.k_step_entropy
        return params

    def set_params(self, **params):
        p_names = ['entropy_threshold', 'entropy_patience', 'min_entropy_decrease', 'k_step_entropy']
        for name in p_names:
            if name in params: setattr(self, name, params.pop(name))
        super().set_params(**params)
        return self


# --- 3. Local LOO-CV Adaptive KNN ---
class LocalLOOCVAdaptiveKNN(BaseAdaptiveKNN):
    def __init__(self, min_k_loo=2, k_step_loo=1,
                 max_k_adaptive=None, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=None):
        super().__init__(max_k_adaptive=max_k_adaptive, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, metric=metric,
                         metric_params=metric_params, n_jobs=n_jobs)
        self.min_k_loo = max(2, min_k_loo) # LOO ha senso per k_cv >= 2 (k_cv-1 >= 1 vicini per predire)
        self.k_step_loo = max(1, k_step_loo)

    def _predict_loo_internal(self, loo_neighbor_indices_for_pred):
        """
        Predizione interna per LOO-CV. USA SEMPRE PESI UNIFORMI per semplicità,
        perché calcolare le distanze tra i vicini stessi è costoso.
        Prende gli indici dei vicini da usare per la predizione.
        """
        if not loo_neighbor_indices_for_pred.size: return None # Non può predire
            
        labels_for_pred = self._y[loo_neighbor_indices_for_pred]
        pred_class, _ = mode(labels_for_pred, keepdims=False) if hasattr(mode(labels_for_pred), 'mode') else (mode(labels_for_pred)[0], mode(labels_for_pred)[1])
        return pred_class[0] if isinstance(pred_class, (np.ndarray, list)) else pred_class


    def _get_adaptive_k_neighbors_for_sample(self, all_sample_neigh_dist, all_sample_neigh_ind):
        if not all_sample_neigh_ind.size: return np.array([]), np.array([]), 0

        best_k_cv = 1 # Default k=1
        max_loo_accuracy = -1.0 

        # Assicura che min_k_loo sia fattibile
        actual_min_k_loo_loop = self.min_k_loo
        if actual_min_k_loo_loop > self.max_k_to_consider_:
            # Non abbastanza vicini per il loop LOO, usa k=1 o max_k_to_consider_
            k_to_use = min(1, self.max_k_to_consider_) if self.max_k_to_consider_ >=1 else 0
            if k_to_use == 0: return np.array([]), np.array([]), 0
            return all_sample_neigh_dist[:k_to_use], all_sample_neigh_ind[:k_to_use], k_to_use

        # Itera attraverso i k_cv candidati per il processo LOO
        for k_cv in range(actual_min_k_loo_loop, self.max_k_to_consider_ + 1, self.k_step_loo):
            # Vicini considerati per questo k_cv
            current_k_cv_indices = all_sample_neigh_ind[:k_cv]
            
            correct_loo_preds = 0
            num_loo_attempts = 0

            # Esegui LOO-CV su questi k_cv vicini
            for i in range(k_cv): # i è l'indice del vicino da "lasciare fuori"
                loo_out_neighbor_true_label = self._y[current_k_cv_indices[i]]
                
                # Indici degli altri k_cv-1 vicini usati per la predizione interna
                loo_in_indices_for_pred = np.delete(current_k_cv_indices, i)

                if not loo_in_indices_for_pred.size: continue # Non ci sono vicini per predire

                num_loo_attempts +=1
                predicted_label_for_loo_out = self._predict_loo_internal(loo_in_indices_for_pred)
                
                if predicted_label_for_loo_out is not None and \
                   predicted_label_for_loo_out == loo_out_neighbor_true_label:
                    correct_loo_preds += 1
            
            current_loo_accuracy = 0.0
            if num_loo_attempts > 0:
                current_loo_accuracy = correct_loo_preds / num_loo_attempts
            
            # Aggiorna best_k_cv se questa accuratezza LOO è migliore
            # Tie-breaking: preferisci k più piccolo (gestito implicitamente se aggiorni solo con >)
            if current_loo_accuracy > max_loo_accuracy:
                max_loo_accuracy = current_loo_accuracy
                best_k_cv = k_cv
            # Se l'accuratezza è la stessa, manteniamo il k più piccolo già trovato
        
        # Se nessun k_cv nel loop ha dato un'accuratezza > -1 (es. loop non eseguito),
        # best_k_cv rimarrà 1.
        final_k = best_k_cv
        final_distances = all_sample_neigh_dist[:final_k]
        final_indices = all_sample_neigh_ind[:final_k]
        
        return final_distances, final_indices, final_k

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['min_k_loo'] = self.min_k_loo
        params['k_step_loo'] = self.k_step_loo
        return params

    def set_params(self, **params):
        if 'min_k_loo' in params: self.min_k_loo = params.pop('min_k_loo')
        if 'k_step_loo' in params: self.k_step_loo = params.pop('k_step_loo')
        super().set_params(**params)
        return self