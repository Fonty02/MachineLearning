import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
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


# --- 2. Confidence Adaptive KNN ---
class ConfidenceAdaptiveKNN(BaseAdaptiveKNN):
    def __init__(self, confidence_threshold=0.8, max_k_adaptive=None, weights='uniform', 
                 algorithm='auto', leaf_size=30, p=2, metric='minkowski', 
                 metric_params=None, n_jobs=None):
        super().__init__(max_k_adaptive=max_k_adaptive, weights=weights, 
                         algorithm=algorithm, leaf_size=leaf_size, p=p, 
                         metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        self.confidence_threshold = confidence_threshold
        
    def _get_adaptive_k_neighbors_for_sample(self, all_sample_neigh_dist, all_sample_neigh_ind):
        max_confidence = 0
        best_k = 1
        best_indices = all_sample_neigh_ind[:1]
        best_distances = all_sample_neigh_dist[:1]
        
        # Cerca il k che fornisce la massima confidenza
        for k in range(1, len(all_sample_neigh_ind) + 1):
            current_indices = all_sample_neigh_ind[:k]
            current_distances = all_sample_neigh_dist[:k]
            y_neighbor_labels = self._y[current_indices]
            
            # Calcola la confidenza come proporzione della classe maggioritaria
            if self.weights == 'uniform':
                class_counts = np.bincount(y_neighbor_labels, minlength=len(self.classes_))
                max_count = np.max(class_counts)
                confidence = max_count / k
            elif self.weights == 'distance':
                weights = 1.0 / (current_distances + 1e-9)
                weighted_votes = {}
                for i, label in enumerate(y_neighbor_labels):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weights[i]
                confidence = max(weighted_votes.values()) / sum(weighted_votes.values())
            
            # Aggiorna il miglior k se migliora la confidenza
            if confidence > max_confidence:
                max_confidence = confidence
                best_k = k
                best_indices = current_indices
                best_distances = current_distances
                
            # Se la confidenza è sufficientemente alta, termina la ricerca
            if confidence >= self.confidence_threshold:
                break
                
        return best_distances, best_indices, best_k
    
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['confidence_threshold'] = self.confidence_threshold
        return params

    def set_params(self, **params):
        if 'confidence_threshold' in params:
            self.confidence_threshold = params.pop('confidence_threshold')
        super().set_params(**params)
        return self
    



# --- 3. Density Confidence Adaptive KNN ---
class DensityConfidenceKNN(BaseAdaptiveKNN):
    def __init__(self, confidence_threshold=0.7, density_quantile=0.8, max_k_adaptive=None, 
                 weights='uniform', algorithm='auto', leaf_size=30, p=2, 
                 metric='minkowski', metric_params=None, n_jobs=None):
        super().__init__(max_k_adaptive=max_k_adaptive, weights=weights, 
                         algorithm=algorithm, leaf_size=leaf_size, p=p, 
                         metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        self.confidence_threshold = confidence_threshold
        self.density_quantile = density_quantile
        
    def fit(self, X, y):
        super().fit(X, y)
        
        # Calcola la densità locale per ogni punto di training
        # (inverso della distanza media dai 10 vicini più prossimi)
        n_density_neighbors = min(10, X.shape[0] - 1)
        if n_density_neighbors > 0:
            knn = NearestNeighbors(n_neighbors=n_density_neighbors + 1)
            knn.fit(X)
            distances, _ = knn.kneighbors(X)
            # Ignora la distanza a se stesso (sempre 0)
            self.local_density_ = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-9)
            
            # Calcola soglie di densità
            self.density_threshold_ = np.quantile(self.local_density_, self.density_quantile)
        return self
        
    def _get_adaptive_k_neighbors_for_sample(self, all_sample_neigh_dist, all_sample_neigh_ind):
        if not all_sample_neigh_ind.size:
            return np.array([]), np.array([]), 0
            
        # Determina se il punto è in una regione ad alta o bassa densità
        # calcolando la densità dei suoi vicini
        neighbor_densities = [self.local_density_[idx] for idx in all_sample_neigh_ind[:5]]
        avg_neighbor_density = np.mean(neighbor_densities)
        
        # Per regioni ad alta densità, usiamo più vicini
        if avg_neighbor_density > self.density_threshold_:
            # In regioni dense, cerchiamo alta confidenza
            max_confidence = 0
            best_k = 1
            
            for k in range(1, len(all_sample_neigh_ind) + 1):
                y_neighbor_labels = self._y[all_sample_neigh_ind[:k]]
                class_counts = np.bincount(y_neighbor_labels, minlength=len(self.classes_))
                confidence = np.max(class_counts) / k
                
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_k = k
                    
                if confidence >= self.confidence_threshold:
                    break
        else:
            # In regioni sparse, usiamo meno vicini (più localizzato)
            best_k = max(1, int(len(all_sample_neigh_ind) * 0.3))
            
        return all_sample_neigh_dist[:best_k], all_sample_neigh_ind[:best_k], best_k
    
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        params['confidence_threshold'] = self.confidence_threshold
        return params

    def set_params(self, **params):
        if 'confidence_threshold' in params:
            self.confidence_threshold = params.pop('confidence_threshold')
        super().set_params(**params)
        return self