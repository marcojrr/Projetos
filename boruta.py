import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from scipy.stats import binom

def boruta_feature_selection(df, target_column, test_size=0.3, n_estimators=100, 
                            alpha=0.05, random_state=None, use_boruta_py=True):
    """
    Função completa para seleção de features com Boruta incluindo divisão de dados.
    
    Parâmetros:
    -----------
    df : pandas DataFrame
        DataFrame contendo todas as features e a variável target
    target_column : str
        Nome da coluna que contém a variável resposta/target
    test_size : float, opcional (default=0.3)
        Proporção do dataset a ser usada como teste
    n_estimators : int, opcional (default=100)
        Número de árvores na Random Forest
    alpha : float, opcional (default=0.05)
        Nível de significância para teste estatístico
    random_state : int, opcional (default=None)
        Seed para reprodutibilidade
    use_boruta_py : bool, opcional (default=True)
        Se True, usa a implementação do BorutaPy (recomendado)
        Se False, usa nossa implementação simplificada
        
    Retorna:
    --------
    result : dict
        Dicionário contendo:
        - 'X_train': Features de treino (apenas as selecionadas)
        - 'X_test': Features de teste (apenas as selecionadas)
        - 'y_train': Target de treino
        - 'y_test': Target de teste
        - 'selected_features': Nomes das features selecionadas
        - 'selected_indices': Índices das features selecionadas
    """
    
    # 1. Preparar os dados
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    feature_names = df.drop(columns=[target_column]).columns.tolist()
    
    # 2. Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # 3. Aplicar Boruta apenas nos dados de treino
    if use_boruta_py:
        # Usando a implementação do BorutaPy (recomendada)
        rf = RandomForestClassifier(n_estimators=n_estimators, 
                                  random_state=random_state,
                                  n_jobs=-1)
        
        boruta_selector = BorutaPy(
            estimator=rf,
            n_estimators='auto',
            alpha=alpha,
            max_iter=100,
            random_state=random_state,
            verbose=0
        )
        
        boruta_selector.fit(X_train, y_train)
        
        # Obter máscara de features selecionadas
        selected_mask = boruta_selector.support_
    else:
        # Usando nossa implementação simplificada
        selected_indices = simplified_boruta(
            X_train, y_train, 
            n_estimators=n_estimators, 
            alpha=alpha, 
            random_state=random_state
        )
        # Criar máscara de seleção
        selected_mask = np.zeros(X.shape[1], dtype=bool)
        selected_mask[selected_indices] = True
    
    # 4. Filtrar features selecionadas
    selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
    selected_indices = np.where(selected_mask)[0]
    
    # Aplicar a seleção aos conjuntos de treino e teste
    X_train_selected = X_train[:, selected_mask]
    X_test_selected = X_test[:, selected_mask]
    
    # 5. Retornar resultados em um dicionário organizado
    return {
        'X_train': X_train_selected,
        'X_test': X_test_selected,
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': selected_features,
        'selected_indices': selected_indices.tolist(),
        'feature_names': feature_names
    }

def simplified_boruta(X, y, n_estimators=100, max_iter=100, alpha=0.05, random_state=None):
    """
    Implementação simplificada do algoritmo Boruta (função auxiliar)
    """
    X, y = check_X_y(X, y)
    n_samples, n_features = X.shape
    
    rf = RandomForestClassifier(n_estimators=n_estimators, 
                               max_depth=5, 
                               random_state=random_state)
    
    hit_history = np.zeros((n_features, max_iter))
    decision = np.zeros(n_features)
    
    for iter_ in range(max_iter):
        np.random.seed(iter_ if random_state is None else random_state + iter_)
        X_shadow = np.random.permutation(X.T).T
        
        X_combined = np.hstack((X, X_shadow))
        rf.fit(X_combined, y)
        
        importances = rf.feature_importances_
        imp_real = importances[:n_features]
        imp_shadow = importances[n_features:]
        max_shadow = imp_shadow.max()
        
        hit_history[:, iter_] = (imp_real > max_shadow)
        
        for feature in range(n_features):
            if decision[feature] == 0:
                hits = hit_history[feature, :iter_+1].sum()
                p_value = 1 - binom.cdf(hits - 1, iter_ + 1, 0.5)
                
                if p_value < alpha:
                    decision[feature] = 1
    
    return np.where(decision == 1)[0]
