import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

def select_features_with_boruta(df, target_column, test_size=0.3, n_estimators=100,
                              alpha=0.05, random_state=None, verbose=True):
    """
    Seleção de features com Boruta retornando apenas as variáveis selecionadas.
    
    Parâmetros:
        df: DataFrame com features e target
        target_column: Nome da coluna alvo
        test_size: Proporção para teste (default: 0.3)
        n_estimators: Nº de árvores na Random Forest (default: 100)
        alpha: Limiar estatístico para seleção (default: 0.05)
        random_state: Seed para reprodutibilidade
        verbose: Se True, mostra logs de progresso (default: True)
    
    Retorna:
        Lista com os nomes das features selecionadas
    """
    # Separa features e target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    feature_names = df.drop(columns=[target_column]).columns.tolist()
    
    # Configura BorutaPy com logging
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    def log_progress(boruta):
        if verbose:
            confirmed = sum(boruta.support_)
            tentative = sum(boruta.support_weak_)
            rejected = len(boruta.support_) - confirmed - tentative
            print(f"Iteração {boruta.n_iter_}: Confirmadas={confirmed} | "
                  f"Tentativas={tentative} | Rejeitadas={rejected}")

    boruta_selector = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        alpha=alpha,
        max_iter=50,
        random_state=random_state,
        verbose=0,
        callback=log_progress
    )
    
    # Executa a seleção (usando todos os dados para seleção de features)
    boruta_selector.fit(X, y)
    
    # Retorna apenas os nomes das features selecionadas
    selected_features = [feature_names[i] 
                        for i in range(len(feature_names)) 
                        if boruta_selector.support_[i]]
    
    return selected_features
