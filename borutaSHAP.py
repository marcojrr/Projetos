import pandas as pd
from lightgbm import LGBMClassifier
from BorutaShap import BorutaShap

def selecionar_features_borutashap(df, 
                                    target_col, 
                                    modelo_base=None, 
                                    n_trials=100, 
                                    normalize=True, 
                                    plotar=True, 
                                    verbose=True):
    """
    Aplica BorutaSHAP para seleção de variáveis usando SHAP + modelo base (default: LGBM).
    
    Parâmetros:
    - df: DataFrame contendo variáveis e target.
    - target_col: nome da coluna alvo (target).
    - modelo_base: modelo usado para calcular SHAP (default: LGBMClassifier).
    - n_trials: número de iterações Boruta (default: 100).
    - normalize: normaliza valores SHAP (default: True).
    - plotar: plota gráfico com importâncias (default: True).
    - verbose: imprime as features selecionadas (default: True).

    Retorna:
    - Lista com os nomes das features selecionadas.
    """
    
    # 1. Separar X e y com base no nome da variável alvo
    X = df.drop(columns=target_col)
    y = df[target_col]

    # 2. Definir modelo base se não for passado
    if modelo_base is None:
        modelo_base = LGBMClassifier(random_state=42)
    
    # 3. Inicializar BorutaSHAP
    selector = BorutaShap(model=modelo_base,
                          importance_measure='shap',
                          classification=True)

    # 4. Aplicar seleção
    selector.fit(X=X, y=y,
                 n_trials=n_trials,
                 sample=False,
                 train_or_test='train',
                 normalize=normalize)

    # 5. Plotar gráfico se solicitado
    if plotar:
        selector.plot(importances_type='shap')

    # 6. Obter features selecionadas
    selected_features = selector.Subset().columns.tolist()
    
    if verbose:
        print(f"✅ {len(selected_features)} features selecionadas:")
        print(selected_features)
    
    return selected_features
