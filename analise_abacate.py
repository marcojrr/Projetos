from typing import Dict, List, Union
import pyspark.sql.dataframe as psd

def cria_dict_splits(df: psd.DataFrame, lista_colunas: List[str]) -> Dict[str, List[Union[float, str]]]:
    """
    Cria um dicionário de splits para features numéricas baseado em percentis.
    
    Args:
        df: DataFrame do PySpark
        lista_colunas: Lista de colunas para analisar
        
    Returns:
        Dicionário onde as chaves são nomes de colunas e os valores são listas
        de splits baseados nos percentis calculados.
    """
    # Filtra colunas numéricas uma única vez
    colunas_num = [
        col for col in lista_colunas 
        if dict(df.dtypes)[col] in ("double", "float", "int", "bigint", "decimal")
    ]
    
    if not colunas_num:
        return {}
    
    # Calcula todos os percentis de uma vez
    percentis = df.approxQuantile(
        colunas_num, 
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
        0.01
    )
    
    # Constrói o dicionário de splits
    dict_splits = {
        feature: [-float('inf')] + sorted(set(percentis[i])) + [float('inf')]
        for i, feature in enumerate(colunas_num)
    }
    
    # Ajusta splits com apenas um valor (além dos infinitos)
    for splits in dict_splits.values():
        if len(splits) == 3:
            splits[1] += 0.001  # Forma mais concisa de incremento
    
    return dict_splits
