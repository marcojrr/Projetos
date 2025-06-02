from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, isnull, lit

def calculate_column_stats(df, columns=None):
    """
    Calcula para cada coluna:
    - Valor mais frequente e sua porcentagem
    - Contagem de valores nulos
    - Porcentagem de valores nulos
    
    Args:
        df: DataFrame do PySpark
        columns: Lista de colunas para analisar (None para todas as colunas)
    
    Returns:
        DataFrame com estatísticas por coluna
    """
    if columns is None:
        columns = df.columns
    
    # Cria uma lista de expressões de agregação para todas as colunas de uma vez
    aggs = []
    
    for column in columns:
        # Para valores mais frequentes
        aggs.append(F.max(F.struct(F.count(lit(1)).alias("count"), F.col(column).alias("value"))).alias(f"{column}_mode"))
        # Para valores nulos
        aggs.append(F.sum(F.when(F.col(column).isNull(), 1).otherwise(0)).alias(f"{column}_null_count"))
        # Contagem total (para calcular porcentagens)
        aggs.append(F.count(F.col(column)).alias(f"{column}_total"))
    
    # Executa todas as agregações em uma única operação
    agg_df = df.agg(*aggs)
    
    # Prepara os resultados em um formato mais legível
    results = []
    total_rows = df.count()
    
    for column in columns:
        # Extrai o valor mais frequente e sua contagem
        mode_row = agg_df.select(
            F.col(f"{column}_mode.value").alias("most_common_value"),
            F.col(f"{column}_mode.count").alias("most_common_count"),
            F.col(f"{column}_null_count"),
            F.col(f"{column}_total")
        ).first()
        
        most_common_value = mode_row["most_common_value"]
        most_common_count = mode_row["most_common_count"]
        null_count = mode_row[f"{column}_null_count"]
        non_null_count = mode_row[f"{column}_total"]
        
        # Calcula porcentagens
        most_common_pct = (most_common_count / non_null_count * 100) if non_null_count > 0 else 0
        null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        
        results.append({
            "column": column,
            "most_common_value": most_common_value,
            "most_common_count": most_common_count,
            "most_common_pct": most_common_pct,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_values": df.select(column).distinct().count()
        })
    
    return spark.createDataFrame(results)

# Exemplo de uso:
# stats_df = calculate_column_stats(df)
# stats_df.show()