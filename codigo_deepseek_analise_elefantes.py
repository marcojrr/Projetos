from pyspark.sql import functions as F

def calculate_monthly_stats(df, date_column, exclude_columns=None):
    """
    Calcula estatísticas mensais simplificadas para todas as colunas, exceto as excluídas.
    
    Args:
        df: DataFrame do PySpark
        date_column: Nome da coluna de data para definir as safras mensais (YYYYMM)
        exclude_columns: Lista de colunas a serem excluídas da análise
        
    Returns:
        DataFrame com formato: safra | feature | valor_mais_comum | concentracao_mais_comum | percentual_nulos
    """
    # Determina as colunas a serem analisadas
    if exclude_columns is None:
        exclude_columns = []
    
    columns = [c for c in df.columns if c not in exclude_columns and c != date_column]
    
    # Cria a safra mensal e prepara o DataFrame
    df_with_safra = df.withColumn("safra", F.date_format(F.col(date_column), "yyyyMM"))
    
    # Agregações básicas para cada coluna
    aggs = []
    for column in columns:
        aggs.extend([
            F.first(F.col("safra")).alias("safra"),
            F.lit(column).alias("feature"),
            F.max(F.struct(
                F.count("*").alias("count"), 
                F.col(column).alias("value")
            )).alias("moda"),
            F.avg(F.when(F.col(column).isNull(), 1).otherwise(0)).alias("percentual_nulos")
        ])
    
    # Executa todas as agregações de uma vez
    result = df_with_safra.groupBy("safra").agg(*aggs).select(
        "safra",
        "feature",
        F.col("moda.value").alias("valor_mais_comum"),
        (F.col("moda.count") / F.count("*")).alias("concentracao_mais_comum"),
        (F.col("percentual_nulos") * 100).alias("percentual_nulos")
    ).orderBy("safra", "feature")
    
    return result

# Exemplo de uso:
# stats = calculate_monthly_stats(df, "data_referencia", exclude_columns=["id", "codigo"])
# stats.show()
