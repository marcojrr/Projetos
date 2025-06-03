from pyspark.sql import functions as F

def calculate_monthly_stats(df, date_column, exclude_columns=None):
    """
    Calcula estatísticas mensais otimizadas para Databricks.
    
    Args:
        df: DataFrame do PySpark
        date_column: Nome da coluna de data para definir as safras (YYYYMM)
        exclude_columns: Lista de colunas a serem excluídas da análise
        
    Returns:
        DataFrame com formato: safra | feature | valor_mais_comum | concentracao_mais_comum | percentual_nulos
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Seleciona colunas para análise
    columns = [c for c in df.columns if c not in exclude_columns and c != date_column]
    
    # Cria coluna de safra mensal
    df_with_safra = df.withColumn("safra", F.date_format(F.col(date_column), "yyyyMM"))
    
    # Calcula totais e nulos por safra para todas as colunas de uma vez
    null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}_nulls") for c in columns]
    base_stats = df_with_safra.groupBy("safra").agg(F.count("*").alias("total"), *null_exprs)
    
    # Processa cada coluna para encontrar a moda
    results = []
    for column in columns:
        # Encontra o valor mais frequente por safra
        mode_df = (df_with_safra
                  .filter(F.col(column).isNotNull())  # Filtra nulos para moda
                  .groupBy("safra", column)
                  .count()
                  .groupBy("safra")
                  .agg(F.max(F.struct("count", column)).alias("mode"))
                  )
        
        # Junta com as estatísticas base e calcula métricas finais
        column_stats = (mode_df
                       .join(base_stats, "safra")
                       .select(
                           "safra",
                           F.lit(column).alias("feature"),
                           F.col(f"mode.{column}").alias("valor_mais_comum"),
                           (F.col("mode.count") / F.col("total")).alias("concentracao_mais_comum"),
                           (F.col(f"{column}_nulls") / F.col("total") * 100).alias("percentual_nulos")
                       ))
        results.append(column_stats)
    
    # Combina todos os resultados corretamente
    if not results:
        return spark.createDataFrame([], "safra string, feature string, valor_mais_comum string, concentracao_mais_comum double, percentual_nulos double")
    
    final_result = results[0]
    for df_temp in results[1:]:
        final_result = final_result.union(df_temp)
    
    return final_result.orderBy("safra", "feature")

# Exemplo de uso:
# stats = calculate_monthly_stats(df, "data_referencia", exclude_columns=["id", "codigo"])
# display(stats)
