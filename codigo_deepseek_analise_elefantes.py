from pyspark.sql import functions as F

def calculate_monthly_stats(df, date_column, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []
    
    columns = [c for c in df.columns if c not in exclude_columns and c != date_column]
    df_with_safra = df.withColumn("safra", F.date_format(F.col(date_column), "yyyyMM"))
    
    # Cria uma lista de expressões condicionais para contar nulos
    null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}_nulls") for c in columns]
    total_expr = F.count("*").alias("total")
    
    # Primeira agregação: conta totais e nulos por safra
    base_stats = df_with_safra.groupBy("safra").agg(total_expr, *null_exprs)
    
    # Para cada coluna, calcula a moda e junta com as estatísticas base
    results = []
    for column in columns:
        mode_df = (df_with_safra
                  .groupBy("safra", column)
                  .count()
                  .groupBy("safra")
                  .agg(F.max(F.struct("count", column)).alias("mode"))
                 )
        
        column_result = (mode_df
                        .join(base_stats, "safra")
                        .select(
                            "safra",
                            F.lit(column).alias("feature"),
                            F.col("mode." + column).alias("valor_mais_comum"),
                            (F.col("mode.count") / F.col("total")).alias("concentracao_mais_comum"),
                            (F.col(f"{column}_nulls") / F.col("total") * 100).alias("percentual_nulos")
                        )
        results.append(column_result)
    
    return results[0].unionByName(*results[1:]).orderBy("safra", "feature")
