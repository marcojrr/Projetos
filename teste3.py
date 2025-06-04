from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import Bucketizer

def cria_bucket_safras(
    df,
    col_data: str,
    prefix: str,
    dict_splits: dict,
    lista_colunas: list
):
    """
    Cria buckets de análise por safras (períodos temporais) de forma otimizada.
    
    Args:
        df: DataFrame de entrada
        col_data: Nome da coluna de data
        prefix: Prefixo para as colunas de resultado
        dict_splits: Dicionário com pontos de corte para variáveis numéricas
        lista_colunas: Lista de colunas para análise
        
    Returns:
        DataFrame com estatísticas agregadas por safra e faixa de valor
    """
    # 1. Pré-processamento inicial
    df = df.withColumn(col_data, F.date_format(col_data, "yyyy-MM-dd")).cache()
    total_rows = df.count()
    
    if total_rows == 0:
        return _create_empty_schema(col_data, prefix)
    
    # 2. Preparação das faixas de corte (otimizada)
    df_corte_spark = _prepare_cut_ranges(dict_splits).cache()
    _ = df_corte_spark.count()  # Força materialização
    
    # 3. Processamento em lote
    results = []
    dtypes = dict(df.dtypes)
    
    for feature in lista_colunas:
        if _is_categorical(dtypes, feature):
            df_grouped = _process_categorical_feature(
                df, feature, col_data, prefix, total_rows
            )
        else:
            df_grouped = _process_numerical_feature(
                df, feature, col_data, prefix, 
                total_rows, dict_splits, df_corte_spark
            )
        results.append(df_grouped)
    
    # 4. Combinação eficiente dos resultados
    df_final = results[0]
    for df in results[1:]:
        df_final = df_final.unionByName(df)
    
    # 5. Liberação de recursos
    df.unpersist()
    df_corte_spark.unpersist()
    
    return df_final.sort(col_data)

# Funções auxiliares
def _create_empty_schema(col_data: str, prefix: str):
    """Cria um DataFrame vazio com o schema correto."""
    schema = StructType([
        StructField(col_data, StringType()),
        StructField("Classe", StringType()),
        StructField("Variavel", StringType()),
        StructField(prefix, DoubleType()),
        StructField("cnpj_cpf_count", IntegerType()),
    ])
    return spark.createDataFrame([], schema)

def _prepare_cut_ranges(dict_splits: dict):
    """Prepara as faixas de corte para variáveis numéricas."""
    window_spec = Window.partitionBy("feature").orderBy("corte")
    
    return (
        spark.createDataFrame(
            [(f, c) for f, splits in dict_splits.items() for c in splits],
            ["feature", "corte"]
        )
        .withColumn("faixa", F.row_number().over(window_spec) - 1)
        .withColumn("corte_lead", F.round(F.lead("corte", 1).over(window_spec), 2))
        .withColumn(
            "Classe",
            F.concat(
                F.lit("c"),
                F.col("faixa"),
                F.lit(":"),
                F.when(F.col("corte") == -float("inf"), "(").otherwise("["),
                F.round(F.col("corte"), 2),
                F.lit(","),
                F.col("corte_lead"),
                F.lit(")")
            )
        )
        .filter(F.col("corte") != float("inf"))
    )

def _is_categorical(dtypes: dict, feature: str) -> bool:
    """Determina se uma feature é categórica."""
    return dtypes[feature] == "string" or feature.startswith('cindcd_')

def _process_categorical_feature(df, feature: str, col_data: str, prefix: str, total_rows: int):
    """Processa uma feature categórica."""
    return (
        df.groupBy(F.col(col_data), F.col(feature))
        .agg(F.count("*").alias("cnpj_cpf_count"))
        .select(
            F.col(col_data),
            F.col(feature).cast(StringType()).alias("Classe"),
            F.lit(feature).cast(StringType()).alias("Variavel"),
            F.round(F.col("cnpj_cpf_count") / total_rows, 5).alias(prefix),
            F.col("cnpj_cpf_count")
        )
    )

def _process_numerical_feature(
    df, feature: str, col_data: str, prefix: str, 
    total_rows: int, dict_splits: dict, df_corte_spark
):
    """Processa uma feature numérica."""
    splits = dict_splits[feature]
    
    if len(splits) <= 2:
        df_bucket = df.select(feature, col_data).withColumn("faixa", F.lit(None))
    else:
        bucketizer = Bucketizer(
            inputCol=feature,
            outputCol="faixa",
            splits=splits,
            handleInvalid="keep"
        )
        df_bucket = bucketizer.transform(df.select(feature, col_data))
    
    return (
        df_bucket
        .groupBy("faixa", col_data)
        .agg(F.count("*").alias("cnpj_cpf_count"))
        .withColumn("faixa", F.col("faixa").cast(IntegerType()))
        .join(
            F.broadcast(df_corte_spark.filter(F.col("feature") == feature)),
            on="faixa",
            how="left"
        )
        .withColumn(
            "Classe",
            F.when(
                F.col("faixa").isNull() & F.col("Classe").isNull(), 
                "Null"
            ).otherwise(F.col("Classe"))
        )
        .select(
            F.col(col_data),
            F.col("Classe"),
            F.lit(feature).cast(StringType()).alias("Variavel"),
            F.round(F.col("cnpj_cpf_count") / total_rows, 5).alias(prefix),
            F.col("cnpj_cpf_count")
        )
    )
