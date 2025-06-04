from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType
from pyspark.ml.feature import Bucketizer

def cria_bucket_baseline(
    df, 
    col_data: str, 
    prefix: str, 
    dict_splits: dict, 
    lista_colunas: list, 
    qtd_meses_baseline: list
):
    """
    Cria buckets de baseline para análise de dados.
    
    Args:
        df: DataFrame do PySpark
        col_data: Nome da coluna de data
        prefix: Prefixo para nomear colunas de resultado
        dict_splits: Dicionário com splits para as features
        lista_colunas: Lista de colunas para análise
        qtd_meses_baseline: Lista de meses para filtro
        
    Returns:
        DataFrame com as estatísticas de baseline por faixa
    """
    # 1. Pré-processamento inicial
    df_filtered = (
        df
        .withColumn(col_data, F.date_format(col_data, "yyyy-MM-dd"))
        .filter(F.col(col_data).isin(qtd_meses_baseline))
    )
    
    total_rows = df_filtered.count()
    if total_rows == 0:
        return spark.createDataFrame([], _get_output_schema(prefix))
    
    # 2. Preparação das faixas de corte
    df_corte_spark = _prepare_cut_ranges(dict_splits)
    df_corte_spark.persist()
    _ = df_corte_spark.count()  # Força a materialização
    
    # 3. Processamento das colunas
    dfs = []
    for feature in lista_colunas:
        if _is_categorical_feature(df_filtered, feature):
            df_grouped = _process_categorical_feature(
                df_filtered, feature, prefix, total_rows
            )
        else:
            df_grouped = _process_numerical_feature(
                df_filtered, feature, prefix, total_rows, dict_splits, df_corte_spark
            )
        dfs.append(df_grouped)
    
    # 4. Combinação dos resultados
    result_df = dfs[0]
    for df in dfs[1:]:
        result_df = result_df.union(df)
    
    df_corte_spark.unpersist()
    
    return result_df

# Funções auxiliares
def _get_output_schema(prefix: str) -> StructType:
    """Retorna o schema de saída padrão."""
    return StructType([
        StructField("Classe", StringType()),
        StructField("Variavel", StringType()),
        StructField(prefix, DoubleType()),
        StructField("cnpj_cpf_count", IntegerType()),
    ])

def _prepare_cut_ranges(dict_splits: dict):
    """Prepara as faixas de corte para variáveis numéricas."""
    window_spec = Window.partitionBy("feature").orderBy("corte")
    
    return (
        spark.createDataFrame(
            [(f, c) for f, splits in dict_splits.items() for c in splits],
            ["feature", "corte"]
        )
        .withColumn("faixa", F.row_number().over(window_spec) - 1)
        .withColumn("corte_lead", F.round(F.lead("corte", 1).over(window_spec), 2)
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
        .filter(F.col("corte") != -float("inf"))
    )

def _is_categorical_feature(df, feature: str) -> bool:
    """Verifica se a feature é categórica."""
    dtypes = dict(df.dtypes)
    return dtypes[feature] == "string" or feature.startswith('cindcd_')

def _process_categorical_feature(df, feature: str, prefix: str, total_rows: int):
    """Processa uma feature categórica."""
    return (
        df.groupBy(F.col(feature))
        .agg(F.count("*").alias("cnpj_cpf_count"))
        .select(
            F.col(feature).cast(StringType()).alias("Classe"),
            F.lit(feature).cast(StringType()).alias("Variavel"),
            F.round(F.col("cnpj_cpf_count") / total_rows, 5).alias(prefix),
            F.col("cnpj_cpf_count")
        )
    )

def _process_numerical_feature(
    df, feature: str, prefix: str, total_rows: int, dict_splits: dict, df_corte_spark
):
    """Processa uma feature numérica."""
    splits = dict_splits[feature]
    
    if len(splits) <= 2:
        df_bucket = df.select(feature).withColumn("faixa", F.lit(None))
    else:
        bucketizer = Bucketizer(
            inputCol=feature,
            outputCol="faixa",
            splits=splits,
            handleInvalid="keep"
        )
        df_bucket = bucketizer.transform(df.select(feature))
    
    return (
        df_bucket
        .groupBy("faixa")
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
            F.col("Classe"),
            F.lit(feature).cast(StringType()).alias("Variavel"),
            F.round(F.col("cnpj_cpf_count") / total_rows, 5).alias(prefix),
            F.col("cnpj_cpf_count")
        )
    )
