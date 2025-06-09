import math

def calcula_tamanho_amostra_surveymonkey(tamanho_populacao, nivel_confianca=95, margem_erro=5):
    """
    Calcula o tamanho da amostra igual à calculadora do SurveyMonkey.
    
    Parâmetros:
    -----------
    tamanho_populacao : int
        Tamanho total da população (use um número grande para população "infinita").
    nivel_confianca : int (90, 95 ou 99)
        Nível de confiança desejado (padrão = 95%).
    margem_erro : int ou float
        Margem de erro desejada em % (padrão = 5%).
    
    Retorna:
    --------
    int
        Tamanho da amostra calculado.
    """
    
    # Valores Z baseados no nível de confiança
    z_scores = {
        90: 1.645,
        95: 1.96,
        99: 2.576
    }
    
    # Verifica se o nível de confiança é válido
    if nivel_confianca not in z_scores:
        raise ValueError("Nível de confiança deve ser 90, 95 ou 99.")
    
    # Converte margem de erro para decimal (ex: 5% → 0.05)
    margem_erro_decimal = margem_erro / 100
    
    # Proporção conservadora (p = 0.5 para o pior caso)
    p = 0.5
    
    # Cálculo inicial (população infinita)
    z = z_scores[nivel_confianca]
    n_infinito = (z ** 2) * p * (1 - p) / (margem_erro_decimal ** 2)
    
    # Ajuste para população finita
    n_ajustado = n_infinito / (1 + (n_infinito - 1) / tamanho_populacao)
    
    # Arredonda para cima (nunca diminuir a amostra)
    return math.ceil(n_ajustado)
