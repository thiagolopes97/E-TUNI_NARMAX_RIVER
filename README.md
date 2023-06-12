# E-TUNI_NARMAX_RIVER

# Dissertação

Repositório oficial sobre a Dissertação de Mestrado Thiago Lopes em Engenharia Eletrônica e Computação - Informática no ITA - Análise comparativa entre a metodologia NARMAX e E-TUNI aplicadas na previsão de fluxo de vazão de bacias hidrográficos. Todo o projeto foi desenvolvido na linguagem Python 3.9


## Descrição do projeto
Este repositório contém os dados, análises, treinamento de modelos e comparação entre as metodologias NARMAX e E-TUNI com derivadas médias na predição de séries temporais.

As contribuições deste trabalho estendem-se: Análise Exploratória de Dados (EDA) aplicado a séries temporais da vazão de rios, avaliação comparativa entre os desempenhos das metodologias NARMAX e E-TUNI utilizando Redes Neurais Artificiais em problemas do tipo escalar e vetorial. Foram analisados dados do período de janeiro de 1931 até dezembro de 2012, totalizando 82 anos de séries históricas, cada uma representa a vazão média em $m^3/s$ mensal das usinas hidroelétricas de Furnas e Camargos. 

Cada série histórica, totalizou 984 pontos no tempo, sendo utilizado os últimos 120 pontos (equivalente a 10 anos) para avaliação do modelo e o restante para treinamento. Foram analisados a eficiência da mesma rede neural na tarefa de predição de séries temporais variando a abordagem NARMAX e E-TUNI no problema Escalar e Vetorial.

### Principais libs
- Pandas
- Tensorflow
- Scikit-learn
- Stats