import pandas as pd
import numpy as np

def df_transform(path, name):
  '''
    O conjunto de dados em relação aos rios é composto por duas séries
    históricas, cada uma representa a vazão média em [m^3/ s] de um
    determinado mês das usinas hidroelétricas de Furnas e Camargos. Estas
    séries históricas apresentam alta correlação que pode ser explicada pelo
    fato de que ambas as usinas estão situadas no Rio Grande, na bacia do
    Rio Paraná. Os dados são de 82 anos do histórico de operação dessas duas
    usinas hidroelétricas, totalizando 984 amostras mensais desde janeiro de
    1931 até dezembro de 2012, adquiridos pela ONS (Operador Nacional do
    Sistema Elétrico).
  '''

  ## Sequence list of data
  #Camargos: https://raw.githubusercontent.com/thiagolopes97/E-TUNI_NARMAX_RIVER/main/Datasets/Rio%2001%20Camargos.txt
  #Furnas: https://raw.githubusercontent.com/thiagolopes97/E-TUNI_NARMAX_RIVER/main/Datasets/Rio%2002%20Furnas.txt

  df = pd.read_csv(path,sep='\t',header=None)
  df.columns = ['01','02','03',
                '04','05','06',
                '07','08','09',
                '10','11','12']
  df_melted = pd.melt(df)
  

  lista_aux = []
  for i in df_melted.variable.unique():
    for year in range(1931,2013):
      lista_aux.append(year)

  df_melted['year'] = lista_aux
  df_melted['date'] = pd.to_datetime(df_melted.variable.astype(str) + '/' + df_melted['year'].astype(str),format='%m/%Y')
  df_melted.sort_values('date',inplace=True)
  df_melted['name'] = name
  df_melted.reset_index(inplace=True,drop=True)

  return df_melted


def vec_2_esc(df):
  temp_list = []
  for i in range(df.shape[0]):
    obj_slice = df.iloc[i,:].to_list()
    for val in obj_slice:
      temp_list.append(val)

  return pd.DataFrame(temp_list)

def cohort_type(x, model_type):
    if model_type == 'escalar':
        if x < pd.to_datetime('1997-12-01'):
            return '1. train'
        elif x <= pd.to_datetime('2002-12-01'):
            return '2. validation'
        else:
            return '3. test'
    
    elif model_type == 'vetorial':
        if x < pd.to_datetime('1997-12-01'):
            return '1. train'
        elif x <= pd.to_datetime('2002-12-01'):
            return '2. validation'
        else:
            return '3. test'
    else:
        raise ValueError('model_type must be either "escalar" or "vetorial"')