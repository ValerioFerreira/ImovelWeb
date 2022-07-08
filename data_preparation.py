
import pandas as pd
import numpy as np
from scipy import stats as sts
import itertools
from datetime import date
from sklearn.model_selection import train_test_split
from numpy.core.fromnumeric import transpose
from sklearn.preprocessing import  StandardScaler

def start_pipeline(df):
  return df.copy()

def col_to_lowercase(df):
  df.columns = df.columns.str.lower()
  return df

def data_to_lowercase(df):
  
  colunas_transform = []
  for coluna in df.columns:
    if df[coluna].dtypes == 'O' and coluna not in ['faixa listings']:
      colunas_transform.append(coluna)

  for coluna in colunas_transform:
    try:
      df[coluna] = df[coluna].str.lower()
    except:
      df[coluna] = df[coluna]

  return df

def drop_cols(df, colunas = None):

  if colunas == None:
    
    colunas =  ["faturamento", "contratado freemium", "utilizado freemium", "frequência de faturamento",
                "executivo carteira atual", "faixa listings", "id navplat", "id crm", 'bairro', 'leads total', 'possui mídia ativa?',
                'cidade', 'valor hoje', 'contratado super destaques', 'contratado ofertas simples', 'contratado destaques', 
                'utilizado super destaques', 'utilizado ofertas simples', 'leads form', 'total de listings', 'unnamed: 37', 
                'unnamed: 38', 'unnamed: 39', 'unnamed: 40', 'unnamed: 41', 'cnpj/cpf', 'listings', 'upsale/downsale']
  
  for col in colunas:
    
    if col in list(df.columns):
      
      df = df.drop(columns = col)

  return df

def drop_rows(df, cols):

  erros = []
  corretos = []

  for item in list(df['id sap']):

    try:
      
      corretos.append(int(item))

    except:

      erros.append(item)

  df = df.drop(index = df[df['id sap'].isin(erros)].index)

  df = df.dropna(subset=cols)
  
  return df

def adjusting_regiao(df):

  erros_regiao = ['são paulo', 'santa catarina', 'goiás', 'paraná','bahia', 'rio de janeiro','espírito santo', 
                'distrito federal', 'mato grosso','rio grande do norte', 'amazonas', 'alagoas', 'minas gerais',
                'paraná ', 'sergipe', 'rio grande do sul', 'ceará', 'piauí','pernambuco', 'mato grosso do sul', 
                'paraíba', 'pará', 'maranhão','mina gerais', 'sao paulo', 'belo horizonte', 'brasília',
                'mogi das cruzes', 'caruaru', 'porto alegre', 'duque de caxias','cotia', 'canoas', 'diadema',
                'campinas', 'amapá', 'curitiba','porto belo', 'mairiporã', 'bastos', 'tocantins', 'fortaleza','acre', 'roraima']

  corretos_regiao = ['sp','sc','go','pr','ba','rj','es','df','mt','rn','am','al','mg','pr','se','rs','ce','pi','pe',
             'ms','pb','pa','ma','mg','sp','mg','df','sp','pe','rs','rj','sp','rs','sp','sp','ap','pr','sc',
             'sp','sp','to','ce','ac','rr']



  for errado, correto in zip(erros_regiao, corretos_regiao):
    df['região'].replace(errado, correto, inplace = True)

  df.drop(df[df['região'] == 'outros paises'].index, inplace = True)

  df.groupby(['id sap']).agg({'região': lambda x:sts.mode(x)[0][0]}).value_counts()

  # criando um dataframe agrupado pelos ids, com os valores mais frequentes (modas) individuais da coluna região

  df_regiao_nulos = pd.DataFrame(df.groupby(['id sap']).agg({'região': lambda x:sts.mode(x)[0][0]}))

  # dataframe com o ids cuja moda da coluna 'região' é nulo

  df_ids_nulos = df_regiao_nulos[df_regiao_nulos['região'] == 0].reset_index()

  # lista com esses ids

  ids_regiao_nula = df_ids_nulos['id sap'].tolist()

  # verificando a quantos registros na tabela original esses ids estão atrelados

  tot_reg_nulos = 0
  for id in ids_regiao_nula:
    tot_reg_nulos += len(df[df['id sap'] == id])

  # por ser um valor mínimo, opta-se por excluir esses registros:

  indexes_idsap_regiao_nulos = []
  for id in ids_regiao_nula:
    indexes_idsap_regiao_nulos.append(df[df['id sap'] == id].index.tolist())
  flat_indexes = list(itertools.chain(*indexes_idsap_regiao_nulos))

  # removendo os registros

  df.drop(flat_indexes, inplace = True)

  # os valores nulos restantes na coluna 'região' possuem ao menos um registro preenchido corretamente
  # logo, iremos salvar esse valor num dicionário, utilizando o id sap como a chave individual
  # e substituir os valores faltantes por estes valores, no dataframe 

  nulos_restantes_id = list(df[df['região'].isnull() == True]['id sap']) # ids sap dos registros nulos

  dict_correcao = {} # dicionário com chave:'id sap' e valor:'moda da coluna região para este id sap 
  for id in nulos_restantes_id:
    dict_correcao[id] = sts.mode(df[df['id sap'] == id]['região'])[0][0]

  modas_total = [] # lista com as modas na ordem em que esses ids sap aparecem
  for id in nulos_restantes_id:
    modas_total.append(dict_correcao[id][0])

  nans = [] # variável de nulos para aplicar a função replace
  for cont in range(len(modas_total)):
    nans.append(np.nan)

  # substituindo os valores nulos pelas modas de cada id sap na tabela

  df['região'] = df['região'].replace(nans,modas_total)
  
  return df

def cat_cols(df, cols):

  # transformando os valores da coluna em valores numéricos, para utilizarmos nos modelos

  for col in cols:
      df[col] = df[col].astype('category')

  cat_columns = df.select_dtypes(['category']).columns

  # salvando os valores num dicionário, para adequar o código à produtização, para que seja utilizado em novas tabelas

  map_categ_cols = {}
  for col in cat_columns:
    map_categ_cols[col] = dict(enumerate(df[col].cat.categories))

  # adicionando os valores no dataframe

  df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

  return df

def get_dummies(df, cols):

  df = pd.get_dummies(df, columns=cols, drop_first=True)

  return df

def round_cols(df, cols):

  for col in cols:
    df[col] = round(df[col], 2)
  return df

def house_time(df):

  # criação de uma coluna que determine o tempo, em meses, que o cliente está na plataforma

  atual = date.today()
  atual = pd.to_datetime(atual, format="%Y %m %d")
  
  df['tempo de casa em meses'] = round(((atual - df['mês'])/30).map(lambda x: x.components.days),0)

  df.drop(columns = ['mês'], inplace = True) # removendo a coluna 'mês', que não mais é necessária

  return df

def drop_error_rows(df):

  indexes = list(df[df['região'] == 'm'].index)
  df = df.drop(index = indexes)

  return df

def mapping_df(df):

  regioes_map = {0: 'ac', 1: 'al', 2: 'am', 3: 'ap', 4: 'ba', 5: 'ce', 6: 'df',
                  7: 'es', 8: 'go', 9: 'to', 10: 'ma', 11: 'mg', 12: 'ms', 13: 'mt', 14: 'pa',
                  15: 'pb', 16: 'pe', 17: 'pi', 18: 'pr', 19: 'rj', 20: 'rn', 21: 'ro', 22: 'rr',
                  23: 'rs', 24: 'sc', 25: 'se', 26: 'sp'}

  regioes_map_2 = {value:key for key, value in regioes_map.items()}

  df['região'] = df['região'].map(regioes_map_2)

  df['região'] = df['região'].astype('category')

  return df

def outliers(df):

  columns = ['valor mensal', 'utilizado destaque', 'leads ver dato',
             'custo por lead total',	'total contratado',	'custo por listing', 'total utilizado']

  values = [999, 70, 163, 343, 100042, 89.8, 1076]

  for col, val in zip (columns, values):
    
    df.loc[df[col] > val, col] = val

  df = df.dropna()

  return df

def freq_encoder(df, cols):

  for col in cols:

    df = df.astype({col:'category'})

    df_freq = pd.DataFrame(df[col].value_counts()/len(df[col]))

    df_freq = df_freq.transpose()

    freq = {}

    for column in df_freq.columns:
      freq[column] = df_freq[column][0]

    df[col] = df[col].replace(freq)

  return df

def to_binary(df):

  cols = ['região', 'equipe']
  
  for col in cols:
    
    df[col] = df[col].map(lambda x: 1 if x > 0.5 else 0)

  return df

def scaler(df):

  if 'status final' in list(df.columns):

    X = df.drop(columns = ['id sap', 'status final'])

  else:

    X = df.drop(columns = ['id sap'])

  scaler = StandardScaler()
  scaler.fit(X)

  X_scaled = scaler.transform(X)

  X = pd.DataFrame(X_scaled, columns = X.columns)
  
  col = df['id sap']

  X['id sap'] = col

  return X.dropna()

def final_df(df):
  
  df = (df.pipe(start_pipeline) 
        .pipe(col_to_lowercase) 
        .pipe(data_to_lowercase) 
        .pipe(drop_cols)
        .pipe(drop_rows, cols = ['utilizado destaque','total utilizado'])
        .pipe(adjusting_regiao)
        .pipe(get_dummies, cols = ['pf/pj', 'oficina','tipo de plano']) 
        .pipe(round_cols, cols = ['custo por lead total', 'custo por listing'])
        .pipe(house_time)
        .pipe(drop_error_rows)
        .pipe(mapping_df)
        .pipe(outliers)              
        .pipe(freq_encoder, cols = ['equipe', 'região'])
        .pipe(to_binary)
        .pipe(scaler)
       )
        
  return df

def return_df(df):

  df = (df.pipe(start_pipeline) 
      .pipe(col_to_lowercase) 
      .pipe(data_to_lowercase) 
      .pipe(drop_cols)
      .pipe(drop_rows, cols = ['utilizado destaque','total utilizado'])
      .pipe(adjusting_regiao)
      .pipe(house_time)
      .pipe(drop_error_rows)
      .pipe(mapping_df)
      .pipe(outliers)
      )
      
  return df