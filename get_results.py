import pandas as pd
import numpy as np
import collections
import streamlit as st



def get_final_df(df, clf):
  
  # função para retornar os n clientes com maior probabilidade de churn

  y_pred_prob = clf.predict_proba(df.drop(columns = ['id sap']))

  probabs_churn = []

  for probs in list(y_pred_prob):

    probabs_churn.append(probs[1])

  prob_churn = dict()

  indexes = list(range(len(y_pred_prob)))

  for prob, index in zip(probabs_churn, indexes):
    
    prob_churn[index] = prob*100
      
  final_keys = sorted(prob_churn, key=prob_churn.get, reverse = True)

  final_churn_dict = dict()

  for k in final_keys:
        
    final_churn_dict[k] = prob_churn[k]
      
  lista_index_churn_ordenada = list(collections.OrderedDict.fromkeys(final_churn_dict.keys()))

  # Lista com os ID SAPs dos clientes preditos
  churns = []

  for item in list(df.iloc[lista_index_churn_ordenada]['id sap']):

    churns.append(int(item))

  df_predict = df.drop(columns = ['id sap']).iloc[lista_index_churn_ordenada]

  df_churn = pd.DataFrame(clf.predict_proba(df_predict)*100, index = churns, columns = ['Permanecer na empresa (%)', 'Churn (%)'])

  for col in df_churn.columns:
    
    df_churn[col] = df_churn[col].apply(lambda x: round(x, 2))

  df_churn = df_churn.sort_values(by = 'Churn (%)', ascending = False)

  return df_churn

def to_excel(df):

    import pandas as pd
    from io import BytesIO

    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):

    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    import base64

    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="tabela_predicoes.xlsx">Baixar arquivo Excel</a>'

def regioes():

    regioes = {0: 'Acre', 1: 'Alagoas', 2: 'Amazonas', 3: 'Amapá', 4: 'Bahia', 5: 'Ceará', 6: 'Distrito Federal',
    7: 'Espírito Santo', 8: 'Goiás', 9: 'Tocantins', 10: 'Maranhão', 11: 'Minas Gerais', 12: 'Mato Grosso do Sul', 13: 'Mato Grosso', 
    14: 'Pará', 15: 'Paraíba', 16: 'Pernambuco', 17: 'Piauí', 18: 'Paraná', 19: 'Rio de Janeiro', 20: 'Rio Grande do Norte', 21: 'Rondônia', 22: 'Roraima', 23: 'Rio Grande do Sul', 24: 'Santa Catarina', 25: 'Sergipe', 26: 'São Paulo'}

    select_reg = {value:key for key, value in regioes.items ()}

    return select_reg

def criar_faixa_preco(df, col):
  
  conditions = [
              df[col].between(0,200),  
              df[col].between(200, 400),
              df[col].between(400,600),  
              df[col].between(600, 1000),
              df[col].between(1000, 5000),  
              df[col].ge(5000)
             ]
             
  choices = ['0 a 200', '201 a 400', '401 a 600', '600 a 1000', '1000 a 5000', 'Acima de 5000']
    
  df["faixa_preco"] = np.select(conditions, choices, default=np.nan)

  return df

def show_filtered_table(df):

    st.write(df)
    st.write(f'Total de linhas = {df.shape[0]} / Total de colunas = {df.shape[1]}')
