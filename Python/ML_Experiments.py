import pandas as pd               
import numpy as np  
import time
from matplotlib import pyplot as plt   
import seaborn as sns            
import plotly.express as px      
from datetime import date, datetime, timedelta  
from sklearn import preprocessing

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.inspection import permutation_importance

from sklearn.model_selection import train_test_split, cross_val_score #
from sklearn.preprocessing import StandardScaler, LabelEncoder #
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, f_regression, f_classif, mutual_info_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_absolute_error, balanced_accuracy_score
from yellowbrick.classifier import ConfusionMatrix
from sklearn.naive_bayes import GaussianNB #
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import LogisticRegression,LinearRegression,BayesianRidge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

## Dataset:
bd = pd.read_csv("Python/coleta_de_características_DB2.csv")
df = bd.dropna()
df = df[['DB_NAME','DTHORACONSULTA','CPU_LOAD_SHORT','USO_TOTAL_CPU','NR_CPU','MEMORIA_TOTAL','MEMORIA_FREE','QTD_CONEXAO','BUILD_BANCO_DB2','BUILD_INST_DB2','QTD_ERROS_ATUALIZACAO','ULTIMO_RUNSTATS','QTD_OBJETOS_INVALIDOS','DT_PACOTE_ANTIGO','PACOTES_INVALIDOS','BUFFERPOLLS_AUTO','MEMORIA_BD','DFT_QUERYOPT','LOGBUFSZ','DB_MEM_THRESH','SELF_TUNING_MEM','DATABASE_MEMORY','DFT_DEGREE','TAMANHO_DO_BANCO_GB','TRIGGERS_AUDLOG','INSTANCE_MEMORY']]

def data_preparation(bd,df):
  bd_numeric = bd.select_dtypes(include=[np.number])
  median_values = bd_numeric.median()

  bd.fillna(median_values, inplace=True)

  #############################################################################
  ##### Funções de conversões de nomes dos databases                    #######
  #############################################################################

  ## Gera o conjunto de valores únicos da coluna DB_NAME:
  xNomeDB = df['DB_NAME'].tolist()
  yNomeDB = set(xNomeDB)

  def rotulo_nome_db(a):
    if a == "AGROMETA":
      return 1
    elif a == "BELO":
      return 2
    elif a == "CRISTAL":
      return 3
    elif a == "DESTRO":
      return 4
    elif a == "DONGO":
      return 5
    elif a == "DEVKOP":
      return 6
    elif a == "GIPIELA":
      return 7
    elif a == "MADESIL":
      return 8
    elif a == "OUTLET":
      return 9
    elif a == "PKENTE":
      return 10
    elif a == "SCHWALM":
      return 11
    else:
      return 0

  df['DB_NAME'] = df['DB_NAME'].map(rotulo_nome_db)

  df["TAMANHO_DO_BANCO_GB"] = df["TAMANHO_DO_BANCO_GB"].str.split(",", n=1, expand = True)[0]

  df.describe()

  #######################################################################################
  ##### Funções de conversões de coluna  ## Funções:
  ##### Qtde Erros de atualização ## valida_erros_atualizacao
  ##### Qtde Objetos inválidos    ## valida_objetos_invalidos
  ##### Qtde Pacotes inválidos    ## valida_pacotes_invalidos
  ##### DFT_QUERYOPT              ## valida_queryopt
  ##### DB_MEM_THRESH             ## valida_mem_thresh
  ##### SELF_TUNING_MEM           ## valida_tuning_mem
  ##### DATABASE_MEMORY           ## valida_db_mem
  ##### TRIGGERS_AUDLOG           ## valida_aud_logs
  #######################################################################################

  def valida_erros_atualizacao (a):
    if a > 0:
      return 0
    else:
      return 1

  def valida_objetos_invalidos(a):
    if a > 0:
      return 0
    else:
      return 1

  def valida_pacotes_invalidos(a):
    if a > 0:
      return 0
    else:
      return 1

  def valida_queryopt(a):
    if a == 5:
      return 0
    else:
      return 1

  def valida_mem_thresh(a):
    if a > 70:
      return 0
    else:
      return 1

  def valida_tuning_mem(a):
    if a == "ON (Active)":
      return 0
    else:
      return 1

  def valida_db_mem(a):
    if a == "ON (Active)":
      return 0
    else:
      return 1

  def valida_instance_mem(a):
    if a == "AUTOMATIC":
      return 0
    else:
      return 1

  def valida_aud_logs(a):
    if a < 4:
      return 0
    else:
      return 1

  from dataclasses import replace
  def converte_float(a):
    x = a.replace(',', '.')
    return x

  #######################################################################################
  ##### Funções de conversões de coluna - Versão de instalação vs Versão de Build #######
  ##### Tratamento: ambas colunas devem ter o mesmo valor (0 - True / 1 = False ) #######
  #######################################################################################

  df['BUILD_BANCO'] = df['BUILD_BANCO_DB2'].isin(df['BUILD_INST_DB2'])

  def checa_build(a):
    if a == True:
      return 0
    else:
      return 1

  #######################################################################################
  #####             Funções de conversões de coluna - Campos de Data              #######
  #######################################################################################

  def converte_datas(a):
    if a > 90:
      return 1
    else:
      return 0

  def remove_dots(string):
    string = string.replace(".", "")
    return string


  ## Tratamento de campos
  df['QTD_ERROS_ATUALIZACAO'] = df['QTD_ERROS_ATUALIZACAO'].map(valida_erros_atualizacao)
  df['QTD_OBJETOS_INVALIDOS'] = df['QTD_OBJETOS_INVALIDOS'].map(valida_objetos_invalidos)
  df['PACOTES_INVALIDOS'] = df['PACOTES_INVALIDOS'].map(valida_pacotes_invalidos)
  df['DFT_QUERYOPT'] = df['DFT_QUERYOPT'].map(valida_queryopt)
  df['DB_MEM_THRESH'] = df['DB_MEM_THRESH'].map(valida_mem_thresh)
  df['SELF_TUNING_MEM'] = df['SELF_TUNING_MEM'].map(valida_tuning_mem)
  df['DATABASE_MEMORY'] = df['DATABASE_MEMORY'].map(valida_db_mem)
  df['INSTANCE_MEMORY'] = df['INSTANCE_MEMORY'].map(valida_instance_mem)
  df['TRIGGERS_AUDLOG'] = df['TRIGGERS_AUDLOG'].map(valida_aud_logs)
  df['BUILD_BANCO'] = df['BUILD_BANCO'].map(checa_build)

  # Tratamento de pacotes:
  df[['DT_PACOTE_ANTIGO','DTHORACONSULTA']] = df[['DT_PACOTE_ANTIGO','DTHORACONSULTA']].apply(pd.to_datetime) #if conversion required
  df['TEMPOPACOTES'] = (df['DTHORACONSULTA'] - df['DT_PACOTE_ANTIGO']).dt.days

  # Tratamento de pacotes:
  df[['ULTIMO_RUNSTATS','DTHORACONSULTA']] = df[['ULTIMO_RUNSTATS','DTHORACONSULTA']].apply(pd.to_datetime) #if conversion required
  df['TEMPORUNSTATS'] = (df['DTHORACONSULTA'] - df['ULTIMO_RUNSTATS']).dt.days

  df['TEMPOPACOTES'] = df['TEMPOPACOTES'].map(converte_datas)
  df['TEMPORUNSTATS'] = df['TEMPORUNSTATS'].map(converte_datas)

  #############################################################################
  ##### Funções de conversões de coluna - load de CPU + consumo de CPU  #######
  #############################################################################
  #uso_cpu = df['USO_TOTAL_CPU']

  def load_cpu(a):
    if float(a) > 5:
      return 1
    else:
      return 0

  def consumo_cpu (a):
    if a < 75:
      return 0
    else:
      return 1

  ##############################################################################################
  ##### Funções de conversões de coluna - consumo de memória                             #######
  ##### Tratamento: Consumo de memória maior que 95 (%), consideramos ruim               #######
  ##### Fórmula de percentual: Vi / Vf = X - 1 * 100 = Resultado       #
  ##############################################################################################
  # - cálculo ok :> df['USO_MEMORIA'] = ((df['MEMORIA_TOTAL'] / df['MEMORIA_FREE']) - 1 * 100) #
  ##############################################################################################
  temp = (df['MEMORIA_FREE'] / df['MEMORIA_TOTAL']) * 100

  df['USO_MEMORIA'] = temp

  def valida_memoria(a):
    if a > 95:
      return 1
    else:
      return 0

  df['CPU_LOAD_SHORT'] = df['CPU_LOAD_SHORT'].map(load_cpu)
  df['USO_TOTAL_CPU'] = df['USO_TOTAL_CPU'].map(consumo_cpu)
  df['USO_MEMORIA'] = df['USO_MEMORIA'].map(valida_memoria)
  df.head(10)

  # Criação de novo atributo classificador:
  df['PERFORMANCE'] = 0

  for i in df.itertuples():
    if df.loc[i.Index, 'CPU_LOAD_SHORT'] == 1 or df.loc[i.Index, 'USO_MEMORIA'] == 1 or int(df.loc[i.Index, 'TAMANHO_DO_BANCO_GB']) <= 25: # OK!
      df.loc[i.Index, 'PERFORMANCE'] = 1

  ## Dataset final

  dataset_final = df[['USO_TOTAL_CPU','CPU_LOAD_SHORT','TAMANHO_DO_BANCO_GB','NR_CPU','QTD_CONEXAO', 'QTD_ERROS_ATUALIZACAO','QTD_OBJETOS_INVALIDOS', 'PACOTES_INVALIDOS', 'BUFFERPOLLS_AUTO','MEMORIA_BD','DFT_QUERYOPT', 'LOGBUFSZ', 'DB_MEM_THRESH','DFT_DEGREE','TRIGGERS_AUDLOG','INSTANCE_MEMORY','USO_MEMORIA','TEMPOPACOTES','TEMPORUNSTATS']]
  dataset_final2 = df[['PERFORMANCE','USO_TOTAL_CPU','QTD_CONEXAO', 'QTD_ERROS_ATUALIZACAO','QTD_OBJETOS_INVALIDOS', 'PACOTES_INVALIDOS', 'BUFFERPOLLS_AUTO','MEMORIA_BD','DFT_QUERYOPT', 'LOGBUFSZ', 'DB_MEM_THRESH','SELF_TUNING_MEM','DATABASE_MEMORY','DFT_DEGREE','TRIGGERS_AUDLOG','INSTANCE_MEMORY','BUILD_BANCO','TEMPOPACOTES','TEMPORUNSTATS']]
  return dataset_final, dataset_final2

def feature_engineering(dataset_final,dataset_final2):

  x = dataset_final.iloc[:,1:20]
  x2 = dataset_final2.iloc[:,1:20]
  y = dataset_final[['USO_TOTAL_CPU']]
  y2 = dataset_final2[['PERFORMANCE']]

  #Conversão de de data frame para uso no PCA sklearn
  x_2 = dataset_final.iloc[:,1:21]
  y_full = dataset_final['USO_TOTAL_CPU']
  y_full2 = dataset_final2['PERFORMANCE']

  print('1: ', dataset_final.shape, '|', '2:',  dataset_final2.shape)


  #######################################################################################
  #####                           Normalização de Dados                           #######
  #######################################################################################

  ## Normalização:
  scale_X = StandardScaler()
  x = scale_X.fit_transform(x)
  x2 = scale_X.fit_transform(x2)
  x_full = scale_X.fit_transform(x_2)

  y_array = y_full.to_numpy()

  target_names = y_array

  pca = PCA(n_components=8)
  X_r = pca.fit(x).transform(x)

  # Percentage of variance explained for each components
  print(
      "explained variance ratio: %s"
  )

  plt.figure()
  colors = ["navy", "darkorange"]
  lw = 2

  for color, i, target_name in zip(colors, [0, 1], target_names):
      plt.scatter(
          #X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
          X_r[y_array == i, 0], X_r[y_array == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
      )
  plt.legend(loc="best", shadow=False, scatterpoints=1)
  plt.title("PCA - dataset")

  plt.show()

  x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, stratify=y, random_state=1)

  ## Contagem de valores únicos na classe de 'Uso total de CPU' :> 0 = Bom / 1 = Ruim
  print("Contagem de valores únicos na classe de 'Uso total de CPU' :> 0 = Bom / 1 = Ruim")
  print("")
  print("")
  print("Quantidade de amostras - Classe USO_TOTAL_CPU:")

  print(np.unique(dataset_final['USO_TOTAL_CPU'], return_counts=True))

  print("")

  sns.countplot(x = dataset_final['USO_TOTAL_CPU']);

  print("Contagem de valores únicos na classe de 'Uso total de CPU' :> 0 = Bom / 1 = Ruim")
  print("")
  print("")
  print("Quantidade de amostras - Classe PERFORMANCE:")

  print(np.unique(dataset_final2['PERFORMANCE'], return_counts=True))
  print("")

  sns.countplot(x = dataset_final2['PERFORMANCE']);

  grafico_matriz = px.scatter_matrix(df, dimensions=['CPU_LOAD_SHORT', 'TAMANHO_DO_BANCO_GB', 'NR_CPU', 'MEMORIA_BD'], color = 'USO_TOTAL_CPU')
  grafico_matriz.show()

  # Dados para impressão de correlação de valores com base nos valores do dataset completo
  df_correlac = df[['USO_TOTAL_CPU','CPU_LOAD_SHORT','NR_CPU','QTD_CONEXAO', 'QTD_ERROS_ATUALIZACAO','QTD_OBJETOS_INVALIDOS', 'PACOTES_INVALIDOS', 'BUFFERPOLLS_AUTO','MEMORIA_BD','DFT_QUERYOPT', 'LOGBUFSZ', 'DB_MEM_THRESH','DFT_DEGREE','TRIGGERS_AUDLOG','TEMPOPACOTES','TEMPORUNSTATS']]
  plt.figure(figsize=(40, 32))
  sns.heatmap(df_correlac.corr(),
              annot = True,
              fmt = '.2f',
              cmap='Blues')
  plt.title('Correlação entre variáveis do dataset DB')
  plt.show()

  ## Verificação de importância de feature, usando árvores de decisão:

  feature_names = [f" Atributo {i}" for i in range(x.shape[1])]
  forest = RandomForestClassifier(random_state=0)
  forest.fit(x_treino, y_treino)

  start_time = time.time()
  importances = forest.feature_importances_
  std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
  elapsed_time = time.time() - start_time

  print(f"Tempo gasto para computar a importância de características: {elapsed_time:.3f} segundos")

  forest_importances = pd.Series(importances, index=feature_names)

  fig, ax = plt.subplots()
  forest_importances.plot.bar(yerr=std, ax=ax)
  ax.set_title("\n Importância de características usando MDI\n\n")
  ax.set_ylabel("Média da impureza")
  fig.tight_layout()

  start_time = time.time()
  result = permutation_importance(
      forest, x_teste, y_teste, n_repeats=10, random_state=42, n_jobs=2
  )
  elapsed_time = time.time() - start_time
  print(f"Tempo gasto para computar a importância de características: {elapsed_time:.3f} segundos")

  forest_importances = pd.Series(result.importances_mean, index=feature_names)

  fig, ax = plt.subplots()
  forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
  ax.set_title("\n Importância de características - Permutação \n\n")
  ax.set_ylabel("média da impureza")
  fig.tight_layout()
  plt.show()

  return x, x2, x_full, y, y2, y_full, y_full2, x_treino, x_teste, y_treino, y_teste

def model_training_evaluation(x, x2, x_full, y, y2, y_full, y_full2, x_treino, x_teste, y_treino, y_teste):
  ## Naive Bayes

  naive_load = GaussianNB()
  cross_naive = naive_load.fit(x_treino, y_treino)

  # Previsores:
  previsoes_naive = cross_naive.predict(x_teste)


  accuracy_score(y_teste, previsoes_naive)

  ## Cross Validation:
  score = cross_val_score(cross_naive,x,y, cv=10)
  cross_naive.score(x_teste, y_teste)

  ## Matriz de confusão Naive Bayes:
  cm_nb = ConfusionMatrix(naive_load)
  cm_nb.fit(x_treino, y_treino)
  print('Score Naive Bayes:\n')
  cm_nb.score(x_teste, y_teste)

  print(classification_report(y_teste, previsoes_naive))

  ###############################################################
  # Aplicação de algoritmo Naive Bayes sem dados de treinamento #
  ###############################################################

  naive_load2 = GaussianNB()
  cross_naive2 = naive_load.fit(x_full, y_full)

  # Previsores:
  previsoes_naive2 = cross_naive.predict(x_full)
  previsoes_naive2

  print('Resultado de acurácia utilizando o dataset completo: ')
  accuracy_score(y_full, previsoes_naive2)

  ## Matriz de confusão Naive Bayes - dataset completo:
  cm_nb2 = ConfusionMatrix(naive_load2)
  cm_nb2.fit(x_full, y_full)
  print('Score Naive Bayes - dataset completo:\n')
  cm_nb2.score(x_full, y_full)

  #resultado_naive = accuracy_score(y_full, previsoes_naive2)*100
  #print(str(resulta_naive) +'%')

  ###############################################################
  # Aplicação de algoritmo Naive Bayes sem dados de treinamento #
  # Classe composta   - y2                                      #
  ###############################################################

  naive_load3 = GaussianNB()
  naive3 = naive_load3.fit(x2, y2)

  # Previsores:
  previsoes_naive3 = naive3.predict(x2)
  previsoes_naive3

  print('Resultado de acurácia utilizando o dataset completo - Classe Composta: ')
  accuracy_score(y2, previsoes_naive3)

  ## Matriz de confusão Naive Bayes - dataset completo - Classe composta:
  cm_nb3 = ConfusionMatrix(naive_load3)
  cm_nb3.fit(x2, y2)
  print('Score Naive Bayes - Dataset completo - Classe composta:\n')
  cm_nb3.score(x2, y2)

  print(classification_report(y2, previsoes_naive3))

  """## <<< Árvore de Decisão >>>



  """

  # Arvore de Decisão:
  arvore_load = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
  cross_arvore = arvore_load.fit(x, y)

  tree.plot_tree(arvore_load)

  # Previsões da árvore de decisão:
  previsoes_arvore = cross_arvore.predict(x_teste)
  previsoes_arvore

  ## Resultado de acurácia árvore:
  score_arvore = accuracy_score(y_teste, previsoes_arvore)
  score_arvore

  ## Cross Validation Arvore de Decisão:
  score = cross_val_score(cross_arvore,x,y, cv=10)
  cross_arvore.score(x_teste, y_teste)

  result_arvore =  cross_arvore.score(x_teste, y_teste)*100
  print(str(result_arvore) + '%')

  ## Plot decision tree:
  plt.figure(figsize=(20,10))
  tree.plot_tree(arvore_load, filled=True, max_depth=3, feature_names=None)
  plt.show()

  #plt.savefig("decistion_tree.png")

  ### Matriz de Árvore de Decisão:
  cm_av = ConfusionMatrix(arvore_load)
  cm_av.fit(x_treino, y_treino)
  print('Score Árvore de Decisão:\n')
  cm_av.score(x_teste, y_teste)

  print(classification_report(y_teste, previsoes_arvore))

  f1score = f1_score(y_teste, previsoes_arvore, average='micro')
  print('F1 Score: %f' % f1score)
  #####################################################################
  # Aplicação de algoritmo Árvore de Decisão sem dados de treinamento #/
  #####################################################################

  arvore_load2 = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
  cross_arvore2 = arvore_load2.fit(x_full, y_full)

  # Previsores:
  previsoes_arvore2 = cross_arvore2.predict(x_full)
  previsoes_arvore2

  print('Resultado de acurácia utilizando o dataset completo: ')
  score_arvore2 = accuracy_score(y_full, previsoes_arvore2)
  score_arvore2

  ## Matriz de confusão Árvore de Decisão - dataset completo:
  cm_av2 = ConfusionMatrix(arvore_load2)
  cm_av2.fit(x_full, y_full)
  print('Score Árvore de Decisão - Dataset completo:\n')
  cm_av2.score(x_full, y_full)

  result_arvore2 = accuracy_score(y_full, previsoes_arvore2)*100
  print(str(result_arvore2) + '%')

  #####################################################################
  # Aplicação de algoritmo Árvore de Decisão sem dados de treinamento #
  # Classe composta - y2                                              #
  #####################################################################

  arvore_load3 = DecisionTreeClassifier(criterion='entropy', random_state=0)
  cross_arvore3 = arvore_load3.fit(x2, y2)

  # Previsores:
  previsoes_arvore3 = cross_arvore3.predict(x2)
  previsoes_arvore3

  print('Resultado de acurácia utilizando o dataset completo - Classe composta: ')
  score_arvore3 = accuracy_score(y2, previsoes_arvore3)
  score_arvore3

  ## Matriz de confusão Naive Bayes - dataset completo - Classe composta:
  cm_av3 = ConfusionMatrix(arvore_load3)
  cm_av3.fit(x2, y2)
  print('Score Árvore de Decisão - Dataset completo - Classe composta:\n')
  cm_av3.score(x2, y2)

  print(classification_report(y2, previsoes_arvore3))

  """## <<< KNN >>>"""

  ## Implementação KNN:
  load_knn = KNeighborsClassifier(n_neighbors=8)
  load_knn.fit(x_treino, y_treino)

  previsoes_knn = load_knn.predict(x_teste)
  previsoes_knn

  ## Resultado de acurácia KNN:
  score_knn = accuracy_score(y_teste, previsoes_knn)
  score_knn

  ## Cross Validation KNN:
  cross_knn = load_knn.fit(x_treino, y_treino)

  previsoes_cross_knn = cross_knn.predict(x_teste)
  previsoes_cross_knn

  accuracy_cross_knn = accuracy_score(y_teste, previsoes_cross_knn)

  score_cross_knn = cross_val_score(cross_knn, x,y, cv=10)

  cross_knn.score(x_teste, y_teste)

  score_cross_knn

  ## Matriz de Confusão Knn:
  cm_knn = ConfusionMatrix(load_knn)
  cm_knn.fit(x_treino, y_treino)
  print('Score KNN:\n')
  cm_knn.score(x_teste, y_teste)

  print(classification_report(y_teste, previsoes_knn))

  #####################################################################
  # Aplicação de algoritmo KNN sem dados de treinamento #
  #####################################################################
  load_knn2 = KNeighborsClassifier(n_neighbors=8)
  load_knn2.fit(x_full, y_full)

  # Previsores:
  previsoes_knn2 = load_knn2.predict(x_full)
  previsoes_knn2

  print('Resultado de acurácia utilizando o dataset completo: ')
  score_knn2 = accuracy_score(y_full, previsoes_knn2)
  score_knn2

  ## Matriz de Confusão Knn - Dataset completo:
  cm_knn2 = ConfusionMatrix(load_knn2)
  cm_knn2.fit(x_full, y_full)
  print('Score KNN - Dataset completo:\n')
  cm_knn2.score(x_full, y_full)

  #####################################################################
  # Aplicação de algoritmo KNN sem dados de treinamento               #
  # Classe composta - y2                                              #
  #####################################################################
  load_knn3 = KNeighborsClassifier(n_neighbors=8)
  load_knn3.fit(x2, y2)

  # Previsores:
  previsoes_knn3 = load_knn3.predict(x2)
  previsoes_knn3

  print('Resultado de acurácia utilizando o dataset completo - Classe composta: ')
  score_knn3 = accuracy_score(y2, previsoes_knn3)
  score_knn3

  ## Matriz de Confusão Knn - Dataset completo - Classe composta:
  cm_knn3 = ConfusionMatrix(load_knn3)
  cm_knn3.fit(x2, y2)
  print('Score KNN - Dataset completo - Classe composta:\n')
  cm_knn3.score(x2, y2)

  print(classification_report(y2, previsoes_knn3))

  """## <<< Regressão Logística >>>"""

  ## Implementação de Regressão Logpistica:
  logistic_regres = LogisticRegression(random_state=1)
  logistic_regres.fit(x_treino, y_treino)

  ## Previsões Regressão:
  previsoes_logistic = logistic_regres.predict(x_teste)
  previsoes_logistic

  ## Resultado de acurácia:
  score_logistic = accuracy_score(y_teste, previsoes_logistic)
  score_logistic

  ## Cross Validation Logistic Regression:
  cross_lr = load_knn.fit(x_treino, y_treino)

  previsoes_cross_lr = cross_lr.predict(x_teste)
  previsoes_cross_lr

  accuracy_cross_lr = accuracy_score(y_teste, previsoes_cross_lr)

  score_cross_lr = cross_val_score(cross_lr, x,y, cv=10)

  cross_lr.score(x_teste, y_teste)

  score_cross_lr

  ## Matriz de Confusão LR:
  cm_lr = ConfusionMatrix(logistic_regres)
  cm_lr.fit(x_treino, y_treino)
  print('Score LR:\n')
  cm_lr.score(x_teste, y_teste)

  print(classification_report(y_teste, previsoes_logistic))

  #######################################################################
  # Aplicação de algoritmo Regressão Logística sem dados de treinamento #
  #######################################################################
  logistic_regres2 = LogisticRegression(random_state=1)
  logistic_regres2.fit(x_full, y_full)

  # Previsores:
  previsoes_logistic2 = logistic_regres2.predict(x_full)
  previsoes_logistic2

  print('Resultado de acurácia utilizando o dataset completo: ')
  score_logistic2 = accuracy_score(y_full, previsoes_logistic2)
  score_logistic2

  ## Matriz de Confusão LR:
  cm_lr2 = ConfusionMatrix(logistic_regres2)
  cm_lr2.fit(x_full, y_full)
  print('Score Regressão Logística - Dataset completo:\n')
  cm_lr2.score(x_full, y_full)

  #######################################################################
  # Aplicação de algoritmo Regressão Logística sem dados de treinamento #
  # Classe composta - y2                                                #
  #######################################################################
  logistic_regres3 = LogisticRegression(random_state=1)
  logistic_regres3.fit(x2, y2)

  # Previsores:
  previsoes_logistic3 = logistic_regres3.predict(x2)
  previsoes_logistic3

  print('Resultado de acurácia utilizando o dataset completo - Classe composta: ')
  score_logistic3 = accuracy_score(y2, previsoes_logistic3)
  score_logistic3

  ## Matriz de Confusão LR - Classe composta:
  cm_lr3 = ConfusionMatrix(logistic_regres3)
  cm_lr3.fit(x2, y2)
  print('Score Regressão Logística - Dataset completo - Classe composta:\n')
  cm_lr3.score(x2, y2)

  print(classification_report(y2, previsoes_logistic3))

  """## <<< Randon Forest >>>"""

  ##
  load_random_forest = RandomForestClassifier(n_estimators=30, criterion='entropy', random_state=0)
  load_random_forest.fit(x_treino, y_treino)

  previsoes_random_forest = load_random_forest.predict(x_teste)
  previsoes_random_forest

  score_random_forest = accuracy_score(y_teste, previsoes_random_forest)
  score_random_forest

  ## Cross Validation Random Forest:
  cross_rf = load_random_forest.fit(x_treino, y_treino)

  previsoes_cross_rf = cross_rf.predict(x_teste)
  previsoes_cross_rf

  accuracy_cross_rf = accuracy_score(y_teste, previsoes_cross_rf)

  score_cross_rf = cross_val_score(cross_rf, x,y, cv=12)

  cross_rf.score(x_teste, y_teste)

  score_cross_rf

  ## Matriz de Confusão RF:
  cm_rf = ConfusionMatrix(load_random_forest)
  cm_rf.fit(x_treino, y_treino)
  print('Score RF:\n')
  cm_rf.score(x_teste, y_teste)

  print(classification_report(y_teste, previsoes_random_forest))

  #######################################################################
  # Aplicação de algoritmo Random Forest sem dados de treinamento #
  #######################################################################
  load_random_forest2 = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=0)
  load_random_forest2.fit(x_full, y_full)

  # Previsores:
  previsoes_random_forest2 = load_random_forest2.predict(x_full)
  previsoes_random_forest2

  print('Resultado de acurácia utilizando o dataset completo: ')
  score_random_forest2 = accuracy_score(y_full, previsoes_random_forest2)
  score_random_forest2

  ## Matriz de Confusão RF - Dataset completo:
  cm_rf2 = ConfusionMatrix(load_random_forest2)
  cm_rf2.fit(x_full, y_full)
  print('Score RF - Dataset completo:\n')
  cm_rf2.score(x_full, y_full)

  #### Redução de dimensionalidade -  PCA no Random Forest:
  #### Utilizando treino e teste
  pca = PCA(n_components=10)
  x_treino_pca = pca.fit_transform(x_treino)
  x_teste_pca = pca.transform(x_teste)

  ## Validação de variância de acerto dos dados:
  pca.explained_variance_ratio_

  # Treinamento  PCA usando Random Forest:
  rf_pca = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=0)
  rf_pca.fit(x_treino_pca, y_treino)

  #Previsões RF/PCA
  previsoes_rf_pca = rf_pca.predict(x_teste_pca)
  #previsoes_mlp_pca

  # Acurácia RF/PCA
  print('Score RF/PCA:\n')
  accuracy_score(y_teste, previsoes_rf_pca)

  #######################################################################
  # Aplicação de algoritmo Random Forest sem dados de treinamento #
  # Classe composta - y2                                          #
  #######################################################################
  load_random_forest3 = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=0)
  load_random_forest3.fit(x2, y2)

  # Previsores:
  previsoes_random_forest3 = load_random_forest3.predict(x2)
  previsoes_random_forest3

  print('Resultado de acurácia utilizando o dataset completo: ')
  score_random_forest3 = accuracy_score(y2, previsoes_random_forest3)
  score_random_forest3

  ## Matriz de Confusão RF - Dataset completo - Classe composta:
  cm_rf3 = ConfusionMatrix(load_random_forest3)
  cm_rf3.fit(x2, y2)
  print('Score RF - Dataset completo - Classe composta:\n')
  cm_rf3.score(x2, y2)

  print(classification_report(y2, previsoes_random_forest3))

  #Fonte:  https://chrisalbon.com/code/machine_learning/model_evaluation/plot_the_validation_curve/
  from sklearn.model_selection import validation_curve

  param_range = np.arange(1, 150, 2)

  # Calculate accuracy on training and test set using range of parameter values
  train_scores, test_scores = validation_curve(RandomForestClassifier(),
                                              x,
                                              y,
                                              param_name="n_estimators",
                                              param_range=param_range,
                                              cv=3,
                                              scoring="accuracy",
                                              n_jobs=-1)

  # Calculate mean and standard deviation for training set scores
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)

  # Calculate mean and standard deviation for test set scores
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)

  # Plot mean accuracy scores for training and test sets
  plt.plot(param_range, train_mean, label="Training score", color="black")
  plt.plot(param_range, test_mean, label="Cross-validation score", color="dimgrey")

  # Plot accurancy bands for training and test sets
  plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
  plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

  # Create plot
  plt.title("Validation Curve With Random Forest")
  plt.xlabel("Number Of Trees")
  plt.ylabel("Accuracy Score")
  plt.tight_layout()
  plt.legend(loc="best")
  plt.show()

  """## <<< Support Vector Machines (SVM) >>>"""

  ##
  load_svm = SVC(kernel='linear', random_state=1)
  load_svm.fit(x_treino, y_treino)

  previsoes_svm = load_svm.predict(x_teste)
  previsoes_svm

  score_svm = accuracy_score(y_teste, previsoes_svm)
  score_svm

  ## Cross Validation SVM:
  cross_svm = load_svm.fit(x_treino, y_treino)

  previsoes_cross_svm = cross_svm.predict(x_teste)
  previsoes_cross_svm

  accuracy_cross_svm = accuracy_score(y_teste, previsoes_cross_svm)

  score_cross_svm = cross_val_score(cross_svm, x,y, cv=12)

  cross_svm.score(x_teste, y_teste)

  score_cross_svm

  ## Matriz de Confusão SVM:
  cm_svm = ConfusionMatrix(load_svm)
  cm_svm.fit(x_treino, y_treino)
  print('Score SVM:\n')
  cm_svm.score(x_teste, y_teste)

  print(classification_report(y_teste, previsoes_svm))

  #######################################################################
  # Aplicação de algoritmo SVM sem dados de treinamento #
  #######################################################################
  load_svm2 = SVC(kernel='linear', random_state=1)
  load_svm2.fit(x_full, y_full)

  # Previsores:
  previsoes_svm2 = load_svm2.predict(x_full)
  previsoes_svm2

  print('Resultado de acurácia utilizando o dataset completo: ')
  score_svm2 = accuracy_score(y_full, previsoes_svm2)
  score_svm2

  ## Matriz de Confusão SVM - Dataset completo:
  cm_svm2 = ConfusionMatrix(load_svm2)
  cm_svm2.fit(x_full, y_full)
  print('Score SVM - Dataset completo:\n')
  cm_svm2.score(x_full, y_full)

  #######################################################################
  # Aplicação de algoritmo SVM sem dados de treinamento #
  # Classe composta - y2                                                #
  #######################################################################
  load_svm3 = SVC(kernel='linear', random_state=1)
  load_svm3.fit(x2, y2)

  # Previsores:
  previsoes_svm3 = load_svm3.predict(x2)
  previsoes_svm3

  print('Resultado de acurácia utilizando o dataset completo: ')
  score_svm3 = accuracy_score(y2, previsoes_svm3)
  score_svm3

  ## Matriz de Confusão SVM - Dataset completo - Classe composta:
  cm_svm3 = ConfusionMatrix(load_svm3)
  cm_svm3.fit(x2, y2)
  print('Score SVM - Dataset completo:\n')
  cm_svm3.score(x2, y2)

  print(classification_report(y2, previsoes_svm3))

  """## <<< MLP >>>"""

  x_treino.shape

  (20+1) /2

  #Treinamento:
  load_mlp = MLPClassifier( hidden_layer_sizes=(10,5), solver='sgd', verbose=True, max_iter=500, tol=0.000010, random_state=0)
  load_mlp.fit(x_treino, y_treino)

  previsoes_mlp = load_mlp.predict(x_teste)
  previsoes_mlp

  score_mlp = accuracy_score(y_teste, previsoes_mlp)
  score_mlp

  load_mlp.score(x_teste,y_teste)

  #Cross validation - MLP:

  cross_mlp = load_mlp.fit(x_treino, y_treino)

  previsoes_cross_mlp = cross_mlp.predict(x_teste)
  previsoes_cross_mlp

  accuracy_cross_mlp = accuracy_score(y_teste, previsoes_cross_mlp)


  score_cross_mlp = cross_val_score(cross_mlp, x,y, cv=10)

  cross_mlp.score(x_teste, y_teste)

  score_cross_mlp

  ## Matriz de Confusão MLP:
  cm_mlp = ConfusionMatrix(load_mlp)
  cm_mlp.fit(x_treino, y_treino)
  print('Score MLP:\n')
  cm_mlp.score(x_teste, y_teste)

  print(classification_report(y_teste, previsoes_mlp))

  #######################################################################
  # Aplicação de algoritmo MLP sem dados de treinamento #
  #######################################################################
  load_mlp2 = MLPClassifier(hidden_layer_sizes=(10,5), solver='sgd', verbose=True, max_iter=500, tol=0.000010, random_state=0)
  load_mlp2.fit(x_full, y_full)

  # Previsores:
  previsoes_mlp2 = load_mlp2.predict(x_full)
  previsoes_mlp2

  print('Resultado de acurácia utilizando o dataset completo: ')
  score_mlp2 = accuracy_score(y_full, previsoes_mlp2)
  score_mlp2

  ## Matriz de Confusão MLP:
  cm_mlp2 = ConfusionMatrix(load_mlp2)
  cm_mlp2.fit(x_full, y_full)
  print('Score MLP - Dataset completo:\n')
  cm_mlp2.score(x_full, y_full)

  #### Redução de dimensionalidade -  PCA no MLP:
  #### Utilizando treino e teste
  #pca = PCA(n_components=8)
  # Teste:
  x_treino_pca = pca.fit_transform(x_treino)
  x_teste_pca = pca.transform(x_teste)

  ## Validação de variância de acerto dos dados:
  pca.explained_variance_ratio_

  ## Soma de validação de variância de acerto dos dados:
  pca.explained_variance_ratio_.sum()

  # Treinamento  PCA usando MLP:
  mlp_pca = MLPClassifier(hidden_layer_sizes=(10,5), solver='sgd', verbose=True, max_iter=500, tol=0.000010, random_state=0)
  mlp_pca.fit(x_treino_pca, y_treino)

  previsoes_mlp_pca = mlp_pca.predict(x_teste_pca)
  #previsoes_mlp_pca

  print('Score MLP/PCA:\n')
  accuracy_score(y_teste, previsoes_mlp_pca)

  #######################################################################
  # Aplicação de algoritmo MLP sem dados de treinamento #
  # Classe composta - y2
  #######################################################################
  load_mlp3 = MLPClassifier(hidden_layer_sizes=(10,5), solver='sgd', verbose=True, max_iter=1000, tol=0.000010, random_state=0)
  load_mlp3.fit(x2, y2)

  # Previsores:
  previsoes_mlp3 = load_mlp3.predict(x2)
  previsoes_mlp3

  print('Resultado de acurácia utilizando o dataset completo - Classe composta: ')
  score_mlp3 = accuracy_score(y2, previsoes_mlp3)
  score_mlp3

  ## Matriz de Confusão MLP - Classe composta:
  cm_mlp3 = ConfusionMatrix(load_mlp3)
  cm_mlp3.fit(x2, y2)
  print('Score MLP - Dataset completo - Classe composta:\n')
  cm_mlp3.score(x2, y2)

  print(classification_report(y2, previsoes_mlp3))

  """## Resultado Gerais dos modelos"""

  # Recall_score:

  print("Naive Bayes: ", balanced_accuracy_score(y2, previsoes_naive3), "\n",
        "Árvore Decisão: ",balanced_accuracy_score(y2, previsoes_arvore3), "\n",
        "KNN: ",balanced_accuracy_score(y2, previsoes_knn3), "\n",
        "Regressão Logística: ",balanced_accuracy_score(y2, previsoes_logistic3), "\n",
        "Random Forest: ",balanced_accuracy_score(y2, previsoes_random_forest3), "\n",
        "SVM: ",balanced_accuracy_score(y2, previsoes_svm3), "\n",
        "MLP: ",balanced_accuracy_score(y2, previsoes_mlp3))
  
dataset_final, dataset_final2 = data_preparation(bd,df)
x, x2, x_full, y, y2, y_full, y_full2, x_treino, x_teste, y_treino, y_teste = feature_engineering(dataset_final,dataset_final2)
model_training_evaluation(x, x2, x_full, y, y2, y_full, y_full2, x_treino, x_teste, y_treino, y_teste)