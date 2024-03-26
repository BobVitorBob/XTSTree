import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import sys


if len(sys.argv) > 1:
  nome_arq = sys.argv[1]
else:
  nome_arq = 'resultados backup.csv'
# Lendo dados e criando lista de modelos e estatísticas de interesse

# Dropa nulo
dados = pd.read_csv(nome_arq).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').reset_index(drop=True)

modelos = [
  # 'full',
  'PageHinkley',
  'TopDownReg',
  'TopDownIndex',
]

# Pega só o que tá em modelos
dados = dados[dados['model'].str.startswith(tuple(modelos))]

# Filtra cópias
dados = dados.set_index('nome').groupby(level=0).first().reset_index(drop=True)

# dados = dados[dados['numero de segmentos'] > 1]

print('Execuções únicas: ', len(dados))
# Filtra execuções que pelo menos um dos modelos falhou
dados = dados[dados['file'].apply(lambda file: len(dados[dados['file'] == file]) == len(modelos))]


lengths = list(set([re.search(r'(.*)dias_umidrelmed2m', file).groups()[0] for file in list(dados['file'])]))

for model in modelos:
  print(f'Número de execuções do {model}: ', len(dados[dados.model.str.startswith(model)]))

for length in lengths:
  dados_len = dados[dados['file'].str.startswith(length)]
  resultados = {
    'mean_comp': [],
    'std_comp': [],
    'mean_MAE': [],
    'std_MAE': [],
    'mean_RMSE': [],
    'std_RMSE': [],
    'mean_std_comp': [],
    'std_std_comp': [],
    'mean_tempo': [],
    'std_tempo': [],
    'mean_num_seg': [],
    'std_num_seg': [],
    'num_res': [],
    'mean_ent_gain': [],
    'std_ent_gain': [],
  }

  # Calculando estatísticas
  for model in modelos:
    dados_estat = dados_len[dados_len.model.str.startswith(model)]
    resultados['mean_comp'].append(np.mean(dados_estat['complexidade (média dos segmentos)']))
    resultados['std_comp'].append(np.std(dados_estat['complexidade (média dos segmentos)']))
    resultados['mean_MAE'].append(np.mean(dados_estat['MAE (erro entre a série inteira e a predição de todos os segmentos)']))
    resultados['std_MAE'].append(np.std(dados_estat['MAE (erro entre a série inteira e a predição de todos os segmentos)']))
    resultados['mean_RMSE'].append(np.mean(dados_estat['RMSE (erro entre a série inteira e a predição de todos os segmentos)']))
    resultados['std_RMSE'].append(np.std(dados_estat['RMSE (erro entre a série inteira e a predição de todos os segmentos)']))
    resultados['mean_std_comp'].append(np.mean(dados_estat['desvio padrão complexidade']))
    resultados['std_std_comp'].append(np.std(dados_estat['desvio padrão complexidade']))
    resultados['mean_tempo'].append(np.mean(dados_estat['tempo']))
    resultados['std_tempo'].append(np.std(dados_estat['tempo']))
    resultados['mean_num_seg'].append(np.mean(dados_estat['numero de segmentos']))
    resultados['std_num_seg'].append(np.std(dados_estat['numero de segmentos']))
    resultados['num_res'].append(len(dados_estat))
    resultados['mean_ent_gain'].append(np.mean(dados_estat['ganho médio de entropia por corte']))
    resultados['std_ent_gain'].append(np.std(dados_estat['ganho médio de entropia por corte']))

  estatisticas_df = pd.DataFrame(resultados, index=modelos)

  # Pegando lista de cores
  colors = plt.colormaps['magma'].reversed()(np.linspace(0, 1, 10)[1:-3])

  # Criando disposição dos plots e tamanho da figura
  fig, ((axMAE, axRMSE, axTempo), (axNseg, axComp, axStdComp)) = plt.subplots(nrows=2, ncols=3, figsize=(14,8))

  # Plotando as figuras nos axes
  axMAE.bar(modelos,  estatisticas_df.mean_MAE, width=0.4, yerr = estatisticas_df.std_MAE, color=colors)
  axMAE.set_xlabel('Model')
  axMAE.tick_params(axis='x', labelrotation=45)
  axMAE.set_ylabel('MAE')
  axMAE.set_ylim((
    0,
    max([estatisticas_df['mean_MAE'][model]+estatisticas_df['std_MAE'][model] for model in modelos]) + 0.5,  
  ))
  axMAE.set_title('MAE (Less is better)')

  axComp.bar(modelos,  estatisticas_df.mean_comp, width=0.4, yerr = estatisticas_df.std_comp, color=colors)
  axComp.set_xlabel('Model')
  axComp.tick_params(axis='x', labelrotation=45)
  axComp.set_ylabel('Complexity')
  axComp.set_title('Complexity (Less is better)')

  axStdComp.bar(modelos,  estatisticas_df.mean_std_comp, width=0.4, yerr = estatisticas_df.std_std_comp, color=colors)
  axStdComp.set_xlabel('Model')
  axStdComp.tick_params(axis='x', labelrotation=45)
  axStdComp.set_ylabel('Complexity standard deviation')
  axStdComp.set_ylim((
    min([estatisticas_df['mean_std_comp'][model]-estatisticas_df['std_std_comp'][model] for model in modelos]) - 0.5,
    max([estatisticas_df['mean_std_comp'][model]+estatisticas_df['std_std_comp'][model] for model in modelos]) + 0.5,  
  ))
  axStdComp.set_title('Complexity standard deviation\nof segments (Less is more consistent)')


  axRMSE.bar(modelos,  estatisticas_df.mean_RMSE, width=0.4, yerr = estatisticas_df.std_RMSE, color=colors)
  axRMSE.set_xlabel('Model')
  axRMSE.tick_params(axis='x', labelrotation=45)
  axRMSE.set_ylabel('RMSE')
  axRMSE.set_ylim((
    0,
    max([estatisticas_df['mean_RMSE'][model]+estatisticas_df['std_RMSE'][model] for model in modelos]) + 0.5,  
  ))
  axRMSE.set_title('RMSE (Less is better)')

  axNseg.bar(modelos,  estatisticas_df.mean_num_seg, width=0.4, yerr = estatisticas_df.std_num_seg, color=colors)
  axNseg.set_xlabel('Model')
  axNseg.tick_params(axis='x', labelrotation=45)
  axNseg.set_ylabel('Number of segments')
  axNseg.set_ylim((
    0,
    max([estatisticas_df['mean_num_seg'][model]+estatisticas_df['std_num_seg'][model] for model in modelos]) + 0.5,
  ))
  axNseg.set_title('Number of segments (Less is better)')

  axTempo.bar(modelos,  estatisticas_df.mean_tempo, width=0.4, yerr = estatisticas_df.std_tempo, color=colors)
  axTempo.set_xlabel('Model')
  axTempo.tick_params(axis='x', labelrotation=45)
  axTempo.set_ylabel('SymbReg execution time')
  axTempo.set_ylim((
    0,
    max([estatisticas_df['mean_tempo'][model]+estatisticas_df['std_tempo'][model] for model in modelos]) + 0.5,
  ))
  axTempo.set_title('SymbReg segment execution time (Less is better)')

  # Ajuste de espaço vertical, salva a imagem e plot final
  plt.subplots_adjust(hspace=0.8, wspace=0.4)
  plt.savefig(f'resultados_{len(modelos)}_modelos_{length}.pdf', bbox_inches='tight')
  print('Tamanho ', length)
  print(estatisticas_df.num_res)
