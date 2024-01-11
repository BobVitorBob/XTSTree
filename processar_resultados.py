import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) > 1:
  nome_arq = sys.argv[1]
else:
  nome_arq = 'resultados.csv'
# Lendo dados e criando lista de modelos e estatísticas de interesse

dados = pd.read_csv(nome_arq).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').reset_index(drop=True)
modelos = [
  'full',
  'PageHinkley',
  'PeriodicCut',
  'RandomCut',
  'TopDownReg',
  'TopDownIndex',
]
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
}

# Calculando estatísticas
for model in modelos:
  dados_estat = dados[dados.model.str.startswith(model)]
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

estatisticas_df = pd.DataFrame(resultados, index=modelos)

# Pegando lista de cores
colors = plt.colormaps['magma'].reversed()(np.linspace(0, 1, 10)[1:-3])

# Criando disposição dos plots e tamanho da figura
fig, ((axMAE, axRMSE, axNseg), (axComp, axStdComp, axTempo)) = plt.subplots(nrows=2, ncols=3, figsize=(16,9))

# Plotando as figuras nos axes
axMAE.bar(modelos,  estatisticas_df.mean_MAE, width=0.4, yerr = estatisticas_df.std_MAE, color=colors)
axMAE.set_xlabel('modelo')
axMAE.tick_params(axis='x', labelrotation=45)
axMAE.set_ylabel('MAE')
axMAE.set_ylim((0, 20))
axMAE.set_title('MAE (Menor é melhor)')

axComp.bar(modelos,  estatisticas_df.mean_comp, width=0.4, yerr = estatisticas_df.std_comp, color=colors)
axComp.set_xlabel('modelo')
axComp.tick_params(axis='x', labelrotation=45)
axComp.set_ylabel('Complexidade')
axComp.set_title('Complexidade (Menor é melhor)')

axStdComp.bar(modelos,  estatisticas_df.mean_std_comp, width=0.4, yerr = estatisticas_df.std_std_comp, color=colors)
axStdComp.set_xlabel('modelo')
axStdComp.tick_params(axis='x', labelrotation=45)
axStdComp.set_ylabel('Desvio padrão Comp.')
axStdComp.set_ylim((0, 20))
axStdComp.set_title('Desvio padrão da complexidade\ndos segmentos (Menor é mais consistente)')


axRMSE.bar(modelos,  estatisticas_df.mean_RMSE, width=0.4, yerr = estatisticas_df.std_RMSE, color=colors)
axRMSE.set_xlabel('modelo')
axRMSE.tick_params(axis='x', labelrotation=45)
axRMSE.set_ylabel('RMSE')
axRMSE.set_ylim((0, 30))
axRMSE.set_title('RMSE (Menor é melhor)')

axNseg.bar(modelos,  estatisticas_df.mean_num_seg, width=0.4, yerr = estatisticas_df.std_num_seg, color=colors)
axNseg.set_xlabel('modelo')
axNseg.tick_params(axis='x', labelrotation=45)
axNseg.set_ylabel('Num. de segmentos')
axNseg.set_ylim((0, 20))
axNseg.set_title('Número de segmentos (Menor é melhor)')

axTempo.bar(modelos,  estatisticas_df.mean_tempo, width=0.4, yerr = estatisticas_df.std_tempo, color=colors)
axTempo.set_xlabel('modelo')
axTempo.tick_params(axis='x', labelrotation=45)
axTempo.set_ylabel('Tempo de exec. SymbReg')
axTempo.set_ylim((0, 400))
axTempo.set_title('Tempo de execução nos segmentos (Menor é melhor)')

# Ajuste de espaço vertical, salva a imagem e plot final
plt.subplots_adjust(hspace=0.6)
plt.savefig('resultados.pdf', bbox_inches='tight')
plt.show()

print(estatisticas_df.num_res)