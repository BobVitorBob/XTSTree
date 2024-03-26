import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import stats


if len(sys.argv) > 1:
  nome_arq = sys.argv[1]
else:
  nome_arq = 'resultados_entropy.csv'
# Lendo dados e criando lista de modelos e estatísticas de interesse

dados = pd.read_csv(nome_arq).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').reset_index(drop=True)

dados = dados[(np.abs(stats.zscore(dados['ganho médio de entropia por corte'])) < 3)]

modelos = [
  'PageHinkley',
  'PeriodicCut',
  'RandomCut',
]
resultados = {
  'mean_ent_gain': [],
  'std_ent_gain': [],
  'num_res': [],
  'mean_num_seg': [],
  'std_num_seg': [],
}

# Calculando estatísticas
for model in modelos:
  dados_estat = dados[dados.model.str.startswith(model)]
  resultados['num_res'].append(len(dados_estat))
  resultados['mean_num_seg'].append(np.mean(dados_estat['numero de segmentos']))
  resultados['std_num_seg'].append(np.std(dados_estat['numero de segmentos']))
  resultados['mean_ent_gain'].append(np.mean(dados_estat['ganho médio de entropia por corte']))
  resultados['std_ent_gain'].append(np.std(dados_estat['ganho médio de entropia por corte']))

estatisticas_df = pd.DataFrame(resultados, index=modelos)

# Pegando lista de cores
colors = plt.colormaps['magma'].reversed()(np.linspace(0, 1, 10)[1:-3])

# Criando disposição dos plots e tamanho da figura
fig, ((axEntGain)) = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

print(estatisticas_df)

# axNseg.bar(modelos,  estatisticas_df.mean_num_seg, width=0.4, yerr = estatisticas_df.std_num_seg, color=colors)
# axNseg.set_xlabel('Model')
# axNseg.tick_params(axis='x', labelrotation=45)
# axNseg.set_ylabel('Number of segments')
# axNseg.set_ylim((
#   min([estatisticas_df['mean_num_seg'][model]-estatisticas_df['std_num_seg'][model] for model in modelos]) - 2,
#   max([estatisticas_df['mean_num_seg'][model]+estatisticas_df['std_num_seg'][model] for model in modelos]) + 2,
# ))
# axNseg.set_title('Number of segments (Smaller is better)')

axEntGain.bar(modelos, estatisticas_df.mean_ent_gain, yerr = estatisticas_df.std_ent_gain, color=colors, width=0.6)
axEntGain.set_xlabel('Model')
axEntGain.tick_params(axis='x', labelrotation=45)
axEntGain.set_ylabel('Entropy')
axEntGain.set_ylim((
  min([estatisticas_df['mean_ent_gain'][model]-estatisticas_df['std_ent_gain'][model] for model in modelos]) - 0.5,
  max([estatisticas_df['mean_ent_gain'][model]+estatisticas_df['std_ent_gain'][model] for model in modelos]) + 0.5,
))
axEntGain.set_title('Mean entropy by cut (Smaller is better)')

# Ajuste de espaço vertical, salva a imagem e plot final
plt.subplots_adjust(hspace=0.2, wspace=0.2)
plt.savefig(f'{nome_arq[:-4]}.pdf', bbox_inches='tight')
plt.show()

