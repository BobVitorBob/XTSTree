# import pandas as pd
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# from scipy import stats


# if len(sys.argv) > 1:
#   nome_arq = sys.argv[1]
# else:
#   nome_arq = 'resultados backup.csv'
# # Lendo dados e criando lista de modelos e estatísticas de interesse

# dados = pd.read_csv(nome_arq).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').reset_index(drop=True)

# print(len(dados))
# print(len(dados['nome'].unique()))
# modelos = [
#   'full',
#   'PageHinkley',
#   'PeriodicCut',
#   'RandomCut',
#   'TopDownReg',
#   'TopDownIndex',
# ]
# resultados = {
#   'mean_comp': [],
#   'std_comp': [],
#   'mean_MAE': [],
#   'std_MAE': [],
#   'mean_RMSE': [],
#   'std_RMSE': [],
#   'mean_std_comp': [],
#   'std_std_comp': [],
#   'mean_tempo': [],
#   'std_tempo': [],
#   'mean_num_seg': [],
#   'std_num_seg': [],
#   'num_res': [],
#   'mean_ent_gain': [],
#   'std_ent_gain': [],
# }

# # Calculando estatísticas
# dados = list(filter(lambda indexrow: int(indexrow[1]['numero de segmentos']) != 1, dados.iterrows()))
# print(len(dados))
# # for model in modelos:
# #   dados_estat = dados[dados.model.str.startswith(model)]
# #   resultados['mean_comp'].append(np.mean(dados_estat['complexidade (média dos segmentos)']))
# #   resultados['std_comp'].append(np.std(dados_estat['complexidade (média dos segmentos)']))
# #   resultados['mean_MAE'].append(np.mean(dados_estat['MAE (erro entre a série inteira e a predição de todos os segmentos)']))
# #   resultados['std_MAE'].append(np.std(dados_estat['MAE (erro entre a série inteira e a predição de todos os segmentos)']))
# #   resultados['mean_RMSE'].append(np.mean(dados_estat['RMSE (erro entre a série inteira e a predição de todos os segmentos)']))
# #   resultados['std_RMSE'].append(np.std(dados_estat['RMSE (erro entre a série inteira e a predição de todos os segmentos)']))
# #   resultados['mean_std_comp'].append(np.mean(dados_estat['desvio padrão complexidade']))
# #   resultados['std_std_comp'].append(np.std(dados_estat['desvio padrão complexidade']))
# #   resultados['mean_tempo'].append(np.mean(dados_estat['tempo']))
# #   resultados['std_tempo'].append(np.std(dados_estat['tempo']))
# #   resultados['mean_num_seg'].append(np.mean(dados_estat['numero de segmentos']))
# #   resultados['std_num_seg'].append(np.std(dados_estat['numero de segmentos']))
# #   resultados['num_res'].append(len(dados_estat))
# #   resultados['mean_ent_gain'].append(np.mean(dados_estat['ganho médio de entropia por corte']))
# #   resultados['std_ent_gain'].append(np.std(dados_estat['ganho médio de entropia por corte']))

# # estatisticas_df = pd.DataFrame(resultados, index=modelos)

# # # Pegando lista de cores
# # colors = plt.colormaps['magma'].reversed()(np.linspace(0, 1, 10)[1:-3])

# # # Criando disposição dos plots e tamanho da figura
# # fig, ((axMAE, axRMSE, axTempo, axNseg), (axComp, axStdComp, axEntGain, _)) = plt.subplots(nrows=2, ncols=4, figsize=(20,12))

# # # Plotando as figuras nos axes
# # axMAE.bar(modelos,  estatisticas_df.mean_MAE, width=0.4, yerr = estatisticas_df.std_MAE, color=colors)
# # axMAE.set_xlabel('modelo')
# # axMAE.tick_params(axis='x', labelrotation=45)
# # axMAE.set_ylabel('MAE')
# # axMAE.set_ylim((0, 20))
# # axMAE.set_title('MAE (Menor é melhor)')

# # axComp.bar(modelos,  estatisticas_df.mean_comp, width=0.4, yerr = estatisticas_df.std_comp, color=colors)
# # axComp.set_xlabel('modelo')
# # axComp.tick_params(axis='x', labelrotation=45)
# # axComp.set_ylabel('Complexidade')
# # axComp.set_title('Complexidade (Menor é melhor)')

# # axStdComp.bar(modelos,  estatisticas_df.mean_std_comp, width=0.4, yerr = estatisticas_df.std_std_comp, color=colors)
# # axStdComp.set_xlabel('modelo')
# # axStdComp.tick_params(axis='x', labelrotation=45)
# # axStdComp.set_ylabel('Desvio padrão Comp.')
# # axStdComp.set_ylim((0, 20))
# # axStdComp.set_title('Desvio padrão da complexidade\ndos segmentos (Menor é mais consistente)')


# # axRMSE.bar(modelos,  estatisticas_df.mean_RMSE, width=0.4, yerr = estatisticas_df.std_RMSE, color=colors)
# # axRMSE.set_xlabel('modelo')
# # axRMSE.tick_params(axis='x', labelrotation=45)
# # axRMSE.set_ylabel('RMSE')
# # axRMSE.set_ylim((0, 30))
# # axRMSE.set_title('RMSE (Menor é melhor)')

# # axNseg.bar(modelos,  estatisticas_df.mean_num_seg, width=0.4, yerr = estatisticas_df.std_num_seg, color=colors)
# # axNseg.set_xlabel('modelo')
# # axNseg.tick_params(axis='x', labelrotation=45)
# # axNseg.set_ylabel('Num. de segmentos')
# # axNseg.set_ylim((0, 20))
# # axNseg.set_title('Número de segmentos (Menor é melhor)')

# # axTempo.bar(modelos,  estatisticas_df.mean_tempo, width=0.4, yerr = estatisticas_df.std_tempo, color=colors)
# # axTempo.set_xlabel('modelo')
# # axTempo.tick_params(axis='x', labelrotation=45)
# # axTempo.set_ylabel('Tempo de exec. SymbReg')
# # axTempo.set_ylim((0, 400))
# # axTempo.set_title('Tempo de execução nos segmentos (Menor é melhor)')

# # axEntGain.bar(modelos,  estatisticas_df.mean_ent_gain, width=0.4, yerr = estatisticas_df.std_ent_gain, color=colors)
# # axEntGain.set_xlabel('modelo')
# # axEntGain.tick_params(axis='x', labelrotation=45)
# # axEntGain.set_ylabel('Ganho de entropia')
# # axEntGain.set_ylim((-.5, .5))
# # axEntGain.set_title('Ganho médio de entropia por corte')

# # # Ajuste de espaço vertical, salva a imagem e plot final
# # plt.subplots_adjust(hspace=0.4, wspace=0.2)
# # plt.savefig(f'{nome_arq[:-4]}.pdf', bbox_inches='tight')
# # plt.show()

# # print(estatisticas_df.num_res)
