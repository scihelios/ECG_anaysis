import numpy as np
import extremum as ext
import os
import pandas as pd




input_file = f'data/1'
frequence = 500


i_max_pic = 0
for signal in os.listdir(f'{input_file}/beats/'):
    indice_enregistrement = int(signal.split(".")[0])
    if indice_enregistrement > 100 or indice_enregistrement <61:
        continue
    data = []
    for beat_id in os.listdir(f'{input_file}/beats/{signal}'):
        if beat_id.endswith('.npy'):
            beat = np.load(f'{input_file}/beats/{signal}/{beat_id}')
            i = np.argmax(beat)
            param, _ = ext.gradient_descent_calibre(beat)
            indice_battement = int(beat_id.split(".")[0])
            data.append([indice_enregistrement, indice_battement] +param.get_params()+[(i+i_max_pic)/frequence])
            i_max_pic = len(beat) - i
    data.sort(key=lambda x: x[1])

    data = pd.DataFrame(data)
    data.columns = ['Numéro enregistrement'] + ['Numéro battement'] + [f'Amplitude {i}' for i in range(1,6)] + [f'Centre {i}' for i in range(1,6)] + [f'Ecart-type {i}' for i in range(1,6)] + ['Période interpics']
    print(f'Enregistrement {indice_enregistrement} : {len(data)} battements')
    data.to_csv(f'{input_file}/parametres/{indice_enregistrement}.csv', index=False)



