import numpy as np
import extremum as ext
import os
import pandas as pd




input_file = f'data/1'

data = []

for signal in os.listdir(f'{input_file}/beats/'):
    for beat_id in os.listdir(f'{input_file}/beats/{signal}'):
        if beat_id.endswith('.npy'):
            beat = np.load(f'{input_file}/beats/{signal}/{beat_id}')
            param, _ = ext.gradient_descent_calibre(beat)
            data.append([int(beat_id.split(".")[0])] +param.get_params())
    break

data.sort(key=lambda x: x[0])

data = pd.DataFrame(data)
print(data)
data.to_csv(f'{input_file}/data.csv', index=False)



