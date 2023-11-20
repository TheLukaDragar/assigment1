#implement grid search for VJ model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import tqdm

# VJ.py returns
# #results = {
#     #     "detected": len(ioius_of_detected),
#     #     "not_detected": len(files) - len(ioius_of_detected),
#     #     "iou": sum(ious)/len(ious),
#     #     "iou_detected": sum(ioius_of_detected)/len(ioius_of_detected)
#     # }


#sklearn grid search
from sklearn.model_selection import ParameterGrid

#grid search parameters
grid = {
    "scaleFactor": list(i for i in np.arange(1.01, 1.1, 0.01)),
    "minNeighbors": [2, 4, 6],
    "minSize": [20, 30, 40],
}

print(f"Grid size: {len(ParameterGrid(grid))}")
print(ParameterGrid(grid))

#run vj.py for each parameter combination
results = []

with open("results.txt", "w") as f:
    for params in tqdm.tqdm(list(ParameterGrid(grid))):
        # Here you would run your VJ.py with the current parameters
        # For example, using subprocess.run if VJ.py is a separate script:
        result = subprocess.run(['/Users/carbs/miniforge3/envs/ai/bin/python', 'VJ.py', '--scale-factor', str(params['scaleFactor']),
                                '--min-neighbors', str(params['minNeighbors']),
                                '--min-size', str(params['minSize'])], capture_output=True)
        results.append(result.stdout)
        print(result.stdout)

        #save results
        f.write(str(params))
        f.write(",")
        f.write(str(result.stdout))
        f.write("\n")

        #write imidiate results to file
        f.flush()
        os.fsync(f.fileno())

    

        



        print(f"Running with params: {params}")
        # ... (call VJ.py and append the result to results list)

