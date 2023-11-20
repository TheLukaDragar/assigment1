import subprocess
import argparse
from sklearn.model_selection import ParameterGrid
import tqdm
import os

# Define the grid search parameters
grid = {
    "lbptype": ["default", "uniform", "rotation"],
    "radius": [1, 2],
    "neighbors": [8, 16],
    "histograms": [False, True],
    "resize_to": [24,64, 128],
    "use_lib": [False, True],
    "use_raw": [False, True]
}

#one more flag that is one element in the 

# Optional flags are simply True or False
# optional_flags = {"save": [False, True]}


def run_lbp_script(params):
    # Construct the command with parameters
    command = ["/Users/carbs/miniforge3/envs/ai/bin/python", "LBP.py"]

    # Add each parameter to the command
    for param, value in params.items():
        if isinstance(value, bool):
            # For true boolean flags, add the flag to the command
            if value:
                command.append(f"--{param}")
        else:
            # For non-boolean parameters, add both the flag and its value
            command.append(f"--{param}")
            command.append(str(value))

    # Add optional flags if they are True
    # for flag, is_active in optional_flags.items():
    #     if is_active:
    #         command.append(f"--{flag}")

    # Run the script with the current parameters

    # print(" ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)

    return result


# Run grid search
results = []

print(f"Grid size: {len(ParameterGrid(grid))}")
with open("results_LBP.txt", "w") as f:
    for params in tqdm.tqdm(list(ParameterGrid(grid))):
            print(f"Running with params: {params} ")
            result = run_lbp_script(params)

            print("Result:", result.stdout.split('Accuracy: ')[-1])
            if result.stderr:
                print("Error:", result.stderr)

            results.append(f"{params},{result.stdout.split('Accuracy: ')[-1]}")

            #save results
            f.write(str(params))
            f.write(",")
            f.write(str(result.stdout.split('Accuracy: ')[-1]))
            f.write("\n")

            #write imidiate results to file
            f.flush()
            os.fsync(f.fileno())





            

# Save or process the collected results from the grid search
# ...
