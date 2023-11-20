import matplotlib.pyplot as plt
import ast
import re

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Function to parse a single line of the file
def parse_line(line):
    # Split the line at the last comma to separate the dictionary part from the score
    dict_part, acc = line.rsplit(',', 1)

    # Remove any trailing whitespace or newline characters from the score part
    acc = acc.strip()

    # Convert string representation of dictionary to an actual dictionary
    parsed_dict = ast.literal_eval(dict_part)

    return {**parsed_dict, 'score': float(acc)}


# Function to read data from a file and parse it
def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            #check if line is empty
            if line.strip():
                
                data.append(parse_line(line))

            
    return data

# Function to plot the graphs
def plot_data(data):
    params = ['histograms', 'lbptype', 'neighbors', 'radius', 'resize_to', 'use_lib','use_raw']
    scores = [d['score'] for d in data]

    # Plotting using plotly use different colors for runs withch contain use_lib or use_raw

    df = pd.DataFrame(data)
    df['use_lib'] = df['use_lib'].astype(bool)
    df['use_raw'] = df['use_raw'].astype(bool)

    #plot scores and use different colors for runs with use_lib or use_raw
    fig = go.Figure()

    colors = ['red', 'blue','green']

    #use raw_is blue and takes precedence over use_lib
    #use lib is red
    #everything else is green

    df['color'] = 'green'
    df.loc[df['use_lib'] == True, 'color'] = 'red'
    df.loc[df['use_raw'] == True, 'color'] = 'blue'


    #sort df so that parameters are in the same order
    sorting_keys = ['histograms', 'lbptype', 'neighbors', 'radius', 'resize_to', 'score', 'use_lib', 'use_raw']
    df = df.sort_values(by=sorting_keys)    





    fig.add_trace(go.Scatter(x=df.index, y=df['score'], mode='markers', marker_color=df['color'], name='Scores', marker=dict(size=12)))
   
    fig.update_layout(title='Scores - blue: raw_comparison, red: lib_lbp, green: my_lbp', xaxis_title='Run', yaxis_title='Score')

    #dark background
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        font_color="white"
    )



    




    fig.show()

    

    # Find the parameter set with the highest score
    highest_score_index = scores.index(max(scores))
    print(f"Highest score at index {highest_score_index}")
    print(f"Score: {scores[highest_score_index]}")

    # Print parameters of the best set
    print(f"Best parameter set: {data[highest_score_index]}")

    #print top 5
    top5 = df.sort_values(by=['score'], ascending=False).head(5)
    print("Top 5:")
    for index, row in top5.iterrows():
        print(f"{index}: {row['score']} - histograms: {row['histograms']}, lbptype: {row['lbptype']}, neighbors: {row['neighbors']}, radius: {row['radius']}, resize_to: {row['resize_to']}, use_lib: {row['use_lib']}, use_raw: {row['use_raw']}")


# Main function to run the script
def main():
    file_path = 'results_LBP.txt'  # Replace with your actual file path
    data = read_data_from_file(file_path)
    print(data)
    plot_data(data)

if __name__ == '__main__':
    main()
