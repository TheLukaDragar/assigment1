import matplotlib.pyplot as plt
import ast
import re
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Function to parse a single line of the file
def parse_line(line):
    # Strip the newline character and split at the comma
    line = line.strip().split('},')

    #remove \n
    line = [l.replace("\\n","") for l in line]
    # Take the first part, add the closing bracket back since we split at "},"
    dict_part = line[0] + '}'
    # Convert string representation of dictionary to an actual dictionary
    parsed_dict = ast.literal_eval(dict_part)
    # Now handle the byte string, decode it, and parse it
    byte_str = line[1].lstrip('b"').rstrip('"\n')
    byte_str_dict = ast.literal_eval(byte_str)
    # Merge both dictionaries into one
    merged_dict = {**parsed_dict, **byte_str_dict}

   
    print(merged_dict)
    #get first float from string
    merged_dict['avg_iou'] = float(re.findall("\d+\.\d+", merged_dict['avg_iou'])[0])
    merged_dict['avg_iou_detected'] = float(re.findall("\d+\.\d+", merged_dict['avg_iou_detected'])[0])
    print(merged_dict)
    return merged_dict

# Function to read data from a file and parse it
def read_data_from_file(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(parse_line(line))
    return data

# Function to plot the graphs
def plot_data(data):
    scale_factors = [d['scaleFactor'] for d in data]
    detected = [d['detected'] for d in data]
    not_detected = [d['not_detected'] for d in data]
    avg_iou = [d['avg_iou'] for d in data]
    avg_iou_detected = [d['avg_iou_detected'] for d in data]

    # # Plotting
    # fig, ax1 = plt.subplots()

    # #set y axis limits
    # ax1.set_ylim(0, 500)

    # color = 'tab:red'
    # ax1.set_xlabel('Scale Factor')
    # ax1.set_ylabel('Detected/Not Detected', color=color)
    # # ax1.plot(scale_factors, detected, label='Detected', color='r', marker='o')
    # ax1.plot(scale_factors, not_detected, label='Not Detected', color='r', linestyle='--', marker='x')
    # ax1.tick_params(axis='y', labelcolor=color)
    # ax1.legend(loc='upper left')

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    # color = 'tab:blue'
    # ax2.set_ylabel('Average IOU', color=color)  # we already handled the x-label with ax1
    # ax2.plot(scale_factors, avg_iou, label='Avg IOU', color='b', marker='s')
    # # ax2.plot(scale_factors, avg_iou_detected, label='Avg IOU Detected', color='b', linestyle='--', marker='^')
    # ax2.tick_params(axis='y', labelcolor=color)
    # ax2.legend(loc='upper right')


    # #find index that maximeises detections and avg iou at the same time
    # loss = [detected[i] + avg_iou[i] for i in range(len(detected))]
    # max_loss = max(loss)
    # max_loss_index = loss.index(max_loss)
    # print(f"Max loss at index {max_loss_index}")

    # #plot
    # plt.axvline(x=scale_factors[max_loss_index], color='k', linestyle='--')
    # plt.text(scale_factors[max_loss_index], 0, f"Max loss at {scale_factors[max_loss_index]}", rotation=90)

    #SAVE
    
    





    # fig.tight_layout()  # otherwise the

    # plt.show()

 


    #plot only the one with the highest avg iou
    highest_avg_iou = max(avg_iou)
    highest_avg_iou_index = avg_iou.index(highest_avg_iou)
    print(f"Highest avg iou at index {highest_avg_iou_index}")
    print(f"Avg iou: {highest_avg_iou}")

    print(f"Scale factor: {scale_factors[highest_avg_iou_index]}")
    print(f"minNeighbors: {data[highest_avg_iou_index]['minNeighbors']}")
    print(f"minSize: {data[highest_avg_iou_index]['minSize']}")
    


    #print parameters
    print(f"Detected: {detected[highest_avg_iou_index]}")
    print(f"Not detected: {not_detected[highest_avg_iou_index]}")
    print(f"Avg iou detected: {avg_iou_detected[highest_avg_iou_index]}")

    #plot
    fig, ax1 = plt.subplots()

    fig.suptitle(f"Scale factor: {scale_factors[highest_avg_iou_index]}")
    ax1.set_ylim(0, 500)

    color = 'tab:red'
    ax1.set_xlabel('Scale Factor')
    ax1.set_ylabel('Detected/Not Detected', color=color)
    ax1.plot(scale_factors, detected, label='Detected', color='r', marker='o')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Average IOU', color=color)  # we already handled the x-label with ax1
    ax2.plot(scale_factors, avg_iou, label='Avg IOU', color='b', marker='s')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    #plot
    plt.axvline(x=scale_factors[highest_avg_iou_index], color='k', linestyle='--')
    # plt.text(scale_factors[highest_avg_iou_index], 0, f"Max loss at {scale_factors[highest_avg_iou_index]}", rotation=90)

    fig.tight_layout()  # otherwise the

    plt.savefig("analysis_VJ_highest_avg_iou.png")


    plt.show()

















# Main function to run the script
def main():
    file_path = 'results.txt'  # Replace with your actual file path
    data = read_data_from_file(file_path)
    plot_data(data)

if __name__ == '__main__':
    main()
