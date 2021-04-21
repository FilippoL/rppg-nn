import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

# %%
path_to = os.path.join(os.path.dirname(__file__), "data", "data_out")

columns = ["confidence", "face_found", "time"]
HOG_full = pd.read_csv(f"{path_to}\\HOG_performance.csv", names=columns)
HOG_small = pd.read_csv(f"{path_to}\\HOG_performance_small_dataset.csv", names=columns)

SSD_full = pd.read_csv(f"{path_to}\\SSD_performance.csv", names=columns)
SSD_small = pd.read_csv(f"{path_to}\\SSD_performance_small_dataset.csv", names=columns)

MTCNN_full = pd.read_csv(f"{path_to}\\MTCNN_performance.csv", names=columns)
MTCNN_small = pd.read_csv(f"{path_to}\\MTCNN_performance_small_dataset.csv", names=columns)


# %%

def plot_pie_chart(array_of_booleans, save=False, model_name=""):
    if save and model_name == "":
        print("If save is true, need a model name for image name.")
    labels = 'Not Detected', 'Detected'
    count = Counter(array_of_booleans)
    sizes = [count[0], count[1]]
    explode = (0, 0.075)
    plt.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%', )
    plt.axis('equal')
    plt.title(f"{model_name}, {len(array_of_booleans)} samples.")
    if save: plt.savefig(f"{model_name}_piechart_detected_{len(array_of_booleans)}.jpg", bbox_inches='tight')
    plt.show()


def plot_histogram(array_of_confidence, save=False, model_name="", value_label=""):
    if save and model_name == "":
        print("If save is true, need a model name for image name.")
    plt.hist(x=array_of_confidence, color='purple', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f"{model_name} {value_label}, {len(array_of_confidence)} samples.")
    if save: plt.savefig(f"{model_name}_{value_label}_detected_{len(array_of_confidence)}.jpg")
    plt.show()


# %%
plot_pie_chart(HOG_full["face_found"], True, "HOG")
plot_pie_chart(HOG_small["face_found"], True, "HOG")

plot_pie_chart(SSD_full["face_found"], True, "SSD")
plot_pie_chart(SSD_small["face_found"], True, "SSD")

plot_pie_chart(MTCNN_full["face_found"], True, "MTCNN")
plot_pie_chart(MTCNN_small["face_found"], True, "MTCNN")

# %%
plot_histogram(HOG_full["confidence"], True, "HOG", "confidence")
plot_histogram(HOG_small["confidence"], True, "HOG", "confidence")

plot_histogram(SSD_full["confidence"], True, "SSD", "confidence")
plot_histogram(SSD_small["confidence"], True, "SSD", "confidence")

plot_histogram(MTCNN_full["confidence"], True, "MTCNN", "confidence")
plot_histogram(MTCNN_small["confidence"], True, "MTCNN", "confidence")

# %%

plot_histogram(HOG_full["time"], True, "HOG", "time")
plot_histogram(HOG_small["time"], True, "HOG", "time")

plot_histogram(SSD_full["time"], True, "SSD", "time")
plot_histogram(SSD_small["time"], True, "SSD", "time")

plot_histogram(MTCNN_full["time"], True, "MTCNN", "time")
plot_histogram(MTCNN_small["time"], True, "MTCNN", "time")
