import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#%%
columns = ["confidence", "face_found", "time"]
HOG_full = pd.read_csv("data/data_out/HOG_performance.csv", names=columns)
HOG_small = pd.read_csv("data/data_out/HOG_performance_small_dataset.csv", names=columns)

SSD_full = pd.read_csv("data/data_out/SSD_performance.csv", names=columns)
SSD_small = pd.read_csv("data/data_out/SSD_performance_small_dataset.csv", names=columns)

MTCNN_full = pd.read_csv("data/data_out/MTCNN_performance.csv", names=columns)
MTCNN_small = pd.read_csv("data/data_out/MTCNN_performance_small_dataset.csv", names=columns)
#%%

