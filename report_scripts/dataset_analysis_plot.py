import _pickle as cPickle
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# %%
def plot_histogram_bbox(data, save=False, path=""):
    data = [d[:2] for d in data]
    c = Counter(data)
    plt.figure(figsize=(20, 10))
    plt.bar([str(k) for k in c.keys()], list(c.values()))
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(range(1, np.max(list(c.values())), 10), fontsize=12)
    plt.title(f"COHFACE, bounding box sizes for {len(data)} data points.", fontsize=20)
    plt.show() if not save else plt.savefig(f"{path}\\cohface_bbox_sizes.jpg")


def plot_line_for_distance(data, save=False, path=""):
    max = np.max(data)
    normalised = [d / max for d in data]
    plt.figure(figsize=(20, 10))
    plt.plot(normalised, linewidth=5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim([0, 1])
    plt.title(f"COHFACE, distance nose tip from center minmax normalised, "
              f"{len(data)} data points.", fontsize=20)
    plt.show() if not save else plt.savefig(f"{path}\\cohface_nose_center_distance.jpg")


def plot_lines_for_orientation(data, save=False, path="", m=2):
    roll = np.asarray(data["roll"], dtype=np.float)
    pitch = np.asarray(data["pitch"], dtype=np.float)
    yaw = np.asarray(data["yaw"], dtype=np.float)

    roll = roll[abs(roll - np.mean(roll)) < m * np.std(roll)]
    pitch = pitch[abs(pitch - np.mean(pitch)) < m * np.std(pitch)]
    yaw = yaw[abs(yaw - np.mean(yaw)) < m * np.std(yaw)]

    plt.figure(figsize=(20, 10))
    plt.plot(roll, linewidth=5, color='red', label="Roll")
    plt.plot(pitch, linewidth=5, color="green", label="Pitch")
    plt.plot(yaw, linewidth=5, color="blue", label="Yaw")
    plt.legend(prop={'size': 20})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim([-50, 50])
    plt.title(f"COHFACE, roll, pitch and yaw of subject head, "
              f"{len(pitch)} data points.", fontsize=20)
    plt.show() if not save else plt.savefig(f"{path}\\cohface_orientation.jpg")


def plot_pie_chart(number_not_detected, total_number, save=False, path=""):
    labels = 'Not Detected', 'Detected'
    sizes = [number_not_detected, total_number - number_not_detected]
    explode = (0, 0.075)
    plt.figure(figsize=(20, 10))
    plt.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%',
            textprops={'fontsize': 20})
    plt.axis('equal')
    plt.title(f"COHFACE, number detected vs not detected, "
              f"{total_number} data points.", fontsize=20, pad=20)
    plt.show() if not save else plt.savefig(f"{path}\\cohface_pie_detected.jpg")


def plot_luminance(data, save=False, path="", normalise=False):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    fig.suptitle(f'COHFACE per quadrant luminance, {len(data["top_right"])} datapoints', fontsize=20)
    if normalise:
        ax1.plot([d / np.max(data["top_right"]) for d in data["top_right"]], linewidth=5)
        ax2.plot([d / np.max(data["top_left"]) for d in data["top_left"]], 'tab:orange', linewidth=5)
        ax3.plot([d / np.max(data["bottom_right"]) for d in data["bottom_right"]], 'tab:green', linewidth=5)
        ax4.plot([d / np.max(data["bottom_left"]) for d in data["bottom_left"]], 'tab:red', linewidth=5)
    else:
        ax1.plot(data["top_right"], linewidth=5)
        ax2.plot(data["top_left"], 'tab:orange', linewidth=5)
        ax3.plot(data["bottom_right"], 'tab:green', linewidth=5)
        ax4.plot(data["bottom_left"], 'tab:red', linewidth=5)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(labelsize=20)
        ax.tick_params(labelsize=20)
        if normalise:
            ax.set_ylim((0, 1))
        else:
            ax.set_ylim((100, np.max([np.max(data[key]) for key in data.keys()])))
        ax.label_outer()
    plt.show() if not save else plt.savefig(f"{path}\\cohface_luminance.jpg")


def plot_bbox_size_coarse(data, save=False, path=""):
    widths = [d[0] for d in data]
    heights = [d[1] for d in data]
    plt.figure(figsize=(20, 10))
    plt.boxplot([widths, heights], widths=0.6, showmeans=True, patch_artist=True, meanline=True,
                medianprops={"linewidth": 5, "color": (0.4, 1, 0)},
                meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 5, "markeredgecolor": "blue",
                           "linewidth": 5, "color": "red"})
    plt.xticks([1, 2], ["Width", "Height"])
    plt.title(f"COHFACE, boxplot coarse data bbox width and height, "
              f"{len(widths)} data points.", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show() if not save else plt.savefig(f"{path}\\cohface_bbox_dimension_coarse.jpg")


def plot_box_distance_coarse(data, save=False, path=""):
    plt.figure(figsize=(20, 10))
    plt.boxplot(data, widths=0.6, showmeans=True, patch_artist=True, meanline=True,
                medianprops={"linewidth": 5, "color": (0.4, 1, 0)},
                meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 5, "markeredgecolor": "blue",
                           "linewidth": 5, "color": "red"})
    plt.xticks([1], ["Distance"])
    plt.title(f"COHFACE, boxplot coarse data distance nose to screen-center, "
              f"{len(data)} data points.", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show() if not save else plt.savefig(f"{path}\\cohface_distance_coarse.jpg")


def plot_luminance_coarse(data, save=False, path="", normalise=False):
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
    (ax1, ax2), (ax3, ax4) = gs.subplots(sharex='col', sharey='row')
    fig.suptitle(f'COHFACE per quadrant luminance coarse, {len(data["top_right"])} datapoints', fontsize=20)
    if normalise:
        ax1.boxplot([d / np.max(data["top_right"]) for d in data["top_right"]],showmeans=True,
                    medianprops={"linewidth": 2, "color": (0.4, 1, 0)},
                    meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 2, "markeredgecolor": "blue",
                               "linewidth": 2, "color": "red"})
        ax2.boxplot([d / np.max(data["top_left"]) for d in data["top_left"]],showmeans=True,
                    medianprops={"linewidth": 2, "color": (0.4, 1, 0)},
                    meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 2, "markeredgecolor": "blue",
                               "linewidth": 2, "color": "red"})
        ax3.boxplot([d / np.max(data["bottom_right"]) for d in data["bottom_right"]],showmeans=True,
                    medianprops={"linewidth": 2, "color": (0.4, 1, 0)},
                    meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 2, "markeredgecolor": "blue",
                               "linewidth": 2, "color": "red"})
        ax4.boxplot([d / np.max(data["bottom_left"]) for d in data["bottom_left"]],showmeans=True,
                    medianprops={"linewidth": 2, "color": (0.4, 1, 0)},
                    meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 2, "markeredgecolor": "blue",
                               "linewidth": 2, "color": "red"})
    else:
        ax1.boxplot(data["top_right"],showmeans = True,meanline=True, medianprops={"linewidth": 2, "color": (0.4, 1, 0)},
                    meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 2, "markeredgecolor": "blue",
                               "linewidth": 2, "color": "red"})
        ax2.boxplot(data["top_left"], showmeans = True,meanline=True,medianprops={"linewidth": 2, "color": (0.4, 1, 0)},
                    meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 2, "markeredgecolor": "blue",
                               "linewidth": 2, "color": "red"})
        ax3.boxplot(data["bottom_right"],showmeans = True,meanline=True, medianprops={"linewidth": 2, "color": (0.4, 1, 0)},
                    meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 2, "markeredgecolor": "blue",
                               "linewidth": 2, "color": "red"})
        ax4.boxplot(data["bottom_left"], showmeans = True,meanline=True,medianprops={"linewidth": 2, "color": (0.4, 1, 0)},
                    meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 2, "markeredgecolor": "blue",
                               "linewidth": 2, "color": "red"})

    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(labelsize=20)
        ax.tick_params(labelsize=20)
        ax.label_outer()
    plt.show() if not save else plt.savefig(f"{path}\\cohface_luminance_coarse.jpg")


def plot_orientation_coarse(data, save=False, path=""):
    roll = data["roll"]
    pitch = data["pitch"]
    yaw = data["yaw"]

    plt.figure(figsize=(20, 10))
    plt.boxplot([roll, pitch, yaw], widths=0.6, showmeans=True, patch_artist=True, meanline=True,
                medianprops={"linewidth": 5, "color": (0.4, 1, 0)},
                meanprops={"marker": "s", "markerfacecolor": "white", "markersize": 5, "markeredgecolor": "blue",
                           "linewidth": 5, "color": "red"})
    plt.xticks([1, 2, 3], ["Roll", "Pitch", "Yaw"])
    plt.title(f"COHFACE, boxplot coarse data head orientation, "
              f"{len(yaw)} data points.", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.show() if not save else plt.savefig(f"{path}\\cohface_head_orientation_coarse.jpg")


# %%
data_directory = "H:\\DatasetsThesis\\COHFACE"
loaded_dictionaries = []
for subdir, dirs, files in os.walk(data_directory):
    for file in files:
        if not file.endswith("pickle"):
            continue
        file_to_read = open(os.path.join(subdir, file), "rb")
        dictionary = cPickle.load(file_to_read)
        end_directory = os.path.join(subdir, "plots")
        # end_directory = "C:\\Users\\Filippo\\Desktop\\" + "\\".join(end_directory.split("\\")[2:])
        if not os.path.isdir(end_directory):
            os.mkdir(end_directory)
        if "aggregated" in file:
            plot_luminance_coarse(dictionary["luminance"], True, end_directory)
            plot_box_distance_coarse(dictionary["distance_nose_from_center"], True, end_directory)
            plot_orientation_coarse(dictionary["head_orientation"], True, end_directory)
            plot_bbox_size_coarse(dictionary["detected_bbox_size"], True, end_directory)

        else:
            plot_luminance(dictionary["luminance"], True, end_directory)
            plot_pie_chart(dictionary["not_detected"], len(dictionary["detected_bbox_size"]), True, end_directory)
            plot_line_for_distance(dictionary["distance_nose_from_center"], True, end_directory)
            plot_lines_for_orientation(dictionary["head_orientation"], True, end_directory)
            plot_histogram_bbox(dictionary["detected_bbox_size"], True, end_directory)

        file_to_read.close()
        plt.close('all')
        print(f"Created plots in {end_directory}")
