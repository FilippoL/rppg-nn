import _pickle as cPickle
import os

import biosppy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

import heartpy as hp

differences = []
variances = []
paths = []
hrs_dataset = {"hr": [], "filtered_hr": [], "path": []}
differences_filtered = []
variances_filtered = []

out_data = {"differences":[],
          "variances":[],
          "deviations":[],
          "paths":[],
          "differences_filtered":[],
          "deviations_filtered":[],
          "variances_filtered":[],
}

root_path = "H:\\DatasetsThesis\\COHFACE"
# root_path = "/data/data_in/sample_COHFACE"

for subdir, dirs, files in os.walk(root_path):
    for file in files:
        if file.endswith("hdf5"):
            data = h5py.File(os.path.join(subdir,file), "r")
            pulse = data["pulse"]
            hr = biosppy.ppg.ppg(pulse, 256, show=False)
            out_data["differences"].append(abs(np.mean(hr[-1]) - len(hr[-1])))
            out_data["variances"].append(np.var(hr[-1]))
            out_data["deviations"].append(np.std(hr[-1]))
            hrs_dataset["hr"].append(hr[-1])

            filtered = hp.filter_signal(pulse,
                                        cutoff=[0.8, 2.5],
                                        filtertype='bandpass',
                                        sample_rate=256,
                                        order=3,
                                        return_top=False)
            working_data, measures = hp.process(filtered, sample_rate=256.0,
                                                high_precision=True, clean_rr=True)

            result = np.where(working_data["binary_peaklist"] == 1)
            peaks = [working_data["peaklist"][idx] for idx in result[0]]
            hr_filtered = [60 / ((j - i) / 256) for i, j in zip(peaks[:-1], peaks[1:])]
            hrs_dataset["filtered_hr"].append(hr_filtered)
            hrs_dataset["path"].append(os.path.join(subdir, file))

            out_data["differences_filtered"].append(abs(len(hr_filtered) - np.mean(hr_filtered)))
            out_data["variances_filtered"].append(np.var(hr_filtered))
            out_data["deviations_filtered"].append(np.std(hr_filtered))
            out_data["paths"].append(os.path.join(subdir, file))

            plt.plot(hr[-1],label='HR')
            plt.plot(hr_filtered,label='HR filtered')
            plt.legend()
            plt.savefig(os.path.join(subdir,"plots", "hr_plot_newbiosppy.jpg"))
            plt.close('all')


df = pd.DataFrame(out_data)

df.to_csv("hr_analysis_biosppy_new.csv")

with open(r"hr_dataset_newbiosppy.pickle", "wb") as output_file:
    cPickle.dump(hrs_dataset, output_file)
