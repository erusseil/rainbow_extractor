import rainbow_extractor as rx
import pandas as pd
import os
import numpy as np
from multiprocessing import Pool

folder = '/media/ELAsTICC/transfer_elasticc_all_SN_like/'
files = os.listdir(folder)
save_name = "features/elasticc_features"
n_cores = 15

lcs = [rx.ELAsTiCC_lightcurve.read_parquet_withclass(folder + file + "/", n=1000000) for file in files]
lcs = [element for sublist in lcs for element in sublist]
splits = np.linspace(0, len(lcs), n_cores + 1, dtype=int)


def extract_and_save(i, filename):
    
    sub_lcs = lcs[splits[int(i)]:splits[int(i)+1]]
    fex = rx.FeatureExtractor(pd.DataFrame(sub_lcs))
    fex.format_all()
    fex.full_feature_extraction()
    fex.features.to_pickle(filename)

def worker(i):
    fname = f"{save_name}_{i}.pkl"
    extract_and_save(i, fname)
    
def main():
    with Pool(n_cores) as pool:
        pool.map(worker, range(n_cores))

if __name__ == "__main__":
    main()