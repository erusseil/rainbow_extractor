import numpy as np
import random
import pandas as pd
from collections import Counter

def balance_objects(features):
    ids = []
    
    for classe in np.unique(features['type']):
        sub = features[features['type'] == classe]
        ids.append(np.unique([lc.objectid for lc in sub['lc']]))
    
    smallest = min([len(a) for a in ids])
    keep_ids = [random.sample(list(a), smallest) for a in ids]
    keep_ids = [item for row in keep_ids for item in row]
    
    mask = features['lc'].apply(lambda x: x.objectid).isin(keep_ids)
    
    final = features[mask]
    alerts_per_obj = dict(Counter([lc.objectid for lc in final['lc']]))
    weights = [1/alerts_per_obj.get(lc.objectid) for lc in final['lc']]
    
    return final, weights

def balance_alerts(features):
    
    alerts_per_class = list(Counter(features['type']).values())
    smallest = min(alerts_per_class)
    
    a = [features[features['type']==classe].sample(n=smallest) for classe in np.unique(features['type'])]

    return pd.concat(a)

def X_y(data, lc_only=False):

    y = data['type']
    
    if lc_only:
        X = data.iloc[:, 4:27]
    else:
        X = data.iloc[:, 4:] 
    return X, y