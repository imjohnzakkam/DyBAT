import pandas as pd 
import wandb
from tqdm import tqdm

api = wandb.Api(timeout=19)
entity, project = "turbo_x", "runs_crop_size_32"  
runs = api.runs(entity + "/" + project) 

summary_list, config_list, name_list = [], [], []
keys = ['Train Loss', 'Train Top-1 Accuracy', 'Train Top-5 Accuracy', 'Val Loss', 'Val Top-1 Accuracy', 'Val Top-5 Accuracy']
rows = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
metric_dict = {}

for run in tqdm(runs): 
    if run.state == 'finished':
        hist = run.history()
        for key in keys:
            metric_dict[f'{run.name}_{"".join(key.split())}'] = hist[key][rows]

runs_df = pd.DataFrame(metric_dict)
runs_df.to_csv("metrics.csv")