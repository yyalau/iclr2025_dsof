import os 
import sys
import shutil
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# from misc.extract_base import find_min_mse, is_min_mse

def base(*path ):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *path)

stats = {
}


for model in os.listdir(base()):
    
    if 'TimesNet' in model: continue
    if os.path.isfile(base(model)):
        continue
    
    
    if 'MLP' not in model: continue
    # print(model)
    
    
    
    for dataset in os.listdir(base(model)):
        
        
        with open(base(model, dataset)) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # print(config['opt_main']['train'].keys())
        
        for opt_model in config.keys():
            
            if not isinstance(config[opt_model], dict): continue     
                   
            if opt_model not in stats.keys():
                stats[opt_model] = {}
                
            for mode in config[opt_model].keys():
                if mode not in stats[opt_model].keys():
                    stats[opt_model][mode] = {}
                
                if mode == 'train': 
                    for phase, v in config[opt_model][mode].items():
                        # print(phase)
                        if phase=='lr': continue
                        if phase not in stats[opt_model][mode].keys():
                            stats[opt_model][mode][phase] = {}
                        
                        if v not in stats[opt_model][mode][phase].keys():
                            stats[opt_model][mode][phase ][v] = 1
                        else:
                            stats[opt_model][mode][phase ][v] += 1
                    continue
                for phase in config[opt_model][mode].keys():
                    
                    # print(config[opt_model][mode][phase], opt_model, mode, phase)
                    for param, v in config[opt_model][mode][phase].items():
                        if param=='lr': continue
                        if f"{phase}.{param}" not in stats[opt_model][mode].keys():
                            stats[opt_model][mode][f"{phase}.{param}"] = {}
                        
                        if v not in stats[opt_model][mode][f"{phase}.{param}"].keys():
                            stats[opt_model][mode][f"{phase}.{param}" ][v] = 1
                        else:
                            stats[opt_model][mode][f"{phase}.{param}" ][v] += 1
                    
            
        # for param, v in stats.items():
            
        #     tally = config[param]
            
        #     if tally not in v.keys():
        #         stats[param][tally] = 1
        #     else:
        #         stats[param][tally] += 1
import json
print(json.dumps(stats, indent = 4))

with open(path :=base('../../..', 'latex', 'lr_stats.json'), 'w+') as f:
    json.dump(stats, f, indent=4)
    
print(f"Saved to {path}")
