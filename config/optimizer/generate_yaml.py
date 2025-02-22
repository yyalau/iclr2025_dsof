import yaml
import os

datasets = [
    "Traffic",
    "Exchange",
    "Weather",
    "ETTm1",
    "Electricity",
    "ETTh2",
]
# models = [
#     "NSTransformer",
#     "TimeMixer",
#     "TimesNet",
#     "PatchTST",
#     "DLinear",
#     "iTransformer",
#     "FSNet",
#     "FITS",
#     # "MLP",
# ]
models = [
    "PatchTST_FSNet",
    "PatchTST_PatchTST",
    "FSNet_FSNet",
    "FSNet_PatchTST",
    # "MLP",
]

default_params = {
    "opt_main": {
        "train": {"opt": "adam", "amsgrad": True, "lr": 0.001},
        "test": {
            "batch": {"opt": "adamw", "amsgrad": True, "lr": 0.0001},
            "online": {"opt": "adam", "amsgrad": True, "lr": 0.0001},
        },
    },
    "opt_student": {
        "train": {"opt": "adamw", "amsgrad": True, "lr": 0.002},
        "test": {
            "batch": {"opt": "adamw", "amsgrad": True, "lr": 0.0015},
            "online": {"opt": "adamw", "amsgrad": True, "lr": 0.0019},
        },
    },
}


def base(path=""):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


for dataset in datasets:
    for model in models:
        path = base(f"residual_ERTDRegS/{model}/{dataset}.yaml")
        print(path)
        if os.path.exists(path):
            with open(path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            for opt_model in config.keys():
                for mode in config[opt_model].keys():
                    if mode == 'train':
                        for param, v in config[opt_model][mode].items():
                            if param == 'lr': continue
                            config[opt_model][mode][param] = default_params[opt_model][mode][param]

                    else:
                        for phase in config[opt_model][mode].keys():
                            for param, v in config[opt_model][mode][phase].items():
                                if param == 'lr': continue
                                config[opt_model][mode][phase][param] = default_params[opt_model][mode][phase][param]

            with open(path, "w+") as f:
                yaml.dump(config, f)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "w+") as f:
                yaml.dump(default_params, f)
