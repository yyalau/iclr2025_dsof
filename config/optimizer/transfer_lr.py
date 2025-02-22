import yaml, os
import shutil

def base(path= ''):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
datasets = ['Electricity', 'ETTh2', 'ETTm1', 'Traffic', 'Exchange',  'Weather',]




############ for non_residual_ERTDRegS category ################
# from_category = "residual_ERTDRegS"

# to_category = "residual_ERTDRegS_003"
# lr = 0.003

# settings = [
#     {
#         'main_model': 'FSNet',
#         'student_model': 'MLP',
#     },
#     {
#         'main_model': 'PatchTST',
#         'student_model': 'MLP',
#     },
#     {
#         'main_model': 'DLinear',
#         'student_model': 'MLP',
#     },
#     {
#         'main_model': 'iTransformer',
#         'student_model': 'MLP',
#     },
#     # {
#     #     'main_model': 'FSNet',
#     #     'student_model': 'FSNet',  
#     # },
#     # {
#     #     'main_model': 'PatchTST',
#     #     'student_model': 'FSNet',  
#     # },
#     # {
#     #     'main_model': 'FSNet',
#     #     'student_model': 'PatchTST',   
#     # },
#     # {
#     #     'main_model': 'PatchTST',
#     #     'student_model': 'PatchTST',
#     # }
# ]

# for sett in settings:
#     main_model = sett['main_model']
#     student_model = sett['student_model']
#     for dataset in datasets:
                
#         new_path = base(f"{to_category}/{main_model}_{student_model}")
#         os.makedirs(new_path, exist_ok=True)
        
#         if student_model == 'MLP':
#             # shutil.copy(base(f"{from_category}/{main_model}_MLP/{dataset}.yaml"), base(f"{to_category}/{main_model}_MLP/{dataset}.yaml"))
#             # print(f"Created {new_path}/{dataset}.yaml")

#             with open(base(f"{from_category}/{main_model}_MLP/{dataset}.yaml")) as f:
#                 new_params = yaml.load(f, Loader=yaml.FullLoader)
#             # import ipdb; ipdb.set_trace()
#             # new_params['opt_main']['test']['online']['lr'] = lr
#             # new_params['opt_student']['test']['online']['lr'] = lr
#             with open(base(f"{to_category}/{main_model}_MLP/{dataset}.yaml"), 'w') as f:
#                 yaml.dump(new_params, f)
                
#             continue
        
#         with open(base(f"{from_category}/{main_model}_MLP/{dataset}.yaml")) as f:
#             main_params = yaml.load(f, Loader=yaml.FullLoader)
            
#         with open(base(f"{from_category}/{student_model}_MLP/{dataset}.yaml")) as f:
#             student_params = yaml.load(f, Loader=yaml.FullLoader)
        
#         new_params = {'opt_main': main_params['opt_main'], 'opt_student': student_params['opt_main']}
        
#         with open(base(f"{new_path}/{dataset}.yaml"), 'w') as f:
#             yaml.dump(new_params, f)
        
#         print(f"Created {new_path}/{dataset}.yaml")
#         # import ipdb; ipdb.set_trace()

########### for without student ERTDReg category ################

from_category = "residual_ERTDRegS"
to_category = "wo_student"
# lr = 0.00003

settings = [
    { 
        'main_model': 'DLinear', 
    },
    {
        'main_model': 'FSNet',
    },
    {
        'main_model': 'PatchTST',
    },
    {
        'main_model': 'iTransformer',
    },
    {
        'main_model': 'NSTransformer',
    },
    {
        'main_model': 'OneNet',
    },
    {
        'main_model': 'FITS',
    }
]

for sett in settings:
    main_model = sett['main_model']
    # student_model = sett['student_model']
    for dataset in datasets:
                
        new_path = base(f"{to_category}/{main_model}")
        os.makedirs(new_path, exist_ok=True)
        
        # if student_model == 'MLP':
        #     shutil.copy(base(f"{from_category}/{main_model}_MLP/{dataset}.yaml"), base(f"{to_category}/{main_model}_MLP/{dataset}.yaml"))
        #     print(f"Created {new_path}/{dataset}.yaml")
        #     continue
        
        with open(base(f"{from_category}/{main_model}_MLP/{dataset}.yaml")) as f:
            main_params = yaml.load(f, Loader=yaml.FullLoader)
        
        # with open(base(f"{from_category}/{student_model}_MLP/{dataset}.yaml")) as f:
        #     student_params = yaml.load(f, Loader=yaml.FullLoader)
        
        new_params = {'opt_main': main_params['opt_main'],}
        # import ipdb; ipdb.set_trace()
        # new_params['opt_main']['test']['online']['lr'] = lr
        # import ipdb; ipdb.set_trace()        
        with open(base(f"{new_path}/{dataset}.yaml"), 'w') as f:
            yaml.dump(new_params, f)
        
        print(f"Created {new_path}/{dataset}.yaml")
