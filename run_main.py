import subprocess
import os
import time

from misc.runnerbase import Utility, BaseCLI


def base(path: str = "") -> str:
    """
    Returns the base directory path joined with the provided path.

    Args:
        path (str): The path to join with the base directory.

    Returns:
        str: The joined path.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


class MainCLI(BaseCLI):
    def __init__(
        self,
        main_models: str,
        student_models: str,
        opts: str,
        trainers: str,
        use_nohup: bool = False,
        mode: str = "zip",
    ):
        """
        Initializes the MainCLI class with the specified parameters.

        Args:
            main_models (str): The main model name.
            student_models (str): The student model name.
            opts (str): The options for the model.
            trainers (str): The trainer name.
            use_nohup (bool): Whether to use nohup for running commands.
            mode (str): The mode for running commands.
        """
        super().__init__(main_models, student_models, opts, trainers, use_nohup, mode)

    def create_cli(self, sett: dict, dataset: str, pred_len: int, gpu_to_use: int, itr: int = 1) -> str:
        """
        Creates the command line interface (CLI) command to run the task.

        Args:
            sett (dict): The settings for the task.
            dataset (str): The dataset name.
            pred_len (int): The prediction length.
            gpu_to_use (int): The GPU to use for the task.
            itr (int): The iteration number.

        Returns:
            str: The CLI command to run the task.
        """
        command = f"CUDA_VISIBLE_DEVICES={gpu_to_use} "

        if self.use_nohup:
            command += "nohup "
        command += f"python -u src/main.py --itr {itr} --pred_len {pred_len} --data {dataset} --num_workers {num_workers} "
        command += f"--y_model_main {sett['main_model']} "
        command += f"--y_model_student {sett['student_model']} "
        command += f"--y_opt {self.get_opt_sett(sett['main_model'], sett['student_model'], sett['opt'], dataset)} "
        command += f"--y_trainer {sett['trainer']} --comments {sett['comment']} "

        if self.use_nohup:
            logger_file = self.get_nohup_filename(sett, dataset, pred_len)
            command += f"> {logger_file} 2>&1 &"
            os.makedirs(os.path.dirname(logger_file), exist_ok=True)

        return command

    def get_nohup_filename(self, sett: dict, dataset: str, pred_len: int) -> str:
        """
        Generates the filename for the nohup log file.

        Args:
            sett (dict): The settings for the task.
            dataset (str): The dataset name.
            pred_len (int): The prediction length.

        Returns:
            str: The filename for the nohup log file.
        """
        folder = f"./logs/nohup/exp/{Utility.get_header(sett['main_model'])}"
        folder += f"-{Utility.get_header(sett['student_model'])}"
        folder += f"-{Utility.get_header(sett['trainer'])}"

        return f"{folder}/{dataset}-{pred_len}.out"

    def save_outputpath_file(self, exp_paths: list) -> None:
        """
        Saves the output paths to a file.

        Args:
            exp_paths (list): The list of experiment paths.
        """
        super().save_outputpath_file(exp_paths, "exps")

    def get_comment(self, main_model: str, student_model: str, trainer: str) -> str:
        """
        Generates a comment string based on the model and trainer names.

        Args:
            main_model (str): The main model name.
            student_model (str): The student model name.
            trainer (str): The trainer name.

        Returns:
            str: The generated comment string.
        """
        cm = f"exp--{Utility.replace_slash(main_model)}"
        cm += f"--{Utility.replace_slash(student_model)}--{Utility.replace_slash(trainer)}"

        return cm

    def get_exp_path(self, sett: dict) -> str:
        """
        Generates the experiment path based on the settings.

        Args:
            sett (dict): The settings for the task.

        Returns:
            str: The generated experiment path.
        """
        path = self.get_output_path(sett)
        return "exps/" + path


gpus = {
    "Electricity": 3,
    "ETTh2": 3,
    "ETTm1": 4,
    "Traffic": 4,
    "Exchange": 4,
    "Weather": 4,
    "others": 4,
}


pred_lens = [48]
#### example pred_lens options
# pred_lens = [6, 12, 24, 48, 96]

# mode = "product"

itr = 1
use_nohup = False

datasets = [
    "ETTh2",
]
#### other dataset options
# datasets =  ["Electricity", "ETTh1", "ETTh2", "Traffic", "Exchange", "Weather",]

main_model = "DLinear"
student_model = "MLP"
opt = "w_student"
trainer = "w_student/residual/dsof"


### other main_model options
# main_models = ['DLinear', 'FITS', 'FSNet', 'iTransformer', 'NSTransformer', 'OneNet', 'PatchTST',  ]


#### option combinations (group 1)
# student_models = ["MLP"]
# opts = [
#     "w_student",
# ]
# trainers = [
#     "w_student/residual/dsof",
#     "w_student/residual/td",
#     "w_student/residual/er",
#     "w_student/residual/baseline",
#     "w_student/non_residual/ertd",
# ]


#### option combinations (group 2)
# student_models = [None]
# opts = [
#     "wo_student",
# ]

# trainers = [
#     "wo_student/batch_learning",
#     "wo_student/td",
#     "wo_student/er",
#     "wo_student/derpp",
#     "wo_student/ertd",
#     "wo_student/plain",
#     "wo_student/tfcl",
# ]


# Initialize the Utility class with a process threshold
utility = Utility(process_threshold=12)

# Initialize the MainCLI class with the specified parameters
mainCLI = MainCLI(
    main_model,
    student_model,
    opt,
    trainer,
    use_nohup,
)

# Set the number of workers to 0
num_workers = 0

# List to store process IDs
pids = []

# Change the current working directory to the base directory
os.chdir(base())

# Create the logs/nohup directory if it doesn't exist
os.makedirs("./logs/nohup", exist_ok=True)

# Print the current working directory
subprocess.run(["pwd"], shell=True)

# Map GPU commands to process IDs
utility.pid_gpu_cmd_mapping()

# Iterate over the settings, datasets, and prediction lengths
for sett in mainCLI.get_settings():
    for dataset in datasets:
        for pred_len in pred_lens:

            # Get the GPU to use for the current dataset
            gpu_to_use = gpus[dataset]

            # Get the main and student model names
            main_model = mainCLI.get_model_name(sett["main_model"], None)
            student_model = mainCLI.get_model_name(sett["student_model"], dataset)

            # Wait until a CPU is available
            while not utility.cpu_available():
                time.sleep(60)

            # Allocate a GPU for the current task
            while True:
                gpu_to_use = utility.allocate_gpu(
                    gpu_to_use, dataset, main_model, student_model, gpus
                )
                if gpu_to_use is not None:
                    break

                time.sleep(60)
                utility.update_estimated_free_gpu()

            # Print the current task details
            print(
                f'Running {dataset} with {sett["comment"]} for pred_len {pred_len} in gpu {gpu_to_use}'
            )

            # Create the command to run the task
            command = mainCLI.create_cli(sett, dataset, pred_len, gpu_to_use, itr)

            # Run the command
            subprocess.run(command, shell=True)

            # Wait for 1 second before starting the next task
            time.sleep(1)

        # Wait for 1 second before starting the next dataset
        time.sleep(1)

    # Wait for 2 seconds before starting the next setting
    time.sleep(2)
