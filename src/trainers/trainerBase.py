from tqdm import tqdm

from src.utils.tools import EarlyStopping, AverageMeter, DataLogger
from src.utils.metrics import metric  # , cumavg

# from src.utils.optim import Adbfgs, AdamSVD #, SAGM
# from utils.optim.scheduler import LinearScheduler
# from src.data.data_loader import Dataset_Custom as Data
from src.utils.tools import Struct, save_model
from misc.runnerbase import Utility
import optuna


# import numpy as np
import time
import torch
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter

# import yaml

import os
import time
import warnings

warnings.filterwarnings("ignore")
import importlib


class TrainerBase:
    # TODO: remove the student_model argument from the constructor
    def __init__(self, args, main_model, student_model):

        # super().__init__(args)
        self.args = args
        self.online = args.online_learning
        self.device = args.device

        self.main_model = main_model(
            Struct(args).dict2attr("main_model"), args.seq_len
        ).to(self.device)
        self.mainFFN = getattr(
            importlib.import_module(
                f'src.trainers.forward.trainerForward_{args.main_model["model"]}'
            ),
            "TrainerForward",
        )(args, self.main_model, self.device)

        self.student_model = None
        self.studentFFN = None

        self.scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
        self.f_dim = -1 if self.args.features == "MS" else 0

        if not self.args.timing:
            self.writer = SummaryWriter(log_dir=args.setting)
            self.writer.add_text("comments", args.comments)

        self.criterion = self._select_criterion()

        self.meters = {
            "train": {
                "loss": AverageMeter(),
            },
            "valid": {
                "loss": AverageMeter(),
            },
            "test": {
                "mae": AverageMeter(),
                "mse": AverageMeter(),
                "rmse": AverageMeter(),
                "mape": AverageMeter(),
                "mspe": AverageMeter(),
            },
        }

        self.datalog = DataLogger(["pred", "true", "mse", "mae", "embeddings"])

    def _get_optim(self, arguments, params):
        opt_str = arguments["opt"].lower()
        Optimizer = {
            "adam": optim.Adam,
            "adamw": optim.AdamW,
            # 'adamsvd': AdamSVD,
            # 'adbfgs': Adbfgs,
            "sgd": optim.SGD,
        }[opt_str]

        d = arguments.copy()
        d.pop("opt")
        return Optimizer(params, **d)

    def _select_optimizer(self):

        opts = {
            "train": [
                self._get_optim(
                    self.args.opt_main["train"], self.main_model.parameters()
                )
            ],
            "test": {
                "batch": [
                    self._get_optim(
                        self.args.opt_main["test"]["batch"],
                        self.main_model.parameters(),
                    )
                ],
                "online": [
                    self._get_optim(
                        self.args.opt_main["test"]["online"],
                        self.main_model.parameters(),
                    )
                ],
            },
        }
        
        # TODO: refractor 
        extended_opts = self.mainFFN.extended_optimizer()
        
        opts['train'] += extended_opts['train']
        opts['test']['batch'] += extended_opts['test']['batch']
        opts['test']['online'] += extended_opts['test']['online']
        
        return opts

    def _select_criterion(self):
        return nn.MSELoss()
    
    def update_ocp(self, prev, curr):
        self.mainFFN.update_ocp(prev, curr)
        if self.studentFFN is not None:
            self.studentFFN.update_ocp(prev, curr)

    def store_grad(self):

        self.mainFFN.store_grad()
        if self.studentFFN is not None:
            self.studentFFN.store_grad()

    def display_msg(self, epoch, train_steps, modes):
        msg = f"Epoch: {epoch + 1}, Steps: {train_steps + 1} "

        for mode in modes:
            for k, v in self.meters[mode].items():
                msg += f"| {mode} {k}: {v.avg:.4f} "
        return msg

    def _process_one_batch(
        self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="train"
    ):
        raise NotImplementedError

    def clean(self, outputs):
        for key, value in outputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            outputs[key] = outputs[key].detach().cpu().numpy()
        return outputs

    def train_step(self, train_data, batch_x, batch_y, batch_x_mark, batch_y_mark):
        outputs = self._process_one_batch(
            train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
        )

        loss = self.criterion(outputs["pred"], outputs["true"])
        if self.args.use_amp:
            self.scaler.scale(loss).backward()
            for opt in self.opt["train"]:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            for opt in self.opt["train"]:
                opt.step()

        self.store_grad()
        self.update_ocp(outputs, outputs)
        

        return {"loss": loss.item()}

    def valid_step(self, vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark):
        outputs = self._process_one_batch(
            vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="vali"
        )
        loss = self.criterion(
            self.mainFFN.get_future(outputs["pred"]).detach().cpu(),
            self.mainFFN.get_future(outputs["true"]).detach().cpu(),
        )
        return {"loss": loss.item()}

    def test_step(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        return self._process_one_batch(
            dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode="test"
        )

    def online_freezing(self):
        if self.online == "regressor":
            for p in self.main_model.encoder.parameters():
                p.requires_grad = False
        elif self.online == "none":
            for p in self.main_model.parameters():
                p.requires_grad = False

    def train(self, train_data, train_loader, vali_data, vali_loader):
        encountered_nan = False
        # train_data, train_loader = self._get_data(flag='train') #2773
        # vali_data, vali_loader = self._get_data(flag='val') # 673
        self.opt = self._select_optimizer()

        path = os.path.join(self.args.setting, "checkpoints")
        if not self.args.timing:
            os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            iter_count = 0

            self.main_model.train()
            if self.student_model is not None:
                self.student_model.train()

            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                if hasattr(self, "augmenter"):
                    if self.augmenter == None:
                        self.get_augmenter(batch_x)

                for opt in self.opt["train"]:
                    opt.zero_grad()

                losses = self.train_step(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                if losses["loss"] != losses["loss"]:
                    encountered_nan = True

                for k, v in losses.items():
                    self.meters["train"][k].update(v)

                if (i + 1) % 100 == 0:
                    print(self.display_msg(epoch, i, ["train"]))

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if encountered_nan:
                    break
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            #### validation
            self.main_model.eval()
            if self.student_model is not None:
                self.student_model.eval()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                losses = self.valid_step(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark
                )
                for k, v in losses.items():
                    self.meters["valid"][k].update(v)
            self.main_model.train()
            if self.student_model is not None:
                self.student_model.train()

            ckpt_content = {
                "main_model": self.main_model.state_dict(),
                "student_model": (
                    self.student_model.state_dict()
                    if self.student_model is not None
                    else None
                ),
            }
            
            if not self.args.timing:
                early_stopping(
                    self.meters["valid"][k].avg,
                    ckpt_content,
                    path,
                )
            #    name=f"e{epoch}_valid_{self.meters['valid'][k].avg:.4f}.pth")

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                #### logging
                print(self.display_msg(epoch, train_steps, ["train", "valid"]))
                for mode in self.meters.keys():
                    for k, v in self.meters[mode].items():
                        self.writer.add_scalar(f"{mode}/{k}", v.avg, epoch)

                for i, opt in enumerate(self.opt["train"]):
                    self.writer.add_scalar(
                        f"learning rate {i}", opt.param_groups[0]["lr"], epoch
                    )

            if self.args.test_run or self.args.timing or encountered_nan:
                break

        if not self.args.timing:
            self.load(early_stopping.best_path)

        return self.main_model

    def test(self, test_data, test_loader, trial=None):
        encountered_nan = False

        if self.online == "none":
            self.main_model.eval()

        self.online_freezing()
        start = time.time()
        pbar = tqdm(test_loader)
        start_cpu = 0

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pbar):
            if self.args.timing and i == 100:
                start = time.time()
                start_cpu = time.process_time()
            
            outputs = self.test_step(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark
            )
            if outputs is None:
                continue
            
            metrics = metric(
                self.mainFFN.get_future(outputs["pred"]),
                self.mainFFN.get_future(outputs["true"]),
            )
            if metrics["mse"] != metrics["mse"]:
                encountered_nan = True

            if not self.args.timing:
                for k, v in metrics.items():
                    self.meters["test"][k].update(v)
                    self.writer.add_scalar(f"test/{k}", v, i)
                    self.writer.add_scalar(f"test/c{k}", self.meters["test"][k].avg, i)

            self.datalog.update({k: metrics[k] for k in metrics.keys()})

            if self.args.data == "Traffic":
                outputs["pred"] = outputs["pred"][:, :, :60]
                outputs["true"] = outputs["true"][:, :, :60]
            self.datalog.update({k: outputs[k] for k in outputs.keys()})

            pbar.set_postfix(
                {"point": metrics["mse"], "cumavg": self.meters["test"]["mse"].avg}
            )
            if self.args.test_run  and i > (self.args.pred_len + 50):
                break
            
            if (self.args.timing and i == 150):
                break

            if trial is not None:
                trial.report(self.meters["test"]["mse"].avg, i)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if encountered_nan:
                break

        print("test shape:", self.datalog["pred"].shape, self.datalog["true"].shape)
        # end = 
        exp_time = time.time() - start
        #### logging and saving
        if self.args.timing:
            cpu_time = time.process_time() - start_cpu

            # TODO: make it customizable
            gpu0 = Utility.get_gpu_usage()[0]
            # with open('latex/timing.csv', 'a+') as f:
            #     ll = f"{self.args.main_model['model']},{self.args.data},{self.args.y_trainer},"
            #     ll += f"{exp_time},{self.args.pred_len },{(self.args.pred_len)/exp_time}\n"
            #     f.write(ll)
            with open('latex/timing_cpu.csv', 'a+') as f:
                ll = f"{self.args.main_model['model']},{self.args.data},{self.args.y_trainer},"
                ll += f"{cpu_time},{50 },{50/cpu_time}\n"
                f.write(ll)
                
            # with open('latex/memory.csv', 'a+') as f:
            #     ll = f"{self.args.main_model['model']},{self.args.data},{self.args.y_trainer},"
            #     ll += f"{gpu0}\n"
            #     f.write(ll)
        
        print(
            f"mse:{self.meters['test']['mse'].avg}, mae:{self.meters['test']['mae'].avg}, time:{exp_time}"
        )
        
        if not self.args.timing:
            self.save(name=f"s{i}_test_{self.meters['test']['mse'].avg:.4f}.pth")

        if not self.args.timing: self.writer.close()

        return (
            [
                self.meters["test"]["mae"].avg,
                self.meters["test"]["mse"].avg,
                self.meters["test"]["rmse"].avg,
                self.meters["test"]["mape"].avg,
                self.meters["test"]["mspe"].avg,
                exp_time,
            ],
            self.datalog["mae"],
            self.datalog["mse"],
            self.datalog["pred"],
            self.datalog["true"],
        )

    def save(self, name="checkpoint.pth"):
        path = os.path.join(self.args.setting, "checkpoints", name)
        ckpt_content = {
            "main_model": self.main_model.state_dict(),
            "student_model": (
                self.student_model.state_dict()
                if self.student_model is not None
                else None
            ),
        }
        save_model(ckpt_content, path)

    def load(self, path):

        ckpt_content = torch.load(path, map_location=self.device)
        self.main_model.load_state_dict(ckpt_content["main_model"])
        if self.student_model is not None:
            self.student_model.load_state_dict(ckpt_content["student_model"])
