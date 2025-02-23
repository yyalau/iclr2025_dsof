from src.trainers.trainerBase import TrainerBase
import warnings

# from collections import deque
import torch
from torch import nn

# import random
import numpy as np

warnings.filterwarnings("ignore")


class TrainerBaseERTD(TrainerBase):
    def __init__(self, args, main_model, student_model):

        super().__init__(args, main_model, student_model)

        self.laststepA = [None, None, None, None]
        self.laststepB = [None, None, None, None]

        self.replayBufferSize = args.replayBufferSize if not args.test_run else 20
        self.batchReplayBufferSize = (
            args.batchReplaySize if not args.test_run else min(10, args.batchReplaySize)
        )
        self.num_ERepochs = args.num_ERepochs
        self.freq_ERupdate = args.freq_ERupdate
        self.count = 0
        self.replayBuffer = [None, None, None, None]

        
        self.MSEReductNone = nn.MSELoss(reduction="none")
        
        self.indices = [ k for k in self.args.td_k if k <= self.args.pred_len]
        self.discountedW = []
        
        for idx in self.indices:
            self.discountedW.append(
                torch.cat(
                    [
                        torch.ones(idx-1),
                        torch.Tensor([self.args.discounted**i for i in range(self.args.pred_len - idx+1)]),
                    ]
                )[None, :, None].to(self.device)
            )
            
        self.discountedW = torch.cat(self.discountedW, dim=0)


    def backward_lossT_train(self, prev):
        raise NotImplementedError

    def backward_lossT_test(self, prev):
        raise NotImplementedError

    def backward_lossFT(self, prev, curr):
        raise NotImplementedError

    def backward_tdlossFT(self, prev, curr):
        raise NotImplementedError

    def er(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):

        g = lambda x, index, i: (
            x[:, index : index + self.args.pred_len]
            if i % 2
            else x[:, index : index + self.args.seq_len]
        )

        if (
            self.laststepA[0] is not None
            and (N_A := self.laststepA[0].shape[1] - self.args.seq_len + 1)
            >= self.args.pred_len
        ):

            # save the first element of the laststepA
            itemA = [g(x, 0, i) for i, x in enumerate(self.laststepA)]

            # remove the first element from the laststepA
            self.laststepA = [x[:, 1:] for x in self.laststepA]

            # enqueue the first element of the laststepA
            self.replayBuffer = [
                (
                    torch.cat([bufferT, newT[:, -1:]], dim=1)
                    if bufferT is not None
                    else newT
                )
                for bufferT, newT in zip(self.replayBuffer, itemA)
            ]

            # dequeue the first element of the replay buffer
            if (
                self.replayBuffer[0].shape[1] - self.args.seq_len + 1
                > self.replayBufferSize
            ):
                self.replayBuffer = [bufferT[:, 1:] for bufferT in self.replayBuffer]

            # number of sequences in the replay buffer
            if (
                N_buffer := self.replayBuffer[0].shape[1] - self.args.seq_len + 1
            ) >= self.batchReplayBufferSize:

                if self.count % self.freq_ERupdate == 0:

                    for _ in range(self.num_ERepochs):

                        randomSampleIndex = np.random.choice(
                            N_buffer, self.batchReplayBufferSize, replace=False
                        )

                        pA_x, pA_y, pA_x_mark, pA_y_mark = [
                            torch.cat(
                                [g(x, index, j) for index in randomSampleIndex], dim=0
                            )
                            for j, x in enumerate(self.replayBuffer)
                        ]

                        prevA = self._process_one_batch(
                            dataset_object,
                            pA_x,
                            pA_y,
                            pA_x_mark,
                            pA_y_mark,
                            mode="test",
                        )

                        self.backward_lossT_test(prevA)
                        self.backward_lossFT(prevA, prevA)

                        for opt in self.opt["test"]["batch"]:
                            opt.step()
                        self.store_grad()
                        self.update_ocp(prevA, prevA)
                        for opt in self.opt["test"]["batch"]:
                            opt.zero_grad()
                        self.clean(prevA)

                self.count = (self.count + 1) % self.freq_ERupdate

        curr_data = [
            batch_x,
            batch_y,
            batch_x_mark,
            batch_y_mark,
        ]
        # add the latest data to the laststepA
        self.laststepA = [
            torch.cat([lastT, currT[:, -1:]], dim=1) if lastT is not None else currT
            for lastT, currT in zip(self.laststepA, curr_data)
        ]

    def td(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):

        b, t, d = batch_y.shape
        g = lambda x, index, i: (
            torch.cat(
                (x[:, -index:], torch.zeros((b, t - index, x.shape[2]), device=x.device)), 
                dim=1
            )
            if (i % 2 == 1)
            else x[:,-(index + self.args.seq_len - 1) : -(index - 1) if index != 1 else None,:]
        )

        if self.laststepB[0] is not None and (
            self.laststepB[0].shape[1] - self.args.seq_len + 1 >= max(self.indices)
        ):

            pB_x, pB_y, pB_x_mark, pB_y_mark = [
                torch.cat([g(x, index, j) for index in self.indices], dim=0)
                for j, x in enumerate(self.laststepB)
            ]

            # previous prediction
            prevB = self._process_one_batch(
                dataset_object, pB_x, pB_y, pB_x_mark, pB_y_mark, mode="test"
            )

            # current time prediction
            with torch.no_grad():
                curr = self._process_one_batch(
                    dataset_object,
                    batch_x,
                    batch_y,
                    batch_x_mark,
                    batch_y_mark,
                    mode="test",
                )
            self.backward_tdlossFT(prevB, curr)

            for opt in self.opt["test"]["online"]:
                opt.step()
            self.store_grad()
            self.update_ocp(prevB, curr)
            for opt in self.opt["test"]["online"]:
                opt.zero_grad()

            self.clean(curr)
            self.clean(prevB)

            # dequeue earliest element
            self.laststepB = [x[:, 1:] for x in self.laststepB]

        # append latest element
        if self.laststepB[0] is None:
            self.laststepB = [
                batch_x,
                batch_y[:,0:1],
                batch_x_mark,
                batch_y_mark[:,0:1]
            ]
            return

        self.laststepB = [
            torch.cat([lastT, currT[:, 0:1]], dim=1) 
            for lastT, currT in zip(
                self.laststepB,
                [
                    batch_x[:, -1:,],
                    batch_y[:, 0:1],
                    batch_x_mark[:, -1:,],
                    batch_y_mark[:, 0:1],
                ],
            )
        ]