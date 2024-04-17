# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
import torch
import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, \
    ParamUpdateHook, EvaluationHook, EMAHook, WANDBHook, AimHook
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, \
    Bn_Controller
from semilearn.core.criterions import CELoss, ConsistencyLoss
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
import torch
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist
import torch.nn as nn
from semilearn.algorithms import get_algorithm, name2alg
from semilearn.core.utils import get_net_builder, get_logger, get_port, send_model_cuda, count_parameters, \
    over_write_args_from_file, TBLog
import os
import heapq
import math
import faiss
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
import torch.nn.functional as F
import time
from sklearn.manifold import TSNE


@ALGORITHMS.register('etfssl')
class FullySupervised(AlgorithmBase):
    """
        Train a etfssl model using labeled data only. This serves as a baseline for initialization.

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
        """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        self.classpolars = self.generate_random_orthogonal_matrix(768, self.num_classes).cuda(0)

    def train_step(self, x_lb, y_lb, x_ulb_s, idx_ulb):
        num_lb = y_lb.shape[0]
        if self.mode == 'pretrain':
            with self.amp_cm():
                feats_x_lb = self.model(x_lb)['feat']
                logits_x_lb = (torch.matmul(feats_x_lb, self.classpolars)/0.1)
                sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
            out_dict = self.process_out_dict(loss=sup_loss)
            log_dict = self.process_log_dict(sup_loss=sup_loss.item())

            return out_dict, log_dict, self.classpolars
        else:
            if self.it % 1024 == 0:
                self.model.eval()
                self.mem_label, self.id = self.obtain_label(self.loader_dict['train_ulb'], self.loader_dict['lb_queue'])
                self.mem_label = torch.from_numpy(self.mem_label).cuda()
                self.model.train()
            with self.amp_cm():
                if self.use_cat:
                    inputs = torch.cat((x_lb, x_ulb_s))
                    outputs = self.model(inputs)
                    feats_x_lb = outputs['feat'][:num_lb]
                    feats_x_ulb_s = outputs['feat'][num_lb:]
                else:
                    feats_x_lb = self.model(x_lb)['feat']
                    feats_x_ulb_s = self.model(x_ulb_s)['feat']
                logits_x_lb = (torch.matmul(feats_x_lb, self.classpolars) / 0.1)
                idx = torch.tensor([self.id.index(element) for element in idx_ulb.cpu().numpy().tolist()])

                sup_loss_l = self.ce_loss(logits_x_lb, y_lb, reduction='mean')
                feat_dict = {'x_ulb_s': logits_x_lb}

                proxy_logits_x_ulb_s = (torch.matmul(feats_x_ulb_s, self.classpolars) / 0.1)
                pred = self.mem_label[idx]

                ulb_loss = self.ce_loss(proxy_logits_x_ulb_s, pred, reduction='mean')

                softmax_out = nn.Softmax(dim=1)(proxy_logits_x_ulb_s)
                entropy_loss = torch.mean(self.Entropy(softmax_out))  # ent

                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))
                entropy_loss -= gentropy_loss

                total_loss = self.lambda_uce * ulb_loss + sup_loss_l + self.lambda_uim * entropy_loss
            out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
            log_dict = self.process_log_dict(sup_loss=ulb_loss.item(),
                                             unsup_loss=sup_loss_l.item(),
                                             total_loss=total_loss.item(), )

            return out_dict, log_dict, self.classpolars

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        P, _ = np.linalg.qr(a)
        P = torch.tensor(P).float()
        assert torch.allclose(torch.matmul(P.T, P), torch.eye(num_classes), atol=1e-07), torch.max(
            torch.abs(torch.matmul(P.T, P) - torch.eye(num_classes)))
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = np.sqrt(num_classes / (num_classes - 1)) * torch.matmul(P, I - ((1 / num_classes) * one))
        return M

    def obtain_label(self, loader, loader_lb):
        start_test = True

        with torch.no_grad():
            iter_test = iter(loader)
            iter_lb = iter(loader_lb)

            for i in range(len(loader)):
                data = iter_test.__next__()
                x_ulb_w = data['x_ulb_w']
                idx = data['idx_ulb']
                if isinstance(x_ulb_w, dict):
                    x_ulb_w = {k: v.cuda() for k, v in x_ulb_w.items()}
                else:
                    x_ulb_w = x_ulb_w.cuda()

                outs_x_ulb_w = self.model(x_ulb_w)
                feats_x_ulb_w = outs_x_ulb_w['feat']
                proxy_logits_x_ulb_w = (torch.matmul(feats_x_ulb_w, self.classpolars) / 0.1)

                if start_test:
                    id = idx
                    all_fea = feats_x_ulb_w.float()
                    all_output = proxy_logits_x_ulb_w.float()
                    start_test = False
                else:
                    id = torch.cat((id, idx), 0)
                    all_fea = torch.cat((all_fea, feats_x_ulb_w.float()), 0)
                    all_output = torch.cat((all_output, proxy_logits_x_ulb_w.float()), 0)
            start_test = True
            for _ in range(len(loader_lb)):
                data = iter_lb.__next__()
                x_lb = data['x_lb']
                y_lb = data['y_lb']
                idx = data['idx_lb']
                id = torch.cat((id, idx), 0)
                if isinstance(x_lb, dict):
                    x_lb = {k: v.cuda() for k, v in x_lb.items()}
                else:
                    x_lb = x_lb.cuda()

                outs_x_lb = self.model(x_lb)
                feats_x_lb = outs_x_lb['feat']
                proxy_logits_x_lb = (torch.matmul(feats_x_lb, self.classpolars) / 0.1)
                if start_test:
                    fea_lb = feats_x_lb.float()
                    y = y_lb.float()
                    all_fea = torch.cat((all_fea, feats_x_lb.float()), 0)
                    all_output = torch.cat((all_output, proxy_logits_x_lb.float()), 0)
                    start_test = False
                else:
                    fea_lb = torch.cat((fea_lb, feats_x_lb.float()), 0)
                    y = torch.cat((y, y_lb.float()), 0)
                    all_output = torch.cat((all_output, proxy_logits_x_lb.float()), 0)
                    all_fea = torch.cat((all_fea, feats_x_lb.float()), 0)
        _, pred = torch.max(all_output, 1)

        result = id.tolist()
        first_occurrences = [i for i, x in enumerate(result) if result.index(x) == i]
        all_fea = all_fea.float().cpu().numpy()[first_occurrences]
        all_output = all_output[first_occurrences].cpu()
        all_output = nn.Softmax(dim=1)(all_output)
        K = len(first_occurrences)
        aff = all_output.float().cpu().numpy()
        pred = pred.cpu().numpy()[first_occurrences]
        id = id[first_occurrences]

        for i in range(2):
            initc = aff.transpose().dot(all_fea)
            initc = initc / np.linalg.norm(initc)
            cls_count = np.eye(K)[pred].sum(axis=0)
            labelset = np.where(cls_count > 0)
            labelset = labelset[0]
            dd = cdist(all_fea, initc[labelset], metric='cosine')
            pred_label = dd.argmin(axis=1)
            pred = labelset[pred_label]
            aff = np.eye(K)[pred]

        pred = np.array(pred)
        id = np.array(id)
        return pred.astype('int'), id.astype('int').tolist()

    def Entropy(self, input_):
        bs = input_.size(0)
        entropy = -input_ * torch.log(input_ + 1e-5)
        entropy = torch.sum(entropy, dim=1)
        return entropy


