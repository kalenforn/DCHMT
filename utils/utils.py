import torch
import numpy as np
from typing import Union
import torch.nn as nn
from torch.nn import functional as F

from sklearn.metrics.pairwise import euclidean_distances


def compute_metrics(x):
    # 取复值的原因在于cosine的值越大说明越相似，但是需要取的是前N个值，所以取符号变为增函数s
    sx = np.sort(-x, axis=1)
    d = np.diag(-x)
    d = d[:, np.newaxis]
    ind = sx - d
    ind = np.where(ind == 0)
    ind = ind[1]
    metrics = {}
    metrics['R1'] = float(np.sum(ind == 0)) * 100 / len(ind)
    metrics['R5'] = float(np.sum(ind < 5)) * 100 / len(ind)
    metrics['R10'] = float(np.sum(ind < 10)) * 100 / len(ind)
    metrics['MR'] = np.median(ind) + 1
    metrics["MedianR"] = metrics['MR']
    metrics["MeanR"] = np.mean(ind) + 1
    metrics["cols"] = [int(i) for i in list(ind)]
    return metrics

def encode_hash(a: Union[torch.Tensor, np.ndarray]):
    if isinstance(a, torch.Tensor):
        hash_a = torch.sign(a)
        # where 是吧所有false值转为0
        # hash_a = torch.where(hash_a>0, hash_a, torch.tensor(0))
        return hash_a
    else:
        hash_a = np.sign(a)
        # hash_a = np.where(hash_a > 0, hash_a, 0)
        return hash_a

def calc_neighbor(a: torch.Tensor, b: torch.Tensor):
    # print(a.dtype, b.dtype)
    return (a.matmul(b.transpose(0, 1)) > 0).float()


def euclidean_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        similarity = torch.cdist(a, b, p=2.0)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        similarity = euclidean_distances(a, b)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))
    return similarity


def euclidean_dist_matrix(tensor1: torch.Tensor, tensor2: torch.Tensor):
    """
    calculate euclidean distance as inner product
    :param tensor1: a tensor with shape (a, c)
    :param tensor2: a tensor with shape (b, c)
    :return: the euclidean distance matrix which each point is the distance between a row in tensor1 and a row in tensor2.
    """
    dim1 = tensor1.shape[0]
    dim2 = tensor2.shape[0]
    multi = torch.matmul(tensor1, tensor2.t())
    a2 = torch.sum(torch.pow(tensor1, 2), dim=1, keepdim=True).expand(dim1, dim2)
    b2 = torch.sum(torch.pow(tensor2, 2), dim=1, keepdim=True).t().expand(dim1, dim2)
    dist = torch.sqrt(a2 + b2 - 2 * multi)
    return dist


def cosine_similarity(a: Union[torch.Tensor, np.ndarray], b: Union[torch.Tensor, np.ndarray]):

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        a = a / a.norm(dim=-1, keepdim=True) if len(torch.where(a != 0)[0]) > 0 else a
        b = b / b.norm(dim=-1, keepdim=True) if len(torch.where(b != 0)[0]) > 0 else b
        return torch.matmul(a, b.t())
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        a = a / np.linalg.norm(a, axis=-1, keepdims=True) if len(np.where(a != 0)[0]) > 0 else a
        b = b / np.linalg.norm(b, axis=-1, keepdims=True) if len(np.where(b != 0)[0]) > 0 else b
        return np.matmul(a, b.T)
    else:
        raise ValueError("input value must in [torch.Tensor, numpy.ndarray], but it is %s, %s"%(type(a), type(b)))

def calc_map_k(qB, rB, query_L, retrieval_L, k=None, rank=0):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # query_L: {0,1}^{mxl}
    # retrieval_L: {0,1}^{nxl}
    num_query = query_L.shape[0]
    qB = torch.sign(qB)
    rB = torch.sign(rB)
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    # print("query num:", num_query)
    for iter in range(num_query):
        q_L = query_L[iter]
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0)      # [1, hash length]
        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32)
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calcHammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.to(rank)
        map = map + torch.mean(count / tindex)
    map = map / num_query
    return map

def softmax_hash(code: Union[torch.Tensor, np.ndarray], dim_alph=0.25):

        device = None
        if isinstance(code, torch.Tensor):
            device = code.device
            if code.is_cuda:
                code = code.detach.cpu().numpy()
            else:
                code = code.cpu().numpy()
        
        softmax_code = np.exp(code) / np.exp(code).sum(axis=-1, keepdims=True)
        # print("max softmax_code:", softmax_code.max())
        # print("min softmax_code:", softmax_code.min())
        # print(1 / int(dim_alph * softmax_code.shape[-1]))
        # print(np.sum(softmax_code[0]))
        hash_code = np.where(softmax_code >= 1 / int(dim_alph * softmax_code.shape[-1]), softmax_code, -1)
        hash_code = np.where(hash_code <= 1 / int(dim_alph * softmax_code.shape[-1]), hash_code, 1)
        # print(hash_code[0])
        # print("hash code sum:", hash_code.sum())
        # print(len(np.where(hash_code == 1.0)[0]))
        # print(len(np.where(hash_code == -1.0)[0]))
        if device is not None:
            hash_code = torch.from_numpy(hash_code).to(device)
        return hash_code

# def calcHammingDist(B1, B2):
#     result = np.zeros((B1.shape[0], B2.shape[0]))
#     for i, data in enumerate(B1):
#         result[i] = np.sum(np.where((data + B2) != 2, data + B2, 0), axis=-1)
#     return result

def calcHammingDist(B1, B2):

    if len(B1.shape) < 2:
        B1.view(1, -1)
    if len(B2.shape) < 2:
        B2.view(1, -1)
    q = B2.shape[1]
    if isinstance(B1, torch.Tensor):
        distH = 0.5 * (q - torch.matmul(B1, B2.t()))
    elif isinstance(B1, np.ndarray):
        distH = 0.5 * (q - np.matmul(B1, B2.transpose()))
    else:
        raise ValueError("B1, B2 must in [torch.Tensor, np.ndarray]")
    return distH

def compute_hash_similarity(visual_embed, text_embed, use_softmax_hash=False, alph=0.25):
    # hamming distance的值越大说明越不相似
    hash_visual = encode_hash(visual_embed) if not use_softmax_hash else softmax_hash(visual_embed, alph)
    hash_text = encode_hash(text_embed) if not use_softmax_hash else softmax_hash(text_embed, alph)
    vt_similarity = calcHammingDist(hash_visual, hash_text)
    # print(vt_similarity[0])
    tv_similarity = calcHammingDist(hash_text, hash_visual)
    return vt_similarity, tv_similarity

class CrossEn(nn.Module):
    def __init__(self, mode="cosine"):
        super(CrossEn, self).__init__()
        # if mode == "euclidean":
        #     self.compute_func = F.softmax
        # else:
        #     self.compute_func = F.log_softmax
        self.mode = mode

    def forward(self, sim_matrix):
        # if self.mode == "cosine":
        #     logpt = F.log_softmax(sim_matrix, dim=-1)
        #     logpt = torch.diag(logpt)
        #     nce_loss = -logpt
        #     sim_loss = nce_loss.mean()
        # elif self.mode == "euclidean":
        #     logpt = F.softmax(sim_matrix, dim=-1)
        #     logpt = torch.diag(sim_matrix)
        #     sim_loss = logpt.mean()
        # else:
        #     raise ValueError("mode paramater is not support.[cosine, euclidean]")
        if self.mode == "euclidean":
            sim_matrix = -sim_matrix
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class CrossEn_mean(nn.Module):
    def __init__(self, mode="cosine"):
        super(CrossEn_mean, self).__init__()
        # if mode == "euclidean":
        #     self.compute_func = F.softmax
        # else:
        #     self.compute_func = F.log_softmax
        self.mode = mode

    def forward(self, sim_matrix):
        # if self.mode == "cosine":
        #     logpt = F.log_softmax(sim_matrix, dim=-1)
        #     logpt = torch.diag(logpt)
        #     nce_loss = -logpt
        #     sim_loss = nce_loss.mean()
        # elif self.mode == "euclidean":
        #     logpt = F.softmax(sim_matrix, dim=-1)
        #     logpt = torch.diag(sim_matrix)
        #     sim_loss = logpt.mean()
        # else:
        #     raise ValueError("mode paramater is not support.[cosine, euclidean]")
        # if self.mode == "euclidean":
        #     sim_matrix = -sim_matrix
        # print(sim_matrix.max(), sim_matrix.min())
        # logpt = F.log_softmax(sim_matrix, dim=-1)
        # logpt = torch.diag(logpt)
        # print(logpt.max())
        sim_loss = sim_matrix.mean()
        return sim_loss
