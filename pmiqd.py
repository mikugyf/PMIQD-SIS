
import torch
from torch import nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from ignite.metrics.metric import Metric
from scipy import stats
import h5py
import numpy as np
from PIL import Image
class PMIQA(nn.Module):
    def __init__(self):
        super(PMIQA, self).__init__()

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1x = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2x = nn.Conv2d(128, 128, 3, padding=1)
        self.conv3x = nn.Conv2d(256, 256, 3, padding=1)
        self.conv4x = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5x = nn.Conv2d(512, 512, 3, padding=1,dilation=(1,1))
        self.conv61   = nn.Conv2d(64, 128, 3, padding=1)
        self.conv12  = nn.Conv2d(128, 256, 3, padding=1)
        self.conv25   = nn.Conv2d(256, 512, 3, padding=1)
        self.maxpool = F.max_pool2d
        self.fcdown  = nn.Linear(1024, 512)
        self.fc1q_nr = nn.Linear(512, 512)
        self.fc2q_nr = nn.Linear(512, 1)
        self.fc1w_nr = nn.Linear(512, 512)
        self.fc2w_nr = nn.Linear(512, 1)
        self.dropout = nn.Dropout()
    def extract_features(self, x):

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1x(h))
        h = F.max_pool2d(h, kernel_size=(2, 2),stride=(2, 2))
        #conv_2
        h = F.relu(self.conv61(h))
        h = F.relu(self.conv2x(h))
        h = F.max_pool2d(h, kernel_size=(2, 2),stride=(2, 2))
        # conv_3
        h = F.relu(self.conv12(h))
        h = F.relu(self.conv3x(h))
        h = F.relu(self.conv3x(h))
        h = F.max_pool2d(h, kernel_size=(2, 2),stride=(2, 2),padding=1)
        # conv_4
        h = F.relu(self.conv25(h))
        h = F.relu(self.conv4x(h))
        h = F.relu(self.conv4x(h))
        h = F.max_pool2d(h, kernel_size=(2, 2),stride=(2, 2),padding=1)
        # conv_5
        h = F.relu(self.conv5x(h))
        h = F.relu(self.conv5x(h))
        h = F.relu(self.conv5x(h))

        h = F.max_pool2d(h, 2)

        h = h.view(-1,512)
        # h = self.fcdown(h)
        # h = self.extractor(x)
        return h

    def forward(self, x):
        batch_size = x.size(0)
        n_patches = x.size(1)
        q = torch.ones((batch_size * n_patches, 1), device=x.device)
        for i in range(batch_size):
            h = self.extract_features(x[i])
            h_ = h  # save intermediate features
            h = F.relu(self.fc1q_nr(h_))
            h = self.dropout(h)
            h = self.fc2q_nr(h)
            q[i * n_patches:(i + 1) * n_patches] = h
        return q
class NystromAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]

        self.num_landmarks = config["num_landmarks"]
        self.seq_len = config["seq_len"]

        if "inv_coeff_init_option" in config:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"

        self.use_conv = "conv_kernel_size" in config
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (config["conv_kernel_size"], 1), padding = (config["conv_kernel_size"] // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask):

        Q = Q * mask[:, None, :, None] / np.math.sqrt(np.math.sqrt(self.head_dim))
        K = K * mask[:, None, :, None] / np.math.sqrt(np.math.sqrt(self.head_dim))

        if self.num_landmarks == self.seq_len:
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, self.seq_len // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel_3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)) - 1e9 * (1 - mask[:, None, None, :]), dim = -1)
            X = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        if self.use_conv:
            X += self.conv(V * mask[:, None, :, None])

        return X

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat

        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0.
            V = 1 / torch.max(torch.sum(K, dim = -2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence.
            V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V
class normer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(dim=dim,dim_head=dim // 8,heads=8,num_landmarks=dim // 2,pinv_iterations=6,residual=True,dropout=0.1)
    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
class Patcher(nn.Module):
    def __init__(self, n_classes):
        super(Patcher, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = normer(dim=512)
        self.layer2 = normer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, **kwargs):
        h = kwargs['data'].float()  # [B, n, 1024]
        h = self._fc1(h)  # [B, n, 512]
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.layer1(h)  # [B, N, 512]
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        h = self.layer2(h)  # [B, N, 512]
        h = self.norm(h)[:, 0]
        logits = self._fc2(h)  # [B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict
def pathload(path):
    return Image.open(path).convert('RGB')
def Crop(im, patch_size=32):

    w, h = im.size
    patches = ()
    stride = patch_size
    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))
            patches = patches + (patch,)
    return torch.stack(patches)
def model():
    model_select = PMIQA()   #use
    return model_select
class Datasetl(Dataset):
    """
    IQA Dataset (less memory)
    """
    def __init__(self, args, status='train', loader=pathload):
        """
        :param args:
        :param status: train/val/test
        :param loader: image loader
        """
        self.status = status
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches
        self.loader = loader
        exp_id = 0
        Info = h5py.File(args.datamat, 'r')
        index = Info['index']
        index = index[:, exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]  #

        K = 5
        k = 5
        testindex = index[int((k-1)/K * len(index)):int(k/K * len(index))]
        valindex = index[range(-int((5-k)/K * len(index)), -int((4-k)/K * len(index)))]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in testindex:
                test_index.append(i)
            elif ref_ids[i] in valindex:
                val_index.append(i)
                train_index.append(i)  #
            else:
                train_index.append(i)
        if 'train' in status:
            self.index = train_index
            # print("# Train Images: {}".format(len(self.index)))
        if 'test' in status:
            self.index = test_index
            # print("# Test Images: {}".format(len(self.index)))
            # findextest = open('indextest.csv', 'w', newline='' "")
            # csv_writer = csv.writer(findextest)
            # csv_writer.writerow(test_index)
        if 'val' in status:
            self.index = val_index
            # print("# Val Images: {}".format(len(self.index)))
            # findexval = open('indexval.csv', 'w', newline='' "")
            # csv_writer = csv.writer(findexval)
            # csv_writer.writerow(val_index)
        # print(self.index)

        self.scale = Info['subjective_scores'][0, :].max()
        self.mos = Info['subjective_scores'][0, self.index] / self.scale #
        self.mos_std = Info['subjective_scoresSTD'][0, self.index] / self.scale #
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]

        self.patches = ()
        self.label = []
        self.label_std = []
        self.im_names = []

        for idx in range(len(self.index)):
            self.im_names.append(os.path.join(args.imgset, im_names[idx]))

            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        im = self.loader(self.im_names[idx])
        patches = Crop(im, self.patch_size)
        return patches, (torch.Tensor([self.label[idx], ]), torch.Tensor([self.label_std[idx], ]))
class LossFuc(torch.nn.Module):
    def __init__(self):
        super(LossFuc, self).__init__()
    def forward(self, y_pred, y):
        n = int(y_pred.size(0) / y[0].size(0))
        w = 0.5
        loss = w*F.smooth_l1_loss(y_pred, y[0].repeat((1, n)).reshape((-1, 1)))+(1-w)*F.multilabel_soft_margin_loss(y_pred, y[0].repeat((1, n)).reshape((-1, 1)))
        return loss
class PerformanceEva(Metric):

    def reset(self):
        self._y_pred = []
        self._y      = []
        self._y_std  = []

    def update(self, output):
        y_pred, y = output

        self._y.append(y[0].item())
        self._y_std.append(y[1].item())
        n = int(y_pred.size(0) / y[0].size(0))
        y_pred_im = y_pred.reshape((y[0].size(0), n)).mean(dim=1, keepdim=True)
        self._y_pred.append(y_pred_im.item())

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))
        sq_std = np.reshape(np.asarray(self._y_std), (-1,))
        q = np.reshape(np.asarray(self._y_pred), (-1,))
        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()
        outlier_ratio = (np.abs(sq - q) > 2 * sq_std).mean()
        return srocc, krocc, plcc, rmse, mae, outlier_ratio,sq,q
class Dataset(Dataset):
    def __init__(self, args, status='train', loader=pathload):

        self.status = status
        self.patch_size = args.patch_size
        self.n_patches = args.n_patches
        exp_id = 0
        Info = h5py.File(args.datamat, 'r')
        index = Info['index']
        index = index[:, exp_id % index.shape[1]]
        ref_ids = Info['ref_ids'][0, :]
        K = 5
        k = 5
        testindex = index[int((k-1)/K * len(index)):int(k/K * len(index))]
        valindex = index[range(-int((5-k)/K * len(index)), -int((4-k)/K * len(index)))]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            if ref_ids[i] in testindex:
                test_index.append(i)
            elif ref_ids[i] in valindex:
                val_index.append(i)
                train_index.append(i)  #
            else:
                train_index.append(i)
        if 'train' in status:
            self.index = train_index
            # print("# Train Images: {}".format(len(self.index)))
        if 'test' in status:
            self.index = test_index
            # print("# Test Images: {}".format(len(self.index)))
            # np.savetxt('test_index.csv', test_index, delimiter=',')
        if 'val' in status:
            self.index = val_index
            # print("# Val Images: {}".format(len(self.index)))
            # print("XXXXX val index",val_index)
            # np.savetxt('val_index.csv', val_index, delimiter=',')

        # print(self.index)

        self.scale = Info['subjective_scores'][0, :].max()
        self.mos = Info['subjective_scores'][0, self.index] / self.scale #
        self.mos_std = Info['subjective_scoresSTD'][0, self.index] / self.scale #
        im_names = [Info[Info['im_names'][0, :][i]][()].tobytes()[::2].decode() for i in self.index]


        self.patches = ()
        self.label = []
        self.label_std = []
        self.ims = []
        for idx in range(len(self.index)):
            print("Preprocessing Image: {}".format(im_names[idx]))
            im = loader(os.path.join(args.imgset, im_names[idx]))
            self.label.append(self.mos[idx])
            self.label_std.append(self.mos_std[idx])

            if status == 'train':
                self.ims.append(im)
            elif status == 'test' or status == 'val':
                patches = Crop(im, args.patch_size)
                self.patches = self.patches + (patches,)  #

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.status == 'train':
            patches = Crop(self.ims[idx], self.patch_size)
        else:
            patches = self.patches[idx]
        return patches, (torch.Tensor([self.label[idx], ]), torch.Tensor([self.label_std[idx], ]))
def get_data_loaders(args):
    train_dataset = Datasetl(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0)  #
    val_dataset = Dataset(args, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)
    test_dataset = Dataset(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset)
    thescale = test_dataset.scale
    return train_loader, val_loader, test_loader, thescale



