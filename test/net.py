# -*- coding: utf-8 -*-
# @Author: Song Dejia
# @Date:   2018-11-05 11:16:24
# @Last Modified by:   Song Dejia
# @Last Modified time: 2018-11-23 15:44:42
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np
from data_loader import TrainDataLoader
from collections import namedtuple
import cv2


from got10k.trackers import Tracker

model_urls = {'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}

class SiameseRPNOld(nn.Module):
    def __init__(self, test_video=False):
        super(SiameseRPN, self).__init__()
        self.features = nn.Sequential(                  #1, 3, 256, 256
            nn.Conv2d(3, 64, kernel_size=11, stride=2), #1, 64,123, 123
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1, 64, 60,  60
            nn.Conv2d(64, 192, kernel_size=5),          #1,192, 56,  56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1,192, 27,  27
            nn.Conv2d(192, 384, kernel_size=3),         #1,384, 25,  25
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3),         #1,256, 23,  23
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3),         #1,256, 21,  21
        )

        self.k = 5
        self.s = 4
        self.conv1 = nn.Conv2d(256, 2*self.k*256, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4*self.k*256, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(256, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(256, 4* self.k, kernel_size = 4, bias = False)

        #self.reset_params() # we will not reset parameter

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Load Alexnet models Done' )

    def forward(self, template, detection):
        template = self.features(template)
        detection = self.features(detection)

        ckernal = self.conv1(template)
        ckernal = ckernal.view(2* self.k, 256, 4, 4)
        cinput  = self.conv3(detection)


        rkernal = self.conv2(template)
        rkernal = rkernal.view(4* self.k, 256, 4, 4)
        rinput  = self.conv4(detection)

        coutput = F.conv2d(cinput, ckernal)
        routput = F.conv2d(rinput, rkernal)
        print('coutput', coutput.shape)
        print('routput', routput.shape)

        #coutput = coutput.squeeze().permute(1,2,0).reshape(-1, 2)
        #routput = routput.squeeze().permute(1,2,0).reshape(-1, 4)
        return routput, coutput

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        print('Resume checkpoint from {}'.format(weight))

class SiameseRPN(nn.Module):

    def __init__(self, test_video=False):
        super(SiameseRPN, self).__init__()
        self.features = nn.Sequential(                  #1, 3, 256, 256
            #conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=2), #1, 64,123, 123
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1, 64, 60,  60
            #conv2
            nn.Conv2d(64, 192, kernel_size=5),          #1,192, 56,  56
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),      #1,192, 27,  27
            #conv3
            nn.Conv2d(192, 384, kernel_size=3),         #1,384, 25,  25
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            #conv4
            nn.Conv2d(384, 256, kernel_size=3),         #1,256, 23,  23
            #nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            #conv5
            nn.Conv2d(256, 256, kernel_size=3),         #1,256, 21,  21
            #nn.BatchNorm2d(256)
        )

        self.k = 5
        self.s = 4
        self.conv1 = nn.Conv2d(256, 2*self.k*256, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 4*self.k*256, kernel_size=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.relu4 = nn.ReLU(inplace=True)

        self.cconv = nn.Conv2d(256, 2* self.k, kernel_size = 4, bias = False)
        self.rconv = nn.Conv2d(256, 4* self.k, kernel_size = 4, bias = False)

        #self.reset_params() # we will not reset parameter

    def reset_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        print('Load Alexnet models Done' )

    def forward(self, template, detection):
        template = self.features(template)
        detection = self.features(detection)

        ckernal = self.conv1(template)
        ckernal = ckernal.view(2* self.k, 256, 4, 4)
        cinput  = self.conv3(detection)


        rkernal = self.conv2(template)
        rkernal = rkernal.view(4* self.k, 256, 4, 4)
        rinput  = self.conv4(detection)

        coutput = F.conv2d(cinput, ckernal)
        routput = F.conv2d(rinput, rkernal)

        #coutput = coutput.squeeze().permute(1,2,0).reshape(-1, 2)
        #routput = routput.squeeze().permute(1,2,0).reshape(-1, 4)
        return routput, coutput

    def resume(self, weight):
        checkpoint = torch.load(weight)
        self.load_state_dict(checkpoint)
        print('Resume checkpoint from {}'.format(weight))

class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)
        self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiameseRPN()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

    def parse_args(self, **kargs):
        self.cfg = {
            'exemplar_sz': 127,
            'instance_sz': 271,
            'total_stride': 8,
            'context': 0.5,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8,],
            'penalty_k': 0.055,
            'window_influence': 0.42,
            'lr': 0.295}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(image.shape[:2]) < 0.004:
            self.cfg = self.cfg._replace(instance_sz=287)

        # generate anchors
        self.response_sz = (self.cfg.instance_sz - \
            self.cfg.exemplar_sz) // self.cfg.total_stride + 1
        self.anchors = self._create_anchors(self.response_sz)

        # create hanning window
        self.hann_window = np.outer(
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            self.cfg.exemplar_sz, self.avg_color)

        # classification and regression kernels
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            #self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image)
            self.exemplar_image = exemplar_image

    def update(self, image):
        image = np.asarray(image)

        # search image
        instance_image = self._crop_and_resize(
            image, self.center, self.x_sz,
            self.cfg.instance_sz, self.avg_color)

        # classification and regression outputs
        instance_image = torch.from_numpy(instance_image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        print('instance_image', instance_image.shape)
        print('self.exemplar_image', self.exemplar_image.shape)
        with torch.set_grad_enabled(False):
            self.net.eval()
            out_reg, out_cls = self.net(self.exemplar_image, instance_image)

        # offsets
        print('out_reg', out_reg.shape)
        print('out_cls', out_cls.shape)

        offsets = out_reg.permute(
            1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]

        # scale and ratio penalty
        penalty = self._create_penalty(self.target_sz, offsets)

        # response
        response = F.softmax(out_cls.permute(
            1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        response = response * penalty
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window

        # peak location
        best_id = np.argmax(response)
        offset = offsets[:, best_id] * self.z_sz / self.cfg.exemplar_sz

        # update center
        self.center += offset[:2][::-1]
        self.center = np.clip(self.center, 0, image.shape[:2])

        # update scale
        lr = response[best_id] * self.cfg.lr
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]
        self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])

        # update exemplar and instance sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def _create_anchors(self, response_sz):
        anchor_num = len(self.cfg.ratios) * len(self.cfg.scales)
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = self.cfg.total_stride * self.cfg.total_stride
        ind = 0
        for ratio in self.cfg.ratios:
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)
            for scale in self.cfg.scales:
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(
            anchors, response_sz * response_sz).reshape((-1, 4))

        begin = -(response_sz // 2) * self.cfg.total_stride
        xs, ys = np.meshgrid(
            begin + self.cfg.total_stride * np.arange(response_sz),
            begin + self.cfg.total_stride * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)

        return anchors

    def _create_penalty(self, target_sz, offsets):
        def padded_size(w, h):
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)

        src_sz = padded_size(
            *(target_sz * self.cfg.exemplar_sz / self.z_sz))
        dst_sz = padded_size(offsets[2], offsets[3])
        change_sz = larger_ratio(dst_sz / src_sz)

        src_ratio = target_sz[1] / target_sz[0]
        dst_ratio = offsets[2] / offsets[3]
        change_ratio = larger_ratio(dst_ratio / src_ratio)

        penalty = np.exp(-(change_ratio * change_sz - 1) * \
            self.cfg.penalty_k)

        return penalty

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

class TrackerSiamRPN12(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)

        # setup GPU device if available
        # self.cuda = torch.cuda.is_available()
        # self.device = torch.device('cuda:0' if self.cuda else 'cpu')


        '''setup model'''
        self.net = SiameseRPN()
        #self.net = self.net.cuda()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        # self.net = self.net.to(self.device)

    def init(self, image, box):

        """ dataloader """
        self.data_loader = TrainDataLoader(image, box)
        self.box = box

    def update(self, image):

        ret = self.data_loader.__get__(image)
        template = ret['template_tensor']#.cuda()
        detection= ret['detection_tensor']#.cuda()

        cout, rout = self.net(template, detection) #[1, 10, 17, 17], [1, 20, 17, 17]

        cout = cout.reshape(-1, 2)
        rout = rout.reshape(-1, 4)
        cout = cout.cpu().detach().numpy()
        score = 1/(1 + np.exp(cout[:,1]-cout[:,0]))
        diff   = rout.cpu().detach().numpy() #1445

        num_proposals = 1
        score_64_index = np.argsort(score)[::-1][:num_proposals]

        score64 = score[score_64_index]
        diffs64 = diff[score_64_index, :]
        anchors64 = ret['anchors'][score_64_index]
        proposals_x = (anchors64[:, 0] + anchors64[:, 2] * diffs64[:, 0]).reshape(-1, 1)
        proposals_y = (anchors64[:, 1] + anchors64[:, 3] * diffs64[:, 1]).reshape(-1, 1)
        proposals_w = (anchors64[:, 2] * np.exp(diffs64[:, 2])).reshape(-1, 1)
        proposals_h = (anchors64[:, 3] * np.exp(diffs64[:, 3])).reshape(-1, 1)
        proposals = np.hstack((proposals_x, proposals_y, proposals_w, proposals_h))

        box = proposals

        #print('self.box', self.box)
        #print('box', box)

        return box

if __name__ == '__main__':
    model = SiameseRPN()

    template = torch.ones((1, 3, 127, 127))
    detection= torch.ones((1, 3, 256, 256))

    y1, y2 = model(template, detection)
    print(y1.shape) #[1, 10, 17, 17]
    print(y2.shape) #[1, 20, 17, 17]15
