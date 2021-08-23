from . import BaseActor
import torch
import torch.nn as nn

from pytracking.analysis.vos_utils import davis_jaccard_measure


class SegActor(BaseActor):
    """Actor for training the JOINT network."""
    def __init__(self, net, objective, loss_weight=None,
                 num_refinement_iter=3,
                 disable_backbone_bn=False,
                 disable_all_bn=False):
        """
        args:
            net - The network model to train
            objective - Loss functions
            loss_weight - Weights for each training loss
            num_refinement_iter - Number of update iterations N^{train}_{update} used to update the target model in
                                  each frame
            disable_backbone_bn - If True, all batch norm layers in the backbone feature extractor are disabled, i.e.
                                  set to eval mode.
            disable_all_bn - If True, all the batch norm layers in network are disabled, i.e. set to eval mode.
        """
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'segm': 1.0}
        self.loss_weight = loss_weight

        self.num_refinement_iter = num_refinement_iter
        self.disable_backbone_bn = disable_backbone_bn
        self.disable_all_bn = disable_all_bn

    def train(self, mode=True):
        """ Set whether the network is in train mode.
        args:
            mode (True) - Bool specifying whether in training mode.
        """
        self.net.train(mode)

        if self.disable_all_bn:
            self.net.eval()
        elif self.disable_backbone_bn:
            for m in self.net.feature_extractor.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'train_images', 'test_images', 'train_masks',
                    'test_masks'
        returns:
            loss    - the training loss
            stats  -  dict containing detailed losses
        """
        segm_pred, enc_tm, enc_tr = self.net(train_imgs=data['train_images'],
                                    test_imgs=data['test_images'],
                                    train_masks=data['train_masks'],
                                    test_masks=data['test_masks'],
                                    num_refinement_iter=self.num_refinement_iter)

        acc = 0
        cnt = 0

        segm_pred = segm_pred.view(-1, 1, *segm_pred.shape[-2:])
        gt_segm = data['test_masks']
        gt_segm = gt_segm.view(-1, 1, *gt_segm.shape[-2:])

        loss_segm = self.loss_weight['segm'] * self.objective['segm'](segm_pred, gt_segm)
        loss_diff = self.loss_weight['diff'] * self.objective['diff'](enc_tm, enc_tr)

        acc_l = [davis_jaccard_measure(torch.sigmoid(rm.detach()).cpu().numpy() > 0.5, lb.cpu().numpy()) for
                 rm, lb in zip(segm_pred.view(-1, *segm_pred.shape[-2:]), gt_segm.view(-1, *segm_pred.shape[-2:]))]
        acc += sum(acc_l)
        cnt += len(acc_l)

        loss = loss_segm + loss_diff

        if torch.isinf(loss) or torch.isnan(loss):
            raise Exception('ERROR: Loss was nan or inf!!!')

        # Log stats
        stats = {'Loss/total': loss.item(),
                 'Loss/segm': loss_segm.item(),
                 'Loss/diff': loss_diff.item(),
                 'Stats/acc': acc / cnt}

        return loss, stats
