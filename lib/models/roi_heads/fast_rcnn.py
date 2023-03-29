# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from numpy import isin
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

from torchvision.ops import nms
import pdb
import time

__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers_baseline"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(boxes, scores, image_shapes, objness_scores,
                        score_thresh, nms_thresh, topk_per_image,
                        use_unknown=False, num_classes=80, reverse_label_converter=None, obj_thresh=0.5):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    scores_cls = scores["probs"]
    scores_ova = scores["probs_ova"]
    scores_ood = scores["probs_ood"]
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, image_shape, objness_scores_per_image,
            score_thresh, nms_thresh, topk_per_image, use_unknown, num_classes,
            reverse_label_converter, score_ova_per_image, obj_thresh, scores_ood_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape, objness_scores_per_image, score_ova_per_image, scores_ood_per_image in zip(scores_cls, boxes, image_shapes, objness_scores, scores_ova, scores_ood)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes, scores, image_shape, objness_scores, score_thresh, nms_thresh, topk_per_image,
    use_unknown=False, num_classes=80, reverse_label_converter=None, score_ova=None, obj_thresh=0.5, scores_ood=None
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)

    if reverse_label_converter is not None:
        ignore_void = reverse_label_converter[-1] == -1
    else:
        ignore_void = scores.shape[1] == num_classes + 1

    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        objness_scores = objness_scores[valid_mask]
        if score_ova is not None:
            score_ova = score_ova[valid_mask]
        if scores_ood is not None:
            scores_ood = scores_ood[valid_mask]

    original_scores = scores.clone()
    if score_ova is not None:
        original_score_ova = score_ova.clone()
    if ignore_void:
        scores = scores[:,:-1]
        if score_ova is not None:
            score_ova = score_ova[:, :, :-1]
    else:
        scores = scores[:, :-2]
        if score_ova is not None:
            score_ova = score_ova[:, :, :-2]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    if scores.shape[1] > num_classes:
        filter_mask = scores[:,:-1] > score_thresh
    else:
        filter_mask = scores > score_thresh  # R x K
    # new constraint: ova score, add by haiming, 2021/12/08
    if score_ova is not None:
        if score_ova.shape[-1] > num_classes:
            known_obj_mask = score_ova[:, :, :-1].argmax(1) == 1
        else:
            known_obj_mask = score_ova.argmax(1) == 1
        filter_mask = torch.logical_and(filter_mask, known_obj_mask)
    # new constraint: ood score, add by haiming, 2021/12/23
    if scores_ood is not None:
        in_dist_mask = scores_ood > 0.5
        filter_mask = torch.logical_and(filter_mask, in_dist_mask.unsqueeze(-1).expand_as(filter_mask))

    if use_unknown:
        new_filter_mask = filter_mask.sum(-1) < 1
        if original_scores.shape[1] > num_classes+1 or not ignore_void:
            new_filter_mask = torch.logical_and(new_filter_mask, original_scores.argmax(-1) == num_classes)
        # objness_scores = objness_scores.sigmoid()
        obj_th = obj_thresh #0.500
        unknown_filter_mask = torch.logical_and(new_filter_mask, objness_scores > obj_th)
        unknown_filter_inds = unknown_filter_mask.nonzero()
        unknown_boxes = boxes[unknown_filter_inds[:,0], 0]
        unknown_scores = objness_scores[unknown_filter_inds[:,0]]
        keep = nms(unknown_boxes, unknown_scores, nms_thresh)
        keep = keep[:int(topk_per_image*0.5)]
        unknown_boxes = unknown_boxes[keep]
        unknown_scores = unknown_scores[keep]
        unknown_filter_inds = unknown_filter_inds[keep]

    if scores.shape[1] > num_classes:
        scores = scores[:,:-1]

    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]


    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    if use_unknown :
        boxes = torch.cat((boxes, unknown_boxes), dim=0)
        scores = torch.cat((scores, unknown_scores), dim=0)
        if ignore_void:
            classes = torch.cat((filter_inds[:,1], -torch.ones(len(unknown_scores), device=filter_inds.device).long()), dim=0)
        else:
            classes = torch.cat((filter_inds[:,1], -2 * torch.ones(len(unknown_scores), device=filter_inds.device).long()), dim=0)

    else:
        classes = filter_inds[:,-1]
    if reverse_label_converter is not None:
        classes = reverse_label_converter.to(classes.device)[classes]

    boxes = boxes[:topk_per_image]
    scores = scores[:topk_per_image]
    classes = classes[:topk_per_image]


    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = classes
    inds = filter_inds[:,0]
    if use_unknown:
        inds = torch.cat((inds, unknown_filter_inds[:,0]))
    inds = inds[:topk_per_image]
    return result, inds


class FastRCNNOutputs:
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        void_logits,
        void_suppression=False,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        box_reg_loss_weight=1.0,
        label_converter=None,
        reverse_label_converter=None,
        add_unlabeled_class=True,
        cls_weight=None,
        bg_class_ind=None,
        is_bce=False,
        num_classes=None,
        ova_clf_activate=False,
        ova_ent_weight=0.0,
        ova_void_suppression=False,
        mix_criterion=None,
        mixup_default_cls_loss=False,
        aux_obj_head=False,
        aux_obj_head_pseudo_inst=False,
        ood_scores=None,
        ood_train_type='separately',
        probabilistic_det=False,
        strong_aug=False,
        void_pseudo_labeling=False,
        adv_obj=False,
        known_inst_only=False,
        centerness_head=False,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            box_reg_loss_weight (float): Weight for box regression loss
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.ova_clf_activate = ova_clf_activate
        self.aux_obj_head = aux_obj_head
        self.aux_obj_head_pseudo_inst = aux_obj_head_pseudo_inst
        self.centerness_head = centerness_head
        self.ova_ent_weight = ova_ent_weight
        self.probabilistic_det = probabilistic_det
        if self.ova_clf_activate:
            assert isinstance(pred_class_logits, tuple)
            self.pred_class_logits, self.pred_class_logits_ova = pred_class_logits
        elif self.aux_obj_head:
            assert isinstance(pred_class_logits, tuple)
            self.pred_class_logits, self.pred_aux_obj = pred_class_logits
        else:
            self.pred_class_logits = pred_class_logits
            self.pred_class_logits_ova = None
            self.pred_aux_obj = None
        
        self.ova_void_suppression = ova_void_suppression
        if ova_void_suppression:
            self.void_logits, self.void_logits_ova = void_logits
        elif probabilistic_det:
            self.void_logits, self.void_aux_obj = void_logits
        else:
            self.void_logits = void_logits
        self.void_suppression = void_suppression
        self.num_classes = num_classes
        self.mix_criterion = mix_criterion
        self.mixup_default_cls_loss = mixup_default_cls_loss

        self.ood_scores = ood_scores
        self.ood_train_type = ood_train_type
        self.strong_aug = strong_aug
        self.void_pseudo_labeling = void_pseudo_labeling
        self.adv_obj = adv_obj
        self.known_inst_only = known_inst_only

        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight
        self.label_converter = label_converter
        self.reverse_label_converter = reverse_label_converter
        self.add_unlabeled_class = add_unlabeled_class
        self.cls_weight = cls_weight
        if bg_class_ind is None:
            bg_class_ind = num_classes - 1
            if self.add_unlabeled_class:
                bg_class_ind = bg_class_ind - 1
        self.bg_class_ind = bg_class_ind
        self.is_bce = is_bce

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                # print(f'heuristicPL num: {(self.gt_classes < 0).sum()}')
                if self.label_converter is not None:
                    self.gt_classes = self.label_converter.to(self.gt_classes.device)[self.gt_classes]

                if self.centerness_head:
                    centerness = []
                    for p in proposals:
                        _gt_box = p.gt_boxes.tensor
                        _proposal_box = p.proposal_boxes.tensor
                        _proposal_center_loc = ((_proposal_box[:, 2] + _proposal_box[:, 0])/2, (_proposal_box[:, 3] + _proposal_box[:, 1])/2)
                        _l = _proposal_center_loc[0] - _gt_box[:, 0]
                        _r = _gt_box[:, 2] - _proposal_center_loc[0]
                        _t = _proposal_center_loc[1] - _gt_box[:, 1]
                        _b = _gt_box[:, 3] - _proposal_center_loc[1]
                        _filter_mask = torch.logical_and(torch.logical_and(_l>=0, _r>=0), torch.logical_and(_t>=0, _b>=0))
                        left_right = torch.stack([_l, _r], 1)
                        top_bottom = torch.stack([_t, _b], 1)
                        _centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
                        _centerness[~_filter_mask] = 0.0
                        centerness.append(torch.sqrt(_centerness))
                    self.gt_centerness = cat(centerness, dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0 or len(self.gt_classes) == 0# no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes != self.bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == self.bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self.mixup_default_cls_loss:
            pred_class_logits_origin, pred_class_logits = self.pred_class_logits
        else:
            pred_class_logits = self.pred_class_logits
        if self._no_instances:
            return 0.0 * pred_class_logits.sum()
        else:
            # self._log_accuracy()
            assert not self.is_bce, 'bce loss is false by default'
            if self.is_bce:
                gt_classes = F.one_hot(self.gt_classes, pred_class_logits.shape[-1])
                prob = pred_class_logits.sigmoid() #softmax(-1)
                loss = F.binary_cross_entropy(prob, gt_classes.float(), self.cls_weight, reduction='sum')
                if len(prob) > 0:
                    return loss / len(prob)
                else:
                    return loss
            
            if self.mix_criterion is not None:
                mixup_loss = self.mix_criterion(pred_class_logits)
                if self.mixup_default_cls_loss:
                    default_loss = F.cross_entropy(pred_class_logits_origin, self.gt_classes, reduction="mean", weight=self.cls_weight)
                    return (mixup_loss + default_loss) / 2
                return mixup_loss
            else:
                if self.probabilistic_det:
                    pred_class_probs = self.pred_class_logits.softmax(-1)
                    num_inst = pred_class_logits.shape[0]
                    obj_probs = self.pred_aux_obj.sigmoid()[:num_inst]
                    known_cls_probs = pred_class_probs[:, :-1].clone() * obj_probs
                    bg_probs = pred_class_probs[:, -1:].clone() * (1 - obj_probs)
                    all_probs = torch.cat([known_cls_probs, bg_probs], 1)
                    gt_classes = F.one_hot(self.gt_classes, pred_class_logits.shape[-1])
                    return -torch.sum(gt_classes * torch.log(all_probs+1e-8), dim=1).mean()
                else:
                    return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean", weight=self.cls_weight)

    def void_suppression_loss(self):
        if len(self.void_logits) > 0:
            if self.probabilistic_det:
                void_probs = self.void_logits.softmax(-1)[:, :self.num_classes-1]
                void_obj_probs = self.void_aux_obj.sigmoid().detach()
                prob_void_probs = void_probs.clone() * void_obj_probs
                void_neg_loss = -torch.log(1-prob_void_probs+1e-8)
            else:
                # v1: only suppress known things
                void_neg_loss = -torch.log(1-self.void_logits.softmax(-1)[:, :self.num_classes-1]+1e-8)
                # void_neg_loss = -torch.log(1-self.void_logits[:, :self.num_classes-1].softmax(-1)+1e-8)  # this implementation is worse than the upper one
                # v2: suppress both known & bg classes
                # void_neg_loss = -torch.log(1-self.void_logits.softmax(-1)+1e-8)
            if len(void_neg_loss) > 0:
                void_neg_loss = void_neg_loss.sum() / len(void_neg_loss)
            else:
                void_neg_loss = void_neg_loss.sum()
        else:
            if isinstance(self.pred_class_logits, tuple):
                void_neg_loss = 0.0 * self.pred_class_logits[0].sum()
            else:
                void_neg_loss = 0.0 * self.pred_class_logits.sum()
        return void_neg_loss
    
    def ova_loss(self):
        logits_open = self.pred_class_logits_ova.view(self.pred_class_logits_ova.size(0), 2, -1)
        logits_open = F.softmax(logits_open, 1)
        label_s_sp = torch.zeros((logits_open.size(0), logits_open.size(2))).long().to(self.gt_classes.device)
        label_range = torch.range(0, logits_open.size(0) - 1).long()
        label_s_sp[label_range, self.gt_classes] = 1
        label_sp_neg = 1 - label_s_sp
        open_loss = torch.mean(torch.sum(-torch.log(logits_open[:, 1, :] + 1e-8) * label_s_sp, 1))
        # open_loss_neg = torch.mean(torch.max(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1)[0])
        open_loss_neg = torch.mean(torch.mean(-torch.log(logits_open[:, 0, :] + 1e-8) * label_sp_neg, 1))
        Lo = open_loss_neg + open_loss

        if self.ova_ent_weight:
            logits_open = logits_open.view(logits_open.size(0), 2, -1)
            logits_open = F.softmax(logits_open, 1)
            Le = torch.mean(torch.mean(torch.sum(-logits_open * torch.log(logits_open + 1e-8), 1), 1))
            return Lo, self.ova_ent_weight * Le
        return Lo

    def ova_void_suppression_loss(self):
        if len(self.void_logits_ova) > 0:
            logits_open = self.void_logits_ova.view(self.void_logits_ova.size(0), 2, -1)
            logits_open = F.softmax(logits_open, 1)
            label_s_sp = torch.ones((logits_open.size(0), logits_open.size(2))).long().to(self.gt_classes.device)
            void_neg_loss = torch.mean(torch.mean(-torch.log(logits_open[:, 0, :] + 1e-8) * label_s_sp, 1))
        else:
            void_neg_loss = 0.0 * self.pred_class_logits.sum()
        return void_neg_loss

    def aux_obj_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_aux_obj.sum()
        else:
            if self.centerness_head:
                obj_gt = self.gt_centerness
            else:
                obj_gt = (self.gt_classes < (self.num_classes-1)).to(torch.float32)
                if self.strong_aug:
                    pseudo_inst_num = self.pred_aux_obj.shape[0] - 2 * obj_gt.shape[0]
                    if self.aux_obj_head_pseudo_inst and pseudo_inst_num > 0:
                        obj_gt = torch.cat([obj_gt, obj_gt, torch.ones(pseudo_inst_num).to(obj_gt.device)])
                    else:
                        obj_gt = torch.cat([obj_gt, obj_gt])
                elif self.adv_obj:
                    pseudo_inst_num = self.pred_aux_obj.shape[0] - 2 * obj_gt.shape[0]
                    if self.aux_obj_head_pseudo_inst and pseudo_inst_num > 0:
                        if self.known_inst_only:
                            filter_mask = torch.cat([torch.ones_like(obj_gt), obj_gt, torch.ones(pseudo_inst_num).to(obj_gt.device)])
                        obj_gt = torch.cat([obj_gt, obj_gt, torch.ones(pseudo_inst_num).to(obj_gt.device)])
                    else:
                        if self.known_inst_only:
                            filter_mask = torch.cat([torch.ones_like(obj_gt), obj_gt])
                        obj_gt = torch.cat([obj_gt, obj_gt])
                    if self.known_inst_only:
                        filter_mask = filter_mask == 1
                        self.pred_aux_obj = self.pred_aux_obj[filter_mask]
                        obj_gt = obj_gt[filter_mask]
                else:
                    pseudo_inst_num = self.pred_aux_obj.shape[0] - obj_gt.shape[0]
                    if (self.void_pseudo_labeling or self.aux_obj_head_pseudo_inst) and pseudo_inst_num > 0:
                        obj_gt = torch.cat([obj_gt, torch.ones(pseudo_inst_num).to(obj_gt.device)])
            return F.binary_cross_entropy_with_logits(self.pred_aux_obj.squeeze(), obj_gt, reduction="mean")

    def ood_loss(self):
        if self._no_instances:
            return 0.0 * self.pred_aux_obj.sum()
        else:
            assert self.ood_scores is not None, 'ood loss is called but ood_scores is None!'
            num_proposal = len(self.gt_classes)
            in_dist_scores = self.ood_scores[:num_proposal][self.gt_classes < (self.num_classes-1)]
            num_in_dist = in_dist_scores.shape[0]
            out_dist_scores = self.ood_scores[num_proposal:]
            num_out_dist = out_dist_scores.shape[0]
            samples = []
            gts = []
            if num_in_dist > 0:
                samples.append(in_dist_scores)
                in_gt = torch.ones(num_in_dist).to(self.ood_scores.device)
                gts.append(in_gt)
            if num_out_dist > 0:
                samples.append(out_dist_scores)
                out_gt = torch.zeros(num_out_dist).to(self.ood_scores.device)
                gts.append(out_gt)
            if len(samples) > 0:
                return F.binary_cross_entropy_with_logits(torch.cat(samples, dim=0).squeeze(), torch.cat(gts), reduction="mean")
            else:
                return 0.0 * self.pred_aux_obj.sum()

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < self.bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg * self.box_reg_loss_weight / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        loss_dict = {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss()}

        if self.void_suppression:
            loss_dict.update({"loss_void_neg": self.void_suppression_loss()})

        if self.ova_clf_activate:
            ova_losses = self.ova_loss()
            if isinstance(ova_losses, tuple):
                ova_loss, ova_ent = ova_losses
            else:
                ova_loss = ova_losses
                ova_ent = torch.zeros_like(ova_loss)
            loss_dict.update({"loss_ova": ova_loss, "loss_ova_ent": ova_ent})
        
        if self.ova_void_suppression:
            loss_dict.update({"loss_ova_void_neg": self.ova_void_suppression_loss()})
        
        if self.aux_obj_head:
            loss_dict.update({"loss_aux_obj": self.aux_obj_loss()})

        if self.ood_scores is not None:
            loss_dict.update({"loss_ood": self.ood_loss()})
        return loss_dict

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image,
            reverse_label_converter=self.reverse_label_converter
        )


class FastRCNNOutputLayers_baseline(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        box_reg_loss_type="smooth_l1",
        box_reg_loss_weight=1.0,
        add_unlabeled_class=False,
        label_converter=None,
        reverse_label_converter=None,
        void_background=False,
        void_ignorance=False,
        void_suppression=False,
        void_suppression_weight=1.0,
        ova_clf_activate=False,
        ova_ent_weight=0.0,
        ova_void_suppression=False,
        mixup_default_cls_loss=False,
        aux_obj_head=False,
        obj_thresh=0.5,
        aux_obj_head_pseudo_inst=False,
        ood_head=False,
        ood_train_type='separately',
        probabilistic_det=False,
        strong_aug=False,
        void_pseudo_labeling=False,
        adv_obj=False,
        known_inst_only=False,
        information_bottleneck=False,
        infoBtnk_compression_rate=0.5,
        infoBtnk_detach_feat=False,
        infoBtnk_single_layer=False,
        centerness_head=False,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            box_reg_loss_weight (float): Weight for box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.label_converter = label_converter
        self.reverse_label_converter = reverse_label_converter
        if add_unlabeled_class: # For old job (before runnning 1027 16:05), it should after the below condition.
            num_classes = num_classes + 1
        if self.reverse_label_converter is not None:
            num_classes = min(num_classes+1, len(reverse_label_converter))
        num_cls = num_classes
        self.add_unlabeled_class = add_unlabeled_class
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_cls - 1
        box_dim = len(box2box_transform.weights)
        self.num_cls = num_cls
        if void_suppression:
            self.cls_score = Linear(input_size, num_cls+128)
        else:
            self.cls_score = Linear(input_size, num_cls)
        if ova_clf_activate:
            self.cls_ova_score = Linear(input_size, 2*num_cls, bias=False)
        if aux_obj_head:
            if information_bottleneck:
                compression_dim = int(input_size * infoBtnk_compression_rate)
                if infoBtnk_single_layer:
                    self.aux_obj_score = nn.Sequential(
                        Linear(input_size, compression_dim),
                        nn.ReLU(),
                        Linear(compression_dim, 1))
                else:
                    self.aux_obj_score = nn.Sequential(
                        Linear(input_size, compression_dim),
                        nn.ReLU(),
                        Linear(compression_dim, compression_dim),
                        nn.ReLU(),
                        Linear(compression_dim, 1))
            else:
                self.aux_obj_score = Linear(input_size, 1)
        if ood_head:
            self.ood_score = Linear(input_size, 1)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        self.box_reg_loss_weight = box_reg_loss_weight

        self.void_background = void_background
        self.void_ignorance = void_ignorance
        self.void_suppression = void_suppression
        self.void_suppression_weight = void_suppression_weight
        self.ova_clf_activate = ova_clf_activate
        self.ova_ent_weight = ova_ent_weight
        self.ova_void_suppression = ova_void_suppression
        self.mixup_default_cls_loss = mixup_default_cls_loss
        self.aux_obj_head = aux_obj_head
        self.information_bottleneck = information_bottleneck
        self.infoBtnk_detach_feat = infoBtnk_detach_feat
        self.obj_thresh = obj_thresh
        self.aux_obj_head_pseudo_inst = aux_obj_head_pseudo_inst
        self.ood_head = ood_head
        self.ood_train_type = ood_train_type
        self.probabilistic_det = probabilistic_det
        self.strong_aug = strong_aug
        self.void_pseudo_labeling = void_pseudo_labeling
        self.adv_obj = adv_obj
        self.known_inst_only = known_inst_only
        self.centerness_head = centerness_head

    @classmethod
    def from_config(cls, cfg, input_shape, label_converter=None, reverse_label_converter=None):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes"           : cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg" : cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta"        : cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh"     : cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh"       : cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image"   : cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type"     : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "box_reg_loss_weight"   : cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT,
            "add_unlabeled_class": cfg.MODEL.EOPSN.UNLABELED_REGION and (not cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION),
            "label_converter": label_converter,
            "reverse_label_converter": reverse_label_converter,
            "void_background": cfg.MODEL.EOPSN.VOID_BACKGROUND,
            "void_ignorance": cfg.MODEL.EOPSN.VOID_IGNORANCE,
            "void_suppression": cfg.MODEL.EOPSN.VOID_SUPPRESSION,
            "void_suppression_weight" : cfg.MODEL.EOPSN.VOID_SUPPRESSION_WEIGHT,
            "ova_clf_activate": cfg.MODEL.OVA.ACTIVATE,
            "ova_ent_weight": cfg.MODEL.OVA.OVA_ENT_WEIGHT,
            "ova_void_suppression": cfg.MODEL.OVA.VOID_SUPPRESSION,
            "aux_obj_head": cfg.MODEL.EOPSN.AUX_OBJ_HEAD,
            "mixup_default_cls_loss": cfg.MODEL.EOPSN.MIXUP_DEFAULT_CLS_LOSS,
            "obj_thresh": cfg.MODEL.EOPSN.OBJ_SCORE_THRESHOLD,
            "aux_obj_head_pseudo_inst": cfg.MODEL.EOPSN.AUX_OBJ_HEAD_PSEUDO_INST,
            "ood_head": cfg.MODEL.EOPSN.OOD_HEAD,
            "ood_train_type": cfg.MODEL.EOPSN.OOD_TRAIN_TYPE,
            "probabilistic_det": cfg.MODEL.EOPSN.PROBABILISTIC_DET,
            "strong_aug": cfg.DATASETS.STRONG_AUG,
            "void_pseudo_labeling": cfg.MODEL.EOPSN.VOID_PSEUDO_LABELING,
            "adv_obj": cfg.MODEL.ADVERSARIAL_OBJHEAD.ACTIVATE,
            "known_inst_only": cfg.MODEL.ADVERSARIAL_OBJHEAD.KNOWN_INST_ONLY,
            "information_bottleneck": cfg.MODEL.INFORMATION_BOTTLENECK.ACTIVATE,
            "infoBtnk_compression_rate": cfg.MODEL.INFORMATION_BOTTLENECK.COMPRESSION_RATE,
            "infoBtnk_detach_feat": cfg.MODEL.INFORMATION_BOTTLENECK.DETACH_FEAT,
            "infoBtnk_single_layer": cfg.MODEL.INFORMATION_BOTTLENECK.SINGLE_LAYER,
            "centerness_head": cfg.MODEL.EOPSN.CENTERNESS_HEAD
            # fmt: on
        }

    def forward(self, x):
        """
        Returns:
            Tensor: shape (N,K+1), scores for each of the N box. Each row contains the scores for
                K object categories and 1 background class.
            Tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4), or (N,4)
                for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        if self.void_suppression:
            scores = self.cls_score(x)[:, :self.num_cls]
        else:
            scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        ret = {
            "scores": scores,
            "proposal_deltas": proposal_deltas,
        }
        if self.ova_clf_activate:
            ret["scores_ova"] = self.cls_ova_score(x)
        if self.aux_obj_head:
            if self.infoBtnk_detach_feat:    
                ret["scores_aux_obj"] = self.aux_obj_score(x.detach())
            else:
                ret["scores_aux_obj"] = self.aux_obj_score(x)
        if self.ood_head:
            ret["scores_ood"] = self.ood_score(x)
        return ret

    def get_logits(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score.forward_freeze(x)
        return scores

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals, void_predictions, mix_criterion=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        proposal_deltas = predictions["proposal_deltas"]
        if self.ova_clf_activate:
            _scores = predictions["scores"]
            scores_ova = predictions["scores_ova"]
            scores = (_scores, scores_ova)

            if void_predictions is not None:
                _void_scores = void_predictions["scores"]
                void_scores_ova = void_predictions["scores_ova"]
                if self.ova_void_suppression:
                    void_scores = (_void_scores, void_scores_ova)
                else:
                    void_scores = _void_scores
            else:
                void_scores = None
        elif self.aux_obj_head:
            _scores = predictions["scores"]
            scores_aux_obj = predictions["scores_aux_obj"]
            scores = (_scores, scores_aux_obj)
            # void-scores are ignored
            if void_predictions is not None:
                if self.probabilistic_det:
                    void_scores = (void_predictions["scores"], void_predictions["scores_aux_obj"])
                else:
                    void_scores = void_predictions["scores"]
            else:
                void_scores = None
        else:
            scores = predictions["scores"]
            if void_predictions is not None:
                void_scores = void_predictions["scores"]
            else:
                void_scores = None
        
        ood_scores = torch.cat([predictions["scores_ood"], void_predictions["scores_ood"]], dim=0) if self.ood_head else None
        losses = FastRCNNOutputs(
            self.box2box_transform,
            scores,
            proposal_deltas,
            proposals,
            void_scores,
            self.void_suppression,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
            self.box_reg_loss_weight,
            self.label_converter,
            add_unlabeled_class=self.add_unlabeled_class,
            num_classes=self.num_cls,
            ova_clf_activate=self.ova_clf_activate,
            ova_ent_weight=self.ova_ent_weight,
            ova_void_suppression=self.ova_void_suppression,
            mix_criterion=mix_criterion,
            mixup_default_cls_loss=self.mixup_default_cls_loss,
            aux_obj_head=self.aux_obj_head,
            aux_obj_head_pseudo_inst=self.aux_obj_head_pseudo_inst,
            ood_scores=ood_scores,
            ood_train_type=self.ood_train_type,
            probabilistic_det=self.probabilistic_det,
            strong_aug=self.strong_aug,
            void_pseudo_labeling=self.void_pseudo_labeling,
            adv_obj=self.adv_obj,
            known_inst_only=self.known_inst_only,
            centerness_head=self.centerness_head,
        ).losses()

        if self.void_suppression:
            losses['loss_void_neg'] *= self.void_suppression_weight
        return losses

    def inference(self, predictions, proposals, use_unknown=False):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        objness_scores = scores["probs_objness"]

        image_shapes = [x.image_size for x in proposals]
        if self.void_suppression or self.void_background or self.void_ignorance:
            num_classes = len(self.reverse_label_converter)
        elif self.ova_clf_activate:
            num_classes = len(self.reverse_label_converter) - 1
        else:
            num_classes = len(self.reverse_label_converter) - 2
        
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            objness_scores,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            use_unknown,
            reverse_label_converter=self.reverse_label_converter,
            num_classes=num_classes,
            obj_thresh=self.obj_thresh,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        proposal_deltas = predictions["proposal_deltas"]
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        num_inst_per_image = [len(p) for p in proposals]
        scores = predictions["scores"]
        probs = F.softmax(scores, dim=-1)
        if self.probabilistic_det:
            obj_probs = predictions["scores_aux_obj"].sigmoid()
            known_cls_probs = probs[:, :-1].clone() * obj_probs
            bg_probs = probs[:, -1:].clone() * (1 - obj_probs)
            all_probs = torch.cat([known_cls_probs, bg_probs], 1)
            probs_split = all_probs.split(num_inst_per_image, dim=0)
        else:
            probs_split = probs.split(num_inst_per_image, dim=0)

        ret_probs = {
            "probs": probs_split
        }

        if self.aux_obj_head:
            _objness_scores = predictions["scores_aux_obj"]
            ret_probs["probs_objness"] = [_objness_scores.squeeze().sigmoid()]
        else:
            ret_probs["probs_objness"] = [x.objectness_logits.sigmoid() for x in proposals]

        if self.ova_clf_activate:
            scores_ova = predictions["scores_ova"]
            logits_ova = scores_ova.view(scores_ova.size(0), 2, -1)
            probs_ova = F.softmax(logits_ova, 1)
            probs_ova_split = probs_ova.split(num_inst_per_image, dim=0)
            ret_probs["probs_ova"] = probs_ova_split
        else:
            ret_probs["probs_ova"] = (None, )
    
        if self.ood_head:
            scores_ood = predictions["scores_ood"]
            probs_ood = torch.sigmoid(scores_ood.squeeze())
            probs_ood_split = probs_ood.split(num_inst_per_image, dim=0)
            ret_probs["probs_ood"] = probs_ood_split
        else:
            ret_probs["probs_ood"] = (None, )
        return ret_probs
