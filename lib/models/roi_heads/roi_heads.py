import inspect
import logging
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torch.nn.functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.roi_heads.box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers_baseline
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads import ROIHeads, ROI_HEADS_REGISTRY
from detectron2.data import MetadataCatalog

from util.misc import add_unlabeled_class

logger = logging.getLogger('__name__')

def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes < bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret

@ROI_HEADS_REGISTRY.register()
class StandardROIHeads_baseline(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        label_converter,
        num_classes_known,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        ignore_unlabeled_region: bool = False,
        void_background=False,
        mixup_alpha,
        mixup_known_unk,
        mixup_unk_unk,
        mixup_default_cls_loss,
        mixup_known_known,
        mixup_unk_known,
        mixup_exchange,
        mixup_loss_type,
        pseudo_suppression,
        aux_obj_head,
        aux_obj_head_pseudo_inst,
        mahalanobis_ood_detector=False,
        strong_aug=False,
        void_pseudo_labeling=False,
        void_obj_confidence=0.95,
        pseudo_inst_pseudo_labeling=False,
        pseudo_inst_obj_confidence=0.5,
        adv_obj=False,
        adv_eps=1.0,
        detach_feat=False,
        **kwargs
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask head.
                None if not using mask head.
            mask_pooler (ROIPooler): pooler to extra region features for mask head
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.label_converter = label_converter
        self.strong_aug = strong_aug
        self.void_pseudo_labeling = void_pseudo_labeling
        self.void_obj_confidence = void_obj_confidence
        self.pseudo_inst_pseudo_labeling = pseudo_inst_pseudo_labeling
        self.pseudo_inst_obj_confidence = pseudo_inst_obj_confidence
        self.adv_obj = adv_obj
        self.adv_eps = adv_eps
        self.detach_feat = detach_feat

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes
        self.ignore_unlabeled_region = ignore_unlabeled_region
        self.void_background = void_background
        self.mixup_alpha = mixup_alpha
        self.mixup_known_unk = mixup_known_unk
        self.mixup_unk_unk = mixup_unk_unk
        self.mixup_default_cls_loss = mixup_default_cls_loss
        self.mixup_known_known = mixup_known_known
        self.mixup_unk_known = mixup_unk_known
        self.mixup_exchange = mixup_exchange
        self.pseudo_suppression = pseudo_suppression
        self.aux_obj_head = aux_obj_head
        self.aux_obj_head_pseudo_inst = aux_obj_head_pseudo_inst
        self.mixup_loss_type = mixup_loss_type
        self.num_classes_known = num_classes_known
        self.mahalanobis_ood_detector = mahalanobis_ood_detector
        self.roi_feats = {'feat': [], 'label': [], 'ood_feat': []}


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        ret["ignore_unlabeled_region"] = cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION
        ret["void_background"] = cfg.MODEL.EOPSN.VOID_BACKGROUND
        ret["mixup_alpha"] = cfg.MODEL.EOPSN.MIXUP_ALPHA
        ret["mixup_known_unk"] = cfg.MODEL.EOPSN.MIXUP_KNOWN_UNK
        ret["mixup_known_known"] = cfg.MODEL.EOPSN.MIXUP_KNOWN_KNOWN
        ret["mixup_unk_unk"] = cfg.MODEL.EOPSN.MIXUP_UNK_UNK
        ret["mixup_unk_known"] = cfg.MODEL.EOPSN.MIXUP_UNK_KNOWN
        ret["mixup_default_cls_loss"] = cfg.MODEL.EOPSN.MIXUP_DEFAULT_CLS_LOSS
        ret["mixup_exchange"] = cfg.MODEL.EOPSN.MIXUP_EXCHANGE
        ret["mixup_loss_type"] = cfg.MODEL.EOPSN.MIXUP_LOSS_TYPE
        ret["pseudo_suppression"] = cfg.MODEL.EOPSN.PSEUDO_SUPPRESSION
        ret["aux_obj_head"] = cfg.MODEL.EOPSN.AUX_OBJ_HEAD
        ret["aux_obj_head_pseudo_inst"] = cfg.MODEL.EOPSN.AUX_OBJ_HEAD_PSEUDO_INST
        ret["mahalanobis_ood_detector"] = cfg.MODEL.MAHALANOBIS_OOD_DETECTOR.ACTIVATE
        ret["strong_aug"] = cfg.DATASETS.STRONG_AUG
        ret["void_pseudo_labeling"] = cfg.MODEL.EOPSN.VOID_PSEUDO_LABELING
        ret["void_obj_confidence"] = cfg.MODEL.EOPSN.VOID_OBJ_CONFIDENCE
        ret["pseudo_inst_pseudo_labeling"] = cfg.MODEL.EOPSN.PSEUDO_INST_PSEUDO_LABELING
        ret["pseudo_inst_obj_confidence"] = cfg.MODEL.EOPSN.PSEUDO_INST_OBJ_CONFIDENCE
        ret["adv_obj"] = cfg.MODEL.ADVERSARIAL_OBJHEAD.ACTIVATE
        ret["adv_eps"] = cfg.MODEL.ADVERSARIAL_OBJHEAD.ADV_EPS
        ret["detach_feat"] = cfg.MODEL.ADVERSARIAL_OBJHEAD.DETACH_FEAT
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))


        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        test_unseen_path = cfg.DATASETS.TEST_UNSEEN_LABEL_SET
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        if unseen_path != '':
            meta_info = {e: i for i, e in enumerate(meta.thing_classes)}
            with open(unseen_path, 'r') as f:
                lines = [meta_info[e.replace('\n','')] for e in f.readlines()]
            if test_unseen_path != '':
                with open(test_unseen_path, 'r') as f:
                    test_unseen_label_set = [meta_info[e.replace('\n','')] for e in f.readlines()]
                lines += test_unseen_label_set
            unseen_label_set = sorted(lines)
            meta.stuff_classes.append('unknown')
            meta.stuff_colors.append([20, 220, 60])
            meta.stuff_dataset_id_to_contiguous_id[201] = 54
            pre_process_unk = cfg.MODEL.EOPSN.PRE_PROCESS_UNK
            pseudo_suppression = cfg.MODEL.EOPSN.PSEUDO_SUPPRESSION
            if pseudo_suppression or (not pre_process_unk and (cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION or not cfg.MODEL.EOPSN.UNLABELED_REGION)):
            # if cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION or not cfg.MODEL.EOPSN.UNLABELED_REGION:
                label_converter = torch.ones(len(meta.thing_classes) + 1)
            else:
                label_converter = torch.ones(len(meta.thing_classes) + 2)
            for i in unseen_label_set:
                label_converter[i] = 0
            reverse_label_converter = label_converter.nonzero()[:,0].long()
            label_converter = torch.cumsum(label_converter, 0).long() - 1
            if cfg.MODEL.EOPSN.UNLABELED_REGION:
                if cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION:
                    reverse_label_converter[-1] = -1
                else:
                    reverse_label_converter[-1] = reverse_label_converter[-2]
                    reverse_label_converter[-2] = -1
        else:
            reverse_label_converter = None
            label_converter = None

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        box_predictor = FastRCNNOutputLayers_baseline(cfg, box_head.output_shape, label_converter, reverse_label_converter)

        num_classes_known = min(cfg.MODEL.ROI_HEADS.NUM_CLASSES+1, len(reverse_label_converter))

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "label_converter": label_converter,
            "num_classes_known": num_classes_known,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["mask_head"] = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["keypoint_head"] = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
        return ret


    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], integral_sem_seg_target=None
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []
        void_proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for i, (proposals_per_image, targets_per_image) in enumerate(zip(proposals, targets)):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            if integral_sem_seg_target is not None and self.void_background is False:
                gt_classes, filtered_idx = add_unlabeled_class(proposals_per_image.proposal_boxes.tensor,
                                    gt_classes, integral_sem_seg_target[i], bg=self.num_classes)
                if self.ignore_unlabeled_region:
                    neg_filtered_idx = torch.logical_not(filtered_idx)
                    void_proposals_per_image = proposals_per_image[neg_filtered_idx]
                    gt_classes = gt_classes[filtered_idx]
                    gt_classes[gt_classes>self.num_classes] = self.num_classes
                    proposals_per_image = proposals_per_image[filtered_idx]
                else:
                    void_proposals_per_image = None
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                if self.ignore_unlabeled_region and self.void_background is False:
                    sampled_targets = sampled_targets[filtered_idx]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)
            if self.ignore_unlabeled_region and self.void_background is False:
                void_proposals_with_gt.append(void_proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))
        if not self.ignore_unlabeled_region or self.void_background is True:
            return proposals_with_gt, None
        return proposals_with_gt, void_proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        integral_sem_seg_target: Optional[List[torch.Tensor]] = None,
        image_path=None, flips=None, pseudo_instances=None, features_str=None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals, void_proposals = self.label_and_sample_proposals(proposals, targets, integral_sem_seg_target)
        if not self.mahalanobis_ood_detector:
            del targets

        if self.training:
            losses = self._forward_box(features, proposals, void_proposals, pseudo_instances=pseudo_instances, features_str=features_str)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            loss_masks = self._forward_mask(features, proposals)
            losses.update(loss_masks)
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            if self.mahalanobis_ood_detector:
                # in-dist inst features
                gt_boxes = [x.gt_boxes for x in targets]
                gt_classes = [x.gt_classes for x in targets]
                gt_boxes_features = self._extract_roi_feat(features, gt_boxes, proposals)
                for gt_box_features, gt_cls in zip(gt_boxes_features, gt_classes):
                    for idx in range(gt_cls.shape[0]):
                        _gt_cls = gt_cls[idx].item()
                        _gt_box_feat = gt_box_features[idx]
                        self.roi_feats['feat'].append(torch.mean(_gt_box_feat.view(_gt_box_feat.size(0), _gt_box_feat.size(1), -1), 2).view(1, -1).cpu().numpy())
                        self.roi_feats['label'].append(_gt_cls)
                # pseudo-inst, potential ood
                if len(pseudo_instances) > 0:
                    ood_boxes = [x.gt_boxes for x in pseudo_instances]
                    ood_boxes_features = self._extract_roi_feat(features, ood_boxes, proposals)[0]
                    for ood_box_features in ood_boxes_features:
                        self.roi_feats['ood_feat'].append(torch.mean(ood_box_features.view(ood_box_features.size(0), ood_box_features.size(1), -1), 2).view(1, -1).cpu().numpy())
            
            pred_instances = self._forward_box(features, proposals, image_path=image_path)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
    
    def _extract_roi_feat(self, features, interested_boxes, proposals):
        new_proposals = add_ground_truth_to_proposals(interested_boxes, proposals)
        interested_proposals = [
            new_proposals[idx][len(_proposal):]
            for idx, _proposal in enumerate(proposals)
        ]
        features = [features[f] for f in self.box_in_features]
        interested_boxes_features = [self.box_pooler(features, [x.proposal_boxes for x in interested_proposals])]
        return interested_boxes_features

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances


    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances], void_proposals: Optional[List[Instances]] = None,
        image_path=None, pseudo_instances=None, features_str=None
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features1 = self.box_head(box_features)
        predictions = self.box_predictor(box_features1)

        if self.training:
            num_cls = predictions['scores'].shape[1]
            if void_proposals is not None:
                void_box_features = self.box_pooler(features, [x.proposal_boxes for x in void_proposals])
                void_box_features1 = self.box_head(void_box_features)
                void_predictions = self.box_predictor(void_box_features1)
                unk_cls = torch.ones(len(void_box_features)).long().to(void_box_features.device)*(num_cls-1)

                if self.aux_obj_head and self.void_pseudo_labeling:
                    void_aux_obj_score = void_predictions["scores_aux_obj"]
                    void_aux_obj_prob = torch.sigmoid(void_aux_obj_score)
                    valid_void = void_aux_obj_score[void_aux_obj_prob >= self.void_obj_confidence].unsqueeze(1)
                    if len(valid_void) > 0:
                        aux_obj_score_aug_void = torch.cat([predictions["scores_aux_obj"], valid_void], dim=0)
                        predictions["scores_aux_obj"] = aux_obj_score_aug_void  # objectiveness on auxObjHead, pseudo-inst label is 1
                        # print(f'void PL num: {len(valid_void)}')
            else:
                void_predictions = None

            if self.strong_aug and features_str is not None:
                features_str = [features_str[f] for f in self.box_in_features]
                box_features_str = self.box_pooler(features_str, [x.proposal_boxes for x in proposals])
                box_features_str1 = self.box_head(box_features_str)
                predictions_str = self.box_predictor(box_features_str1)
                aux_obj_score_aug_str = torch.cat([predictions["scores_aux_obj"], predictions_str["scores_aux_obj"]], dim=0)
                predictions["scores_aux_obj"] = aux_obj_score_aug_str  # objectiveness on auxObjHead, same label as weak aug version proposal

            if self.adv_obj:
                box_features_adv = self.adv_feat(box_features, eps=self.adv_eps)
                box_features_adv1 = self.box_head(box_features_adv)
                predictions_adv = self.box_predictor(box_features_adv1)
                # self.adv_check(predictions['scores'], predictions_adv['scores'])
                aux_obj_score_aug_adv = torch.cat([predictions["scores_aux_obj"], predictions_adv["scores_aux_obj"]], dim=0)
                predictions["scores_aux_obj"] = aux_obj_score_aug_adv  # objectiveness on auxObjHead, same label as weak aug version proposal

            if self.mixup_alpha > 0:
                # mixup among known insts
                if self.mixup_known_known or self.mixup_known_unk:
                    gt_cls = torch.cat([p.gt_classes.clone() for p in proposals], dim=0)
                    gt_cls = self.label_converter.to(gt_cls.device)[gt_cls]
                    if self.mixup_known_unk:
                        mix_input_feat = torch.cat([box_features, void_box_features], dim=0)
                        mix_input_gt = torch.cat([gt_cls, unk_cls])
                    else:
                        mix_input_feat = box_features
                        mix_input_gt = gt_cls
                    mixed_box_features, mix_criterion = self.mixup_data(mix_input_feat, mix_input_gt, alpha=self.mixup_alpha, mix_kn=self.mixup_known_known, mixup_exchange=self.mixup_exchange)
                    mixed_box_features1 = self.box_head(mixed_box_features)
                    mixed_predictions = self.box_predictor(mixed_box_features1)
                    if self.mixup_default_cls_loss:
                        _predictions = (predictions["scores"], mixed_predictions["scores"]),
                    else:
                        _predictions = (mixed_predictions["scores"]),
                    predictions["scores"] = _predictions
                    del mixed_box_features, mixed_box_features1, mix_input_feat
                else:
                    mix_criterion = None

                # mix unk & unk
                if self.mixup_unk_unk:
                    mixed_void_box_features, _ = self.mixup_data(void_box_features, unk_cls, alpha=self.mixup_alpha)
                    mixed_void_box_features1 = self.box_head(mixed_void_box_features)
                    mixed_void_predictions = self.box_predictor(mixed_void_box_features1)
                    _void_predictions = (torch.cat([void_predictions["scores"], mixed_void_predictions["scores"]], dim=0)),
                    void_predictions["scores"] = _void_predictions
                    del mixed_void_box_features, mixed_void_box_features1
                elif self.mixup_unk_known:
                    gt_cls = torch.cat([p.gt_classes for p in proposals], dim=0)
                    gt_cls = self.label_converter.to(gt_cls.device)[gt_cls]
                    bg_cls = gt_cls.max().item()
                    kn_inst_mask = gt_cls != bg_cls
                    if kn_inst_mask.sum() > 0 and len(void_box_features) > 0:
                        kn_inst_feat = box_features[kn_inst_mask].clone().detach()
                        index = list(torch.utils.data.WeightedRandomSampler(torch.ones(kn_inst_mask.sum()), len(void_box_features), replacement=True))
                        mix_lambda = 0.9
                        mixed_x = mix_lambda * void_box_features + (1 - mix_lambda) * kn_inst_feat[index,:]
                        mixed_void_box_features1 = self.box_head(mixed_x)
                        mixed_void_predictions = self.box_predictor(mixed_void_box_features1)
                        _void_predictions = (torch.cat([void_predictions["scores"], mixed_void_predictions["scores"]], dim=0)),
                        void_predictions["scores"] = _void_predictions
            else:
                mix_criterion = None

            if len(pseudo_instances) > 0:
                # suppress pseudo-inst boxes
                pseudo_boxes = []
                feat_mask = torch.ones(len(pseudo_instances))
                for idx, x in enumerate(pseudo_instances):
                    if x == None:
                        feat_mask[idx] -= 1
                    else:
                        pseudo_boxes.append(x.gt_boxes)
                feat4pseudobox = [feat[feat_mask == 1].clone() for feat in features]
                pseudo_box_features = self.box_pooler(feat4pseudobox, pseudo_boxes)
                pseudo_box_features1 = self.box_head(pseudo_box_features)
                pseudo_predictions = self.box_predictor(pseudo_box_features1)
                _void_predictions = torch.cat([void_predictions["scores"], pseudo_predictions["scores"]], dim=0)
                void_predictions["scores"] = _void_predictions  # void supppression on classification head, do not need label info

                if self.aux_obj_head and self.aux_obj_head_pseudo_inst:
                    if self.pseudo_inst_pseudo_labeling:
                        pseudoInst_aux_obj_score = pseudo_predictions["scores_aux_obj"]
                        pseudoInst_aux_obj_prob = torch.sigmoid(pseudoInst_aux_obj_score)
                        valid_pseudoInst = pseudoInst_aux_obj_score[pseudoInst_aux_obj_prob >= self.pseudo_inst_obj_confidence].unsqueeze(1)
                        if len(valid_pseudoInst) > 0:
                            aux_obj_score_aug_pseudoInst = torch.cat([predictions["scores_aux_obj"], valid_pseudoInst], dim=0)
                            predictions["scores_aux_obj"] = aux_obj_score_aug_pseudoInst  # objectiveness on auxObjHead, pseudo-inst label is 1
                            # print(f'pseudoInst PL num: {len(valid_pseudoInst)}, total PL num: {len(pseudoInst_aux_obj_score)}')
                    else:
                        aux_obj_score_aug = torch.cat([predictions["scores_aux_obj"], pseudo_predictions["scores_aux_obj"]], dim=0)
                        predictions["scores_aux_obj"] = aux_obj_score_aug  # objectiveness on auxObjHead, pseudo-inst label is 1
                
                if "scores_ood" in pseudo_predictions:
                    void_predictions["scores_ood"] = torch.cat([void_predictions["scores_ood"], pseudo_predictions["scores_ood"]], dim=0)

            del box_features, box_features1
                
            losses = self.box_predictor.losses(predictions, proposals, void_predictions, mix_criterion)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, get_inds = self.box_predictor.inference(predictions, proposals, use_unknown=True)
            del box_features
            return pred_instances

    def adv_check(self, origin_pred, adv_pred):
        score_max, idx_max = torch.softmax(origin_pred, -1).max(1)
        score_adv_max, idx_adv_max = torch.softmax(adv_pred, -1).max(1)
        score_max[idx_max != 64], score_adv_max[idx_max != 64], idx_adv_max[idx_max != 64]
        

    @staticmethod
    def _l2_normalize(d):
        d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
        d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
        return d

    def adv_feat(self, roi_feat, xi=10.0, eps=1.0, ip=1):
        # prepare random unit tensor
        d = torch.rand(roi_feat.shape).sub(0.5).to(roi_feat.device)
        d = self._l2_normalize(d)
        
        target = torch.ones((roi_feat.shape[0], self.num_classes_known-1)).to(roi_feat.device) / (self.num_classes_known-1)
        for _ in range(ip):
            d.requires_grad_()
            roi_feat_noisy = roi_feat.detach() + xi * d
            box_features = self.box_head(roi_feat_noisy)
            predictions = self.box_predictor(box_features)
            scores = predictions["scores"][:, :-1]
            logp_hat = F.log_softmax(scores, dim=1)
            adv_distance = F.kl_div(logp_hat, target, reduction='batchmean')
            adv_distance.backward()
            d = self._l2_normalize(d.grad)
            self.box_head.zero_grad()
            self.box_predictor.zero_grad()
        
        if self.detach_feat:
            roi_feat_adv = roi_feat.detach() - d.detach() * eps
        else:
            roi_feat_adv = roi_feat - d.detach() * eps
        return roi_feat_adv

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True, mix_kn=False, mixup_exchange=False):
        '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda
        mix_kn: mix among known insts
        '''
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        batch_size, chn_num = x.shape[:2]
        num_cls = self.num_classes_known
        bg_cls = num_cls - 1
        kn_inst_mask = y != bg_cls
        if mix_kn and kn_inst_mask.sum() > 0:
            kn_inst_idx = kn_inst_mask.nonzero(as_tuple=True)[0]
            index = torch.arange(0, batch_size).long().to(y.device)
            index[kn_inst_mask] = kn_inst_idx[torch.randperm(kn_inst_mask.sum())]
        else:
            if use_cuda:
                index = torch.randperm(batch_size).to(x.device)
            else:
                index = torch.randperm(batch_size)
        if mixup_exchange:
            chn_num1 = int(chn_num * lam)
            mix_mask = torch.zeros(batch_size, chn_num).to(x.device)
            mix_mask[:, :chn_num1] = 1
            mix_mask = mix_mask[:, torch.randperm(mix_mask.size()[1])].unsqueeze(-1).unsqueeze(-1)
            mixed_x = x * mix_mask + x[index] * (1 - mix_mask)
        else:
            mixed_x = lam * x + (1 - lam) * x[index,:]
        y_a, y_b = y, y[index]
        mix_criterion = self.mixup_criterion(y_a, y_b, lam, loss_type=self.mixup_loss_type, num_cls=num_cls)
        return mixed_x, mix_criterion
    
    @staticmethod
    def mixup_criterion(y_a, y_b, lam, loss_type='ce', num_cls=None):
        if loss_type == 'ce':
            criterion = nn.CrossEntropyLoss(reduction="mean")
            return lambda pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
        else:
            # bce
            y_a_onehot = nn.functional.one_hot(y_a, num_cls)
            y_b_onehot = nn.functional.one_hot(y_b, num_cls)
            # lam = torch.from_numpy(np.array([lam]).astype('float32')).to(y_a.device)
            # y_reweighted = y_a_onehot * lam.expand_as(y_a_onehot) + y_b_onehot * (1 - lam.expand_as(y_b_onehot))
            y_reweighted = y_a_onehot * lam + y_b_onehot * (1 - lam)
            criterion = nn.BCELoss(reduction="none").to(y_a.device)
            return lambda pred: criterion(nn.functional.softmax(pred, dim=1), y_reweighted).sum(dim=1).mean()

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.mask_in_features]
        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.keypoint_in_features]

        if self.training:
            # The loss is defined on positive proposals with >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)
