import copy
import logging
import os
from PIL import Image

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks
from detectron2.structures import polygons_to_bitmask
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)

from util.misc import filter_unseen_class, filter_unseen_class_oracle, cum_map
import fvcore.transforms.transform as FT
from .unk_retrival import get_potential_unk

__all__ = ["DatasetMapper", "DatasetMapperExtractor", "DatasetMapperOrigin", "DatasetMapperOracleVerify", "DatasetMapperGT"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by EOPSN.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.unlabeled_region_on = cfg.MODEL.EOPSN.UNLABELED_REGION
        self.sem_seg_filter_unk = cfg.MODEL.EOPSN.SEM_SEG_FILTER_UNK
        self.pre_process_unk = cfg.MODEL.EOPSN.PRE_PROCESS_UNK
        if self.pre_process_unk:
            # detectron2/detectron2/modeling/roi_heads/roi_heads.py, Line 207
            # Since RPN takes the -1 as an ignored label by default, so we set -2 here
            # also, the last 2nd class represents the unk among the "label_converter"
            self.unk_cate_id = -2

        # Semantic Segmentation
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.sem_seg_unlabeled_region_on = cfg.MODEL.EOPSN.SEM_SEG_UNLABELED_REGION
        self.num_sem_seg_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        if unseen_path != '' and self.is_train:
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            self.unseen_label_set = self._get_unseen_label_set(meta, unseen_path)
        else:
            self.unseen_label_set = None

        if cfg.MODEL.LOAD_PROPOSALS:
            self.proposal_topk = (
                                    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                                    if is_train
                                    else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
                                )
        else:
            self.proposal_topk = None

        self.box_min_w = cfg.DATASETS.OPENPS.BOX_MIN_W
        self.box_min_h = cfg.DATASETS.OPENPS.BOX_MIN_H
        self.mask_box_ratio = cfg.DATASETS.OPENPS.MASK_BOX_RATIO
        self.box_range_ratio = cfg.DATASETS.OPENPS.BOX_RANGE_RATIO
        self.pseudo_suppression = cfg.MODEL.EOPSN.PSEUDO_SUPPRESSION

        self.strong_aug_flag = cfg.DATASETS.STRONG_AUG

    def _get_unseen_label_set(self, meta, path):
        meta = {e: i for i, e in enumerate(meta)}
        with open(path, 'r') as f:
            lines = [meta[e.replace('\n','')] for e in f.readlines()]

        return lines

    def strong_aug(self, img, mask):
        from .transform_yf import Rand_Augment
        # img_rgb = img[:, :, ::-1]
        # img_pil = Image.fromarray(img_rgb, mode="RGB")
        strong_trans = Rand_Augment(Numbers=3, Magnitude=20, max_Magnitude=40, p=1.0)
        img_aug, _ = strong_trans(img, mask)
        return img_aug
        
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        original_image  = image

        if self.crop_gen is None or np.random.rand() > 0.5:
            tfm_gens = self.tfm_gens
        else:
            tfm_gens = self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:]


        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            if self.is_train and self.pre_process_unk:
                unk_anno = get_potential_unk(sem_seg_gt, self.box_min_w, self.box_min_h, self.mask_box_ratio, self.box_range_ratio, category_id=self.unk_cate_id)
                if unk_anno is not None:
                    dataset_dict["annotations"] += unk_anno
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(original_image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(tfm_gens)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        # Image.fromarray(image[:, :, ::-1], mode="RGB").save('output/openps/')
        
        if self.strong_aug_flag:
            image_aug = self.strong_aug(image, sem_seg_gt)
            dataset_dict["image_aug"] = torch.as_tensor(np.ascontiguousarray(image_aug.transpose(2, 0, 1)))


        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            return dataset_dict

        if type(transforms[0]) is FT.NoOpTransform:
            flip = 0
        elif type(transforms[0]) is FT.HFlipTransform:
            flip = 1
        else:
            flip = 2
        dataset_dict["flip"] = flip

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.sem_seg_unlabeled_region_on:
                sem_seg_gt[sem_seg_gt==self.ignore_value] = self.num_sem_seg_classes
            dataset_dict["sem_seg"] = sem_seg_gt

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                   anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            if self.pseudo_suppression and unk_anno is not None:
                # split gt and pseudo insts
                dataset_dict["instances_pseudo"] = dataset_dict["instances"][-len(unk_anno):]
                dataset_dict["instances"] = dataset_dict["instances"][:-len(unk_anno)]

            if self.unseen_label_set is not None:
                dataset_dict["instances"] = filter_unseen_class(dataset_dict["instances"], self.unseen_label_set)
                if len(dataset_dict["instances"]) == 0:
                    return None

        if self.unlabeled_region_on:
            sem_seg = dataset_dict["sem_seg"].clone()
            if self.sem_seg_filter_unk:
                for idx, inst_cls in enumerate(dataset_dict["instances"].gt_classes):
                    if inst_cls == self.unk_cate_id:
                        _bit_mask = BitMasks.from_polygon_masks(dataset_dict["instances"].gt_masks[idx], *image_shape)
                        sem_seg[_bit_mask.tensor.squeeze()] = 0
            if self.sem_seg_unlabeled_region_on:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.num_sem_seg_classes)
            else:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.ignore_value)
            dataset_dict["integral_sem_seg"] = cum_sem_seg

        return dataset_dict


class DatasetMapperOrigin:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by EOPSN.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.unlabeled_region_on = cfg.MODEL.EOPSN.UNLABELED_REGION
        self.sem_seg_filter_unk = cfg.MODEL.EOPSN.SEM_SEG_FILTER_UNK
        self.pre_process_unk = cfg.MODEL.EOPSN.PRE_PROCESS_UNK
        if self.pre_process_unk:
            # detectron2/detectron2/modeling/roi_heads/roi_heads.py, Line 207
            # Since RPN takes the -1 as an ignored label by default, so we set -2 here
            # also, the last 2nd class represents the unk among the "label_converter"
            self.unk_cate_id = -2

        # Semantic Segmentation
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.sem_seg_unlabeled_region_on = cfg.MODEL.EOPSN.SEM_SEG_UNLABELED_REGION
        self.num_sem_seg_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        if unseen_path != '' and self.is_train:
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            self.unseen_label_set = self._get_unseen_label_set(meta, unseen_path)
        else:
            self.unseen_label_set = None

        if cfg.MODEL.LOAD_PROPOSALS:
            self.proposal_topk = (
                                    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                                    if is_train
                                    else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
                                )
        else:
            self.proposal_topk = None

        self.box_min_w = cfg.DATASETS.OPENPS.BOX_MIN_W
        self.box_min_h = cfg.DATASETS.OPENPS.BOX_MIN_H
        self.mask_box_ratio = cfg.DATASETS.OPENPS.MASK_BOX_RATIO
        self.box_range_ratio = cfg.DATASETS.OPENPS.BOX_RANGE_RATIO
        self.pseudo_suppression = cfg.MODEL.EOPSN.PSEUDO_SUPPRESSION

    def _get_unseen_label_set(self, meta, path):
        meta = {e: i for i, e in enumerate(meta)}
        with open(path, 'r') as f:
            lines = [meta[e.replace('\n','')] for e in f.readlines()]

        return lines
    
    def _filter_unk_inst(self, sem_seg, annos, image_size):
        annos_copy = copy.deepcopy(annos)
        sem_seg_copy = copy.deepcopy(sem_seg)
        for idx, anno in enumerate(annos_copy):
            cat_id = anno["category_id"]
            iscrowd = anno["iscrowd"]
            if self.unseen_label_set is not None and cat_id in self.unseen_label_set and iscrowd == 0:
                segmentation = anno['segmentation']  # list
                # polygons
                polygons = [np.array(p) for p in segmentation]
                masks = polygons_to_bitmask(polygons, *image_size)
                # sem_seg.setflags(write=1)
                sem_seg_copy[masks==1] = 255
                annos.remove(anno)
        return sem_seg_copy, annos

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        original_image  = image

        if self.crop_gen is None or np.random.rand() > 0.5:
            tfm_gens = self.tfm_gens
        else:
            tfm_gens = self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:]


        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            if self.is_train:
                sem_seg_gt, _annos = self._filter_unk_inst(sem_seg_gt, dataset_dict['annotations'], image.shape[:2])
                dataset_dict['annotations'] = _annos
                if self.pre_process_unk:
                    unk_anno = get_potential_unk(sem_seg_gt, self.box_min_w, self.box_min_h, self.mask_box_ratio, self.box_range_ratio, category_id=self.unk_cate_id)
                    if unk_anno is not None:
                        dataset_dict["annotations"] += unk_anno
                if len(dataset_dict['annotations']) == 0:
                    return None
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(original_image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(tfm_gens)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            return dataset_dict

        if type(transforms[0]) is FT.NoOpTransform:
            flip = 0
        elif type(transforms[0]) is FT.HFlipTransform:
            flip = 1
        else:
            flip = 2
        dataset_dict["flip"] = flip

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.sem_seg_unlabeled_region_on:
                sem_seg_gt[sem_seg_gt==self.ignore_value] = self.num_sem_seg_classes
            dataset_dict["sem_seg"] = sem_seg_gt

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                   anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            if self.pseudo_suppression and unk_anno is not None:
                # split gt and pseudo insts
                dataset_dict["instances_pseudo"] = dataset_dict["instances"][-len(unk_anno):]
                dataset_dict["instances"] = dataset_dict["instances"][:-len(unk_anno)]

            if self.unseen_label_set is not None:
                dataset_dict["instances"] = filter_unseen_class(dataset_dict["instances"], self.unseen_label_set)
                if len(dataset_dict["instances"]) == 0:
                    return None

        if self.unlabeled_region_on:
            sem_seg = dataset_dict["sem_seg"].clone()
            if self.sem_seg_filter_unk:
                for idx, inst_cls in enumerate(dataset_dict["instances"].gt_classes):
                    if inst_cls == self.unk_cate_id:
                        _bit_mask = BitMasks.from_polygon_masks(dataset_dict["instances"].gt_masks[idx], *image_shape)
                        sem_seg[_bit_mask.tensor.squeeze()] = 0
            if self.sem_seg_unlabeled_region_on:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.num_sem_seg_classes)
            else:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.ignore_value)
            dataset_dict["integral_sem_seg"] = cum_sem_seg

        return dataset_dict


class DatasetMapperOracleVerify:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by EOPSN.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.unlabeled_region_on = cfg.MODEL.EOPSN.UNLABELED_REGION
        self.sem_seg_filter_unk = cfg.MODEL.EOPSN.SEM_SEG_FILTER_UNK
        self.pre_process_unk = cfg.MODEL.EOPSN.PRE_PROCESS_UNK
        if self.pre_process_unk:
            # detectron2/detectron2/modeling/roi_heads/roi_heads.py, Line 207
            # Since RPN takes the -1 as an ignored label by default, so we set -2 here
            # also, the last 2nd class represents the unk among the "label_converter"
            self.unk_cate_id = -2

        # Semantic Segmentation
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.sem_seg_unlabeled_region_on = cfg.MODEL.EOPSN.SEM_SEG_UNLABELED_REGION
        self.num_sem_seg_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        if unseen_path != '' and self.is_train:
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            self.unseen_label_set = self._get_unseen_label_set(meta, unseen_path)
        else:
            self.unseen_label_set = None

        if cfg.MODEL.LOAD_PROPOSALS:
            self.proposal_topk = (
                                    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                                    if is_train
                                    else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
                                )
        else:
            self.proposal_topk = None

        self.box_min_w = cfg.DATASETS.OPENPS.BOX_MIN_W
        self.box_min_h = cfg.DATASETS.OPENPS.BOX_MIN_H
        self.mask_box_ratio = cfg.DATASETS.OPENPS.MASK_BOX_RATIO
        self.box_range_ratio = cfg.DATASETS.OPENPS.BOX_RANGE_RATIO
        self.pseudo_suppression = cfg.MODEL.EOPSN.PSEUDO_SUPPRESSION

    def _get_unseen_label_set(self, meta, path):
        meta = {e: i for i, e in enumerate(meta)}
        with open(path, 'r') as f:
            lines = [meta[e.replace('\n','')] for e in f.readlines()]

        return lines



    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        original_image  = image

        if self.crop_gen is None or np.random.rand() > 0.5:
            tfm_gens = self.tfm_gens
        else:
            tfm_gens = self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:]


        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            # if self.is_train and self.pre_process_unk:
            #     unk_anno = get_potential_unk(sem_seg_gt, self.box_min_w, self.box_min_h, self.mask_box_ratio, self.box_range_ratio, category_id=self.unk_cate_id)
            #     if unk_anno is not None:
            #         dataset_dict["annotations"] += unk_anno
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(original_image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(tfm_gens)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            return dataset_dict

        if type(transforms[0]) is FT.NoOpTransform:
            flip = 0
        elif type(transforms[0]) is FT.HFlipTransform:
            flip = 1
        else:
            flip = 2
        dataset_dict["flip"] = flip

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.sem_seg_unlabeled_region_on:
                sem_seg_gt[sem_seg_gt==self.ignore_value] = self.num_sem_seg_classes
            dataset_dict["sem_seg"] = sem_seg_gt

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                   anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            # if self.pseudo_suppression and unk_anno is not None:
            #     # split gt and pseudo insts
            #     dataset_dict["instances_pseudo"] = dataset_dict["instances"][-len(unk_anno):]
            #     if not self.obj_head_pseudo_inst:
            #         dataset_dict["instances"] = dataset_dict["instances"][:-len(unk_anno)]

            if self.unseen_label_set is not None:
                dataset_dict["instances"], known_idx, dataset_dict["instances_pseudo"] = filter_unseen_class_oracle(dataset_dict["instances"], self.unseen_label_set, category_id=self.unk_cate_id)
                dataset_dict["instances"] = dataset_dict["instances"][known_idx]
                if len(dataset_dict["instances"]) == 0:
                    return None

        if self.unlabeled_region_on:
            sem_seg = dataset_dict["sem_seg"].clone()
            if self.sem_seg_filter_unk:
                for idx, inst_cls in enumerate(dataset_dict["instances"].gt_classes):
                    if inst_cls == self.unk_cate_id:
                        _bit_mask = BitMasks.from_polygon_masks(dataset_dict["instances"].gt_masks[idx], *image_shape)
                        sem_seg[_bit_mask.tensor.squeeze()] = 0
            if self.sem_seg_unlabeled_region_on:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.num_sem_seg_classes)
            else:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.ignore_value)
            dataset_dict["integral_sem_seg"] = cum_sem_seg

        return dataset_dict


class DatasetMapperExtractor:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by EOPSN.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.unlabeled_region_on = cfg.MODEL.EOPSN.UNLABELED_REGION
        self.sem_seg_filter_unk = cfg.MODEL.EOPSN.SEM_SEG_FILTER_UNK
        self.pre_process_unk = cfg.MODEL.EOPSN.PRE_PROCESS_UNK
        if self.pre_process_unk:
            # detectron2/detectron2/modeling/roi_heads/roi_heads.py, Line 207
            # Since RPN takes the -1 as an ignored label by default, so we set -2 here
            # also, the last 2nd class represents the unk among the "label_converter"
            self.unk_cate_id = -2

        # Semantic Segmentation
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.sem_seg_unlabeled_region_on = cfg.MODEL.EOPSN.SEM_SEG_UNLABELED_REGION
        self.num_sem_seg_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        if unseen_path != '': # and self.is_train
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            self.unseen_label_set = self._get_unseen_label_set(meta, unseen_path)
        else:
            self.unseen_label_set = None

        if cfg.MODEL.LOAD_PROPOSALS:
            self.proposal_topk = (
                                    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                                    if is_train
                                    else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
                                )
        else:
            self.proposal_topk = None

        self.box_min_w = cfg.DATASETS.OPENPS.BOX_MIN_W
        self.box_min_h = cfg.DATASETS.OPENPS.BOX_MIN_H
        self.mask_box_ratio = cfg.DATASETS.OPENPS.MASK_BOX_RATIO
        self.box_range_ratio = cfg.DATASETS.OPENPS.BOX_RANGE_RATIO
        self.pseudo_suppression = cfg.MODEL.EOPSN.PSEUDO_SUPPRESSION
        self.obj_head_pseudo_inst = cfg.MODEL.EOPSN.OBJ_HEAD_PSEUDO_INST

    def _get_unseen_label_set(self, meta, path):
        meta = {e: i for i, e in enumerate(meta)}
        with open(path, 'r') as f:
            lines = [meta[e.replace('\n','')] for e in f.readlines()]

        return lines



    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        original_image  = image

        if self.crop_gen is None or np.random.rand() > 0.5:
            tfm_gens = self.tfm_gens
        else:
            tfm_gens = self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:]


        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            if self.pre_process_unk:
                unk_anno = get_potential_unk(sem_seg_gt, self.box_min_w, self.box_min_h, self.mask_box_ratio, self.box_range_ratio, category_id=self.unk_cate_id)
                if unk_anno is not None:
                    dataset_dict["annotations"] += unk_anno
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(original_image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(tfm_gens)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     return dataset_dict

        if type(transforms[0]) is FT.NoOpTransform:
            flip = 0
        elif type(transforms[0]) is FT.HFlipTransform:
            flip = 1
        else:
            flip = 2
        dataset_dict["flip"] = flip

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.sem_seg_unlabeled_region_on:
                sem_seg_gt[sem_seg_gt==self.ignore_value] = self.num_sem_seg_classes
            dataset_dict["sem_seg"] = sem_seg_gt

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                   anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            if self.pseudo_suppression and unk_anno is not None:
                # split gt and pseudo insts
                dataset_dict["instances_pseudo"] = dataset_dict["instances"][-len(unk_anno):]
                if not self.obj_head_pseudo_inst:
                    dataset_dict["instances"] = dataset_dict["instances"][:-len(unk_anno)]

            if self.unseen_label_set is not None:
                dataset_dict["instances"] = filter_unseen_class(dataset_dict["instances"], self.unseen_label_set)
                if len(dataset_dict["instances"]) == 0:
                    return None

        if self.unlabeled_region_on:
            sem_seg = dataset_dict["sem_seg"].clone()
            if self.sem_seg_filter_unk:
                for idx, inst_cls in enumerate(dataset_dict["instances"].gt_classes):
                    if inst_cls == self.unk_cate_id:
                        _bit_mask = BitMasks.from_polygon_masks(dataset_dict["instances"].gt_masks[idx], *image_shape)
                        sem_seg[_bit_mask.tensor.squeeze()] = 0
            if self.sem_seg_unlabeled_region_on:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.num_sem_seg_classes)
            else:
                cum_sem_seg = cum_map(dataset_dict["sem_seg"], self.ignore_value)
            dataset_dict["integral_sem_seg"] = cum_sem_seg

        return dataset_dict


class DatasetMapperGT:
    """
    dataset mapper for vis gt mask

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.mask_on = cfg.MODEL.MASK_ON
        self.unlabeled_region_on = cfg.MODEL.EOPSN.UNLABELED_REGION
        self.sem_seg_filter_unk = cfg.MODEL.EOPSN.SEM_SEG_FILTER_UNK
        self.pre_process_unk = cfg.MODEL.EOPSN.PRE_PROCESS_UNK
        if self.pre_process_unk:
            # detectron2/detectron2/modeling/roi_heads/roi_heads.py, Line 207
            # Since RPN takes the -1 as an ignored label by default, so we set -2 here
            # also, the last 2nd class represents the unk among the "label_converter"
            self.unk_cate_id = -1

        # Semantic Segmentation
        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        self.sem_seg_unlabeled_region_on = cfg.MODEL.EOPSN.SEM_SEG_UNLABELED_REGION
        self.num_sem_seg_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        unseen_path = cfg.DATASETS.UNSEEN_LABEL_SET
        test_unseen_path = cfg.DATASETS.TEST_UNSEEN_LABEL_SET
        if unseen_path != '':
            meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
            self.unseen_label_set = self._get_unseen_label_set(meta, unseen_path)
            if test_unseen_path != '':
                self.test_unseen_label_set = self._get_unseen_label_set(meta, test_unseen_path)
                self.unseen_label_set += self.test_unseen_label_set
        else:
            self.unseen_label_set = None
        
        

        if cfg.MODEL.LOAD_PROPOSALS:
            self.proposal_topk = (
                                    cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                                    if is_train
                                    else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
                                )
        else:
            self.proposal_topk = None

        self.box_min_w = cfg.DATASETS.OPENPS.BOX_MIN_W
        self.box_min_h = cfg.DATASETS.OPENPS.BOX_MIN_H
        self.mask_box_ratio = cfg.DATASETS.OPENPS.MASK_BOX_RATIO
        self.box_range_ratio = cfg.DATASETS.OPENPS.BOX_RANGE_RATIO

    def _get_unseen_label_set(self, meta, path):
        meta = {e: i for i, e in enumerate(meta)}
        with open(path, 'r') as f:
            lines = [meta[e.replace('\n','')] for e in f.readlines()]

        return lines

    def _detect_test_unseen_inst(self, annos):
        for anno in annos:
            if anno.get("iscrowd", 0) == 0 and anno['category_id'] in self.test_unseen_label_set:
                return True
        return False

    def _filter_unk_inst(self, sem_seg, annos, image_size):
        annos_copy = copy.deepcopy(annos)
        sem_seg_copy = copy.deepcopy(sem_seg)
        for idx, anno in enumerate(annos_copy):
            cat_id = anno["category_id"]
            iscrowd = anno["iscrowd"]
            if self.unseen_label_set is not None and cat_id in self.unseen_label_set and iscrowd == 0:
                segmentation = anno['segmentation']  # list
                # polygons
                polygons = [np.array(p) for p in segmentation]
                masks = polygons_to_bitmask(polygons, *image_size)
                # sem_seg.setflags(write=1)
                sem_seg_copy[masks==1] = 255
                annos.remove(anno)
        return sem_seg_copy, annos

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if self.is_train and self._detect_test_unseen_inst(dataset_dict['annotations']):
            file_name = os.path.basename(dataset_dict['file_name'])
            # print(f'{file_name} is filtered!')
            return None
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        original_image  = image

        if self.crop_gen is None or np.random.rand() > 0.5:
            tfm_gens = self.tfm_gens
        else:
            tfm_gens = self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:]


        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
            # for heuristic PL of training GT
            # unk_anno = get_potential_unk(sem_seg_gt, self.box_min_w, self.box_min_h, self.mask_box_ratio, self.box_range_ratio, category_id=self.unk_cate_id)
            # if unk_anno is not None:
            #     dataset_dict["annotations"] += unk_anno
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        else:
            sem_seg_gt = None

        # aug_input = T.StandardAugInput(original_image, sem_seg=sem_seg_gt)
        # transforms = aug_input.apply_augmentations(tfm_gens)
        # image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        # if type(transforms[0]) is FT.NoOpTransform:
        #     flip = 0
        # elif type(transforms[0]) is FT.HFlipTransform:
        #     flip = 1
        # else:
        #     flip = 2
        dataset_dict["flip"] = 0

        if sem_seg_gt is not None:
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            if self.sem_seg_unlabeled_region_on:
                sem_seg_gt[sem_seg_gt==self.ignore_value] = self.num_sem_seg_classes
            dataset_dict["sem_seg"] = sem_seg_gt

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                   anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     obj
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]
            annos = []
            for obj in dataset_dict["annotations"]:
                if obj.get("iscrowd", 0) == 0:
                    # obj["bbox"] = BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
                    # obj["bbox_mode"] = BoxMode.XYXY_ABS
                    annos.append(obj)
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
            if self.unseen_label_set is not None:
            # for testing
                # for i, c in enumerate(dataset_dict["instances"].gt_classes):
                #     if c in self.unseen_label_set:
                #         dataset_dict["instances"].gt_classes[i] = -1
            # for training
                dataset_dict["instances"] = filter_unseen_class(dataset_dict["instances"], self.unseen_label_set)
            if len(dataset_dict["instances"]) == 0:
                return None
        
        panoptic_r = self._combine_semantic_and_instance_outputs(dataset_dict["instances"], dataset_dict["sem_seg"])
        dataset_dict["panoptic_seg"] = panoptic_r

        return dataset_dict
    
    def _combine_semantic_and_instance_outputs(self, instance_results, semantic_results):
        panoptic_seg = torch.zeros_like(semantic_results, dtype=torch.int32)
        H, W = semantic_results.shape

        current_segment_id = 0
        segments_info = []

        # instance
        masks = instance_results.gt_masks.to(dtype=torch.bool, device=panoptic_seg.device)
        gt_classes = instance_results.gt_classes
        mask_area_list = list()
        bit_mask_list = list()
        for inst_id in range(len(gt_classes)):
            mask = masks[inst_id]  # H,W
            mask = polygons_to_bitmask(mask.polygons[0], H, W) * 1.0
            mask = torch.from_numpy(mask)
            bit_mask_list.append(mask)

            mask_area = mask.sum().item()
            mask_area_list.append(mask_area)
        mask_area_increase_idx = np.array(mask_area_list).argsort()

        for inst_id in mask_area_increase_idx:
        # for inst_id in range(len(gt_classes)):
            # mask = masks[inst_id]  # H,W
            # mask = polygons_to_bitmask(mask.polygons[0], H, W) * 1.0
            # mask = torch.from_numpy(mask)
            mask = bit_mask_list[inst_id]
            mask_area = mask.sum().item()
            if mask_area == 0:
                continue

            intersect = (mask > 0) & (panoptic_seg > 0)
            intersect_area = intersect.sum().item()
            if intersect_area > 0:
                mask = (mask==1) & (panoptic_seg == 0)

            current_segment_id += 1
            panoptic_seg[mask == 1] = current_segment_id
            c = gt_classes[inst_id].item()
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": True,
                    "category_id": c,
                    "instance_id": inst_id,
                }
            )

        # Add semantic results to remaining empty areas
        semantic_labels = torch.unique(semantic_results).cpu().tolist()
        for semantic_label in semantic_labels:
            if semantic_label == 0 or semantic_label == 54:  # 0 is a special "thing" class
                continue
            mask = (semantic_results == semantic_label) & (panoptic_seg == 0)
            mask_area = mask.sum().item()

            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append(
                {
                    "id": current_segment_id,
                    "isthing": False,
                    "category_id": semantic_label,
                    "area": mask_area,
                }
            )
        return panoptic_seg, segments_info