# -*- coding: utf-8 -*-
from detectron2.config import CfgNode as CN


def add_config(cfg):
    """
    Add config for EOPSN
    """
    cfg.MODEL.EOPSN = CN()
    cfg.MODEL.EOPSN.NUM_CENTROID = 0
    cfg.MODEL.EOPSN.CLUSTERING_INTERVAL = 200

    cfg.DATASETS.UNSEEN_LABEL_SET = ''
    cfg.DATASETS.TEST_UNSEEN_LABEL_SET = ''  # labels not appear in train set
    cfg.DATASETS.ONLINE_PROCESS_UNK = False  # process unk area in `sem_seg_gt` during dataset mapper
    cfg.DATASETS.UNK_ORACLE_VERIFY = False  # verification with oracle unk inst info
    cfg.DATASETS.STRONG_AUG = False  # strong aug(randAug), expect to regularize the objHead BUT no improvement shown
    cfg.MODEL.EOPSN.PREDICTOR = "baseline"
    cfg.MODEL.EOPSN.UNLABELED_REGION = False
    cfg.MODEL.EOPSN.IGNORE_UNLABELED_REGION = False
    cfg.MODEL.EOPSN.SEM_SEG_UNLABELED_REGION = False
    cfg.MODEL.EOPSN.VOID_BACKGROUND = False  # void bg flag
    cfg.MODEL.EOPSN.VOID_IGNORANCE = False  # void bg flag
    cfg.MODEL.EOPSN.VOID_SUPPRESSION = False
    cfg.MODEL.EOPSN.PSEUDO_SUPPRESSION = False  # generate heuristic pseudo-inst for void-suppression
    cfg.MODEL.EOPSN.VOID_SUPPRESSION_WEIGHT = 1.0  # void suppression loss weight
    cfg.MODEL.EOPSN.PROBABILISTIC_DET = False  # use probabilistic detection for joint modeling p(C_kn, d|x)=p(C_kn|x, d=0)p(d=0|x)
    cfg.MODEL.EOPSN.VOID_PSEUDO_LABELING = False  # do pseudo-labeling for void proposal on objHead
    cfg.MODEL.EOPSN.VOID_OBJ_CONFIDENCE = 0.95  # confidence threshold

    cfg.MODEL.EOPSN.MIXUP_ALPHA = -1.0  # alpha param of beta distribution for mixup op. alpha=-1 means no mixup
    cfg.MODEL.EOPSN.MIXUP_KNOWN_KNOWN = False  # mixup known & known inst for classification, ie, bg inst is filtered
    cfg.MODEL.EOPSN.MIXUP_KNOWN_UNK = False  # mixup known & unk inst for classification
    cfg.MODEL.EOPSN.MIXUP_UNK_UNK = False  # mixup unk & unk inst for suppression
    cfg.MODEL.EOPSN.MIXUP_UNK_KNOWN = False  # mixup unk & known inst for suppression
    cfg.MODEL.EOPSN.MIXUP_DEFAULT_CLS_LOSS = False  # activate default classification loss when mixup is used
    cfg.MODEL.EOPSN.MIXUP_EXCHANGE = False  # echange channel
    cfg.MODEL.EOPSN.MIXUP_LOSS_TYPE = 'ce'  # mixup loss type

    cfg.MODEL.EOPSN.AUX_OBJ_HEAD = False  # use aux obj head at the second stage
    cfg.MODEL.EOPSN.OBJ_SCORE_THRESHOLD = 0.5  # obj score threshold, default is 0.5
    cfg.MODEL.EOPSN.AUX_OBJ_HEAD_PSEUDO_INST = False  # calc loss for pseudo-inst on aux obj head
    cfg.MODEL.EOPSN.OBJ_HEAD_PSEUDO_INST = False  # calc loss for pseudo-inst on default obj head in RPN stage
    cfg.MODEL.EOPSN.CENTERNESS_HEAD = False  # use objHead do centerness task
    cfg.MODEL.EOPSN.PSEUDO_INST_PSEUDO_LABELING = False  # do pseudo-labeling for pseudo-inst on aux obj head
    cfg.MODEL.EOPSN.PSEUDO_INST_OBJ_CONFIDENCE = 0.5  # confidence threshold for pseudo-inst on aux obj head

    cfg.MODEL.EOPSN.OOD_HEAD = False  # add ood detection head at the second stage
    cfg.MODEL.EOPSN.OOD_TRAIN_TYPE = 'separately'  # traverase three types: separately, semi-joint, joint

    cfg.MODEL.EOPSN.SEM_SEG_FILTER_UNK = False
    cfg.MODEL.EOPSN.PRE_PROCESS_UNK = False

    cfg.MODEL.EOPSN.N_SAMPLE = 20
    cfg.MODEL.EOPSN.NMS_THRESH = 0.3
    cfg.MODEL.EOPSN.CLUSTER_OBJ_THRESH = 0.9
    cfg.MODEL.EOPSN.COS_THRESH = 0.15
    cfg.MODEL.EOPSN.COUPLED_COS_THRESH = 0.15
    cfg.MODEL.EOPSN.COUPLED_OBJ_THRESH = 0.9


    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.DATASETS.OPENPS = CN()
    cfg.DATASETS.OPENPS.BOX_MIN_W = 40
    cfg.DATASETS.OPENPS.BOX_MIN_H = 40
    cfg.DATASETS.OPENPS.MASK_BOX_RATIO = 0.8
    cfg.DATASETS.OPENPS.BOX_RANGE_RATIO = 0.9

    # one-vs-all classifier, does not bring improvement may because of void-supp is already an OOD method: outlier exposure
    cfg.MODEL.OVA = CN()
    cfg.MODEL.OVA.ACTIVATE = False
    cfg.MODEL.OVA.OVA_ENT_WEIGHT = 0.0
    cfg.MODEL.OVA.VOID_SUPPRESSION = False  # void suppression on ova head

    cfg.MODEL.MAHALANOBIS_OOD_DETECTOR = CN()
    cfg.MODEL.MAHALANOBIS_OOD_DETECTOR.ACTIVATE = False

    cfg.MODEL.ADVERSARIAL_OBJHEAD = CN()
    cfg.MODEL.ADVERSARIAL_OBJHEAD.ACTIVATE = False
    cfg.MODEL.ADVERSARIAL_OBJHEAD.DETACH_FEAT = False  # whether detach original roi_feature or note
    cfg.MODEL.ADVERSARIAL_OBJHEAD.ADV_EPS = 10.0  #  gradient update step size
    cfg.MODEL.ADVERSARIAL_OBJHEAD.KNOWN_INST_ONLY = False  # adv op performed on known-inst only

    cfg.MODEL.INFORMATION_BOTTLENECK = CN()
    cfg.MODEL.INFORMATION_BOTTLENECK.ACTIVATE = False
    cfg.MODEL.INFORMATION_BOTTLENECK.COMPRESSION_RATE = 0.5    # default out dim is 1024
    cfg.MODEL.INFORMATION_BOTTLENECK.DETACH_FEAT = False  # if True, objHead's loss will not backpropagate into bkbn
    cfg.MODEL.INFORMATION_BOTTLENECK.SINGLE_LAYER = False  # only has single layer
