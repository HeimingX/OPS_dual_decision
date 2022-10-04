# Dual Decision Improves Open-Set Panoptic Segmentation            

This project contains the implementation of our work for open-set panoptic segmentation:
    
> Dual Decision Improves Open-Set Panoptic Segmentation ,   
> Hai-Ming Xu, Hao Chen, Lingqiao Liu and Yufei Yin,   
> *To be appeared in BMVC 2022*
    
The full paper is available at: [Arxiv Link](https://arxiv.org/abs/2207.02504)

## News
* [2022-10-04] Repo is created. Code will come soon.

## Abstract

Open-set panoptic segmentation (OPS) problem is a new research direction aiming to perform segmentation for both known classes and unknown classes, i.e., the objects (``things'') that are never annotated in the training set. The main challenges of OPS are twofold: (1) the infinite possibility of the unknown object appearances makes it difficult to model them from a limited number of training data. (2) at training time, we are only provided with the ``void'' category, which essentially mixes the ``unknown thing'' and ``background'' classes. We empirically find that directly using ``void'' category to supervise known class or ``background'' classifiers without screening will lead to an unsatisfied OPS result. In this paper, we propose a divide-and-conquer scheme to develop a dual decision process for OPS. We show that by properly combining a known class discriminator with an additional class-agnostic object prediction head, the OPS performance can be significantly improved. Specifically, we first propose to create a classifier with only known categories and let the ``void'' class proposals achieve low prediction probability from those categories. Then we distinguish the ``unknown things'' from the background by using the additional object prediction head. To further boost performance, we introduce ``unknown things''  pseudo-labels generated from up-to-date models to enrich the training set. Our extensive experimental evaluation shows that our approach significantly improves unknown class panoptic quality, with more than 30\% relative improvements than the existing best-performed method.
