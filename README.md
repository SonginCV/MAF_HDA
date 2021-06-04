# GMPHD_MAF: Online Multi-Object Tracking and Segmentation with GMPHD Filter and Mask based Affinity Fusion
This repository includes an implementation of the GMPHD_SAF tracker in C/C++ with a demo code and supplementary materials.

# Progress
We submiited a journal extended from the arXiv preprint.

We are expecting that the first upload of the implementation codes will be done by **_the second week of Jun in 2021_**.

# Paper 

The paper is available in two versions:
[[BMTT2020 website](https://motchallenge.net/workshops/bmtt2020/index.html)]
and
[[arxiv](https://arxiv.org/abs/2009.00100)].

The arXiv preprint is an extension of the BMTT worshop paper.

# Output
#### Demo examples in KITTI object tracking test 0018 sequence.
![public segmentation](GMPHD_MAF/img/KITTI_test-0018_det_256bits.gif)

`▶ Public segmentation results by MaskRCNN`

![public segmentation](GMPHD_MAF/img/KITTI_test-0018_trk_256bits.gif)

`▶ MOTS results by GMPHD_MAF (Ours)`

# Experimental Results (available at the benchmark websites)

We participated "tracking only" track in 5th BMTT MOTChallenge Workshop: Multi-Object Tracking and Segmentation in conjunction with CVPR 2020.

The results are available in 
CVPR 2020 MOTSChallenge [[link](https://motchallenge.net/results/CVPR_2020_MOTS_Challenge/)], 
MOTS20 [[link](https://motchallenge.net/results/MOTS/)],

KITTI-MOTS w/ sMOTSA measure [[link](http://www.cvlibs.net/datasets/kitti/eval_mots.php)] and w/ HOTA measure [[link](http://www.cvlibs.net/datasets/kitti/eval_mots.php)].

# References

[1] Ba-Ngu Vo and Wing-Kin Ma, "The Gaussian Mixture Probability Hypothesis Density Filter," _IEEE Trans. Signal Process._, vol. 54, no. 11, pp. 4091–4104, Oct. 2006. [[paper]](https://ieeexplore.ieee.org/document/1710358)

[2] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick "Mask R-CNN" _in Proc. IEEE Int. Conf. Comput. Vis. (ICCV)_, Oct. 2017, pp. 2961–2969. 
[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)
[[arxiv]](https://arxiv.org/abs/1703.06870)

