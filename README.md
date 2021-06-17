# 1. Introduction
## Online Multi-Object Tracking and Segmentation <br> with GMPHD Filter and Mask-based Affinity Fusion (GMPHD_MAF)
This repository includes an implementation of the GMPHD_MAF tracker in C/C++ with a demo code and supplementary materials.

## Progress

+ **_[2021.06.00] Writing this README.md now..._** <br>
+ [2021.06.17] First full upload of a C/C++ implementation in VS2017 project (with VC15), **_v0.2.0_**. <br>
+ [2021.06.11] Full manuscript upload in **_arXiv_**

## Paper 

+ The paper is available in two versions:
[[BMTT2020 website](https://motchallenge.net/workshops/bmtt2020/index.html)]
and [[arxiv](https://arxiv.org/abs/2009.00100)]. <br>
+ The arXiv preprint is an extension of the BMTT worshop paper.

# 2. User guide

## Development Environment
+ Windows 10  (64 bit) <br>
+ Visual Studio 2017  (64 bit)

#### Programming Languages
+ Visual C/C++ (VC15)

#### Libaries
[OpenCV 3.4.1](https://www.opencv.org/opencv-3-4-1.html) and 
[boost 1.74.0 (Windows Binaries)](https://sourceforge.net/projects/boost/files/boost-binaries/1.74.0/) 
were used to implement the GMPHD_MAF tracker.
> Download [OpenCV Win Pack](https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.4.1/opencv-3.4.1-vc14_vc15.exe/download) and [boost_1_74_0-msvc-14.1-64.exe](https://sourceforge.net/projects/boost/files/boost-binaries/1.74.0/boost_1_74_0-msvc-14.1-64.exe/download) to run our tracker in Visual Studio 2017 (64 bit).

## Project Source
### File Tree
```
PROJECT_HOME
├── GMPHD_MAF.sln  <-- **solution file for the project**
└── GMPHD_MAF      
    ├── demo_GMPHD_MAF.cpp                            <-- **the main function** including demos.
    ├── OnlineTracker.h, OnlineTracker.cpp            <-- the parent class of GMPHD_MAF
    ├── GMPHD_MAF.h, GMPHD_MAF.cpp                    <-- *a class implementation of the GMPHD_MAF tracker* inherited from OnlineTracker class
    ├── kcftracker.hpp&cpp, VOT.h, ffttols.hpp, fhog.hpp&cpp, labdata.hpp, recttols.hpp <-- a C++ code set of KCF [2] implementaion
    ├── utils.hpp                                     <-- a header file including essential containers and preprocessors for GMPHD_MAF
    ├── io_mots.hpp&cpp, mask_api.h&cpp               <-- read/write functions for MOTS data format (e.g., run-length encoding)
    ├── drawing.hpp, drawing.cpp                      <-- drawing functions for MOTS results visualization
    ├── hungarian.h, hungarian.cpp                    <-- a class implementation of the Hungarian Algorithm 
    ├── pch.h                                         <-- precompiled header including essential header files
    ├── GMPHD_MAF.vcxproj, GMPHD_MAF.vcxproj.filters  <-- VS project file, VS project filter file
    ├── params                                        <-- text files containing scene parameters
    |   └── KITTI_test.txt, KITTI_train.txt, MOTS20_test.txt, MOTS20_train.txt
    ├── seq                                           <-- text files containing dataset paths and sequences' names
    |   └── KITTI_test.txt, KITTI_train.txt, MOTS20_test.txt, MOTS20_train.txt
    ├── img                                           <-- MOTS results are saved in {seqname}/*.jpg
    |   ├── KITTI
    |   |   └── test, train                           
    |   └── MOTS20
    |       └── test, train 
    └── res                                           <-- MOTS results are saved in {seqname}.txt
        ├── KITTI
        |   └── test, train 
        └── MOTS20
            └── test, train 
```

C++ implementation of the Hungarian Algorithm : 
`
hungarian.h, hungarian.cpp 
`
, refering to [#mcximing/hungarian-algorithm-cpp](https://github.com/mcximing/hungarian-algorithm-cpp) <br> 

C++ implementaion of KCF [2] :
`
kcftracker.hpp&cpp, VOT.h, ffttols.hpp, fhog.hpp&cpp, labdata.hpp, recttols.hpp
`
, refering to [#joaofaro/KCFcpp](https://github.com/joaofaro/KCFcpp) <br> 

## How to run
1. Open the solution file **GMPHD_MAF.sln**.
2. Link and include **OpenCV3.4.1** and **boost1.74.0** libraries to the project w/ VC15_x64.
3. Press Ctrl+F5 in **Release mode (x64)**

## Input
## Output
## Demo
#### Examples in KITTI object tracking test 0018 sequence.
![public segmentation](GMPHD_MAF/img/KITTI_test-0018_det_256bits.gif)

`▶ Public segmentation results by MaskRCNN [2]`

![public segmentation](GMPHD_MAF/img/KITTI_test-0018_trk_256bits.gif)

`▶ MOTS results by GMPHD_MAF (Ours)`

## Experimental Results (available at the benchmark websites)

We participated "tracking only" track in 5th BMTT MOTChallenge Workshop: Multi-Object Tracking and Segmentation in conjunction with CVPR 2020.

The results are available in 
CVPR 2020 MOTSChallenge [[link](https://motchallenge.net/results/CVPR_2020_MOTS_Challenge/)], 
MOTS20 [[link](https://motchallenge.net/results/MOTS/)],

KITTI-MOTS w/ sMOTSA measure [[link](http://www.cvlibs.net/datasets/kitti/eval_mots.php)] and w/ HOTA measure [[link](http://www.cvlibs.net/datasets/kitti/eval_mots.php)].

# 3. References

### Soure codes
### Papers 

[1] Ba-Ngu Vo and Wing-Kin Ma, "The Gaussian Mixture Probability Hypothesis Density Filter," _IEEE Trans. Signal Process._, vol. 54, no. 11, pp. 4091–4104, Oct. 2006. [[paper]](https://ieeexplore.ieee.org/document/1710358)

[2] João F. Henriques, Rui Caseiro, Pedro Martins, and Jorge Batista "High-Speed Tracking with Kernelized Correlation Filters" _IEEE Trans. Pattern Anal. Mach. Intell._, vol. 37, no. 3, pp. 583–596, Mar. 2015.
[[paper]](https://ieeexplore.ieee.org/abstract/document/6870486)
[[arxiv]](https://arxiv.org/abs/1404.7584)

[3] Kaiming He, Georgia Gkioxari, Piotr Dollar, and Ross Girshick "Mask R-CNN" _in Proc. IEEE Int. Conf. Comput. Vis. (ICCV)_, Oct. 2017, pp. 2961–2969. 
[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)
[[arxiv]](https://arxiv.org/abs/1703.06870)

## Citation [[arxiv]](https://arxiv.org/abs/1907.13347)

```
\bibitem{gmphdmaf}
  Y. Song, Y.-C. Yoon, K. Yoon, M. Jeon, S.-W. Lee, and W. Pedrycz, 
  ``Online Multi-Object Tracking and Segmentation with GMPHD Filter and Mask-based Affinity Fusion,'' 2021, 
  [{O}nline]. Available: ar{X}iv:2009.00100.
```

## 5. [License](https://github.com/SonginCV/GMPHD_MAF/blob/master/LICENSE)
BSD 2-Clause "Simplified" License

