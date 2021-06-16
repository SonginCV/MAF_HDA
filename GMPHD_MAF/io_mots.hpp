/*
BSD 2-Clause License

Copyright (c) 2021, Young-min Song,
Machine Learning and Vision Lab (https://sites.google.com/view/mlv/),
Gwangju Institute of Science and Technology(GIST), South Korea.
All rights reserved.

This software is an implementation of the GMPHD_MAF tracker,
which not only refers to the paper entitled
"Online Multi-Object Tracking and Segmentation with GMPHD Filter and Mask-Based Affinity Fusion"
but also is available at https://github.com/SonginCV/GMPHD_MAF.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/algorithm/string.hpp>

#include <io.h>

#include "utils.hpp"
#include "mask_api.h"

#include <ppl.h>

using namespace std;
using namespace boost::filesystem;

// User-defined functions for I/O,..

// Read
void ReadDatasetInfo(const int& DB_TYPE, const string& MODE, const string& detNAME, const string& seqFile, const string& paramsFile,
	vector<string>& seqNames, vector<string>& seqPaths, vector<string>& detTxts, vector<string>& trkTxtsGT, vector<MOTparams>& params_out);
vector<string> ReadFilesInPath(boost::filesystem::path p);
VECx2xBBDet ReadDetectionsSeq(const int& DB_TYPE, const string& detNAME, const string& detTxt,
	VECx2xBBDet& carDets, VECx2xBBDet& personDets);
VECx2xBBTrk ReadTracksSeq(const int& DB_TYPE, const string& trkNAME, const string& trkTxt,
	VECx2xBBTrk& carTrks, VECx2xBBTrk& persoTrks, cv::Mat& carHeatMap, cv::Mat& perHeatMap);
// The Function for Sorting Detection Responses by frame number (ascending order)
vector<string> SortAllDetections(const vector<string>& allLines, int DB_TYPE = DB_TYPE_MOT17);

// Write
void SaveResultImgs(const int& DB_TYPE, const string& MODE, const string& detNAME, const string& seqNAME,
	const int& iFrmCnt, const cv::Mat& img, const float& ths_det = 0.0, const string& tag = "");

// Convert, divide, interpolate
// convert
std::string CvtMAT2RleSTR(const cv::Mat& in_maskMAT, const cv::Size& in_frmImgSz, const cv::Rect& bbox, const bool& viewDetail = false);
int CvtRleSTR2MATVecSeq(VECx2xBBDet& in_dets, VECx2xBBDet& out_dets, const cv::Size& frm_sz, const float& DET_SCORE_TH = 0.0);
void CvtRleSTR2MAT(const std::string &in_maskRleSTR, const cv::Size& in_segImgSz, cv::Mat& out_maskMAT, cv::Rect& out_objRec);
cv::Rect CvtMAT2RECT(const cv::Size& in_segImgSz, const cv::Mat& in_maskMAT);