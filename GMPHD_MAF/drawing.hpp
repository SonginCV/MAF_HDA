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

#include "utils.hpp"

// #include <opencv2\opencv.hpp>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include "opencv2\video\tracking.hpp"	// Kalman Filtering
#include <opencv2/imgproc.hpp>			// image transformation


// user-define funcstions for drawing
void InitColorMapTab(cv::Mat& color_map, cv::Scalar* color_tab);

void DrawDetections(cv::Mat& img_det, const std::vector<BBDet>& dets, const cv::Scalar* color_tab, const int& DB_TYPE, const bool& VIS_BB = false);
void DrawTracker(cv::Mat& img_trk, const std::vector<BBTrk>& trks, const std::string& trackerName, const int& DB_TYPE,
	const cv::Scalar *color_tab, int thick = BOUNDING_BOX_THICK, double fontScale = ID_CONFIDENCE_FONT_SIZE - 1);
void DrawTrackerInstances(cv::Mat& img_trk, const std::vector<BBTrk>& trks, const std::string& trackerName, const int& DB_TYPE,
	const cv::Scalar *color_tab, int thick = BOUNDING_BOX_THICK, double fontScale = ID_CONFIDENCE_FONT_SIZE - 1);
void DrawTrackerInstance(cv::Mat& img_trk, const track_info& trk, const std::string& trackerName, const int& DB_TYPE,
	const cv::Scalar *color_tab, const bool& vis_bb_on = false, int thick = BOUNDING_BOX_THICK, double fontScale = ID_CONFIDENCE_FONT_SIZE - 1);

/// Draw the Detection Bounding Box (2D)
void DrawDetBB(cv::Mat& img, int iter, cv::Rect bb, double conf, double conf_th, int digits, cv::Scalar color, int thick = 3);
/// Draw the Tracking Bounding Box (2D)
void DrawTrackBB(cv::Mat& img, const cv::Rect& rec, const cv::Scalar& color, const int& thick, const int& id, const double& fontScale, std::string type, const bool& idBGinBB = false);


// Draw Frame Number and FPS
void DrawFrameNumberAndFPS(int iFrameCnt, cv::Mat& img, double scale, int thick, int frameOffset = 0, int frames_skip_interval = 1, double sec = -1.0);

void cvPrintMat(cv::Mat matrix, std::string name);