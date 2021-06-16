/*

Tracker based on Kernelized Correlation Filter (KCF) [1] and Circulant Structure with Kernels (CSK) [2].
CSK is implemented by using raw gray level features, since it is a single-channel filter.
KCF is implemented by using HOG features (the default), since it extends CSK to multiple channels.

[1] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"High-Speed Tracking with Kernelized Correlation Filters", TPAMI 2015.

[2] J. F. Henriques, R. Caseiro, P. Martins, J. Batista,
"Exploiting the Circulant Structure of Tracking-by-detection with Kernels", ECCV 2012.

Authors: Joao Faro, Christian Bailer, Joao F. Henriques
Contacts: joaopfaro@gmail.com, Christian.Bailer@dfki.de, henriques@isr.uc.pt
Institute of Systems and Robotics - University of Coimbra / Department Augmented Vision DFKI


Constructor parameters, all boolean:
    hog: use HOG features (default), otherwise use raw pixels
    fixed_window: fix window size (default), otherwise use ROI size (slower but more accurate)
    multiscale: use multi-scale tracking (default; cannot be used with fixed_window = true)

Default values are set for all properties of the tracker depending on the above choices.
Their values can be customized further before calling init():
    interp_factor: linear interpolation factor for adaptation
    sigma: gaussian kernel bandwidth
    lambda: regularization
    cell_size: HOG cell size
    padding: area surrounding the target, relative to its size
    output_sigma_factor: bandwidth of gaussian target
    template_size: template size in pixels, 0 to use ROI size
    scale_step: scale step for multi-scale estimation, 1 to disable it
    scale_weight: to downweight detection scores of other scales for added stability

For speed, the value (template_size/cell_size) should be a power of 2 or a product of small prime numbers.

Inputs to init():
   image is the initial frame.
   roi is a cv::Rect with the target positions in the initial frame

Inputs to update():
   image is the current frame.

Outputs of update():
   cv::Rect with target positions for the current frame


By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */
#pragma once
#include "pch.h"

#ifndef _KCFTRACKER_HEADERS
#include "kcftracker.hpp"
#include "ffttools.hpp"
#include "recttools.hpp"
#include "fhog.hpp"
#include "labdata.hpp"
#endif

// Constructor
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab, bool roi_only)
{

    // Parameters equal in all cases
    lambda = 0.0001;
    padding = 2.5; 
    //output_sigma_factor = 0.1;
    output_sigma_factor = 0.125;


    if (hog) {    // HOG
        // VOT
        interp_factor = 0.012;
        sigma = 0.6; 
        // TPAMI
        //interp_factor = 0.02;
        //sigma = 0.5; 
        cell_size = 4;
        _hogfeatures = true;

        if (lab) {
            interp_factor = 0.005;
            sigma = 0.4; 
            //output_sigma_factor = 0.025;
            output_sigma_factor = 0.1;

            _labfeatures = true;
            _labCentroids = cv::Mat(labdata::nClusters, 3, CV_32FC1, &labdata::data);
            cell_sizeQ = cell_size*cell_size;
        }
        else{
            _labfeatures = false;
        }
    }
    else {   // RAW
        interp_factor = 0.075;
        sigma = 0.2; 
        cell_size = 1;
        _hogfeatures = false;

        if (lab) {
            printf("Lab features are only used with HOG features.\n");
            _labfeatures = false;
        }
    }


    if (multiscale) { // multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1.05;
        scale_weight = 0.95;
        if (!fixed_window) {
            //printf("Multiscale does not support non-fixed window.\n");
            fixed_window = true;
        }
    }
    else if (fixed_window) {  // fit correction without multiscale
        template_size = 96;
        //template_size = 100;
        scale_step = 1;
    }
    else {						// Caculate features and similarity extracted from ROI
        template_size = 1;
        scale_step = 1;
    }

	if (roi_only) {
		template_size = 1;
		scale_step = 1;
	}

	_scale_MOT = 1.0;
	this->InitializeColorMap();
}

// Initialize tracker 
void KCFTracker::init(const cv::Mat& image, const cv::Rect &roi, const cv::Mat &mask, const bool& USE_MASK)
{
    _roi = roi;
    assert(roi.width >= 0 && roi.height >= 0);

	this->fWidth = image.cols;
	this->fHeight = image.rows;

	if (USE_MASK) {
		cv::Mat masked_image = image.clone();
		cv::Mat bin_mask;

		cv::Rect _roi_int((int)_roi.x, (int)_roi.y, (int)_roi.width, (int)_roi.height);
		cv::Rect _roi_int_frm = this->RectExceptionHandling(this->fWidth, this->fHeight, _roi_int);
		//printf("i(%dx%d)", _roi_int_frm.width, _roi_int_frm.height);
		//printf("imask(%dx%d)", mask.cols, mask.rows);
		cv::resize(mask, bin_mask, cv::Size(_roi_int_frm.width, _roi_int_frm.height));

		std::vector<cv::Point2i> bg_pts, fg_pts, ag_pts;
		cv::Vec3d bg_sum_(0.0, 0.0, 0.0), fg_sum_(0.0, 0.0, 0.0), ag_sum_(0.0, 0.0, 0.0);

		for (int r = _roi_int_frm.y; r < _roi_int_frm.y + _roi_int_frm.height; ++r) {
			for (int c = _roi_int_frm.x; c < _roi_int_frm.x + _roi_int_frm.width; ++c) {

				cv::Vec3b bgr = masked_image.at<cv::Vec3b>(r, c);
				if ((int)bin_mask.at<uchar>(r - _roi_int_frm.y, c - _roi_int_frm.x) == 0) {

					// masked_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
					// or 배경의 영향력을 줄이기 위해 average 된 rgb value 를 넣어 보는것은 어떤가?

					bg_pts.push_back(cv::Point2i(c, r));
					bg_sum_ += (cv::Vec3d)bgr;
				}
				else {
					fg_pts.push_back(cv::Point2i(c, r));
					fg_sum_ += (cv::Vec3d)bgr;
				}
				ag_pts.push_back(cv::Point2i(c, r));
				ag_sum_ += (cv::Vec3d)bgr;
			}
		}
		// sum -> average
		bg_sum_[0] /= bg_pts.size(); bg_sum_[1] /= bg_pts.size(); bg_sum_[2] /= bg_pts.size();
		fg_sum_[0] /= fg_pts.size(); fg_sum_[1] /= fg_pts.size(); fg_sum_[2] /= fg_pts.size();
		ag_sum_[0] /= ag_pts.size(); ag_sum_[1] /= ag_pts.size(); ag_sum_[2] /= ag_pts.size();

		for (const auto& p : bg_pts) {
			masked_image.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(0, 0, 0);
			//masked_image.at<cv::Vec3b>(p.y, p.x) = (cv::Vec3b)bg_sum_;
			//masked_image.at<cv::Vec3b>(p.y, p.x) = (cv::Vec3b)fg_sum_;
			//masked_image.at<cv::Vec3b>(p.y, p.x) = (cv::Vec3b)ag_sum_;
		}

		_tmpl = getFeatures(masked_image, 1);
		masked_image.release();
	}
	else {
		_tmpl = getFeatures(image, 1);
	}
	_prob = createGaussianPeak(size_patch[0], size_patch[1]);
	_alphaf = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_num = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    //_den = cv::Mat(size_patch[0], size_patch[1], CV_32FC2, float(0));
    train(_tmpl, 1.0); // train with initial frame

 }

// Update position based on the new frame
cv::Rect KCFTracker::update(cv::Mat& image)
{
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

    float cx = _roi.x + _roi.width / 2.0f;
    float cy = _roi.y + _roi.height / 2.0f;


    float peak_value;
	cv::Mat res_mat;
    cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value,res_mat);
    if (scale_step != 1) { 
        // Test at a smaller _scale
        float new_peak_value;
        cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);

        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale /= scale_step;
            _roi.width /= scale_step;
            _roi.height /= scale_step;
        }
        // Test at a bigger _scale
        new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);
        if (scale_weight * new_peak_value > peak_value) {
            res = new_res;
            peak_value = new_peak_value;
            _scale *= scale_step;
            _roi.width *= scale_step;
            _roi.height *= scale_step;
        }
    }
    // Adjust by cell size and _scale
    _roi.x = cx - _roi.width / 2.0f + ((float) res.x * cell_size * _scale);
    _roi.y = cy - _roi.height / 2.0f + ((float) res.y * cell_size * _scale);

    if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
    if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
    if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
    if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

    assert(_roi.width >= 0 && _roi.height >= 0);
    cv::Mat x = getFeatures(image, 0);

	// printf("image:%d by %d (type:%d, nc:%d)\n", image.rows, image.cols, image.type(), image.channels());
	
	// printf("res_mat:%d by %d (type:%d, nc:%d)\n", res_mat.rows, res_mat.cols, res_mat.type(), res_mat.channels());
	//cvPrintMat(res_mat, "res_mat");
	cv::Mat res_color(cv::Size(res_mat.cols, res_mat.rows), CV_8UC3);
	cv::Mat res_gray; 
	
	//cvPrintMat(res_mat, "res_mat");
	for (int r = 0; r < res_mat.rows; r++) {
		for (int c = 0; c < res_mat.cols; c++) {
			res_mat.at<float>(r, c) += 1.0;
			res_mat.at<float>(r, c) /= 2.0;
		}
	}
	res_mat.convertTo(res_gray, CV_8UC1, 255.0 );
	/*cv::threshold(res_mat, res_mat, 0.0, 2.0, cv::THRESH_TOZERO);
	res_mat.convertTo(res_gray, CV_8UC1, 255.0);*/
	//cvPrintMat(res_mat, "res_mat_threshold");
	cv::normalize(res_gray, res_gray, 255, 0, cv::NORM_MINMAX);
	for (int r = 0; r < res_gray.rows; r++) {
		for (int c = 0; c < res_gray.cols; c++) {
			res_color.at<cv::Vec3b>(r, c) = this->color_map.at<cv::Vec3b>(0, res_gray.at<uchar>(r,c));
		}
	}
	cv::Mat res_tmpl;// (cv::Size(res_color.cols * 10, res_color.rows * 10), CV_8UC3);
	cv::Rect _roi_int((int)_roi.x, (int)_roi.y, (int)_roi.width, (int)_roi.height);
	_roi_int = this->RectExceptionHandling(this->fWidth, this->fHeight, _roi_int);
	if (_roi_int.width >= 1 && _roi_int.height >= 1) {
		
		cv::resize(res_color, res_tmpl, cv::Size(_roi_int.width, _roi_int.height));

		train(x, interp_factor);
	}
	
	/*printf("    [VOT] %d by %d features (type:%d, nc:%d)\n",x.rows, x.cols, x.type(), x.channels());
	printf("    [VOT] roi (%.2f,%.2f,%.2f,%.2f)\n", _roi.x, _roi.y, _roi.width, _roi.height);
	printf("    [VOT] roi (%d,%d,%d,%d)\n", _roi_int.x, _roi_int.y, _roi_int.width, _roi_int.height);*/
	
	//cv::imshow("color map", color_map);
	//cv::imshow("feature map", res_tmpl);

	//cv::Mat imgTmp = image.clone();

	/*cv::addWeighted(imgTmp(_roi_int), 0.5, res_tmpl, 0.5, 0.0, imgTmp(_roi_int));
	cv::imshow("image with feature map", imgTmp);
	cv::waitKey(1);*/

	if (!res_mat.empty()) res_mat.release();
	//if (!imgTmp.empty()) imgTmp.release();

    return _roi_int;
}
cv::Rect KCFTracker::update(cv::Mat& image, float& confProb)
{
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 2;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 2;

	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;


	float peak_value;
	cv::Mat res_mat;
	cv::Point2f res = detect(_tmpl, getFeatures(image, 0, 1.0f), peak_value, res_mat);
	if (scale_step != 1) {
		// Test at a smaller _scale
		float new_peak_value;
		cv::Point2f new_res = detect(_tmpl, getFeatures(image, 0, 1.0f / scale_step), new_peak_value);

		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale /= scale_step;
			_roi.width /= scale_step;
			_roi.height /= scale_step;
		}
		// Test at a bigger _scale
		new_res = detect(_tmpl, getFeatures(image, 0, scale_step), new_peak_value);
		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale *= scale_step;
			_roi.width *= scale_step;
			_roi.height *= scale_step;
		}
	}
	// Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	cv::Mat res_color(cv::Size(res_mat.cols, res_mat.rows), CV_8UC3);
	cv::Mat res_gray;

	// Re-scaling from -1.0~1.0 to 0.0~1.0
	for (int r = 0; r < res_mat.rows; r++) {
		for (int c = 0; c < res_mat.cols; c++) {
			res_mat.at<float>(r, c) += 1.0;
			res_mat.at<float>(r, c) /= 2.0;
		}
	}

	res_mat.convertTo(res_gray, CV_8UC1, 255.0);
	cv::normalize(res_gray, res_gray, 255, 0, cv::NORM_MINMAX);

	float sumResNorm = 0, sumResNormPerSize;
	for (int r = 0; r < res_gray.rows; r++) {
		for (int c = 0; c < res_gray.cols; c++) {
			res_color.at<cv::Vec3b>(r, c) = this->color_map.at<cv::Vec3b>(0, res_gray.at<uchar>(r, c));
			sumResNorm += ((float)(res_gray.at<uchar>(r, c)) / (float)255.0);
		}
	}
	int area = res_gray.rows*res_gray.cols;
	if (area>0)	sumResNormPerSize = sumResNorm / area;
	else		sumResNormPerSize = 0.99;

	confProb = sumResNormPerSize;

	assert(_roi.width >= 0 && _roi.height >= 0);
	cv::Mat x = getFeatures(image, 0);

	train(x, interp_factor);

	return _roi;
}
cv::Rect KCFTracker::update(const cv::Mat& image, float& confProb, 
	cv::Mat &confMapVis, const cv::Rect &roi, const bool& GET_VIS,
	const cv::Mat &mask, const bool& USE_MASK) {

	if (roi.width > 0 && roi.height > 0) {
		_roi = (cv::Rect2f)roi;
	}
	//assert(roi.width >= 0 && roi.height >= 0);
	cv::Mat proc_image;
	//printf("(%d:%d,%d)\n    [MOT] init_roi (%d,%d,%d,%d)", image.empty(), image.cols, image.rows
	//	, roi.x, roi.y, roi.width, roi.height);

	if (_roi.x + _roi.width <= 0)	_roi.x = -_roi.width + 1;
	if (_roi.y + _roi.height <= 0)	_roi.y = -_roi.height + 1;
	if (_roi.x >= image.cols - 1)	 _roi.x = proc_image.cols - 2;
	if (_roi.y >= image.rows - 1)	_roi.y = proc_image.rows - 2;

	//printf("->(%.f,%.f,%.f,%.f)\n", _roi.x, _roi.y, _roi.width, _roi.height);

	float cx = _roi.x + _roi.width / 2.0f;
	float cy = _roi.y + _roi.height / 2.0f;

	cv::Rect _roi_mask;
	cv::Mat bin_mask;
	if (USE_MASK) {
		cv::Mat masked_image = image.clone();
		cv::Mat bin_mask_resize;
		cv::Mat bin_mask_crop;

		cv::Rect _size_frame(0,0, image.cols, image.rows);
		cv::Rect _roi_int((int)_roi.x, (int)_roi.y, (int)_roi.width, (int)_roi.height);
		
		// rect in image frame coordinate
		cv::Rect _roi_inter = (_size_frame & _roi_int);
		// rect in mark coordinate
		cv::Rect _roi_inter_on_mask(_roi_inter.x - _roi_inter.x, _roi_inter.y - _roi_inter.y, _roi_inter.width, _roi_inter.height);
		//printf("		(%d,%d)~roi_mask2(%dx%d)", _roi_inter.x, _roi_inter.y, _roi_inter_on_mask.width, _roi_inter_on_mask.height);
		cv::resize(mask(_roi_inter_on_mask), bin_mask_crop, cv::Size(_roi_inter_on_mask.width, _roi_inter_on_mask.height));
		
		//cv::Rect _roi_int_frm = this->RectExceptionHandling(this->fWidth, this->fHeight, _roi_int);
		//printf("/roi_mask1(%dx%d)", _roi_int_frm.width, _roi_int_frm.height);
		//cv::resize(mask, bin_mask_resize, cv::Size(_roi_int_frm.width, _roi_int_frm.height));

		bin_mask = bin_mask_crop; // bin_mask_crop, bin_mask_resize
		_roi_mask = _roi_inter; // _roi_inter, _roi_int_frm
		/*if (_roi_int_frm.width==15 && _roi_int_frm.height==122 && mask.cols == 63 && mask.rows == 122) {
			cv::imshow("mask",mask);
			cv::imshow("mask_resize", bin_mask_resize);
			cv::imshow("mask_crop", bin_mask_crop);
			cv::waitKey();
		}*/
		//printf("-org_mask(%dx%d)\n", mask.cols, mask.rows);

		std::vector<cv::Point2i> bg_pts, fg_pts, ag_pts;
		cv::Vec3d bg_sum_(0.0,0.0,0.0), fg_sum_(0.0, 0.0, 0.0), ag_sum_(0.0, 0.0, 0.0);

		for (int r = _roi_mask.y; r < _roi_mask.y + _roi_mask.height; ++r) {
			for (int c = _roi_mask.x; c < _roi_mask.x + _roi_mask.width; ++c) {
				
				cv::Vec3b bgr = masked_image.at<cv::Vec3b>(r, c);
				if ((int)bin_mask.at<uchar>(r - _roi_mask.y, c - _roi_mask.x) == 0) {
					
					// masked_image.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
					// or 배경의 영향력을 줄이기 위해 average 된 rgb value 를 넣어 보는것은 어떤가?
					// -> 그냥 검정색그대로 하는게 제일 좋았음

					bg_pts.push_back(cv::Point2i(c, r)); 
					bg_sum_ += (cv::Vec3d)bgr;
				}
				else {
					fg_pts.push_back(cv::Point2i(c, r));
					fg_sum_ += (cv::Vec3d)bgr;
				}
				ag_pts.push_back(cv::Point2i(c, r));
				ag_sum_ += (cv::Vec3d)bgr;
			}
		}
		// sum -> average
		bg_sum_[0] /= bg_pts.size(); bg_sum_[1] /= bg_pts.size(); bg_sum_[2] /= bg_pts.size();
		fg_sum_[0] /= fg_pts.size(); fg_sum_[1] /= fg_pts.size(); fg_sum_[2] /= fg_pts.size();
		ag_sum_[0] /= ag_pts.size(); ag_sum_[1] /= ag_pts.size(); ag_sum_[2] /= ag_pts.size();

		if (false) {
			cv::Vec3b fg_avg_inv;
			if (fg_sum_[0] >= 128)	fg_avg_inv[0] = fg_sum_[0] - 128;
			else					fg_avg_inv[0] = fg_sum_[0] + 128;
			if (fg_sum_[1] >= 128)	fg_avg_inv[1] = fg_sum_[1] - 128;
			else					fg_avg_inv[1] = fg_sum_[1] + 128;
			if (fg_sum_[2] >= 128)	fg_avg_inv[2] = fg_sum_[2] - 128;
			else					fg_avg_inv[2] = fg_sum_[2] + 128;
		}

		for (const auto& p : bg_pts) {
			masked_image.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(0, 0, 0);
			//masked_image.at<cv::Vec3b>(p.y, p.x) = (cv::Vec3b)bg_sum_;
			//masked_image.at<cv::Vec3b>(p.y, p.x) = (cv::Vec3b)fg_sum_;
			//masked_image.at<cv::Vec3b>(p.y, p.x) = (cv::Vec3b)ag_sum_;
			//masked_image.at<cv::Vec3b>(p.y, p.x) = cv::Vec3b(255, 255, 255);
			//masked_image.at<cv::Vec3b>(p.y, p.x) = fg_avg_inv;
		}

		proc_image = masked_image;

		// Considering Mask based ROI and Excluding region-out of Frame
		// because of Error in Person Tracking in train0004-frame230
		// Velocity Update Alpha = 0.1f
		_roi = _roi_mask;
		cx = _roi.x + _roi.width / 2.0f;
		cy = _roi.y + _roi.height / 2.0f;


	}
	else {
		proc_image = image;
	}


	float peak_value;
	cv::Mat res_mat;
	//printf("(%d:%d,%d)", image.empty(),image.cols,image.rows);
	//printf("-(%d:%d,%d)", _tmpl.empty(), _tmpl.cols, _tmpl.rows);
	cv::Point2f res = detect(_tmpl, getFeatures(proc_image, 0, 1.0f), peak_value, res_mat);

	//std::cout << "\n";

	// Calculate KCF detection in one smaller and one bigger scales (two more).
	if (scale_step != 1) {
		// Test at a smaller _scale
		float new_peak_value;
		cv::Point2f new_res = detect(_tmpl, getFeatures(proc_image, 0, 1.0f / scale_step), new_peak_value);

		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale /= scale_step;
			_roi.width /= scale_step;
			_roi.height /= scale_step;
		}

		// Test at a bigger _scale
		new_res = detect(_tmpl, getFeatures(proc_image, 0, scale_step), new_peak_value);

		if (scale_weight * new_peak_value > peak_value) {
			res = new_res;
			peak_value = new_peak_value;
			_scale *= scale_step;
			_roi.width *= scale_step;
			_roi.height *= scale_step;
		}
	}
	//cout << "3";
	// Adjust by cell size and _scale
	_roi.x = cx - _roi.width / 2.0f + ((float)res.x * cell_size * _scale);
	_roi.y = cy - _roi.height / 2.0f + ((float)res.y * cell_size * _scale);

	if (_roi.x >= image.cols - 1) _roi.x = image.cols - 1;
	if (_roi.y >= image.rows - 1) _roi.y = image.rows - 1;
	if (_roi.x + _roi.width <= 0) _roi.x = -_roi.width + 2;
	if (_roi.y + _roi.height <= 0) _roi.y = -_roi.height + 2;

	assert(_roi.width >= 0 && _roi.height >= 0);
	cv::Mat x = getFeatures(proc_image, 0);
	train(x, interp_factor);

	cv::Rect _roi_int((int)_roi.x, (int)_roi.y, (int)_roi.width, (int)_roi.height);
	cv::Rect _roi_int_frm = this->RectExceptionHandling(this->fWidth, this->fHeight, _roi_int);

	//printf("    [MOT] %d by %d features (type:%d, nc:%d)\n", x.rows, x.cols, x.type(), x.channels());
	//printf("    [MOT] roi (%.2f,%.2f,%.2f,%.2f)\n", _roi.x, _roi.y, _roi.width, _roi.height);
	//printf("    [MOT] res_roi (%d,%d,%d,%d)\n", _roi_int.x, _roi_int.y, _roi_int.width, _roi_int.height);
	//printf("    [MOT] res_roi_frm (%d,%d,%d,%d)\n", _roi_int_frm.x, _roi_int_frm.y, _roi_int_frm.width, _roi_int_frm.height);
	//printf("    [MOT] track prob values' sum %f (%f)\n", sumResNorm, sumResNormPerSize);


	if (GET_VIS) {

		if (_roi_int_frm.width > 0 && _roi_int_frm.height > 0) {
			cv::Mat res_color(cv::Size(res_mat.cols, res_mat.rows), CV_8UC3);
			cv::Mat res_gray;

			//cvPrintMat(res_mat, "res_mat");

			// Re-scaling from -1.0~1.0 to 0.0~1.0
			//float sumRes = 0;
			for (int r = 0; r < res_mat.rows; r++) {
				for (int c = 0; c < res_mat.cols; c++) {
					res_mat.at<float>(r, c) += 1.0;
					res_mat.at<float>(r, c) /= 2.0;

					//sumRes += res_mat.at<float>(r, c);
				}
			}

			//cv::threshold(res_mat, res_mat, 0.0, 2.0, cv::THRESH_TOZERO);
			//std::cout << "5";
			res_mat.convertTo(res_gray, CV_8UC1, 255.0);
			//cvPrintMat(res_gray, "res_gray");
			cv::normalize(res_gray, res_gray, 255, 0, cv::NORM_MINMAX);
			//cvPrintMat(res_gray, "res_gray_norm");

			// Updated roi results
			cv::Mat res_vis;	// (cv::Size(res_color.cols * 10, res_color.rows * 10), CV_8UC3);
			float sumResNorm = 0, sumResNormPerSize;

			if (!USE_MASK) {
				const int area = res_gray.rows*res_gray.cols;

				for (int r = 0; r < res_gray.rows; r++) {
					for (int c = 0; c < res_gray.cols; c++) {
						res_color.at<cv::Vec3b>(r, c) = this->color_map.at<cv::Vec3b>(0, res_gray.at<uchar>(r, c));
						sumResNorm += ((float)(res_gray.at<uchar>(r, c)) / (float)255.0);
					}
				}
				if (area > 0)	sumResNormPerSize = sumResNorm / area;
				else			sumResNormPerSize = 0.99;
			}
			else {
				cv::Mat _bin_mask, res_gray_mask;
				cv::resize(res_gray, res_gray_mask, cv::Size(_roi_mask.width, _roi_mask.height));
				cv::resize(mask, _bin_mask, cv::Size(_roi_mask.width, _roi_mask.height));

				/*cv::imshow("mask", mask);
				cv::imshow("bin_mask", bin_mask);
				cv::waitKey();*/

				// re-normalized in mask area
				cv::normalize(res_gray_mask, res_gray_mask, 255, 0, cv::NORM_MINMAX, -1, bin_mask);

				int pixels_area = 0;
				cv::Mat res_color_mask(cv::Size(res_gray_mask.cols, res_gray_mask.rows), CV_8UC3);

				for (int r = 0; r < res_gray_mask.rows; r++) {
					for (int c = 0; c < res_gray_mask.cols; c++) {
						if (_bin_mask.at<uchar>(r, c) > 0) {
							res_color_mask.at<cv::Vec3b>(r, c) = this->color_map.at<cv::Vec3b>(0, res_gray_mask.at<uchar>(r, c));
							sumResNorm += ((float)(res_gray_mask.at<uchar>(r, c)) / (float)255.0);
							++pixels_area;
						}
						else {
							res_color_mask.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
						}
					}
				}
				if (pixels_area > 0)	sumResNormPerSize = sumResNorm / pixels_area;
				else					sumResNormPerSize = 0.99;

				res_color.release();
				res_color = res_color_mask;

				if (!res_gray_mask.empty())res_gray_mask.release();
				if (!_bin_mask.empty())_bin_mask.release();
			}

			
			cv::resize(res_color, res_vis, cv::Size(_roi_int_frm.width, _roi_int_frm.height));
			confMapVis = res_vis.clone();
			confProb = sumResNormPerSize;

			train(x, interp_factor);

			if (!res_mat.empty()) res_mat.release();
			if (!res_vis.empty()) res_vis.release();
		}
		else { // out of frame
			confMapVis = cv::Mat(roi.height, roi.width, CV_8UC3, cv::Scalar(0, 0, 0));
			confProb = 0.99;
		}	
	}

	if (USE_MASK) {
		proc_image.release();
		bin_mask.release();
	}

	return _roi_int_frm;
}

// Detect object in the current frame.
cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, z);
    cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));

    //minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
    cv::Point2i pi;
    double pv;
    cv::minMaxLoc(res, NULL, &pv, NULL, &pi);
    peak_value = (float) pv;

    //subpixel peak estimation, coordinates will be non-integer
    cv::Point2f p((float)pi.x, (float)pi.y);

    if (pi.x > 0 && pi.x < res.cols-1) {
        p.x += subPixelPeak(res.at<float>(pi.y, pi.x-1), peak_value, res.at<float>(pi.y, pi.x+1));
    }

    if (pi.y > 0 && pi.y < res.rows-1) {
        p.y += subPixelPeak(res.at<float>(pi.y-1, pi.x), peak_value, res.at<float>(pi.y+1, pi.x));
    }

    p.x -= (res.cols) / 2;
    p.y -= (res.rows) / 2;

    return p;
}
cv::Point2f KCFTracker::detect(cv::Mat k1, cv::Mat k2, float &peak_value, cv::Mat& res_output) {
	using namespace FFTTools;
	cv::Mat k = gaussianCorrelation(k2, k1);
	//printf("k:%d by %d (t:%d, nc:%d)\n",k.rows, k.cols, k.type(), k.channels());
	cv::Mat res = (real(fftd(complexMultiplication(_alphaf, fftd(k)), true)));
	//minMaxLoc only accepts doubles for the peak, and integer points for the coordinates
	cv::Point2i pi,mini;
	double pv,minv;
	cv::minMaxLoc(res, &minv, &pv, &mini, &pi);
	peak_value = (float)pv;

	//subpixel peak estimation, coordinates will be non-integer
	cv::Point2f p((float)pi.x, (float)pi.y);
	if (pi.x > 0 && pi.x < res.cols - 1) {
		p.x += subPixelPeak(res.at<float>(pi.y, pi.x - 1), peak_value, res.at<float>(pi.y, pi.x + 1));
	}

	if (pi.y > 0 && pi.y < res.rows - 1) {
		p.y += subPixelPeak(res.at<float>(pi.y - 1, pi.x), peak_value, res.at<float>(pi.y + 1, pi.x));
	}
	p.x -= (res.cols) / 2;
	p.y -= (res.rows) / 2;

	res_output = res.clone();

	return p;
}
// train tracker with a single image
void KCFTracker::train(cv::Mat x, float train_interp_factor)
{
    using namespace FFTTools;

    cv::Mat k = gaussianCorrelation(x, x);
    cv::Mat alphaf = complexDivision(_prob, (fftd(k) + lambda));
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _alphaf = (1 - train_interp_factor) * _alphaf + (train_interp_factor) * alphaf;


    /*cv::Mat kf = fftd(gaussianCorrelation(x, x));
    cv::Mat num = complexMultiplication(kf, _prob);
    cv::Mat den = complexMultiplication(kf, kf + lambda);
    
    _tmpl = (1 - train_interp_factor) * _tmpl + (train_interp_factor) * x;
    _num = (1 - train_interp_factor) * _num + (train_interp_factor) * num;
    _den = (1 - train_interp_factor) * _den + (train_interp_factor) * den;

    _alphaf = complexDivision(_num, _den);*/

}

// Evaluates a Gaussian kernel with bandwidth SIGMA for all relative shifts between input images X and Y, which must both be MxN. They must    also be periodic (ie., pre-processed with a cosine window).
cv::Mat KCFTracker::gaussianCorrelation(cv::Mat x1, cv::Mat x2)
{
    using namespace FFTTools;
    cv::Mat c = cv::Mat( cv::Size(size_patch[1], size_patch[0]), CV_32F, cv::Scalar(0) );
    // HOG features
    if (_hogfeatures) {
        cv::Mat caux;
        cv::Mat x1aux;
        cv::Mat x2aux;
        for (int i = 0; i < size_patch[2]; i++) {
            x1aux = x1.row(i);   // Procedure do deal with cv::Mat multichannel bug
            x1aux = x1aux.reshape(1, size_patch[0]);
            x2aux = x2.row(i).reshape(1, size_patch[0]);
            cv::mulSpectrums(fftd(x1aux), fftd(x2aux), caux, 0, true); 
            caux = fftd(caux, true);
            rearrange(caux);
            caux.convertTo(caux,CV_32F);
            c = c + real(caux);
        }
    }
    // Gray features
    else {
        cv::mulSpectrums(fftd(x1), fftd(x2), c, 0, true);
        c = fftd(c, true);
        rearrange(c);
        c = real(c);
    }
    cv::Mat d; 
    cv::max(( (cv::sum(x1.mul(x1))[0] + cv::sum(x2.mul(x2))[0])- 2. * c) / (size_patch[0]*size_patch[1]*size_patch[2]) , 0, d);
	
    cv::Mat k;
    cv::exp((-d / (sigma * sigma)), k);
    return k;
}

// Create Gaussian Peak. Function called only in the first frame.
cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    cv::Mat_<float> res(sizey, sizex);

    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;

    float output_sigma = std::sqrt((float) sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5 / (output_sigma * output_sigma);

    for (int i = 0; i < sizey; i++)
        for (int j = 0; j < sizex; j++)
        {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float) (ih * ih + jh * jh));
        }
    return FFTTools::fftd(res);
}

// Obtain sub-window from image, with replication-padding and extract features
cv::Mat KCFTracker::getFeatures(const cv::Mat & image, bool inithann, float scale_adjust)
{
    cv::Rect extracted_roi;

    float cx = _roi.x + _roi.width / 2;
    float cy = _roi.y + _roi.height / 2;

    if (inithann) {
        int padded_w = _roi.width * padding;
        int padded_h = _roi.height * padding;
        
        if (template_size > 1) {  // Fit largest dimension to the given template size
            if (padded_w >= padded_h)  //fit to width
                _scale = padded_w / (float) template_size;
            else
                _scale = padded_h / (float) template_size;

            _tmpl_sz.width = padded_w / _scale;
            _tmpl_sz.height = padded_h / _scale;
        }
        else {  //No template size given, use ROI size
			printf("No template size given, use ROI size");
            _tmpl_sz.width = padded_w;
            _tmpl_sz.height = padded_h;
            _scale = 1;
            // original code from paper:
            /*if (sqrt(padded_w * padded_h) >= 100) {   //Normal size
                _tmpl_sz.width = padded_w;
                _tmpl_sz.height = padded_h;
                _scale = 1;
            }
            else {   //ROI is too big, track at half size
                _tmpl_sz.width = padded_w / 2;
                _tmpl_sz.height = padded_h / 2;
                _scale = 2;
            }*/
        }

        if (_hogfeatures) {
            // Round to cell size and also make it even
            _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
            _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * cell_size)) ) * 2 * cell_size ) + cell_size*2;
        }
        else {  //Make number of pixels even (helps with some logic involving half-dimensions)
            _tmpl_sz.width = (_tmpl_sz.width / 2) * 2;
            _tmpl_sz.height = (_tmpl_sz.height / 2) * 2;
        }
    }

    extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
    extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

    // center roi with new size
    extracted_roi.x = cx - extracted_roi.width / 2;
    extracted_roi.y = cy - extracted_roi.height / 2;

	//printf("ExROI(%d,%d,%d,%d)\n", extracted_roi.x, extracted_roi.y, extracted_roi.width, extracted_roi.height);

    cv::Mat FeaturesMap;  
    cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);
	if (z.cols == 0 || z.rows == 0) {
		return cv::Mat();
	}

    if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
        cv::resize(z, z, _tmpl_sz);
    } 
	
	//cout << "o";
	/*printf("(%f,%f)(%d,%d)(%d,%d,%d,%d)", 
		scale_adjust, _scale, _tmpl_sz.width, _tmpl_sz.height, 
		extracted_roi.x, extracted_roi.y, extracted_roi.width, extracted_roi.height);*/
	// HOG features
    if (_hogfeatures) { 
        IplImage z_ipl = z;
		//printf("z(%d,%d) by using tmpl(%d,%d)->",z.rows,z.cols, _tmpl_sz.height,_tmpl_sz.width);
        CvLSVMFeatureMapCaskade *map;
        getFeatureMaps(&z_ipl, cell_size, &map);
		//printf("map(%d:%d:%d)", map->sizeY, map->sizeX, map->numFeatures);
		if(map->sizeY>2 && map->sizeX >2) // sym
			normalizeAndTruncate(map,0.2f);
		//printf("mapNorm(%d:%d:%d)", map->sizeY, map->sizeX, map->numFeatures);
        PCAFeatureMaps(map);
        size_patch[0] = map->sizeY;
        size_patch[1] = map->sizeX;
        size_patch[2] = map->numFeatures;
		//printf("mapNormPCA(%d:%d:%d) with cell(%d)\n", map->sizeY, map->sizeX, map->numFeatures, cell_size);

        FeaturesMap = cv::Mat(cv::Size(map->numFeatures,map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
        FeaturesMap = FeaturesMap.t();
        freeFeatureMapObject(&map);

        // Lab features
        if (_labfeatures) {
            cv::Mat imgLab;
            cvtColor(z, imgLab, CV_BGR2Lab);
            unsigned char *input = (unsigned char*)(imgLab.data);

            // Sparse output vector
            cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0]*size_patch[1], CV_32F, float(0));

            int cntCell = 0;
            // Iterate through each cell
            for (int cY = cell_size; cY < z.rows-cell_size; cY+=cell_size){
                for (int cX = cell_size; cX < z.cols-cell_size; cX+=cell_size){
                    // Iterate through each pixel of cell (cX,cY)
                    for(int y = cY; y < cY+cell_size; ++y){
                        for(int x = cX; x < cX+cell_size; ++x){
                            // Lab components for each pixel
                            float l = (float)input[(z.cols * y + x) * 3];
                            float a = (float)input[(z.cols * y + x) * 3 + 1];
                            float b = (float)input[(z.cols * y + x) * 3 + 2];

                            // Iterate trough each centroid
                            float minDist = FLT_MAX;
                            int minIdx = 0;
                            float *inputCentroid = (float*)(_labCentroids.data);
                            for(int k = 0; k < _labCentroids.rows; ++k){
                                float dist = ( (l - inputCentroid[3*k]) * (l - inputCentroid[3*k]) )
                                           + ( (a - inputCentroid[3*k+1]) * (a - inputCentroid[3*k+1]) ) 
                                           + ( (b - inputCentroid[3*k+2]) * (b - inputCentroid[3*k+2]) );
                                if(dist < minDist){
                                    minDist = dist;
                                    minIdx = k;
                                }
                            }
                            // Store result at output
                            outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ; 
                            //((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ; 
                        }
                    }
                    cntCell++;
                }
            }
            // Update size_patch[2] and add features to FeaturesMap
            size_patch[2] += _labCentroids.rows;
            FeaturesMap.push_back(outputLab);
        }
    }
    else {
        FeaturesMap = RectTools::getGrayImage(z);
        FeaturesMap -= (float) 0.5; // In Paper;
        size_patch[0] = z.rows;
        size_patch[1] = z.cols;
        size_patch[2] = 1;  
    }
	//cout << "p";
    if (inithann) {
		//cout << "q";
        createHanningMats();
    }
	//printf("(%dby%d(%d) mul %dby%d(%d))", hann.rows, hann.cols, hann.type(), FeaturesMap.rows, FeaturesMap.cols, FeaturesMap.type());
    FeaturesMap = hann.mul(FeaturesMap);
	//cout << "r" << endl;
	return FeaturesMap;
}
cv::Mat KCFTracker::getFeaturesROI(const cv::Mat & image, const cv::Rect &roi_specified, float scale_adjust) {
	cv::Rect extracted_roi;

	float cx = _roi.x + _roi.width / 2;
	float cy = _roi.y + _roi.height / 2;

	//_tmpl_sz.width = roi_specified.width;
	//_tmpl_sz.height = roi_specified.height;

	//printf("%f * (%d,%d)-(%.2f,%.2f)(%d,%d)", _scale, _tmpl_sz.width, _tmpl_sz.height, _roi.width, _roi.height,roi_specified.width, roi_specified.height);

	extracted_roi.width = scale_adjust * _scale * _tmpl_sz.width;
	extracted_roi.height = scale_adjust * _scale * _tmpl_sz.height;

	//extracted_roi.width = roi_specified.width;  //scale_adjust * _scale * _tmpl_sz.width;
	//extracted_roi.height = roi_specified.height; //scale_adjust * _scale * _tmpl_sz.height;

	//printf("->(%d,%d)\n", extracted_roi.width, extracted_roi.height);

	// center roi with new size
	extracted_roi.x = cx - extracted_roi.width / 2;
	extracted_roi.y = cy - extracted_roi.height / 2;

	cv::Mat FeaturesMap;
	cv::Mat z = RectTools::subwindow(image, extracted_roi, cv::BORDER_REPLICATE);

	if (z.cols != _tmpl_sz.width || z.rows != _tmpl_sz.height) {
		cv::resize(z, z, _tmpl_sz);
	}

	// HOG features
	if (_hogfeatures) {
		IplImage z_ipl = z;
		CvLSVMFeatureMapCaskade *map;
		getFeatureMaps(&z_ipl, cell_size, &map);
		normalizeAndTruncate(map, 0.2f);
		PCAFeatureMaps(map);
		size_patch[0] = map->sizeY;
		size_patch[1] = map->sizeX;
		size_patch[2] = map->numFeatures;

		FeaturesMap = cv::Mat(cv::Size(map->numFeatures, map->sizeX*map->sizeY), CV_32F, map->map);  // Procedure do deal with cv::Mat multichannel bug
		FeaturesMap = FeaturesMap.t();
		freeFeatureMapObject(&map);

		// Lab features
		if (_labfeatures) {
			cv::Mat imgLab;
			cvtColor(z, imgLab, CV_BGR2Lab);
			unsigned char *input = (unsigned char*)(imgLab.data);

			// Sparse output vector
			cv::Mat outputLab = cv::Mat(_labCentroids.rows, size_patch[0] * size_patch[1], CV_32F, float(0));

			int cntCell = 0;
			// Iterate through each cell
			for (int cY = cell_size; cY < z.rows - cell_size; cY += cell_size) {
				for (int cX = cell_size; cX < z.cols - cell_size; cX += cell_size) {
					// Iterate through each pixel of cell (cX,cY)
					for (int y = cY; y < cY + cell_size; ++y) {
						for (int x = cX; x < cX + cell_size; ++x) {
							// Lab components for each pixel
							float l = (float)input[(z.cols * y + x) * 3];
							float a = (float)input[(z.cols * y + x) * 3 + 1];
							float b = (float)input[(z.cols * y + x) * 3 + 2];

							// Iterate trough each centroid
							float minDist = FLT_MAX;
							int minIdx = 0;
							float *inputCentroid = (float*)(_labCentroids.data);
							for (int k = 0; k < _labCentroids.rows; ++k) {
								float dist = ((l - inputCentroid[3 * k]) * (l - inputCentroid[3 * k]))
									+ ((a - inputCentroid[3 * k + 1]) * (a - inputCentroid[3 * k + 1]))
									+ ((b - inputCentroid[3 * k + 2]) * (b - inputCentroid[3 * k + 2]));
								if (dist < minDist) {
									minDist = dist;
									minIdx = k;
								}
							}
							// Store result at output
							outputLab.at<float>(minIdx, cntCell) += 1.0 / cell_sizeQ;
							//((float*) outputLab.data)[minIdx * (size_patch[0]*size_patch[1]) + cntCell] += 1.0 / cell_sizeQ; 
						}
					}
					cntCell++;
				}
			}
			// Update size_patch[2] and add features to FeaturesMap
			size_patch[2] += _labCentroids.rows;
			FeaturesMap.push_back(outputLab);
		}
	}
	else {
		FeaturesMap = RectTools::getGrayImage(z);
		FeaturesMap -= (float) 0.5; // In Paper;
		size_patch[0] = z.rows;
		size_patch[1] = z.cols;
		size_patch[2] = 1;
	}

	FeaturesMap = hann.mul(FeaturesMap);
	return FeaturesMap;
}
// Deep Copty by using Operator "="
KCFTracker& KCFTracker::operator=(const KCFTracker& copy) {
	if (this == &copy) // if same instance (mermory address)
		return *this;

	this->interp_factor = copy.interp_factor;
	this->sigma = copy.sigma;
	this->lambda = copy.lambda;
	this->cell_size = copy.cell_size;
	this->cell_sizeQ = copy.cell_sizeQ;
	this->padding = copy.padding;
	this->output_sigma_factor = copy.output_sigma_factor;
	this->template_size = copy.template_size;
	this->scale_step = copy.scale_step;
	this->scale_weight = copy.scale_weight;

	// cv::Mat
	/// Release
	if (!this->_alphaf.empty()) this->_alphaf.release();
	if (!this->_prob.empty()) this->_prob.release();
	if (!this->_tmpl.empty()) this->_tmpl.release();
	if (!this->_num.empty()) this->_num.release();
	if (!this->_den.empty()) this->_den.release();
	if (!this->_labCentroids.empty()) this->_labCentroids.release();
	if (!this->hann.empty()) this->hann.release();
	/// Copy
	this->_alphaf = copy._alphaf.clone();
	this->_prob = copy._prob.clone();
	this->_tmpl = copy._tmpl.clone();
	this->_num = copy._num.clone();
	this->_den = copy._den.clone();
	this->_labCentroids = copy._labCentroids.clone();
	this->hann = copy.hann.clone();

	this->size_patch[0] = copy.size_patch[0];
	this->size_patch[1] = copy.size_patch[1];
	this->size_patch[2] = copy.size_patch[2];

	this->_tmpl_sz = copy._tmpl_sz;
	this->_scale = copy._scale;
	this->_scale_MOT = copy._scale_MOT;
	this->_gaussian_size = copy._gaussian_size;
	this->_hogfeatures = copy._hogfeatures;
	this->_labfeatures = copy._labfeatures;

	this->fWidth = copy.fWidth;
	this->fHeight = copy.fHeight;

	return *this;
}
// Deep Copy
void KCFTracker::copyTo(KCFTracker& dst) {
	dst.interp_factor = this->interp_factor;
	dst.sigma = this->sigma;
	dst.lambda = this->lambda;
	dst.cell_size = this->cell_size;
	dst.cell_sizeQ = this->cell_sizeQ;
	dst.padding = this->padding;
	dst.output_sigma_factor = this->output_sigma_factor;
	dst.template_size = this->template_size;
	dst.scale_step = this->scale_step;
	dst.scale_weight = this->scale_weight;

	// cv::Mat
	/// Release
	if (!dst._alphaf.empty()) dst._alphaf.release();
	if (!dst._prob.empty()) dst._prob.release();
	if (!dst._tmpl.empty()) dst._tmpl.release();
	if (!dst._num.empty()) dst._num.release();
	if (!dst._den.empty()) dst._den.release();
	if (!dst._labCentroids.empty()) dst._labCentroids.release();
	if (!dst.hann.empty()) dst.hann.release();
	/// Copy
	dst._alphaf = this->_alphaf.clone();
	dst._prob = this->_prob.clone();
	dst._tmpl = this->_tmpl.clone();
	dst._num = this->_num.clone();
	dst._den = this->_den.clone();
	dst._labCentroids = this->_labCentroids.clone();
	dst.hann = this->hann.clone();

	dst.size_patch[0] = this->size_patch[0];
	dst.size_patch[1] = this->size_patch[1];
	dst.size_patch[2] = this->size_patch[2];

	dst._tmpl_sz = this->_tmpl_sz;
	dst._scale = this->_scale;
	dst._scale_MOT = this->_scale_MOT;
	dst._gaussian_size = this->_gaussian_size;
	dst._hogfeatures = this->_hogfeatures;
	dst._labfeatures = this->_labfeatures;

	dst.fWidth = this->fWidth;
	dst.fHeight = this->fHeight;
}
// Initialize Hanning window. Function called only in the first frame.
void KCFTracker::createHanningMats()
{
	//printf("[%d:%d:%d]\n", size_patch[0], size_patch[1], size_patch[2]);
	if (size_patch[0] == 0 || size_patch[1] == 0 || size_patch[2]==0) //ymsong
		printf("[ERROR] KCFTracker::createHanningMats (line 1105) with (size_patch[0] == 0 || size_patch[1] == 0 || size_patch[2]==0)\n");

    cv::Mat hann1t = cv::Mat(cv::Size(size_patch[1],1), CV_32F, cv::Scalar(0));
    cv::Mat hann2t = cv::Mat(cv::Size(1,size_patch[0]), CV_32F, cv::Scalar(0)); 
    
	for (int i = 0; i < hann1t.cols; i++)
        hann1t.at<float > (0, i) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann1t.cols - 1)));
    for (int i = 0; i < hann2t.rows; i++)
        hann2t.at<float > (i, 0) = 0.5 * (1 - std::cos(2 * 3.14159265358979323846 * i / (hann2t.rows - 1)));

    cv::Mat hann2d = hann2t * hann1t;
    // HOG features
    if (_hogfeatures) {
        cv::Mat hann1d = hann2d.reshape(1,1); // Procedure do deal with cv::Mat multichannel bug
        
        hann = cv::Mat(cv::Size(size_patch[0]*size_patch[1], size_patch[2]), CV_32F, cv::Scalar(0));
        for (int i = 0; i < size_patch[2]; i++) {
            for (int j = 0; j<size_patch[0]*size_patch[1]; j++) {
                hann.at<float>(i,j) = hann1d.at<float>(0,j);
            }
        }
    }
    // Gray features
    else {
        hann = hann2d;
    }
}

// Calculate sub-pixel peak for one dimension
float KCFTracker::subPixelPeak(float left, float center, float right)
{   
    float divisor = 2 * center - right - left;

    if (divisor == 0)
        return 0;
    
    return 0.5 * (right - left) / divisor;
}
