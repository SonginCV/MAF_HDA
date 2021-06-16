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
#include "hungarian.h"

class OnlineTracker
{
public:
	OnlineTracker();
	~OnlineTracker();

	cv::Mat *imgBatch;

	std::vector<std::vector<BBTrk>> allLiveReliables;

	cv::Scalar color_tab[MAX_OBJECTS];

	// getter and setter
	string GetSeqName();
	void SetSeqName(const string seqName);

	struct MOTparams GetParams();
	void SetParams(struct MOTparams params);

	int GetTotalFrames();
	void SetTotalFrames(int nFrames);

	int	GetObjType();
	void SetObjType(int type);

	int frmWidth, frmHeight;
protected:
	string seqName;

	struct MOTparams params;
	int trackObjType;
	int iTotalFrames;

	int sysFrmCnt;
	int usedIDcnt;

	std::vector<BBDet> *detsBatch;
	std::unordered_map<int, std::vector<BBTrk>> tracksbyID;

	std::vector<BBTrk> liveTrkVec;		// live tracks at time t (now)
	std::vector<BBTrk> *liveTracksBatch;
	std::vector<BBTrk> lostTrkVec;		// lost tracks at time t (was live tracks before time t)
	std::vector<BBTrk> *lostTracksBatch;

	// The container for Group management, index (0:t-d-2, 1:t-d-1, 2:t-d), d: delayed time
	std::unordered_map<int, std::vector<RectID>> *groupsBatch;

	vector<vector<int>> HungarianMethod(vector<vector<double>>& costMatrix, const int& nObs, const int& nStats);
	//----------------------------------------------------------------------------
	/** @brief convert sparse matrix to dense matrix to accelerate Hungarian method.
	@param sparseMatrix : source matrix (m1)
	@param denseMatrix : destination matrix (m2)
	@param sparse_value : sparsity value (0 for cost maximization, a big number for cost minimization)
	@return vector for translating m2 to m1 indices
	@remark */
	//----------------------------------------------------------------------------
	vector<vector<cv::Vec2i>> cvtSparseMatrix2Dense(const vector<vector<double>>& sparseMatrix, vector<vector<double>>& denseMatrix,
		const double& sparse_value = 10000);

	HungarianAlgorithm HungAlgo;

public:
	void InitColorTab();
	cv::Rect RectExceptionHandling(int fWidth, int fHeight, cv::Rect rect);	// Rect Region Correction for preventing out of frame
	cv::Rect RectExceptionHandling(const int& fWidth, const int& fHeight, cv::Point p1, cv::Point p2);
	cv::Mat cvPerspectiveTrans2Rect(const cv::Mat img, const vector<cv::Point2f> corners, const cv::Vec3f dims, float ry);
	void cvPrintMat(cv::Mat matrix, string name = "");
	void cvPrintVec2Vec(const vector<vector<double>>& costs, const string& name = "");
	void cvPrintVec2Vec(const vector<vector<int>>& costs, const string& name = "");
	void cvPrintVec2Vec(const vector<vector<cv::Vec2i>>& costs, const string& name = "");
	void cvPrintVec2Vec(const vector<vector<bool>>& costs, const string& name = "");
	string cvPrintRect(const cv::Rect& rec);
protected:
	void InitImagesQueue(int width, int height);
	void UpdateImageQueue(const cv::Mat img, int Q_SIZE);
	void InitializeTrackletsContainters();

	void CollectTracksbyID(unordered_map<int, vector<BBTrk>>& tracksbyID, vector<BBTrk>& tracks, bool isInit = false);
	void ClearOldEmptyTracklet(int current_fn, unordered_map<int, vector<BBTrk>>& tracks, int MAXIMUM_OLD);

	// Motion Estimation
	cv::Vec4f LinearMotionEstimation(vector<BBTrk> tracklet, int& idx_first_last_fd, int& idx_first, int& idx_last, int MODEL_TYPE, int reverse_offset = 0, int required_Q_size = 0);
	cv::Vec4f LinearMotionEstimation(unordered_map<int, vector<BBTrk>> tracks, int id, int& idx_first_last_fd, int& idx_first, int& idx_last, int MODEL_TYPE, int reverse_offset = 0, int required_Q_size = 0);

	BBTrk KalmanMotionbasedPrediction(BBTrk lostStat, BBTrk liveObs);

	cv::Point cvtRect2Point(const cv::Rect rec);
	bool CopyCovMatDiag(const cv::Mat src, cv::Mat& dst);

	void NormalizeWeight(vector<BBDet>& detVec);
	void NormalizeWeight(vector<vector<BBDet>>& detVecs);
	void NormalizeCostVec2Vec(vector<vector<double>>& m_cost, double& min_cost, double& max_cost, const int& MODEL_TYPE);

	cv::Rect MergeRect(const cv::Rect A, const cv::Rect B, const float alpha = 0.5);
	cv::Rect MergeMultiRect(const vector<cv::Rect> recs);

	cv::Point2f CalcSegMaskCenter(const cv::Rect& rec, const cv::Mat& mask);

	double CalcGaussianProbability(const int dims, const cv::Mat x, const cv::Mat mean, cv::Mat& cov);
public:
	float CalcIOU(const cv::Rect& A, const cv::Rect& B);
	float Calc3DIOU(vector<cv::Vec3f> Ca, cv::Vec3f Da, vector<cv::Vec3f> Cb, cv::Vec3f Db); // Calculate 3D IOU between two Cuboids
	float CalcSIOA(cv::Rect A, cv::Rect B);
	float CalcMIOU(const cv::Rect& Ri, const cv::Mat& Si, const cv::Rect& Rj, const cv::Mat& Sj, cv::Mat& mask_union);
protected:
	bool IsOutOfFrame(cv::Rect rec, int fWidth, int fHeight, int margin_x = 0, int margin_y = 0);
	bool IsPointInRect(cv::Point pt, cv::Rect rec);

	void DrawTrkBBS(cv::Mat& img, cv::Rect rec, cv::Scalar color, int thick, int id, double fontScale, string type, bool idBGinBB = false);

	template < typename Type > std::string to_str(const Type & t)
	{
		std::ostringstream os;
		os << t;
		return os.str();
	}
};

