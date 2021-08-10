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

// user-defined pre-processor,...
// user-defined containters, data-type,..

// #include <opencv2\opencv.hpp>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include "opencv2\video\tracking.hpp"	// Kalman Filtering
#include <opencv2/imgproc.hpp>			// image transformation

// IOU between 3D Boxes
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>

#include "VOT.h"
#include "kcftracker.hpp"
#include <unordered_map>
//#include "siamRPN_tracker.hpp"
//#include "DaSiamTracker.h"

#define MAX_OBJECTS	27	

#define SIZE_CONSTRAINT_RATIO	2

// Parameters for the GMPHD filter
#define PI_8				3.14159265
#define e_8				2.71828182
#define T_th				(0.0)
#define W_th				(0.0)
#define Q_TH_LOW_20			0.00000000000000000001f		// 0.1^20
#define Q_TH_LOW_20_INVERSE		100000000000000000000.0f	// 10^20
#define Q_TH_LOW_15			0.00000000000001f		// 0.1^15
#define Q_TH_LOW_15_INVERSE		1000000000000000.0f		// 10^15
#define Q_TH_LOW_10			0.0000000001f			// 0.1^10
#define Q_TH_LOW_10_INVERSE		10000000000.0f			// 10^10
#define Q_TH_LOW_8			0.00000001f			// 0.1^8
#define Q_TH_LOW_8_INVERSE		100000000.0f			// 10^8
#define P_SURVIVE_LOW			0.99				// object number >=2 : 0.99, else 0.95
#define P_SURVIVE_MID			0.95
#define VAR_X				25 //42.96 // 279.13 // 1116.520// 0.1 //25 // 171.87
#define VAR_Y				100//29.27// 47.685 // 190.740 //0.1 //100 // 117.09
#define VAR_D				25//0.1 //25
#define VAR_R				0.09 // 0.1, 0.04, 0.01
#define VAR_X_VEL			25// 42.96 // 279.13 // 1116.520// 0.1 //25 // 171.871
#define VAR_Y_VEL			100// 29.27//47.685 // 190.740//0.1//100 // 117.090
#define VAR_D_VEL			25//0.1 // 25	// 1, 4, 9, 16, 25, 100, 225, 400, 900
#define VAR_R_VEL			100 // 0.1, 0.04, 0.01
#define VAR_WIDTH			100
#define VAR_HEIGHT			400
#define VAR_X_MID			100
#define VAR_Y_MID			400
#define VAR_X_VEL_MID			100
#define VAR_Y_VEL_MID			400

//#define VELOCITY_UPDATE_ALPHA	0.4f // moving camera scene with low fps ( <= 15)
#define CONFIDENCE_UPDATE_ALPHA	0.95f

// Parameters for Data Association
#define TRACK_ASSOCIATION_FRAME_DIFFERENCE	0	// 0: equal or later, 1:later
#define ASSOCIATION_STAGE_1_GATING_ON		0
#define ASSOCIATION_STAGE_2_GATING_ON		0
#define AFFINITY_COST_L1_OR_L2_NORM_FW		1	// 0: L1-norm, 1: L2-norm, frame-wise
#define AFFINITY_COST_L1_OR_L2_NORM_TW		1	// 0: L1-norm, 1: L2-norm, tracklet-wise
// SOT Tracker Options
#define SOT_USE_KCF_TRACKER			1
#define SOT_USE_SIAMRPN_TRACKER			2
#define SOT_USE_DASIAMRPN_TRACKER		3
#define SOT_TRACK_OPT				SOT_USE_KCF_TRACKER

// (D2TA) GMPHD-KCF Fusion (D2TA)
#define APPEARANCE_STRICT_UPDATE_ON		0
#define APPEARANCE_UPDATE_DET_TH		0.85f

// (T2TA) GMPHD-KCF Fusion (T2TA)
//#define SAF_MASK_T2TA_ON			1

// #define KCF_SOT_T2TA_ON			0
//#define IOU_UPPER_TH_T2TA_ON			1

// MOTS
#define COV_UPDATE_T2TA				0

#define USE_LINEAR_MOTION			0
#define USE_KALMAN_MOTION			1
#define T2TA_MOTION_OPT				USE_LINEAR_MOTION			
#define KALMAN_INIT_BY_LINEAR			1

// Parameters for Occlusion Group Management (Merge and Occlusion Group Energy Minimization)
#define MERGE_DET_ON			1
#define MERGE_TRACK_ON			1
#define MERGE_METRIC_OPT		2	// 0: SIOA, 1:IOU, 2:sIOU
#define MERGE_THRESHOLD_RATIO	0.4f
#define MERGE_METRIC_SIOA		0
#define MERGE_METRIC_IOU		1
#define MERGE_METRIC_mIOU		2

#define GROUP_MANAGEMENT_FRAME_WISE_ON	1

#define USE_GMPHD_NAIVE			0
#define USE_GMPHD_HDA			1
#define GMPHD_TRACKER_MODE		USE_GMPHD_HDA

// Tracking Results Writing Option
#define EXCLUDE_BBOX_OUT_OF_FRAME	0

// Visualization Option (main function)
#define VISUALIZATION_MAIN_ON		1
#define SKIP_FRAME_BY_FRAME		0
#define VISUALIZATION_RESIZE_ON		0
#define BOUNDING_BOX_THICK		5
#define TRAJECTORY_THICK		5
#define	ID_CONFIDENCE_FONT_SIZE		2
#define	ID_CONFIDENCE_FONT_THICK	2
#define FRAME_COUNT_FONT_SIZE		3
#define FRAME_COUNT_FONT_THICK		3
// Visualization Option (tracker class)
#define VIS_D2TA_DETAIL			0
#define VIS_T2TA_DETAIL			0
#define VIS_TRACK_MERGE			0

// Image Save Option (main function)
#define SAVE_IMG_ON			0

// Dataset Indices
#define DB_TYPE_MOT15			0	// MOT Challenge 2015 Dataset (-1)
#define DB_TYPE_MOT17			1	// MOT Challenge 2017 Dataset
#define DB_TYPE_CVPR19			2   	// MOT Challenge 2019 (CVPR 2019) Dataset
#define DB_TYPE_MOT20			2   	// MOT Challenge 2020
#define DB_TYPE_KITTI			3	// KITTI 	(2D box, 3D box, 3D Point Cloud)
#define DB_TYPE_KITTI_MOTS		4	// KITTI-MOTS 	(2D Instance Segment)
#define DB_TYPE_MOTS20			5	// MOTS20	(2D Instacne Segment)

#define DEBUG_PRINT			0

/*--------------------------------------------------------------------------------------*/


typedef struct boundingbox_id {
	int id;
	int idx;
	int min_id;		// minimum value ID in a group (is considered as group ID)
	double weight = 0.0;
	cv::Rect rec;	// t, t-1, t-2
	boundingbox_id() {
	}
	boundingbox_id(int id, cv::Rect occRect = cv::Rect()) :id(id) {
		// Deep Copy
		id = id;
		rec = occRect;
	}
	boundingbox_id& operator=(const boundingbox_id& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->idx = copy.idx;
		this->id = copy.id;
		this->min_id = copy.min_id;
		this->rec = copy.rec;
		this->weight = copy.weight;
	}
	bool operator<(const boundingbox_id& rect) const {
		return (id < rect.id);
	}
}RectID;

typedef struct track_id_bb_mask {
	int id;
	int fn;
	int object_type;
	cv::Rect bb;
	cv::Mat mask;

	//track_id_bb_mask() {

	//}
	//track_id_bb_mask(int id_, int fn_, cv::Rect bb, cv::Mat mask_=cv::Mat()) {
	//	this->id = id_;
	//	this->fn = fn_;
	//	this->bb = bb;

	//	this->mask = mask_.clone();
	//}

	track_id_bb_mask& operator=(const track_id_bb_mask& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->id = copy.id;
		this->fn = copy.fn;
		this->object_type = copy.object_type;
		this->bb = copy.bb;

		if (!this->mask.empty()) this->mask.release();
		this->mask = copy.mask.clone();

		return *this;
	}
} track_info;

typedef struct bbTrack {
	//VOT *papp ;
	KCFTracker papp;
	cv::Mat segMask;
	std::string segMaskRle;
	std::vector<float> reid_features;

	int fn;
	int id;
	int id_associated = -1; // (index) it is able to be used in Tracklet-wise association
	/*----MOTS Tracking Only---*/
	int det_id;
	float det_confidence;
	/*----MOTS Tracking Only---*/
	int fn_latest_T2TA = 0;
	cv::Rect rec;
	cv::Rect rec_corr;
	float rec6D[6];
	std::vector<cv::Vec3f> top_corners;
	std::vector<cv::Vec3f> bottom_corners;
	float vx;
	float vy;
	float vd;
	float vr;
	float weight;
	float conf;
	cv::KalmanFilter KF;
	cv::Mat cov;
	cv::Mat tmpl;
	cv::Mat hist;
	float density;
	float depth;
	float ratio_yx;
	float rotation_y;
	bool isNew;
	bool isAlive;
	bool isMerged = false;
	int isInterpolated = 0; // 0: Online, 1: Interpolated, -1:Ignored (Falsely Interpolated)
	int iGroupID = -1;
	bool isOcc = false;
	int size = 0;
	int objType;
	std::vector<RectID> occTargets;
	bbTrack() {}
	bbTrack(int fn, int id, int isOcc, cv::Rect rec, cv::Mat obj = cv::Mat(), cv::Mat hist = cv::Mat()) :
		fn(fn), id(id), isOcc(isOcc), rec(rec) {
		if (!hist.empty()) {
			this->hist.release();
			this->hist = hist.clone(); // deep copy
		}
		else {
			//this->hist.release();
			//printf("[ERROR]target_bb's parameter \"hist\" is empty!\n");
			this->hist = hist;
		}
		if (!obj.empty()) {
			this->tmpl.release();
			this->tmpl = obj.clone(); // deep copy
		}
		else {
			//this->obj_tmpl.release();
			this->tmpl = obj;
		}
		//isOccCorrNeeded = false; // default

	}
	bool operator<(const bbTrack& trk) const {
		return (id < trk.id);
	}
	bbTrack& operator=(const bbTrack& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		//this->KCFfeatures = copy.KCFfeatures;
		this->papp = copy.papp;
		if (!this->segMask.empty()) this->segMask.release();
		if (!copy.segMask.empty())	this->segMask = copy.segMask;
		if (!this->reid_features.empty()) this->reid_features.clear();
		this->reid_features = copy.reid_features;
		
		this->fn = copy.fn;
		this->id = copy.id;
		this->id_associated = copy.id_associated;
		this->size = copy.size;
		this->fn_latest_T2TA = copy.fn_latest_T2TA;
		this->rec = copy.rec;
		this->rec6D[0] = copy.rec6D[0];
		this->rec6D[1] = copy.rec6D[1];
		this->rec6D[2] = copy.rec6D[2];
		this->rec6D[3] = copy.rec6D[3];
		this->rec6D[4] = copy.rec6D[4];
		this->rec6D[5] = copy.rec6D[5];
		this->top_corners = copy.top_corners;
		this->bottom_corners = copy.bottom_corners;
		this->vx = copy.vx;
		this->vy = copy.vy;
		this->vd = copy.vd;
		this->vr = copy.vr;
		this->rec_corr = copy.rec_corr;
		this->density = copy.density;
		this->depth = copy.depth;
		this->rotation_y = copy.rotation_y;
		this->ratio_yx = copy.ratio_yx;
		this->isNew = copy.isNew;
		this->isAlive = copy.isAlive;
		this->isMerged = copy.isMerged;
		this->isInterpolated = copy.isInterpolated;
		this->iGroupID = copy.iGroupID;
		this->isOcc = copy.isOcc;
		this->weight = copy.weight;
		this->conf = copy.conf;
		this->objType = copy.objType;
		this->segMaskRle = copy.segMaskRle;

		if (!this->cov.empty()) this->cov.release();
		if (!this->tmpl.empty()) this->tmpl.release();
		if (!this->hist.empty()) this->hist.release();
	
		this->cov = copy.cov.clone();
		this->tmpl = copy.tmpl.clone();
		this->hist = copy.hist.clone();

		/*----MOTS Tracking Only---*/
		this->det_id = copy.det_id;
		this->det_confidence = copy.det_confidence;
		/*----MOTS Tracking Only---*/

		return *this;
	}
	void CopyTo(bbTrack& dst) {
		dst.papp = this->papp;
		if (!this->segMask.empty()) dst.segMask = this->segMask.clone();
		if (!this->reid_features.empty()) dst.reid_features = this->reid_features;

		dst.fn = this->fn;
		dst.id = this->id;
		dst.id_associated = this->id_associated;
		dst.rec = this->rec;
		dst.rec6D[0] = this->rec6D[0];
		dst.rec6D[1] = this->rec6D[1];
		dst.rec6D[2] = this->rec6D[2];
		dst.rec6D[3] = this->rec6D[3];
		dst.rec6D[4] = this->rec6D[4];
		dst.rec6D[5] = this->rec6D[5];
		dst.top_corners = this->top_corners;
		dst.bottom_corners = this->bottom_corners;
		dst.vx = this->vx;
		dst.vy = this->vy;
		dst.vd = this->vd;
		dst.vr = this->vr;
		dst.rec_corr = this->rec_corr;
		dst.density = this->density;
		dst.depth = this->depth;
		dst.rotation_y = this->rotation_y;
		dst.isNew = this->isNew;
		dst.isAlive = this->isAlive;
		dst.isMerged = this->isMerged;
		dst.isInterpolated = this->isInterpolated;
		dst.iGroupID = this->iGroupID;
		dst.isOcc = this->isOcc;
		dst.weight = this->weight;
		dst.conf = this->conf;
		dst.objType = this->objType;
		dst.segMaskRle = this->segMaskRle;

		if (!this->cov.empty()) dst.cov = this->cov.clone();
		if (!this->tmpl.empty()) dst.tmpl = this->tmpl.clone();
		if (!this->hist.empty()) dst.hist = this->hist.clone();

		/*----MOTS Tracking Only-- - */
		dst.det_id = this->det_id;
		dst.det_confidence = this->det_confidence;
		/*----MOTS Tracking Only---*/
	}
	void Destroy() {
		if (!this->cov.empty()) this->cov.release();
		if (!this->tmpl.empty()) this->tmpl.release();
		if (!this->hist.empty()) this->hist.release();
	}
	void InitKF(int nStats, int mObs, float x_var, float y_var) {
		this->KF = cv::KalmanFilter(nStats, mObs, 0);

		// F, constant
		setIdentity(KF.transitionMatrix);
		this->KF.transitionMatrix.at<float>(0, 2) = 1;
		this->KF.transitionMatrix.at<float>(1, 3) = 1;

		// H, constant
		this->KF.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);

		setIdentity(this->KF.processNoiseCov, cv::Scalar::all(x_var / 2));	// Q, constant, 4^2/2, 4x4
		this->KF.processNoiseCov.at<float>(1, 1) = y_var / 2;
		this->KF.processNoiseCov.at<float>(3, 3) = y_var / 2;
		//cout << this->KF.processNoiseCov.rows << "x" << this->KF.processNoiseCov.cols << endl;

		setIdentity(this->KF.measurementNoiseCov, cv::Scalar::all(x_var));	// R, constant, 4^2, 2x2
		this->KF.measurementNoiseCov.at<float>(1, 1) = y_var;
		//cout << this->KF.measurementNoiseCov.rows << "x" << this->KF.measurementNoiseCov.cols << endl;

		setIdentity(this->KF.errorCovPre, cv::Scalar::all(1));				// P_t|t-1, predicted from P_t-1
		setIdentity(this->KF.errorCovPost, cv::Scalar::all(1));				// P_0 or P_t,	just for init (variable)
	}
}BBTrk;

typedef struct bbDet {
	int fn;
	int object_type = -1;
	cv::Rect rec;
	float depth;		// used to represent the 2.5D Bounding Box (2D + Depth from Point Cloud)
	float rotation_y;
	float ratio_yx;
	float rec6D[6];
	std::vector<cv::Vec3f> top_corners;
	std::vector<cv::Vec3f> bottom_corners;
	std::vector<float> distances;
	float confidence;
	float weight;		// normalization value of confidence at time t
	int id;				// Used in Looking Back Association
	int input_type;		// 2D Box, 3D Box, 3D Point Cloud, 2D Instance 
	cv::Mat segMask;
	std::string segMaskRle;
	std::vector<float> reid_features;

	bbDet& operator=(const bbDet& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		if (!this->segMask.empty()) this->segMask.release();
		this->segMask = copy.segMask;
		this->segMaskRle = copy.segMaskRle;

		if (!this->reid_features.empty()) this->reid_features.clear();
		this->reid_features = copy.reid_features;
		if (!this->distances.empty()) this->distances.clear();
		this->distances = copy.distances;

		this->fn = copy.fn;
		this->object_type = copy.object_type;
		this->rec = copy.rec;
		this->rec6D[0] = copy.rec6D[0];
		this->rec6D[1] = copy.rec6D[1];
		this->rec6D[2] = copy.rec6D[2];
		this->rec6D[3] = copy.rec6D[3];
		this->rec6D[4] = copy.rec6D[4];
		this->rec6D[5] = copy.rec6D[5];

		if (!this->top_corners.empty()) this->top_corners.clear();
		this->top_corners = copy.top_corners;
		if (!this->bottom_corners.empty()) this->bottom_corners.clear();
		this->bottom_corners = copy.bottom_corners;

		this->depth = copy.depth;
		this->rotation_y = copy.rotation_y;
		this->ratio_yx = copy.ratio_yx;
		this->confidence = copy.confidence;

		this->weight = copy.weight;
		this->id = copy.id;
		this->input_type = copy.input_type;

		return *this;
	}
}BBDet;

// 함수의 정의, 변수의 초기화를 header 파일에서 하면서
// 다른 파일(소스)에서 해당 함수, 변수를 가져다 쓰고 싶으면 static 밖에는 방법이 없다.
// 아니면, 선언은 헤더파일, 정의 및 초기화는 cpp 에서 한뒤
// extern 키워드를 쓰면된다.
// https://zerobell.tistory.com/22 참고
// 어짜피 값의 변동이 없을 때는 static으로 하는게 제일 편하다
// 함수는 자동으로 extern 이 생략되어있다. (파일 내 전역 변수도 마찬가지)

namespace sym {
	static enum MODEL_VECTOR {
		XY, XYWH, XYRyx
	};

	static int DIMS_STATE[3] = { 4, 6, 6};
	static int DIMS_OBS[3] = { 2, 4, 3};

	static enum OBJECT_TYPE {
		CAR, VAN,
		PEDESTRIAN, PERSON_SITTING,
		CYCLIST, TRUCK, TRAM,
		MISC, DONTCARE, BUS
	};
	static std::string OBJECT_STRINGS[10] = \
	{ "Car", "Van",
		"Pedestrian", "Person_sitting",
		"Cyclist", "Truck", "Tram",
		"Misc", "DontCare", "Bus" };

	static std::string DB_NAMES[6] = { "MOT15", "MOT17", "MOT20", "KITTI", "KITTI", "MOTS20" };
	static int FRAME_OFFSETS[6] = { 1,1,1,0,0,1 };

	static CvScalar OBJECT_TYPE_COLORS[9] = {
		cvScalar(255, 255, 0),	/*Car: Lite Blue (Mint)*/
		cvScalar(255, 0, 0),	/*Van: Blue*/
		cvScalar(255, 0, 255),	/*Pedestrian: Pink*/
		cvScalar(0, 0, 255),	/*Person_sitting: Red*/
		cvScalar(42, 42, 165),	/*Cyclist: Brown*/
		cvScalar(0, 255, 0),	/*Truck: Lite Green*/
		cvScalar(0, 255, 255),	/*Tram: Yellow*/
		cvScalar(64, 64, 64),	/*Misc: Gray*/
		cvScalar(0, 0, 0) };	/*DontCare: Black*/

	// Parameters: 11*10*7 = 770 cases * num of scenes * num of detectors * num of object classes, too many.
	static float DET_SCORE_THS[15] = { -100.0, 0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.0 };
	//static float DET_SCORE_THS[11] = { 0.0,0.52,0.54,0.55,0.56,0.58,0.62,0.64,0.65,0.66,0.68};
	static int TRACK_INIT_MIN[10] = { 1,2,3,4,5,6,7,8,9,10 };
	static int TRACK_T2TA_MAX[8] = { 5,10,15,20,30,60,80,100 };
	static float VEL_UP_ALPHAS[12] = { 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99 };
	static float MERGE_THS[8] = { 0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 };
	static float DET_SCORE_THS_LOWER[7] = { 0.0,0.1,0.2,0.3,0.4,0.5,0.6 };

	static std::string MERGE_METRICS_STR[3] = { "SIOA","IOU", "mIOU" };

	static enum AFFINITY_OPT { GMPHD, KCF, MAF };
}

typedef std::vector<std::vector<std::vector<std::vector<float>>>> VECx4xFLT;
typedef std::vector<std::vector<std::vector<float>>> VECx3xFLT;
typedef std::vector<std::vector<std::vector<cv::Mat>>> VECx3xMAT;
typedef std::vector<std::vector<std::vector<std::string>>> VECx3xSTR;
typedef std::vector<std::vector<std::vector<BBDet>>> VECx3xBBDet;
typedef std::vector<std::vector<BBDet>> VECx2xBBDet;
typedef std::vector<std::vector<std::vector<BBTrk>>> VECx3xBBTrk;
typedef std::vector<std::vector<BBTrk>> VECx2xBBTrk;

#define IS_DONTCARE(x) (((int)x)==sym::OBJECT_TYPE::DONTCARE)
#define IS_VEHICLE_EVAL(x) (((int)x==sym::OBJECT_TYPE::CAR) ||((int)x==sym::OBJECT_TYPE::VAN))
#define IS_VEHICLE_ALL(x) (((int)x==sym::OBJECT_TYPE::CAR) ||((int)x==sym::OBJECT_TYPE::VAN)||((int)x==sym::OBJECT_TYPE::TRUCK)||((int)x==sym::OBJECT_TYPE::BUS))
#define IS_PERSON_KITTI(x) (((int)x==sym::OBJECT_TYPE::PEDESTRIAN) ||((int)x==sym::OBJECT_TYPE::PERSON_SITTING))
#define IS_PERSON_EVAL(x) (((int)x==sym::OBJECT_TYPE::PEDESTRIAN) ||((int)x==sym::OBJECT_TYPE::PERSON_SITTING)||((int)x==sym::OBJECT_TYPE::CYCLIST))

typedef struct MOTparams {
	/*TARGET CLASS*/
	int OBJ_TYPE = sym::OBJECT_TYPE::PEDESTRIAN;
	/*PARAMS*/
	float DET_MIN_CONF = 0.0f; // upper bound
	int TRACK_MIN_SIZE = 1;
	int QUEUE_SIZE = 10;
	int FRAMES_DELAY_SIZE = TRACK_MIN_SIZE - 1;
	int T2TA_MAX_INTERVAL = 10;
	float VEL_UP_ALPHA = 0.5;
	int MERGE_METRIC = MERGE_METRIC_OPT;
	float MERGE_RATIO_THRESHOLD = MERGE_THRESHOLD_RATIO;
	int GROUP_QUEUE_SIZE = TRACK_MIN_SIZE * 10;
	int FRAME_OFFSET = 0;
	/*Gating */
	bool GATE_D2TA_ON = true;
	bool GATE_T2TA_ON = true;

	/*Simple Affinity Fusion*/
	/// 0: GMPHD (Position and Motion) only, 1: KCF (Appearance) Only, 2: Simple Affinity Fuion On
	int SAF_D2TA_MODE = 2;
	int SAF_T2TA_MODE = 2;
	bool SAF_MASK_D2TA = true; // compute appearance considering pixel-wise mask
	bool SAF_MASK_T2TA = true;
	cv::Vec2f KCF_BOUNDS_D2TA;
	cv::Vec2f KCF_BOUNDS_T2TA;
	cv::Vec2f IOU_BOUNDS_D2TA;
	cv::Vec2f IOU_BOUNDS_T2TA;
	/*Simple Affinity Fusion*/

	MOTparams() {
	}
	MOTparams(int obj_type, float dConf_th, int trk_min, int t2ta_max, int mg_metric, float mg_th, float vel_alpha, int group_q_size, int frames_offset,
		int saf_d2ta = 2, int saf_t2ta = 2, bool mask_d2ta = true, bool mask_t2ta = true,
		const cv::Vec2f& kcf_bounds_d2ta = cv::Vec2f(0.5, 0.9), const cv::Vec2f& kcf_bounds_t2ta = cv::Vec2f(0.5, 0.9),
		const cv::Vec2f& iou_bounds_d2ta = cv::Vec2f(0.1, 0.9), const cv::Vec2f& iou_bounds_t2ta = cv::Vec2f(0.1, 0.9),
		const cv::Vec2b& GATES_ONOFF = cv::Vec2b(true, true)) {

		this->OBJ_TYPE = obj_type;

		this->DET_MIN_CONF = dConf_th;
		this->TRACK_MIN_SIZE = trk_min;
		this->FRAMES_DELAY_SIZE = trk_min - 1;
		this->T2TA_MAX_INTERVAL = t2ta_max;
		this->VEL_UP_ALPHA = vel_alpha;
		this->MERGE_METRIC = mg_metric;
		this->MERGE_RATIO_THRESHOLD = mg_th;
		this->GROUP_QUEUE_SIZE = group_q_size;
		this->FRAME_OFFSET = frames_offset;

		this->GATE_D2TA_ON = GATES_ONOFF[0];
		this->GATE_T2TA_ON = GATES_ONOFF[1];

		this->SAF_D2TA_MODE = saf_d2ta;
		this->SAF_T2TA_MODE = saf_t2ta;
		this->SAF_MASK_D2TA = mask_d2ta;
		this->SAF_MASK_T2TA = mask_t2ta;

		this->KCF_BOUNDS_D2TA = kcf_bounds_d2ta;
		this->KCF_BOUNDS_T2TA = kcf_bounds_t2ta;
		this->IOU_BOUNDS_D2TA = iou_bounds_d2ta;
		this->IOU_BOUNDS_T2TA = iou_bounds_t2ta;
	}
	MOTparams& operator=(const MOTparams& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->OBJ_TYPE = copy.OBJ_TYPE;
		this->DET_MIN_CONF = copy.DET_MIN_CONF;
		this->TRACK_MIN_SIZE = copy.TRACK_MIN_SIZE;
		this->FRAMES_DELAY_SIZE = copy.FRAMES_DELAY_SIZE;
		this->T2TA_MAX_INTERVAL = copy.T2TA_MAX_INTERVAL;
		this->VEL_UP_ALPHA = copy.VEL_UP_ALPHA;
		this->MERGE_METRIC = copy.MERGE_METRIC;
		this->MERGE_RATIO_THRESHOLD = copy.MERGE_RATIO_THRESHOLD;
		this->GROUP_QUEUE_SIZE = copy.GROUP_QUEUE_SIZE;
		this->FRAME_OFFSET = copy.FRAME_OFFSET;

		this->GATE_D2TA_ON = copy.GATE_D2TA_ON;
		this->GATE_T2TA_ON = copy.GATE_T2TA_ON;

		this->SAF_D2TA_MODE = copy.SAF_D2TA_MODE;
		this->SAF_T2TA_MODE = copy.SAF_T2TA_MODE;
		this->SAF_MASK_D2TA = copy.SAF_MASK_D2TA;
		this->SAF_MASK_T2TA = copy.SAF_MASK_T2TA;

		this->KCF_BOUNDS_D2TA = copy.KCF_BOUNDS_D2TA;
		this->KCF_BOUNDS_T2TA = copy.KCF_BOUNDS_T2TA;
		this->IOU_BOUNDS_D2TA = copy.IOU_BOUNDS_D2TA;
		this->IOU_BOUNDS_T2TA = copy.IOU_BOUNDS_T2TA;

		return *this;
	}
} GMPHDMAFparams;
