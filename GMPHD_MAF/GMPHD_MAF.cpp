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
// GMPHD_MAF.cpp
#include "pch.h"
#include "GMPHD_MAF.h"

GMPHD_MAF::GMPHD_MAF()
{
	this->iTotalFrames = 0;
	this->sysFrmCnt = 0;
	this->usedIDcnt = 0;

	this->isInitialization = false;

	this->InitColorTab();

	// Moved in OnlineTracker::SetParams()
	//this->InitializeTrackletsContainters();

	this->InitializeMatrices(F_xy, Q_xy, Ps_xy, R_xy, H_xy, sym::MODEL_VECTOR::XY);
}


GMPHD_MAF::~GMPHD_MAF()
{
}
void GMPHD_MAF::Destory() {

	for (auto& ftr : this->allLiveReliables) {
		for (auto& tr : ftr) {
			tr.Destroy();
		}
		ftr.clear();
	}
	this->allLiveReliables.clear();

	for (auto& ftr : this->tracksbyID) {
		for (auto& tr : ftr.second) {
			tr.Destroy();
		}
		ftr.second.clear();
	}
	this->tracksbyID.clear();
}
/**
*	@brief
*	@details
*	@param iFrmCnt : frame number (const int&)
*	@param img : input image (const cv::Mat&)
*	@param dets : input detections and segmentations (const vector<BBDet>&)
*	@param out_tracks : tracking results (vector<BBTrk>& output)
*
*	@return the number of processed detections (int)
*
*/
int GMPHD_MAF::RunMOTS(const int& iFrmCnt, const cv::Mat& img, const vector<BBDet>& dets, vector<BBTrk>& out_tracks) {
	if (this->iTotalFrames == 0)
		cerr << "[ERROR] Set the number of total frames !!" << endl;

	if (this->sysFrmCnt == 0)
		this->InitImagesQueue(img.cols, img.rows);
	this->UpdateImageQueue(img, this->params.QUEUE_SIZE);

	const int MODEL_TYPE = sym::MODEL_VECTOR::XY; // sym::MODEL_VECTOR::XYRyx
	// Load Detection (truncating and filtering)
	std::vector<BBDet> detVec, detVecLow;

	int d = 0;
	for (const auto& in_det : dets) {
		if (in_det.confidence >= this->params.DET_MIN_CONF /*&& bbd.rec.height < this->frmHeight/2.0 */) {

			float confidence;
			if (in_det.confidence < 0.0)	confidence = 0.001; // 1.0 / nDets;
			else							confidence = in_det.confidence;

			BBDet det = in_det;
			det.fn = iFrmCnt;
			det.confidence = confidence;
			det.ratio_yx = (float)det.rec.y / (float)det.rec.x;

			if (det.segMask.cols > 0 && det.segMask.rows > 0) { // empty segment filtering for (BB2Seg+RRC)
				detVec.push_back(det);
			}
			d++;
		}
	}

	if (MERGE_DET_ON) { // 안하는게 좋네 어짜피 track level 에서 하는데
		std::vector<BBDet> mereged_dets = this->MergeDetInstances(detVec, true);
		detVec.clear();
		detVec = mereged_dets;
	}
	int nProcDets = detVec.size();
	// Normalize the Weights (Detection Scores)
	this->NormalizeWeight(detVec);

	/*-------------------- Stage 1. Detection-to-Track Association (D2TA) --------------------*/
	/// Init, Predict -> Data Association -> Update -> Pruning -> Merge
	if (!this->liveTrkVec.empty()) {
		// Predict
		this->PredictFrmWise(iFrmCnt, img, this->liveTrkVec, MODEL_TYPE);
		if (EXCLUDE_BBOX_OUT_OF_FRAME) {
			ArrangeTargetsVecsBatchesLiveLost();
		}
		// Empty Check Again
		if (this->liveTrkVec.size() == 0) {
			this->InitTracks(iFrmCnt, img, detVec, MODEL_TYPE);
		}
		else {
			// Data Association -> Update -> Pruning -> Merge
			this->DataAssocFrmWise(iFrmCnt, img, this->liveTrkVec, detVec, this->P_survive, MODEL_TYPE);
		}
	}
	else if (/*this->sysFrmCnt == 0 || */this->liveTrkVec.size() == 0) {
		// Init
		this->InitTracks(iFrmCnt, img, detVec, MODEL_TYPE);
	}
	// Arrange the targets which have been alive or not (live or lost)
	this->ArrangeTargetsVecsBatchesLiveLost();

	// Push the Tracking Results (live, lost) into the each Tracks Queue (low level)
	/// Keep only the tracking targets at now (except the loss targets)
	this->PushTargetsVecs2BatchesLiveLost();

	if (GMPHD_TRACKER_MODE) {

		// Tracklet Categorization (live or lost) for Track-to-Track Association (T2TA)
		// A Reliable Tracklet has the size (length) >= params.TRACK_MIN_SIZE.
		// A Unreliable Tracklet has the size < params.TRAK_MIN_SIZE.
		// Only the Reliable Tracklets are used for T2TA
		/// Put the re-arranged targets into tracklets according to ID, frame by frame
		/// Insert the re-arranged tracklets to tracklets map according to ID as a key
		this->CollectTracksbyID(this->tracksbyID, this->liveTrkVec);
		this->ClassifyTrackletReliability(iFrmCnt, this->tracksbyID, this->tracks_reliable, this->tracks_unreliable);

		vector<BBTrk> liveReliables, lostReliables, obss;
		this->ClassifyReliableTracklets2LiveLost(iFrmCnt, this->tracks_reliable, liveReliables, lostReliables, obss);

		/*-------------------- Stage 2. Track-to-Track Association (T2TA) --------------------*/
		/// Data Association -> Update Tracklets -> Occlusion Group Enerege Minimization (OGEM)
		if (!lostReliables.empty() && !obss.empty()/* !liveReliables.empty()*/) {

			cv::Mat img_latency = imgBatch[this->params.QUEUE_SIZE - 1 - this->params.FRAMES_DELAY_SIZE].clone();	// image for no latent association tracking
			//this->DataAssocTrkWise(iFrmCnt - this->params.FRAMES_DELAY_SIZE, img_latency, lostReliables, liveReliables, MODEL_TYPE);
			this->DataAssocTrkWise(iFrmCnt - this->params.FRAMES_DELAY_SIZE, img_latency, lostReliables, obss, MODEL_TYPE);

			//this->ArrangeRevivedTracklets(this->tracks_reliable, liveReliables); // it can be tracks_unreliabe, liveUnreliables
			this->ArrangeRevivedTracklets(this->tracks_reliable, obss);
			img_latency.release();
		}
		if (!obss.empty()) {
			for (auto &live_t2ta : obss) {
				liveReliables.push_back(live_t2ta);
			}
			//liveReliables.insert(std::end(liveReliables), std::begin(obss), std::end(obss));
		}

		// Tracklets' Containters Management (for practical implementation)
		if ((this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE > this->params.T2TA_MAX_INTERVAL) && (((this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE) % this->params.T2TA_MAX_INTERVAL) == 0)) {

			//cout << "this->tracksbyID: ";
			ClearOldEmptyTracklet(this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE, this->tracksbyID, this->params.T2TA_MAX_INTERVAL);
			//cout << "this->tracks_reliable: ";
			ClearOldEmptyTracklet(this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE, this->tracks_reliable, this->params.T2TA_MAX_INTERVAL);
			//cout << "this->tracks_unreliable: ";
			if (this->params.TRACK_MIN_SIZE > 1)
				ClearOldEmptyTracklet(this->sysFrmCnt - this->params.FRAMES_DELAY_SIZE, this->tracks_unreliable, this->params.T2TA_MAX_INTERVAL);
		}

		// Return the Tracking Results in frame iFrmCnt
		out_tracks = this->GetTrackingResults(iFrmCnt - this->params.FRAMES_DELAY_SIZE, liveReliables, MODEL_TYPE);

		liveReliables.clear();
		lostReliables.clear();
		obss.clear();
	}
	else {
		// Return the Tracking Results in frame iFrmCnt
		out_tracks = this->GetTrackingResults(iFrmCnt, this->liveTrkVec, MODEL_TYPE);
	}

	this->sysFrmCnt++;

	return nProcDets;
}
void GMPHD_MAF::InitTracks(const int& iFrmCnt, const cv::Mat& img, const vector<BBDet>& dets, const int& MODEL_TYPE) {
	std::vector<BBDet>::const_iterator iterD;
	for (iterD = dets.begin(); iterD != dets.end(); ++iterD)
	{
		int id = this->usedIDcnt++;
		if (IS_VEHICLE_ALL(this->trackObjType))		id = 2 * id + 1;
		else if (IS_PERSON_EVAL(this->trackObjType))	id = 2 * id;

		BBTrk bbt;
		bbt.isAlive = true;
		bbt.id = id;
		bbt.fn = iFrmCnt;
		bbt.rec = iterD->rec;
		bbt.depth = iterD->depth;
		bbt.rotation_y = iterD->rotation_y;
		bbt.ratio_yx = iterD->ratio_yx;
		bbt.vx = 0.0;
		bbt.vy = 0.0;
		bbt.vd = 0.0;
		bbt.vr = 0.0;
		bbt.weight = iterD->weight;
		bbt.conf = iterD->confidence;
		if (VIS_D2TA_DETAIL || VIS_T2TA_DETAIL) {
			cv::Rect objRec = this->RectExceptionHandling(this->frmWidth, this->frmHeight, iterD->rec);
			bbt.tmpl = img(objRec).clone();
			//cv::imshow(std::to_string(id).c_str(), bbt.tmpl);
			//cv::waitKey(0);
		}

		/*----MOTS Tracking Only-- - */
		//if (MOT_SEGMENTATION_ON)
		bbt.segMask = iterD->segMask.clone();

		/*----MOTS Tracking Only-- - */
		bbt.det_id = iterD->id;
		bbt.det_confidence = iterD->confidence;

		if (SOT_TRACK_OPT == SOT_USE_KCF_TRACKER)				bbt.papp = new KCFTracker();
		//else if (SOT_TRACK_OPT == SOT_USE_SIAMRPN_TRACKER)	bbt.papp = new SiamRPNTracker();
		//else if (SOT_TRACK_OPT == SOT_USE_DASIAMRPN_TRACKER)	bbt.papp = new DaSiamTracker();

		if (this->params.SAF_D2TA_MODE > sym::AFFINITY_OPT::GMPHD || this->params.SAF_T2TA_MODE > sym::AFFINITY_OPT::GMPHD)
			bbt.papp.init(img, bbt.rec, bbt.segMask, this->params.SAF_MASK_D2TA || this->params.SAF_MASK_T2TA);

		if (MODEL_TYPE == sym::MODEL_VECTOR::XY ) {
			bbt.cov = (cv::Mat_<double>(sym::DIMS_STATE[MODEL_TYPE], sym::DIMS_STATE[MODEL_TYPE]) << \
				VAR_X, 0, 0, 0, \
				0, VAR_Y, 0, 0,
				0, 0, VAR_X_VEL, 0,
				0, 0, 0, VAR_Y_VEL);
		}

		this->liveTrkVec.push_back(bbt);
	}
}
void GMPHD_MAF::PredictFrmWise(int iFrmCnt, const cv::Mat& img, vector<BBTrk>& stats, const int MODEL_TYPE)
{

	vector<BBTrk>::iterator iter;

	for (iter = stats.begin(); iter < stats.end(); ++iter) {

		iter->fn = iFrmCnt;

		if (MODEL_TYPE == sym::MODEL_VECTOR::XY)
		{
			// copy current stat bounding box and velocity info to previous stat
			iter->rec.x += iter->vx; // Same as xk|k-1=F*xk-1
			iter->rec.y += iter->vy;

			cv::Mat Ps_temp = this->Q_xy + this->F_xy*iter->cov*this->F_xy.t();

			// make covariance matrix diagonal
			//Ps_temp.copyTo(iter->cov); 
			iter->cov.at<double>(0, 0) = Ps_temp.at<double>(0, 0);
			iter->cov.at<double>(1, 1) = Ps_temp.at<double>(1, 1);
			iter->cov.at<double>(2, 2) = Ps_temp.at<double>(2, 2);
			iter->cov.at<double>(3, 3) = Ps_temp.at<double>(3, 3);

			bool is_out = false;
			if (EXCLUDE_BBOX_OUT_OF_FRAME) {
				cv::Rect future_rec = iter->rec;

				future_rec.x += iter->vx;
				future_rec.y += iter->vy;

				is_out = this->IsOutOfFrame(future_rec, this->frmWidth, this->frmHeight);
				if (is_out) iter->isAlive = false;

			}
			if (!is_out) {
				if (this->params.SAF_D2TA_MODE > sym::AFFINITY_OPT::GMPHD) {
					float tConf;
					cv::Mat tMat;
					//iter->papp.update(img);
					iter->papp.update(img, tConf, tMat, iter->rec, true, iter->segMask, this->params.SAF_MASK_D2TA);
					tMat.release();
				}
			}
		}	
	}
}
void GMPHD_MAF::DataAssocFrmWise(int iFrmCnt, const cv::Mat& img, vector<BBTrk>& stats, vector<BBDet>& obss, double P_survive, const int &MODEL_TYPE) {

	//FILE* fp_s2ta_affs[2] ;
	//string dirHome = "res\\KITTI\\train\\";
	//string fileTags[2] = { "_s2ta_not_norm","_s2ta_norm_maf"};
	//string objName = sym::OBJECT_STRINGS[this->trackObjType];
	//string filePaths[2];

	//for (int f = 0; f < 2; ++f) {
	//	filePaths[f] = dirHome + this->seqName + "_" + objName + fileTags[f] + ".txt";
	//	if(iFrmCnt ==this->params.FRAME_OFFSET+1) // after init
	//		fp_s2ta_affs[f] = fopen(filePaths[f].c_str(), "w+");
	//	else
	//		fp_s2ta_affs[f] = fopen(filePaths[f].c_str(), "a");
	//}


	int nObs = obss.size();
	int mStats = stats.size();
	vector<vector<double>> m_cost;
	m_cost.resize(mStats, vector<double>(nObs, 0.0));

	vector<vector<double>> kcf_cost, kcf_cost_not_norm, gmphd_cost;
	if (this->params.SAF_D2TA_MODE > sym::AFFINITY_OPT::GMPHD) {
		kcf_cost.resize(mStats, vector<double>(nObs, 0.0));
		kcf_cost_not_norm.resize(mStats, vector<double>(nObs, 0.0));
		gmphd_cost.resize(mStats, vector<double>(nObs, 0.0));
	}

	vector<vector<double>> q_values;
	q_values.resize(nObs, std::vector<double>(mStats, 0.0));
	vector<vector<float>> ry_norms;
	ry_norms.resize(nObs, std::vector<float>(mStats, 0.0));
	vector<vector<float>> IOUs;
	IOUs.resize(nObs, std::vector<float>(mStats, 0.0));
	vector<vector<BBTrk>> stats_matrix; // It can boost parallel processing?
	stats_matrix.resize(nObs, std::vector<BBTrk>(mStats, BBTrk()));
	for (int r = 0; r < nObs; ++r) {
		for (int c = 0; c < mStats; ++c)
		{
			stats.at(c).CopyTo(stats_matrix[r][c]);
		}
	}

	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for (int r = 0; r < nObs; ++r){
		for (int c = 0; c < stats_matrix[r].size(); ++c) {
			// Calculate the Affinity between detection (observations) and tracking (states)
			q_values[r][c] = FrameWiseAffinity(obss[r], stats_matrix[r][c], MODEL_TYPE);

			if (ASSOCIATION_STAGE_1_GATING_ON) {
				if (q_values[r][c] < Q_TH_LOW_20) {
					q_values[r][c] = 0.0;

					if (MODEL_TYPE == sym::MODEL_VECTOR::XY ) {
						float overlapping_ratio = 0.0;
						if (this->params.MERGE_METRIC == MERGE_METRIC_IOU)
							overlapping_ratio = this->CalcIOU(obss[r].rec, stats_matrix[r][c].rec);
						else if(this->params.MERGE_METRIC == MERGE_METRIC_SIOA)
							overlapping_ratio = this->CalcSIOA(obss[r].rec, stats_matrix[r][c].rec);
						else {
							cv::Mat m_uni;
							overlapping_ratio = this->CalcMIOU(obss[r].rec, obss[r].segMask, stats_matrix[r][c].rec, stats_matrix[r][c].segMask, m_uni);
							m_uni.release();
						}

						if (overlapping_ratio >= this->params.MERGE_RATIO_THRESHOLD) // threshold
							q_values[r][c] = obss[r].weight*overlapping_ratio;
					}
				}
			}
		}
	}
	);

	// Calculate States' Weights by GMPHD filtering with q values
	// Then the Cost Matrix is filled with States's Weights to solve Assignment Problem.
	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for (int r = 0; r < nObs; ++r){
		int nStats = stats_matrix[r].size();
		double denominator = 0.0;
		for (int c = 0; c < stats_matrix[r].size(); ++c) {		
			denominator += (stats_matrix[r][c].weight * q_values[r][c]);
		}

		for (int c = 0; c < stats_matrix[r].size(); ++c) {

			double numerator =  /*P_detection*/stats_matrix[r][c].weight*q_values[r][c];

			if (numerator > DBL_MAX || numerator < DBL_MIN) {
				stats_matrix[r][c].weight = 0.0;
				// numerator = 0.0;
			}
			else
				stats_matrix[r][c].weight = numerator / denominator;							// (19), numerator(분자), denominator(분모)

			// Scaling the affinity value to Integer
			if (stats_matrix[r][c].weight > 0.0) {
				if ((double)(/*stats_matrix[r][c].weight*/numerator) < (double)(FLT_MIN)) {
					std::cerr << "[" << iFrmCnt << "] weight(FW) < FLT_MIN" << std::endl;
					if (!AFFINITY_COST_L1_OR_L2_NORM_FW)	m_cost[c][r] = -1;
					else									m_cost[c][r] = 10000;	// 어짜피 아주작은 값이 나올만한 애들은 GATING 에서 날아갈테니 같은 10000
				}
				else {
					if (!AFFINITY_COST_L1_OR_L2_NORM_FW)	m_cost[c][r] = -1.0*Q_TH_LOW_10_INVERSE * numerator;
					else									m_cost[c][r] = -100.0*log(numerator); //log2l((double)numerator);
				}
			}
			else {
				if (!AFFINITY_COST_L1_OR_L2_NORM_FW)			m_cost[c][r] = 0;
				else											m_cost[c][r] = 10000;	// upper bound, double 로 표현 못하는값 -> association 후보 조차 제외
			}
			/*printf("(FW) Obs%d(%d,%d,%d,%d,%.4f) -> Stat%d(id%d):%.lf (n:%lf, q:%.20lf, W':%.5f)\n",\
			r, obss[r].rec.x, obss[r].rec.y, obss[r].rec.width, obss[r].rec.height, obss[r].weight,
			c, stats_matrix[r][c].id, m_cost[c][r], numerator, q_values[r][c], stats_matrix[r][c].weight);*/
		}
		//printf("\n");
	}
	);

	// KCF features based Affinity Calculation
	if (this->params.SAF_D2TA_MODE > sym::AFFINITY_OPT::GMPHD) {
		// KCF Vis Window Setting
		string winTitleKCF = "MOT with SOT"; // (ID: " + boost::to_string(id) +")";
		cv::Mat frameVis;

		int FW = this->frmWidth;
		int FH = this->frmHeight;
		if (VISUALIZATION_RESIZE_ON) {
			FW *= (2.0 / 3.0);
			FH *= (2.0 / 3.0);
		}

		if (VIS_D2TA_DETAIL) {

			frameVis = img.clone();
		}
		//
		if (mStats > 0) {

			string winTitleDA = "States x Observations (D2TA)"; // (ID: " + boost::to_string(id) +")";

			int margin = 40;
			int cellWH = 140;
			cv::Mat canvasDA;

			if (VIS_D2TA_DETAIL) {
				cv::namedWindow(winTitleDA);	cv::moveWindow(winTitleDA, FW, 0);
				canvasDA = cv::Mat(cellWH * (mStats)+margin, cellWH * (nObs + 1) + margin, CV_8UC3, cv::Scalar(200, 200, 200));
			}

			vector<BBDet> kcf_dummy;
			vector<vector<BBTrk>> stats_dummy;
			Concurrency::parallel_for(0, mStats, [&](int c) {
				//for (int c = 0; c < mStats; ++c) {
					//cv::Mat framePrev = this->imgBatch[this->params.QUEUE_SIZE-2];
				cv::Mat frameProc = img.clone();

				int id = stats[c].id;
				if (VIS_D2TA_DETAIL) {
					cv::Mat statResize;
					cv::Rect statRecInFrame = this->RectExceptionHandling(this->frmWidth, this->frmHeight, stats[c].rec);

					if (statRecInFrame.width < 1 || statRecInFrame.height < 1)
						statResize = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
					else
						cv::resize(frameProc(statRecInFrame), statResize, cv::Size(100, 100)); // predicted state

					statResize.copyTo(canvasDA(cv::Rect(margin, margin + cellWH * c, 100, 100)));

					char strIDConf[64];
					sprintf_s(strIDConf, "ID%d (%.3f)", id, stats[c].conf);
					cv::putText(canvasDA, string(strIDConf), cv::Point(margin, margin / 2 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.5, this->color_tab[(id / 2) % (MAX_OBJECTS - 1)], 2);
				}

				if (VIS_D2TA_DETAIL)
					printf("[%d] Affinity between ID%d (%d,%d,%d,%d) and %d Detections..\n", iFrmCnt, id, stats[c].rec.x, stats[c].rec.y, stats[c].rec.width, stats[c].rec.height, nObs);

				if (nObs > 0) {

					for (int r = 0; r < nObs; ++r) {
						float objConf = obss[r].confidence;
						float iou = this->CalcIOU(obss[r].rec, stats_matrix[r][c].rec);
						IOUs[r][c] = iou;

						cv::Rect obsRecInFrame = this->RectExceptionHandling(this->frmWidth, this->frmHeight, obss[r].rec);
						//cout << "(a)";
						cv::Mat confMap;
						float confProb;
						if (this->params.GATE_D2TA_ON) {

							//if (iou > 0) printf("[%d] ID%d-Det%d (%.3f-%.3f)\n", this->sysFrmCnt, id, r, iou, m_cost[c][r]);

							if (iou < 0.1) {
								confProb = 0.99;
								//cv::rectangle(frameVis, obss[r].rec, cv::Scalar(0, 0, 0), -1);
								if (VIS_D2TA_DETAIL)
									confMap = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
							}
							else {
								cv::Rect res = stats_matrix[r][c].papp.update(frameProc, confProb, confMap, obss[r].rec, true, obss[r].segMask, this->params.SAF_MASK_D2TA);
								//cv::addWeighted(frameVis(res), 0.5, confMap, 0.5, 0.0, frameVis(res));
								//confMap.release();

								//if (objConf >= 0.95) {
								//	if (!confMap.empty()) confMap.release();

								//	stats_matrix[r][c].papp.init(obss[r].rec, frameProc);
								//	cv::Rect res = stats_matrix[r][c].papp.update(obss[r].rec, frameProc, confMap, confProb);
								//	//cv::addWeighted(frameVis(res), 0.5, confMap, 0.5, 0.0, frameVis(res));
								//	//confMap.release();
								//}

								//DrawTrkBBS(frameVis, obss[r].rec, this->color_tab[id % (MAX_OBJECTS - 1)], 2, id, 0.5, "2D", false);
								if (res.width < 1 || res.height < 1) {
									//confProb = 0.99;
									if (VIS_D2TA_DETAIL)
										confMap = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
								}
							}
						}
						else {
							cv::Rect res = stats_matrix[r][c].papp.update(frameProc, confProb, confMap, obss[r].rec, true, obss[r].segMask, this->params.SAF_MASK_D2TA);

							if (res.width < 1 || res.height < 1) {
								//confProb = 0.99;
								if (VIS_D2TA_DETAIL)
									confMap = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
							}
						}
						cv::Mat obsResize;
						cv::Rect obsDAcell;
						cv::Mat confMapResize;
						if (VIS_D2TA_DETAIL) {
							obsDAcell = cv::Rect(margin + cellWH * (r + 1), margin + cellWH * c, 100, 100);

							if (obsRecInFrame.width < 1 || obsRecInFrame.height < 1)
								obsResize = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
							else
								cv::resize(frameProc(obsRecInFrame), obsResize, cv::Size(100, 100));

							obsResize.copyTo(canvasDA(obsDAcell));
							cv::resize(confMap, confMapResize, cv::Size(100, 100));
						}

						char scores[256];
						double score_gmphd_temp;// = pow(10, m_cost[c][r] / (-100.0));
						if (m_cost[c][r] < 10000) {
							gmphd_cost[c][r] = pow(e_8, m_cost[c][r] / (-100.0));
							score_gmphd_temp = gmphd_cost[c][r];
						}
						else {
							gmphd_cost[c][r] = 0.0;
							score_gmphd_temp = gmphd_cost[c][r];
						}

						float score_kcf_temp; // = (1.0 - confProb)/**objConf*/;
						if (SOT_TRACK_OPT == SOT_USE_KCF_TRACKER)	score_kcf_temp = 1.0 - confProb;
						else										score_kcf_temp = confProb;

						kcf_cost[c][r] = score_kcf_temp;

						float CONF_LOWER_THRESH = this->params.KCF_BOUNDS_D2TA[0];
						float CONF_UPPER_THRESH = this->params.KCF_BOUNDS_D2TA[1];

						if (this->params.SAF_D2TA_MODE == sym::AFFINITY_OPT::KCF) {

							if (score_kcf_temp >= CONF_UPPER_THRESH)
								m_cost[c][r] = -100 * log(score_kcf_temp);
							else if (score_kcf_temp < CONF_UPPER_THRESH && score_kcf_temp >= CONF_LOWER_THRESH && iou >= 0.1)
								m_cost[c][r] = -100 * log(score_kcf_temp);
							else
								m_cost[c][r] = 10000;
						}

						// 
						//kcf_cost[c][r] = scaleComb * -100 * log2f(1.0 - confProb);
						// m_cost[c][r] = m_cost[c][r] + kcf_cost[c][r];

						/*if (confProb > CONF_THRESH)
							m_cost[c][r] = 10000;
						else*/
						//	m_cost[c][r] = -100.0*log2l((double)(score_gmphd_temp*(1.0 - confProb)));
						if (VIS_D2TA_DETAIL) {
							printf("	Det%d (%d,%d,%d,%d,%.5f)-IOU(%.3f), Cost(GMPHD:%s, KCF:%.5f)\n",
								r, obss[r].rec.x, obss[r].rec.y, obss[r].rec.width, obss[r].rec.height, objConf,
								iou, this->to_str(score_gmphd_temp).c_str(), score_kcf_temp);
						}
						if (m_cost[c][r] < 10000) {
							if (VIS_D2TA_DETAIL) {
								cv::putText(canvasDA, this->to_str(score_gmphd_temp).c_str(), cv::Point(margin + cellWH * (r + 1), margin - 30 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
							}
						}
						else {
							if (VIS_D2TA_DETAIL)
								cv::putText(canvasDA, "INF", cv::Point(margin + cellWH * (r + 1), margin - 30 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
						}
						if (VIS_D2TA_DETAIL) {
							cv::addWeighted(canvasDA(obsDAcell), 0.6, confMapResize, 0.4, 0.0, canvasDA(obsDAcell));
							cv::putText(canvasDA, this->to_str(score_kcf_temp).c_str(), cv::Point(margin + cellWH * (r + 1), margin - 15 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, r), 1);
						}

						if (!confMap.empty())		confMap.release();
						if (!confMapResize.empty()) confMapResize.release();
					}

				}
				frameProc.release();
			}
			);
			// Fusing GMPHD and KCF scores
			float CONF_LOWER_THRESH = this->params.KCF_BOUNDS_D2TA[0];
			float CONF_UPPER_THRESH = this->params.KCF_BOUNDS_D2TA[1];
			float IOU_LOWER_THRESH = this->params.IOU_BOUNDS_D2TA[0]; // 0.1: loose (moving cam) 0.4: static cam (지금까지 다 이렇게한것만 제출함)
			float IOU_UPPER_THRESH = this->params.IOU_BOUNDS_D2TA[1];

			if (this->params.SAF_D2TA_MODE == sym::AFFINITY_OPT::MAF) {
				// (1) Min-Max Normalization (D2TA)
				for (int c = 0; c < mStats; ++c)
					for (int r = 0; r < nObs; ++r)
						kcf_cost_not_norm[c][r] = kcf_cost[c][r];

				/*for (int c = 0; c < mStats; ++c){
					for (int r = 0; r < nObs; ++r) {

						if (gmphd_cost[c][r] > 0)
							fprintf_s(fp_s2ta_affs[0], "%d\t%.10lf\t%.30lf\n", iFrmCnt, kcf_cost[c][r], gmphd_cost[c][r]);
					}
				}
				fclose(fp_s2ta_affs[0]);*/

				this->FusionMinMaxNorm(nObs, mStats, m_cost, gmphd_cost, kcf_cost, IOUs, CONF_LOWER_THRESH, CONF_UPPER_THRESH, IOU_LOWER_THRESH, IOU_UPPER_THRESH);
				// (2) Z-Score Normalization
				//ms = this->FusionZScoreNorm(nObs, mStats, m_cost, gmphd_cost, kcf_cost, IOUs, 0.95, CONF_LOWER_THRESH, CONF_UPPER_THRESH, IOU_LOWER_THRESH, IOU_UPPER_THRESH);
				// (3) TanH Normalization
				//this->FusionTanHNorm(nObs, mStats, m_cost, gmphd_cost, kcf_cost, IOUs, CONF_LOWER_THRESH, CONF_UPPER_THRESH, IOU_LOWER_THRESH, IOU_UPPER_THRESH);

				/*for (int c = 0; c < mStats; ++c) {
					for (int r = 0; r < nObs; ++r) {
						double m_cost_norm_tmp = 0.0;
						if (m_cost[c][r] == 10000) m_cost_norm_tmp = 0.0;
						else m_cost_norm_tmp = pow(e, m_cost[c][r] / -100.0);

						if (gmphd_cost[c][r] > 0)
							fprintf_s(fp_s2ta_affs[1], "%d\t%.10lf\t%.30lf\t%.30lf\n", iFrmCnt, kcf_cost[c][r], gmphd_cost[c][r], m_cost_norm_tmp);
					}
				}

				fclose(fp_s2ta_affs[1]);*/
			}

			//cv::waitKey();
			//);
			if (VIS_D2TA_DETAIL) {
				cv::Mat frameVisResize;
				this->cvPrintVec2Vec(m_cost, "COST_D2TA");

				cv::imshow(winTitleDA, canvasDA);
				cv::waitKey(1);
				canvasDA.release();
				if (VISUALIZATION_RESIZE_ON)
					frameVisResize.release();
				frameVis.release();
			}
		}
	}

	// Build a Cost Matrix for Detection-to-Track Association
	double min_cost = INT_MAX, max_cost = 0;
	if (!AFFINITY_COST_L1_OR_L2_NORM_FW) {

		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {
				if (min_cost > m_cost[c][r]) min_cost = m_cost[c][r];
				if (max_cost < m_cost[c][r]) max_cost = m_cost[c][r];
			}
		}
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {
				m_cost[c][r] = m_cost[c][r] - min_cost + 1;
			}
		}
		max_cost = 1 - min_cost;
	}
	else {
		max_cost = 10000;
		int r_min = -1, c_min = -1;

	}

	// Solving data association problem (=linear assignment problem) (find the min cost assignment pairs)
	std::vector<vector<int>> assigns_sparse, assigns_dense;
	vector<vector<double>> m_cost_d;
	vector<vector<cv::Vec2i>> trans_indices = this->cvtSparseMatrix2Dense(m_cost, m_cost_d/*, max_cost*/);

	if (m_cost_d.empty() || m_cost_d[0].empty()) {
		assigns_sparse = this->HungarianMethod(m_cost, obss.size(), stats.size());
	}
	else {
		assigns_sparse.resize(obss.size(), std::vector<int>(stats.size(), 0));

		assigns_dense = this->HungarianMethod(m_cost_d, trans_indices[0].size(), trans_indices.size());
		for (int c = 0; c < trans_indices.size(); ++c) {		// states
			for (int r = 0; r < trans_indices[c].size(); ++r) { // observations
				int rt = trans_indices[c][r](1);
				int ct = trans_indices[c][r](0);
				assigns_sparse[rt][ct] = assigns_dense[r][c];
			}
		}
	}

	//if(nObs>0)
	//	this->cvPrintVec2Vec(m_cost, "D2TA_COST_GMPHD-REID");

	bool *isAssignedStats = new bool[stats.size()];	memset(isAssignedStats, 0, stats.size());
	bool *isAssignedObs = new bool[obss.size()];	memset(isAssignedObs, 0, obss.size());
	int *isAssignedObsIDs = new int[stats.size()];	memset(isAssignedObsIDs, 0, stats.size()); // only used in LB_ASSOCIATION

	// Update
	for (int c = 0; c < stats.size(); ++c) {

		for (int r = 0; r < obss.size(); ++r) {
			if (assigns_sparse[r][c] == 1 && m_cost[c][r] < max_cost) {

				// Velocity Update
				float vx_t_1 = stats[c].vx;
				float vy_t_1 = stats[c].vy;
				//float vd_t_1 = stats[c].vd;
				float vr_t_1 = stats[c].vr;
				float vx_t, vy_t, vd_t, vr_t;

				if (MODEL_TYPE == sym::MODEL_VECTOR::XYRyx) {
					vr_t = (obss[r].ratio_yx) - (stats[c].ratio_yx);
				}
				else {

					vx_t = (obss[r].rec.x + obss[r].rec.width / 2.0) - (stats[c].rec.x + stats[c].rec.width / 2.0);
					vy_t = (obss[r].rec.y + obss[r].rec.height / 2.0) - (stats[c].rec.y + stats[c].rec.height / 2.0);

					stats[c].rec = obss[r].rec;
					//if (MOT_SEGMENTATION_ON)
					stats[c].segMask = obss[r].segMask.clone();
				}

				stats[c].vx = vx_t_1 * this->params.VEL_UP_ALPHA + vx_t * (1.0 - this->params.VEL_UP_ALPHA);
				stats[c].vy = vy_t_1 * this->params.VEL_UP_ALPHA + vy_t * (1.0 - this->params.VEL_UP_ALPHA);
				//stats[c].vd = vd_t_1 * this->params.VEL_UP_ALPHA + vd_t * (1.0 - this->params.VEL_UP_ALPHA);
				stats[c].vr = vr_t_1 * this->params.VEL_UP_ALPHA + vr_t * (1.0 - this->params.VEL_UP_ALPHA);

				stats[c].weight = stats_matrix[r][c].weight;
				if (obss[r].confidence >= 0.95 && stats[c].conf < obss[r].confidence)
					stats[c].conf = obss[r].confidence;
				else
					stats[c].conf = stats[c].conf*CONFIDENCE_UPDATE_ALPHA + obss[r].confidence*(1.0 - CONFIDENCE_UPDATE_ALPHA);

				// Covariance Matrix Update
				stats_matrix[r][c].cov.copyTo(stats[c].cov);

				// KCF Features Update
				// cost 계산 부분에서 0.25 이상이면 init 아니면 update 를 해서 stats_matrix[r][c] 에 가지고 있도록 하는게 일관성이 있겠다.
				/*if (stats[c].conf > 0.95)
				{
					stats_matrix[r][c].papp.init(obss[r].rec, img);
				}*/
				if (this->params.SAF_D2TA_MODE > sym::AFFINITY_OPT::GMPHD) {

					cv::Mat frameProc = img.clone();
					float confProb;
					bool update_on = true;
					if (APPEARANCE_STRICT_UPDATE_ON) {
						//update_on = (obss[r].confidence >= 0.95);
						update_on = (kcf_cost_not_norm[c][r] >= APPEARANCE_UPDATE_DET_TH);
						// 여기가 아니라, kcf score 값이 일정 (0.85) 이하면, 이전것으로 되돌리는게 맞는것 같다.

					}

					if (update_on) { // newly init
						cv::Mat frameProc = img.clone();
						cv::Mat confMap;

						/*printf("obs(%d,%d,%d,%d,m%dx%d)-",
							obss[r].rec.x, obss[r].rec.y, obss[r].rec.width, obss[r].rec.height,
							obss[r].segMask.cols, obss[r].segMask.rows);*/

						stats_matrix[r][c].papp.init(frameProc, obss[r].rec, obss[r].segMask, this->params.SAF_MASK_D2TA);
						// line 2271, 2254 obss 에 추가해주는 dummy 에 segMask 복사 안해서 resize src size width, height >0 에서 걸림
						cv::Rect res = stats_matrix[r][c].papp.update(frameProc, confProb, confMap, obss[r].rec, true, obss[r].segMask, this->params.SAF_MASK_D2TA);

						stats[c].papp = stats_matrix[r][c].papp;

						if (!stats[c].tmpl.empty()) stats[c].tmpl.release();
						stats[c].tmpl = frameProc(this->RectExceptionHandling(this->frmWidth, this->frmHeight, obss[r].rec)).clone();

						confMap.release();
						frameProc.release();
					}
					stats[c].papp = stats_matrix[r][c].papp;
				}

				/*----MOTS Tracking Only-- - */
				stats[c].det_id = obss[r].id;
				stats[c].det_confidence = obss[r].confidence;
				/*----MOTS Tracking Only---*/

				isAssignedStats[c] = true;
				isAssignedObs[r] = true;
				isAssignedObsIDs[c] = obss[r].id; // only used in LB_ASSOCIATION
				break;
			}
			isAssignedStats[c] = false;
		}
		stats[c].isAlive = isAssignedStats[c];
	}

	// Weight Normalization after GMPHD association process
	double sumWeight = 0.0;
	for (int c = 0; c < stats.size(); ++c) {
		sumWeight += stats[c].weight;
	}
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			stats[c].weight /= sumWeight;

		}
	}

	vector<int> newTracks;
	for (int r = 0; r < obss.size(); ++r) {
		if (!isAssignedObs[r]) {
			newTracks.push_back(r);

			int id_new = this->usedIDcnt++;
			if (IS_VEHICLE_ALL(this->trackObjType))			id_new = 2 * id_new + 1;
			else if (IS_PERSON_EVAL(this->trackObjType))	id_new = 2 * id_new;

			BBTrk newTrk;
			newTrk.fn = iFrmCnt;
			newTrk.id = id_new;
			newTrk.rec = obss[r].rec;
			newTrk.isAlive = true;
			newTrk.vx = 0.0;
			newTrk.vy = 0.0;
			newTrk.vd = 0.0;
			newTrk.vr = 0.0;
			newTrk.weight = obss[r].weight;
			newTrk.conf = obss[r].confidence;
			newTrk.rotation_y = obss[r].rotation_y;
			newTrk.ratio_yx = obss[r].ratio_yx;

			/*----MOTS Tracking Only-- - */
			newTrk.det_id = obss[r].id;
			newTrk.det_confidence = obss[r].confidence;
			/*----MOTS Tracking Only---*/

			//printf("[%d] a new track ID%d(%d,%d,%d,%d)\n", iFrmCnt, newTrk.id, newTrk.rec.x, newTrk.rec.y, newTrk.rec.width, newTrk.rec.height);

			//if (MOT_SEGMENTATION_ON)
			newTrk.segMask = obss[r].segMask.clone();

			// KCF init
			if (this->params.SAF_D2TA_MODE > sym::AFFINITY_OPT::GMPHD) {

				newTrk.papp.init(img, obss[r].rec, obss[r].segMask, this->params.SAF_MASK_D2TA);
				newTrk.tmpl = img(this->RectExceptionHandling(this->frmWidth, this->frmHeight, obss[r].rec)).clone();
			}

			if (MODEL_TYPE == sym::MODEL_VECTOR::XY) {
				newTrk.cov = (cv::Mat_<double>(sym::DIMS_STATE[MODEL_TYPE], sym::DIMS_STATE[MODEL_TYPE]) << \
					VAR_X, 0, 0, 0, \
					0, VAR_Y, 0, 0,
					0, 0, VAR_X_VEL, 0,
					0, 0, 0, VAR_Y_VEL);
			}

			stats.push_back(newTrk);
		}
	}
	if (MODEL_TYPE == sym::MODEL_VECTOR::XY) {
		if (MERGE_TRACK_ON) { // post-processing to prevent overlapped segments after D2TA
			stats = this->MergeTrkInstances(stats, 0.1);
		}
	}

	// Weight Normalization After Birth Processing
	sumWeight = 0.0;
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			sumWeight += stats[c].weight;
		}
	}
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			stats[c].weight /= sumWeight;
		}
	}

	/// Memory Deallocation
	delete[]isAssignedStats;
	delete[]isAssignedObs;
	delete[]isAssignedObsIDs;
}
float GMPHD_MAF::FrameWiseAffinity(BBDet ob, BBTrk &stat_temp, const int MODEL_TYPE) {

	const int dims_obs = sym::DIMS_OBS[MODEL_TYPE];
	const int dims_stat = sym::DIMS_STATE[MODEL_TYPE];

	if (MODEL_TYPE == sym::MODEL_VECTOR::XY) {
		// Bounding box size contraint
		if ((stat_temp.rec.area() >= ob.rec.area() * SIZE_CONSTRAINT_RATIO) || (stat_temp.rec.area() * SIZE_CONSTRAINT_RATIO <= ob.rec.area())) return 0.0;

		// Bounding box location contraint(gating)
		if (stat_temp.rec.area() >= ob.rec.area()) {
			if ((stat_temp.rec & ob.rec).area() < ob.rec.area() / 2) return 0.0;
		}
		else {
			if ((stat_temp.rec & ob.rec).area() < stat_temp.rec.area() / 2) return 0.0;
		}
	}

	// Step2: Update each Gaussian for every observation
	// find the observation which makes the Gaussian's weight into maximum among every observation

	// Step 2: Update phase1
	double q_value = 0.0;

	cv::Mat K(dims_stat, dims_obs, CV_64FC1);

	cv::Mat z_cov_dbl(dims_obs, dims_obs, CV_64FC1);
	cv::Mat z_cov_flt(dims_obs, dims_obs, CV_32FC1);

	cv::Mat Ps_temp(dims_stat, dims_stat, CV_64FC1);

	cv::Mat z_temp(dims_obs, 1, CV_32FC1);

	cv::Mat mean_obs(dims_obs, 1, CV_32FC1);

	// (20) Make the Mean Vector (cv::Mat) from the state (BBTrk)
	// (23) z_cov_temp = H*Ps*H.t() + R;
	// (22-a) K = Ps*H.t()*z_cov_temp.inv(DECOMP_SVD);
	// (22-b) Ps_temp = Ps - K*H*Ps;
	// Make the observation vector (cv::Mat) from the observation (BBDet)
	if (MODEL_TYPE == sym::MODEL_VECTOR::XY) {
		mean_obs.at<float>(0, 0) = (float)stat_temp.rec.x + (float)stat_temp.rec.width / 2.0;
		mean_obs.at<float>(1, 0) = (float)stat_temp.rec.y + (float)stat_temp.rec.height / 2.0;

		z_cov_dbl = this->H_xy*stat_temp.cov*this->H_xy.t() + this->R_xy;
		K = stat_temp.cov*this->H_xy.t()*z_cov_dbl.inv(cv::DECOMP_SVD);
		Ps_temp = stat_temp.cov - K * this->H_xy*stat_temp.cov;
		z_temp = (cv::Mat_<float>(dims_obs, 1) << ob.rec.x + (float)ob.rec.width / 2.0, ob.rec.y + (float)ob.rec.height / 2.0);
	}

	// Update the Covariance Matrix
	Ps_temp.copyTo(stat_temp.cov);

	// CV_64FC1 to CV_32FC1
	for (int r = 0; r < z_cov_flt.rows; r++)
		for (int c = 0; c < z_cov_flt.cols; c++)
			z_cov_flt.at<float>(r, c) = z_cov_dbl.at<double>(r, c);

	q_value = this->CalcGaussianProbability(dims_obs, z_temp, mean_obs, z_cov_flt);

	if (q_value < FLT_MIN) q_value = 0.0;
	return q_value;
}
float GMPHD_MAF::TrackletWiseAffinity(BBTrk &stat_pred, const BBTrk& obs, const int MODEL_TYPE) {

	const int dims_obs = sym::DIMS_OBS[MODEL_TYPE];
	const int dims_stat = sym::DIMS_STATE[MODEL_TYPE];

	if (MODEL_TYPE == sym::MODEL_VECTOR::XY) {
		// Bounding box size contraint
		if ((stat_pred.rec.area() >= obs.rec.area() * SIZE_CONSTRAINT_RATIO) || (stat_pred.rec.area() * SIZE_CONSTRAINT_RATIO <= obs.rec.area())) return 0.0;

		// Bounding box location contraint(gating)
		if (stat_pred.rec.area() >= obs.rec.area()) {
			if ((stat_pred.rec & obs.rec).area() <= 0 /*obs.rec.area() / 4*/) return 0.0;
		}
		else {
			if ((stat_pred.rec & obs.rec).area() <= 0  /*stat_pred.rec.area() / 4*/) return 0.0;
		}
	}
	// Step2: Update each Gaussian for every observation
	// find the observation which makes the Gaussian's weight into maximum among every observation

	// Step 2: Update phase1
	double q_value = 0.0;

	cv::Mat K(dims_stat, dims_obs, CV_64FC1);

	cv::Mat mean_obs(dims_obs, 1, CV_32FC1);

	cv::Mat z_cov_dbl(dims_obs, dims_obs, CV_64FC1);
	cv::Mat z_cov_flt(dims_obs, dims_obs, CV_32FC1);
	cv::Mat Ps_temp(dims_stat, dims_stat, CV_64FC1);

	cv::Mat z_temp(dims_obs, 1, CV_32FC1);

	// (20) Make the Mean Vector (cv::Mat) from the state (BBTrk)
	// (23) z_cov_temp = H*Ps*H.t() + R;
	// (22-a) K = Ps*H.t()*z_cov_temp.inv(DECOMP_SVD);
	// (22-b) Ps_temp = Ps - K*H*Ps;
	// Make the observation vector (cv::Mat) from the observation (BBTrk)
	if (MODEL_TYPE == sym::MODEL_VECTOR::XY ) {
		mean_obs.at<float>(0, 0) = (float)stat_pred.rec.x + (float)stat_pred.rec.width / 2.0;
		mean_obs.at<float>(1, 0) = (float)stat_pred.rec.y + (float)stat_pred.rec.height / 2.0;

		z_cov_dbl = this->H_xy*stat_pred.cov*this->H_xy.t() + this->R_xy;
		K = stat_pred.cov*this->H_xy.t()*z_cov_dbl.inv(cv::DECOMP_SVD);
		Ps_temp = stat_pred.cov - K * this->H_xy*stat_pred.cov;
		z_temp = (cv::Mat_<float>(dims_obs, 1) << obs.rec.x + (float)obs.rec.width / 2.0, obs.rec.y + (float)obs.rec.height / 2.0);
	}

	// Update the Covariance Matrix
	Ps_temp.copyTo(stat_pred.cov);

	// CV_64FC1 to CV_32FC1
	for (int r = 0; r < z_cov_flt.rows; r++)
		for (int c = 0; c < z_cov_flt.cols; c++)
			z_cov_flt.at<float>(r, c) = z_cov_dbl.at<double>(r, c);

	q_value = this->CalcGaussianProbability(dims_obs, z_temp, mean_obs, z_cov_flt);

	if (q_value < FLT_MIN) q_value = 0.0;
	return q_value;
}
float GMPHD_MAF::TrackletWiseAffinityKF(BBTrk &stat_pred, const BBTrk& obs, const int MODEL_TYPE) {

	const int dims_obs = sym::DIMS_OBS[MODEL_TYPE];

	if (MODEL_TYPE == sym::MODEL_VECTOR::XY) {
		// Bounding box size contraint
		if ((stat_pred.rec.area() >= obs.rec.area() * SIZE_CONSTRAINT_RATIO) || (stat_pred.rec.area() * SIZE_CONSTRAINT_RATIO <= obs.rec.area())) return 0.0;

		// Bounding box location contraint(gating)
		if (stat_pred.rec.area() >= obs.rec.area()) {
			if ((stat_pred.rec & obs.rec).area() <= 0 /*obs.rec.area() / 4*/) return 0.0;
		}
		else {
			if ((stat_pred.rec & obs.rec).area() <= 0  /*stat_pred.rec.area() / 4*/) return 0.0;
		}
	}
	// Step2: Update each Gaussian for every observation
	// find the observation which makes the Gaussian's weight into maximum among every observation

	// Step 2: Update phase1
	double q_value = 0.0;

	cv::Mat mean_obs(dims_obs, 1, CV_32FC1);
	cv::Mat z_cov_flt(dims_obs, dims_obs, CV_32FC1);
	cv::Mat z_temp(dims_obs, 1, CV_32FC1);

	if (MODEL_TYPE == sym::MODEL_VECTOR::XY ) {
		mean_obs.at<float>(0, 0) = (float)stat_pred.rec.x + (float)stat_pred.rec.width / 2.0;
		mean_obs.at<float>(1, 0) = (float)stat_pred.rec.y + (float)stat_pred.rec.height / 2.0;

		z_temp = (cv::Mat_<float>(dims_obs, 1) << obs.rec.x + (float)obs.rec.width / 2.0, obs.rec.y + (float)obs.rec.height / 2.0);
	}

	// CV_64FC1 4x4 to CV_32FC1 2x2
	for (int r = 0; r < z_cov_flt.rows; r++)
		for (int c = 0; c < z_cov_flt.cols; c++)
			z_cov_flt.at<float>(r, c) = stat_pred.cov.at<double>(r, c);

	q_value = this->CalcGaussianProbability(dims_obs, z_temp, mean_obs, z_cov_flt);

	if (q_value < FLT_MIN) q_value = 0.0;
	return q_value;
}
void GMPHD_MAF::FusionMinMaxNorm(const int& nObs, const int& mStats, vector<vector<double>> &m_cost, vector<vector<double>>& gmphd_cost, vector<vector<double>>& kcf_cost, const vector<vector<float>>& IOUs,
	const float& CONF_LOWER_THRESH, const float& CONF_UPPER_THRESH, const float& IOU_LOWER_THRESH, const float& IOU_UPPER_THRESH, const bool& T2TA_ON) {
	double min_gmphd_cost = FLT_MAX;
	double max_gmphd_cost = 0.0;
	double range_gmphd_cost;
	for (int r = 0; r < nObs; ++r) {
		for (int c = 0; c < mStats; ++c) {
			if (min_gmphd_cost > gmphd_cost[c][r]) min_gmphd_cost = gmphd_cost[c][r];
			if (max_gmphd_cost < gmphd_cost[c][r]) max_gmphd_cost = gmphd_cost[c][r];
		}
	}

	if (min_gmphd_cost == max_gmphd_cost) {
		if (max_gmphd_cost == 0.0)	max_gmphd_cost = 1.0;
		else						min_gmphd_cost = 0.0;
	}
	range_gmphd_cost = max_gmphd_cost - min_gmphd_cost;

	double min_kcf_cost = FLT_MAX;
	double max_kcf_cost = 0.0;
	double range_kcf_cost;
	for (int r = 0; r < nObs; ++r) {
		for (int c = 0; c < mStats; ++c) {

			if (min_kcf_cost > kcf_cost[c][r]) min_kcf_cost = kcf_cost[c][r];
			if (max_kcf_cost < kcf_cost[c][r]) max_kcf_cost = kcf_cost[c][r];
		}
	}
	if (min_kcf_cost == max_kcf_cost) {
		if (max_kcf_cost == 0.0)	max_kcf_cost = 1.0;
		else						min_kcf_cost = 0.0;
	}
	range_kcf_cost = max_kcf_cost - min_kcf_cost;

	if (false/*nObs > 0*/) {
		printf("GMPHD (min:%s-max:%s)(%s)", this->to_str(min_gmphd_cost).c_str(), this->to_str(max_gmphd_cost).c_str(), this->to_str(range_gmphd_cost).c_str());
		printf(", KCF (min:%s-max:%s)(%s)\n", this->to_str(min_kcf_cost).c_str(), this->to_str(max_kcf_cost).c_str(), this->to_str(range_kcf_cost).c_str());
	}
	if (T2TA_ON) {
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {

				double gmphd_temp = gmphd_cost[c][r];
				float kcf_temp = kcf_cost[c][r];
				float iou = IOUs[r][c];

				if (gmphd_temp == 0.0 && kcf_temp <= 0.01) {
					m_cost[c][r] = 10000;
					continue;
				}

				gmphd_cost[c][r] = (gmphd_cost[c][r] - min_gmphd_cost) / (range_gmphd_cost);

				// T2TA 를 길게 한경우 GMPHD cost (position) 가 높아도 KCF (appearance) 안닯은건 제외하기 위함
				if (kcf_temp < CONF_LOWER_THRESH) kcf_cost[c][r] = 0.0;
				else kcf_cost[c][r] = (kcf_cost[c][r] - min_kcf_cost) / (range_kcf_cost);

				//m_cost[c][r] = log(1 + gmphd_cost[c][r] * kcf_cost[c][r]);
				double cost_fusion = gmphd_cost[c][r] * kcf_cost[c][r];
				if (gmphd_cost[c][r] == 0.0 && kcf_cost[c][r] == 0.0) {
					m_cost[c][r] = 10000;
					continue;
				}
				// alternative cost using IOU instead of gmphd_cost when KCF cost is high
				// GMPHD cost 가 낮아도 KCF 매우 닮은거나 IOU 가 매우 높은것은 포함시키기 위함
				if (gmphd_cost[c][r] == 0 && (kcf_temp >= CONF_UPPER_THRESH || IOUs[r][c] >= IOU_UPPER_THRESH)) {
					cost_fusion = IOUs[r][c] * kcf_cost[c][r];
				}
				/*else {
					if (gmphd_cost[c][r] == 0 || IOUs2D[r][c] > IOU_TRESH) {
						cost_fusion = IOUs[r][c] * kcf_cost[c][r];
					}
				}*/

				if (cost_fusion == 0.0)
					m_cost[c][r] = 10000;
				else
					m_cost[c][r] = std::abs(-100 * log(cost_fusion));
			}
		}
	}
	else {
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {

				gmphd_cost[c][r] = (gmphd_cost[c][r] - min_gmphd_cost) / (range_gmphd_cost);

				/*if (kcf_cost[c][r] < 0.5) kcf_cost[c][r] = 0.0;
				else */kcf_cost[c][r] = (kcf_cost[c][r] - min_kcf_cost) / (range_kcf_cost);

				//m_cost[c][r] = log(1 + gmphd_cost[c][r] * kcf_cost[c][r]);
				double cost_fusion = gmphd_cost[c][r] * kcf_cost[c][r];
				if (gmphd_cost[c][r] == 0.0 && kcf_cost[c][r] == 0.0) {
					m_cost[c][r] = 10000;
					continue;
				}
				// alternative cost using IOU instead of gmphd_cost
				// && 로 바꿔볼까 -> || 가 더 좋음
				if (gmphd_cost[c][r] == 0 || IOUs[r][c] > IOU_LOWER_THRESH) {
					cost_fusion = IOUs[r][c] * kcf_cost[c][r];
				}

				if (cost_fusion == 0.0)
					m_cost[c][r] = 10000;
				else
					m_cost[c][r] = std::abs(-100 * log(cost_fusion));
			}
		}
	}
}
vector<double> GMPHD_MAF::FusionZScoreNorm(const int& nObs, const int& mStats,
	vector<vector<double>> &m_cost, vector<vector<double>>& gmphd_cost, vector<vector<double>>& kcf_cost, const vector<vector<float>>& IOUs, double conf_interval,
	const float& CONF_LOWER_THRESH, const float& CONF_UPPER_THRESH, const float& IOU_LOWER_THRESH, const float& IOU_UPPER_THRESH, const bool& T2TA_ON) {

	double mean_kcf_cost = 0.0;
	double stddev_kcf = 1.0;
	double mean_gmphd_cost = 0.0;
	double stddev_gmphd = 1.0;

	this->GetMeanStdDev(nObs, mStats, gmphd_cost, mean_gmphd_cost, stddev_gmphd, kcf_cost, mean_kcf_cost, stddev_kcf);

	if ((T2TA_ON & VIS_T2TA_DETAIL) || (!T2TA_ON && VIS_D2TA_DETAIL)) {
		if (nObs > 0) {
			printf("    GMPHD_COST (m: %s, stddev: %s)\n", this->to_str(mean_gmphd_cost).c_str(), this->to_str(stddev_gmphd).c_str());
			printf("    KCF_COST (m: %s, stddev: %s)\n", this->to_str(mean_kcf_cost).c_str(), this->to_str(stddev_kcf).c_str());
		}
	}

	if (T2TA_ON) {
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {

				double gmphd_temp = gmphd_cost[c][r];
				float kcf_temp = kcf_cost[c][r];
				float iou = IOUs[r][c];

				if (gmphd_temp == 0.0 && kcf_temp <= 0.01) {
					m_cost[c][r] = 10000;
					continue;
				}

				gmphd_cost[c][r] = (gmphd_cost[c][r] - mean_gmphd_cost) / (stddev_gmphd);
				if (gmphd_cost[c][r] < 0.0) gmphd_cost[c][r] = 0.0;

				// T2TA 를 길게 한경우 GMPHD cost (position) 가 높아도 KCF (appearance) 안닯은건 제외하기 위함
				if (kcf_temp < CONF_LOWER_THRESH) kcf_cost[c][r] = 0.0;
				else kcf_cost[c][r] = (kcf_cost[c][r] - mean_kcf_cost) / (stddev_kcf);

				if (kcf_cost[c][r] < 0.0) kcf_cost[c][r] = 0.0;

				//m_cost[c][r] = log(1 + gmphd_cost[c][r] * kcf_cost[c][r]);
				double cost_fusion = gmphd_cost[c][r] * kcf_cost[c][r];
				if (gmphd_cost[c][r] == 0.0 && kcf_cost[c][r] == 0.0) {
					m_cost[c][r] = 10000;
					continue;
				}
				// alternative cost using IOU instead of gmphd_cost when KCF cost is high
				// GMPHD cost 가 낮아도 KCF 매우 닮은거나 IOU가 매우 높은 것은 포함시키기 위함
				if (gmphd_cost[c][r] == 0 && (kcf_temp >= CONF_UPPER_THRESH || IOUs[r][c] >= IOU_UPPER_THRESH)) {
					cost_fusion = IOUs[r][c] * kcf_cost[c][r];
				}

				if (cost_fusion == 0.0)
					m_cost[c][r] = 10000;
				else if (cost_fusion >= 1.0)
					m_cost[c][r] = 0.0;
				else
					m_cost[c][r] = std::abs(-100 * log(cost_fusion));

			}
		}
	}
	else {
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {

				gmphd_cost[c][r] = (gmphd_cost[c][r] - mean_gmphd_cost) / (stddev_gmphd);
				if (gmphd_cost[c][r] < 0.0) gmphd_cost[c][r] = 0.0;

				kcf_cost[c][r] = (kcf_cost[c][r] - mean_kcf_cost) / (stddev_kcf);
				if (kcf_cost[c][r] < 0.0) kcf_cost[c][r] = 0.0;

				//m_cost[c][r] = log(1 + gmphd_cost[c][r] * kcf_cost[c][r]);
				double cost_fusion = gmphd_cost[c][r] * kcf_cost[c][r];
				if (gmphd_cost[c][r] == 0.0 && kcf_cost[c][r] == 0.0) {
					m_cost[c][r] = 10000;
					continue;
				}
				// alternative cost using IOU instead of gmphd_cost
				// && 로 바꿔볼까 -> || 가 더 좋음
				if (gmphd_cost[c][r] == 0 || IOUs[r][c] > IOU_LOWER_THRESH) {
					cost_fusion = IOUs[r][c] * kcf_cost[c][r];
				}

				if (cost_fusion == 0.0)
					m_cost[c][r] = 10000;
				else if (cost_fusion >= 1.0)
					m_cost[c][r] = 0.0;
				else
					m_cost[c][r] = std::abs(-100 * log(cost_fusion));
			}
		}
	}

	// Mean and Stddev of GMPHD and KCF costs 
	vector<double> gmphd_kcf_mean_sdtddev;
	gmphd_kcf_mean_sdtddev.push_back(mean_gmphd_cost);
	gmphd_kcf_mean_sdtddev.push_back(stddev_gmphd);
	gmphd_kcf_mean_sdtddev.push_back(mean_kcf_cost);
	gmphd_kcf_mean_sdtddev.push_back(stddev_kcf);

	return gmphd_kcf_mean_sdtddev;
}
void GMPHD_MAF::FusionDblSigmoidNorm(const int& nObs, const int& mStats,
	vector<vector<double>> &m_cost, vector<vector<double>>& gmphd_cost, vector<vector<double>>& kcf_cost, const vector<vector<float>>& IOUs,
	const float& CONF_LOWER_THRESH, const float& CONF_UPPER_THRESH, const float& IOU_LOWER_THRESH, const float& IOU_UPPER_THRESH, const bool& T2TA_ON) {

}
void GMPHD_MAF::FusionTanHNorm(const int& nObs, const int& mStats,
	vector<vector<double>> &m_cost, vector<vector<double>>& gmphd_cost, vector<vector<double>>& kcf_cost, const vector<vector<float>>& IOUs,
	const float& CONF_LOWER_THRESH, const float& CONF_UPPER_THRESH, const float& IOU_LOWER_THRESH, const float& IOU_UPPER_THRESH, const bool& T2TA_ON) {

	double mean_kcf_cost = 0.0;
	double stddev_kcf = 1.0;
	double mean_gmphd_cost = 0.0;
	double stddev_gmphd = 1.0;

	this->GetMeanStdDev(nObs, mStats, gmphd_cost, mean_gmphd_cost, stddev_gmphd, kcf_cost, mean_kcf_cost, stddev_kcf);

	if ((T2TA_ON & VIS_T2TA_DETAIL) || (!T2TA_ON && VIS_D2TA_DETAIL)) {
		printf("    GMPHD_COST (m: %s, stddev: %s)\n", this->to_str(mean_gmphd_cost).c_str(), this->to_str(stddev_gmphd).c_str());
		printf("    KCF_COST (m: %s, stddev: %s)\n", this->to_str(mean_kcf_cost).c_str(), this->to_str(stddev_kcf).c_str());
	}

	if (T2TA_ON) {
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {

				double gmphd_temp = gmphd_cost[c][r];
				float kcf_temp = kcf_cost[c][r];
				float iou = IOUs[r][c];

				if (gmphd_temp == 0.0 && kcf_temp <= 0.01) {
					m_cost[c][r] = 10000;
					continue;
				}

				gmphd_cost[c][r] = 0.5*(std::tanh(0.01*(gmphd_cost[c][r] - mean_gmphd_cost) / (stddev_gmphd)) + 1);

				// T2TA 를 길게 한경우 GMPHD cost (position) 가 높아도 KCF (appearance) 안닯은건 제외하기 위함
				if (kcf_temp < CONF_LOWER_THRESH) kcf_cost[c][r] = 0.0;
				else kcf_cost[c][r] = 0.5*(std::tanh(0.01*(kcf_cost[c][r] - mean_kcf_cost) / (stddev_kcf)) + 1);

				//m_cost[c][r] = log(1 + gmphd_cost[c][r] * kcf_cost[c][r]);
				double cost_fusion = gmphd_cost[c][r] * kcf_cost[c][r];

				// alternative cost using IOU instead of gmphd_cost when KCF cost is high
				// GMPHD cost 가 낮아도 KCF 매우 닮은건 포함시키기 위함
				if (gmphd_temp == 0 && (kcf_temp >= CONF_UPPER_THRESH || IOUs[r][c] >= IOU_UPPER_THRESH)) {
					cost_fusion = iou * kcf_cost[c][r];
				}

				if (cost_fusion == 0.0)
					m_cost[c][r] = 10000;
				else
					m_cost[c][r] = std::abs(-100 * log(cost_fusion));

			}
		}
	}
	else {
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {

				double gmphd_temp = gmphd_cost[c][r];
				float kcf_temp = kcf_cost[c][r];
				float iou = IOUs[r][c];

				gmphd_cost[c][r] = 0.5*(std::tanh(0.01*(gmphd_cost[c][r] - mean_gmphd_cost) / (stddev_gmphd)) + 1);
				kcf_cost[c][r] = 0.5*(std::tanh(0.01*(kcf_cost[c][r] - mean_kcf_cost) / (stddev_kcf)) + 1);

				//m_cost[c][r] = log(1 + gmphd_cost[c][r] * kcf_cost[c][r]);
				double cost_fusion = gmphd_cost[c][r] * kcf_cost[c][r];
				if (gmphd_temp == 0.0 && kcf_temp == 0.01) {
					m_cost[c][r] = 10000;
					continue;
				}
				// alternative cost using IOU instead of gmphd_cost
				// && 로 바꿔볼까 -> || 가 더 좋음
				if (gmphd_temp == 0 || iou > IOU_LOWER_THRESH) {
					cost_fusion = iou * kcf_cost[c][r];
				}

				if (cost_fusion == 0.0)
					m_cost[c][r] = 10000;
				else
					m_cost[c][r] = std::abs(-100 * log(cost_fusion));
			}
		}
	}
}
void GMPHD_MAF::GetMeanStdDev(const int& nObs, const int& mStats,
	vector<vector<double>>&gmphd_cost, double &mean_gmphd, double& stddev_gmphd, vector<vector<double>>&kcf_cost, double &mean_kcf, double& stddev_kcf) {
	double m_kcf_cost = 0.0;
	double m_kcf_2 = 0.0;
	double m_gmphd_cost = 0.0;
	double m_gmphd_2 = 0.0;
	double nSamples = (double)(nObs*mStats);

	for (int r = 0; r < nObs; ++r) {
		for (int c = 0; c < mStats; ++c) {
			m_gmphd_cost += gmphd_cost[c][r];
			m_kcf_cost += kcf_cost[c][r];

			m_gmphd_2 += (gmphd_cost[c][r] * gmphd_cost[c][r]);
			m_kcf_2 += (kcf_cost[c][r] * kcf_cost[c][r]);
		}
	}

	if (nSamples > 1) {
		mean_gmphd = m_gmphd_cost / nSamples;
		mean_kcf = m_kcf_cost / nSamples;

		m_gmphd_2 /= (nSamples);
		m_kcf_2 /= (nSamples);

		stddev_gmphd = sqrt(m_gmphd_2 - mean_gmphd * mean_gmphd);
		stddev_kcf = sqrt(m_kcf_2 - mean_kcf * mean_kcf);
	}
	if (stddev_gmphd == 0.0) stddev_gmphd = 1.0;
	if (stddev_kcf == 0.0) stddev_kcf = 1.0;
}
void GMPHD_MAF::DataAssocTrkWise(int iFrmCnt, cv::Mat& img, vector<BBTrk>& stats_lost, vector<BBTrk>& obss_live, const int &MODEL_TYPE) {
	//FILE* fp_t2ta_affs[2];
	//string dirHome = "res\\KITTI\\train\\";
	//string fileTags[2] = { "_t2ta_not_norm","_t2ta_norm_maf" };
	//string objName = sym::OBJECT_STRINGS[this->trackObjType];
	//string filePaths[2];

	//for (int f = 0; f < 2; ++f) {
	//	filePaths[f] = dirHome + this->seqName+"_"+objName +fileTags[f] + ".txt";
	//	if (iFrmCnt == this->params.FRAME_OFFSET+1) // after init
	//		fp_t2ta_affs[f] = fopen(filePaths[f].c_str(), "w+");
	//	else
	//		fp_t2ta_affs[f] = fopen(filePaths[f].c_str(), "a");
	//}


	double min_cost_dbl = DBL_MAX;
	int nObs = obss_live.size();
	int mStats = stats_lost.size();
	vector<vector<double>> m_cost;
	m_cost.resize(mStats, vector<double>(nObs, 0.0));

	vector<vector<double>> kcf_cost;
	kcf_cost.resize(mStats, vector<double>(nObs, 0.0));

	vector<vector<double>> gmphd_cost;
	gmphd_cost.resize(mStats, vector<double>(nObs, 0.0));

	vector<vector<float>> IOUs;
	IOUs.resize(nObs, std::vector<float>(mStats, 0.0));

	vector<vector<float>> IOUs2D;
	IOUs2D.resize(nObs, std::vector<float>(mStats, 0.0));

	vector<vector<double>> q_values;
	q_values.resize(nObs, std::vector<double>(mStats, 0.0));

	vector<vector<int>> fd_ex;
	fd_ex.resize(nObs, std::vector<int>(mStats, -1));

	vector<vector<float>> ry_norms;
	ry_norms.resize(nObs, std::vector<float>(mStats, 0.0));

	vector<vector<BBTrk>> stats_matrix; // It can boost parallel processing?
	stats_matrix.resize(nObs, std::vector<BBTrk>(mStats, BBTrk()));

	//cv::Mat tempVis = img.clone();

	for (int r = 0; r < obss_live.size(); ++r) {
		//stats.assign(stats_matrix[r].begin(), stats_matrix[r].end());
		for (int c = 0; c < stats_lost.size(); ++c)
		{
			stats_lost.at(c).CopyTo(stats_matrix[r][c]);
		}
	}

	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for(int r=0;r<nObs;++r){
		for (int c = 0; c < stats_matrix[r].size(); ++c) {

			int lostID = stats_matrix[r][c].id;
			int liveID = obss_live[r].id;

			int fd = this->tracks_reliable[liveID].front().fn - this->tracks_reliable[lostID].back().fn;
			fd_ex[r][c] = fd;

			double Taff = 1.0;
			int trk_fd = 0; // frame_difference
			if (fd >= TRACK_ASSOCIATION_FRAME_DIFFERENCE && fd <= this->params.T2TA_MAX_INTERVAL) { // ==0 일때는 occlusion 을 감안해야 할듯 일단 >0 으로 해보자

				// Linear Motion Estimation
				int idx_first = 0;
				int idx_last = this->tracks_reliable[lostID].size() - 1;

				cv::Vec4f v = this->LinearMotionEstimation(this->tracks_reliable, lostID, trk_fd, idx_first, idx_last, MODEL_TYPE);

				if (T2TA_MOTION_OPT == USE_LINEAR_MOTION) {

					BBTrk stat_pred_LM;
					this->tracks_reliable[lostID].back().CopyTo(stat_pred_LM);

					stat_pred_LM.vx = v[0];
					stat_pred_LM.vy = v[1];
					stat_pred_LM.vd = v[2];
					stat_pred_LM.vr = v[3];

					//printf("(%lf,%lf)\n",v.x,v.y);

					
					stat_pred_LM.rec.x = stat_pred_LM.rec.x + stat_pred_LM.vx*fd;
					stat_pred_LM.rec.y = stat_pred_LM.rec.y + stat_pred_LM.vy*fd;
					stat_pred_LM.ratio_yx = stat_pred_LM.ratio_yx + stat_pred_LM.vr*fd;
					//stat_pred_LM.depth = stat_pred_LM.depth + stat_pred_LM.vd*fd;

					stat_pred_LM.CopyTo(stats_matrix[r][c]);
					//this->cvBoundingBox(img, obs_pred.rec, this->color_tab[obs_pred.id % 26], 3);
				}
				else if (T2TA_MOTION_OPT == USE_KALMAN_MOTION) {
					if (KALMAN_INIT_BY_LINEAR) {
						stats_matrix[r][c].vx = v[0];
						stats_matrix[r][c].vy = v[1];
						stats_matrix[r][c].vd = v[2];
						stats_matrix[r][c].vr = v[3];
					}
					BBTrk stat_pred_KF = this->KalmanMotionbasedPrediction(stats_matrix[r][c], obss_live[r]);
					stat_pred_KF.CopyTo(stats_matrix[r][c]);
				}
				// Calculate the affinity between the n-th observation and the m-th state
				bool isOut = false;

				isOut = this->IsOutOfFrame(stats_matrix[r][c].rec, this->frmWidth, this->frmHeight);
				if (isOut) {
					q_values[r][c] = 0;
				}
				
				if (!isOut) { // MODEL_TYPE==sym::MODEL_VECTOR::XYZ || !isOut
					// Linear Motion based Prediction
					if (T2TA_MOTION_OPT == USE_LINEAR_MOTION)
						q_values[r][c] = /*pow(0.9,fd)**/TrackletWiseAffinity(stats_matrix[r][c], this->tracks_reliable[liveID].front(), MODEL_TYPE);
					// Kalman Motion based Prediction
					else if (T2TA_MOTION_OPT == USE_KALMAN_MOTION)
						q_values[r][c] = /*pow(0.9,fd)**/ TrackletWiseAffinityKF(stats_matrix[r][c], obss_live[r], MODEL_TYPE);
				}
				//q_values[r][c] = /*pow(0.9,fd)**/TrackletWiseAffinityVelocity(stat_pred, this->tracks_reliable[liveID].front(), 4);
				//-> this is too sensitive to deal with unstable detection bounding boxes	

				if (ASSOCIATION_STAGE_2_GATING_ON) {
					if (q_values[r][c] < Q_TH_LOW_20) {
						q_values[r][c] = 0.0;


						float overlapping_ratio = 0.0;
						if (this->params.MERGE_METRIC == MERGE_METRIC_IOU)
							overlapping_ratio = this->CalcIOU(obss_live[r].rec, stats_matrix[r][c].rec);
						else if (this->params.MERGE_METRIC == MERGE_METRIC_SIOA)
							overlapping_ratio = this->CalcSIOA(obss_live[r].rec, stats_matrix[r][c].rec);
						else {
							cv::Mat m_uni;
							overlapping_ratio = this->CalcMIOU(obss_live[r].rec, obss_live[r].segMask, stats_matrix[r][c].rec, stats_matrix[r][c].segMask, m_uni);
							m_uni.release();
						}

						if (overlapping_ratio >= this->params.MERGE_RATIO_THRESHOLD) // threshold
							q_values[r][c] = obss_live[r].weight*overlapping_ratio;

					}
				}
			}
			else {
				q_values[r][c] = 0;
			}
			// Calculate the Affinity between detection (observations) and tracking (states)
		}
	}
	);

	// Calculate States' Weights by GMPHD filtering with q values
	// Then the Cost Matrix is filled with States's Weights to solve Assignment Problem.
	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for (int r = 0; r < nObs; ++r){
		int nStats = stats_matrix[r].size();
		// (19)
		double denominator = 0.0;													// (19)'s denominator(분모)
		for (int l = 0; l < stats_matrix[r].size(); ++l) {
			denominator += (stats_matrix[r][l].weight * q_values[r][l]);
		}
		for (int c = 0; c < stats_matrix[r].size(); ++c) {
			double numerator = /*P_detection*/stats_matrix[r][c].weight*q_values[r][c];


			stats_matrix[r][c].weight = numerator / denominator;
			IOUs2D[r][c] = this->CalcIOU(stats_matrix[r][c].rec, obss_live[r].rec);

			// Scaling the Affinity value into Integer
			if (stats_matrix[r][c].weight > 0.0) {
				if ((double)(stats_matrix[r][c].weight) < (double)(FLT_MIN) / (double)10.0) {
					std::cerr << "[" << iFrmCnt << "] weight(TW) < 0.1*FLT_MIN" << std::endl;
					if (!AFFINITY_COST_L1_OR_L2_NORM_TW)	m_cost[c][r] = -1;
					else									m_cost[c][r] = 10000;	// upper bound, 작지만 double로 표현가능한 값이 존재하므로 association 후보임
				}
				else {
					if (!AFFINITY_COST_L1_OR_L2_NORM_TW)	m_cost[c][r] = -1.0*Q_TH_LOW_10_INVERSE * numerator;
					else									m_cost[c][r] = -100.0*log(numerator);
				}
			}
			else {
				if (!AFFINITY_COST_L1_OR_L2_NORM_TW)		m_cost[c][r] = 0;
				else										m_cost[c][r] = 10000;	// upper bound, double로도 표현 불가능한 값이므로 association 후보조차 제외
			}

			/*printf("(TW) Obs_id%d(%d,%d,%d,%d,%.4f) -> Stat_id%d:%.lf (n:%lf, q:%.20lf, W':%.5f)\n", \
				obss_live[r].id, obss_live[r].rec.x, obss_live[r].rec.y, obss_live[r].rec.width, obss_live[r].rec.height, obss_live[r].weight,
				stats_matrix[r][c].id, m_cost[c][r], numerator, q_values[r][c], stats_matrix[r][c].weight);*/
		}
	}
	);

	//imshow("KF in T2TA", tempVis);
	//cv::waitKey(1);
	//tempVis.release();


	// KCF features based Affinity Calculation
	if (this->params.SAF_T2TA_MODE > sym::AFFINITY_OPT::GMPHD) {
		// KCF Vis Window Setting
		//string winTitleT2TA = "MOT with T2TA"; // (ID: " + boost::to_string(id) +")";
		cv::Mat frameVis;
		if (VIS_T2TA_DETAIL) {
			//cv::namedWindow(winTitleT2TA);	cv::moveWindow(winTitleT2TA, 0, (this->frmHeight+30) * 3.0);
			frameVis = img.clone();
		}
		//
		if (mStats > 0) {

			string winTitleT2TA = "Lost x Live Tracks"; // (ID: " + boost::to_string(id) +")";

			int margin = 40;
			int cellWH = 140;
			cv::Mat canvasDA;

			if (VIS_T2TA_DETAIL) {
				cv::namedWindow(winTitleT2TA);	cv::moveWindow(winTitleT2TA, this->frmWidth, 0);
				canvasDA = cv::Mat(cellWH * (mStats)+margin, cellWH * (nObs + 1) + margin, CV_8UC3, cv::Scalar(200, 200, 200));

				//this->cvPrintVec2Vec(fd_ex, "Frame Diffs");
			}

			vector<vector<BBTrk>> stats_dummy;
			Concurrency::parallel_for(0, mStats, [&](int c) {
				//for (int c = 0; c < mStats; ++c) {
					//cv::Mat framePrev = this->imgBatch[this->params.QUEUE_SIZE-2];
				cv::Mat frameProc = img.clone(); // current frame considering latency

				int id_lost = stats_lost[c].id;
				if (VIS_T2TA_DETAIL) {
					cv::Mat statResize;
					/*printf("%d-%dx%d(%d,%d,%d,%d)\n",
						stats_lost[c].tmpl.empty(), stats_lost[c].tmpl.rows, stats_lost[c].tmpl.cols,
						stats_lost[c].rec.x, stats_lost[c].rec.y, stats_lost[c].rec.width, stats_lost[c].rec.height);*/
					if (stats_lost[c].tmpl.empty())
						statResize = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0.0, 0));
					else
						cv::resize(stats_lost[c].tmpl, statResize, cv::Size(100, 100)); // predicted state

					statResize.copyTo(canvasDA(cv::Rect(margin, margin + cellWH * c, 100, 100)));

					char strIDConf[64];
					sprintf_s(strIDConf, "ID%d (%.3f)", id_lost, stats_lost[c].conf);
					cv::putText(canvasDA, string(strIDConf), cv::Point(margin, margin / 2 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.5, this->color_tab[(id_lost / 2) % (MAX_OBJECTS - 1)], 2);
					cv::putText(canvasDA, "~" + std::to_string(stats_lost[c].fn), cv::Point(margin, margin / 2 + cellWH * (c + 1) - 105), CV_FONT_HERSHEY_COMPLEX, 0.5, this->color_tab[(id_lost / 2) % (MAX_OBJECTS - 1)], 2);
				}


				if (VIS_T2TA_DETAIL)
					printf("[%d] Affinity between ID%d [%d~%d](%d,%d,%d,%d) and %d Live Tracks..\n", iFrmCnt, id_lost,
						this->tracks_reliable[id_lost].front().fn, this->tracks_reliable[id_lost].back().fn,
						stats_lost[c].rec.x, stats_lost[c].rec.y, stats_lost[c].rec.width, stats_lost[c].rec.height, nObs);

				if (nObs > 0) {

					for (int r = 0; r < nObs; ++r) {
						if (fd_ex[r][c] >= TRACK_ASSOCIATION_FRAME_DIFFERENCE && fd_ex[r][c] <= this->params.T2TA_MAX_INTERVAL) {

							int id_live = obss_live[r].id;

							float objConf = obss_live[r].conf;
							float iou = this->CalcIOU(obss_live[r].rec, stats_matrix[r][c].rec);
							IOUs[r][c] = iou;

							cv::Rect obsRecInFrame = this->RectExceptionHandling(this->frmWidth, this->frmHeight, obss_live[r].rec);
							//cout << "(aobss_live
							cv::Mat confMap;
							float confProb;
							if (this->params.GATE_T2TA_ON) {
								if (iou < 0.1) {
									confProb = 0.99;
									//cv::rectangle(frameVis, obss[r].rec, cv::Scalar(0, 0, 0), -1);
									if (VIS_T2TA_DETAIL)
										confMap = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
								}
								else {
									cv::Rect res = stats_matrix[r][c].papp.update(frameProc, confProb, confMap, obss_live[r].rec, true, obss_live[r].segMask, this->params.SAF_MASK_T2TA);
									//cv::addWeighted(frameVis(res), 0.5, confMap, 0.5, 0.0, frameVis(res));
									//confMap.release();

									if (res.width < 1 || res.height < 1) {
										//confProb = 0.99;
										if (VIS_T2TA_DETAIL)
											confMap = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
									}
								}
							}
							else {
								cv::Rect res = stats_matrix[r][c].papp.update(frameProc, confProb, confMap, obss_live[r].rec, true, obss_live[r].segMask, this->params.SAF_MASK_T2TA);

								if (res.width < 1 || res.height < 1) {
									//confProb = 0.99;
									if (VIS_T2TA_DETAIL)
										confMap = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
								}
							}
							cv::Mat obsResize;
							cv::Rect obsDAcell;
							cv::Mat confMapResize;
							if (VIS_T2TA_DETAIL) {
								obsDAcell = cv::Rect(margin + cellWH * (r + 1), margin + cellWH * c, 100, 100);

								if (obsRecInFrame.width < 1 || obsRecInFrame.height < 1)
									obsResize = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
								else
									cv::resize(frameProc(obsRecInFrame), obsResize, cv::Size(100, 100));

								obsResize.copyTo(canvasDA(obsDAcell));
								cv::resize(confMap, confMapResize, cv::Size(100, 100));
							}

							char scores[256];
							double score_gmphd_temp;// = pow(10, m_cost[c][r] / (-100.0));
							if (m_cost[c][r] < 10000) {
								gmphd_cost[c][r] = pow(e_8, m_cost[c][r] / (-100.0));
								score_gmphd_temp = gmphd_cost[c][r];
							}
							else {
								gmphd_cost[c][r] = 0.0;
								score_gmphd_temp = gmphd_cost[c][r];
							}

							float score_kcf_temp = (1.0 - confProb)/**objConf*/;
							if (SOT_TRACK_OPT == SOT_USE_KCF_TRACKER)	score_kcf_temp = 1.0 - confProb;
							else										score_kcf_temp = confProb;

							kcf_cost[c][r] = score_kcf_temp;

							float CONF_LOWER_THRESH = this->params.KCF_BOUNDS_T2TA[0];
							float CONF_UPPER_THRESH = this->params.KCF_BOUNDS_T2TA[1];

							if (this->params.SAF_T2TA_MODE == sym::AFFINITY_OPT::KCF) {

								if (score_kcf_temp >= CONF_UPPER_THRESH)
									m_cost[c][r] = -100 * log(score_kcf_temp);
								else if (score_kcf_temp < CONF_UPPER_THRESH && score_kcf_temp >= CONF_LOWER_THRESH && iou >= 0.1)
									m_cost[c][r] = -100 * log(score_kcf_temp);
								else
									m_cost[c][r] = 10000;
							}

							// 
							//kcf_cost[c][r] = scaleComb * -100 * log2f(1.0 - confProb);
							// m_cost[c][r] = m_cost[c][r] + kcf_cost[c][r];

							/*if (confProb > CONF_THRESH)
							m_cost[c][r] = 10000;
							else*/
							//	m_cost[c][r] = -100.0*log2l((double)(score_gmphd_temp*(1.0 - confProb)));

							if (VIS_T2TA_DETAIL) {
								printf("	+%d[%d:%d] ~ ID%d (%d,%d,%d,%d,%.5f)-", fd_ex[r][c],
									this->tracks_reliable[id_live].front().fn, this->tracks_reliable[id_live].back().fn,
									id_live, obss_live[r].rec.x, obss_live[r].rec.y, obss_live[r].rec.width, obss_live[r].rec.height, objConf);
								printf("IOU(%.2f), Cost:%s(GMPHD:%s, KCF:%.5f)\n", iou, this->to_str(m_cost[c][r]), this->to_str(score_gmphd_temp).c_str(), score_kcf_temp);

								if (m_cost[c][r] < 10000) {
									cv::putText(canvasDA, this->to_str(score_gmphd_temp).c_str(), cv::Point(margin + cellWH * (r + 1), margin - 30 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
								}
								else {
									char strINF_IOU[64];
									sprintf_s(strINF_IOU, "INF (IOU:%.3f)", IOUs2D[r][c]);
									cv::putText(canvasDA, strINF_IOU, cv::Point(margin + cellWH * (r + 1), margin - 30 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);
								}
								cv::addWeighted(canvasDA(obsDAcell), 0.6, confMapResize, 0.4, 0.0, canvasDA(obsDAcell));
								cv::putText(canvasDA, this->to_str(score_kcf_temp).c_str(), cv::Point(margin + cellWH * (r + 1), margin - 15 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, r), 1);

								char strIDConf[64];
								sprintf_s(strIDConf, "ID%d(%.3f)", id_live, obss_live[r].conf);
								cv::putText(canvasDA, string(strIDConf), cv::Point(margin + cellWH * (r + 1), margin - 45 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

								confMap.release();
								confMapResize.release();
							}
						}
						else {
							gmphd_cost[c][r] = 0.0;
							kcf_cost[c][r] = 0.0;
							m_cost[c][r] = 10000;

							/*if(this->sysFrmCnt>=120)
								cv::waitKey();*/

							if (VIS_T2TA_DETAIL) {
								cv::Mat obsResize, confMapResize;
								cv::Rect obsDAcell;

								int id_live = obss_live[r].id;
								double score_kcf_temp = 0.0;
								cv::Rect obsRecInFrame = this->RectExceptionHandling(this->frmWidth, this->frmHeight, obss_live[r].rec);
								cv::Mat confMap = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
								obsDAcell = cv::Rect(margin + cellWH * (r + 1), margin + cellWH * c, 100, 100);

								if (obsRecInFrame.width < 1 || obsRecInFrame.height < 1)
									obsResize = cv::Mat(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
								else
									cv::resize(frameProc(obsRecInFrame), obsResize, cv::Size(100, 100));

								obsResize.copyTo(canvasDA(obsDAcell));
								cv::resize(confMap, confMapResize, cv::Size(100, 100));

								char strINF_IOU[64];
								sprintf_s(strINF_IOU, "INF (IOU:%.3f)", IOUs2D[r][c]);
								cv::putText(canvasDA, strINF_IOU, cv::Point(margin + cellWH * (r + 1), margin - 30 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

								cv::addWeighted(canvasDA(obsDAcell), 0.6, confMapResize, 0.4, 0.0, canvasDA(obsDAcell));
								cv::putText(canvasDA, this->to_str(score_kcf_temp).c_str(), cv::Point(margin + cellWH * (r + 1), margin - 15 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.4, cv::Scalar(0, 0, r), 1);

								char strIDConf[64];
								sprintf_s(strIDConf, "ID%d(%.3f)", id_live, obss_live[r].conf);
								cv::putText(canvasDA, string(strIDConf), cv::Point(margin + cellWH * (r + 1), margin - 45 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

								// Excluded in T2TA because of the last frame of the lost track > the first frame of the live track
								cv::putText(canvasDA, std::to_string(stats_lost[c].fn + fd_ex[r][c]) + "~", cv::Point(margin + cellWH * (r + 1), margin - 90 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
								cv::putText(canvasDA, "Excluded", cv::Point(margin + cellWH * (r + 1), margin - 110 + cellWH * (c + 1)), CV_FONT_HERSHEY_COMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

								confMap.release();
								confMapResize.release();
							}
						}
					}

				}
				frameProc.release();
			}
			);

			// Fusing GMPHD and KCF scores
			float CONF_LOWER_THRESH = this->params.KCF_BOUNDS_T2TA[0]; // used
			float CONF_UPPER_THRESH = this->params.KCF_BOUNDS_T2TA[1]; // used (moving : 0.8) (static: 0.9) see line 4610 iou < 0.01 also
			float IOU_LOWER_THRESH = this->params.IOU_BOUNDS_T2TA[0];// not used
			float IOU_UPPER_THRESH = this->params.IOU_BOUNDS_T2TA[1];// used

			if (this->params.SAF_T2TA_MODE == sym::AFFINITY_OPT::MAF) {

				/*for (int c = 0; c < mStats; ++c) {
					for (int r = 0; r < nObs; ++r) {

						if(gmphd_cost[c][r]>0)
							fprintf_s(fp_t2ta_affs[0], "%d\t%.10lf\t%.30lf\n", iFrmCnt, kcf_cost[c][r], gmphd_cost[c][r]);
					}
				}
				fclose(fp_t2ta_affs[0]);*/

				// (1) Min-Max Normalization (T2TA)			
				this->FusionMinMaxNorm(nObs, mStats, m_cost, gmphd_cost, kcf_cost, IOUs2D, CONF_LOWER_THRESH, CONF_UPPER_THRESH, IOU_LOWER_THRESH, IOU_UPPER_THRESH, true);
				// (2) Z-Score Normalization (T2TA)
				//this->FusionZScoreNorm(nObs, mStats, m_cost, gmphd_cost, kcf_cost, IOUs2D, 0.95, CONF_LOWER_THRESH, CONF_UPPER_THRESH, IOU_LOWER_THRESH, IOU_UPPER_THRESH, true);
				// (3) TanH Normalization
				//this->FusionTanHNorm(nObs, mStats, m_cost, gmphd_cost, kcf_cost, IOUs2D, CONF_LOWER_THRESH, CONF_UPPER_THRESH, IOU_LOWER_THRESH, IOU_UPPER_THRESH, true);

				//for (int c = 0; c < mStats; ++c) {
				//	for (int r = 0; r < nObs; ++r) {
				//		double m_cost_norm_tmp = 0.0;
				//		if (m_cost[c][r] == 10000) m_cost_norm_tmp = 0.0;
				//		else m_cost_norm_tmp = pow(e, m_cost[c][r] / -100.0);

				//		if (gmphd_cost[c][r] > 0)
				//			fprintf_s(fp_t2ta_affs[1], "%d\t%.10lf\t%.30lf\t%.30lf\n", iFrmCnt, kcf_cost[c][r], gmphd_cost[c][r], m_cost_norm_tmp);
				//	}
				//}
				//fclose(fp_t2ta_affs[1]);
			}

			//cv::waitKey();
			//);
			if (VIS_T2TA_DETAIL) {
				this->cvPrintVec2Vec(m_cost, "COST_T2TA");

				cv::imshow(winTitleT2TA, canvasDA);
				cv::waitKey(1);
				canvasDA.release();

				frameVis.release();
			}
		}
	}


	double min_cost = INT_MAX;
	double max_cost = 0;
	if (!AFFINITY_COST_L1_OR_L2_NORM_TW) { // -xxxxxxxx.xxxx ~ 0
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {
				if (min_cost > m_cost[c][r]) min_cost = m_cost[c][r];
				if (max_cost < m_cost[c][r]) max_cost = m_cost[c][r];
			}
		}
		for (int r = 0; r < nObs; ++r) {
			for (int c = 0; c < mStats; ++c) {
				m_cost[c][r] = m_cost[c][r] - min_cost + 1;
			}
		}
		max_cost = 1 - min_cost;
	}
	else {
		max_cost = 10000;
		int r_min = -1, c_min = -1;
	}


	// Hungarian Method for solving data association problem (find the max cost assignment pairs)

	std::vector<vector<int>> assigns_sparse, assigns_dense;
	vector<vector<double>> m_cost_d;
	vector<vector<cv::Vec2i>> trans_indices = this->cvtSparseMatrix2Dense(m_cost, m_cost_d/*, max_cost*/);

	if (m_cost_d.empty() || m_cost_d[0].empty()) {
		assigns_sparse = this->HungarianMethod(m_cost, obss_live.size(), stats_lost.size());
		//printf("[%dx%d] ", stats_lost.size(), obss_live.size());
		//this->cvPrintVec2Vec(m_cost, "Sparse Costs (T2TA-"+sym::OBJECT_STRINGS[this->trackObjType]+")");
		//this->cvPrintVec2Vec(assigns_sparse, "Sparse Assignments");
	}
	else {
		assigns_sparse.resize(obss_live.size(), std::vector<int>(stats_lost.size(), 0));
		//printf("[%dx%d]", trans_indices.size(), trans_indices[0].size());

		// m_cost (mStats x nObs) --transition--> assings (nObs x mStats)
		// trans_indices (mStats x nObs)
		assigns_dense = this->HungarianMethod(m_cost_d, trans_indices[0].size(), trans_indices.size());
		//this->cvPrintVec2Vec(m_cost_d, "Dense Costs (T2TA-" + sym::OBJECT_STRINGS[this->trackObjType] + ")");
		//this->cvPrintVec2Vec(trans_indices, "Dense-to-Sparse Transition");
		//this->cvPrintVec2Vec(assigns_dense, "Dense Assignments");
		for (int c = 0; c < trans_indices.size(); ++c) {		// states
			for (int r = 0; r < trans_indices[c].size(); ++r) { // observations
				int rt = trans_indices[c][r](1);
				int ct = trans_indices[c][r](0);
				//printf("Costs [%d,%d]->[%d,%d] ", c, r, ct, rt);
				//printf("Assigments [%d,%d]->[%d,%d] ", r,c, rt, ct);
				assigns_sparse[rt][ct] = assigns_dense[r][c];
			}
			//printf("\n");
		}
		//this->cvPrintVec2Vec(assigns_sparse, "Sparse Assignments");
	}

	bool *isAssignedStats = new bool[stats_lost.size()];	memset(isAssignedStats, 0, stats_lost.size());
	bool *isAssignedObs = new bool[obss_live.size()];	memset(isAssignedObs, 0, obss_live.size());

	for (int r = 0; r < obss_live.size(); ++r) {
		obss_live[r].id_associated = -1; // faild to tracklet association (init before check)
		for (int c = 0; c < stats_lost.size(); ++c) {
			if (assigns_sparse[r][c] == 1 && m_cost[c][r] < max_cost) {

				// obss_live[r].id = stats_lost[c].id;
				obss_live[r].id_associated = stats_lost[c].id;
				obss_live[r].fn_latest_T2TA = iFrmCnt;

				obss_live[r].conf = stats_lost[c].conf*CONFIDENCE_UPDATE_ALPHA + obss_live[r].conf*(1.0 - CONFIDENCE_UPDATE_ALPHA);

				isAssignedStats[c] = true;
				isAssignedObs[r] = true;

				// Copy the Covariance Matrix calculated from Motion based Prediction
				if (COV_UPDATE_T2TA)
					stats_matrix[r][c].cov.copyTo(obss_live[r].cov);

				stats_lost[c].isAlive = true;

				break;
			}
		}
	}

	delete[]isAssignedObs;
	delete[]isAssignedStats;
}
void GMPHD_MAF::ArrangeTargetsVecsBatchesLiveLost() {
	vector<BBTrk> liveTargets;
	vector<BBTrk> lostTargets;
	for (int tr = 0; tr < this->liveTrkVec.size(); ++tr) {
		if (this->liveTrkVec[tr].isAlive) {
			liveTargets.push_back(this->liveTrkVec[tr]);
		}
		else if (!this->liveTrkVec[tr].isAlive && !this->liveTrkVec[tr].isMerged) {

			lostTargets.push_back(this->liveTrkVec[tr]);
		}
		else {
			// abandon the merged targets (When target a'ID and b'TD are merged with a'ID < b'TD, target b is abandoned and not considered as LB_ASSOCIATION) 
		}
	}
	this->liveTrkVec.swap(liveTargets);	// swapping the alive targets
	this->lostTrkVec.swap(lostTargets);	// swapping the loss tragets	
	liveTargets.clear();
	lostTargets.clear();
}
void GMPHD_MAF::PushTargetsVecs2BatchesLiveLost() {
	if (this->sysFrmCnt >= this->params.TRACK_MIN_SIZE) {
		for (int q = 0; q < this->params.FRAMES_DELAY_SIZE; q++) {
			for (int i = 0; i < this->liveTracksBatch[q].size(); i++)this->liveTracksBatch[q].at(i).Destroy();
			this->liveTracksBatch[q].clear();
			this->liveTracksBatch[q] = liveTracksBatch[q + 1];

			for (int i = 0; i < this->lostTracksBatch[q].size(); i++)this->lostTracksBatch[q].at(i).Destroy();
			this->lostTracksBatch[q].clear();
			this->lostTracksBatch[q] = lostTracksBatch[q + 1];
		}
		this->liveTracksBatch[this->params.FRAMES_DELAY_SIZE] = this->liveTrkVec;
		this->lostTracksBatch[this->params.FRAMES_DELAY_SIZE] = this->lostTrkVec;
	}
	else if (this->sysFrmCnt < this->params.TRACK_MIN_SIZE) {
		this->liveTracksBatch[this->sysFrmCnt] = this->liveTrkVec;
		this->lostTracksBatch[this->sysFrmCnt] = this->lostTrkVec;
	}
}
void GMPHD_MAF::ClassifyTrackletReliability(int iFrmCnt, unordered_map<int, vector<BBTrk>>& tracksbyID,
	unordered_map<int, vector<BBTrk>>& reliables, unordered_map<int, std::vector<BBTrk>>& unreliables) {

	unordered_map<int, vector<BBTrk>>::iterator iterID;

	for (iterID = tracksbyID.begin(); iterID != tracksbyID.end(); iterID++) {
		if (!iterID->second.empty()) {

			if (iterID->second.back().fn == iFrmCnt) {

				vector<BBTrk> tracklet;
				vector<BBTrk>::reverse_iterator rIterT;
				bool isFound = false;
				for (rIterT = iterID->second.rbegin(); rIterT != iterID->second.rend(); rIterT++) {
					if (rIterT->fn == iFrmCnt - this->params.FRAMES_DELAY_SIZE) {

						tracklet.push_back(rIterT[0]);
						isFound = true;
						break;
					}
				}
				if (isFound /*iterID->second.back().fn - iterID->second.front().fn >= FRAMES_DELAY*/) { // reliable (with latency)
					pair< unordered_map<int, vector<BBTrk>>::iterator, bool> isEmpty = reliables.insert(unordered_map<int, vector<BBTrk>>::value_type(iterID->first, tracklet));
					if (isEmpty.second == false) {
						reliables[iterID->first].push_back(tracklet[0]);
					}

					unreliables[iterID->first].clear();
				}
				else {																					// unreliable (witout latency)
					pair< unordered_map<int, vector<BBTrk>>::iterator, bool> isEmpty = unreliables.insert(unordered_map<int, vector<BBTrk>>::value_type(iterID->first, iterID->second));
					if (isEmpty.second == false)
						unreliables[iterID->first].push_back(iterID->second.back());
				}
			}

		}

	}
}
void GMPHD_MAF::ClassifyReliableTracklets2LiveLost(int iFrmCnt, const unordered_map<int, vector<BBTrk>>& reliables, vector<BBTrk>& liveReliables, vector<BBTrk>& lostReliables, vector<BBTrk>& obss) {

	unordered_map<int, vector<BBTrk>>::const_iterator iterT;
	for (iterT = reliables.begin(); iterT != reliables.end(); iterT++) {
		if (!iterT->second.empty()) {
			//int live_fn_first = iterT->second.front().fn;	// birth at now 
			int live_fn_last = iterT->second.back().fn;		// all live tracks

			if (live_fn_last == iFrmCnt - this->params.FRAMES_DELAY_SIZE) {

				//if (this->params.TRACK_MIN_SIZE == 2) { 
				if (iterT->second.size() == 1) {
					obss.push_back(iterT->second.back());
				}
				else if (iterT->second.size() > 1) {
					liveReliables.push_back(iterT->second.back());
				}
				////}
				////else {	// Other Scenes for DPM
				////if (iterT->second.size() >= this->params.TRACK_MIN_SIZE && iterT->second.size() <= this->params.T2TA_MAX_INTERVAL) {
				////	obss.push_back(iterT->second.back());
				////}
				////else if (iterT->second.size() > this->params.T2TA_MAX_INTERVAL) {
				////liveReliables.push_back(iterT->second.back());
				////}
				////}

				/*cv::imshow((string("live_r_") + std::to_string(iterT->first)).c_str(), liveReliables.back().tmpl);
				cv::waitKey(1);*/
			}
			else if (iterT->second.back().fn < iFrmCnt - this->params.FRAMES_DELAY_SIZE) {
				//if(iterT->second.size()>1)
				lostReliables.push_back(iterT->second.back());
				//cv::imshow((string("lost") + std::to_string(iterT->first)).c_str(), lostReliables.back().tmpl);
				//cv::waitKey(0);
			}
		}
	}
}
void GMPHD_MAF::ArrangeRevivedTracklets(unordered_map<int, vector<BBTrk>>& tracks, vector<BBTrk>& lives) {

	// ID Management
	vector<BBTrk>::iterator iterT;
	for (iterT = lives.begin(); iterT != lives.end(); ++iterT) {
		if (iterT->id_associated >= 0) { // id != -1, succeed in ID recovery;

			// input parameter 1: tracks
			int size_old = tracks[iterT->id_associated].size();
			tracks[iterT->id_associated].insert(tracks[iterT->id_associated].end(), tracks[iterT->id].begin(), tracks[iterT->id].end());
			int size_new = tracks[iterT->id_associated].size();
			for (int i = size_old; i < size_new; ++i) {  // 뒤에 새로 붙은것의 ID를 복원시켜줌(associated 된 lostTrk의 것으로)
				tracks[iterT->id_associated].at(i).id = iterT->id_associated;
				tracks[iterT->id_associated].at(i).id_associated = iterT->id_associated;
				tracks[iterT->id_associated].at(i).conf = iterT->conf;

				// Copy the Covariance Matrix
				if (COV_UPDATE_T2TA)
					iterT->cov.copyTo(tracks[iterT->id_associated].at(i).cov);
			}
			tracks[iterT->id].clear();

			// this->tracksbyID
			size_old = this->tracksbyID[iterT->id_associated].size();
			this->tracksbyID[iterT->id_associated].insert(this->tracksbyID[iterT->id_associated].end(), this->tracksbyID[iterT->id].begin(), this->tracksbyID[iterT->id].end());
			size_new = this->tracksbyID[iterT->id_associated].size();
			for (int i = size_old; i < size_new; ++i) {
				this->tracksbyID[iterT->id_associated].at(i).id = iterT->id_associated;
				this->tracksbyID[iterT->id_associated].at(i).id_associated = iterT->id_associated;
				this->tracksbyID[iterT->id_associated].at(i).conf = iterT->conf;

				// Copy the Covariance Matrix
				if (COV_UPDATE_T2TA)
					iterT->cov.copyTo(this->tracksbyID[iterT->id_associated].at(i).cov);
			}
			this->tracksbyID[iterT->id].clear();

			// this->liveTrkVec (no letancy tracking)
			vector<BBTrk>::iterator iterTfw; // frame-wise (no latency)
			for (iterTfw = this->liveTrkVec.begin(); iterTfw != this->liveTrkVec.end(); ++iterTfw) {
				if (iterTfw->id == iterT->id) {
					iterTfw->id = iterT->id_associated;
					iterTfw->id_associated = iterT->id_associated;
					iterTfw->conf = iterT->conf;

					// Copy the Covariance Matrix
					if (COV_UPDATE_T2TA)
						iterT->cov.copyTo(iterTfw->cov);
					break;
				}
			}
			// this->liveTracksBatch (at t-2, t-1, t)
			for (int b = 0; b < this->params.TRACK_MIN_SIZE; ++b) {
				for (iterTfw = this->liveTracksBatch[b].begin(); iterTfw != this->liveTracksBatch[b].end(); ++iterTfw) {
					if (iterTfw->id == iterT->id) {
						iterTfw->id = iterT->id_associated;
						iterTfw->id_associated = iterT->id_associated;
						iterTfw->conf = iterT->conf;

						// Copy the Covariance Matrix
						if (COV_UPDATE_T2TA)
							iterT->cov.copyTo(iterTfw->cov);
						break;
					}
				}
			}

			// input parameter 2: lives
			if (VIS_T2TA_DETAIL)
				printf("[T2TA] Live track ID %d to Lost Track ID %d\n", iterT->id, iterT->id_associated);
			iterT->id = iterT->id_associated;
		}
	}
}
// Initialize Color Tab
void GMPHD_MAF::InitializeMatrices(cv::Mat &F, cv::Mat &Q, cv::Mat &Ps, cv::Mat &R, cv::Mat &H, const int MODEL_TYPE)
{
	const int dims_state = sym::DIMS_STATE[MODEL_TYPE];
	const int dims_obs = sym::DIMS_OBS[MODEL_TYPE];

	if (MODEL_TYPE == sym::MODEL_VECTOR::XY) {
		/* Initialize the transition matrix F, from state_t-1 to state_t

		1	0  △t	0
		0	1	0  △t
		0	0	1	0
		0	0	0	1

		△t = 구현시에는 △frame으로 즉 1이다.
		*/

		F = cv::Mat::eye(dims_state, dims_state, CV_64FC1); // identity matrix
		F.at<double>(0, 2) = 1.0;///30.0; // 30fps라 가정, 나중에 계산할때 St = St-1 + Vt-1△t (S : location) 에서 
		F.at<double>(1, 3) = 1.0;///30.0; // Vt-1△t 의해 1/30 은 사라진다. Vt-1 (1frame당 이동픽셀 / 0.0333..), △t = 0.0333...

		Q = (cv::Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0, \
			0, 0, VAR_X_VEL, 0, \
			0, 0, 0, VAR_Y_VEL);
		Q = 0.5 * Q;

		Ps = (cv::Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0, \
			0, 0, VAR_X_VEL, 0, \
			0, 0, 0, VAR_Y_VEL);

		R = (cv::Mat_<double>(dims_obs, dims_obs) << \
			VAR_X, 0, \
			0, VAR_Y);

		/*	Initialize the transition matrix H, transing the state_t to the observation_t(measurement) */
		H = (cv::Mat_<double>(dims_obs, dims_state) << \
			1, 0, 0, 0, \
			0, 1, 0, 0);
	}

}
vector<BBDet> GMPHD_MAF::MergeDetInstances(vector<BBDet>& obss, const bool& IS_MOTS, const float& sOCC_TH) {

	bool VIS_MERGE_ON = false;

	const int nFrmDets = obss.size();
	//vector<vector<float>> sIoUTable;
	//sIoUTable.resize(nFrmDets, vector<float>(nFrmDets, 0.0));

	vector<vector<int>> mergeIdxList;
	mergeIdxList.resize(nFrmDets, vector<int>());

	cv::Mat frameMerge = imgBatch[this->params.QUEUE_SIZE - 1].clone();

	/// Merge Process
	// Calculate Overlapping Ratio between Tracks' Instances
	bool merge_check = false;
	const float sMERGE_TH = this->params.MERGE_RATIO_THRESHOLD;
	for (int i = 0; i < nFrmDets; ++i) {
		//sIoUTable[i][i] = 1.0;

		for (int j = i + 1; j < nFrmDets; ++j) {

			cv::Rect ri = obss[i].rec;
			cv::Rect rj = obss[j].rec;
			cv::Mat segi, segj;

			if (IS_MOTS) {
				segi = obss[i].segMask;
				segj = obss[j].segMask;
			}

			cv::Mat mu; // union of masks

			int id_i = obss[i].id;
			int id_j = obss[j].id;

			float m_value, iou, siou; // merge metric value (measure)

			if (this->params.MERGE_METRIC == MERGE_METRIC_IOU)
				m_value = this->CalcIOU(ri, rj);
			if (this->params.MERGE_METRIC == MERGE_METRIC_mIOU) {
				m_value = this->CalcMIOU(ri, segi, rj, segj, mu);
				//sIoUTable[i][j] = m_value;
			}

			if (VIS_MERGE_ON) {
				iou = this->CalcIOU(ri, rj);
				siou = this->CalcMIOU(ri, segi, rj, segj, mu);
			}


			if (m_value >= sMERGE_TH) {

				// 조건부 인접 리스트
				mergeIdxList[i].push_back(j);
				mergeIdxList[j].push_back(i);

				if (VIS_MERGE_ON) {
					merge_check = true;

					cv::Mat mu3c; // (h, w, CV_8UC3);
					cv::cvtColor(mu, mu3c, CV_GRAY2BGR/*, CV_8UC3*/);

					cv::Vec3b overlap_color = { 255, 0, 0 };			// blue
					if (m_value >= sOCC_TH && m_value < sMERGE_TH)	overlap_color = { 0, 255, 0 }; // green
					else if (m_value >= sMERGE_TH)					overlap_color = { 0, 0, 255 }; // red


					cv::Rect ru((ri.x < rj.x) ? ri.x : rj.x, (ri.y < rj.y) ? ri.y : rj.y, mu.cols, mu.rows);

					for (int c = 0; c < ru.width; ++c) {
						for (int r = 0; r < ru.height; ++r) {
							if (mu3c.at<cv::Vec3b>(r, c)[0] >= 254) {
								mu3c.at<cv::Vec3b>(r, c) = overlap_color;
							}
						}
					}

					addWeighted(frameMerge(ru), 0.5, mu3c, 0.5, 0.0, frameMerge(ru));

					printf("[%d] Det%d(%d,%d,%d,%d,%.3f)-Det%d(%d,%d,%d,%d,%.3f) (IOU:%f, SIOU:%f)\n", this->sysFrmCnt,
						obss[i].id, ri.x, ri.y, ri.width, ri.height, obss[i].confidence,
						obss[j].id, rj.x, rj.y, rj.width, rj.height, obss[j].confidence,
						iou, siou);

					cv::imshow(string("Det") + std::to_string(id_i) + string("-Det") + std::to_string(id_j), mu3c);
					mu3c.release();
				}
			}
			mu.release();
		}
	}
	// Find the optimal mergign sets of the track instances
	// To make {ID1,ID3}{ID2,ID3} into {ID1,ID2,ID3}

	vector<int> group_min_idx = BuildMergeGroupsMaxConfID(obss, mergeIdxList);

	for (int i = 0; i < mergeIdxList.size(); ++i) {
		// printf("[%d] ID%d (group ID:%d)\n", iFrmCnt, frmTracks[i].id, frmTracks[group_min_idx[i]].id);
		mergeIdxList[i].clear();
		if (i != group_min_idx[i]) {
			mergeIdxList[group_min_idx[i]].push_back(i);
		}
	}

	// Merge
	vector<bool> merged(nFrmDets, false);
	for (int i = 0; i < nFrmDets; ++i) {

		vector<cv::Rect> rects;
		vector<cv::Mat> masks;

		rects.push_back(obss[i].rec);

		if (IS_MOTS)
			masks.push_back(obss[i].segMask);

		if (!mergeIdxList[i].empty()) {
			if (VIS_MERGE_ON) printf("[%d] Merge Det%d with", this->sysFrmCnt, obss[i].id);
			for (const auto& j : mergeIdxList[i]) {

				if (VIS_MERGE_ON) {
					printf(" Det%d,", obss[j].id);
					cv::Scalar class_color = cv::Scalar(255, 255, 255);
					DrawDetBB(frameMerge, obss[j].id, obss[j].rec, obss[j].confidence, this->params.DET_MIN_CONF, 2, class_color);
				}

				merged[j] = true;

				rects.push_back(obss[j].rec);
				if (IS_MOTS)
					masks.push_back(obss[j].segMask);
			}

			if (IS_MOTS)
				MergeSegMasksRects(masks, rects, obss[i].segMask, obss[i].rec);

			if (VIS_MERGE_ON) {
				printf("\n");
				cv::Scalar class_color = sym::OBJECT_TYPE_COLORS[this->trackObjType];
				DrawDetBB(frameMerge, obss[i].id, obss[i].rec, obss[i].confidence, this->params.DET_MIN_CONF, 3, class_color);
			}

		}
	}

	// Arrange Merged Tracks
	vector<BBDet> frmDetsM;
	int j = 0;
	for (auto& det : obss) {
		if (!merged[j]) {
			frmDetsM.push_back(det);
		}
		++j;
	}

	if (VIS_MERGE_ON) {
		//int j = 0;
		//for (auto& det : obss) {
		//	if (!merged[j]) {
		//		//cv::rectangle(frameMerge, det.rec, cv::Scalar(255, 255, 255), 3);
		//		cv::Scalar class_color = sym::OBJECT_TYPE_COLORS[this->trackObjType];
		//		DrawDetBB(frameMerge, det.id, det.rec, det.confidence, this->params.DET_MIN_CONF_CA, 3, class_color);
		//	}
		//	j++;
		//}
		//DrawFrameNumberAndFPS(iFrmCnt, frameMerge, 2.0, 2, 0, 1, 0.1);
		cv::imshow("Merged Detection Instance", frameMerge);

		int key;
		if (merge_check)	cv::waitKey();
	}

	frameMerge.release();

	return frmDetsM;
}
vector<BBTrk> GMPHD_MAF::MergeTrkInstances(vector<BBTrk>& stats, const float& sOCC_TH) {

	bool VIS_MERGE_ON = VIS_TRACK_MERGE;

	const int nFrmTracks = stats.size();
	vector<vector<float>> sIoUTable;
	sIoUTable.resize(nFrmTracks, vector<float>(nFrmTracks, 0.0));

	vector<vector<int>> mergeIdxList;
	mergeIdxList.resize(nFrmTracks, vector<int>());

	cv::Mat frameMerge = imgBatch[this->params.QUEUE_SIZE - 1].clone();

	// Decode a run-length string data to cv::Mat
	//for (auto& track : stats) {
	//	/*if (!track.isAlive) {
	//		printf("[ERROR][%d] ID%d a lost track exists\n",this->sysFrmCnt, track.id);
	//	}*/
	//	
	//	//if (VIS_MERGE_ON) {
	//	//	cv::rectangle(frameMerge, track.rec, cv::Scalar(255, 255, 255), 3);
	//	//	cv::Scalar id_color = color_tab[track.id % (MAX_OBJECTS - 1)];
	//	//	this->DrawTrkBBS(frameMerge, track.rec, id_color, 2, track.id, 0.5, "CAR");
	//	//	//cv::Mat seg3c; // (h, w, CV_8UC3);
	//	//	//cv::cvtColor(track.segMask, seg3c, CV_GRAY2BGR/*, CV_8UC3*/);
	//	//	//seg3c.setTo(id_color, seg3c);
	//	//	//addWeighted(frame(track.rec), 0.5, seg3c, 0.5, 0.0, frame(track.rec));
	//	//}
	//}
	/// Merge Process
	// Calculate Overlapping Ratio between Tracks' Instances
	bool merge_check = false;
	const float sMERGE_TH = this->params.MERGE_RATIO_THRESHOLD;
	for (int i = 0; i < nFrmTracks; ++i) {
		sIoUTable[i][i] = 1.0;
		if (stats[i].isAlive) {
			for (int j = i + 1; j < nFrmTracks; ++j) {

				if (stats[i].isAlive) {
					cv::Rect ri = stats[i].rec;
					cv::Mat segi = stats[i].segMask;
					cv::Rect rj = stats[j].rec;
					cv::Mat segj = stats[j].segMask;

					cv::Mat mu; // union of masks

					int id_i = stats[i].id;
					int id_j = stats[j].id;

					float m_value, iou, siou; // merge metric value (measure)

					if (this->params.MERGE_METRIC == MERGE_METRIC_IOU)
						m_value = this->CalcIOU(ri, rj);
					if (this->params.MERGE_METRIC == MERGE_METRIC_mIOU) {
						m_value = this->CalcMIOU(ri, segi, rj, segj, mu);
						sIoUTable[i][j] = m_value;
					}

					if (VIS_MERGE_ON) {
						iou = this->CalcIOU(ri, rj);
						siou = this->CalcMIOU(ri, segi, rj, segj, mu);
					}


					if (m_value >= sMERGE_TH) {

						// 조건부 인접 리스트
						mergeIdxList[i].push_back(j);
						mergeIdxList[j].push_back(i);

						if (VIS_MERGE_ON) {
							merge_check = true;

							cv::Mat mu3c; // (h, w, CV_8UC3);
							cv::cvtColor(mu, mu3c, CV_GRAY2BGR/*, CV_8UC3*/);

							cv::Vec3b overlap_color = { 255, 0, 0 };			// blue
							if (m_value >= sOCC_TH && m_value < sMERGE_TH)	overlap_color = { 0, 255, 0 }; // green
							else if (m_value >= sMERGE_TH)					overlap_color = { 0, 0, 255 }; // red


							cv::Rect ru((ri.x < rj.x) ? ri.x : rj.x, (ri.y < rj.y) ? ri.y : rj.y, mu.cols, mu.rows);

							for (int c = 0; c < ru.width; ++c) {
								for (int r = 0; r < ru.height; ++r) {
									if (mu3c.at<cv::Vec3b>(r, c)[0] >= 254) {
										mu3c.at<cv::Vec3b>(r, c) = overlap_color;
									}
								}
							}

							addWeighted(frameMerge(ru), 0.5, mu3c, 0.5, 0.0, frameMerge(ru));

							printf("[%d] ID%d(%d,%d,%d,%d)-ID%d(%d,%d,%d,%d) (IOU:%f, SIOU:%f)\n", this->sysFrmCnt,
								stats[i].id, ri.x, ri.y, ri.width, ri.height,
								stats[j].id, rj.x, rj.y, rj.width, rj.height,
								iou, siou);

							cv::imshow(string("ID") + std::to_string(id_i) + string("-ID") + std::to_string(id_j), mu3c);
							mu3c.release();
						}
					}
					mu.release();
				}
			}
		}
	}
	// Find the optimal mergign sets of the track instances
	// To make {ID1,ID3}{ID2,ID3} into {ID1,ID2,ID3}
	vector<int> group_min_idx = BuildMergeGroupsMinID(stats, mergeIdxList);

	for (int i = 0; i < mergeIdxList.size(); ++i) {
		// printf("[%d] ID%d (group ID:%d)\n", iFrmCnt, frmTracks[i].id, frmTracks[group_min_idx[i]].id);
		mergeIdxList[i].clear();
		if (i != group_min_idx[i]) {
			mergeIdxList[group_min_idx[i]].push_back(i);
		}
	}

	// Merge
	vector<bool> merged(nFrmTracks, false);
	for (int i = 0; i < nFrmTracks; ++i) {
		if (stats[i].isAlive) {
			vector<cv::Mat> masks;
			vector<cv::Rect> rects;

			masks.push_back(stats[i].segMask);
			rects.push_back(stats[i].rec);

			if (!mergeIdxList[i].empty()) {
				if (VIS_MERGE_ON) printf("[%d] Merge ID%d with", this->sysFrmCnt, stats[i].id);
				for (const auto& j : mergeIdxList[i]) {

					if (VIS_MERGE_ON) printf(" ID%d,", stats[j].id);

					stats[j].isAlive = false;
					merged[j] = true;

					masks.push_back(stats[j].segMask);
					rects.push_back(stats[j].rec);
				}
				if (VIS_MERGE_ON) printf("\n");

				MergeSegMasksRects(masks, rects, stats[i].segMask, stats[i].rec);
			}
		}
	}

	// Arrange Merged Tracks
	vector<BBTrk> frmTracksM;
	int j = 0;
	for (auto& track : stats) {
		if (!merged[j]) {
			frmTracksM.push_back(track);
		}
		++j;
	}

	if (VIS_MERGE_ON) {
		for (auto& track : stats) {

			if (track.isAlive) {
				cv::rectangle(frameMerge, track.rec, cv::Scalar(255, 255, 255), 3);
				cv::Scalar id_color = color_tab[track.id % (MAX_OBJECTS - 1)];
				this->DrawTrkBBS(frameMerge, track.rec, id_color, 2, track.id, 0.5, "CAR");
			}
		}

		int key;
		if (merge_check) {
			DrawFrameNumberAndFPS(this->sysFrmCnt, frameMerge, 2.0, 2, 0, 1, 0.1);
			cv::imshow("Merged Track Instance", frameMerge);
			cv::waitKey();
		}
	}

	frameMerge.release();

	return frmTracksM;
}
vector<int> GMPHD_MAF::BuildMergeGroupsMinID(const vector<BBTrk>& frmTracks, vector<vector<int>>& mergeIdxList) {
	vector<int> group_min_idx(mergeIdxList.size(), -1);

	for (int i = 0; i < mergeIdxList.size(); ++i) {
		int min_id_idx = i;

		// find global min idx and id
		int nOcc = mergeIdxList[i].size();
		if (nOcc > 0) {
			vector<bool> visitTable(mergeIdxList.size(), false);
			// depth-first search
			min_id_idx = FindGroupMinIDRecursive(i, visitTable, frmTracks, mergeIdxList, min_id_idx);
		}
		group_min_idx[i] = min_id_idx;
	}

	return group_min_idx;
}
vector<int> GMPHD_MAF::BuildMergeGroupsMaxConfID(const vector<BBDet>& frmDets, vector<vector<int>>& mergeIdxList) {
	vector<int> group_min_idx(mergeIdxList.size(), -1);

	for (int i = 0; i < mergeIdxList.size(); ++i) {
		int min_id_idx = i;

		// find global min idx and id
		int nOcc = mergeIdxList[i].size();
		if (nOcc > 0) {
			vector<bool> visitTable(mergeIdxList.size(), false);
			// depth-first search
			min_id_idx = FindGroupMaxConfIDRecursive(i, visitTable, frmDets, mergeIdxList, min_id_idx);
		}
		group_min_idx[i] = min_id_idx;
	}

	return group_min_idx;
}
void GMPHD_MAF::MergeSegMasksRects(const vector<cv::Mat>& in_masks, const vector<cv::Rect>& in_rects, cv::Mat& out_mask, cv::Rect& out_rect) {

	int min[2] = { in_rects[0].x , in_rects[0].y };
	int max[2] = { in_rects[0].x + in_rects[0].width , in_rects[0].y + in_rects[0].height };

	for (const auto& R : in_rects) {
		if (min[0] > R.x)				min[0] = R.x;
		if (min[1] > R.y)				min[1] = R.y;
		if (max[0] < (R.x + R.width))	max[0] = R.x + R.width;
		if (max[1] < (R.y + R.height))	max[1] = R.y + R.height;

	}

	int x = min[0];
	int y = min[1];
	int w = max[0] - min[0];
	int h = max[1] - min[1];

	cv::Mat Mu(h, w, CV_8UC1, cv::Scalar(0));

	int i = 0;
	float nR = 1.0 / (float)in_rects.size();
	for (const auto& R : in_rects) {

		cv::Rect Ri(R.x - x, R.y - y, R.width, R.height);
		cv::Mat Mi = Mu(Ri);

		/*if (Mi.cols != in_masks[i].cols || Mi.rows != in_masks[i].rows) {

			printf("[%d]d(%d,%d)(%dx%d)+0.5*(%dx%d)\n", this->sysFrmCnt,
				x, y, Mi.cols, Mi.rows,
				in_masks[i].cols, in_masks[i].rows);

			imshow("Mask in Frame", Mi);
			imshow("Orginal Mask", in_masks[i]);
			cv::waitKey();
		}*/

		cv::Mat in_mask_resize;
		cv::resize(in_masks[i], in_mask_resize, cv::Size(Ri.width, Ri.height));

		Mi = Mi + nR * in_mask_resize;
		in_mask_resize.release();

		i++;
	}

	// 0, 127-in_rects.size() 등 여러가지 옵션이 있을 수 있겠다. -in_rects.size() 는 CV_8UC 를 소수점 연산시 버리게되는 현상 때문
	double th = 255.0 * nR - in_rects.size();
	cv::threshold(Mu, Mu, th, 255, cv::THRESH_BINARY);
	if (!out_mask.empty()) out_mask.release();
	out_mask = Mu;
	out_rect = cv::Rect(x, y, w, h);
}
int GMPHD_MAF::FindGroupMinIDRecursive(const int& i, vector<bool>& visitTable, const vector<BBTrk>& frmTracks, const vector<vector<int>>& mergeIdxList, const int& cur_min_idx) {

	if (visitTable[i]) {
		return cur_min_idx;
	}
	else {
		visitTable[i] = true;

		int tmp_min_idx = cur_min_idx;
		int tmp_min_id = frmTracks[cur_min_idx].id;

		// find local min idx and id
		for (const auto& j : mergeIdxList[i]) {

			if (frmTracks[j].id < tmp_min_id) {
				tmp_min_idx = j;
				tmp_min_id = frmTracks[j].id;
			}
			tmp_min_idx = FindGroupMinIDRecursive(j, visitTable, frmTracks, mergeIdxList, tmp_min_idx);
		}
		return tmp_min_idx;
	}
}
int GMPHD_MAF::FindGroupMaxConfIDRecursive(const int& i, vector<bool>& visitTable, const vector<BBDet>& frmDets, const vector<vector<int>>& mergeIdxList, const int& cur_min_idx) {
	if (visitTable[i]) {
		return cur_min_idx;
	}
	else {
		visitTable[i] = true;

		int tmp_max_conf_idx = cur_min_idx;
		int tmp_max_conf_id = frmDets[cur_min_idx].id;

		// find local min idx and id
		for (const auto& j : mergeIdxList[i]) {

			if (frmDets[j].confidence > frmDets[tmp_max_conf_idx].confidence) {
				tmp_max_conf_idx = j;
				tmp_max_conf_id = frmDets[j].id;
			}
			tmp_max_conf_idx = FindGroupMaxConfIDRecursive(j, visitTable, frmDets, mergeIdxList, tmp_max_conf_idx);
		}
		return tmp_max_conf_idx;
	}
}

vector<BBTrk> GMPHD_MAF::GetTrackingResults(const int& iFrmCnt, vector<BBTrk>& liveReliables, const int& MODEL_TYPE) {

	vector<BBTrk> trackResults;

	if (iFrmCnt>=0 && (iFrmCnt + this->params.FRAMES_DELAY_SIZE) < this->iTotalFrames) {

		vector<BBTrk>::const_iterator iLive;
		for (iLive = liveReliables.begin(); iLive != liveReliables.end(); ++iLive) {
			if (iLive[0].isAlive) {

				BBTrk trk = iLive[0];
				trk.fn = iFrmCnt;
				trackResults.push_back(trk);
			}
		}
		this->allLiveReliables.push_back(trackResults);
	}
	if (GMPHD_TRACKER_MODE && this->params.FRAMES_DELAY_SIZE && ((iFrmCnt + this->params.FRAMES_DELAY_SIZE) == (this->iTotalFrames - 1))) {
		for (int OFFSET = 1; OFFSET < this->params.TRACK_MIN_SIZE; OFFSET++) {
			trackResults.clear();
			
			vector<BBTrk>::iterator iterT;
			for (iterT = this->liveTracksBatch[OFFSET].begin(); iterT != this->liveTracksBatch[OFFSET].end(); ++iterT) {

				bool isReliable = false;
				vector<BBTrk>::iterator iterTR;
				for (iterTR = liveReliables.begin(); iterTR != liveReliables.end(); ++iterTR) {
					if (iterT->id == iterTR->id) {
						isReliable = true;
						// It Must be used
						iterTR->fn = iterT->fn;
						iterTR->id = iterT->id;

						iterTR->rec = iterT->rec;
						iterTR->conf = iterT->conf;

						//if (MOT_SEGMENTATION_ON)
						iterTR->segMask = iterT->segMask.clone();

						break;
					}
				}
				// Select the targets which exists reliable targets vector.
				if (iterT->isAlive == true && isReliable) {

					BBTrk trk = iterTR[0];
					trk.fn = iterTR->fn; // (iFrmCnt + OFFSET);
					trackResults.push_back(trk);
				}
			}
			this->allLiveReliables.push_back(trackResults);
		}
	}
	return trackResults;
}