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

#include "pch.h"
#include <iostream>


// Funtions for Demo (visualization) in train or test dataset 
void RunMOTDataset(const vector<string>& seqNAMEs, const vector<string>& seqPATHs, const vector<string>& detTXTs, const vector<MOTparams>& params_in=vector<MOTparams>());
void RunMOTSequence(const int& sq, const string& seqName, const string& seqPath, const string& detTxt,
	cv::Vec2i& procObjs, int &procFrames, float &procSecs, const vector<MOTparams>& params_in=vector<MOTparams>());

void WriteMOTResults(const string& train_or_test, const vector<string>& imgPaths, const string& seqName, const int& seqNum, const string& detName,
	boost::shared_ptr<OnlineTracker> tracker);
void WriteMOTResults(const string& train_or_test, const vector<string>& imgPaths, const string& seqName, const int& seqNum, const string& detName,
	boost::shared_ptr<OnlineTracker> carTracker, boost::shared_ptr<OnlineTracker> personTracker);
void WriteFPSTxt(const string& train_or_test, const string& detName, const vector<string>& seqNAMES, 
	int& totalProcFrames, float& totalProcSecs, float& totalProcFPS,
	const vector<int>& frames= vector<int>(), const vector<float>& secs=vector<float>());

// Global Utiliy Settings for Visualization
cv::Mat color_map;
cv::Scalar color_tab[MAX_OBJECTS];

bool VIS_ON = VISUALIZATION_MAIN_ON;
bool VIS_FRAME_BY_KEY = SKIP_FRAME_BY_FRAME;

//	Dataset Type: KITTI-MOTS or MOTS20
//		Mode: Train or Test
//		Detector Name
//		Tracker Name
//		Sequence List(*.txt)

const int DB_TYPE = DB_TYPE_KITTI_MOTS;	// DB_TYPE_KITTI_MOTS, DB_TYPE_MOTS20
const string MODE = "train";			// 'train', 'test'
const string DETECTOR = "maskrcnn";		// 'maskrcnn'
const string TRACKER = "GMPHD_MAF";		// Mask-Based Affinity Fusion
const string SEQ_TXT = "seq/" + sym::DB_NAMES[DB_TYPE] + "_" + MODE +".txt" ;
const string PARAMS_TXT = "params/"+ sym::DB_NAMES[DB_TYPE] + "_" + MODE + ".txt";

const float DET_SCORE_TH_SCALE = 1.0;	// maskrcnn: 1.0
const float DET_SCORE_ALPHA = 0.0;

const bool PARALLEL_PROC_ON = true;
const int TARGET_OBJ = 0;	// 0: Car, 1: Person
 
vector<string> trkTXTsGT;

string class_names[2] = { "CARS", "PERSONS" };
cv::Point2f avg_var_all[2];
int all_occ[2] = { 0,0 };

VECx3xBBDet detectionsALL[2];
VECx3xBBTrk tracksGT[2];
vector<bool> DET_READ_FINs;

int main()
{
	/// Init
	InitColorMapTab(color_map, color_tab);

	string m_buffer = "2";
	
	cout << "Select MOTS mode in [1] a single scene or [2] a set of scenes: ";
	cin >> m_buffer;

	// [1] Run MOTS with a Single Scene in a Direct Path
	char m = m_buffer.back();
	if (m =='1') {
		DET_READ_FINs.push_back(false);
		detectionsALL[0].resize(1, VECx2xBBDet());
		detectionsALL[1].resize(1, VECx2xBBDet());

		cv::Vec2i procObjs(0, 0);
		int seq_num = 0;
		int procFrames = 0;	float procSecs = 0.0; float procFPS = 10.0;
		string seq_name = "0000";
		vector<string> seqNAMES = vector<string>(1, seq_name);

		RunMOTSequence(seq_num,
			seq_name,
			"F:/KITTI/tracking/train/image_02/"+ seq_name,
			"F:/KITTI/tracking/train/det_02_maskrcnn/"+ seq_name+".txt",
			procObjs, procFrames, procSecs);

		WriteFPSTxt(MODE, DETECTOR, seqNAMES, procFrames, procSecs, procFPS);
		printf("[Total] %d frames / %.3f secs = (%.3f FPS)\n", procFrames, procSecs, procFPS);
		printf("-------------------------------------------------------\n");
	}
	// [2]	Run MOTS with a Set of Scenes in Dataset Dirs
	else if(m == '2') {
		
		vector<string> seqNAMEs, seqPATHs;
		vector<string> detTXTs, trkTXTs;
		vector<MOTparams> seqParams;

		ReadDatasetInfo(DB_TYPE, MODE, DETECTOR, SEQ_TXT, PARAMS_TXT, seqNAMEs, seqPATHs, detTXTs, trkTXTsGT, seqParams);

		DET_READ_FINs.resize(seqPATHs.size(), false);
		detectionsALL[0].resize(seqPATHs.size(), VECx2xBBDet());
		detectionsALL[1].resize(seqPATHs.size(), VECx2xBBDet());

		//	Run MOTS in the Dataset (including loading the dataset images & segmentation results)
		RunMOTDataset(seqNAMEs, seqPATHs, detTXTs, seqParams);
	}
	return 0;
}
void RunMOTDataset(const vector<string>& seqNAMEs, const vector<string>& seqPATHs, const vector<string>& detTXTs, const vector<MOTparams>& params_in) {

	int totalProcFrames = 0;  float totalProcSecs = 1.0;  float totalProcFPS = 10.0;
	vector<int> frames(seqPATHs.size(), 0);
	vector<float> secs(seqPATHs.size(), 0.0);
	vector<cv::Vec2i> objects(seqPATHs.size(), cv::Vec2i(0, 0));

	for (int sq = 0; sq < seqPATHs.size(); ++sq) {

		cv::Vec2i procObjs(0, 0);	int procFrames = 0;	float procSecs = 0.0;
		/*-------------------------------------------------------------------------------------------------------*/
		RunMOTSequence(sq, seqNAMEs[sq], seqPATHs[sq], detTXTs[sq], procObjs, procFrames, procSecs, params_in);
		/*-------------------------------------------------------------------------------------------------------*/
		frames[sq] = procFrames;
		secs[sq] = procSecs;
		objects[sq] = procObjs;
		// cout << procFrames << " " << procSecs << " " << procObjs <<" "<< endl;
	}

	WriteFPSTxt(MODE, DETECTOR, seqNAMEs, totalProcFrames, totalProcSecs, totalProcFPS, frames, secs);
	printf("[Total] %d frames / %.3f secs = (%.3f FPS)\n", totalProcFrames, totalProcSecs, totalProcFPS);
	printf("-------------------------------------------------------\n");
}
void RunMOTSequence(const int& sq, const string& seqName, const string& seqPath, const string& detTxt, 
	cv::Vec2i& procObjs, int &procFrames, float &procSecs, const vector<MOTparams>& params_in) {

	// Read Sequence Data (Images and Detections)
	vector<string> imgs = ReadFilesInPath(boost::filesystem::path(seqPath));
	//cout << "[ "<< imgs.size() <<" images ]";

	// Types: 2D Bounding Box, 3D Box, 3D Point Cloud, 2D Intance Segments 
	VECx2xBBDet *detsSeq[2]; // [0] Car, [1] Person, 
	if (!DET_READ_FINs[sq]) {
		cout << "		Reading the Detections of Seq:" << seqName << "....";
		VECx2xBBDet input_dets = ReadDetectionsSeq(DB_TYPE, DETECTOR, detTxt, detectionsALL[0][sq], detectionsALL[1][sq]);
		cout << "[done]" << endl;
		input_dets.clear();
	}
	DET_READ_FINs[sq] = true;

	detsSeq[0] = &detectionsALL[0][sq];
	detsSeq[1] = &detectionsALL[1][sq];

	// Specification of the Data (Images and Detections)
	const int& nImages = (int)imgs.size();
	cv::Mat tImg = cv::imread(imgs[0]);
	int frmWidth = tImg.cols, frmHeight = tImg.rows;
	int frmSize = frmWidth * frmHeight;
	tImg.release();
	const int& nDetFrames = (detsSeq[0]->size() > detsSeq[1]->size()) ? (int)detsSeq[0]->size() : (int)detsSeq[1]->size();
	const int& nTrackers = 2;

	// Init a Tracker with Parameters
	boost::shared_ptr<GMPHD_MAF> MOTSParallel[2];	// [0] car and [1] person

	// Car Tracker
	MOTSParallel[0] = boost::shared_ptr<GMPHD_MAF>(new  GMPHD_MAF());
	MOTSParallel[0]->SetSeqName(seqName);
	MOTSParallel[0]->SetTotalFrames(nImages);

	// Pedestrian Tracker
	MOTSParallel[1] = boost::shared_ptr<GMPHD_MAF>(new  GMPHD_MAF());
	MOTSParallel[1]->SetSeqName(seqName);
	MOTSParallel[1]->SetTotalFrames(nImages);

	// Tracker Settings by using a parmeter file , e.g., KITTI_train.txt
	if (params_in.size()) {
		for (const auto& param : params_in) {
			int obj_idx = param.OBJ_TYPE - 1;
			MOTSParallel[obj_idx]->SetParams(param);
		}
	}
	// Default Tracker Settings
	else {
		printf("Default parameters are set.\n");
		MOTSParallel[0]->SetParams(MOTparams(sym::OBJECT_TYPE::CAR,
			0.6, \
			sym::TRACK_INIT_MIN[0],	/*{1,2,3,4,5}*/\
			sym::TRACK_T2TA_MAX[3],	/*{5,10,15,20,30,60,80,100}*/\
			MERGE_METRIC_mIOU,		/*{0:SIOA, 1:IOU, 2:sIOU}*/\
			sym::MERGE_THS[1],		/*{0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}*/\
			sym::VEL_UP_ALPHAS[4],	/*{0.0(0),...,0.4(4),0.5(5),...,0.99(11)}, KITTI-MOTS: 0.4(car)*/\
			sym::TRACK_INIT_MIN[0] * 10,
			sym::FRAME_OFFSETS[DB_TYPE],
			sym::AFFINITY_OPT::KCF, sym::AFFINITY_OPT::KCF/*KITTI-MOTS (maskrcnn, BB2Seg+RRC): KCF, KCF*/,
			false, false/*MASK_ON*//*KITTI-MOTS (maskrcnnm): false, false*/,
			cv::Vec2f(0.5f, 0.85f), cv::Vec2f(0.5f, 0.85f)/*KITTI-MOTS (maskrcnn): 0.5, 0.85*/,
			cv::Vec2f(0.1f, 0.9f), cv::Vec2f(0.1f, 0.9f),
			cv::Vec2b(false, false)));/*GATE_ON, MOTS20: true, true, KITTI-MOTS: false, false*/

		MOTSParallel[1]->SetParams(MOTparams(sym::OBJECT_TYPE::PEDESTRIAN,
			0.7, \
			sym::TRACK_INIT_MIN[0],	/*{1,2,3,4,5}*/\
			sym::TRACK_T2TA_MAX[3],	/*{5,10,15,20,30,60,80,100}, MOT:30?, MOTS20:30, KITTI-MOTS: 20*/\
			MERGE_METRIC_mIOU,		/*{0:SIOA, 1:IOU, 2:sIOU}*/\
			sym::MERGE_THS[1],		/*{0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}*/\
			sym::VEL_UP_ALPHAS[5],	/*{0.0(0),...,0.4(4),0.5(5),...0.95(9),...,0.99(11)}, MOTS20:0.4, KITTI-MOTS: 0.5(ped)*/\
			sym::TRACK_INIT_MIN[0] * 10,
			sym::FRAME_OFFSETS[DB_TYPE],
			sym::AFFINITY_OPT::KCF, sym::AFFINITY_OPT::KCF/*MOTS20: MAF, GMPHD, KITTI-MOTS: KCF, KCF*/,
			true, true/*MASK_ON, MOTS20: true, true, KITTI-MOTS: true, true*/,
			cv::Vec2f(0.5f, 0.85f), cv::Vec2f(0.5f, 0.85f)/*MOTS20: 0.1f, 0.7f, KITTI-MOTS: 0.5f, 0.85f*/,
			cv::Vec2f(0.1f, 0.9f), cv::Vec2f(0.1f, 0.9f),
			cv::Vec2b(false, false)));/*GATE_ON, MOTS20: true, true, KITTI-MOTS: false, false*/
	}
	
	int sumValidObjs[2] = { 0,0 };

	// Run Tracking Process (MOTS)
	int iFrmCnt;
	for (iFrmCnt = 0; (iFrmCnt < nImages) && (iFrmCnt < nDetFrames); ++iFrmCnt) {
	
		cerr << "(" << iFrmCnt + sym::FRAME_OFFSETS[DB_TYPE] << "/" << nImages << ")";
		cerr << "\r";

		cv::Mat img = cv::imread(imgs[iFrmCnt]);
		cv::Mat imgDet = img;
		cv::Mat imgTrk = img.clone();
		
		vector<BBTrk> out_trks[2];
		int nProcDets[2] = { 0,0 };

		// Online Tracking (frame by frame process)
		int64 t_start = cv::getTickCount();
		if (PARALLEL_PROC_ON) {
			Concurrency::parallel_for(0, nTrackers, [&](int p) {
				//for (int p=0;p< nTrackers;++p){
				cv::Mat imgTrkProc = imgTrk.clone();
				MOTSParallel[p]->RunMOTS(iFrmCnt, imgTrkProc, (*detsSeq[p])[iFrmCnt], out_trks[p]);
				sumValidObjs[p] += nProcDets[p];
				imgTrkProc.release();
			}
			);
		}
		else {
			nProcDets[TARGET_OBJ] = MOTSParallel[TARGET_OBJ]->RunMOTS(iFrmCnt, imgTrk, (*detsSeq[TARGET_OBJ])[iFrmCnt], out_trks[TARGET_OBJ]);
			sumValidObjs[TARGET_OBJ] += nProcDets[TARGET_OBJ];
		}
		int64 t_end = cv::getTickCount();
		double secs = (t_end - t_start) / cv::getTickFrequency();
		procSecs += secs;

		// Visualization
		cv::Mat imgTrk_vis = MOTSParallel[TARGET_OBJ]->imgBatch[MOTSParallel[TARGET_OBJ]->GetParams().QUEUE_SIZE-1-MOTSParallel[TARGET_OBJ]->GetParams().FRAMES_DELAY_SIZE].clone();
		if (VISUALIZATION_MAIN_ON) {

			// Drawing MOTS Results
			if (PARALLEL_PROC_ON) {
				DrawDetections(imgDet, (*detsSeq[0])[iFrmCnt], color_tab, DB_TYPE);
				//DrawTracker(imgTrk_vis, out_trks[0], "Car", DB_TYPE, color_tab, 3, 0.8);
				DrawTrackerInstances(imgTrk_vis, out_trks[0], class_names[0], DB_TYPE, color_tab, 2, 0.7);

				DrawDetections(imgDet, (*detsSeq[1])[iFrmCnt], color_tab, DB_TYPE);
				//DrawTracker(imgTrk_vis, out_trks[1], "Person", DB_TYPE, color_tab, 3, 0.8);
				DrawTrackerInstances(imgTrk_vis, out_trks[1], class_names[1], DB_TYPE, color_tab, 2, 0.7);
			}
			else {
				DrawDetections(imgDet, (*detsSeq[TARGET_OBJ])[iFrmCnt], color_tab, DB_TYPE);
				//DrawTracker(imgTrk_vis, out_trks[TARGET_OBJ], class_names[TARGET_OBJ], DB_TYPE, color_tab, 3, 0.8);
				DrawTrackerInstances(imgTrk_vis, out_trks[TARGET_OBJ], class_names[TARGET_OBJ], DB_TYPE, color_tab, 2, 0.7);
			}

			// Resizing 
			cv::Mat visDet = imgDet, visTrk = imgTrk_vis;
			cv::Mat imgDetResized, imgTrkResized;
			string winTitle[2] = { "Detection", "Tracking" };
			cv::Size viewSize(frmWidth, frmHeight);
			if (VISUALIZATION_MAIN_ON && !iFrmCnt) {
				cv::namedWindow(winTitle[0]);	cv::namedWindow(winTitle[1]);
				cv::moveWindow(winTitle[0], 0, 0);
				cv::moveWindow(winTitle[1], 0, viewSize.height + 20);
			}
			// Resizing
			if (VISUALIZATION_RESIZE_ON) {
				float resize_ratio = 2.0 / 3.0;

				cv::resize(imgDet, imgDetResized, cv::Size(frmWidth * resize_ratio, frmHeight * resize_ratio));
				cv::resize(imgTrk_vis, imgTrkResized, cv::Size(frmWidth * resize_ratio, frmHeight * resize_ratio));

				visDet = imgDetResized;
				visTrk = imgTrkResized;

				viewSize.width = imgDetResized.cols;
				viewSize.height = imgDetResized.rows;
			}

			DrawFrameNumberAndFPS(iFrmCnt, visDet, 2.0, 2, sym::FRAME_OFFSETS[DB_TYPE], 1, secs);
			DrawFrameNumberAndFPS(iFrmCnt- MOTSParallel[TARGET_OBJ]->GetParams().FRAMES_DELAY_SIZE, visTrk, 2.0, 2, sym::FRAME_OFFSETS[DB_TYPE], 1, secs);
		
			// Displaying
			if (VISUALIZATION_MAIN_ON) {
				cv::imshow(winTitle[0], visDet);
				cv::imshow(winTitle[1], visTrk);

				if (SKIP_FRAME_BY_FRAME)	cv::waitKey();
				else				cv::waitKey(10);
			}
			if (SAVE_IMG_ON) {
				SaveResultImgs(DB_TYPE, MODE, DETECTOR, seqName, iFrmCnt + sym::FRAME_OFFSETS[DB_TYPE], visDet, MOTSParallel[TARGET_OBJ]->GetParams().DET_MIN_CONF, "det");
				SaveResultImgs(DB_TYPE, MODE, DETECTOR, seqName, iFrmCnt + sym::FRAME_OFFSETS[DB_TYPE], visTrk, MOTSParallel[TARGET_OBJ]->GetParams().DET_MIN_CONF, "trk");
			}
			// Release
			if (VISUALIZATION_RESIZE_ON) {
				imgDetResized.release();
				imgTrkResized.release();
			}
		}
		// Release Memory
		/*for (int t = 0; t < nTrackers; ++t) {
			for (auto& det : in_dets[t]) {
				det.segMask.release();
			}
			in_dets[t].clear();

			for (auto& track : out_trks[t]) track.Destroy();
			out_trks[t].clear();
		}*/
		imgDet.release();
		imgTrk.release();
		imgTrk_vis.release();
	}

	procObjs[0] = sumValidObjs[0];
	procObjs[1] = sumValidObjs[1];
	procFrames = iFrmCnt;
	printf(" ... %d frames / %.3f secs = (%.3f FPS)\n", procFrames, procSecs, (double)procFrames / procSecs);

	// Write Tracking Results (*.txt)
	if (PARALLEL_PROC_ON) {
		WriteMOTResults(MODE, imgs, seqName, sq, DETECTOR, MOTSParallel[0], MOTSParallel[1]);
		MOTSParallel[0]->Destory();
	}
	else {
		WriteMOTResults(MODE, imgs, seqName, sq, DETECTOR, MOTSParallel[TARGET_OBJ]);
	}
	MOTSParallel[TARGET_OBJ]->Destory();

}
void WriteFPSTxt(const string& train_or_test, const string& detName, const vector<string>& seqNAMES,
	int& totalProcFrames, float& totalProcSecs, float& totalProcFPS,
	const vector<int>& frames, const vector<float>& secs) {

	char filePath[256], filePathINTP[256];

	if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS) {
		sprintf_s(filePath, 256, "res\\KITTI\\%s\\_speed_%s.txt", train_or_test.c_str(), detName.c_str());
		//sprintf_s(filePathINTP, 256, "res\\KITTI\\%s\\%s\\%s_intp\\_speed.txt", train_or_test, detName, strThDetConf);
	}
	else if (DB_TYPE == DB_TYPE_MOTS20) {
		sprintf_s(filePath, 256, "res\\MOTS20\\%s\\_speed_%s.txt", train_or_test.c_str(), detName.c_str());
		//sprintf_s(filePathINTP, 256, "res\\MOTS20\\%s\\%s\\%s_intp\\_speed.txt", train_or_test, detName, strThDetConf);
	}

	FILE *fp;
	//FILE *fp_intp;
	fopen_s(&fp, filePath, "w+");
	//fopen_s(&fp_intp, filePathINTP, "w+");

	int totalFrames = 0;
	double totalSecs = 0.0;
	if (seqNAMES.size() > 1) {
		for (int sq = 0; sq < seqNAMES.size(); ++sq) {

			totalFrames += frames[sq];
			totalSecs += secs[sq];

			fprintf_s(fp, "%s: %d/%.3f = %.3f FPS\n", seqNAMES[sq].c_str(), frames[sq], secs[sq], frames[sq] / secs[sq]);
			//fprintf_s(fp_intp, "%s: %d/%.2lf = %.2lf FPS\n", seqNAMES[sq].c_str(), frames[sq], secs[sq], frames[sq] / secs[sq]);
		}

		totalProcFrames = totalFrames;
		totalProcSecs = totalSecs;
		totalProcFPS = totalFrames / totalSecs;

		fprintf_s(fp, "[Total]: %d/%.3f = %.3f FPS\n", totalFrames, totalSecs, totalProcFPS);
		//fprintf_s(fp_intp, "[Total]: %d/%.2lf = %.2lf FPS\n", totalFrames, totalSecs, totalProcFPS);
	}
	else {
		totalProcFPS = totalProcFrames / totalProcSecs;
		fprintf_s(fp, "%s: %d/%.3f = %.3f FPS\n", seqNAMES[0].c_str(), totalProcFrames, totalProcSecs, totalProcFPS);
	}
	fclose(fp);
	//fclose(fp_intp);
}
// The Function for Writing the Tracking Results into a Text File
void WriteMOTResults(const string& train_or_test, const vector<string>& imgPaths, const string& seqName, const int& seqNum, const string& detName, boost::shared_ptr<OnlineTracker> tracker) {

	char filePath[256];

	if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS) {
		sprintf_s(filePath, 256, "res\\KITTI\\%s\\%s.txt", train_or_test.c_str(), seqName.c_str());
	}
	else if (DB_TYPE == DB_TYPE_MOTS20) {
		sprintf_s(filePath, 256, "res\\MOTS20\\%s\\%s.txt", train_or_test.c_str(), seqName.c_str());
	}

	cout << "   " << TRACKER << ":" << filePath << endl;

	// 1. Write the tracking results without interpolation 
	FILE* fp;
	fopen_s(&fp, filePath, "w+");

	for (int i = 0; i < tracker->allLiveReliables.size(); ++i) { // frame by frame
		if (!tracker->allLiveReliables[i].empty()) {

			// Handling the Ovelapping Segments
			if (DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {
				int frmW = tracker->frmWidth; int frmH = tracker->frmHeight;
				cv::Mat maskMAT(frmH, frmW, CV_8UC1, cv::Scalar(0));

				for (int tr = 0; tr < tracker->allLiveReliables[i].size(); ++tr) {

					cv::Rect bboxTmp = tracker->allLiveReliables[i][tr].rec;
					bboxTmp.width = tracker->allLiveReliables[i][tr].segMask.cols;
					bboxTmp.height = tracker->allLiveReliables[i][tr].segMask.rows;

					cv::Rect bboxInFrm = tracker->RectExceptionHandling(tracker->frmWidth, tracker->frmHeight,
						cv::Point(bboxTmp.x, bboxTmp.y), cv::Point(bboxTmp.x + bboxTmp.width, bboxTmp.y + bboxTmp.height));
					tracker->allLiveReliables[i][tr].rec = bboxInFrm;

					cv::Rect bbox_ = cv::Rect(bboxInFrm.x - bboxTmp.x, bboxInFrm.y - bboxTmp.y, bboxInFrm.width, bboxInFrm.height);

					cv::Mat segTmp = tracker->allLiveReliables[i][tr].segMask.clone();
					tracker->allLiveReliables[i][tr].segMask.release();
					tracker->allLiveReliables[i][tr].segMask = segTmp(bbox_).clone();
					segTmp.release();

					cv::Mat *segMaskPtr = &(tracker->allLiveReliables[i][tr].segMask);
					cv::Rect bbox = tracker->allLiveReliables[i][tr].rec;
					bbox.width = segMaskPtr->cols;
					bbox.height = segMaskPtr->rows;

					for (int i = 0; i < segMaskPtr->cols; i++) {
						for (int j = 0; j < segMaskPtr->rows; j++) {
							if (segMaskPtr->at<uchar>(j, i) > 0) {

								if (maskMAT.at<uchar>(bbox.y + j, bbox.x + i) > 0)
									segMaskPtr->at<uchar>(j, i) = 0;
								else
									maskMAT.at<uchar>(bbox.y + j, bbox.x + i) = segMaskPtr->at<uchar>(j, i);
							}
						}
					}
				}
				maskMAT.release();
			}

			for (int tr = 0; tr < tracker->allLiveReliables[i].size(); ++tr) {
				if (DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20) {

					fprintf_s(fp, "%d,%d,%.2f,%.2f,%.2f,%.2f,-1,-1,-1,-1\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
						tracker->allLiveReliables[i][tr].id, \
						(float)tracker->allLiveReliables[i][tr].rec.x, (float)tracker->allLiveReliables[i][tr].rec.y, \
						(float)(tracker->allLiveReliables[i][tr].rec.width), \
						(float)(tracker->allLiveReliables[i][tr].rec.height));
				}
				else if (DB_TYPE == DB_TYPE_KITTI) {
					/// Detection File Format in the KITTI Benchmark
					// token: " "

					// [frame id type -1 -1 -10 left top right bottom -1000 -1000 -1000 -10 -1 -1 -1]
					// int(0)..,str(2)...floats(6-9)...float(17)

					// type: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
					// interested types: 'Car', 'Van', 'Pedestrian', 'Person_sitting', (optional)'Cyclist'

					cv::Rect bbox = tracker->allLiveReliables[i][tr].rec;
					if (EXCLUDE_BBOX_OUT_OF_FRAME) {
						bbox = tracker->RectExceptionHandling(tracker->frmWidth, tracker->frmHeight, bbox);
					}

					fprintf_s(fp, "%d %d %s -1 -1 -10 %.2f %.2f %.2f %.2f -1000 -1000 -1000 -10 -1 -1 %.5f\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
						tracker->allLiveReliables[i][tr].id, sym::OBJECT_STRINGS[tracker->GetObjType()].c_str(), \
						(float)bbox.x, (float)bbox.y, (float)(bbox.x + bbox.width), (float)(bbox.y + bbox.height), \
						tracker->allLiveReliables[i][tr].conf);

				}
				else if (DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {

					int id = tracker->allLiveReliables[i][tr].id;
					int objTypeMOTS = (tracker->GetObjType() == sym::OBJECT_TYPE::CAR) ? 1 : 2;

					// Convert "cv::Mat mask" to "std::string Rle Encoded mask"
					cv::Rect bbox = tracker->allLiveReliables[i][tr].rec;

					// Frame ID objectType imgWidth imgHeight RleMask
					if (bbox.width > 0 && bbox.height > 0/* && id!=480 */) {

						std::string maskSTR = CvtMAT2RleSTR(tracker->allLiveReliables[i][tr].segMask, cv::Size(tracker->frmWidth, tracker->frmHeight), bbox, false);

						fprintf_s(fp, "%d %d %d %d %d %s\n", i + sym::FRAME_OFFSETS[DB_TYPE],
							id, objTypeMOTS, tracker->frmHeight, tracker->frmWidth, maskSTR.c_str());

						//fprintf_s(fp_tracking_only, "%d %d %.6f\n", tracker->allLiveReliables[i][tr].det_id, id, tracker->allLiveReliables[i][tr].det_confidence);

					}
				}
			}
		}
	}

	fclose(fp);
}
// The Function for Writing the Tracking Results into a Text File
void WriteMOTResults(const string& train_or_test, const vector<string>& imgPaths, const string& seqName, const int& seqNum,
	const string& detName, boost::shared_ptr<OnlineTracker> carTracker, boost::shared_ptr<OnlineTracker> personTracker) {

	char filePath[256];

	if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS) {
		sprintf_s(filePath, 256, "res\\KITTI\\%s\\%s.txt", train_or_test.c_str(), seqName.c_str());
	}
	else if (DB_TYPE == DB_TYPE_MOTS20) {
		sprintf_s(filePath, 256, "res\\MOTS20\\%s\\%s.txt", train_or_test.c_str(), seqName.c_str());
	}

	cout << "   " << TRACKER << ":" << filePath << endl;

	// 1. Write the tracking resulsts without interpolation 
	FILE* fp;
	fopen_s(&fp, filePath, "w+");

	int nCarTrackFrames = carTracker->allLiveReliables.size();
	int mPersonTrackFrames = personTracker->allLiveReliables.size();
	int nMaxTrackFrames = nCarTrackFrames > mPersonTrackFrames ? nCarTrackFrames : mPersonTrackFrames;
	for (int i = 0; i < nMaxTrackFrames; ++i) { // frame by frame
		bool bCarWrite = false;
		bool bPersonWrite = false;

		if (i >= nCarTrackFrames)		bCarWrite = false;
		else {
			bCarWrite = !carTracker->allLiveReliables[i].empty();
		}

		if (i >= mPersonTrackFrames)	bPersonWrite = false;
		else {
			bPersonWrite = !personTracker->allLiveReliables[i].empty();
		}
		if (bCarWrite || bPersonWrite) {

			// Handling the Ovelapping Segments
			if (DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {
				int frmW = carTracker->frmWidth; int frmH = carTracker->frmHeight;

				cv::Mat maskMAT(frmH, frmW, CV_8UC1, cv::Scalar(0));
				//printf("1");
				for (int tr = 0; tr < carTracker->allLiveReliables[i].size(); ++tr) {

					cv::Rect bboxTmp = carTracker->allLiveReliables[i][tr].rec;
					int segW = carTracker->allLiveReliables[i][tr].segMask.cols;
					int segH = carTracker->allLiveReliables[i][tr].segMask.rows;
					bboxTmp.width = (bboxTmp.width > segW) ? segW : bboxTmp.width;
					bboxTmp.height = (bboxTmp.height > segH) ? segH : bboxTmp.height;

					cv::Rect bboxInFrm = carTracker->RectExceptionHandling(carTracker->frmWidth, carTracker->frmHeight,
						cv::Point(bboxTmp.x, bboxTmp.y), cv::Point(bboxTmp.x + bboxTmp.width, bboxTmp.y + bboxTmp.height));
					carTracker->allLiveReliables[i][tr].rec = bboxInFrm;

					cv::Rect bbox_ = cv::Rect(bboxInFrm.x - bboxTmp.x, bboxInFrm.y - bboxTmp.y, bboxInFrm.width, bboxInFrm.height);

					cv::Mat segTmp = carTracker->allLiveReliables[i][tr].segMask.clone();
					carTracker->allLiveReliables[i][tr].segMask.release();
					carTracker->allLiveReliables[i][tr].segMask = segTmp(bbox_).clone();

					cv::Mat *segMaskPtr = &(carTracker->allLiveReliables[i][tr].segMask);
					cv::Rect bbox = carTracker->allLiveReliables[i][tr].rec;
					bbox.width = segMaskPtr->cols;
					bbox.height = segMaskPtr->rows;

					if (false/*i >= 69*/) {
						bool viewDetail = true;
						if (viewDetail/*i >= 106*/) {
							//cv::imshow(to_string(id), tracker->allLiveReliables[i][tr].segMask);
							printf("%d car%d img%dx%d (%d,%d,%d,%d)(%dx%d)->(%d,%d,%d,%d)(%dx%d)\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
								carTracker->allLiveReliables[i][tr].id, carTracker->frmHeight, carTracker->frmWidth,
								bboxTmp.x, bboxTmp.y, bboxTmp.width, bboxTmp.height,
								segTmp.cols, segTmp.rows,
								bbox.x, bbox.y, bbox.width, bbox.height,
								carTracker->allLiveReliables[i][tr].segMask.cols, carTracker->allLiveReliables[i][tr].segMask.rows);
							//cv::waitKey();
							viewDetail = true;
						}
					}
					segTmp.release();

					for (int i = 0; i < segMaskPtr->cols; i++) {
						for (int j = 0; j < segMaskPtr->rows; j++) {
							if (segMaskPtr->at<uchar>(j, i) > 0) {
								if (maskMAT.at<uchar>(bbox.y + j, bbox.x + i) > 0)
									segMaskPtr->at<uchar>(j, i) = 0;
								else
									maskMAT.at<uchar>(bbox.y + j, bbox.x + i) = segMaskPtr->at<uchar>(j, i);
							}
						}
					}
				}
				//printf("2");
				for (int tr = 0; tr < personTracker->allLiveReliables[i].size(); ++tr) {

					cv::Rect bboxTmp = personTracker->allLiveReliables[i][tr].rec;
					int segW = personTracker->allLiveReliables[i][tr].segMask.cols;
					int segH = personTracker->allLiveReliables[i][tr].segMask.rows;
					bboxTmp.width = (bboxTmp.width > segW) ? segW : bboxTmp.width;
					bboxTmp.height = (bboxTmp.height > segH) ? segH : bboxTmp.height;

					cv::Rect bboxInFrm = personTracker->RectExceptionHandling(personTracker->frmWidth, personTracker->frmHeight,
						cv::Point(bboxTmp.x, bboxTmp.y), cv::Point(bboxTmp.x + bboxTmp.width, bboxTmp.y + bboxTmp.height));

					personTracker->allLiveReliables[i][tr].rec = bboxInFrm;

					cv::Rect bbox_ = cv::Rect(bboxInFrm.x - bboxTmp.x, bboxInFrm.y - bboxTmp.y, bboxInFrm.width, bboxInFrm.height);

					cv::Mat segTmp = personTracker->allLiveReliables[i][tr].segMask.clone();
					personTracker->allLiveReliables[i][tr].segMask.release();
					personTracker->allLiveReliables[i][tr].segMask = segTmp(bbox_).clone();

					cv::Mat *segMaskPtr = &(personTracker->allLiveReliables[i][tr].segMask);
					cv::Rect bbox = personTracker->allLiveReliables[i][tr].rec;
					bbox.width = segMaskPtr->cols;
					bbox.height = segMaskPtr->rows;

					segTmp.release();

					for (int i = 0; i < segMaskPtr->cols; i++) {
						for (int j = 0; j < segMaskPtr->rows; j++) {
							if (segMaskPtr->at<uchar>(j, i) > 0) {
								if (maskMAT.at<uchar>(bbox.y + j, bbox.x + i) > 0)
									segMaskPtr->at<uchar>(j, i) = 0;
								else
									maskMAT.at<uchar>(bbox.y + j, bbox.x + i) = segMaskPtr->at<uchar>(j, i);
							}
						}
					}
				}
				maskMAT.release();
			}
			//printf("3");
			for (int tr = 0; tr < carTracker->allLiveReliables[i].size(); ++tr) {
				if (DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20) {
					fprintf_s(fp, "%d,%d,%.2lf,%.2f,%.2f,%.2f,-1,-1,-1,-1\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
						carTracker->allLiveReliables[i][tr].id, \
						(float)carTracker->allLiveReliables[i][tr].rec.x, (float)carTracker->allLiveReliables[i][tr].rec.y, \
						(float)(carTracker->allLiveReliables[i][tr].rec.width), \
						(float)(carTracker->allLiveReliables[i][tr].rec.height));
				}
				else if (DB_TYPE == DB_TYPE_KITTI ) {
					/// Detection File Format in the KITTI Benchmark
					// token: " "

					// [frame id type -1 -1 -10 left top right bottom -1000 -1000 -1000 -10 -1 -1 -1]
					// int(0)..,str(2)...floats(6-9)...float(17)

					// type: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
					// interested types: 'Car', 'Van', 'Pedestrian', 'Person_sitting', (optional)'Cyclist'

					cv::Rect bbox = carTracker->allLiveReliables[i][tr].rec;
					if (EXCLUDE_BBOX_OUT_OF_FRAME) {
						bbox = carTracker->RectExceptionHandling(carTracker->frmWidth, carTracker->frmHeight, bbox);
					}

					fprintf_s(fp, "%d %d %s -1 -1 -10 %.2f %.2f %.2f %.2f -1000 -1000 -1000 -10 -1 -1 %.5f\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
						carTracker->allLiveReliables[i][tr].id, sym::OBJECT_STRINGS[carTracker->GetObjType()].c_str(), \
						(float)bbox.x, (float)bbox.y, (float)(bbox.x + bbox.width), (float)(bbox.y + bbox.height), \
						carTracker->allLiveReliables[i][tr].conf);


				}
				else if (DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {
					int id = carTracker->allLiveReliables[i][tr].id;
					int objTypeMOTS = (carTracker->GetObjType() == sym::OBJECT_TYPE::CAR) ? 1 : 2;

					// Convert "cv::Mat mask" to "std::string Rle Encoded mask"
					cv::Rect bbox = carTracker->allLiveReliables[i][tr].rec;

					// Frame ID objectType imgWidth imgHeight RleMask
					if (bbox.width > 0 && bbox.height > 0 /*maskSTR.compare("TUT>") != 0*/) {
						std::string maskSTR = CvtMAT2RleSTR(carTracker->allLiveReliables[i][tr].segMask,
							cv::Size(carTracker->frmWidth, carTracker->frmHeight), bbox);

						fprintf_s(fp, "%d %d %d %d %d %s\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
							id, objTypeMOTS, carTracker->frmHeight, carTracker->frmWidth, maskSTR.c_str());

						//fprintf_s(fp_tracking_only, "%d %d %f\n", carTracker->allLiveReliables[i][tr].det_id, id, carTracker->allLiveReliables[i][tr].det_confidence);
					}
				}
			}
			//printf("4");
			for (int tr = 0; tr < personTracker->allLiveReliables[i].size(); ++tr) {
				if (DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20) {
					fprintf_s(fp, "%d,%d,%.2lf,%.2f,%.2f,%.2f,-1,-1,-1,-1\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
						personTracker->allLiveReliables[i][tr].id, \
						(float)personTracker->allLiveReliables[i][tr].rec.x, (float)personTracker->allLiveReliables[i][tr].rec.y, \
						(float)(personTracker->allLiveReliables[i][tr].rec.width), \
						(float)(personTracker->allLiveReliables[i][tr].rec.height));
				}
				else if (DB_TYPE == DB_TYPE_KITTI) {
					/// Detection File Format in the KITTI Benchmark
					// token: " "

					// [frame id type -1 -1 -10 left top right bottom -1000 -1000 -1000 -10 -1 -1 -1]
					// int(0)..,str(2)...floats(6-9)...float(17)

					// type: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'
					// interested types: 'Car', 'Van', 'Pedestrian', 'Person_sitting', (optional)'Cyclist'

					cv::Rect bbox = personTracker->allLiveReliables[i][tr].rec;
					if (EXCLUDE_BBOX_OUT_OF_FRAME) {
						bbox = personTracker->RectExceptionHandling(personTracker->frmWidth, personTracker->frmHeight, bbox);
					}

					fprintf_s(fp, "%d %d %s -1 -1 -10 %.2f %.2f %.2f %.2f -1000 -1000 -1000 -10 -1 -1 %.5f\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
						personTracker->allLiveReliables[i][tr].id, sym::OBJECT_STRINGS[personTracker->GetObjType()].c_str(), \
						(float)bbox.x, (float)bbox.y, (float)(bbox.x + bbox.width), (float)(bbox.y + bbox.height), \
						personTracker->allLiveReliables[i][tr].conf);

				}
				else if (DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {

					int id = personTracker->allLiveReliables[i][tr].id;
					int objTypeMOTS = (personTracker->GetObjType() == sym::OBJECT_TYPE::CAR) ? 1 : 2;

					// Convert "cv::Mat mask" to "std::string Rle Encoded mask"
					cv::Rect bbox = personTracker->allLiveReliables[i][tr].rec;

					// Frame ID objectType imgWidth imgHeight RleMask
					if (bbox.width > 0 && bbox.height > 0 /*maskSTR.compare("TUT>") != 0*/) {
						std::string maskSTR = CvtMAT2RleSTR(personTracker->allLiveReliables[i][tr].segMask,
							cv::Size(personTracker->frmWidth, personTracker->frmHeight), bbox);

						fprintf_s(fp, "%d %d %d %d %d %s\n", i + sym::FRAME_OFFSETS[DB_TYPE], \
							id, objTypeMOTS, personTracker->frmHeight, personTracker->frmWidth, maskSTR.c_str());
					}
				}
			}
		}
	}
	fclose(fp);

}
