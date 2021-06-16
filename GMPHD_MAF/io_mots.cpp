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
#include "io_mots.hpp"


void ReadDatasetInfo(const int& DB_TYPE, const string& MODE, const string& detNAME, const string& seqFile, const string& paramsFile,
	vector<string>& seqNames, vector<string>& seqPaths, vector<string>& detTxts, vector<string>& trkTxtsGT, vector<MOTparams>& params_out) {

	// Scene Info Load.
	vector<string> allLines;

	if (_access(seqFile.c_str(), 0) == 0) {
		cout << "Scene Info is loaded from \""<< seqFile <<"\"."<< endl;
		std::ifstream infile(seqFile);

		string seqNAME, dataHomeDIR;
		if (getline(infile, seqNAME)) {
			dataHomeDIR = seqNAME;// The first line indicates dataset's root location.
		}
		cout << "  "<< dataHomeDIR << endl;
		int sq = 1;
		while (getline(infile, seqNAME)) {

			string imgPath = "";
			string detPath = "";
			string trackGTPath = "";
			if (DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20) {
				imgPath = dataHomeDIR + seqNAME + "\\img1\\";
				detPath = dataHomeDIR + seqNAME + "\\det\\det.txt";
				if (!MODE.compare("train"))
					trackGTPath = dataHomeDIR + seqNAME + "\\gt\\gt.txt";
			}
			else if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS) {
				imgPath = dataHomeDIR + "image_02\\" + seqNAME + "\\";
				detPath = dataHomeDIR + "det_02_" + detNAME + "\\" + seqNAME + ".txt";
				if (!MODE.compare("train")) {
					if (DB_TYPE == DB_TYPE_KITTI)
						trackGTPath = dataHomeDIR + "label_02\\" + seqNAME + ".txt";
					else if (DB_TYPE == DB_TYPE_KITTI_MOTS)
						trackGTPath = dataHomeDIR + "instance_02\\" + seqNAME + ".txt";
				}
			}
			else if (DB_TYPE == DB_TYPE_MOTS20) {
				imgPath = dataHomeDIR + seqNAME + "\\";
				detPath = dataHomeDIR + detNAME + "\\" + seqNAME + ".txt";
				if (!MODE.compare("train")) {
					trackGTPath = dataHomeDIR + "instances_txt\\" + seqNAME + ".txt";
				}
			}
			cout <<"    "<< sq++ << ": " << seqNAME << endl;
			seqNames.push_back(seqNAME);
			if (_access(imgPath.c_str(), 0) == 0)	seqPaths.push_back(imgPath);
			else									cout << imgPath << " doesn't exist!! (1)" << endl;
			if (_access(detPath.c_str(), 0) == 0)	detTxts.push_back(detPath);
			else									cout << detPath << " doesn't exist!! (2)" << endl;
			if (!MODE.compare("train")) {
				if (_access(trackGTPath.c_str(), 0) == 0)	trkTxtsGT.push_back(trackGTPath);
				else										cout << trackGTPath << " doesn't exist!! (3)" << endl;
			}
		}
	}
	else {
		printf("%s doesn't exist!! (4)\n", seqFile.c_str());
	}
	// Paramter load.
	if (_access(paramsFile.c_str(), 0) == 0) {

		cout << "Scene parameters are loaded from \"" << paramsFile << "\"." << endl;
		std::ifstream infile(paramsFile);

		params_out.resize(2);// if the file exists,

		string param_line;		
		int p;
		while (getline(infile, param_line)) {

			boost::char_separator<char> bTok(": ");
			boost::tokenizer < boost::char_separator<char>>tokens(param_line, bTok);
			vector<string> vals;
			for (const auto& t : tokens) vals.push_back(t);

			string cls_tmp = vals[1].substr(0, 3);
			std::transform(cls_tmp.begin(), cls_tmp.end(), cls_tmp.begin(), std::tolower);

			int obj_type;
			if (cls_tmp.compare("car"))		obj_type = 1;// sym::OBJECT_TYPE::CAR;
			else if (cls_tmp.compare("ped"))obj_type = 2;// sym::OBJECT_TYPE::PEDESTRIAN;
			else							obj_type = sym::OBJECT_TYPE::MISC;
			//cout << cls_tmp <<endl;
			vector<float> params;

			for (p = 0; p < 13; ++p) {
				if(!getline(infile, param_line)) break;

				boost::tokenizer < boost::char_separator<char>>tokens(param_line, bTok);
				vector<string> vals;
				for (const auto& t : tokens) vals.push_back(t);
				float val = boost::lexical_cast<float>(vals[1]);
				//cout << vals[0] <<": "<< vals[1] << endl;
				params.push_back(val);
			}

			params_out[obj_type-1] =\
				MOTparams(obj_type, /*OBJECT_TYPE*/\
				params[0], /*DET_SCORE_THRESH*/\
				(int)params[1], /*TRACK_MIN*/\
				(int)params[2], /*T2TA_MAX_INTERVAL*/\
				MERGE_METRIC_mIOU, /*MERGE_MEASURE*/\
				params[3],	/*MERGE_THRESH*/\
				params[4],	/*VEL_UPDATE_RATIO*/\
				(int)params[1] * 10, /*TRACK_QUEUE_SIZE*/\
				sym::FRAME_OFFSETS[DB_TYPE],
				(int)params[5], (int)params[6], /*AFFINITY_OPTS*/\
				(bool)params[7], (bool)params[8],/*MASK_ON*/\
				cv::Vec2f(params[9], params[10]), cv::Vec2f(params[9], params[10]),/*KCF_THRESHOLDS*/\
				cv::Vec2f(0.1f, 0.9f), cv::Vec2f(0.1f, 0.9f),/*IOU_THRESHOLDS*/\
				cv::Vec2b((bool)params[11], (bool)params[12]));/*GATE_ON*/
		}
		if (p!=13) {
			printf("Insufficient parameters (%d%14!=0) !!\n",p);
		}
	}
	else {
		printf("%s doesn't exist!! (5)\n", paramsFile.c_str());
	}
}
vector<string> ReadFilesInPath(path p) {

	vector<string> pathVec;

	directory_iterator end_itr;

	// cycle through the directory
	for (directory_iterator itr(p); itr != end_itr; ++itr)
	{
		// If it's not a directory, list it. If you want to list directories too, just remove this check.
		if (is_regular_file(itr->path())) {
			// assign current file name to current_file and echo it out to the console.
			string current_file = itr->path().string();
			pathVec.push_back(current_file);
		}
	}
	return pathVec;
}
VECx2xBBDet ReadDetectionsSeq(const int& DB_TYPE, const string& detNAME, const string& detTxt, VECx2xBBDet& carDets, VECx2xBBDet& personDets) {

	VECx2xBBDet detsSeq_out;

	if (_access(detTxt.c_str(), 0)) {
		cout << "[ERROR] Detection file does not exist!\n" << endl;
	}
	else { // if (_access(detTxt.c_str(),0)==0) {
			//cout << "[WORK] Detection file path have been loaded." << endl;

		vector<string> detLines;

		std::ifstream infile(detTxt);
		string line;

		while (!infile.eof()) {
			getline(infile, line);
			detLines.push_back(line);
			//cout << line << endl;
		}
		detLines = SortAllDetections(detLines, DB_TYPE);

		// Convert Strings into vector<BBDet>
		vector<BBDet> detsFrmAll, detsFrmCar, detsFrmPerson;
		vector<string>::iterator itLines;
		int iFrmCnt = sym::FRAME_OFFSETS[DB_TYPE];
		int detFrmCnt = 0;
		int det_id = 0;
		for (const auto& detSTR : detLines) {

			boost::char_separator<char> bTok;
			if (DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20)
				bTok = boost::char_separator<char>(", ");
			else if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20)
				bTok = boost::char_separator<char>(" ");

			boost::tokenizer < boost::char_separator<char>>tokens(detSTR, bTok);

			vector<string> vals;
			for (const auto& t : tokens)
			{
				vals.push_back(t);
			}
			if (vals.empty()) {
				detsSeq_out.push_back(detsFrmAll);
				detsFrmAll.clear();

				carDets.push_back(detsFrmCar);
				detsFrmCar.clear();

				personDets.push_back(detsFrmPerson);
				detsFrmPerson.clear();
				break;
			}
			int curFrm = (int)boost::lexical_cast<float>(vals.at(0));		// frame number

			BBDet det;
			det.fn = curFrm;

			if (DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {
				/// Detection File Format in the KITTI-MOTS and MOTSChallenge Benchmark
				// token: " "

				// [frame bbox(x1, y1, x2, y2) score class_id img_width img_height rle ReID_association_vector(A 128-D Feature Vector)
				// "%d / %f %f %f %f / %f / %d / %d %d / %s / 128 %f\n"

				// class_id
				// 1: car
				// 2: pedestrian

				float x, y, w, h;
				x = boost::lexical_cast<float>(vals.at(1));						// x
				y = boost::lexical_cast<float>(vals.at(2));						// y

				if (!detNAME.compare("maskrcnn")) {
					w = boost::lexical_cast<float>(vals.at(3));					// width
					h = boost::lexical_cast<float>(vals.at(4));					// height
				}

				char objType = boost::lexical_cast<char>(vals.at(6));
				int objTypeINT = 0.0;
				if (objType == '1')			objTypeINT = sym::OBJECT_TYPE::CAR;
				else if (objType == '2')	objTypeINT = sym::OBJECT_TYPE::PEDESTRIAN;

				// Detection -> BBDet 
				det.object_type = objTypeINT;
				det.rec = cv::Rect(x, y, w, h);
				det.confidence = boost::lexical_cast<float>(vals.at(5));			// detection score
				det.segMaskRle = vals.at(9);										// multi-byte, a run-length data
				det.id = det_id++;

				CvtRleSTR2MAT(det.segMaskRle, cv::Size(boost::lexical_cast<int>(vals.at(7)), boost::lexical_cast<int>(vals.at(8))), det.segMask, det.rec);

				/*printf("[%d] %d(%d,%d,%d,%d,%.3f)(%d,%d)\n", iFrmCnt, detFrmCnt, det.rec.x, det.rec.y, det.rec.width, det.rec.height, det.confidence,
					boost::lexical_cast<int>(vals.at(7)), boost::lexical_cast<int>(vals.at(8)));*/

			}
			else if (DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20) {
				/// Tracking File Format in the MOT Benchmark (2D)
				// token: ", "
				// [frame (1~) id x y width height]
				// "%d,%d,%.2lf,%.2f,%.2f,%.2f,-1,-1,-1,-1\n"

				/// Detection File Format in the MOT Benchmark (2D)
				// token: ", "
				// [frame (1~) -1 x y width height confidence -1 -1 -1]
				// "%d,-1,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1,-1\n"

				// object class: only person
				float x, y, w, h;
				x = boost::lexical_cast<float>(vals.at(2));		// x
				y = boost::lexical_cast<float>(vals.at(3));		// y
				w = boost::lexical_cast<float>(vals.at(4));		// w
				h = boost::lexical_cast<float>(vals.at(5));		// h

				int objTypeINT = sym::OBJECT_TYPE::PEDESTRIAN;

				// Detection -> BBDet 
				det.object_type = objTypeINT;
				det.rec = cv::Rect(x, y, w, h);
				// detection confidence score
				det.confidence = boost::lexical_cast<float>(vals.at(6));
				det.id = det_id++;
			}

			if (iFrmCnt == curFrm) {
				detFrmCnt++;
			}
			else if (iFrmCnt < curFrm) {		// Next frame
				cerr << "\r";
				cerr << "(" << iFrmCnt + sym::FRAME_OFFSETS[DB_TYPE] << ")";
				do {
					detFrmCnt = 0;

					detsSeq_out.push_back(detsFrmAll);
					detsFrmAll.clear();

					// Frame 에 아무 객체가 없어도 들어가야 한다
					// index = frame 이라
					carDets.push_back(detsFrmCar);
					detsFrmCar.clear();

					personDets.push_back(detsFrmPerson);
					detsFrmPerson.clear();


					iFrmCnt++;
				} while (iFrmCnt < curFrm);
			}
			detsFrmAll.push_back(det);

			if (IS_VEHICLE_ALL(det.object_type)) {
				detsFrmCar.push_back(det);
			}
			else if (IS_PERSON_EVAL(det.object_type)) {
				detsFrmPerson.push_back(det);
			}
			else {
				// MISC or DONTCARE

			}
		}
		// End Frame
		cerr << "\r";
		cerr << "(" << iFrmCnt + 1 << ")";

		detsSeq_out.push_back(detsFrmAll);
		carDets.push_back(detsFrmCar);
		personDets.push_back(detsFrmPerson);
	}

	return detsSeq_out;
}
VECx2xBBTrk ReadTracksSeq(const int& DB_TYPE, const string& trkNAME, const string& trkTxt, VECx2xBBTrk& carTrks, VECx2xBBTrk& personTrks, cv::Mat& carHeatMap, cv::Mat& perHeatMap) {
	VECx2xBBTrk trksSeq_out;

	if (_access(trkTxt.c_str(), 0)) {
		cout << "[ERROR] Tracking file does not exist!\n" << endl;
	}
	else { // if (_access(trkTxt.c_str(),0)==0) {
			//cout << "[WORK] Trkection file path have been loaded." << endl;

		vector<string> trkLines;

		std::ifstream infile(trkTxt);
		string line;

		while (!infile.eof()) {
			getline(infile, line);
			trkLines.push_back(line);
			//cout << line << endl;
		}
		//trkLines = SortAllTrkections(trkLines, DB_TYPE);

		// Convert Strings into vector<BBTrk>
		vector<BBTrk> trksFrmAll;
		vector<string>::iterator itLines;
		int iFrmCnt = sym::FRAME_OFFSETS[DB_TYPE];
		int trkFrmCnt = 0;
		for (const auto& trkSTR : trkLines) {

			boost::char_separator<char> bTok;
			if (DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20)
				bTok = boost::char_separator<char>(", ");
			else if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20)
				bTok = boost::char_separator<char>(" ");

			boost::tokenizer < boost::char_separator<char>>tokens(trkSTR, bTok);

			vector<string> vals;
			for (const auto& t : tokens)
			{
				vals.push_back(t);
			}
			if (vals.empty()) {
				trksSeq_out.push_back(trksFrmAll);
				trksFrmAll.clear();
				break;
			}
			int curFrm = (int)boost::lexical_cast<float>(vals.at(0));		// frame number

			BBTrk trk;
			trk.fn = curFrm;

			if (DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {
				/// Trkection File Format in the KITTI-MOTS and MOTSChallenge Benchmark
				// token: " "

				// [frame object_id class_id img_width img_height rle
				// "%d / %f %f %f %f / %f / %d / %d %d / %s \n"

				// class_id
				// 1: car -> object_id: 1000+id
				// 2: pedestrian -> object_id: 2000+id
				// 10: dont care -> object_id: 10000

				// Track Info -> BBTrk 
				trk.conf = 1.0;						// trkection score (GT)
				trk.segMaskRle = vals.at(5);		// multi-byte, a run-length data

				int object_id = boost::lexical_cast<int>(vals.at(1));	// object_id
				char objType = boost::lexical_cast<int>(vals.at(2));
				int objTypeINT = 0.0;
				if (objType == 1) {
					objTypeINT = sym::OBJECT_TYPE::CAR;
					object_id -= 1000;
				}
				else if (objType == 2) {
					objTypeINT = sym::OBJECT_TYPE::PEDESTRIAN;
					object_id -= 2000;
				}
				else if (objType == 10) {
					objTypeINT = sym::OBJECT_TYPE::DONTCARE;
					object_id = 10000;
					trk.conf = 0.0;
				}
				trk.id = object_id;
				trk.objType = objTypeINT;

				int img_witdh = boost::lexical_cast<int>(vals.at(3));
				int img_height = boost::lexical_cast<int>(vals.at(4));

				CvtRleSTR2MAT(trk.segMaskRle, cv::Size(img_witdh, img_height), trk.segMask, trk.rec);

				/*printf("[%d] ID%d(%d)(%d,%d,%d,%d,%.3f)(%d,%d)\n", iFrmCnt, trk.id, trk.objType, trk.rec.x, trk.rec.y, trk.rec.width, trk.rec.height, trk.conf,
					img_witdh, img_height);*/

			}
			else if (DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20) {
				/// Tracking File Format in the MOT Benchmark (2D)
				// token: ", "
				// [frame (1~) id x y width height]
				// "%d,%d,%.2lf,%.2f,%.2f,%.2f,-1,-1,-1,-1\n"

				/// Trkection File Format in the MOT Benchmark (2D)
				// token: ", "
				// [frame (1~) -1 x y width height confidence -1 -1 -1]
				// "%d,-1,%.2f,%.2f,%.2f,%.2f,%.4f,-1,-1,-1,-1\n"

				// object class: only person
				float x, y, w, h;
				x = boost::lexical_cast<float>(vals.at(2));		// x
				y = boost::lexical_cast<float>(vals.at(3));		// y
				w = boost::lexical_cast<float>(vals.at(4));		// w
				h = boost::lexical_cast<float>(vals.at(5));		// h

				int objTypeINT = sym::OBJECT_TYPE::PEDESTRIAN;

				// Trkection -> BBTrk 
				trk.objType = objTypeINT;
				trk.rec = cv::Rect(x, y, w, h);
				// trkection confidence score
				trk.conf = boost::lexical_cast<float>(vals.at(6));
				//trk.id = trk_id++;
			}

			if (iFrmCnt == curFrm) {
				trkFrmCnt++;
			}
			else if (iFrmCnt < curFrm) {		// Next frame
				cerr << "\r";
				cerr << "(" << iFrmCnt + sym::FRAME_OFFSETS[DB_TYPE] << ")";
				do {
					trkFrmCnt = 0;

					trksSeq_out.push_back(trksFrmAll);
					trksFrmAll.clear();

					// Frame 에 아무 객체가 없어도 들어가야 한다
					// index = frame 이라
					//carTrks.push_back(trksFrmCar);
					//trksFrmCar.clear();

					//personTrks.push_back(trksFrmPerson);
					//trksFrmPerson.clear();

					iFrmCnt++;
				} while (iFrmCnt < curFrm);
			}
			trksFrmAll.push_back(trk);

			if (IS_VEHICLE_ALL(trk.objType)) {
				//trksFrmCar.push_back(trk);
				// Compute Car Trkection Heat Map
				if (!carHeatMap.empty()) {
					cv::Mat segMask64;
					trk.segMask.convertTo(segMask64, CV_64FC1, 1.0 / (255.0*255.0 * 10000));

					segMask64 = segMask64 * trk.conf;

					cv::Mat* roi = &(carHeatMap(trk.rec));
					*roi = *roi + segMask64; // cv::add(segMask64, roi, roi);
				}
			}
			else if (IS_PERSON_EVAL(trk.objType)) {
				//trksFrmPerson.push_back(trk);
				// Compute Person Trkection Heat Map
				if (!perHeatMap.empty()) {
					cv::Mat segMask64;
					trk.segMask.convertTo(segMask64, CV_64FC1, 1.0 / (255.0*255.0 * 10000));

					segMask64 = segMask64 * trk.conf;

					cv::Mat* roi = &(perHeatMap(trk.rec));
					*roi = *roi + segMask64; // cv::add(segMask64, roi, roi);
				}
			}
			else {
				// MISC or DONTCARE

			}
		}
		// End Frame
		cerr << "\r";
		cerr << "(" << iFrmCnt + sym::FRAME_OFFSETS[DB_TYPE] << ")";

		trksSeq_out.push_back(trksFrmAll);
		//carTrks.push_back(trksFrmCar);
		//personTrks.push_back(trksFrmPerson);
	}

	return trksSeq_out;
}
vector<string> SortAllDetections(const vector<string>& allLines, int DB_TYPE) {
	// ascending sort by frame number
	/// http://azza.tistory.com/entry/STL-vector-%EC%9D%98-%EC%A0%95%EB%A0%AC

	class T {
	public:
		int frameNum;
		string line;
		T(string s, int DB_TYPE) {
			line = s;

			char tok[8];
			if (DB_TYPE == DB_TYPE_MOT15 || DB_TYPE == DB_TYPE_MOT17 || DB_TYPE == DB_TYPE_MOT20) {
				strcpy_s(tok, ", ");
			}
			if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {
				strcpy_s(tok, " ");
			}

			boost::char_separator<char> bTok(tok);

			boost::tokenizer < boost::char_separator<char>>tokens(s, bTok);
			vector<string> vals;
			for (const auto& t : tokens)
			{
				vals.push_back(t);
			}
			frameNum = boost::lexical_cast<int>(vals.at(0));
		}
		bool operator<(const T &t) const {
			return (frameNum < t.frameNum);
		}
	};


	// Reconstruct the vector<T> from vector<string> for sorting
	vector<T> tempAllLines;
	vector<string>::const_iterator iter = allLines.begin();
	for (; iter != allLines.end(); iter++) {
		if (iter[0].size() < 2) continue;

		tempAllLines.push_back(T(iter[0], DB_TYPE));
	}
	// Sort the vector<T> by frame number
	std::sort(tempAllLines.begin(), tempAllLines.end());

	// Copy the sorted vector<T> to vector<string>
	vector<string> sortedAllLines;
	vector<T>::iterator iterT = tempAllLines.begin();
	for (; iterT != tempAllLines.end(); iterT++) {

		sortedAllLines.push_back(iterT[0].line);
	}

	return sortedAllLines;
}
void SaveResultImgs(const int& DB_TYPE, const string& MODE, const string& detNAME, const string& seqNAME, const int& iFrmCnt, const cv::Mat& img, const float& ths_det, const string& tag) {
	
	std::string strThDetConf;
	float DET_MIN_CONF = ths_det;// sym::DET_SCORE_THS[iDET_TH] / DET_SCORE_TH_SCALE - DET_SCORE_ALPHA;
	if (DET_MIN_CONF <= 0.9 || DET_MIN_CONF == 1.0)
		strThDetConf = boost::str(boost::format("%.1f") % (DET_MIN_CONF));
	else if (DET_MIN_CONF < 0.0)
		strThDetConf = "_all";
	else
		strThDetConf = boost::str(boost::format("%.2f") % (DET_MIN_CONF));


	char folderPath[256], filePath[256];// , filePathINTP[256];

	if (DB_TYPE == DB_TYPE_MOT15) {
		sprintf_s(folderPath, 256, "img\\MOT15\\%s\\%s\\%s\\%s", MODE.c_str(), detNAME.c_str(), seqNAME.c_str(), strThDetConf.c_str());
		//sprintf_s(filePathINTP, 256, "res\\MOT15\\%s\\_speed.txt", MODE);
	}
	else if (DB_TYPE == DB_TYPE_MOT17) {
		sprintf_s(folderPath, 256, "img\\MOT17\\%s\%s\\%s\\%s", MODE.c_str(), detNAME.c_str(), seqNAME.c_str(), strThDetConf.c_str());
		//sprintf_s(filePathINTP, 256, "res\\MOT17\\%s\\_speed.txt", MODE);
	}
	else if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS) {
		sprintf_s(folderPath, 256, "img\\KITTI\\%s\\%s\\%s", MODE.c_str(), detNAME.c_str(), seqNAME.c_str());
		//sprintf_s(filePathINTP, 256, "res\\KITTI\\%s\\%s\\%s_intp\\_speed.txt", MODE, detNAME, strThDetConf);
	}
	else if (DB_TYPE == DB_TYPE_MOTS20) {
		sprintf_s(folderPath, 256, "img\\MOTS20\\%s\\%s\\%s", MODE.c_str(), detNAME.c_str(), seqNAME.c_str());
		//sprintf_s(filePathINTP, 256, "res\\MOTSChallenge\\%s\\%s\\%s_intp\\_speed.txt", MODE, detNAME, strThDetConf);
	}

	//cout << folderPath << endl;
	if (!boost::filesystem::exists(folderPath)) {
		boost::filesystem::create_directory(folderPath);
	}
	sprintf_s(folderPath, 256, "%s\\%s", folderPath, tag.c_str());
	if (!boost::filesystem::exists(folderPath)) {
		boost::filesystem::create_directory(folderPath);
		cout << folderPath << " is created." << endl;
	}

	sprintf_s(filePath, 256, "%s\\%.5d.jpg", folderPath, iFrmCnt);

	cv::imwrite(filePath, img);
}
// convert
int CvtRleSTR2MATVecSeq(VECx2xBBDet& in_dets, VECx2xBBDet& out_dets, const cv::Size& frm_sz, const float& DET_SCORE_TH) {
	VECx2xBBDet detsSeq_trunc;
	int nValidObjs = 0;

	for (auto& detFrm : in_dets) {
		vector<BBDet> detsFrm_trunc;

		int nDets = detFrm.size();
		Concurrency::parallel_for(0, nDets, [&](int d) {
			//	for (auto& det : detFrm) {

			if (detFrm[d].confidence >= DET_SCORE_TH) {
				int iObjType = (int)detFrm[d].object_type;

				cv::Rect recSeg;
				CvtRleSTR2MAT(detFrm[d].segMaskRle, cv::Size(frm_sz.height, frm_sz.width), detFrm[d].segMask, recSeg);
				detFrm[d].rec = recSeg;

				//detsFrm_trunc.push_back(detFrm[d]);

				nValidObjs++;
			}
		}
		);
		for (auto& det : detFrm) {
			if (!det.segMask.empty())
				detsFrm_trunc.push_back(det);
		}
		detsSeq_trunc.push_back(detsFrm_trunc);
	}

	// Release
	for (auto& detFrm : in_dets) {
		for (auto& det : detFrm) {
			det.segMask.release();
		}
		detFrm.clear();
	}
	in_dets.clear();

	// new link
	out_dets = detsSeq_trunc;

	return nValidObjs;
}
// Convert "cv::Mat mask" to "std::string Rle Encoded mask"
std::string CvtMAT2RleSTR(const cv::Mat& in_maskMAT, const cv::Size& in_frmImgSz, const cv::Rect& bbox, const bool& viewDetail) {

	int frmW = in_frmImgSz.width;
	int frmH = in_frmImgSz.height;

	cv::Mat maskMATinFrm(frmH, frmW, CV_8UC1, cv::Scalar(0));
	byte *mask = new byte[frmW*frmH * 1];
	RLE maskRLE;
	// 아니.. 왜 이렇게 짰지.. 이럼 조금이라도 밖으로 나가면 그냥 0이 잖아
	if (viewDetail) printf("(1)");
	if (bbox.width <= 0 || bbox.height <= 0)
	{
		/*printf("[ERROR] Segment is out of frame in CvtMAT2RleSTR() at line 461 %d%d%d%d%d%d\n",
			bbox.x < 0, bbox.y < 0 ,bbox.width >= frmW,bbox.height >= frmH,
			(bbox.x + bbox.width) >= frmW, (bbox.y + bbox.height) >= frmH);*/

		memset(mask, 0, frmW*frmH);
	}
	else {
		if (viewDetail) printf("(2)");
		in_maskMAT.copyTo(maskMATinFrm(bbox));
		for (int i = 0; i < frmW; i++) {
			for (int j = 0; j < frmH; j++) {
				mask[i*frmH + j] = maskMATinFrm.at<uchar>(j, i) ? 255 : 0;
			}
		}
		if (viewDetail) printf("(3)");
	}
	rleEncode(&maskRLE, mask, frmW, frmH, 1);
	if (viewDetail) printf("(4)");

	delete[]mask;
	maskMATinFrm.release();
	if (viewDetail) printf("(5)");
	char *maskCharPtr = rleToString(&maskRLE);
	if (viewDetail) printf("(6)");
	return string(maskCharPtr);
}
// Convert  "std::string Rle Encoded mask" to "cv::Mat mask"
void CvtRleSTR2MAT(const std::string &in_maskRleSTR, const cv::Size& in_segImgSz, cv::Mat& out_maskMAT, cv::Rect& out_objRec) {

	// Decode a run-length data
	string rleStr = in_maskRleSTR;

	//wstring relStr_w;			// unicode
	//relStr_w.assign(rleStr.begin(), rleStr.end());
	//string rleStrUTF8 = boost::locale::conv::utf_to_utf<char>(rleStr);
	//cout << rleStr << endl;
	//cout << rleStrUTF8 << endl;
	//cout << relStr_w << endl; // "를 인식하네, 어쨌든 안됨

	RLE rleTemp;
	siz segImgW = (siz)in_segImgSz.width; siz segImgH = (siz)in_segImgSz.height;

	// frame image size = transposistion of segment image size
	/// rows (height), cols (width) 
	cv::Mat matMaskFrm(in_segImgSz.width, in_segImgSz.height, CV_8UC1, cv::Scalar(0));

	int nSize = rleStr.length() + 1;
	int frmSize = in_segImgSz.area();
	byte *mask = new byte[frmSize * 1];

	char *s = new char[nSize];
	sprintf_s(s, nSize, "%s", rleStr.c_str());
	rleFrString(&rleTemp, s, segImgH, segImgW);
	rleDecode(&rleTemp, mask, 1);
	uint a = 0;

	rleFree(&rleTemp);

	for (int i = 0; i < (int)segImgH; i++)
		for (int j = 0; j < (int)segImgW; j++)
			matMaskFrm.at<uchar>(j, i) = (mask[i*segImgW + j] > 0) ? 255 : 0;

	// Output
	cv::Rect recObj = CvtMAT2RECT(in_segImgSz, matMaskFrm);
	cv::Mat matMaskObj;

	if (recObj.width > 0 && recObj.height > 0)
		matMaskObj = matMaskFrm(recObj).clone();

	out_objRec = recObj;
	out_maskMAT = matMaskObj;

	if (!matMaskFrm.empty()) matMaskFrm.release();
	if (!matMaskObj.empty()) matMaskObj.release();

	delete[]s;
	delete[]mask;
}
cv::Rect CvtMAT2RECT(const cv::Size& in_segImgSz, const cv::Mat& in_maskMAT) {
	siz segImgW = (siz)in_segImgSz.width; siz segImgH = (siz)in_segImgSz.height;

	int min[2] = { (int)segImgH - 1, (int)segImgW - 1 };
	int max[2] = { 0 ,0 };

	for (int i = 0; i < (int)segImgH; i++) {
		for (int j = 0; j < (int)segImgW; j++) {
			if (in_maskMAT.at<uchar>(j, i) > 0) {
				if (min[0] > i) min[0] = i;
				if (min[1] > j) min[1] = j;

				if (max[0] < i) max[0] = i;
				if (max[1] < j) max[1] = j;
			}
		}
	}
	int objW = max[0] - min[0];
	int objH = max[1] - min[1];

	return cv::Rect(min[0], min[1], objW, objH);
}