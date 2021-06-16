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
#include "drawing.hpp"


void InitColorMapTab(cv::Mat& color_map, cv::Scalar* color_tab)
{
	// Init Color Table for object IDs and Map for intensity
	int a;
	for (a = 1; a*a*a < MAX_OBJECTS; a++);
	int n = 255 / (a - 1);;
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < a; j++) {
			for (int k = 0; k < a; k++) {
				if (i*a*a + j * a + k == MAX_OBJECTS) break;
				color_tab[i*a*a + j * a + k] = CV_RGB(i*n, j*n, k*n);
			}
		}
	}

	cv::Mat img_gray(32, 256, CV_8UC1);
	for (int r = 0; r < img_gray.rows; r++)
		for (int c = 0; c < img_gray.cols; c++)
			img_gray.at<uchar>(r, c) = (uchar)(255 - c);

	if (img_gray.empty()) {
		CV_Error(CV_StsBadArg, "Sample image is empty. Please adjust your path, so it points to a valid input image!");
	}
	color_map = cv::Mat(32, 256, CV_8UC3);

	// Apply the colormap:
	cv::applyColorMap(img_gray(cv::Rect(0, 0, 256, 32)), color_map(cv::Rect(0, 0, 256, 32)), cv::COLORMAP_JET);
}
void DrawDetections(cv::Mat& img_det, const std::vector<BBDet>& dets, const cv::Scalar* color_tab, const int& DB_TYPE, const bool& VIS_BB) {

	if ((DB_TYPE_MOT15 <= DB_TYPE && DB_TYPE <= DB_TYPE_MOT20)) {
		int d = 0;
		for (auto& det : dets) {
			cv::Scalar objClr = sym::OBJECT_TYPE_COLORS[det.object_type];
			DrawDetBB(img_det, d, det.rec, det.confidence, 0.02, 5, objClr, 2);
			d++;
		}
	}
	else if (DB_TYPE == DB_TYPE_KITTI_MOTS || DB_TYPE == DB_TYPE_MOTS20) {
		float alpha = 0.4;
		float beta = 1.0 - alpha;

		//cv::Mat bg(img_det.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		//cv::addWeighted(img_det, 0.5, bg, 0.5, 0.0, img_det);

		int d = 0;
		//printf("[%d] %d detections\n",iFrmCnt, input_dets[iFrmCnt].size());
		for (auto& det : dets) {

			int iObjType = (int)det.object_type;

			cv::Scalar instClr = color_tab[d % (MAX_OBJECTS - 1)];
			cv::Scalar semaClr = sym::OBJECT_TYPE_COLORS[det.object_type];

			for (int r = det.rec.y; r < det.rec.y + det.rec.height; r++) {
				for (int c = det.rec.x; c < det.rec.x + det.rec.width; c++) {
					int i = r - det.rec.y;
					int j = c - det.rec.x;
					if (det.segMask.at<uchar>(i, j) > 0) {
						// Sementic Segmentation
						//imgSemantic.at<Vec3b>(r, c)[0] = alpha*imgSemantic.at<Vec3b>(r, c)[0] + beta*semaClr[0];
						//imgSemantic.at<Vec3b>(r, c)[1] = alpha*imgSemantic.at<Vec3b>(r, c)[1] + beta*semaClr[1];
						//imgSemantic.at<Vec3b>(r, c)[2] = alpha*imgSemantic.at<Vec3b>(r, c)[2] + beta*semaClr[2];

						// Instance Segmentation
						img_det.at<cv::Vec3b>(r, c)[0] = alpha * img_det.at<cv::Vec3b>(r, c)[0] + beta * instClr[0];
						img_det.at<cv::Vec3b>(r, c)[1] = alpha * img_det.at<cv::Vec3b>(r, c)[1] + beta * instClr[1];
						img_det.at<cv::Vec3b>(r, c)[2] = alpha * img_det.at<cv::Vec3b>(r, c)[2] + beta * instClr[2];
					}
				}
			}
			if (VIS_BB) {
				DrawDetBB(img_det, d, det.rec, det.confidence, 0.02, 5, semaClr, 2);
			}
			d++;
		}
		//bg.release();
	}
}
void DrawTracker(cv::Mat& img_trk, const std::vector<BBTrk>& trks, const std::string& trackerName, const int& DB_TYPE, const cv::Scalar* color_tab, int thick, double fontScale) {

	for (const auto& track : trks) {

		int id = track.id;

		cv::Rect rec = track.rec;
		int objType = 0;
		if (DB_TYPE == DB_TYPE_KITTI || DB_TYPE == DB_TYPE_KITTI_MOTS)	objType = track.objType;
		else															objType = sym::OBJECT_TYPE::PEDESTRIAN;

		if (IS_VEHICLE_EVAL(objType) || IS_PERSON_EVAL(objType)) {
			int idTmp = ((id < 0) ? 0 : id);
			bool idBGinBB = (trackerName.compare("3D")) ? false : true;
			DrawTrackBB(img_trk, rec, color_tab[idTmp % (MAX_OBJECTS - 1)], thick, id, fontScale, trackerName, idBGinBB);
		}
	}
}
void DrawTrackerInstance(cv::Mat& img_trk, const track_info& track, const std::string& trackerName, const int& DB_TYPE, const cv::Scalar* color_tab, const bool& vis_bb_on,
	int thick, double fontScale) {

	int id = (int)track.id;

	cv::Rect rec = track.bb;
	int objType = track.object_type;

	int idTmp = ((id < 0) ? 0 : id);
	cv::Scalar color = color_tab[idTmp % (MAX_OBJECTS - 1)];
	bool idBGinBB = (trackerName.compare("T2")) ? false : true;

	const cv::Mat& mask = track.mask;

	cv::Scalar instClr = color_tab[id % (MAX_OBJECTS - 2) + 1]; // exclude black

	std::string strID;
	if (!trackerName.compare("") || trackerName.empty()) {
		strID = sym::OBJECT_STRINGS[objType].substr(0, 1) + std::to_string(id);

		if (IS_DONTCARE(objType)) {
			color = cv::Scalar(25, 25, 25);
			instClr = color;
		}
	}
	else {
		strID = trackerName.substr(0, 2) + ":" + std::to_string(id);
	}

	float alpha = 0.4;
	float beta = 1.0 - alpha;

	for (int r = rec.y; r < rec.y + rec.height; r++) {
		for (int c = rec.x; c < rec.x + rec.width; c++) {
			int i = r - rec.y;
			int j = c - rec.x;
			if (mask.at<uchar>(i, j) > 0) {
				// Instance Segmentation
				img_trk.at<cv::Vec3b>(r, c)[0] = alpha * img_trk.at<cv::Vec3b>(r, c)[0] + beta * instClr[0];
				img_trk.at<cv::Vec3b>(r, c)[1] = alpha * img_trk.at<cv::Vec3b>(r, c)[1] + beta * instClr[1];
				img_trk.at<cv::Vec3b>(r, c)[2] = alpha * img_trk.at<cv::Vec3b>(r, c)[2] + beta * instClr[2];
			}
		}
	}
	if (vis_bb_on) {
		cv::rectangle(img_trk, track.bb, instClr, thick);
	}
	int idNChars = 0;
	if (id == 0)	idNChars = 0;
	else			idNChars = log10f(id);

	int idTextWidth = fontScale * (int)(idNChars + 4) * 20;
	int idTextHeight = fontScale * 40;

	if (!IS_DONTCARE(objType)) {
		cv::Point pt(rec.x + rec.width / 2.0 - idTextWidth / 2.0, rec.y + rec.height / 2.0 + idTextHeight / 2.0);
		cv::putText(img_trk, strID, pt, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255)/*color*/, thick);
	}
}
void DrawTrackerInstances(cv::Mat& img_trk, const std::vector<BBTrk>& trks, const std::string& trackerName, const int& DB_TYPE, const cv::Scalar* color_tab, int thick, double fontScale) {

	const int fWidth = img_trk.cols;
	const int fHeight = img_trk.rows;

	for (auto track : trks) {

		int id = (int)track.id;

		int objType = track.objType;

		int idTmp = ((id < 0) ? 0 : id);
		cv::Scalar color = color_tab[idTmp % (MAX_OBJECTS - 1)];
		bool idBGinBB = (trackerName.compare("T2")) ? false : true;

		const cv::Mat& mask = track.segMask;

		cv::Scalar instClr = color_tab[(id / 2) % (MAX_OBJECTS - 1)];
		/*if(id>=870)
			instClr = color_tab[(id) % (MAX_OBJECTS - 1)];*/
			//cv::Scalar instClr = color_tab[id % (MAX_OBJECTS - 2) + 1]; // exclude black


		std::string strID;
		if (!trackerName.compare("") || trackerName.empty()) {
			strID = sym::OBJECT_STRINGS[objType].substr(0, 1) + std::to_string(id);

			if (IS_DONTCARE(objType)) {
				color = cv::Scalar(25, 25, 25);
				instClr = color;
			}
		}
		else {
			strID = trackerName.substr(0, 1) + std::to_string(id); // C1, P1
			//strID = trackerName.substr(0, 3) + std::to_string(id); // CAR1, PED1
			//strID = trackerName.substr(0, 2) +":"+ std::to_string(id); // CA1, PE1
		}

		float alpha = 0.4;
		float beta = 1.0 - alpha;

		cv::Rect bbox = track.rec;
		bbox.width = track.segMask.cols;
		bbox.height = track.segMask.rows;

		cv::Point p1(bbox.x, bbox.y);
		cv::Point p2(bbox.x + bbox.width, bbox.y + bbox.height);

		if (p1.x < 0) p1.x = 0;
		if (p2.x < 0) p2.x = 0;
		if (p1.x >= fWidth) p1.x = fWidth - 1;
		if (p2.x >= fWidth) p2.x = fWidth - 1;
		if (p1.y < 0) p1.y = 0;
		if (p2.y < 0) p2.y = 0;
		if (p1.y >= fHeight) p1.y = fHeight - 1;
		if (p2.y >= fHeight) p2.y = fHeight - 1;

		cv::Rect bboxInFrm(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
		track.rec = bboxInFrm;

		cv::Rect bbox_ = cv::Rect(bboxInFrm.x - bbox.x, bboxInFrm.y - bbox.y, bboxInFrm.width, bboxInFrm.height);

		cv::Mat segTmp = track.segMask.clone();
		track.segMask.release();
		track.segMask = segTmp(bbox_).clone();
		segTmp.release();

		for (int r = track.rec.y; r < track.rec.y + track.rec.height; r++) {
			for (int c = track.rec.x; c < track.rec.x + track.rec.width; c++) {
				int i = r - track.rec.y;
				int j = c - track.rec.x;
				if (mask.at<uchar>(i, j) > 0) {
					// Instance Segmentation
					img_trk.at<cv::Vec3b>(r, c)[0] = alpha * img_trk.at<cv::Vec3b>(r, c)[0] + beta * instClr[0];
					img_trk.at<cv::Vec3b>(r, c)[1] = alpha * img_trk.at<cv::Vec3b>(r, c)[1] + beta * instClr[1];
					img_trk.at<cv::Vec3b>(r, c)[2] = alpha * img_trk.at<cv::Vec3b>(r, c)[2] + beta * instClr[2];
				}
			}
		}

		int idNChars = 0;
		if (id == 0)	idNChars = 0;
		else			idNChars = log10f(id);

		int idTextWidth = fontScale * (int)(idNChars + 4) * 20;
		int idTextHeight = fontScale * 40;

		if (!IS_DONTCARE(objType)) {
			cv::Point pt(track.rec.x + track.rec.width / 2.0 - idTextWidth / 2.0, track.rec.y + track.rec.height / 2.0 + idTextHeight / 2.0);
			cv::putText(img_trk, strID, pt, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255)/*color*/, thick);
		}

	}
}
void DrawTrackBB(cv::Mat& img, const cv::Rect& rec, const cv::Scalar& color, const int& thick, const int& id, const double& fontScale, std::string type, const bool& idBGinBB) {

	if ((int)id >= 0) {
		std::string strID;
		if (type.empty())	strID = std::to_string(id);
		else				strID = type.substr(0, 1) + std::to_string(id); // type.substr(0, 2) + ":" + std::to_string(id);

		int idNChars = 0;
		if (id == 0)	idNChars = 0;
		else			idNChars = log10f(id);

		if (!type.compare("GT")) {
			cv::rectangle(img, rec, cv::Scalar(255, 255, 255), thick + 2);

			int idTextWidth = fontScale * (int)(idNChars + 4) * 20;
			int idTextHeight = fontScale * 40;

			cv::Mat gt_mat = cv::Mat(rec.height, rec.width, CV_8UC3);
			gt_mat.setTo(color);

			addWeighted(img(rec), 0.5, gt_mat, 0.5, 0.0, img(rec));
			gt_mat.release();

			cv::Point pt(rec.x + rec.width / 2.0 - idTextWidth / 2.0, rec.y + rec.height / 2.0 + idTextHeight / 2.0);
			cv::putText(img, strID, pt, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(255, 255, 255)/*color*/, thick);

		}
		else if (!type.compare("DC")) { // don't care
			cv::Mat gt_mat = cv::Mat(rec.height, rec.width, CV_8UC3);
			gt_mat.setTo(color);

			addWeighted(img(rec), 0.5, gt_mat, 0.5, 0.0, img(rec));
			gt_mat.release();
		}
		else {
			cv::rectangle(img, rec, color, thick);

			int bgRecWidth = fontScale * (int)(idNChars + 3) * 20;
			int bgRecHeight = fontScale * 40;
			cv::Point pt;
			cv::Rect bg;
			pt.x = rec.x;
			int margin = 5;
			if (!idBGinBB) {
				if ((rec.y + rec.height / 2.0) < (img.rows / 2)) { // y < height/2 (higher)
					pt.y = rec.y + rec.height + bgRecHeight - margin;// should be located on the bottom of the bouding box
					bg = cv::Rect(pt.x - margin, pt.y - bgRecHeight + margin, bgRecWidth, bgRecHeight);
				}
				else { // y >= height/2 (lower)
					pt.y = rec.y - margin;	// should be located on the top of the bouding box
					bg = cv::Rect(pt.x - margin, pt.y - bgRecHeight + margin, bgRecWidth, bgRecHeight);
				}
			}
			else {
				if ((rec.y + rec.height / 2.0) < (img.rows / 2)) { // y < height/2 (higher)
					pt.y = rec.y + rec.height - margin;// should be located on the bottom of the bouding box
					bg = cv::Rect(pt.x - margin, pt.y - bgRecHeight + margin, bgRecWidth, bgRecHeight);
				}
				else { // y >= height/2 (lower)
					pt.y = rec.y + bgRecHeight - margin;	// should be located on the top of the bouding box
					bg = cv::Rect(pt.x - margin, pt.y - bgRecHeight + margin, bgRecWidth, bgRecHeight);
				}
			}
			cv::rectangle(img, bg, cv::Scalar(50, 50, 50), -1);
			cv::putText(img, strID, pt, cv::FONT_HERSHEY_SIMPLEX, fontScale, cv::Scalar(0, 255, 255)/*color*/, thick);

		}
	}
	else { // object class: dontcare
		if (!type.compare("GT")) {
			cv::rectangle(img, rec, cv::Scalar(0, 0, 0), -1);
		}
		else {
			cv::rectangle(img, rec, cv::Scalar(0, 0, 0), -1);
		}
	}
}
// Draw the Detection Bounding Box
void DrawDetBB(cv::Mat& img, int iter, cv::Rect bb, double conf, double conf_th, int digits, cv::Scalar color, int thick) {

	int xc = bb.x + bb.width / 2;
	int yc = bb.y + bb.height / 2;

	std::ostringstream ost;
	ost << conf;
	std::string str = ost.str();
	char cArrConfidence[8];
	int c;
	for (c = 0; c < digits && c < str.size(); c++) cArrConfidence[c] = str.c_str()[c];
	cArrConfidence[c] = '\0';

	if (conf >= conf_th) {
		/// Draw Detection Bounding Boxes with detection cofidence score
		cv::rectangle(img, bb, color, thick);
		cv::rectangle(img, cv::Point(bb.x, bb.y), cv::Point(bb.x + bb.width, bb.y - 30), color, -1);
		cv::putText(img, cArrConfidence, cvPoint(bb.x, bb.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, CV_RGB(0, 0, 0), 2);

		/// Draw Observation ID (not target ID)
		char cArrObsID[8];
		sprintf_s(cArrObsID, 8, "%d", iter);
		cv::putText(img, cArrObsID, cvPoint(bb.x + 5, bb.y + 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, color, 2);
	}
	else {
		cv::rectangle(img, bb, color, 1);
		cv::rectangle(img, cv::Point(bb.x, bb.y + bb.height), cv::Point(bb.x + bb.width, bb.y + bb.height + 20), color, -1);
		cv::putText(img, cArrConfidence, cvPoint(bb.x, bb.y + bb.height + 15), cv::FONT_HERSHEY_SIMPLEX, 0.4, CV_RGB(0, 0, 0), 1);
	}
}
// Draw Frame Number on Image
void DrawFrameNumberAndFPS(int iFrameCnt, cv::Mat& img, double scale, int thick, int frameOffset, int frames_skip_interval, double sec) {
	// Draw Frame Number
	char frameCntBuf[8];
	sprintf_s(frameCntBuf, 8, "%d", (iFrameCnt + frameOffset) / frames_skip_interval);
	cv::putText(img, frameCntBuf, cv::Point(10, 65), CV_FONT_HERSHEY_SIMPLEX, scale, cvScalar(255, 255, 255), thick);

	// Draw Frames Per Second
	if (sec > 0.0) {
		std::string text = cv::format("%0.1f fps", 1.0 / sec * frames_skip_interval);
		cv::Scalar textColor(0, 0, 250);
		cv::putText(img, text, cv::Point(10, 100), cv::FONT_HERSHEY_PLAIN, 2, textColor, 2);
	}
}

void cvPrintMat(cv::Mat matrix, std::string name)
{
	/*
	<Mat::type()>
	depth에 channels까지 포함하는 개념 ex. CV_64FC1
	<Mat::depth()>
	CV_8U - 8-bit unsigned integers ( 0..255 )
	CV_8S - 8-bit signed integers ( -128..127 )
	CV_16U - 16-bit unsigned integers ( 0..65535 )
	CV_16S - 16-bit signed integers ( -32768..32767 )
	CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
	CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
	CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
	*/
	printf("Matrix %s=\n", name);
	if (matrix.depth() == CV_64F) {
		//int channels = matrix.channels();
		for (int r = 0; r < matrix.rows; r++) {
			for (int c = 0; c < matrix.cols; c++) {
				//printf("(");
				//for( int cn=0 ; cn<channels ; cn++){
				if (matrix.at<double>(r, c) > 0.0)
					printf("%6.20lf ", matrix.at<double>(r, c)/*[cn]*/);
				//} printf(")");
			}
			printf("\n");
		}
	}
	else if (matrix.depth() == CV_32F) {
		//int channels = matrix.channels();
		for (int r = 0; r < matrix.rows; r++) {
			for (int c = 0; c < matrix.cols; c++) {
				//printf("(");
				//for( int cn=0 ; cn<channels ; cn++){
				printf("%6.2f ", matrix.at<float>(r, c)/*[cn]*/);
				//} printf(")");
			}
			printf("\n");
		}
	}
	else if (matrix.depth() == CV_8U) {
		//int channels = matrix.channels();
		for (int r = 0; r < matrix.rows; r++) {
			for (int c = 0; c < matrix.cols; c++) {
				//printf("(");
				//for( int cn=0 ; cn<channels ; cn++){
				printf("%6d ", (int)matrix.at<uchar>(r, c)/*[cn]*/);
				//} printf(")");
			}
			printf("\n");
		}
	}

}