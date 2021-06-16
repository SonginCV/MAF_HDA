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
#include "OnlineTracker.h"

// BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)
typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<float> > Polygon2D;

OnlineTracker::OnlineTracker()
{

}


OnlineTracker::~OnlineTracker()
{
}
void OnlineTracker::SetParams(const struct MOTparams params) {
	this->params = params;
	this->trackObjType = params.OBJ_TYPE;
	this->InitializeTrackletsContainters();
}
string OnlineTracker::GetSeqName() {
	return this->seqName;
}
void OnlineTracker::SetSeqName(const string seqName) {
	this->seqName = seqName;
}
struct MOTparams OnlineTracker::GetParams() {
	return this->params;
}
void  OnlineTracker::SetTotalFrames(int nFrames) {
	this->iTotalFrames = nFrames;
}
int  OnlineTracker::GetTotalFrames() {
	return this->iTotalFrames;
}
void OnlineTracker::SetObjType(int type) {
	this->trackObjType = type;
}
int OnlineTracker::GetObjType() {
	return this->trackObjType;
}
vector<vector<int>> OnlineTracker::HungarianMethod(vector<vector<double>>& costMatrix, const int& nObs, const int& nStats) {

	vector<vector<int>> assigns;
	assigns.resize(nObs, std::vector<int>(nStats, 0));

	vector<int> assignment;
	this->HungAlgo.iFrmCnt = this->sysFrmCnt;
	double cost = this->HungAlgo.Solve(costMatrix, assignment);

	for (unsigned int x = 0; x < costMatrix.size(); x++) {
		if (assignment[x] >= 0)
			assigns[assignment[x]][x] = 1;
	}
	return assigns;
}
vector<vector<cv::Vec2i>> OnlineTracker::cvtSparseMatrix2Dense(const vector<vector<double>>& sparseMatrix, vector<vector<double>>& denseMatrix,
	const double& sparse_value) {
	/*
	Convert sparse cost matrix to be dense. (it may boost Association speed)
	by removing the rows (states) that do not have any tenative observations. (all costs are sparse_value)
	*/

	// cross check
	// sparseMatrix[stat:c][obs:r]
	const int mStats = sparseMatrix.size();
	const int nObs = sparseMatrix[0].size();

	vector<bool> rows_sparsity(mStats, true);
	vector<bool> cols_sparsity(nObs, true);

	int m2_i = 0;
	for (const auto& row : sparseMatrix) {	// states
		int si = &row - &sparseMatrix[0];
		int m2_j = 0;

		for (const auto& cost : row) {		// observations
			int oj = &cost - &row[0];

			if (cost < sparse_value) {
				rows_sparsity[si] = rows_sparsity[si] && false;
				cols_sparsity[oj] = cols_sparsity[oj] && false;
			}
		}
	}

	vector<int> trans_row, trans_col;
	int si = 0;
	for (const auto& row : rows_sparsity) { // state

		if (!row) {
			trans_row.push_back(si);
		}
		++si;
	}

	int oj = 0;
	for (const auto& col : cols_sparsity) { // observation
		if (!col) {
			trans_col.push_back(oj);
		}
		++oj;
	}

	// index: source indices (sparse matrix: m2)
	// value: translated indices (dense matrix: m1)
	vector<vector<cv::Vec2i>> trans_idx_matrix; // m2 to m1
	trans_idx_matrix.resize(trans_row.size(), std::vector<cv::Vec2i>(trans_col.size()));

	denseMatrix.resize(trans_row.size(), std::vector<double>(trans_col.size()));

	si = 0;
	for (auto& row : trans_idx_matrix) {

		int oj = 0;
		for (auto& ti : row) { // translated idx

			ti = cv::Vec2i(trans_row[si], trans_col[oj]);
			denseMatrix[si][oj] = sparseMatrix[ti(0)][ti(1)];
			++oj;
		}
		++si;
	}

	return trans_idx_matrix;
}
void OnlineTracker::InitColorTab()
{
	int a;
	for (a = 1; a*a*a < MAX_OBJECTS; a++);
	int n = 255 / (a - 1);
	IplImage *temp = cvCreateImage(cvSize(40 * (MAX_OBJECTS), 32), IPL_DEPTH_8U, 3);
	cvSet(temp, CV_RGB(0, 0, 0));
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < a; j++) {
			for (int k = 0; k < a; k++) {
				//if(i*a*a+j*a+k>MAX_OBJECTS) break;
				//printf("%d:(%d,%d,%d)\n",i*a*a +j*a+k,i*n,j*n,k*n);
				if (i*a*a + j * a + k == MAX_OBJECTS) break;
				color_tab[i*a*a + j * a + k] = CV_RGB(i*n, j*n, k*n);
				cvLine(temp, cvPoint((i*a*a + j * a + k) * 40 + 20, 0), cvPoint((i*a*a + j * a + k) * 40 + 20, 32), CV_RGB(i*n, j*n, k*n), 32);
			}
		}
	}
	//cvShowImage("(private)Color tap", temp);
	cvWaitKey(1);
	cvReleaseImage(&temp);
}
void OnlineTracker::InitImagesQueue(int width, int height) {
	this->imgBatch = new cv::Mat[this->params.QUEUE_SIZE];
	for (int i = 0; i < this->params.QUEUE_SIZE; i++) {
		this->imgBatch[i].release();
		this->imgBatch[i] = cv::Mat(height, width, CV_8UC3);
	}
	this->frmWidth = width;
	this->frmHeight = height;
}
void OnlineTracker::UpdateImageQueue(const cv::Mat img, int Q_SIZE) {
	// Keep the images and observations into vector array within the recent this->params.TRACK_MIN_SIZE frames 

	for (int q = 0; q < Q_SIZE - 1; q++) {
		imgBatch[q + 1].copyTo(imgBatch[q]);
	}
	img.copyTo(imgBatch[Q_SIZE - 1]);

}
// Initialized the STL Containters for Online Tracker
void OnlineTracker::InitializeTrackletsContainters() {
	this->detsBatch = new std::vector<BBDet>[this->params.TRACK_MIN_SIZE];
	this->liveTracksBatch = new std::vector<BBTrk>[this->params.TRACK_MIN_SIZE];
	this->lostTracksBatch = new std::vector<BBTrk>[this->params.TRACK_MIN_SIZE];
	this->groupsBatch = new std::unordered_map<int, std::vector<RectID>>[this->params.GROUP_QUEUE_SIZE];
}
cv::Point2f OnlineTracker::CalcSegMaskCenter(const cv::Rect& rec, const cv::Mat& mask) {
	cv::Point2f cp_seg(0, 0);
	int nPoints = 0;

	if ((rec.width != mask.cols) || (rec.height != mask.rows)) {
		printf("[ERROR] if((rec.width(%d) != mask.cols(%d)) || (rec.height(%d) != mask.rows(%d)))\n",
			rec.width, mask.cols, rec.height, mask.rows);
	}
	else {
		for (int x = 0; x < mask.cols; ++x) {
			for (int y = 0; y < mask.rows; ++y) {
				if (mask.at<uchar>(y, x) > 0) {
					cp_seg.x += (x + rec.x);
					cp_seg.y += (y + rec.y);
					nPoints++;
				}
			}
		}
	}
	cp_seg.x /= (float)nPoints;
	cp_seg.y /= (float)nPoints;

	return cp_seg;
}
double OnlineTracker::CalcGaussianProbability(const int dims, const cv::Mat x, const cv::Mat mean, cv::Mat& cov) {
	double probability = -1.0;
	//cout << x.rows << " " << mean.rows << " " << cov.rows << "x" << cov.cols << ":" << dims << endl;
	if ((x.rows != mean.rows) || (cov.rows != cov.cols) || (x.rows != dims)) {
		printf("[ERROR](x.rows!=mean.rows) || (cov.rows!=cov.cols) || (x.rows!=dims) (line:248)\n");
	}
	else {

		cv::Mat cov_dbl(dims, dims, CV_64FC1);

		for (int r = 0; r < dims; ++r) {
			for (int c = 0; c < dims; ++c) {
				cov_dbl.at<double>(r, c) = cov.at<float>(r, c);
			}
		}

		cv::Mat sub(dims, 1, CV_64FC1);
		cv::Mat power(1, 1, CV_64FC1);

		double exponent = 0.0;
		double coefficient = 1.0;

		//sub = x - mean;
		for (int r = 0; r < dims; ++r) {
			sub.at<double>(r, 0) = x.at<float>(r, 0) - mean.at<float>(r, 0);
		}

		power = sub.t() * cov_dbl.inv(cv::DECOMP_SVD) * sub;

		coefficient = ((1.0) / (pow(2.0*PI_8, (double)dims / 2.0)*pow(cv::determinant(cov_dbl), 0.5)));
		exponent = (-0.5)*(power.at<double>(0, 0));
		probability = coefficient * pow(e_8, exponent);

		sub.release();
		power.release();
	}
	if (probability < Q_TH_LOW_10) probability = 0.0;

	//if (0) { // GpuMat
	//	cv::cuda::GpuMat subGpu;
	//	cv::cuda::GpuMat powerGpu;
	//}
	return probability;
}
float OnlineTracker::CalcIOU(const cv::Rect& Ra, const cv::Rect& Rb) {

	float IOU = (float)(Ra & Rb).area() / (float)(Ra | Rb).area(); // Intersection-over-Union (IOU)

	return IOU;
}
/**
* @brief		The Function for Calculating 3D IOU between two cuboids
* @details		Refer:	the method: box3DOverlap (KITTI_detection_eval.cpp)
* @param[in]	Ca		Four 3D corners of top plane
* @param[in]	Da		Dimensions represented by height, width, length
* @param[in]	Cb		Four 3D corners of top plane
* @param[in]	Db		Dimensions represented by height, width, length
* @return		float	The intersection-over-union between two 3D boxes (cuoids) which are measured with Volume
* @throws
*/
float OnlineTracker::Calc3DIOU(vector<cv::Vec3f> Ca, cv::Vec3f Da, vector<cv::Vec3f> Cb, cv::Vec3f Db) {

	float pointsA[5][2] = { {Ca[0][0],Ca[0][2]},{ Ca[1][0],Ca[1][2] },{ Ca[2][0],Ca[2][2] },{ Ca[3][0],Ca[3][2] },{ Ca[0][0],Ca[0][2] } };
	Polygon2D polyA;
	boost::geometry::append(polyA, pointsA);

	float pointsB[5][2] = { { Cb[0][0],Cb[0][2] },{ Cb[1][0],Cb[1][2] },{ Cb[2][0],Cb[2][2] },{ Cb[3][0],Cb[3][2] },{ Cb[0][0],Cb[0][2] } };
	Polygon2D polyB;
	boost::geometry::append(polyB, pointsB);

	std::vector<Polygon2D> in;
	boost::geometry::intersection(polyA, polyB, in);

	//std::vector<Polygon2D> un;
	// boost::geometry::union_(gp, dp, un);
	//double union_area = boost::geometry::area(un.front());

	float ymax = std::min(Ca[0][1], Cb[0][1]);					// y positions of the top planes
	float ymin = std::max(Ca[0][1] - Da[0], Cb[0][1] - Db[0]);	// y positions of the bottom planes

	float inter_area = in.empty() ? 0 : boost::geometry::area(in.front());
	float inter_vol = inter_area * std::max((float)0.0, ymax - ymin);

	float volA = Da[0] * Da[1] * Da[2];
	float volB = Db[0] * Db[1] * Db[2];

	float IOU_3D = inter_vol / (volA + volB - inter_vol);

	return IOU_3D;
}
float OnlineTracker::CalcSIOA(cv::Rect Ra, cv::Rect Rb) {
	// check overlapping region
	float Ua = (float)(Ra & Rb).area() / (float)Ra.area();
	float Ub = (float)(Ra & Rb).area() / (float)Rb.area();

	float SIOA = (Ua + Ub) / 2.0; // Symmetric, The Sum-of-Intersection-over-Area (SIOA)

	return SIOA;
}
float OnlineTracker::CalcMIOU(const cv::Rect& Ri, const cv::Mat& Si, const cv::Rect& Rj, const cv::Mat& Sj, cv::Mat& mask_union) {

	float iou = (float)(Ri&Rj).area() / (float)(Ri | Rj).area();
	float siou = 0.0;

	if (iou > 0) {
		int min[2] = { ((Ri.x < Rj.x) ? Ri.x : Rj.x) , ((Ri.y < Rj.y) ? Ri.y : Rj.y) };
		int max[2] = { (((Ri.x + Ri.width) > (Rj.x + Rj.width)) ? (Ri.x + Ri.width) : (Rj.x + Rj.width)) , (((Ri.y + Ri.height) > (Rj.y + Rj.height)) ? (Ri.y + Ri.height) : (Rj.y + Rj.height)) };
		int x = min[0];
		int y = min[1];
		int w = max[0] - min[0];
		int h = max[1] - min[1];

		cv::Mat mu(h, w, CV_8UC1, cv::Scalar(0));
		cv::Rect ri(Ri.x - x, Ri.y - y, Ri.width, Ri.height);
		cv::Mat mi = mu(ri);

		//printf("[%d]i(%dx%dx%d)(%dx%dx%d)\n", this->sysFrmCnt, mi.cols, mi.rows, mi.channels(), Si.cols, Si.rows, Si.channels());
		cv::Mat Si_resize;
		cv::resize(Si, Si_resize, cv::Size(Ri.width, Ri.height));
		mi = mi + 0.5*Si_resize;
		Si_resize.release();

		cv::Rect rj(Rj.x - x, Rj.y - y, Rj.width, Rj.height);
		cv::Mat mj = mu(rj);

		//printf("[%d]j(%dx%dx%d)(%dx%dx%d)\n", this->sysFrmCnt, mj.cols, mj.rows, mj.channels(), Sj.cols, Sj.rows, Sj.channels());
		cv::Mat Sj_resize;
		cv::resize(Sj, Sj_resize, cv::Size(Rj.width, Rj.height));
		mj = mj + 0.5*Sj_resize;
		Sj_resize.release();

		int seg_union_area = 0;
		int seg_intersection_area = 0;
		for (int c = 0; c < w; ++c) {
			for (int r = 0; r < h; ++r) {
				if (mu.at<uchar>(r, c) > 0) {
					++seg_union_area;
					if (mu.at<uchar>(r, c) >= 254) {
						++seg_intersection_area;
					}
				}
			}
		}
		siou = (float)seg_intersection_area / (float)seg_union_area;

		mask_union = mu;
	}
	else {
		siou = 0.0;
	}

	return siou;
}
bool OnlineTracker::IsOutOfFrame(cv::Rect obj_rec, int fWidth, int fHeight, int margin_x, int margin_y) {
	cv::Rect obj = obj_rec;
	cv::Rect frm(margin_x, margin_y, fWidth - 2 * margin_x, fHeight - 2 * margin_y);

	float TH_IN_FRAME_RATIO = 3.0;

	if (margin_x > 0 || margin_y > 0)
		TH_IN_FRAME_RATIO = 1.0;

	if ((obj&frm).area() < obj.area() / TH_IN_FRAME_RATIO)
		return true;
	else
		return false;
}
bool OnlineTracker::IsPointInRect(cv::Point pt, cv::Rect rec) {

	return (pt.x >= rec.x) && (pt.x < rec.x + rec.width) && (pt.y >= rec.y) && (pt.y < rec.y + rec.height);
}
void OnlineTracker::DrawTrkBBS(cv::Mat& img, cv::Rect rec, cv::Scalar color, int thick, int id, double fontScale, string type, bool idBGinBB) {


	if ((int)id >= 0) {
		string strID;
		if (type.empty())	strID = to_string(id);
		else				strID = type.substr(0, 2) + ":" + to_string(id);

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
		else {
			cv::rectangle(img, rec, color, thick);

			int bgRecWidth = fontScale * (int)(idNChars + 4) * 20;
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
// Rect Region Correction for preventing out of frame
cv::Rect OnlineTracker::RectExceptionHandling(int fWidth, int fHeight, cv::Rect rect) {

	if (rect.x < 0) {
		rect.width += rect.x;
		rect.x = 0;
	}
	if (rect.width < 0) rect.width = 0;
	if (rect.x >= fWidth) rect.x = fWidth - 1;
	if (rect.width > fWidth) rect.width = fWidth;
	if (rect.x + rect.width >= fWidth) rect.width = fWidth - rect.x - 1;

	if (rect.y < 0) {
		rect.height += rect.y;
		rect.y = 0;
	}
	if (rect.height < 0) rect.height = 0;
	if (rect.y >= fHeight) rect.y = fHeight - 1;
	if (rect.height > fHeight) rect.height = fHeight;
	if (rect.y + rect.height >= fHeight) rect.height = fHeight - rect.y - 1;

	return rect;
}
cv::Rect OnlineTracker::RectExceptionHandling(const int& fWidth, const int& fHeight, cv::Point p1, cv::Point p2) {

	cv::Rect rect;
	if (p1.x < 0) p1.x = 0;
	if (p2.x < 0) p2.x = 0;
	if (p1.x >= fWidth) p1.x = fWidth - 1;
	if (p2.x >= fWidth) p2.x = fWidth - 1;
	if (p1.y < 0) p1.y = 0;
	if (p2.y < 0) p2.y = 0;
	if (p1.y >= fHeight) p1.y = fHeight - 1;
	if (p2.y >= fHeight) p2.y = fHeight - 1;

	rect.x = p1.x;
	rect.y = p1.y;
	rect.width = p2.x - p1.x;
	rect.height = p2.y - p1.y;

	return rect;
}
void OnlineTracker::cvPrintMat(cv::Mat matrix, string name)
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
	printf("Matrix %s=\n", name.c_str());
	if (matrix.depth() == CV_64F) {
		//int channels = matrix.channels();
		for (int r = 0; r < matrix.rows; r++) {
			for (int c = 0; c < matrix.cols; c++) {
				//printf("(");
				//for( int cn=0 ; cn<channels ; cn++){
				printf("%6.2lf ", matrix.at<double>(r, c)/*[cn]*/);
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

}
void OnlineTracker::cvPrintVec2Vec(const vector<vector<double>>& costs, const string& name)
{
	printf("Matrix %s=\n", name.c_str());
	for (const auto& row : costs) {
		for (const auto& val : row) {
			printf("%6.2lf ", val);
		}
		printf("\n");
	}
}
void OnlineTracker::cvPrintVec2Vec(const vector<vector<int>>& costs, const string& name)
{
	printf("Matrix %s=\n", name.c_str());
	for (const auto& row : costs) {
		for (const auto& val : row) {
			printf("%2d ", val);
		}
		printf("\n");
	}
}
void OnlineTracker::cvPrintVec2Vec(const vector<vector<cv::Vec2i>>& costs, const string& name)
{
	printf("Matrix %s=\n", name.c_str());
	for (const auto& row : costs) {
		for (const auto& val : row) {
			printf("[%2d,%2d] ", val(0), val(1));
		}
		printf("\n");
	}
}
void OnlineTracker::cvPrintVec2Vec(const vector<vector<bool>>& assigns, const string& name)
{
	printf("Matrix %s=\n", name.c_str());
	for (const auto& row : assigns) {
		for (const auto& val : row) {
			printf("%d ", (int)val);
		}
		printf("\n");
	}
}
string  OnlineTracker::cvPrintRect(const cv::Rect& rec) {

	char carr[256];
	sprintf_s(carr, "(%d,%d,%d,%d)", rec.x, rec.y, rec.width, rec.height);
	//printf("(%.2f,%.2f,%.2f,%.2f)");

	return string(carr);
}
cv::Mat OnlineTracker::cvPerspectiveTrans2Rect(const cv::Mat img, const vector<cv::Point2f> corners, const cv::Vec3f dims, float ry) {
	// https://blog.naver.com/PostView.nhn?blogId=pckbj123&logNo=100205803400&proxyReferer=https%3A%2F%2Fwww.google.com%2F
	/*
	0	1
	┌───┐
	│	│
	└───┘
	2	3
	*/

	cv::Mat imgFloat(cv::Size(img.cols, img.rows), CV_32FC3);
	img.convertTo(imgFloat, CV_32FC3);

	cv::Point2f srcPts[4], dstPts[4];

	// Car Pose 
	if (true) {
		// left plane
		// 4->6
		// 0->2
		srcPts[0] = corners[4];
		srcPts[1] = corners[6];
		srcPts[2] = corners[0];
		srcPts[3] = corners[2];
	}

	float minP[2] = { srcPts[0].x, srcPts[0].y };
	float maxP[2] = { srcPts[0].x, srcPts[0].y };

	for (int c = 0; c < 4; ++c)
	{
		//printf("(%.2f,%.2f:)", srcPts[c].x, srcPts[c].y);
		if (minP[0] > srcPts[c].x) minP[0] = srcPts[c].x;
		if (minP[1] > srcPts[c].y) minP[1] = srcPts[c].y;
		if (maxP[0] < srcPts[c].x) maxP[0] = srcPts[c].x;
		if (maxP[1] < srcPts[c].y) maxP[1] = srcPts[c].y;
	}
	float widthP = maxP[0] - minP[0];
	float heightP = maxP[1] - minP[1];

	// 0->1
	// 2->3
	int widthA = std::sqrtf((srcPts[0].x - srcPts[1].x)*(srcPts[0].x - srcPts[1].x) + (srcPts[0].y - srcPts[1].y)*(srcPts[0].y - srcPts[1].y));
	int heightA = std::sqrtf((srcPts[0].x - srcPts[2].x)*(srcPts[0].x - srcPts[2].x) + (srcPts[0].y - srcPts[2].y)*(srcPts[0].y - srcPts[2].y));
	int widthB = std::sqrtf((srcPts[2].x - srcPts[3].x)*(srcPts[2].x - srcPts[3].x) + (srcPts[2].y - srcPts[3].y)*(srcPts[2].y - srcPts[3].y));
	int heightB = std::sqrtf((srcPts[1].x - srcPts[3].x)*(srcPts[1].x - srcPts[3].x) + (srcPts[1].y - srcPts[3].y)*(srcPts[1].y - srcPts[3].y));

	/*int maxWidth = (widthA > widthB) ? widthA : widthB;
	int maxHeight = (heightA > heightB) ? heightA : heightB;*/
	float h3D = dims[0];
	float w3D = dims[1];
	float l3D = dims[2];

	int maxWidth = 300;
	int maxHeight = 300 * dims[0] / dims[2];// *dims[2] / dims[1];

	// 0->1
	// 2->3
	dstPts[0] = cv::Point2f(0, 0);
	dstPts[1] = cv::Point2f(maxWidth - 1, 0);
	dstPts[2] = cv::Point2f(0, maxHeight - 1);
	dstPts[3] = cv::Point2f(maxWidth - 1, maxHeight - 1);


	cv::Mat transMatrix = cv::getPerspectiveTransform(srcPts, dstPts);

	cv::Mat srcImg = imgFloat(cv::Rect(minP[0], minP[1], widthP, heightP));
	cv::Size outputSize(maxWidth, maxHeight);
	cv::Mat dstImg(outputSize, CV_32FC3);
	cv::warpPerspective(imgFloat, dstImg, transMatrix, outputSize);

	cv::Mat srcImgDisplay(cv::Size(srcImg.cols, srcImg.rows), CV_8UC3);
	srcImg.convertTo(srcImgDisplay, CV_8UC3);
	cv::Mat dstImgDisplay(outputSize, CV_8UC3);
	dstImg.convertTo(dstImgDisplay, CV_8UC3);

	if (!imgFloat.empty()) imgFloat.release();
	if (!srcImg.empty()) srcImg.release();
	if (!srcImgDisplay.empty()) srcImgDisplay.release();
	if (!dstImg.empty()) dstImg.release();
	//if (!dstImgDisplay.empty()) dstImgDisplay.release();
	/*========================================================================================================================================*/
	return dstImgDisplay;
}
void OnlineTracker::CollectTracksbyID(unordered_map<int, vector<BBTrk>>& tracksbyID, vector<BBTrk>& targets, bool isInit) {
	pair< unordered_map<int, vector<BBTrk>>::iterator, bool> isEmpty;
	if (isInit) {
		for (int j = 0; j < targets.size(); j++)
		{

			// ArrangeTargetsVecsBatchesLiveLost 에 의해 alive track 들만 모여있는 상태
			int id = targets.at(j).id;

			if (targets.at(j).isNew && targets.at(j).isAlive) { // only add new tracks
				vector<BBTrk> tracklet;
				tracklet.push_back(targets.at(j));

				pair< unordered_map<int, vector<BBTrk>>::iterator, bool> isEmpty = tracksbyID.insert(unordered_map<int, vector<BBTrk>>::value_type(id, tracklet));

				if (isEmpty.second == false) { // already has a element with target.at(j).id
					tracksbyID[id].push_back(targets.at(j));
					//if (DEBUG_PRINT)
					//printf("[%d][%d-%d]ID%d is updated into tracksbyID\n", (int)isInit, this->sysFrmCnt, targets.at(j).fn, id);
				}
				else {
					//if (DEBUG_PRINT)
					//printf("[%d][%d-%d]ID%d is newly added into tracksbyID\n", (int)isInit, this->sysFrmCnt, targets.at(j).fn, id);
				}

				targets.at(j).isNew = false; // Change isNew state to false after added to tracksbyID
			}
		}
	}
	else {
		for (int j = 0; j < targets.size(); j++)
		{

			// ArrangeTargetsVecsBatchesLiveLost 에 의해 alive track 들만 모여있는 상태
			int id = targets.at(j).id;

			if (targets.at(j).isAlive) {
				// targets.at(j).fn = this->sysFrmCnt; // 이게 왜 안넘어 갔는지 미스테리다, 와.. prediction 에서 framenumber를 update 안해줬네..

				vector<BBTrk> tracklet;
				tracklet.push_back(targets.at(j));

				pair< unordered_map<int, vector<BBTrk>>::iterator, bool> isEmpty = tracksbyID.insert(unordered_map<int, vector<BBTrk>>::value_type(id, tracklet));

				if (isEmpty.second == false) { // already has a element with target.at(j).id
					tracksbyID[id].push_back(targets.at(j));
					//if (DEBUG_PRINT)
					//printf("[%d][%d-%d]ID%d is updated into tracksbyID\n", (int)isInit, this->sysFrmCnt, targets.at(j).fn, id);
				}
				else {
					//if (DEBUG_PRINT)
					//printf("[%d][%d-%d]ID%d is newly added into tracksbyID\n", (int)isInit, this->sysFrmCnt, targets.at(j).fn, id);
				}
			}
		}
	}
}
void OnlineTracker::ClearOldEmptyTracklet(int current_fn, unordered_map<int, vector<BBTrk>>& tracklets, int MAXIMUM_OLD) {

	unordered_map<int, vector<BBTrk>> recent_tracks;

	vector<int> keys_old_vec;
	unordered_map<int, vector<BBTrk>>::iterator iter;

	for (iter = tracklets.begin(); iter != tracklets.end(); ++iter) {

		if (!iter->second.empty()) {
			if (iter->second.back().fn >= current_fn - MAXIMUM_OLD) {

				vector<BBTrk> track;

				vector<BBTrk>::iterator iterT;
				for (iterT = iter->second.begin(); iterT != iter->second.end(); ++iterT)
					track.push_back(iterT[0]);

				pair<unordered_map<int, vector<BBTrk>>::iterator, bool> isEmpty = recent_tracks.insert(unordered_map<int, vector<BBTrk>>::value_type(iter->first, track));

				if (isEmpty.second == false) { // already exists

				}
				else {

				}
			}
			else {
				keys_old_vec.push_back(iter->first);
			}
		}
		else {
			keys_old_vec.push_back(iter->first);
		}
	}
	//printf("[%d]-%d<=Recent tracks: ", current_fn, MAXIMUM_OLD);
	//for (iter = recent_tracks.begin(); iter != recent_tracks.end(); ++iter) {
	//	printf("ID%d(%d), ",iter->second.back().id, iter->second.back().fn);
	//}
	//printf("\n");

	// Swap and Clear Old Tracklets
	tracklets.clear();
	recent_tracks.swap(tracklets);
	for (iter = recent_tracks.begin(); iter != recent_tracks.end(); ++iter) {
		iter->second.clear();
	}
	recent_tracks.clear();

}
cv::Vec4f OnlineTracker::LinearMotionEstimation(vector<BBTrk> tracklet, int& idx1_2_fd, int& idx1, int& idx2, \
	int MODEL_TYPE, int reverse_offset, int required_Q_size) {
	int idx_last, idx_first;

	int T_SIZE = tracklet.size();
	if ((T_SIZE - 1 - reverse_offset) >= 0)	idx_last = (T_SIZE - 1 - reverse_offset);
	else									idx_last = (T_SIZE - 1);

	if (!required_Q_size) idx_first = 0;
	else {
		if (idx_last >= (required_Q_size - 1))	idx_first = (idx_last - required_Q_size + 1);
		else									idx_first = 0;
	}
	idx1 = idx_first;
	idx2 = idx_last;

	float fd = tracklet[idx_last].fn - tracklet[idx_first].fn;
	idx1_2_fd = (int)fd;


	if (fd <= 0) {
		return cv::Vec4f(0, 0, 0);
	}
	else {
		cv::Vec4f v(0, 0, 0, 0);

		cv::Rect r1, r2;
		cv::Vec3f cp1, cp2;

		r1 = tracklet[idx_first].rec;
		r2 = tracklet[idx_last].rec;

		cp1 = cv::Vec3f(r1.x + r1.width / 2.0, r1.y + r1.height / 2.0, tracklet[idx_first].ratio_yx);
		cp2 = cv::Vec3f(r2.x + r2.width / 2.0, r2.y + r2.height / 2.0, tracklet[idx_last].ratio_yx);


		v[0] = (cp2[0] - cp1[0]) / fd;
		v[1] = (cp2[1] - cp1[1]) / fd;
		v[2] = (cp2[2] - cp1[2]) / fd;

		return v;
	}
}
cv::Vec4f OnlineTracker::LinearMotionEstimation(unordered_map<int, vector<BBTrk>> tracks, int id, int &idx1_2_fd, int& idx1, int& idx2, \
	int MODEL_TYPE, int reverse_offset, int required_Q_size) {

	int idx_last, idx_first;
	int T_SIZE = tracks[id].size();
	if ((T_SIZE - 1 - reverse_offset) >= 0)	idx_last = (T_SIZE - 1 - reverse_offset);
	else									idx_last = (T_SIZE - 1);

	if (!required_Q_size) idx_first = 0;
	else {
		if (idx_last >= (required_Q_size - 1))	idx_first = (idx_last - required_Q_size + 1);
		else									idx_first = 0;
	}
	idx1 = idx_first;
	idx2 = idx_last;

	float fd = tracks[id][idx_last].fn - tracks[id][idx_first].fn;
	idx1_2_fd = (int)fd;

	if (fd <= 0) {
		return cv::Vec4f(0, 0, 0, 0);
	}
	else { // idx_first < idx_last
		cv::Vec4f v(0, 0, 0, 0);

		cv::Rect r1, r2;
		cv::Vec3f cp1, cp2;

		r1 = tracks[id][idx_first].rec;
		r2 = tracks[id][idx_last].rec;

		cp1 = cv::Vec3f(r1.x + r1.width / 2.0, r1.y + r1.height / 2.0, (float)tracks[id][idx_first].ratio_yx);
		cp2 = cv::Vec3f(r2.x + r2.width / 2.0, r2.y + r2.height / 2.0, (float)tracks[id][idx_last].ratio_yx);

		v[0] = (cp2[0] - cp1[0]) / fd;
		v[1] = (cp2[1] - cp1[1]) / fd;
		v[2] = (cp2[2] - cp1[2]) / fd;

		return v;
	}

}
BBTrk OnlineTracker::KalmanMotionbasedPrediction(BBTrk lostStat, BBTrk liveObs) {

	int fd = liveObs.fn - lostStat.fn;

	cv::Mat stateMat(4, 1, CV_32FC1);
	cv::KalmanFilter stateKF = cv::KalmanFilter(4, 2, 0);

	// statePre, mean (state vector in KalmanFilter), variable, init
	stateKF.statePost.at<float>(0, 0) = (float)lostStat.rec.x + (float)(lostStat.rec.width) / 2.0;
	stateKF.statePost.at<float>(1, 0) = (float)lostStat.rec.y + (float)(lostStat.rec.height) / 2.0;
	stateKF.statePost.at<float>(2, 0) = lostStat.vx;
	stateKF.statePost.at<float>(3, 0) = lostStat.vy;

	// statePost, mean (just for init, used after predict()), variable
	//stateKF.statePost.copyTo(stateKF.statePre);

	// F, constant
	setIdentity(stateKF.transitionMatrix);
	stateKF.transitionMatrix.at<float>(0, 2) = 1;
	stateKF.transitionMatrix.at<float>(1, 3) = 1;

	// H, constant
	stateKF.measurementMatrix = (cv::Mat_<float>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);

	// Q, constant, 4x4
	setIdentity(stateKF.processNoiseCov, cv::Scalar::all(VAR_X / 2));
	stateKF.processNoiseCov.at<float>(1, 1) = VAR_Y / 2;
	stateKF.processNoiseCov.at<float>(3, 3) = VAR_Y / 2;
	//cout << this->KF.processNoiseCov.rows << "x" << this->KF.processNoiseCov.cols << endl;

	setIdentity(stateKF.measurementNoiseCov, cv::Scalar::all(VAR_X));	// R, constant, 2x2
	stateKF.measurementNoiseCov.at<float>(1, 1) = VAR_Y;
	//cout << this->KF.measurementNoiseCov.rows << "x" << this->KF.measurementNoiseCov.cols << endl;

	// P_t|t-1, predicted from P_t-1
	/*stateKF.errorCovPre = (cv::Mat_<float>(4,4) << \
		(float)lostStat.cov.at<double>(0,0), 0, 0, 0, \
		0, (float)lostStat.cov.at<double>(1, 1), 0, 0,
		0, 0, (float)lostStat.cov.at<double>(2, 2), 0,
		0, 0, 0, (float)lostStat.cov.at<double>(3, 3));*/

		// P_0 or P_t,	just for init (variable)
	stateKF.errorCovPost = (cv::Mat_<float>(4, 4) << \
		(float)lostStat.cov.at<double>(0, 0), 0, 0, 0, \
		0, (float)lostStat.cov.at<double>(1, 1), 0, 0,
		0, 0, (float)lostStat.cov.at<double>(2, 2), 0,
		0, 0, 0, (float)lostStat.cov.at<double>(3, 3));

	cv::Mat correctedCov(2, 2, CV_32FC1);


	BBTrk statePredRes;
	lostStat.CopyTo(statePredRes);

	for (int p = 0; p < fd; ++p) {
		cv::Point2f ptPrev(stateMat.at<float>(0, 0), stateMat.at<float>(1, 0));
		stateMat = stateKF.predict();

		statePredRes.rec.x = stateMat.at<float>(0, 0) - statePredRes.rec.width / 2.0;
		statePredRes.rec.y = stateMat.at<float>(1, 0) - statePredRes.rec.height / 2.0;
		stateMat.at<float>(2, 0) = stateMat.at<float>(0, 0) - statePredRes.rec.width / 2.0 - ptPrev.x;
		stateMat.at<float>(3, 0) = stateMat.at<float>(1, 0) - statePredRes.rec.height / 2.0 - ptPrev.y;

		statePredRes.vx = this->params.VEL_UP_ALPHA*statePredRes.vx + (1.0 - this->params.VEL_UP_ALPHA)*stateMat.at<float>(2, 0);
		statePredRes.vy = this->params.VEL_UP_ALPHA*statePredRes.vy + (1.0 - this->params.VEL_UP_ALPHA)*stateMat.at<float>(3, 0);
		stateMat.at<float>(2, 0) = statePredRes.vx;
		stateMat.at<float>(3, 0) = statePredRes.vy;

		correctedCov = stateKF.measurementNoiseCov + stateKF.measurementMatrix*stateKF.errorCovPre*stateKF.measurementMatrix.t();
	}

	// CV_32FC1 to CV_64FC1
	for (int r = 0; r < correctedCov.rows; r++)
		for (int c = 0; c < correctedCov.cols; c++)
			statePredRes.cov.at<double>(r, c) = correctedCov.at<float>(r, c);

	return statePredRes;
}
cv::Point OnlineTracker::cvtRect2Point(const cv::Rect rec) {

	return cv::Point(rec.x + rec.width / 2.0, rec.y + rec.height / 2.0);
}
bool OnlineTracker::CopyCovMatDiag(const cv::Mat src, cv::Mat& dst) {
	if (src.rows != dst.rows || src.cols != dst.cols) {
		cerr << "src.rows!=dst.rows || src.cols != dst.cols" << endl;
		return false;
	}
	else if (src.type() != dst.type()) {
		cerr << "src.type!=dst.type" << endl;
		return false;
	}
	else {
		for (int r = 0; r < dst.rows; ++r) {
			for (int c = 0; c < dst.cols; ++c) {
				if (r == c) {
					dst.at<double>(r, c) = src.at<double>(r, c);
				}
				else {
					dst.at<double>(r, c) = 0.0;
				}
			}
		}
		return true;
	}
}
void OnlineTracker::NormalizeWeight(vector<BBDet>& detVec) {
	// Weight normalization
	std::vector<BBDet>::iterator iterD;
	float sumConfs = 0.0;
	for (iterD = detVec.begin(); iterD != detVec.end(); ++iterD)
	{
		sumConfs += iterD[0].confidence;
	}
	if (sumConfs > 0.0) {
		for (iterD = detVec.begin(); iterD != detVec.end(); iterD++) {
			iterD->weight = iterD->confidence / sumConfs;
		}
	}
	else if (sumConfs <= 0.0) {
		for (iterD = detVec.begin(); iterD != detVec.end(); iterD++) {
			iterD->weight = 0.001;
		}
	}
}
void OnlineTracker::NormalizeWeight(vector<vector<BBDet>>& detVecs) {
	// Weight normalization
	float sumConfs = 0.0;

	for (auto& detVec : detVecs) {
		std::vector<BBDet>::iterator iterD;

		for (iterD = detVec.begin(); iterD != detVec.end(); ++iterD)
		{
			sumConfs += iterD[0].confidence;
		}
	}
	for (auto& detVec : detVecs) {
		std::vector<BBDet>::iterator iterD;
		if (sumConfs > 0.0) {
			for (iterD = detVec.begin(); iterD != detVec.end(); iterD++) {
				iterD->weight = iterD->confidence / sumConfs;
			}
		}
		else if (sumConfs <= 0.0) {
			for (iterD = detVec.begin(); iterD != detVec.end(); iterD++) {
				iterD->weight = 0.001;
			}
		}
	}
}
void OnlineTracker::NormalizeCostVec2Vec(vector<vector<double>>& m_cost, double& min_cost, double& max_cost, const int& MODEL_TYPE) {
	const int nObs = m_cost[0].size();
	const int mStats = m_cost.size();

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
}
cv::Rect OnlineTracker::MergeRect(const cv::Rect A, const cv::Rect B, const float a) {
	cv::Rect rec;
	rec.x = a * A.x + (1.0 - a)*B.x;
	rec.y = a * A.y + (1.0 - a)*B.y;
	rec.width = a * A.width + (1.0 - a)*B.width;
	rec.height = a * A.height + (1.0 - a)*B.height;

	return rec;
}
cv::Rect OnlineTracker::MergeMultiRect(const vector<cv::Rect> recs) {
	cv::Rect res;

	return res;
}
