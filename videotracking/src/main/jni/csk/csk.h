/*******************************************************************************
* Created by Qiang Wang on 16/7.24
* Copyright 2016 Qiang Wang.  [wangqiang2015-at-ia.ac.cn]
* Licensed under the Simplified BSD License
*******************************************************************************/
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <cmath>

using namespace std;
using namespace cv;

void GetSubWindow(const Mat &frame, Mat &subWindow, Point centraCoor, Size sz, Mat &cos_window);

void CalculateHann(Mat &cos_window, Size sz);

void DenseGaussKernel(float sigma, const Mat &x, const Mat &y, Mat &k);

cv::Mat CreateGaussian1(int n, double sigma, int ktype);

cv::Mat CreateGaussian2(Size sz, double sigma, int ktype);

cv::Mat ComplexMul(const Mat &x1, const Mat &x2);

cv::Mat ComplexDiv(const Mat &x1, const Mat &x2);

static inline cv::Point2d centerRect(const cv::Rect& r)
{
  return cv::Point(r.x + cvFloor(double(r.width) / 2.0), r.y + cvFloor(double(r.height) / 2.0));
}

static inline cv::Rect scale_rect(const cv::Rect& r, float scale)
{
    cv::Point m=centerRect(r);
    float width  = r.width  * scale;
    float height = r.height * scale;
    int x=cvRound(m.x - width/2.0);
    int y=cvRound(m.y - height/2.0);
    
    return cv::Rect(x, y, cvRound(width), cvRound(height));
}

static inline cv::Size scale_size(const cv::Size& r, double scale)
{
    double width  = double(r.width)  * scale;
    double height = double(r.height) * scale;
    
    return cv::Size(cvFloor(width), cvFloor(height));
}

static inline cv::Size scale_sizexy(const cv::Size& r, float scalex,float scaley)
{
    float width  = r.width  * scalex;
    float height = r.height * scaley;
    
    return cv::Size(cvRound(width), cvRound(height));
}

#define PI 3.141592653589793

void CircShift(Mat &x, Size k) {
	int cx, cy;
	if (k.width < 0)
		cx = -k.width;
	else
		cx = x.cols - k.width;

	if (k.height < 0)
		cy = -k.height;
	else
		cy = x.rows - k.height;

	Mat q0(x, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(x, Rect(cx, 0, x.cols - cx, cy));  // Top-Right
	Mat q2(x, Rect(0, cy, cx, x.rows - cy));  // Bottom-Left
	Mat q3(x, Rect(cx, cy, x.cols - cx, x.rows - cy)); // Bottom-Right

	Mat tmp1, tmp2;                           // swap quadrants (Top-Left with Bottom-Right)
	hconcat(q3, q2, tmp1);
	hconcat(q1, q0, tmp2);
	vconcat(tmp1, tmp2, x);

}

void GetSubWindow(const Mat &frame, Mat &subWindow, Point centraCoor, Size sz, Mat &cos_window) {

	Point lefttop(cvFloor(centraCoor.x) - cvFloor(float(sz.width) / 2.0) + 1, cvFloor(centraCoor.y) - cvFloor(float(sz.height) / 2.0) + 1);
	Point rightbottom(cvFloor(centraCoor.x) - cvFloor(float(sz.width) / 2.0) + sz.width + 1, cvFloor(centraCoor.y) - cvFloor(float(sz.height) / 2.0) + sz.height + 1);
	Rect idea_rect(lefttop, rightbottom);
	Rect true_rect = idea_rect&Rect(0, 0, frame.cols, frame.rows);
	Rect border(0, 0, 0, 0);
	if (true_rect.area() == 0)
	{
		int x_start, x_width, y_start, y_height;

		x_start = min(frame.cols - 1, max(0, idea_rect.x));
		x_width = max(1, min(idea_rect.x + idea_rect.width, frame.cols) - x_start);
		y_start = min(frame.rows - 1, max(0, idea_rect.y));
		y_height = max(1, min(idea_rect.y + idea_rect.height, frame.rows) - y_start);

		true_rect = Rect(x_start, y_start, x_width, y_height);

		if ((idea_rect.x + idea_rect.width - 1) < 0)
			border.x = sz.width - 1;
		else if (idea_rect.x > (frame.cols - 1))
			border.width = sz.width - 1;
		else
		{
			if (idea_rect.x < 0)
				border.x = -idea_rect.x;
			if ((idea_rect.x + idea_rect.width) > frame.cols)
				border.width = idea_rect.x + idea_rect.width - frame.cols;
		}

		if ((idea_rect.y + idea_rect.height - 1) < 0)
			border.y = sz.height - 1;
		else if (idea_rect.y > (frame.rows - 1))
			border.height = sz.height - 1;
		else
		{
			if (idea_rect.y < 0)
				border.y = -idea_rect.y;
			if ((idea_rect.y + idea_rect.height) > frame.rows)
				border.height = idea_rect.y + idea_rect.height - frame.rows;
		}

		frame(true_rect).copyTo(subWindow);
	}
	else if (true_rect.area() == idea_rect.area())
	{
		frame(true_rect).copyTo(subWindow);
	}
	else
	{
		frame(true_rect).copyTo(subWindow);
		border.y = true_rect.y - idea_rect.y;
		border.height = idea_rect.y + idea_rect.height - true_rect.y - true_rect.height;
		border.x = true_rect.x - idea_rect.x;
		border.width = idea_rect.x + idea_rect.width - true_rect.x - true_rect.width;
	}

	if (border != Rect(0, 0, 0, 0))
	{
		cv::copyMakeBorder(subWindow, subWindow, border.y, border.height, border.x, border.width, cv::BORDER_REPLICATE);
	}

	subWindow.convertTo(subWindow, CV_64FC1, 1.0 / 255.0, -0.5);
	subWindow = subWindow.mul(cos_window);
}

void CalculateHann(Mat &cos_window, Size sz) {
	Mat temp1(Size(sz.width, 1), CV_64FC1);
	Mat temp2(Size(sz.height, 1), CV_64FC1);
	for (int i = 0; i < sz.width; ++i)
		temp1.at<double>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.width - 1)));
	for (int i = 0; i < sz.height; ++i)
		temp2.at<double>(0, i) = 0.5*(1 - cos(2 * PI*i / (sz.height - 1)));
	cos_window = temp2.t()*temp1;
}

void DenseGaussKernel(float sigma, const Mat &x, const Mat &y, Mat &k) {
	Mat xf, yf;
	dft(x, xf, DFT_COMPLEX_OUTPUT);
	dft(y, yf, DFT_COMPLEX_OUTPUT);
	double xx = norm(x);
	xx = xx*xx;
	double yy = norm(y);
	yy = yy*yy;

	Mat xyf;
	mulSpectrums(xf, yf, xyf, 0, true);

	Mat xy;
	cv::idft(xyf, xy, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
	CircShift(xy, scale_size(x.size(), 0.5));
	double numelx1 = x.cols*x.rows;
	//exp((-1 / (sigma*sigma)) * abs((xx + yy - 2 * xy) / numelx1), k); //thsi setting is fixed by version 2(KCF)
	exp((-1 / (sigma*sigma)) * max(0, (xx + yy - 2 * xy) / numelx1), k);
}

cv::Mat CreateGaussian1(int n, double sigma, int ktype)
{
	CV_Assert(ktype == CV_32F || ktype == CV_64F);
	Mat kernel(n, 1, ktype);
	float* cf = kernel.ptr<float>();
	double* cd = kernel.ptr<double>();

	double sigmaX = sigma > 0 ? sigma : ((n - 1)*0.5 - 1)*0.3 + 0.8;
	double scale2X = -0.5 / (sigmaX*sigmaX);

	int i;
	for (i = 0; i < n; i++)
	{
		double x = i - floor(n / 2) + 1;
		double t = std::exp(scale2X*x*x);
		if (ktype == CV_32F)
		{
			cf[i] = (float)t;
		}
		else
		{
			cd[i] = t;
		}
	}

	return kernel;
}

cv::Mat CreateGaussian2(Size sz, double sigma, int ktype)
{
	Mat a = CreateGaussian1(sz.height, sigma, ktype);
	Mat b = CreateGaussian1(sz.width, sigma, ktype);
	return a*b.t();
}

cv::Mat ComplexMul(const Mat &x1, const Mat &x2)
{
	vector<Mat> planes1;
	split(x1, planes1);
	vector<Mat> planes2;
	split(x2, planes2);
	vector<Mat>complex(2);
	complex[0] = planes1[0].mul(planes2[0]) - planes1[1].mul(planes2[1]);
	complex[1] = planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0]);
	Mat result;
	merge(complex, result);
	return result;
}

cv::Mat ComplexDiv(const Mat &x1, const Mat &x2)
{
	vector<Mat> planes1;
	split(x1, planes1);
	vector<Mat> planes2;
	split(x2, planes2);
	vector<Mat>complex(2);
	Mat cc = planes2[0].mul(planes2[0]);
	Mat dd = planes2[1].mul(planes2[1]);

	complex[0] = (planes1[0].mul(planes2[0]) + planes1[1].mul(planes2[1])) / (cc + dd);
	complex[1] = (-planes1[0].mul(planes2[1]) + planes1[1].mul(planes2[0])) / (cc + dd);
	Mat result;
	merge(complex, result);
	return result;
}