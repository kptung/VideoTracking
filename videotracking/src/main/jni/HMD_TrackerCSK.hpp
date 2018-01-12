#ifndef __HMD_TrackerCSK_hpp__
#define __HMD_TrackerCSK_hpp__

#include <list>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>    // std::min
#include <opencv2/opencv.hpp>


#include <stdlib.h>     /* NULL */
#include <assert.h>     /* assert */

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <cstring>
#include <stdio.h>
#include <ctype.h>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <functional>

#include "HMD_AbstractTracker.hpp"
#include "csk/csk.h"

using namespace cv;
using namespace std;



class TrackerCSK : public AbstractTracker
{
public:

	TrackerCSK() : AbstractTracker()
	{
		padding = 2;
		output_sigma_factor = 1. / 16;
		sigma = 0.2;
		lambda = 1e-2;
		interp_factor = 0.075;

	}

	/// add a tracking obj
	virtual int addTrackingObject(const cv::Mat& source, const cv::Rect& obj_roi)
	{

		int obj_id = AbstractTracker::addTrackingObject(source, obj_roi);

		ratio = ratioObtain(source);

		cv::Mat im_gray;
		cv::cvtColor(source, im_gray, CV_RGB2GRAY);
		if (obj_id >= 0)
		{
			cv::Rect roi = obj_roi;
			roi.x /= ratio;
			roi.y /= ratio;
			roi.width /= ratio;
			roi.height /= ratio;
			cv::resize(im_gray, im_gray, Size(cvRound(source.cols / ratio), cvRound(source.rows / ratio)));

			pos = centerRect(roi);
			target_sz=cv::Size(roi.width, roi.height);
			sz = scale_size(target_sz, (1.0 + padding));
			output_sigma = sqrt(double(target_sz.area())) * output_sigma_factor;
			y = CreateGaussian2(sz, output_sigma, CV_64F);
			dft(y, yf, DFT_COMPLEX_OUTPUT);
			cos_window=cv::Mat(sz, CV_64FC1);
			CalculateHann(cos_window, sz);
		
			//get subwindow at current estimated target position, to train classifer
			GetSubWindow(im_gray, x, pos, sz, cos_window);
			DenseGaussKernel(sigma, x, x, k);
			dft(k, kf, DFT_COMPLEX_OUTPUT);
			new_alphaf = ComplexDiv(yf, kf + Scalar(lambda, 0));
			new_z = x;
			alphaf = new_alphaf;
			z = x;
		}
		return obj_id;
	}

	double ratioObtain(const Mat& im)
	{
		int rows = im.rows;
		int cols = im.cols;
		double sx = max(rows, cols) / max(frame_size.x, frame_size.y);
		double sy = min(rows, cols) / min(frame_size.x, frame_size.y);
		return min(sx, sy);
	}
	
	/// Run the algorithm
	bool runObjectTrackingAlgorithm(const cv::Mat& target, std::map<int, cv::Rect>& objects)
	{
		objects.clear();

		frame_id++;

		cv::Mat im_gray;
		cv::cvtColor(target, im_gray, CV_RGB2GRAY);
		
		cv::resize(im_gray, im_gray, cv::Size(cvRound(target.cols / ratio), cvRound(target.rows / ratio)));

		auto itr = m_active_objects.begin();
		for (; itr != m_active_objects.end(); ++itr) {
			/************************************************************************/
			/* CSK                                                                  */
			/************************************************************************/
			DenseGaussKernel(sigma, x, z, k);
			cv::dft(k, kf, DFT_COMPLEX_OUTPUT);
			cv::idft(ComplexMul(alphaf, kf), response, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Applying IDFT
			cv::Point maxLoc;
			minMaxLoc(response, NULL, NULL, NULL, &maxLoc);
			pos.x = pos.x - cvFloor(float(sz.width) / 2.0) + maxLoc.x + 1;
			pos.y = pos.y - cvFloor(float(sz.height) / 2.0) + maxLoc.y + 1;
			//get subwindow at current estimated target position, to train classifer
			GetSubWindow(im_gray, x, pos, sz, cos_window);
			DenseGaussKernel(sigma, x, x, k);
			cv::dft(k, kf, DFT_COMPLEX_OUTPUT);
			new_alphaf = ComplexDiv(yf, kf + Scalar(lambda, 0));
			new_z = x;
			alphaf = (1.0 - interp_factor) * alphaf + interp_factor * new_alphaf;
			z = (1.0 - interp_factor) * z + interp_factor * new_z;
			cv::Rect match(pos.x - target_sz.width / 2, pos.y - target_sz.height / 2, target_sz.width, target_sz.height);
			/************************************************************************/
			/* CSK                                                                  */
			/************************************************************************/
			match.x *= ratio;
			match.y *= ratio;
			match.width *= ratio;
			match.height *= ratio;
			objects.insert(std::make_pair(itr->first, match));
			
		}
		

		return (objects.size() > 0);
	}

	

private:

	cv::Point frame_size = cv::Point(320, 240);
	double ratio;
	/// debug
	int frame_id;
	bool debug_flag;
	/************************************************************************/
	/* CSK                                                                */
	/************************************************************************/
	double padding;
	double output_sigma_factor;
	double sigma;
	double lambda;
	double interp_factor;
	cv::Point pos;
	cv::Size target_sz, sz;
	bool resize_image;
	double output_sigma;
	cv::Mat y, yf, cos_window, im, im_gray, z, new_z, alphaf, new_alphaf, x, k, kf, response;
	
};


#endif
