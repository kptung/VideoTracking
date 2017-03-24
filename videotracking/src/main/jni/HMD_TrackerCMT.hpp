#ifndef __HMD_TrackerCMT_hpp__
#define __HMD_TrackerCMT_hpp__

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
#include "CMT.h"

using namespace cv;
using namespace std;



class TrackerCMT : public AbstractTracker
{
public:

	TrackerCMT() : AbstractTracker()
	{
		//frame_id = 92;
		frame_id = 1;
		ratio = 1;
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
			resize(im_gray, im_gray, Size(cvRound(source.cols / ratio), cvRound(source.rows / ratio)));
			cv::Point2f initTopLeft(roi.x, roi.y);
			cv::Point2f initBottomDown(roi.x + roi.width - 1, roi.y + roi.height - 1);
			cmt.initialise(im_gray, initTopLeft, initBottomDown);
			cmt.processFrame(im_gray);
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
		
		resize(im_gray, im_gray, Size(cvRound(target.cols / ratio), cvRound(target.rows / ratio)));

		auto itr = m_active_objects.begin();
		for (; itr != m_active_objects.end(); ++itr) {
			cmt.processFrame(im_gray);
			int width = cmt.topRight.x - cmt.topLeft.x + 1;
			int height = cmt.bottomLeft.y - cmt.topLeft.y + 1;
			Rect match_roi(cmt.topLeft.x, cmt.topLeft.y, width, height);
			cv::Rect match = match_roi;
			match.x *= ratio;
			match.y *= ratio;
			match.width *= ratio;
			match.height *= ratio;
			objects.insert(std::make_pair(itr->first, match));
		}
		

		return (objects.size() > 0);
	}

	

private:
	//////////////////////////////////////////////////////////////////////////
	CMT cmt;
	//////////////////////////////////////////////////////////////////////////
	cv::Point frame_size = cv::Point(320, 240);
	double ratio;
	/// debug
	int frame_id;
	bool debug_flag;
};


#endif
