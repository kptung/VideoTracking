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

	int ratioObtain(const Mat& image)
	{
		bool resiez_flag = true;
		int rows = image.rows;
		int cols = image.cols;
		while (resiez_flag)
		{
			int shortedge = min(rows, cols);
			if (shortedge > 240)
			{
				rows = rows >> 1;
				cols = cols >> 1;
			}
			else
				resiez_flag = false;
		}

		return image.cols / cols;
	}

	/// add a tracking obj
	virtual int addTrackingObject(const cv::Mat& source, const cv::Rect& obj_roi)
	{

		int obj_id = AbstractTracker::addTrackingObject(source, obj_roi);
		cv::Mat im_gray;
		cv::cvtColor(source, im_gray, CV_RGB2GRAY);
		ratio = ratioObtain(im_gray);
		Rect roi = obj_roi;
		if (ratio > 1)
		{
			cv::resize(im_gray, im_gray, Size(cvRound(im_gray.cols / ratio), cvRound(im_gray.rows / ratio)));
			roi.x /= ratio;
			roi.y /= ratio;
			roi.width /= ratio;
			roi.height /= ratio;
		}
		cout << "ratio: " << ratio << endl;
		cv::Point2f initTopLeft(roi.x, roi.y);
		cv::Point2f initBottomDown(roi.x+roi.width-1, roi.y+roi.height-1);
		cmt.initialise(im_gray, initTopLeft, initBottomDown);
		cmt.processFrame(im_gray);

		return obj_id;
	}

	
	/// Run the algorithm
	bool runObjectTrackingAlgorithm(const cv::Mat& target, std::map<int, cv::Rect>& objects)
	{
		objects.clear();

		frame_id++;

		cv::Mat im_gray;
		cv::cvtColor(target, im_gray, CV_RGB2GRAY);
		if (ratio > 1)
			cv::resize(im_gray, im_gray, Size(cvRound(im_gray.cols / ratio), cvRound(im_gray.rows / ratio)));
		cmt.processFrame(im_gray);

		auto itr = m_active_objects.begin();
		for (; itr != m_active_objects.end(); ++itr) {
			int width = cmt.topRight.x - cmt.topLeft.x + 1;
			int height = cmt.bottomLeft.y - cmt.topLeft.y + 1;
			Rect match_roi(cmt.topLeft.x, cmt.topLeft.y, width, height);
			if (ratio > 1)
			{
				match_roi.x *= ratio;
				match_roi.y *= ratio;
				match_roi.width *= ratio;
				match_roi.height *= ratio;
			}
			objects.insert(std::make_pair(itr->first, match_roi));
		}
		

		return (objects.size() > 0);
	}

	

private:
	//////////////////////////////////////////////////////////////////////////
	CMT cmt;
	//////////////////////////////////////////////////////////////////////////
	int ratio;
	/// debug
	int frame_id;
	bool debug_flag;
};


#endif
