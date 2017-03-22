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
#ifdef _WIN32
#else
const std::string db_output("/sdcard/output/");
#include <jni.h>
#include <android/log.h>
#endif

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
	}

	/// add a tracking obj
	virtual int addTrackingObject(const cv::Mat& source, const cv::Rect& obj_roi)
	{

		int obj_id = AbstractTracker::addTrackingObject(source, obj_roi);
		cv::Mat im_gray;
		cv::cvtColor(source, im_gray, CV_RGB2GRAY);
		cv::Point2f initTopLeft(obj_roi.x, obj_roi.y);
		cv::Point2f initBottomDown(obj_roi.x+obj_roi.width-1, obj_roi.y+obj_roi.height-1);
		cmt.initialise(im_gray, initTopLeft, initBottomDown);
		cmt.processFrame(im_gray);

/*		Mat img = source.clone();*/

// 		for (int i = 0; i < cmt.trackedKeypoints.size(); i++)
// 			cv::circle(img, cmt.trackedKeypoints[i].first.pt, 3, cv::Scalar(255, 255, 255));
// 		cv::line(img, cmt.topLeft, cmt.topRight, cv::Scalar(255, 255, 255));
// 		cv::line(img, cmt.topRight, cmt.bottomRight, cv::Scalar(255, 255, 255));
// 		cv::line(img, cmt.bottomRight, cmt.bottomLeft, cv::Scalar(255, 255, 255));
// 		cv::line(img, cmt.bottomLeft, cmt.topLeft, cv::Scalar(255, 255, 255));
		return obj_id;
	}

	
	/// Run the algorithm
	bool runObjectTrackingAlgorithm(const cv::Mat& target, std::map<int, cv::Rect>& objects)
	{
		objects.clear();

		frame_id++;

		cv::Mat im_gray;
		cv::cvtColor(target, im_gray, CV_RGB2GRAY);
		cmt.processFrame(im_gray);
// 		Mat img = target.clone();
// 		for (int i = 0; i < cmt.trackedKeypoints.size(); i++)
// 			cv::circle(img, cmt.trackedKeypoints[i].first.pt, 3, cv::Scalar(255, 255, 255));
// 		cv::line(img, cmt.topLeft, cmt.topRight, cv::Scalar(255, 255, 255));
// 		cv::line(img, cmt.topRight, cmt.bottomRight, cv::Scalar(255, 255, 255));
// 		cv::line(img, cmt.bottomRight, cmt.bottomLeft, cv::Scalar(255, 255, 255));
// 		cv::line(img, cmt.bottomLeft, cmt.topLeft, cv::Scalar(255, 255, 255));
		auto itr = m_active_objects.begin();
		for (; itr != m_active_objects.end(); ++itr) {
			int width = cmt.topRight.x - cmt.topLeft.x + 1;
			int height = cmt.bottomLeft.y - cmt.topLeft.y + 1;
			Rect match_roi(cmt.topLeft.x, cmt.topLeft.y, width, height);
			objects.insert(std::make_pair(itr->first, match_roi));
		}
		

		return (objects.size() > 0);
	}

	

private:

	CMT cmt;

	/// debug
	int frame_id;
	bool debug_flag;
};


#endif
