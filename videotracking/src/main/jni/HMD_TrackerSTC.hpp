#ifndef __HMD_TrackerSTC_hpp__
#define __HMD_TrackerSTC_hpp__

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
#include "stc/STCTracker.h"

using namespace cv;
using namespace std;

/************************************************************************/
/* With LAB Color and 3-scale detection                                 */
/************************************************************************/

class TrackerSTC : public AbstractTracker
{
public:

	TrackerSTC() : AbstractTracker()
	{
		frame_id = 1;
		ratio = 1;
	}

	/// add a tracking obj
	virtual int addTrackingObject(const cv::Mat& source, const cv::Rect& obj_roi)
	{

		int obj_id = AbstractTracker::addTrackingObject(source, obj_roi);

		ratio = ratioObtain(source);

		cv::Mat im_gray, im;
		source.copyTo(im);
		resize(im, im, Size(cvRound(source.cols / ratio), cvRound(source.rows / ratio)));
		cv::cvtColor(source, im_gray, CV_RGB2GRAY);

		if (obj_id >= 0)
		{
			cv::Rect roi = obj_roi;
			roi.x /= ratio;
			roi.y /= ratio;
			roi.width /= ratio;
			roi.height /= ratio;
			iniRoi = roi;
			STCTracker tracker;
			tracker.init(im, roi, boxRegion);
			m_tracker.insert(std::make_pair(obj_id, tracker));
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

		cv::Mat im_gray, im;
		cv::cvtColor(target, im_gray, CV_RGB2GRAY);
		target.copyTo(im);
		resize(im, im, Size(cvRound(target.cols / ratio), cvRound(target.rows / ratio)));

		auto itr = m_active_objects.begin();
		for (; itr != m_active_objects.end(); ++itr) {
			STCTracker tracker = m_tracker.at(itr->first);
			cv::Rect match=iniRoi;
			tracker.tracking(im, match, boxRegion, frame_id);

				match.x *= ratio;
				match.y *= ratio;
				match.width *= ratio;
				match.height *= ratio;
				objects.insert(std::make_pair(itr->first, match));
				m_tracker.at(itr->first) = tracker;

		}
		return (objects.size() > 0);
	}

	

private:
	//////////////////////////////////////////////////////////////////////////
	std::map<int, STCTracker> m_tracker;
	cv::Rect boxRegion;
	//////////////////////////////////////////////////////////////////////////

	cv::Point frame_size = cv::Point(320, 240);
	double ratio;

	cv::Rect iniRoi;

	/// debug
	int frame_id;
	bool debug_flag;
};


#endif
