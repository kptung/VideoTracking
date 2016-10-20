#ifndef __HMD_TrackerTM_hpp__
#define __HMD_TrackerTM_hpp__

#include <list>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>    // std::min
#include <opencv2/opencv.hpp>

#include <stdlib.h>     /* NULL */
#include <assert.h>     /* assert */
#ifdef _WIN32
const std::string db_output("./tmp/");
#else
const std::string db_output("/sdcard/output/");
#include <jni.h>
#include <android/log.h>
#endif

#include <iostream>
#include <cstring>
#include <stdio.h>
#include <ctype.h>
#include <iomanip>
#include <fstream>

#include "HMD_AbstractTracker.hpp"

using namespace cv;
using namespace std;

class TrackerTM : public AbstractTracker
{
public:

	TrackerTM() :  AbstractTracker()
	{
		m_frame_id = 0;
		match_method = 5;
	}

	float Dist(Point &p, Rect &q)
	{
			Point qq;
			qq.x = q.x;
			qq.y = q.y;
			Point diff = p - qq;
			return cv::sqrt(float(diff.x*diff.x + diff.y*diff.y));
	}

	virtual int addTrackingObject(const cv::Mat& image, const cv::Rect& obj_roi)
	{
		int obj_id = AbstractTracker::addTrackingObject(image, obj_roi);

		if (obj_id >= 0)
		{
			m_active_roi.insert(std::make_pair(obj_id, obj_roi));
		}

		return obj_id;
	}
	
	
	bool runObjectTrackingAlgorithm (const cv::Mat& image, std::map<int, cv::Rect>& objects)
	{
		objects.clear();

		//std::cout << std::string("This is a tracker implmented using Template Matching algorithm.") << std::endl;
		++m_frame_id;



		//
		Mat edges,trackim;
		GaussianBlur(image, trackim, Size(3, 3), 0, 0);
		cvtColor(trackim, trackim, CV_BGR2GRAY);
		Canny(trackim, edges, 50, 200);

		//
		Mat result, result2;
		//
		auto itr = m_active_objects.begin();
		for ( ; itr != m_active_objects.end(); ++itr )
		{
			// Previous frame ROI position
			cv::Rect roi= m_active_roi[itr->first];
			
			Mat tmplate = itr->second;
			//Mat tmplate = Mat(image, roi);
			GaussianBlur(tmplate, tmplate, Size(3, 3), 0, 0);
			cvtColor(tmplate, tmplate, CV_BGR2GRAY);
			Mat edged;
			Canny(tmplate, edged, 50, 200);

			//Create the result matrix
			result.create(trackim.size(), CV_32FC1);
			result2.create(trackim.size(), CV_32FC1);
			matchTemplate(edges, edged, result, match_method);
			matchTemplate(trackim, tmplate, result2, match_method);
			normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
			normalize(result2, result2, 0, 1, NORM_MINMAX, -1, Mat());
			
			/// Localizing the best match with minMaxLoc
			double minVal, minVal2, maxVal, maxVal2;
			Point minLoc, minLoc2; 
			Point maxLoc, maxLoc2;
			Point matchLoc;
			minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
			minMaxLoc(result2, &minVal2, &maxVal2, &minLoc2, &maxLoc2, Mat());
			double minDis1 = Dist(minLoc, roi);
			double minDis2 = Dist(minLoc2, roi);
			double maxDis1 = Dist(maxLoc, roi);
			double maxDis2 = Dist(maxLoc2, roi);
			if (match_method == 0 || match_method == 1)
			{
				if (minDis1 < minDis2)
					matchLoc = minLoc;
				else
					matchLoc = minLoc2;
			}
			else 
			{
				if (maxDis1 < maxDis2)
					matchLoc = maxLoc;
				else
					matchLoc = maxLoc2;
			}
			objects.insert(std::make_pair(itr->first, cv::Rect(matchLoc.x, matchLoc.y, roi.width, roi.height)));
			auto itr2 = m_active_roi.find(itr->first);
			itr2->second = cv::Rect(matchLoc.x, matchLoc.y, roi.width, roi.height);
		}
		
		return (objects.size() > 0);
	}

private:
	//
	int m_frame_id;
	//
	int match_method;
	//
	std::map<int, cv::Rect> m_active_roi;

};


#endif
