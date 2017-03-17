#ifndef __HMD_TrackerKCL_hpp__
#define __HMD_TrackerKCL_hpp__

#include <list>
#include <vector>
#include <string>
#include <limits>
#include <algorithm>    // std::min
#include <opencv2/opencv.hpp>

#include <stdlib.h>     /* NULL */
#include <assert.h>     /* assert */

#include <iostream>
#include <cstring>
#include <stdio.h>
#include <ctype.h>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <functional>
#include "HMD_AbstractTracker.hpp"
#include "III_TrackerKCFImpl.hpp"

using namespace cv;
using namespace std;

class HMD_TrackerKCF :
	public AbstractTracker
{
public:

	HMD_TrackerKCF() :AbstractTracker()
	{
		m_tracker_kcf_target = std::map<int, Ptr<III_TrackerKCFImpl>>();
	}

	int addTrackingObject(const cv::Mat& source, const cv::Rect& obj_roi)
	{
		Rect2d roi = obj_roi;
		bool success;

		int obj_id = AbstractTracker::addTrackingObject(source, obj_roi);

		if (obj_id >= 0)
		{
			m_active_roi_target.insert(std::make_pair(obj_id, obj_roi));
			Ptr< III_TrackerKCFImpl > tracker = Ptr<III_TrackerKCFImpl>(new III_TrackerKCFImpl());
			Ptr< III_TrackerKCFImpl > parent = Ptr<III_TrackerKCFImpl>(new III_TrackerKCFImpl());
			Rect2d parent_roi = Rect2d();
			//calculate parent roix
			parent_roi.x = round(roi.x - ceil((double)source.cols / (double)roi.width / 2 - 1)*roi.width / 2);
			if (parent_roi.x < 0) {
				parent_roi.x = 0;
			}
			parent_roi.width = roi.width * ceil((double)source.cols / (double)roi.width / 2);
			if ((parent_roi.x + parent_roi.width) > (source.cols - 1)) {
				parent_roi.width = source.cols - 1 - parent_roi.x;
			}
			//calculate parent roiy
			parent_roi.y = round(roi.y - ceil((double)source.rows / (double)roi.height / 2 - 1)*roi.height / 2);
			if (parent_roi.y < 0) {
				parent_roi.y = 0;
			}
			parent_roi.height = roi.height * ceil((double)source.rows / (double)roi.height / 2);
			if ((parent_roi.y + parent_roi.height) > (source.rows - 1)) {
				parent_roi.height = source.rows - 1 - parent_roi.y;
			}
			success = tracker->init(source.clone(), roi);
			success = parent->init(source.clone(), parent_roi);
			//imshow("", source(parent_roi));
			//waitKey(0);
			if (success) {
				tracker->update(source.clone(), roi, false);
				parent->update(source.clone(), parent_roi, false);
				tracker_offset.insert(std::make_pair(obj_id, cv::Point(roi.x - parent_roi.x, roi.y - parent_roi.y)));
				m_active_roi_target.insert(std::make_pair(obj_id, roi));
				m_active_roi_parent.insert(std::make_pair(obj_id, parent_roi));
				m_tracker_kcf_target.insert(std::make_pair(obj_id, tracker));
				m_tracker_kcf_parent.insert(std::make_pair(obj_id, parent));
				m_tracker_kcf_target_state.insert(std::make_pair(obj_id, true));
			}
			else {
				obj_id = -1;
			}

		}

		return obj_id;
	}

	/// Run the algorithm
	bool runObjectTrackingAlgorithm(const cv::Mat& target, std::map<int, cv::Rect>& objects)
	{
		bool status = false;
		objects.clear();
		frame_id++;
		Ptr< III_TrackerKCFImpl > tracker;
		Ptr< III_TrackerKCFImpl > parent;

		auto itr_object = m_active_objects.begin();
		for (; itr_object != m_active_objects.end(); ++itr_object) {
			tracker = m_tracker_kcf_target.at(itr_object->first);
			parent = m_tracker_kcf_parent.at(itr_object->first);
			cv::Rect2d roi_target = m_active_roi_target.at(itr_object->first);
			cv::Rect2d roi_parent = m_active_roi_parent.at(itr_object->first);

			if (m_tracker_kcf_target_state.at(itr_object->first)) {
				bool rt_target = tracker->update(target, roi_target, false);

				if (rt_target) {
					m_active_roi_target.erase(itr_object->first);
					m_active_roi_target.insert(std::make_pair(itr_object->first, roi_target));
					objects.insert(std::make_pair(itr_object->first, roi_target));
					tracker_offset.erase(itr_object->first);
					tracker_offset.insert(std::make_pair(itr_object->first, cv::Point(roi_target.x - roi_parent.x, roi_target.y - roi_parent.y)));
					status = true;
				}
				else {
					m_tracker_kcf_target_state.erase(itr_object->first);
					m_tracker_kcf_target_state.insert(std::make_pair(itr_object->first, false));
				}
			}

			bool rt_parent = parent->update(target, roi_parent, false);
			if (rt_parent) {
				if (!m_tracker_kcf_target_state.at(itr_object->first)) {
					Point offset = tracker_offset.at(itr_object->first);
					if ((roi_parent.x + offset.x) > 0 && (roi_parent.y + offset.y) > 0 &&
						(roi_parent.x + offset.x) < target.cols && (roi_parent.y + offset.y) < target.rows) {
						Rect2d newroi = Rect2d(roi_parent.x + offset.x, roi_parent.y + offset.y, 0, 0);
						bool rt = tracker->update(target, newroi, true);
						if (rt) {
							m_tracker_kcf_target_state.erase(itr_object->first);
							m_tracker_kcf_target_state.insert(std::make_pair(itr_object->first, true));
						}
					}
				}
				m_active_roi_parent.erase(itr_object->first);
				m_active_roi_parent.insert(std::make_pair(itr_object->first, roi_parent));
				//objects.insert(std::make_pair(itr_object->first + 2, roi_parent));
			} else {
				m_tracker_kcf_target_state.erase(itr_object->first);
				m_tracker_kcf_target_state.insert(std::make_pair(itr_object->first, false));
			}
		}

		return status;
	}

	/// Debug: imshow 
	void imgshow(const cv::Mat& img)
	{
		namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", img);
		waitKey(0);
		cvDestroyAllWindows();
	}

	void imgshow(const cv::Mat& img, const cv::Rect& roi)
	{
		//cv::Mat new_img = extrROI(img, roi);
		/*namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", new_img);
		waitKey(0);
		cvDestroyAllWindows();*/
	}
	void imgshow(const cv::Mat& img, const cv::Mat& mask)
	{
		cv::Mat tmp;
		img.copyTo(tmp, mask);
		namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", tmp);
		waitKey(0);
		cvDestroyAllWindows();
	}
	void imgshow(const cv::Mat& img, const cv::Mat& mask, const cv::Rect& roi)
	{
		cv::Mat tmp;
		img.copyTo(tmp, mask);
		namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", tmp(roi));
		waitKey(0);
		cvDestroyAllWindows();
	}

private:

	/// image size definition
	Point imsizew, imsizeh;

	/// template matching method: 0/1: SQDIFF/Normalized-SQDIFF; 2/3: TM_CCORR/Normalized-TM_CCORR; 4/5: CCOEFF/Normalized-CCOEFF; 
	int match_method;

	/// target roi 
	std::map<int, cv::Rect> m_active_roi_target;

	/// parent roi 
	std::map<int, cv::Rect> m_active_roi_parent;

	/// target tracker
	std::map<int, Ptr<III_TrackerKCFImpl >> m_tracker_kcf_target;

	/// parent tracker
	std::map<int, Ptr<III_TrackerKCFImpl >> m_tracker_kcf_parent;

	/// target tracker state
	std::map<int, bool> m_tracker_kcf_target_state;

	/// parent tracker state
	std::map<int, bool> m_tracker_kcf_parent_state;

	std::map<int, cv::Point> tracker_offset;

	///
	bool imflag;

	/// debug
	int frame_id;
	bool debug_flag;
};

#endif

