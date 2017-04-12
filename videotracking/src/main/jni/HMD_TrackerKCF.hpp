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
			
			//calculate parent roi
			success = tracker->init(source.clone(), roi);
			
			//imshow("", source(parent_roi));
			//waitKey(0);
			if (success) {
				tracker->update(source.clone(), roi, false);
				m_active_roi_target.insert(std::make_pair(obj_id, roi));			
				m_tracker_kcf_target.insert(std::make_pair(obj_id, tracker));
				m_tracker_kcf_target_state.insert(std::make_pair(obj_id, true));
				parent_associate.insert(std::make_pair(obj_id, std::map<int, bool>()));
				for (int i = 1; i < 10; i++) {
					parent_associate.at(obj_id).insert(std::make_pair(i, false));
				}
				tracker_trust_point.insert(std::make_pair(obj_id, 0));
			} else {
				obj_id = -1;
			}

		}

		return obj_id;
	}

	/// Run the algorithm
	bool runObjectTrackingAlgorithm(const cv::Mat& target, std::map<int, cv::Rect>& objects)
	{
		bool status = false;
		bool add_flag = false;
		objects.clear();
		frame_id++;
		Ptr< III_TrackerKCFImpl > tracker;
		
		runParentTracking(target);
		auto itr_object = m_active_objects.begin();
		for (; itr_object != m_active_objects.end(); ++itr_object) {
			if (m_tracker_kcf_target.count(itr_object->first) == 0)continue;
			tracker = m_tracker_kcf_target.at(itr_object->first);
			cv::Rect2d roi_target = m_active_roi_target.at(itr_object->first);

			if (m_tracker_kcf_target_state.at(itr_object->first)) {
				bool rt_target = tracker->update(target, roi_target, false);
				double trust = tracker->get_trust_point();
				tracker_trust_point.erase(itr_object->first);
				tracker_trust_point.insert(std::make_pair(itr_object->first, trust));
				if (trust > 4.5) {
					add_flag = true;
					printf("trust = %f\n", trust);
				}
				if (rt_target) {
					m_active_roi_target.erase(itr_object->first);
					m_active_roi_target.insert(std::make_pair(itr_object->first, roi_target));
					objects.insert(std::make_pair(itr_object->first, roi_target));
					status = true;
				} else {
					objects.insert(std::make_pair(itr_object->first, Rect2d(-1, -1, -1, -1)));
					m_tracker_kcf_target_state.erase(itr_object->first);
					m_tracker_kcf_target_state.insert(std::make_pair(itr_object->first, false));
				}
			}
		}

		if (add_flag) {
			addParentObject(target);
			addOffsetInfo();
		}

		/*
		for (int i = 0; i < m_active_roi_parent.size(); i++) {
			if (m_active_roi_parent.count(i) == 0)continue;
			if (m_tracker_kcf_parent_state.count(i)==0)continue;
			if (!m_tracker_kcf_parent_state.at(i))continue;
			objects.insert(std::make_pair(i+6, m_active_roi_parent.at(i)));
		}*/

		return status;
	}

	bool remTrackingObject(int obj_id) {
		vector<int> keyset_parent;
		for (std::map<int, std::map<int, cv::Point>>::iterator it = parent_tracker_offset.begin(); it != parent_tracker_offset.end(); ++it) {
			keyset_parent.push_back(it->first);
		}

		AbstractTracker::remTrackingObject(obj_id);
		m_tracker_kcf_target.erase(obj_id);
		m_tracker_kcf_target_state.erase(obj_id);
		tracker_trust_point.erase(obj_id);
		parent_associate.erase(obj_id);
		m_active_roi_target.erase(obj_id);
		tracker_tracker_offset.erase(obj_id);
		for (int i = 0; i < parent_tracker_offset.size(); i++) {
			if (!parent_tracker_offset.count(keyset_parent.at(i)))continue;
			parent_tracker_offset.at(keyset_parent.at(i)).erase(obj_id);
			if (parent_tracker_offset.at(keyset_parent.at(i)).empty()) {
				parent_tracker_offset.erase(keyset_parent.at(i));
				m_active_roi_parent.erase(keyset_parent.at(i));
				m_tracker_kcf_parent_state.erase(keyset_parent.at(i));
				m_tracker_kcf_parent.erase(keyset_parent.at(i));
			}
		}

		return true;
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

	void addParentObject(const cv::Mat& source) {
		
		bool add_flag = false;
		int area;
		int area_x;
		int area_y;
		int acc_x[3];
		int acc_y[3];
		int med_x;
		int med_y;
		int maximum_x = 0;
		int maximum_y = 0;
		acc_x[0] = 0;
		acc_x[1] = 0;
		acc_x[2] = 0;
		acc_y[0] = 0;
		acc_y[1] = 0;
		acc_y[2] = 0;
		vector<int> keyset;
		for (std::map<int, cv::Rect>::iterator it = m_active_roi_target.begin(); it != m_active_roi_target.end(); ++it) {
			keyset.push_back(it->first);
		}

		for (int i = 0; i < keyset.size(); i++) {
			if (m_active_roi_target.count(keyset.at(i)) == 0)continue;
			Rect tracker_i = m_active_roi_target.at(keyset.at(i));
			area_x = (tracker_i.x + tracker_i.width / 2) / (source.cols / 3);
			if (area_x < 0 || area_x>2)continue;
			area_y = (tracker_i.y + tracker_i.height / 2) / (source.rows / 3);
			if (area_y < 0 || area_y>2)continue;
			area = area_y * 3 + area_x + 1;
			if (parent_associate.at(keyset.at(i)).at(area))continue;
			if (tracker_trust_point.at(keyset.at(i)) < 3)continue;
			if (calcActivatedParent() > 5)continue;
			parent_associate.at(keyset.at(i)).erase(area);
			parent_associate.at(keyset.at(i)).insert(std::make_pair(area, true));
			add_flag = true;
			acc_x[area_x] += 1;
			acc_y[area_y] += 1;
		}
		
		if (!add_flag)return;

		for (int i = 0; i < 3; i++) {
			if (acc_x[i] > maximum_x) {
				maximum_x = acc_x[i];
				med_x = i;
			}
			if (acc_y[i] > maximum_y) {
				maximum_y = acc_y[i];
				med_y = i;
			}
		}
	
		Ptr< III_TrackerKCFImpl > parent = Ptr<III_TrackerKCFImpl>(new III_TrackerKCFImpl(III_TrackerKCFImpl::Params(true)));
		Rect2d parent_roi = Rect2d();
		bool success = false;
		
		int parent_id = m_tracker_kcf_parent.size();
		for (int i = 0; i < m_tracker_kcf_parent.size(); i++) {
			if (m_tracker_kcf_parent.count(i) == 0)parent_id = i;
		}
		parent_roi.width = 0.6*source.cols;
		parent_roi.height = 0.6*source.rows;
		parent_roi.x = (0.4 - (0.2*med_x))*source.cols;
		parent_roi.y = (0.4 - (0.2*med_y))*source.rows;
		success = parent->init(source.clone(), parent_roi);
		if (success) {
			parent->update(source.clone(), parent_roi, false);
			m_tracker_kcf_parent.insert(std::make_pair(parent_id, parent));
			m_active_roi_parent.insert(std::make_pair(parent_id, parent_roi));
		}
	}

	void runParentTracking(const cv::Mat& target) {
		Ptr< III_TrackerKCFImpl > parent;
		vector<int> keyset_parent;
		for (std::map<int, cv::Rect>::iterator it = m_active_roi_parent.begin(); it != m_active_roi_parent.end(); ++it) {
			keyset_parent.push_back(it->first);
		}
		vector<int> keyset_tracker;
		for (std::map<int, cv::Rect>::iterator it = m_active_roi_target.begin(); it != m_active_roi_target.end(); ++it) {
			keyset_tracker.push_back(it->first);
		}

		for (int i=0; i< m_active_roi_parent.size(); i++) {
			if (m_active_roi_parent.count(keyset_parent.at(i)) == 0)continue;
			if (parent_tracker_offset.count(keyset_parent.at(i)) == 0)continue;
			parent = m_tracker_kcf_parent.at(keyset_parent.at(i));
			cv::Rect2d roi_parent = (Rect2d)m_active_roi_parent.at(keyset_parent.at(i));
			bool rt_parent = parent->update(target, roi_parent, false);
			m_tracker_kcf_parent_state.erase(keyset_parent.at(i));
			m_tracker_kcf_parent_state.insert(std::make_pair(keyset_parent.at(i), rt_parent));
			if (rt_parent) {
				for (int j = 0; j < parent_tracker_offset.at(keyset_parent.at(i)).size(); j++) {
					Point offset;
					if (j >= keyset_tracker.size()) {
						break;
					}
					if (parent_tracker_offset.at(keyset_parent.at(i)).count(keyset_tracker.at(j)) == 0)continue;
					offset = parent_tracker_offset.at(keyset_parent.at(i)).at(keyset_tracker.at(j));
					Rect2d roi_target = m_active_roi_target.at(keyset_tracker.at(j));
					Ptr<III_TrackerKCFImpl> tracker = m_tracker_kcf_target.at(keyset_tracker.at(j));
					if ((!m_tracker_kcf_target_state.at(keyset_tracker.at(j))) &&
						(roi_parent.x + offset.x) > 0 &&
						(roi_parent.y + offset.y) > 0 &&
						(roi_parent.x + offset.x) < target.cols &&
						(roi_parent.y + offset.y) < target.rows) {
						Rect2d newroi = Rect2d(roi_parent.x + offset.x, roi_parent.y + offset.y, roi_target.width, roi_target.height);
						bool rt = tracker->update(target, newroi, true);
						if (rt) {
							m_tracker_kcf_target_state.erase(keyset_tracker.at(j));
							m_tracker_kcf_target_state.insert(std::make_pair(keyset_tracker.at(j), true));
						}
					}
				}
				m_active_roi_parent.erase(keyset_parent.at(i));
				m_active_roi_parent.insert(std::make_pair(keyset_parent.at(i), roi_parent));
				m_tracker_kcf_parent_state.erase(keyset_parent.at(i));
				m_tracker_kcf_parent_state.insert(std::make_pair(keyset_parent.at(i), true));

			} else {
				roi_parent = m_active_roi_parent.at(keyset_parent.at(i));
				roi_parent.x = 0;
				roi_parent.y = 0;
				m_active_roi_parent.erase(keyset_parent.at(i));
				m_active_roi_parent.insert(std::make_pair(keyset_parent.at(i), roi_parent));
				m_tracker_kcf_parent_state.erase(keyset_parent.at(i));
				m_tracker_kcf_parent_state.insert(std::make_pair(keyset_parent.at(i), false));
			}
		}

	}

	void addOffsetInfo() {

		vector<int> keyset_parent;
		for (std::map<int, cv::Rect>::iterator it = m_active_roi_parent.begin(); it != m_active_roi_parent.end(); ++it) {
			keyset_parent.push_back(it->first);
		}
		vector<int> keyset_tracker;
		for (std::map<int, cv::Rect>::iterator it = m_active_roi_target.begin(); it != m_active_roi_target.end(); ++it) {
			keyset_tracker.push_back(it->first);
		}

		for (int i = 0;i< m_active_roi_parent.size(); i++) {
			if (m_active_roi_parent.count(keyset_parent.at(i)) == 0)continue;
			Rect parent_roi = m_active_roi_parent.at(keyset_parent.at(i));
			for (int j = 0;j<m_active_roi_target.size(); j++) {
				if (m_tracker_kcf_target_state.count(keyset_tracker.at(j)) == 0)continue;
				if (m_tracker_kcf_target_state.at(keyset_tracker.at(j))) {
					parent_tracker_offset.insert(std::make_pair(keyset_parent.at(i), std::map<int, cv::Point>()));
					Rect roi = m_active_roi_target.at(keyset_tracker.at(j));
					parent_tracker_offset.at(keyset_parent.at(i)).insert(std::make_pair(keyset_tracker.at(j), cv::Point(roi.x - parent_roi.x, roi.y - parent_roi.y)));
				}
			}
		}
	}

	int calcActivatedParent() {
		int ret = 0;
		vector<int> keyset_parent;
		for (std::map<int, cv::Rect>::iterator it = m_active_roi_parent.begin(); it != m_active_roi_parent.end(); ++it) {
			keyset_parent.push_back(it->first);
		}

		for (int i = 0; i < m_tracker_kcf_parent_state.size(); i++) {
			if (m_tracker_kcf_parent_state.count(keyset_parent.at(i)) == 0)continue;
			if (m_tracker_kcf_parent_state.at(keyset_parent.at(i)))ret += 1;
		}

		return ret;
	}

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

	//trackerID, point
	std::map<int, double> tracker_trust_point;

	//trackerID, map(associate parent position, TorF)
	std::map<int, std::map<int, bool>> parent_associate;

	//parentID, map(trackerID, position xy)
	std::map<int, std::map<int, cv::Point>> parent_tracker_offset;

	//trackerID, map(trackerID, position xy)
	std::map<int, std::map<int, cv::Point>> tracker_tracker_offset;

	///
	bool imflag;

	/// debug
	int frame_id;
	bool debug_flag;
};

#endif

