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
#include <queue>

#include "HMD_AbstractTracker.hpp"

#define UNKNOWN_FLOW_THRESH 1e9 

using namespace cv;
using namespace std;

class TrackerTM : public AbstractTracker
{
public:

	TrackerTM() :  AbstractTracker()
	{
		match_method = 5;
	}

	/// add a tracking obj
	virtual int addTrackingObject(const cv::Mat& image, const cv::Rect& obj_roi)
	{
		int obj_id = AbstractTracker::addTrackingObject(image, obj_roi);

		if (obj_id >= 0)
		{
			m_active_roi.insert(std::make_pair(obj_id, obj_roi));
			m_active_prev_roi.insert(std::make_pair(obj_id, obj_roi));
			m_active_pts_vector.insert(std::make_pair(obj_id, Point(0,0)));
			m_active_missing.insert(std::make_pair(obj_id, 0));
		}

		return obj_id;
	}

	/// calculate the distance between 2 regions
	float getDist(const Rect &p, const Rect &q)
	{
		Point p1 = Point(p.x + p.width / 2, p.y + p.height / 2);
		Point p2 = Point(q.x + q.width / 2, q.y + q.height / 2);
		return getDist(p1, p2);
	}

	/// calculate the distance between 2 points
	float getDist(const Point &p, const Point &q)
	{
		Point diff = p - q;
		return cv::sqrt(float(diff.x*diff.x + diff.y*diff.y));
	}
	
	/// obtain the sum of absolute difference (SAD) between 2 images 
	double getSAD(const Mat& a, const Mat& b)
	{
		Scalar x = cv::sum(cv::abs(a - b));
		if (a.channels()==1 || b.channels()==1)
			return x.val[0];
		else
			return (x.val[0] + x.val[1] + x.val[2])/3;
	}

	/// calculate the moving vector from the current frame and the previous frame
	Point getLinevec(const Rect& prev_roi, const Rect& roi)
	{
		Point p1 = Point(prev_roi.x, prev_roi.y);
		Point p2 = Point(roi.x, roi.y);
		Point vec = p2 - p1;
		return vec;
	}

	/// use the moving vector from the current frame and the previous frame 2 predict the region
	Rect getLinepts(const Rect& prev_roi, const Point& vec)
	{
		Rect match=prev_roi;
		double unitvec = 0.4;
		match.x = prev_roi.x + vec.x * unitvec;
		match.y = prev_roi.y + vec.y * unitvec;
		return match;
	}

	/// convert the image 2 its edge image
	void convert2edge(const Mat& source, Mat& destination)
	{
		/// initialization
		int edge_flag = 3;
		int th1 = 100, th2 = 200;
		Mat gx, absgx, gy, absgy, dst;

		/// selection
		switch (edge_flag)
		{
		case 1:
			Canny(source, destination, th1, th2);
			break;
		case 2:
			Laplacian(source, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT);
			convertScaleAbs(dst, destination);
			break;
		case 3:		
			Sobel(source, gx, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
			convertScaleAbs(gx, absgx);
			Sobel(source, gy, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
			convertScaleAbs(gy, absgy);
			addWeighted(absgx, 0.5, absgy, 0.5, 0, destination);
			break;
		default:
			break;
		}
	}

	/// the Gaussian weight of the distance between two regions
	double getLocWeight(const Rect& r1, const Rect& r2)
	{
		/// initialization
		Point diff = Point(r1.x+r1.width/2,r1.y+r1.height/2) - Point(r2.x+r2.width/2, r2.y+r2.height/2);
		
		/// define the searching range
		int sigma = (min(r1.width, r1.height) - 1) / 2;
		
		/// calculate the distance between 2 points
		double dist = getDist(Point(r1.x + r1.width / 2, r1.y + r1.height / 2), Point(r2.x + r2.width / 2, r2.y + r2.height / 2));
		
		if (dist <= 2 * sigma)
			return (1 / sqrt(2 * CV_PI * sigma * sigma)) * exp(double(-(diff.x*diff.x + diff.y*diff.y) / (2 * sigma * sigma)));
		else
			return -1;
	}

	/// Obtain the possibility regions by thresholding
	vector<Point> getLocs(const Mat& src)
	{
		/// threshold
		float maxValue = 0.9;
		vector<Point> locs;
		
		for (int j = 0; j < src.rows; j++)
		{
			for (int i = 0; i < src.cols; i++)
			{
				float v = src.at<float>(Point(i,j));
				if (v > maxValue)
					locs.push_back(Point(i, j));
			}
		}
		
		return locs;
	}
	
	/// Obtain the highest possibility region by using the fusion weight
	Rect getMaxloc(const Mat& weight, const vector<Point>& locs, const Rect& prev_roi, const Mat& tmplate, const Mat& target)
	{
		/// weighting ratio
		double r = 0.5;

		/// initialization
		vector<Point> pos;
		double sum = 1e10;
		Rect rec;

		/// looping
		for (int i = 0; i < locs.size(); i++)
		{
			Point loc = locs.at(i);
			
			/// weight 1 is obtained from the template matching; if the value is higher, it means that the region is like the compared one 
			double w1= weight.at<float>(loc);

			/// weight 2 is obtained from the SAD difference; if the value is lower, it means that the region is like the compared one 
			Rect roi = Rect(loc.x, loc.y, prev_roi.width, prev_roi.height);
			double w2 = getSAD(tmplate, target(roi));

			/// total weight is the fusion weight: half weight is from the template matching and another half is from the SAD difference
			/// since the weight 1 needs higher value and the weight 2 prefers the lower one, it can be modify the weight 1 to match the requirement (find the smallest)
			double w = (1-r) * (1-w1) + r * w2;

			/// by comparing the fusion weight, the suitable region is found (the smallest value)
			if (w < sum)
			{
				sum = w;
				rec = roi;
			}	
		}
		return rec;
	}
	
	// Template Matching 
	Rect TMatch(const Mat& target, const Mat& tmplate, const Rect prev_roi)
	{
		/// set search range of the target image (3 times larger than the template)
		Rect new_roi = prev_roi;
		new_roi.x = new_roi.x - new_roi.width;
		new_roi.y = new_roi.y - new_roi.height;
		new_roi.width = new_roi.width * 3;
		new_roi.height = new_roi.height * 3;

		/// boundary check
		new_roi.x = max(0, new_roi.x);
		new_roi.x = min(new_roi.x, target.cols);
		new_roi.y = max(0, new_roi.y);
		new_roi.y = min(new_roi.y, target.rows);
		int w = new_roi.x + new_roi.width;
		w = max(0, w);
		w = min(w, target.cols);
		new_roi.width = w - new_roi.x;
		int h = new_roi.y + new_roi.height;
		h = max(0, h);
		h = min(h, target.rows);
		new_roi.height = h - new_roi.y;

		/// Extract the searching image using mask
		Mat mask = cv::Mat::zeros(target.rows, target.cols, CV_8U);
		mask(new_roi) = 255;
		Mat new_search;
		target.copyTo(new_search,mask);
		
		/// Do the Matching and Normalize
		Mat result;
		matchTemplate(new_search, tmplate, result, match_method);
		normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
		if (match_method == 0 || match_method == 1)
			result = 1 - result;

		/// find the possible regions
		vector<Point>locs = getLocs(result);

		/// return the highest possibility region
		return getMaxloc(result, locs, prev_roi, tmplate, target);
	}

	/// Run the algorithm
	bool runObjectTrackingAlgorithm (const cv::Mat& source, std::map<int, cv::Rect>& objects)
	{
		objects.clear();

		//std::cout << std::string("This is a tracker implmented using Template Matching algorithm.") << std::endl;
		//m_frame_id++;
		//printf("Frame %d\n", m_frame_id);

		/// convert ARGB image 2 RGB image  
		Mat image;
		if (source.channels() == 4)
			cvtColor(source, image, CV_BGRA2BGR);
		else
			image = source.clone();

		/// The target image and its edge image
		Mat edges,gray_track_im;
		cvtColor(image, gray_track_im, CV_BGR2GRAY);
		convert2edge(gray_track_im, edges);
				
		/// the original frame templates
		auto itr = m_active_objects.begin();
		for ( ; itr != m_active_objects.end(); ++itr )
		{
			/// original template
			Mat tmplate = itr->second;
			cv::Rect roi = m_active_roi.at(itr->first);
			Mat gray_tmplate;
			cvtColor(tmplate, gray_tmplate, CV_BGR2GRAY);
			Mat edged;
			convert2edge(gray_tmplate, edged);

			/// Previous frame ROI position & previous template
			cv::Rect prev_roi = m_active_prev_roi.at(itr->first);

			/// Template matching 2 find the most similar region; m1/m2: the original target andits edge image  
			Rect m1rec = TMatch(image, tmplate, prev_roi);
			Rect m2rec = TMatch(edges, edged, prev_roi);

			/// Using the Gaussian weight of the distance 2 compare the regions
			double s1 = getLocWeight(prev_roi, m1rec);
			double s2 = getLocWeight(prev_roi, m2rec);
			
			/// comparison: the higher value means the region is closed to the previous frame's roi
			Rect match_roi;
			if (s1 > 0 && s2 > 0)
			{
				if (s1 > s2)
					match_roi = m1rec;
				else
					match_roi = m2rec;
			}
			else if (s1 > 0 && s2 < 0)
				match_roi = m1rec;
			else if (s2 > 0 && s1 < 0)
				match_roi = m2rec;

			/// update 
			/// if the found regions are far away, do rematch or predict the possible region by using the moving vector 
			if (s1 < 0 && s2 < 0)
			{
				/// if the template region is missing in the current frame, the missing frame number adds 1
				m_active_missing.at(itr->first) += 1;
				
				/// If the number of frame's missing template is > 4, do rematch
				/// else use the moving vector obtained from the previous frame and the previous 2 frame 2 predict the region in the current frame  
				Rect adj_roi;
				if (m_active_missing.at(itr->first) > 4)
					adj_roi = TMatch(image, tmplate, prev_roi);
				else
					adj_roi = getLinepts(prev_roi, m_active_pts_vector.at(itr->first));

				/// boundary check
				adj_roi.x = max(-1, adj_roi.x);
				adj_roi.x = min(adj_roi.x,image.cols);
				adj_roi.y = max(-1, adj_roi.y);
				adj_roi.y = min(adj_roi.y, image.rows);

				/// draw + save tracking region
				objects.insert(std::make_pair(itr->first, adj_roi));

				/// update the roi 
				m_active_prev_roi.at(itr->first) = adj_roi;

				/// update the moving vector from the current frame and the previous frame 
				m_active_pts_vector.at(itr->first) = getLinevec(prev_roi, adj_roi);
			}
			else if (s1 > 0 || s2 > 0 )
			{
				/// draw + save tracking region
				objects.insert(std::make_pair(itr->first, match_roi));

				/// update the roi 
				m_active_prev_roi.at(itr->first) = match_roi;

				/// update the moving vector from the current frame and the previous frame
				m_active_pts_vector.at(itr->first) = getLinevec(prev_roi, match_roi);

				/// if the template region is found, set the missing frame number 2 zero 
				m_active_missing.at(itr->first)=0 ;
			}
		}
		return (objects.size() > 0);
	}

private:

	/// template matching method: 0/1: SQDIFF/Normalized-SQDIFF; 2/3: TM_CCORR/Normalized-TM_CCORR; 4/5: CCOEFF/Normalized-CCOEFF; 
	int match_method;

	/// the template roi in the start frame 
	std::map<int, cv::Rect> m_active_roi;

	/// the template roi in the previous frame 
	std::map<int, cv::Rect> m_active_prev_roi;

	/// the template moving vector 
	std::map<int, cv::Point> m_active_pts_vector;

	/// the frequency of missing template 
	std::map<int, int> m_active_missing;
};


#endif
