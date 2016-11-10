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
		match_method = 5;
		resampling = true;
		imsize = Point(160, 120);
		
		//
		frame_id = 224;
		debug_flag = false;
	}

	/// add a tracking obj
	virtual int addTrackingObject(const cv::Mat& source, const cv::Rect& obj_roi)
	{
		
		int obj_id = AbstractTracker::addTrackingObject(source, obj_roi);

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
		Point p1 = Point(cvRound(p.x + p.width * 0.5), cvRound(p.y + p.height * 0.5));
		Point p2 = Point(cvRound(q.x + q.width * 0.5), cvRound(q.y + q.height * 0.5));
		return getDist(p1, p2);
	}

	/// calculate the distance between 2 points
	float getDist(const Point &p, const Point &q)
	{
		Point diff = p - q;
		return cv::sqrt(float(diff.x*diff.x + diff.y*diff.y));
	}

	/// obtain the sum of absolute difference (SAD) between 2 images 
	double getSAD(const Mat& source, const Mat& target)
	{
		Scalar x = cv::sum(cv::abs(source - target));
		double val = 0;
		return val = (source.channels() == 1 || source.channels() == 1) ? x.val[0] : (x.val[0] + x.val[1] + x.val[2]) / 3;
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
		match.x = cvRound(prev_roi.x + vec.x * unitvec);
		match.y = cvRound(prev_roi.y + vec.y * unitvec);
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
		int sigma = cvRound((min(r1.width, r1.height)) * 0.5);
		
		/// calculate the distance between 2 points
		double dist = getDist(Point(cvRound(r1.x + r1.width * 0.5), cvRound(r1.y + r1.height * 0.5)), Point(cvRound(r2.x + r2.width * 0.5), cvRound(r2.y + r2.height * 0.5)));
		
		return (dist <= (3 * sigma)) ? (1 / sqrt(2 * CV_PI * sigma * sigma)) * exp(double(-(diff.x*diff.x + diff.y*diff.y) / (2 * sigma * sigma))) : -1;
	}

	/// Obtain the possibility regions by thresholding
	vector<Point> getLocs(const Mat& src)
	{
		/// threshold
		double maxValue = 0.7;
		vector<Point> locs;
		
		for (int j = 0; j < src.rows; j++)
			for (int i = 0; i < src.cols; i++)
			{
				float v = src.at<float>(Point(i,j));
				if (v > maxValue)
					locs.push_back(Point(i, j));
			}
		
		return locs;
	}
	
	/// Obtain the highest possibility region by using the fusion weight
	Rect getMaxloc(const Mat& weight, const vector<Point>& locs, const Rect& prev_roi, const Mat& tmplate, Mat& target)
	{
		/// predict the possible roi & its center
		int x1 = prev_roi.x + cvRound(prev_roi.width * 0.5);
		int y1 = prev_roi.y + cvRound(prev_roi.height * 0.5);

		vector<double> v1(locs.size()), v2(locs.size()), v3(locs.size()), v4(locs.size()), w(locs.size());
		/// looping
		for (int i = 0; i < locs.size(); i++)
		{
			Point loc = locs.at(i);
			/// v1 is obtained from the template matching; if the value is higher, it means that the region is like the compared one 
			v1.at(i) = weight.at<float>(loc);
			/// v2 is obtained from the SAD difference; if the value is lower, it means that the region is like the compared one 
			Rect roi = Rect(loc.x, loc.y, prev_roi.width, prev_roi.height);
			v2.at(i) = getSAD(tmplate, target(roi));
			// v3 is the distance weight; if the value is lower, it means that the region is closed to the compared one 
			int x2 = loc.x + cvRound(prev_roi.width * 0.46);
			int y2 = loc.y + cvRound(prev_roi.height * 0.5);
			v3.at(i) = getDist(Point(x1,y1), Point(x2,y2));
		}
		//normalize(v1, v1, 0, 1, NORM_MINMAX, -1, Mat());
		normalize(v2, v2, 0, 1, NORM_MINMAX, -1, Mat());
		normalize(v3, v3, 0, 1, NORM_MINMAX, -1, Mat());
		/// 
		for (int i = 0; i < locs.size(); i++)
		{
			/// total weight w is the fusion weight; v1 is the template matching weight;  v2 is the SAD weight;  v3 is the distance weight;
			v2.at(i) = 1 - v2.at(i);
			v3.at(i) = 1 - v3.at(i);
			w.at(i) = v1.at(i) + v2.at(i) + v3.at(i);
		}

		/// get the max weight and its position
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		minMaxLoc(w, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
		int id = maxLoc.x;
		//if (debug_flag)
		//	checkim(target, Rect(locs.at(id).x, locs.at(id).y, prev_roi.width, prev_roi.height));
		return Rect(locs.at(id).x, locs.at(id).y, prev_roi.width, prev_roi.height);
	}
	
	void checkim(Mat& im, const Rect& roi)
	{
		rectangle(im, roi, Scalar(0, 0, 0));
		namedWindow("Display window", WINDOW_AUTOSIZE);
		imshow("Display window", im);
		waitKey(0);
	}

	/// set search range of the target image (3 times larger than the template)
	Rect searchArea(const Rect& roi)
	{
		/// Down-sampling scale ratio is ok at scale ratio 5 
		double scale = 5;
		Rect new_roi = roi;
		int cx = cvRound(new_roi.x + new_roi.width * 0.5);
		int cy = cvRound(new_roi.y + new_roi.height * 0.5);
		int radius = cvRound(min(roi.width, roi.height) * 0.5);
		new_roi.x = cx - radius * cvRound(scale * 0.5);
		new_roi.y = cy - radius * cvRound(scale * 0.5);
		new_roi.width = cvRound(scale * radius);
		new_roi.height = cvRound(scale * radius);
		return new_roi;
	}

	void boundaryCheck(Rect& roi, const Mat& target)
	{
		roi.x = max(0, roi.x);
		roi.x = min(roi.x, target.cols - 1);
		roi.y = max(0, roi.y);
		roi.y = min(roi.y, target.rows - 1);
		int w = roi.x + roi.width;
		w = max(0, w);
		w = min(w, target.cols);
		roi.width = w - roi.x;
		int h = roi.y + roi.height;
		h = max(0, h);
		h = min(h, target.rows);
		roi.height = h - roi.y;
	}
	
	// Template Matching 
	Rect TMatch(const Mat& target, const Mat& tmplate, const Rect prev_roi)
	{
		/// set search range of the target image (3 times larger than the template)
		Rect new_roi = searchArea(prev_roi);
		
		/// boundary check
		boundaryCheck(new_roi, target);
		
		/// Extract the searching image using mask
		Mat mask = cv::Mat::zeros(target.rows, target.cols, CV_8U);
		mask(new_roi) = 255;
		Mat new_search;
		target.copyTo(new_search,mask);

		if (debug_flag)
			checkim(new_search,prev_roi);

		/// Do the Matching and Normalize
		Mat weight;
		matchTemplate(new_search, tmplate, weight, match_method);
		normalize(weight, weight, 0, 1, NORM_MINMAX, -1, Mat());
		if (match_method == 0 || match_method == 1)
			weight = 1 - weight;

		/// find the possible regions
		vector<Point>locs = getLocs(weight);

		/// return the highest possibility region
		return getMaxloc(weight, locs, prev_roi, tmplate, new_search);
	}

	/// rectangle re-sampling: 0:up-sampling; 1:down-sampling
	void rectResample(Rect& roi, const Point& ratio, const bool& resample_flag)
	{
		/// down-sampling
		if (resample_flag)
		{
			roi.x /= ratio.x;
			roi.y /= ratio.y;
			roi.width /= ratio.x;
			roi.height /= ratio.y;
		}
		/// up-sampling
		else
		{
			roi.x *= ratio.x;
			roi.y *= ratio.y;
			roi.width *= ratio.x;
			roi.height *= ratio.y;
		}
	}

	/// image re-sampling: 0:up-sampling; 1:down-sampling
	void imResample(Mat& image, const Point& ratio, const bool& resample_flag)
	{
		/// down-sampling or up-sampling
		if (resample_flag)
			cv::resize(image, image, Size(image.cols / ratio.x, image.rows / ratio.y));
			//pyrDown(image, image, Size(image.cols / ratio.x, image.rows / ratio.y));
		else
			cv::resize(image, image, Size(image.cols * ratio.x, image.rows * ratio.y));
			//pyrUp(image, image, Size(image.cols * ratio.x, image.rows * ratio.y));
	}

	/// calculating the image re-sampling ratio
	Point resampleRatio(const Mat& source)
	{
		/// re-sampling ratio
		Point ratio;
		ratio.y = source.rows / imsize.y;
		ratio.x = source.cols / imsize.x;
		resampling = (ratio.x == 1 && ratio.y == 1) ? false : true;
		return ratio;
	}

	/// Run the algorithm
	bool runObjectTrackingAlgorithm (const cv::Mat& source, std::map<int, cv::Rect>& objects)
	{
		objects.clear();

		frame_id++;

		/// image initialization
		Mat image = source.clone();

		/// return the image re-sampling ratio
		scale_ratio = resampleRatio(source);
		
		/// check re-sampling is essential
		if (resampling)
			/// down-sampling (1) 
			imResample(image, scale_ratio, 1);

		/// The target image and its edge image
		Mat edges,gray_track_im;
		cvtColor(image, gray_track_im, CV_BGR2GRAY);
		convert2edge(gray_track_im, edges);
				
		
		auto itr = m_active_objects.begin();
		for ( ; itr != m_active_objects.end(); ++itr )
		{
			/// original template & its roi
			Mat tmplate = itr->second;
			cv::Rect roi = m_active_roi.at(itr->first);

			/// down-sampling
			if (resampling)
			{
				imResample(tmplate, scale_ratio, 1);
				rectResample(roi, scale_ratio, 1);
			}
			
			/// Create gray-template and its edge image
			Mat gray_tmplate;
			cvtColor(tmplate, gray_tmplate, CV_BGR2GRAY);
			Mat edged;
			convert2edge(gray_tmplate, edged);

			/// Previous frame ROI position & down-sampling
			cv::Rect prev_roi = m_active_prev_roi.at(itr->first);
			if (resampling)
				rectResample(prev_roi, scale_ratio, 1);

			//debug_flag = (frame_id == 300 && itr->first == 2) ? true : false;

			/// Template matching 2 find the most similar region; m1/m2: the original target andits edge image  
			Rect m1rec = TMatch(image, tmplate, prev_roi);
			Rect m2rec = TMatch(edges, edged, prev_roi);

			/// Using the Gaussian weight of the distance 2 compare the regions
			double s1 = getLocWeight(prev_roi, m1rec);
			double s2 = getLocWeight(prev_roi, m2rec);
			
			/// comparison: the higher value means the region is closed to the previous frame's roi
			Rect match_roi;
			if (s1 < 0 && s2 < 0)
			{
				/// if the template region is missing in the current frame, the missing frame number adds 1
				m_active_missing.at(itr->first) += 1;

				/// If the number of frame's missing template is > 4, do rematch
				/// else use the moving vector obtained from the previous frame and the previous 2 frame 2 predict the region in the current frame  
				Rect adj_roi = (m_active_missing.at(itr->first) > 4) ? TMatch(image, tmplate, prev_roi) : getLinepts(prev_roi, m_active_pts_vector.at(itr->first));

				/// boundary check
				adj_roi.x = max(0, adj_roi.x);
				adj_roi.x = min(adj_roi.x, image.cols - 1);
				adj_roi.y = max(0, adj_roi.y);
				adj_roi.y = min(adj_roi.y, image.rows - 1);

				/// update match_roi
				match_roi = adj_roi;
			}
			else
				match_roi = max(s1, s2) ? m1rec : m2rec;

			/// Up-sampling
			if (resampling)
				rectResample(match_roi, scale_ratio, 0);

			/// if the template region is found, set the missing frame number 2 zero 
			if (s1 > 0 || s2 > 0)
				m_active_missing.at(itr->first) = 0;
			
			/// draw + save tracking region
			objects.insert(std::make_pair(itr->first, match_roi));
			
			/// Update Parameter: update the roi 
			m_active_prev_roi.at(itr->first) = match_roi;
			
			/// update the moving vector from the current frame and the previous frame 
			m_active_pts_vector.at(itr->first) = getLinevec(prev_roi, match_roi);
 
		}
		return (objects.size() > 0);
	}

private:

	/// image size definition
	Point imsize;

	/// image re-sampling ratio
	Point scale_ratio;

	/// image re-sampling flag
	bool resampling;

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

	/// debug
	int frame_id;
	bool debug_flag;
};


#endif
