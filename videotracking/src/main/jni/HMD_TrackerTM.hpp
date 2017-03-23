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

#include <jni.h>
#include <android/log.h>


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

using namespace cv;
using namespace std;



class TrackerTM : public AbstractTracker
{
public:

	TrackerTM() : AbstractTracker()
	{
		match_method = 5;
		resampling = true;
		//imsize = cv::Point(80, 60);
		imsizew = cv::Point(80, 60);
		imsizeh = cv::Point(60, 80);
		debug_flag = false;
		scale_ratio = cv::Point(1, 1);
		frame_id = 1;
	}

	/// add a tracking obj
	virtual int addTrackingObject(const cv::Mat& source, const cv::Rect& obj_roi)
	{

		int obj_id = AbstractTracker::addTrackingObject(source, obj_roi);

		if (obj_id == 0) srcpHash = getpHash(source);

		/// extract obj by graph-cut segmentation
		cv::Mat fg;
		cv::Mat tmp = extrObj(source, obj_roi, fg);

		if (obj_id >= 0)
		{
			m_active_objects.at(obj_id)= tmp;
			m_active_roi.insert(std::make_pair(obj_id, obj_roi));
			m_active_prev_roi.insert(std::make_pair(obj_id, obj_roi));
			m_active_prev_tmplate.insert(std::make_pair(obj_id, tmp));
			m_active_tmplate_hist.insert(std::make_pair(obj_id, getHist(tmp)));
			m_active_tmplate_colorange.insert(std::make_pair(obj_id, getColorange(fg, obj_roi)));
			m_active_tmplate_probth.insert(std::make_pair(obj_id, vector<double>(1, -1)));
			m_active_tmplate_miss.insert(std::make_pair(obj_id, 0));
		}

		return obj_id;
	}

	cv::Mat extrObj(const cv::Mat& source, const cv::Rect& obj_roi, cv::Mat& fg)
	{
		//
		fg = cv::Mat(source.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		cv::Mat result;
		cv::Mat bgModel, fgModel;
		Mat tmpp = source(obj_roi);
		cv::grabCut(source, result, obj_roi, bgModel, fgModel, 1, cv::GC_INIT_WITH_RECT);
		cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
		source.copyTo(fg, result);
		cv::Mat tmp = fg(obj_roi);
		return tmp;
	}

	cv::Mat getHist(const cv::Mat& tmplate)
	{
		cv::Mat tmphsv, tmpmask, tmphue, tmphist;
		int hsize = 32;
		float hranges[] = { 0, 180 };
		int ch[] = { 0, 0 };
		const float* phranges = hranges;
		/// tmplate HSV
		cvtColor(tmplate, tmphsv, CV_BGR2HSV);
		inRange(tmphsv, Scalar(0, 60, 32), Scalar(180, 255, 255), tmpmask);
		cv::Mat cha[3];
		split(tmphsv, cha);
		calcHist(&cha[0], 1, 0, tmpmask, tmphist, 1, &hsize, &phranges);
		normalize(tmphist, tmphist, 0, 255, NORM_MINMAX);
		return tmphist;
	}

	cv::Rect camshiftArea(const cv::Mat& target, const cv::Mat& tmphist, const cv::Rect& prev_roi, const std::vector<double>& colorange, cv::Mat& backproj)
	{
		cv::Mat tarColor, tar, tarmask;
		int r = 5;
		double t = 2.2;
		Rect selection = Rect(prev_roi.x - r, prev_roi.y - r, prev_roi.width + r, prev_roi.height + r) & Rect(0, 0, target.cols, target.rows);;
		float hranges[] = { 0, 180 };
		const float* phranges = hranges;
		/// camshift pre-processing
		cvtColor(target, tarColor, CV_BGR2HSV);
		double m1 = (colorange.at(3) - colorange.at(2)) / r;
		double m2 = (colorange.at(5) - colorange.at(4)) / r;
		inRange(tarColor, Scalar(0, (colorange.at(2) - m1), (colorange.at(4) - m2)), Scalar(180, (colorange.at(3) - m1), (colorange.at(5) - m2)), tarmask);
		tar = extrROI(tarColor, tarmask);
		cv::Mat chs[3];
		split(tar, chs);
		/// camshift
		calcBackProject(&chs[0], 1, 0, tmphist, backproj, &phranges);
		RotatedRect trackBox = CamShift(backproj, selection, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
		selection.x -= r;
		selection.y -= r;
		selection.width += (int) (t * r);
		selection.height += (int) (t * r);
		selection &= Rect(0, 0, target.cols, target.rows);
		return selection;
	}

	/// calculate Perceptual hash algorithm (pHash)
	std::string getpHash(const cv::Mat &src)
	{
		cv::Mat img, dst;
		std::string rst(64, '\0');
		double dIdex[64];
		double mean = 0.0;
		int k = 0;
		if (src.channels() == 3)
			cvtColor(src, img, CV_BGR2GRAY);
		else
			img = src.clone();
		img = Mat_<double>(img);

		/* resize */
		cv::resize(img, img, Size(8, 8));

		/* dct*/
		cv::dct(img, dst);

		/* the left-top 8 x 8 corner */
		for (int i = 0; i < 8; ++i) {
			for (int j = 0; j < 8; ++j)
			{
				dIdex[k] = dst.at<double>(i, j);
				mean += dst.at<double>(i, j) / 64;
				++k;
			}
		}

		/* hash value。*/
		for (int i = 0; i < 64; ++i)
		{
			if (dIdex[i] >= mean)
				rst[i] = '1';
			else
				rst[i] = '0';

		}
		return rst;
	}

	/// rectangle re-sampling: 0:up-sampling; 1:down-sampling
	void rectResample(cv::Rect& roi, const cv::Point& ratio, const bool& resample_flag)
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
	void imResample(cv::Mat& image, const cv::Point& ratio, const bool& resample_flag)
	{
		/// down-sampling or up-sampling
		if (resample_flag)
			cv::resize(image, image, Size(cvRound(image.cols / ratio.x), cvRound(image.rows / ratio.y)));
		else
			cv::resize(image, image, Size(cvRound(image.cols * ratio.x), cvRound(image.rows * ratio.y)));
	}

	/// calculating the image re-sampling ratio
	cv::Point resampleRatio(const cv::Mat& source)
	{
		cv::Point imsize;
		if (source.rows >= source.cols)
			imsize = imsizeh;
		else
			imsize = imsizew;
		/// re-sampling ratio
		cv::Point ratio;
		ratio.y = source.rows / imsize.y;
		ratio.x = source.cols / imsize.x;
		resampling = (ratio.x == 1 && ratio.y == 1) ? false : true;
		return ratio;
	}

	/// calculate the distance between 2 regions
	float getDist(const cv::Rect &p, const cv::Rect &q)
	{
		cv::Point p1 = cv::Point(p.x + cvRound(p.width >> 1), p.y + cvRound(p.height >> 1));
		cv::Point p2 = cv::Point(q.x + cvRound(q.width >> 1), q.y + cvRound(q.height >> 1));
		return getDist(p1, p2);
	}

	/// calculate the distance between 2 points
	float getDist(const cv::Point &p, const cv::Point &q)
	{
		cv::Point diff = p - q;
		return cv::sqrt(float(diff.x * diff.x + diff.y * diff.y));
	}

	/// get the color difference by using CIE Luv color space
	double getColordiff(const cv::Mat& a, const cv::Mat& b)
	{
		Mat m, n;
		if (a.channels() > 1 && b.channels() > 1)
		{
			cvtColor(a, m, CV_BGR2Luv);
			cvtColor(b, n, CV_BGR2Luv);
		}
		else
		{
			m = a.clone();
			n = b.clone();
		}
		return cv::norm(m - n);
	}

	/// obtain the sum of absolute difference (SAD) between 2 images 
	double getSAD(const cv::Mat& source, const cv::Mat& target)
	{
		cv::Scalar x = cv::sum(cv::abs(source - target));
		return (source.channels() == 1 || target.channels() == 1) ? x.val[0] : (x.val[0] + x.val[1] + x.val[2]) / 3;
	}

	/// convert the image 2 its edge image
	cv::Mat convert2edge(const cv::Mat& source)
	{
		/// initialization
		cv::Mat gx, absgx, gy, absgy, dst;
		cv::Mat destination;

		Sobel(source, gx, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(gx, absgx);
		Sobel(source, gy, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);
		convertScaleAbs(gy, absgy);
		addWeighted(absgx, 0.5, absgy, 0.5, 0, destination);

		return destination;
	}

	/// Obtain the possibility regions by thresholding
	std::vector<Point> getLocs(const cv::Mat& src)
	{
		/// threshold
		double maxValue = 0.7;
		double th = mean(src).val[0];
		std::vector<Point> locs;

		for (int j = 0; j < src.rows; j++)
			for (int i = 0; i < src.cols; i++)
			{
				float v = src.at<float>(Point(i, j));
				if (v > maxValue)
					locs.push_back(Point(i, j));
			}

		return locs;
	}

	/// get Cosine distance
	double getCosineDist(const cv::Mat& src, const cv::Mat& tar)
	{
		cv::Mat sp, tp;
		if (src.channels() == 3 || tar.channels() == 3)
		{
			cvtColor(src, sp, CV_BGR2Luv);
			cvtColor(tar, tp, CV_BGR2Luv);
		}
		else
		{
			sp = src.clone();
			tp = tar.clone();
		}
		/*return sum(sp.dot(tp)).val[0] / sqrt(sum(sp.dot(sp)).val[0]) * sqrt(sum(tp.dot(tp)).val[0]);*/
		double ab = sp.dot(tp);
		double aa = sp.dot(sp);
		double bb = tp.dot(tp);
		return ab / sqrt(aa*bb);
	}

	/// Obtain the highest possibility region by using the fusion weight
	cv::Rect getMaxloc(const cv::Mat& weight, const std::vector<Point>& locs, const cv::Rect& prev_roi, const cv::Mat& tmplate, const cv::Mat& target)
	{
		/// predict the possible roi & its center
		int x1 = prev_roi.x + cvRound(prev_roi.width >> 1);
		int y1 = prev_roi.y + cvRound(prev_roi.height >> 1);
		//
		std::string tmpHash = getpHash(tmplate);

		std::vector<double> v1(locs.size()), v2(locs.size()), v3(locs.size()), v4(locs.size()), v5(locs.size()), v6(locs.size()), w(locs.size());
		/// looping
		for (int i = 0; i < locs.size(); i++)
		{
			cv::Point loc = locs.at(i);
			/// v1 is obtained from the template matching; if the value is higher, it means that the region is like the compared one 
			v1.at(i) = 1.0 - weight.at<float>(loc);
			/// v2 is obtained from the SAD difference; if the value is lower, it means that the region is like the compared one 
			cv::Rect roi = cv::Rect(loc.x, loc.y, prev_roi.width, prev_roi.height);
			v2.at(i) = getSAD(tmplate, target(roi));
			// v3 is the distance weight; if the value is lower, it means that the region is closed to the compared one 
			int x2 = loc.x + cvRound(prev_roi.width >> 1);
			int y2 = loc.y + cvRound(prev_roi.height >> 1);
			v3.at(i) = getDist(cv::Point(x1, y1), cv::Point(x2, y2));
			/// v4 is the color weight(Luv); if the image between the objects is similiar, the value is low
			v4.at(i) = getColordiff(tmplate, target(roi));
			/// v5 is the color weight(Luv); if the image between the objects is similiar, the value is high
			v5.at(i) = 1.0 - getCosineDist(tmplate, target(roi));
			// v6 is the pHash value; if the image between the objects is similiar, the value is low
			std::string tpHash = getpHash(target(roi));
			v6.at(i) = getHammingDist(tmpHash, tpHash) / 64.0;
		}

		normalize(v2, v2, 0, 1, NORM_MINMAX, -1, cv::Mat());
		normalize(v3, v3, 0, 1, NORM_MINMAX, -1, cv::Mat());
		normalize(v4, v4, 0, 1, NORM_MINMAX, -1, cv::Mat());

		/// looping
		for (int i = 0; i < locs.size(); i++)
		{	/// total weight w is the fusion weight; v1 is the template matching weight;  v2 is the SAD weight;  v3 is the distance weight;
			w.at(i) = v1.at(i) + v2.at(i) + v5.at(i) + v6.at(i);
// 			if (target.channels() > 1)
// 				w.at(i) += ( v3.at(i) + v4.at(i)  );
		}

		/// get the max weight and its position
		cv::Point minLoc;
		minMaxLoc(w, NULL, NULL, &minLoc, NULL, cv::Mat());
		int id = minLoc.x;
		return cv::Rect(locs.at(id).x, locs.at(id).y, prev_roi.width, prev_roi.height);
	}

	cv::Mat extrWt(const cv::Mat& image, const cv::Mat& weight, const cv::Rect& prev_roi)
	{
		cv::Mat mask(image.size(), weight.type(), Scalar(0,0,0));
		for (int j = 0; j < weight.rows; j++)
			for (int i = 0; i < weight.cols; i++)
			{
				int q = j + prev_roi.height - 1;
				int p = i + prev_roi.width -1;
				mask.at<float>(Point(p, q))=weight.at<float>(Point(i, j));
			}
		return mask;

	}

	/// Template Matching 
	cv::Rect TMatch(const cv::Mat& target, const cv::Mat& tmplate, const cv::Rect prev_roi, const cv::Rect& selection, Mat& weight)
	{
		cv::Mat tar = target.clone(), tmp = tmplate.clone(), fg;
		//extrObj(target, selection, fg);
		int cols = target.cols - tmplate.cols + 1;
		int rows = target.rows - tmplate.rows + 1;
		weight = cv::Mat(rows, cols, CV_32FC1);
		/// Do the Matching and Normalize
		matchTemplate(extrROI(target, selection), tmp, weight, match_method);
		normalize(weight, weight, 0, 1, NORM_MINMAX, -1, Mat());
		if (match_method == 0 || match_method == 1)
			weight = 1 - weight;

		/// find the possible regions
		std::vector<Point>locs = getLocs(weight);

		/// return the highest possibility region
		return (locs.size() > 0) ? getMaxloc(weight, locs, prev_roi, tmp, tar) : prev_roi;
	}

	//get Hamming Distance
	int getHammingDist(std::string& str1, std::string& str2)
	{
		if ((str1.size() != 64) || (str2.size() != 64))
			return -1;
		int difference = 0;
		for (int i = 0; i < 64; i++)
		{
			if (str1[i] != str2[i])
				difference++;
		}
		return difference;
	}

	/// set search range of the target image (3.1 times larger than the template)
	cv::Rect searchArea(const Rect& roi)
	{
		/// Down-sampling scale ratio is ok at scale ratio 5 
		double range = 6.2;
		Rect new_roi = roi;
		int cx = new_roi.x + cvRound(new_roi.width >> 1);
		int cy = new_roi.y + cvRound(new_roi.height >> 1);
		int radius = cvRound(min(roi.width, roi.height) >> 1);
		new_roi.x = cx - radius * cvRound(range * 0.5);
		new_roi.y = cy - radius * cvRound(range * 0.5);
		new_roi.width = cvRound(range * radius);
		new_roi.height = cvRound(range * radius);
		return new_roi;
	}

	/// get image color range by using HSV
	std::vector<double> getColorange(const cv::Mat& source, const cv::Rect& roi)
	{
		/// narrow the rect. 
		cv::Rect rec;
		int r = min(roi.width, roi.height)>>1;
		rec.x = roi.x + r;
		rec.y = roi.y + r;
		rec.width = roi.width - r;
		rec.height = roi.height - r;
		cv::Mat tmp = source(rec);
		cv::Mat tmpColor, ch[3], tmpmask;
		cvtColor(tmp, tmpColor, CV_BGR2HSV);
		split(tmpColor, ch);
		double minh = 0, maxh = 0, mins = 0, maxs = 0, minv = 0, maxv = 0;
		minMaxLoc(ch[0], &minh, &maxh);
		minMaxLoc(ch[1], &mins, &maxs);
		minMaxLoc(ch[2], &minv, &maxv);
		std::vector<double> colorange;
		colorange.push_back(minh);
		colorange.push_back(maxh);
		colorange.push_back(mins);
		colorange.push_back(maxs);
		colorange.push_back(minv);
		colorange.push_back(maxv);
		return colorange;
	}

	/// convert bgr image 2 normalized bgr 2 enhance the image objs  
	cv::Mat bgr2normalized(const cv::Mat& src)
	{
		cv::Mat tmp(src.size(), src.type());
		for (int i = 0; i < src.rows; i++)
			for (int j = 0; j < src.cols; j++)
			{
				Vec3f intensity = src.at<Vec3b>(i, j);
				double sum = intensity.val[0] + intensity.val[1] + intensity.val[2];
				double b = intensity.val[0] / sum;
				double g = intensity.val[1] / sum;
				double r = intensity.val[2] / sum;
				tmp.data[tmp.step[0] * i + tmp.step[1] * j + 0] = (uchar)(b * 255);
				tmp.data[tmp.step[0] * i + tmp.step[1] * j + 1] = (uchar)(g * 255);
				tmp.data[tmp.step[0] * i + tmp.step[1] * j + 2] = (uchar)(r * 255);
			}
		return tmp;
	}

	/// check the prediction is inside the image cols & rows; if the rectangle is near the image boundary, it will disappear
	bool boundCheck(const cv::Rect& roi, const cv::Mat& image)
	{
		int br = 2;
		Rect tmprec(br, br, image.cols - 2 * br, image.rows - 2 * br);
		Mat tt, kk = Mat::ones(image.size(), CV_8UC1);
		kk = kk * 255;
		Mat newbound = extrROI(kk, tmprec);
		Mat tmp = extrROI(kk, roi);
		bitwise_not(newbound, newbound);
		bitwise_and(tmp, newbound, tt);
		return (countNonZero(tt) > 0) ? false : true;
	}

	// return contour area
	int contourCheck(const cv::Mat& src)
	{
		cv::Mat gray, edge;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		cvtColor(src, gray, CV_BGR2GRAY);
		edge = convert2edge(gray);
		/// Find contours
		findContours(edge, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		Mat drawing = Mat::zeros(edge.size(), CV_8UC1);
		for (int i = 0; i < contours.size(); i++)
		{
			Scalar color = Scalar(255, 255, 255);
			drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
		}
		bitwise_not(drawing, drawing);
		return countNonZero(drawing);
	}

	double getAvgWt(const cv::Mat& weight, const cv::Rect& rect)
	{
		int r = 2;
		cv::Mat patch = weight(cv::Rect(rect.x - r, rect.y - r, rect.width + r, rect.height + r));
		return mean(patch).val[0];
	}

	double histCompare(const cv::Mat& src, const cv::Mat& tar)
	{
		cv::Mat src_hsv, tar_hsv;
		cvtColor(src, src_hsv, COLOR_BGR2HSV);
		cvtColor(tar, tar_hsv, COLOR_BGR2HSV);

		/// Using 50 bins for hue and 60 for saturation
		int h_bins = 30; int s_bins = 32;
		int histSize[] = { h_bins, s_bins };

		// hue varies from 0 to 179, saturation from 0 to 255
		float h_ranges[] = { 0, 180 };
		float s_ranges[] = { 0, 256 };

		const float* ranges[] = { h_ranges, s_ranges };

		// Use the o-th and 1-st channels
		int channels[] = { 0, 1 };

		/// Histograms
		MatND hist_src;
		MatND hist_tar;

		/// Calculate the histograms for the HSV images
		calcHist(&src_hsv, 1, channels, Mat(), hist_src, 2, histSize, ranges, true, false);
		normalize(hist_src, hist_src, 0, 1, NORM_MINMAX, -1, Mat());
		calcHist(&tar_hsv, 1, channels, Mat(), hist_tar, 2, histSize, ranges, true, false);
		normalize(hist_tar, hist_tar, 0, 1, NORM_MINMAX, -1, Mat());
		return compareHist(hist_src, hist_tar, CV_COMP_BHATTACHARYYA);

	}
	

	bool comProb(const cv::Mat& tmplate, const cv::Mat& image, const cv::Rect& m1rec, cv::Rect& match_roi, std::vector<double>& probth, const cv::Mat& backproj, const cv::Mat& wt, const cv::Rect& selection, const cv::Rect& prev_roi, const int& miss)
	{
		match_roi = m1rec;
		Mat fg;
		cv::Mat match = extrObj(image, match_roi, fg);

		cv::Mat tmp = bgr2normalized(tmplate);
		match = bgr2normalized(match);
		cv::Mat ws;
		matchTemplate(match, tmp, ws, match_method);
		double th1 = 1.0 - ws.at<float>(Point(0, 0));
		double th2 = histCompare(tmp, match);

		double th = th1;
		bool pbflag = false;
		if (probth.at(0) == -1)
		{
			pbflag = true;
			probth.at(0) = th;
		}
		Scalar thmean, thstd;
		meanStdDev(probth, thmean, thstd);
		bool thflag = (th <= thmean.val[0]*4) ? true : false;
		if (!pbflag && thflag)
			probth.push_back(th);

		/// pflag: position flag; 
		bool pflag = boundCheck(match_roi, image);

		/// cflag: contour flag
		int tmparea = contourCheck(tmplate);
		int matcharea = contourCheck(match);
		bool cflag = (matcharea >= (tmparea >> 1)) ? true : false;
		
		/// tracking flag (Draw or not)
		return (pflag && cflag && thflag) ? true : false;

	}

	/// the Gaussian weight of the distance between two regions
	double getLocWeight(const Rect& r1, const Rect& r2)
	{
		/// initialization
		Point diff = Point(r1.x + cvRound(r1.width >> 1), r1.y + cvRound(r1.height >> 1)) - Point(r2.x + cvRound(r2.width >> 1), r2.y + cvRound(r2.height >> 1));

		/// define the searching range
		int sigma = cvRound((min(r1.width, r1.height)) >> 1);

		/// calculate the distance between 2 points
		double dist = getDist(Point(r1.x + cvRound(r1.width >> 1), r1.y + cvRound(r1.height >> 1)), Point(r2.x + cvRound(r2.width >> 1), r2.y + cvRound(r2.height >> 1)));

		return (dist <= (3 * sigma)) ? (1 / sqrt(2 * CV_PI * sigma * sigma)) * exp(double(-(diff.x*diff.x + diff.y*diff.y) / (2 * sigma * sigma))) : -1;
	}

	/// Run the algorithm
	bool runObjectTrackingAlgorithm(const cv::Mat& target, std::map<int, cv::Rect>& objects)
	{
		objects.clear();

		frame_id++;

		/// tracking image pHash value
		std::string tarpHash = getpHash(target);
		imflag = ((1.0 - getHammingDist(srcpHash, tarpHash) / 64.0) > 0.5) ? true : false;


		/// image initialization; target: original image size; image: resized image
		cv::Mat image = target.clone();

		/// return the image re-sampling ratio
		scale_ratio = resampleRatio(target);

		/// check re-sampling is essential
		if (resampling)
			/// down-sampling (1) 
			imResample(image, scale_ratio, 1);

		/// The target image and its edge image
		cv::Mat gray_track_im;
		cvtColor(image, gray_track_im, CV_BGR2GRAY);
		cv::Mat edges = convert2edge(gray_track_im);

		auto itr = m_active_objects.begin();
		for (; itr != m_active_objects.end(); ++itr)
		{
			cout << "frame " << frame_id << " obj " << itr->first << endl;

			/// original template & its roi
			cv::Mat tmplate = itr->second;
			/// Previous frame ROI position & down-sampling
			cv::Rect prev_roi = m_active_prev_roi.at(itr->first);
			cv::Mat prev_tmplate = m_active_prev_tmplate.at(itr->first);
			/// Color 
			cv::Mat tmphist = m_active_tmplate_hist.at(itr->first);
			std::vector<double> colorange = m_active_tmplate_colorange.at(itr->first);
			std::vector<double> probth = m_active_tmplate_probth.at(itr->first);
			int miss = m_active_tmplate_miss.at(itr->first);

			/// down-sampling
			if (resampling)
			{
				imResample(tmplate, scale_ratio, 1);
				rectResample(prev_roi, scale_ratio, 1);
				imResample(prev_tmplate, scale_ratio, 1);
			}

			/// searching area
			cv::Rect distsel, camsel, selection;
			/// camshift
			cv::Mat backproj;
			/// set search range of the target image (3 times larger than the template)
			distsel = searchArea(prev_roi) & cv::Rect(0, 0, image.cols, image.rows);
			//set search range of the target image(using camshift)
			camsel = camshiftArea(image, tmphist, prev_roi, colorange, backproj);
			///
			if (miss < 2)
				selection = distsel&camsel;
			else
				selection = camsel;
			
			/// debug
			//debug_flag = (frame_id == 74 && itr->first == 0) ? true : false;
			if (debug_flag) imgshow(image, selection);
			cv::Mat im = extrROI(image, selection);
			Mat wt, ewt;
			/// Template matching 2 find the most similar region; m1/m2: the original target andits edge image
			cv::Rect m1rec = TMatch(image, tmplate, prev_roi, selection, wt);

			cv::Rect match_roi;
			bool trflag;
		
			trflag = comProb(tmplate, image, m1rec, match_roi, probth, backproj, wt, selection, prev_roi, miss);

			if (resampling)
			{
				rectResample(match_roi, scale_ratio, 0);
				rectResample(prev_roi, scale_ratio, 0);
			}
			//if (trflag || trflag == false && m_active_tmplate_miss.at(itr->first) < 2)
			if (trflag)
			{
				/// draw + save tracking region
				objects.insert(std::make_pair(itr->first, match_roi));

				/// Update Parameter: update the roi 
				m_active_prev_roi.at(itr->first) = match_roi;

				///  Update Parameter: update the template 
				m_active_prev_tmplate.at(itr->first) = target(match_roi);

				/// Update Parameter: update the color range
				m_active_tmplate_colorange.at(itr->first) = getColorange(target, match_roi);

				/// 
				if(trflag)
					m_active_tmplate_miss.at(itr->first) = 0;
				else
					m_active_tmplate_miss.at(itr->first) += 1;

			}
			else
			{
				cout << "obj " << itr->first << " not draw" << endl;

				/// Update Parameter: restore the previous tmplate back 2 the original tmplate 
				m_active_prev_tmplate.at(itr->first) = m_active_objects.at(itr->first);

				/// Update Parameter: restore the previous tmplate back 2 the original tmplate 
				m_active_tmplate_miss.at(itr->first) += 1;
			}

			/// the probability between a pair image  
			m_active_tmplate_probth.at(itr->first) = probth;

			/// the image similarity(SAD) between a pair image  
			//m_active_tmplate_sth.at(itr->first) = sth;

		}
		/// Update the pHash value
		if (imflag)
			srcpHash = tarpHash;

		return (objects.size() > 0);
	}

	cv::Mat extrROI(const cv::Mat& img, const cv::Rect& roi)
	{
		cv::Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		mask(roi) = 255;
		cv::Mat new_img;
		//cv::Mat new_img(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		img.copyTo(new_img, mask);
		return new_img;
	}

	cv::Mat extrROI(const cv::Mat& img, const cv::Mat& mask)
	{
		//cv::Mat new_img(img.size(), CV_8UC3, cv::Scalar(255, 255, 255));
		cv::Mat new_img;
		img.copyTo(new_img, mask);
		return new_img;
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
		cv::Mat new_img = extrROI(img, roi);
		namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", new_img);
		waitKey(0);
		cvDestroyAllWindows();
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

	/// image re-sampling ratio
	Point scale_ratio;

	/// image re-sampling flag
	bool resampling;

	/// template matching method: 0/1: SQDIFF/Normalized-SQDIFF; 2/3: TM_CCORR/Normalized-TM_CCORR; 4/5: CCOEFF/Normalized-CCOEFF; 
	int match_method;

	/// a source's pHash value 
	std::string srcpHash;

	/// the template roi in the start frame 
	std::map<int, cv::Rect> m_active_roi;

	/// the template roi in the previous frame 
	std::map<int, cv::Rect> m_active_prev_roi;

	/// the previous tmplate  
	std::map<int, cv::Mat> m_active_prev_tmplate;

	/// the tmplate color histogram image  
	std::map<int, cv::Mat> m_active_tmplate_hist;

	/// the tmplate color range and it will update in every frame  
	std::map<int, std::vector<double>> m_active_tmplate_colorange;

	/// the threshold between a pair image 
	std::map<int, std::vector<double>> m_active_tmplate_probth;

	/// the threshold between a pair image 
	std::map<int, int> m_active_tmplate_miss;

	///
	bool imflag;

	/// debug
	int frame_id;
	bool debug_flag;
};


#endif
