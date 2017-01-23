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
#include <numeric>
#include <functional>

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
		imsize = Point(80, 60);
		frame_id = 92;
		debug_flag = false;
		scale_ratio = Point(1, 1);
	}

	/// add a tracking obj
	virtual int addTrackingObject(const cv::Mat& source, const cv::Rect& obj_roi)
	{
		
		int obj_id = AbstractTracker::addTrackingObject(source, obj_roi);
		
		if (obj_id == 0) srcpHash = getpHash(source);

		if (obj_id >= 0)
		{
			m_active_roi.insert(std::make_pair(obj_id, obj_roi));
			m_active_prev_roi.insert(std::make_pair(obj_id, obj_roi));
			m_active_pts_vector.insert(std::make_pair(obj_id, Point(0,0)));
			m_active_missing.insert(std::make_pair(obj_id, 0));
			m_active_prev_tmplate.insert(std::make_pair(obj_id, source(obj_roi)));
			m_active_tmplate_hist.insert(std::make_pair(obj_id, getHist(source(obj_roi))));
			//m_active_tmplate_prev_hist.insert(std::make_pair(obj_id, getHist(source(obj_roi))));
			m_active_tmplate_colorange.insert(std::make_pair(obj_id, getColorange(source, obj_roi)));
			vector<double> initTh(1);
			initTh.at(0) = 0;
			m_active_tmplate_cth.insert(std::make_pair(obj_id, initTh));
			m_active_tmplate_sth.insert(std::make_pair(obj_id, initTh));
		}

		return obj_id;
	}


	Mat getHist(const Mat& tmplate)
	{
		Mat tmphsv, tmpmask, tmphue, tmphist;
		int hsize = 16;
		float hranges[] = { 0, 180 };
		int ch[] = { 0, 0 };
		const float* phranges = hranges;
		/// tmplate HSV
		cvtColor(tmplate, tmphsv, CV_BGR2HSV);
		inRange(tmphsv, Scalar(0, 60, 32), Scalar(180, 255, 255), tmpmask);
		Mat cha[3];
		split(tmphsv, cha);
		calcHist(&cha[0], 1, 0, tmpmask, tmphist, 1, &hsize, &phranges);
		normalize(tmphist, tmphist, 0, 255, NORM_MINMAX);
		return tmphist;
	}

	Rect camshiftArea(const Mat& target, const Mat& tmphist, const Rect& prev_roi, const vector<double>& colorange)
	{
		Mat tarColor, tar, tarmask, backproj;
		int r = 5;
		Rect selection = Rect(prev_roi.x - r, prev_roi.y - r, prev_roi.width + r, prev_roi.height + r) & Rect(0, 0, target.cols, target.rows);;
		int hsize = 16;
		float hranges[] = { 0, 180};
		int ch[] = { 0, 0 };
		const float* phranges = hranges;
		
		/// camshift
		cvtColor(target, tarColor, CV_BGR2HSV);
		double m1 = (colorange.at(3) - colorange.at(2)) / r;
		double m2 = (colorange.at(5) - colorange.at(4)) / r;
		inRange(tarColor, Scalar(0, max(0.0, colorange.at(2) - m1), max(0.0, colorange.at(4) - m2)), Scalar(180, min(colorange.at(3) - m1, 255.0), min(colorange.at(5) - m2, 255.0)), tarmask);
		//inRange(tarColor, Scalar(0, 60, 32), Scalar(180, 255, 255), tarmask);
		tar = extrROI(tarColor, tarmask);
		//imgshow(target, tarmask);
		Mat chs[3];
		split(tar, chs);
		calcBackProject(&chs[0], 1, 0, tmphist, backproj, &phranges);
		RotatedRect trackBox = CamShift(backproj, selection, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
		//if (debug_flag) { Mat tt = target.clone();  ellipse(tt, trackBox, Scalar(255, 0, 0)); imgshow(tt); }
		selection.x -= r;
		selection.y -= r;
		selection.width += 4*r;
		selection.height += 4*r;
		selection &= Rect(0, 0, target.cols, target.rows);
		return selection;
	}

	/// calculate Perceptual hash algorithm (pHash)
	string getpHash(const Mat &src)
	{
		Mat img, dst;
		string rst(64, '\0');
		double dIdex[64];
		double mean = 0.0;
		int k = 0;
		if (src.channels() == 3)
		{
			cvtColor(src, img, CV_BGR2GRAY);
			img = Mat_<double>(img);
		}
		else
		{
			img = src.clone();
			img = Mat_<double>(src);
		}

		/* resize */
		resize(img, img, Size(8, 8));

		/* dct*/
		dct(img, dst);

		/* the left-top 8 x 8 corner */
		for (int i = 0; i < 8; ++i) {
			for (int j = 0; j < 8; ++j)
			{
				dIdex[k] = dst.at<double>(i, j);
				mean += dst.at<double>(i, j) / 64;
				++k;
			}
		}

		/* hash value¡C*/
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
			cv::resize(image, image, Size(cvRound(image.cols / ratio.x), cvRound(image.rows / ratio.y)));
		else
			cv::resize(image, image, Size(cvRound(image.cols * ratio.x), cvRound(image.rows * ratio.y)));
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

	/// calculate the distance between 2 regions
	float getDist(const Rect &p, const Rect &q)
	{
		Point p1 = Point(p.x + cvRound(p.width >> 1), p.y + cvRound(p.height >> 1));
		Point p2 = Point(q.x + cvRound(q.width >> 1), q.y + cvRound(q.height >> 1));
		return getDist(p1, p2);
	}

	/// calculate the distance between 2 points
	float getDist(const Point &p, const Point &q)
	{
		Point diff = p - q;
		return cv::sqrt(float(diff.x * diff.x + diff.y * diff.y));
	}

	/// get the color difference by using CIE Luv color space
	double getColordiff(const Mat& a, const Mat& b)
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
	double getSAD(const Mat& source, const Mat& target)
	{
		Scalar x = cv::sum(cv::abs(source - target));
		return (source.channels() == 1 || source.channels() == 1) ? x.val[0] : (x.val[0] + x.val[1] + x.val[2]) / 3;
	}

	/// convert the image 2 its edge image
	Mat convert2edge(const Mat& source)
	{
		/// initialization
		int edge_flag = 3;
		int th1 = 100, th2 = 200;
		Mat gx, absgx, gy, absgy, dst;
		Mat destination;

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
		return destination;
	}

	/// Obtain the possibility regions by thresholding
	vector<Point> getLocs(const Mat& src)
	{
		/// threshold
		double maxValue = 0.75;
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
	
	/// get Cosine distance
	double getCosineDist(const Mat& src, const Mat& tar)
	{
		Mat sp, tp;
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
		return sp.dot(tp) / (norm(sp) * norm(tp));
	}

	/// Obtain the highest possibility region by using the fusion weight
	Rect getMaxloc(const Mat& weight, const vector<Point>& locs, const Rect& prev_roi, const Mat& tmplate, const Mat& target)
	{
		/// predict the possible roi & its center
		int x1 = prev_roi.x + cvRound(prev_roi.width >> 1);
		int y1 = prev_roi.y + cvRound(prev_roi.height >> 1);
		//
		String tmpHash = getpHash(tmplate);

		vector<double> v1(locs.size()), v2(locs.size()), v3(locs.size()), v4(locs.size()), v5(locs.size()), v6(locs.size()), w(locs.size());
		/// looping
		for (int i = 0; i < locs.size(); i++)
		{
			Point loc = locs.at(i);
			/// v1 is obtained from the template matching; if the value is higher, it means that the region is like the compared one 
			v1.at(i) = 1.0 - weight.at<float>(loc);
			/// v2 is obtained from the SAD difference; if the value is lower, it means that the region is like the compared one 
			Rect roi = Rect(loc.x, loc.y, prev_roi.width, prev_roi.height);
			v2.at(i) = getSAD(tmplate, target(roi));
			// v3 is the distance weight; if the value is lower, it means that the region is closed to the compared one 
			int x2 = loc.x + cvRound(prev_roi.width >> 1);
			int y2 = loc.y + cvRound(prev_roi.height >> 1);
			v3.at(i) = getDist(Point(x1, y1), Point(x2, y2));
			/// v4 is the color weight; if the image between the objects is similiar, the value is low
			v4.at(i) = getColordiff(tmplate, target(roi));
			/// v5 is the color weight; if the image between the objects is similiar, the value is high
			v5.at(i) = 1.0 - getCosineDist(tmplate, target(roi));
			// 
			String tpHash = getpHash(target(roi));
			v6.at(i) = getHammingDist(tmpHash, tpHash) / 64.0;
		}

		normalize(v2, v2, 0, 1, NORM_MINMAX, -1, Mat());
		normalize(v3, v3, 0, 1, NORM_MINMAX, -1, Mat());
		normalize(v4, v4, 0, 1, NORM_MINMAX, -1, Mat());

		/// looping
		for (int i = 0; i < locs.size(); i++)
		{	/// total weight w is the fusion weight; v1 is the template matching weight;  v2 is the SAD weight;  v3 is the distance weight;
			w.at(i) = v1.at(i) + v2.at(i) + v3.at(i) + v6.at(i);
			if(target.channels() > 1)
				w.at(i)+= (v4.at(i) + v5.at(i));
		}
		
		/// get the max weight and its position
		Point minLoc;
		minMaxLoc(w, NULL, NULL, &minLoc, NULL, Mat());
		int id = minLoc.x;
		return Rect(locs.at(id).x, locs.at(id).y, prev_roi.width, prev_roi.height);
	}


	/// Template Matching 
	Rect TMatch(const Mat& target, const Mat& tmplate, const Rect prev_roi)
	{ 
		Mat weight;
		/// Do the Matching and Normalize
		matchTemplate(target, tmplate, weight, match_method);
		normalize(weight, weight, 0, 1, NORM_MINMAX, -1, Mat());
		if (match_method == 0 || match_method == 1)
			weight = 1 - weight;

		/// find the possible regions
		vector<Point>locs = getLocs(weight);

		/// return the highest possibility region
		return getMaxloc(weight, locs, prev_roi, tmplate, target);
	}

	//get Hamming Distance
	int getHammingDist(string &str1, string &str2)
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

	double getHistcomp(const Mat& src, const Mat& tar)
	{
		/// Using 50 bins for hue and 60 for saturation
		int bins = 256; 
		int histSize[] = { bins};

		// hue varies from 0 to 179, saturation from 0 to 255
		float range[] = { 0, 256 };

		const float* ranges[] = { range};

		// Use the o-th and 1-st channels
		int ch0[] = { 0 };
		int ch1[] = { 1 };
		int ch2[] = { 2 };

		/// Histograms
		Mat hist_sa, hist_sb, hist_sc;
		Mat hist_ta, hist_tb, hist_tc;
		Mat hsv_base = src.clone();
		Mat hsv_test = tar.clone();


		/// Calculate the histograms for the HSV images
		calcHist(&hsv_base, 1, ch0, Mat(), hist_sa, 1, histSize, ranges, true, false);
		normalize(hist_sa, hist_sa, 0, 1, NORM_MINMAX, -1, Mat());
		calcHist(&hsv_base, 1, ch1, Mat(), hist_sb, 1, histSize, ranges, true, false);
		normalize(hist_sb, hist_sb, 0, 1, NORM_MINMAX, -1, Mat());
		calcHist(&hsv_base, 1, ch2, Mat(), hist_sc, 1, histSize, ranges, true, false);
		normalize(hist_sc, hist_sc, 0, 1, NORM_MINMAX, -1, Mat());

		calcHist(&hsv_test, 1, ch0, Mat(), hist_ta, 1, histSize, ranges, true, false);
		normalize(hist_ta, hist_ta, 0, 1, NORM_MINMAX, -1, Mat());
		calcHist(&hsv_test, 1, ch1, Mat(), hist_tb, 1, histSize, ranges, true, false);
		normalize(hist_tb, hist_tb, 0, 1, NORM_MINMAX, -1, Mat());
		calcHist(&hsv_test, 1, ch2, Mat(), hist_tc, 1, histSize, ranges, true, false);
		normalize(hist_tc, hist_tc, 0, 1, NORM_MINMAX, -1, Mat());

		double v0 = 1.0 - compareHist(hist_sa, hist_ta, 3);
		double v1 = 1.0 - compareHist(hist_sb, hist_tb, 3);
		double v2 = 1.0 - compareHist(hist_sc, hist_tc, 3);
		double v=max(max(v0, v1), v2);
		double vv = (v0 + v1 + v2) / 3;
		return (v > vv) ? v : vv;
	}

	vector<double> getColorange(const Mat& source, const Rect& roi)
	{
		Rect rec;
		int r = 13;
		rec.x = roi.x + r;
		rec.y = roi.y + r;
		rec.width = roi.width - (r);
		rec.height = roi.height - (r);
		Mat tmp = source(rec);
		Mat tmpColor, ch[3], tmpmask;
		//tmpColor = bgr2chromaticity(tmp);
		cvtColor(tmp, tmpColor, CV_BGR2HSV);
		split(tmpColor, ch);
		double minh = 0, maxh = 0, mins = 0, maxs = 0, minv = 0, maxv = 0;
		minMaxLoc(ch[0], &minh, &maxh);
		minMaxLoc(ch[1], &mins, &maxs);
		minMaxLoc(ch[2], &minv, &maxv);
		vector<double> colorange;
		colorange.push_back(minh);
		colorange.push_back(maxh);
		colorange.push_back(mins);
		colorange.push_back(maxs);
		colorange.push_back(minv);
		colorange.push_back(maxv);
		return colorange;
	}

	Mat bgr2chromaticity(const Mat& src)
	{
		Mat tmp(src.size(), src.type());
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

	bool boundCheck(const Rect& roi, const Mat& image)
	{
		int br = 2;
		int left = br, right = br + image.cols - 2 * br, up = br, down = br + image.rows - 2 * br;
		return ((roi.x < left || roi.y < up) || (roi.x < left || roi.y + roi.height > down) || (roi.x + roi.width > right || roi.y < left) || (roi.x + roi.width > right || roi.y + roi.height > down)) ? false : true;
	}

	bool compColor(const Mat& prev_tmplate, const Mat& tmplate, const Mat& image, const Rect& m1rec, const Rect& m2rec, Rect& match_roi, const vector<double>& colorange, vector<double>& cth, const Rect& prev_roi, const Mat& tmphist, const int& id, vector<double>& sth)
	{
		Mat tmpColor, prevtmpColor, m1Color, m2Color, m1, m2;
		tmpColor = tmplate.clone();
		prevtmpColor = prev_tmplate.clone();
		m1 = image(m1rec);
		m2 = image(m2rec);
		tmpColor = bgr2chromaticity(tmpColor);
		m1Color = bgr2chromaticity(m1);
		m2Color = bgr2chromaticity(m2);
		if (debug_flag) { imgshow(tmpColor); imgshow(m1Color); imgshow(m2Color); }

		double s1 = getSAD(tmpColor, m1Color) * getSAD(tmplate, m1);
		double s2 = getSAD(tmpColor, m2Color) * getSAD(tmplate, m2);
		double c1 = 1.0 - (tmpColor.dot(m1Color) / (norm(tmpColor) * norm(m1Color))) * (tmplate.dot(m1) / (norm(tmplate) * norm(m1)));
		double c2 = 1.0 - (tmpColor.dot(m2Color) / (norm(tmpColor) * norm(m2Color))) * (tmplate.dot(m2) / (norm(tmplate) * norm(m2)));

		double  ss = 0, cc = 0;
		if ((s1 <= s2 && c1 <= c2) || (s1 > s2 && c1 <= c2))
		{
			match_roi = m1rec;
			ss = s1;
			cc = c1;
		}
		else if ((s1 > s2 && c1 > c2) || (s1 <= s2 && c1 > c2))
		{
			match_roi = m2rec;
			ss = s2;
			cc = c2;
		}

		Scalar cthmean, cthstd, sthmean, sthstd;
		meanStdDev(sth, sthmean, sthstd);
		double stmean = (double)sthmean.val[0];
		double ststd = (double)sthstd.val[0];
		meanStdDev(cth, cthmean, cthstd);
		double ctmean = (double)cthmean.val[0];
		double ctstd = (double)cthstd.val[0];
		
		bool pflag = boundCheck(match_roi, image);
		bool cflag = true, sflag = true;
		double clb = (ctmean - ctstd) / 1.5;
		double cub = 2 * (ctmean + ctstd);
		double slb = (stmean - ststd) / 1.5;
		double sub = 2 * (stmean + ststd);

		if(cth.size()>3)
			cflag = (cc <= cub && cc >= clb) ? true : false;
		if (cflag)
		{
			if (cth.at(0) == 0)
				cth.at(0) = cc;
			else
				cth.push_back(cc);
		}
		if (sth.size() > 3)
			sflag = (ss <= sub && ss >= slb) ? true : false;
		if (sflag)
		{
			if (sth.at(0) == 0)
				sth.at(0) = ss;
			else
				sth.push_back(ss);
		}
		bool trflag = (pflag && cflag && sflag) ? true : false;
 		return trflag;


	}

	/// Run the algorithm
	bool runObjectTrackingAlgorithm (const cv::Mat& target, std::map<int, cv::Rect>& objects)
	{
		objects.clear();
	
		frame_id++;

		///
		String tarpHash = getpHash(target);
		imflag = ((1.0 - getHammingDist(srcpHash, tarpHash) / 64.0) > 0.5) ? true : false;
		

		/// image initialization; target: original image size; image: resized image
		Mat image = target.clone();

		/// return the image re-sampling ratio
		scale_ratio = resampleRatio(target);
		
		/// check re-sampling is essential
		if (resampling)
			/// down-sampling (1) 
			imResample(image, scale_ratio, 1);

		/// The target image and its edge image
		Mat gray_track_im;
		cvtColor(image, gray_track_im, CV_BGR2GRAY);
		Mat edges = convert2edge(gray_track_im);
		
		auto itr = m_active_objects.begin();
		for ( ; itr != m_active_objects.end(); ++itr )
		{
			cout<<"frame "<< frame_id<<" obj "<<itr->first<<endl;

			/// original template & its roi
			Mat tmplate = itr->second;
			/// Previous frame ROI position & down-sampling
			cv::Rect prev_roi = m_active_prev_roi.at(itr->first);
			Mat prev_tmplate = m_active_prev_tmplate.at(itr->first);
			Mat tmphist= m_active_tmplate_hist.at(itr->first);
			vector<double> colorange = m_active_tmplate_colorange.at(itr->first);
			vector<double> cth = m_active_tmplate_cth.at(itr->first);
			vector<double> sth = m_active_tmplate_sth.at(itr->first);

			/// down-sampling
			if (resampling)
			{
				imResample(tmplate, scale_ratio, 1);
				rectResample(prev_roi, scale_ratio, 1);
				imResample(prev_tmplate, scale_ratio, 1);
			}

			/// Create gray-template and its edge image
			Mat gray_tmplate, edged, prev_edged;
			cvtColor(tmplate, gray_tmplate, CV_BGR2GRAY);
			edged = convert2edge(gray_tmplate);
			
			cvtColor(prev_tmplate, gray_tmplate, CV_BGR2GRAY);
			prev_edged = convert2edge(gray_tmplate);

			//debug_flag = (frame_id == 314 && itr->first == 0 || frame_id == 345 && itr->first == 0) ? true : false;
		
			/// camshift
			Rect selection = camshiftArea(image, tmphist, prev_roi, colorange);
			if(debug_flag) imgshow(image, selection);
			Mat im = extrROI(image, selection);
			Mat ed = extrROI(edges, selection);

			/// Template matching 2 find the most similar region; m1/m2: the original target andits edge image
			Rect m1rec = TMatch(im, tmplate, prev_roi);
			Rect m2rec = TMatch(ed, edged, prev_roi);
			
			Rect match_roi;
			bool trflag = compColor(prev_tmplate, tmplate, image, m1rec, m2rec, match_roi, colorange, cth, prev_roi, tmphist, itr->first, sth);
			if (resampling)
			{
				rectResample(match_roi, scale_ratio, 0);
				rectResample(prev_roi, scale_ratio, 0);
			}
			if (trflag) 
			{			
				/// draw + save tracking region
				objects.insert(std::make_pair(itr->first, match_roi));
			
				/// Update Parameter: update the roi 
				m_active_prev_roi.at(itr->first) = match_roi;
			
				///
				m_active_prev_tmplate.at(itr->first) = target(match_roi);

				/// 
				m_active_tmplate_colorange.at(itr->first) = getColorange(target, match_roi);

			}
			else
			{
				cout << "obj " << itr->first << " not draw" << endl;

				/// Update Parameter: restore the previous tmplate back 2 the original tmplate 
				m_active_prev_tmplate.at(itr->first) = m_active_objects.at(itr->first);
			}
			// 
			m_active_tmplate_cth.at(itr->first) = cth;
				
			/// 
			m_active_tmplate_sth.at(itr->first) = sth;
 
		}
		///
		if(imflag) 
			srcpHash = tarpHash;

		return (objects.size() > 0);
	}

	Mat extrROI(const Mat& img, const Rect& roi)
	{
		Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		mask(roi) = 255;
		Mat new_img;
		img.copyTo(new_img, mask);
		return new_img;
	}

	Mat extrROI(const Mat& img, const Rect& r1, const Rect& r2)
	{
		Mat mask = cv::Mat::zeros(img.rows, img.cols, CV_8U);
		mask(r1) = 255;
		mask(r2) = 255;
		Mat new_img;
		img.copyTo(new_img, mask);
		return new_img;
	}

	Mat extrROI(const Mat& img, const Mat& mask)
	{
		Mat new_img;
		img.copyTo(new_img, mask);
		return new_img;
	}



	/// Debug: imshow 
	void imgshow(const Mat& img)
	{
		namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", img);
		waitKey(0);
		cvDestroyAllWindows();
	}

	void imgshow(const Mat& img, const Rect& roi)
	{
		Mat new_img = extrROI(img, roi);
		namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", new_img);
		waitKey(0);
		cvDestroyAllWindows();
	}
	void imgshow(const Mat& img, const Mat& mask)
	{
		Mat tmp;
		img.copyTo(tmp, mask);
		namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", tmp);
		waitKey(0);
		cvDestroyAllWindows();
	}
	void imgshow(const Mat& img, const Mat& mask, const Rect& roi)
	{
		Mat tmp;
		img.copyTo(tmp, mask);
		namedWindow("Display window", WINDOW_NORMAL);
		imshow("Display window", tmp(roi));
		waitKey(0);
		cvDestroyAllWindows();
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

	String srcpHash;

	/// the template roi in the start frame 
	std::map<int, cv::Rect> m_active_roi;

	/// the template roi in the previous frame 
	std::map<int, cv::Rect> m_active_prev_roi;

	/// the template moving vector 
	std::map<int, cv::Point> m_active_pts_vector;

	/// the frequency of missing template 
	std::map<int, int> m_active_missing;

	/// the previous tmplate  
	std::map<int, Mat> m_active_prev_tmplate;

	/// the previous tmplate  
	std::map<int, Mat> m_active_tmplate_hist;

	/// the previous tmplate  
	std::map<int, Mat> m_active_tmplate_prev_hist;

	/// the previous tmplate  
	std::map<int, vector<double>> m_active_tmplate_colorange;

	/// the frequency of missing template 
	std::map<int, vector<double>> m_active_tmplate_sth;

	/// the frequency of missing template 
	std::map<int, vector<double>> m_active_tmplate_cth;


	///
	bool imflag;

	/// debug
	int frame_id;
	bool debug_flag;
};


#endif
