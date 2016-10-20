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

using namespace cv;
using namespace std;

class TrackerTM : public AbstractTracker
{
public:

	TrackerTM() :  AbstractTracker()
	{
		m_frame_id = 224;
		match_method = 5;
	}

	float Dist(Rect &p, Rect &q)
	{
		Point p1 = Point(p.x + p.width / 2, p.y + p.height / 2);
		Point p2 = Point(q.x + q.width / 2, q.y + q.height / 2);
		Point diff = p1 - p2;
		return cv::sqrt(float(diff.x*diff.x + diff.y*diff.y));
	}
	
	void convert2edge(Mat source, Mat &destination)
	{
		int edge_flag = 3;
		int th1 = 100, th2 = 200;
		Mat gx, absgx, gy, absgy, dst;
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
		//
		
		
	}

	Scalar getMSSIM(const Mat& i1, const Mat& i2)
	{
		const double C1 = 6.5025, C2 = 58.5225;
		/***************************** INITS **********************************/
		int d = CV_32F;

		Mat I1, I2;
		i1.convertTo(I1, d);           // cannot calculate on one byte large values
		i2.convertTo(I2, d);

		Mat I2_2 = I2.mul(I2);        // I2^2
		Mat I1_2 = I1.mul(I1);        // I1^2
		Mat I1_I2 = I1.mul(I2);        // I1 * I2

									   /***********************PRELIMINARY COMPUTING ******************************/

		Mat mu1, mu2;   //
		GaussianBlur(I1, mu1, Size(11, 11), 1.5);
		GaussianBlur(I2, mu2, Size(11, 11), 1.5);

		Mat mu1_2 = mu1.mul(mu1);
		Mat mu2_2 = mu2.mul(mu2);
		Mat mu1_mu2 = mu1.mul(mu2);

		Mat sigma1_2, sigma2_2, sigma12;

		GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
		sigma1_2 -= mu1_2;

		GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
		sigma2_2 -= mu2_2;

		GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
		sigma12 -= mu1_mu2;

		///////////////////////////////// FORMULA ////////////////////////////////
		Mat t1, t2, t3;

		t1 = 2 * mu1_mu2 + C1;
		t2 = 2 * sigma12 + C2;
		t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

		t1 = mu1_2 + mu2_2 + C1;
		t2 = sigma1_2 + sigma2_2 + C2;
		t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

		Mat ssim_map;
		divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

		Scalar mssim = mean(ssim_map); // mssim = average of ssim map
		return mssim;
	}

	double getPSNR(const Mat& I1, const Mat& I2)
	{
		Mat s1;
		absdiff(I1, I2, s1);       // |I1 - I2|
		s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
		s1 = s1.mul(s1);           // |I1 - I2|^2

		Scalar s = sum(s1);         // sum elements per channel

		double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		if (sse <= 1e-10) // for small values return zero
			return 0;
		else
		{
			double  mse = sse / (double)(I1.channels() * I1.total());
			double psnr = 10.0*log10((255 * 255) / mse);
			return psnr;
		}
	}

	void maxLocs(const Mat& src, queue<Point>& dst)
	{
		//float maxValue = -1.0f * numeric_limits<float>::max();
		float maxValue = 0.95;
		float* srcData = reinterpret_cast<float*>(src.data);

		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				float v = srcData[i*src.cols + j];
				if ( v > maxValue)
				{
					dst.push(Point(j, i));
				}
			}
		}

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
		m_frame_id++;

		//
		int th1 = 100, th2 = 180;
		Mat edges,trackim;
		GaussianBlur(image, trackim, Size(3, 3), 0, 0);
		cvtColor(trackim, trackim, CV_BGR2GRAY);
		convert2edge(trackim, edges);
		//
		auto itr = m_active_objects.begin();
		for ( ; itr != m_active_objects.end(); ++itr )
		{
			// Previous frame ROI position
			cv::Rect roi= m_active_roi.at(itr->first);
			
			Mat tmplate = itr->second;
			GaussianBlur(tmplate, tmplate, Size(3, 3), 0, 0);
			cvtColor(tmplate, tmplate, CV_BGR2GRAY);
			Mat edged;
			convert2edge(tmplate, edged);

			//Create the result matrix
			Mat result, result2;
			int rc = trackim.cols - tmplate.cols + 1;
			int rr = trackim.rows - tmplate.rows + 1;
			result.create(rc, rr, CV_32FC1);
			result2.create(rc, rr, CV_32FC1);
			matchTemplate(trackim, tmplate, result, match_method);
			matchTemplate(edges, edged, result2, match_method);
			normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
			normalize(result2, result2, 0, 1, NORM_MINMAX, -1, Mat());
			
			if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
			{
				result = 1.0 - result;
				result2 = 1.0 - result2;
			}

			// get the top 10 maximums from the first match
			queue<Point> locs1,locs2;
			maxLocs(result, locs1);
			maxLocs(result2, locs2);
			Mat img = trackim;
			//double m1p = 1e-7,m2p=1e-7;
			double m1p = 100000, m2p = 100000;
			Rect m1rec,m2rec;
			while (!locs1.empty())
			{
				Point tmploc = locs1.front();
				Rect tmproi = Rect(tmploc, Point(tmploc.x + roi.width, tmploc.y + roi.height));
				Mat tmp = img(tmproi).clone();
				//double b=cvNorm(&tmplate.reshape(1), &tmp.reshape(1),CV_L2);
				double dist = Dist(tmproi, roi);
				if (dist < m1p)
				{
					m1p = dist;
					m1rec = tmproi;
				}
				locs1.pop();
			}
			while (!locs2.empty())
			{
				Point matchLoc = locs2.front();
				Rect tmproi = Rect(matchLoc, Point(matchLoc.x + roi.width, matchLoc.y + roi.height));
				double dist = Dist(tmproi, roi);
				if (dist < m2p)
				{
					m2p = dist;
					m2rec = tmproi;
				}
				locs2.pop();
			}
			Mat m1 = img(m1rec);
			Mat m2 = img(m2rec);
			double p1= getPSNR(tmplate, m1);
			double p2 = getPSNR(tmplate, m2);
			Rect matchROI;
			if (p1>p2)
			//if (dis1 < dis2)
				matchROI = m1rec;
			else
				matchROI = m2rec;

			objects.insert(std::make_pair(itr->first, matchROI));
			auto itr2 = m_active_roi.find(itr->first);
			itr2->second=matchROI;
			Mat match = image;
			match = match(matchROI);
			auto itr3 = m_active_objects.find(itr->first);
			itr3->second = match;
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
