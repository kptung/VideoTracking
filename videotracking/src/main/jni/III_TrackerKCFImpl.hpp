#include <complex>
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
#include "featureColorName.hpp"
  /*---------------------------
  |  TrackerKCF
  |---------------------------*/
namespace cv {

	/*
	* Prototype
	*/
	class III_TrackerKCFImpl {
	public:
		enum MODE {
			GRAY = (1u << 0),
			CN = (1u << 1),
			CUSTOM = (1u << 2)
		};

		struct CV_EXPORTS Params
		{
			/**
			* \brief Constructor
			*/
			Params();

			/**
			* \brief Read parameters from file, currently unused
			*/
			void read(const FileNode& /*fn*/);

			/**
			* \brief Read parameters from file, currently unused
			*/
			void write(FileStorage& /*fs*/) const;

			double sigma;                 //!<  gaussian kernel bandwidth
			double lambda;                //!<  regularization
			double interp_factor;         //!<  linear interpolation factor for adaptation
			double output_sigma_factor;   //!<  spatial bandwidth (proportional to target)
			double pca_learning_rate;     //!<  compression learning rate
			bool resize;                  //!<  activate the resize feature to improve the processing speed
			bool split_coeff;             //!<  split the training coefficients into two matrices
			bool wrap_kernel;             //!<  wrap around the kernel values
			bool compress_feature;        //!<  activate the pca method to compress the features
			int max_patch_size;           //!<  threshold for the ROI size
			int compressed_size;          //!<  feature size after compression
			unsigned int desc_pca;        //!<  compressed descriptors of TrackerKCF::MODE
			unsigned int desc_npca;       //!<  non-compressed descriptors of TrackerKCF::MODE
		};

		struct CV_EXPORTS TrackerState
		{
			/**
			* \brief Constructor
			*/
			int parentId = -1;
			bool success = false;

		};

		III_TrackerKCFImpl(const Params &parameters = Params());
		void read(const FileNode& /*fn*/);
		void write(FileStorage& /*fs*/) const;
		void setFeatureExtractor(void(*f)(const Mat, const Rect, Mat&), bool pca_func = false);
		bool init(const Mat& image, const Rect2d& boundingBox);
		bool update(const Mat& image, Rect2d& boundingBox, bool override_roi);
		

	protected:
		/*
		* basic functions and vars
		*/
		bool initImpl(const Mat& image, const Rect2d& boundingBox);
		bool updateImpl(const Mat& image, Rect2d& boundingBox, bool override_roi);
		bool isInit;
		Params params;

		/*
		* KCF functions and vars
		*/
		void createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) const;
		void inline fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const;
		void inline fft2(const Mat src, Mat & dest) const;
		void inline ifft2(const Mat src, Mat & dest) const;
		void inline pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB = false) const;
		void inline sumChannels(std::vector<Mat> src, Mat & dest) const;
		void inline updateProjectionMatrix(const Mat src, Mat & old_cov, Mat &  proj_matrix, double pca_rate, int compressed_sz,
			std::vector<Mat> & layers_pca, std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat v) const;
		void inline compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const;
		bool getSubWindow(const Mat img, const Rect roi, Mat& feat, Mat& patch, MODE desc = GRAY) const;
		bool getSubWindow(const Mat img, const Rect roi, Mat& feat, void(*f)(const Mat, const Rect, Mat&)) const;
		void extractCN(Mat patch_data, Mat & cnFeatures) const;
		void denseGaussKernel(const double sigma, const Mat, const Mat y_data, Mat & k_data,
			std::vector<Mat> & layers_data, std::vector<Mat> & xf_data, std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf) const;
		void calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) const;
		void calcResponse(const Mat alphaf_data, const Mat alphaf_den_data, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) const;

		void shiftRows(Mat& mat) const;
		void shiftRows(Mat& mat, int n) const;
		void shiftCols(Mat& mat, int n) const;

	private:
		double output_sigma;
		Rect2d roi;
		Mat hann; 	//hann window filter
		Mat hann_cn; //10 dimensional hann-window filter for CN features,

		Mat y, yf; 	// training response and its FFT
		Mat x; 	// observation and its FFT
		Mat k, kf;	// dense gaussian kernel and its FFT
		Mat kf_lambda; // kf+lambda
		Mat new_alphaf, alphaf;	// training coefficients
		Mat new_alphaf_den, alphaf_den; // for splitted training coefficients
		Mat z; // model
		Mat response; // detection result
		Mat old_cov_mtx, proj_mtx; // for feature compression
		Mat prev_frame;
		Mat flow;
		Mat splitflow[2];

								   // pre-defined Mat variables for optimization of private functions
		Mat spec, spec2;
		std::vector<Mat> layers;
		std::vector<Mat> vxf, vyf, vxyf;
		Mat xy_data, xyf_data;
		Mat data_temp, compress_data;
		std::vector<Mat> layers_pca_data;
		std::vector<Scalar> average_data;
		Mat img_Patch;

		// storage for the extracted features, KRLS model, KRLS compressed model
		Mat X[2], Z[2], Zc[2];

		// storage of the extracted features
		std::vector<Mat> features_pca;
		std::vector<Mat> features_npca;
		std::vector<MODE> descriptors_pca;
		std::vector<MODE> descriptors_npca;

		// optimization variables for updateProjectionMatrix
		Mat data_pca, new_covar, w_data, u_data, vt_data;

		// custom feature extractor
		bool use_custom_extractor_pca;
		bool use_custom_extractor_npca;
		bool learning_flag;
		std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_pca;
		std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_npca;

		bool resizeImage; // resize the image whenever needed and the patch size is large
		int imageResizeFactor = 2;
		int templateResizeFactor = 2;
		int roi_rate_x;
		int roi_rate_y;
		int frame;
		int prev_max = -1;
	};

	III_TrackerKCFImpl::III_TrackerKCFImpl(const Params &parameters):params(parameters)
	{
		isInit = false;
		resizeImage = false;
		use_custom_extractor_pca = false;
		use_custom_extractor_npca = false;
		learning_flag = true;
	}

	void III_TrackerKCFImpl::read(const cv::FileNode& fn) {
		params.read(fn);
	}

	void III_TrackerKCFImpl::write(cv::FileStorage& fs) const {
		params.write(fs);
	}

	/*
	* Initialization:
	* - creating hann window filter
	* - ROI padding
	* - creating a gaussian response for the training ground-truth
	* - perform FFT to the gaussian response
	*/
	bool III_TrackerKCFImpl::initImpl(const Mat& image, const Rect2d& boundingBox) {
		frame = 0;
		roi = boundingBox;

		//calclulate output sigma
		output_sigma = sqrt(roi.width*roi.height)*params.output_sigma_factor;
		output_sigma = -0.5 / (output_sigma*output_sigma);

		//resize the ROI whenever needed
		if (params.resize && roi.width*roi.height>params.max_patch_size) {
			resizeImage = true;
			int shortImageSide = image.rows > image.cols ? image.cols : image.rows;
			for (int i = 0;;i++) {
				if (((double)image.rows / pow(2, i)) < 180) {
					imageResizeFactor = pow(2, i);
					printf("resizeFactor = %d\n", imageResizeFactor);
					break;
				}
			}
			roi.x = floor(roi.x /imageResizeFactor);
			roi.y  = floor(roi.y / imageResizeFactor);
			roi.width  = floor(roi.width / imageResizeFactor);
			roi.height  = floor(roi.height / imageResizeFactor);
		}

		// add padding to the roi
		/*roi.x -= roi.width / 2;
		roi.y -= roi.height / 2;
		roi.width *= 2;
		roi.height *= 2;*/

		roi_rate_x = image.cols / roi.width - 2;
		roi_rate_y = image.rows / roi.height - 2;
		if (roi_rate_x > 5)roi_rate_x = 5;
		if (roi_rate_x < 2)roi_rate_x = 2;
		if (roi_rate_y > 5)roi_rate_y = 5;
		if (roi_rate_y < 2)roi_rate_y = 2;
		roi.x -= (roi_rate_x -1) * roi.width/2;
		roi.y -= (roi_rate_y -1) * roi.height/2;
		roi.width *= roi_rate_x;
		roi.height *= roi_rate_y;

		// initialize the hann window filter
		createHanningWindow(hann, roi.size(), CV_64F);

		int shortTemplateSide = hann.rows > hann.cols ? hann.cols : hann.rows;
		for (int i = 0;; i++) {
			if (((double)shortTemplateSide / pow(2, i)) < 70) {
				templateResizeFactor = pow(2, i);
				printf("templateResizeFactor = %d\n", templateResizeFactor);
				break;
			}
		}

		resize(hann, hann, Size(hann.cols / templateResizeFactor, hann.rows / templateResizeFactor));

		// hann window filter for CN feature
		Mat _layer[] = { hann, hann, hann, hann, hann, hann, hann, hann, hann, hann };
		merge(_layer, 10, hann_cn);

		// create gaussian response
		y = Mat::zeros((int)floor(roi.height), (int)floor(roi.width), CV_64F);
		for (unsigned i = 0; i<floor(roi.height); i++) {
			for (unsigned j = 0; j<floor(roi.width); j++) {
				y.at<double>(i, j) = (i - roi.height / 2 + 1)*(i - roi.height / 2 + 1) + (j - roi.width / 2 + 1)*(j - roi.width / 2 + 1);//a^2+b^2
			}
		}

		y *= (double)output_sigma;
		cv::exp(y, y);

		// perform fourier transfor to the gaussian response
		resize(y, y, Size(y.cols / templateResizeFactor, y.rows / templateResizeFactor));
		fft2(y, yf);
		// record the non-compressed descriptors
		if ((params.desc_npca & GRAY) == GRAY)descriptors_npca.push_back(GRAY);
		if ((params.desc_npca & CN) == CN)descriptors_npca.push_back(CN);
		if (use_custom_extractor_npca)descriptors_npca.push_back(CUSTOM);
		features_npca.resize(descriptors_npca.size());

		// record the compressed descriptors
		if ((params.desc_pca & GRAY) == GRAY)descriptors_pca.push_back(GRAY);
		if ((params.desc_pca & CN) == CN)descriptors_pca.push_back(CN);
		if (use_custom_extractor_pca)descriptors_pca.push_back(CUSTOM);
		features_pca.resize(descriptors_pca.size());

		// accept only the available descriptor modes
		CV_Assert(
			(params.desc_pca & GRAY) == GRAY
			|| (params.desc_npca & GRAY) == GRAY
			|| (params.desc_pca & CN) == CN
			|| (params.desc_npca & CN) == CN
			|| use_custom_extractor_pca
			|| use_custom_extractor_npca
		);

		// TODO: return true only if roi inside the image
		return true;
	}

	/*
	* Main part of the KCF algorithm
	*/
	bool III_TrackerKCFImpl::updateImpl(const Mat& image, Rect2d& boundingBox, bool override_roi) {
		double minVal, maxVal;	// min-max response
		Point minLoc, maxLoc;	// min-max location
		Mat img = image.clone();
		// check the channels of the input image, grayscale is preferred
		CV_Assert(img.channels() == 1 || img.channels() == 3);
		Rect2d roi2 = roi;
		roi2.x = round(roi.x);
		roi2.y = round(roi.y);
		roi2.width = round(roi.width);
		roi2.height = round(roi.height);

		// resize the image whenever needed
		if (resizeImage)resize(img.clone(), img, Size(img.cols / imageResizeFactor, img.rows / imageResizeFactor));

		if (override_roi) {
			
			roi.x = floor(boundingBox.x / imageResizeFactor);
			roi.y = floor(boundingBox.y / imageResizeFactor);
			roi.x -= (roi_rate_x - 1) * roi.width/ roi_rate_x / 2;
			roi.y -= (roi_rate_y - 1) * roi.height/ roi_rate_y / 2;
		}

		learning_flag = true;
		// detection part
		if (frame > 0) {

			// return false if roi is outside the image
			if ((roi.x + roi.width < 0)
				|| (roi.y + roi.height < 0)
				|| (roi.x >= img.cols)
				|| (roi.y >= img.rows)
				) {
				learning_flag = false;
			};

			// extract patch inside the image
			if (roi.x<0 || roi.y<0 || (roi.x + roi.width > img.cols) || (roi.y + roi.height > img.rows)
				|| roi.width>img.cols || roi.height>img.rows) {
				learning_flag = false;
			}

			// extract and pre-process the patch
			// get non compressed descriptors
			for (unsigned i = 0; i < descriptors_npca.size() - extractor_npca.size(); i++) {
				if (!getSubWindow(img, roi, features_npca[i], img_Patch, descriptors_npca[i]))return false;
			}
			//get non-compressed custom descriptors
			for (unsigned i = 0, j = (unsigned)(descriptors_npca.size() - extractor_npca.size()); i < extractor_npca.size(); i++, j++) {
				if (!getSubWindow(img, roi, features_npca[j], extractor_npca[i]))return false;
			}
			if (features_npca.size() > 0)merge(features_npca, X[1]);

			// get compressed descriptors
			for (unsigned i = 0; i < descriptors_pca.size() - extractor_pca.size(); i++) {
				if (!getSubWindow(img, roi, features_pca[i], img_Patch, descriptors_pca[i]))return false;
			}
			//get compressed custom descriptors
			for (unsigned i = 0, j = (unsigned)(descriptors_pca.size() - extractor_pca.size()); i < extractor_pca.size(); i++, j++) {
				if (!getSubWindow(img, roi, features_pca[j], extractor_pca[i]))return false;
			}
			if (features_pca.size() > 0)merge(features_pca, X[0]);

			//compress the features and the KRSL model
			if (params.desc_pca != 0) {
				compress(proj_mtx, X[0], X[0], data_temp, compress_data);
				compress(proj_mtx, Z[0], Zc[0], data_temp, compress_data);
			}

			// copy the compressed KRLS model
			Zc[1] = Z[1];

			// merge all features
			if (features_npca.size() == 0) {
				x = X[0];
				z = Zc[0];
			}
			else if (features_pca.size() == 0) {
				x = X[1];
				z = Z[1];
			}
			else {
				merge(X, 2, x);
				merge(Zc, 2, z);
			}
			/*imshow("dGK", k);
			waitKey(0);
			cvDestroyAllWindows();*/
			//compute the gaussian kernel
			denseGaussKernel(params.sigma, x, z, k, layers, vxf, vyf, vxyf, xy_data, xyf_data);
			/*imshow("dGK",k);
			waitKey(0);
			cvDestroyAllWindows();*/
			// compute the fourier transform of the kernel
			fft2(k, kf);
			if (frame == 1)spec2 = Mat_<Vec2d >(kf.rows, kf.cols);

			// calculate filter response
			if (params.split_coeff)
				calcResponse(alphaf, alphaf_den, kf, response, spec, spec2);
			else
				calcResponse(alphaf, kf, response, spec);

			// extract the maximum response
			minMaxLoc(response, &minVal, &maxVal, &minLoc, &maxLoc);
			if (prev_max == -1) {
				prev_max = maxVal;
			}
			else if (maxVal < (prev_max*0.5)) {
				if (override_roi) {
				return false;
				}
				learning_flag = false;
			}
			else {
				if (override_roi && (maxVal < (prev_max*0.9))) {
					return false;
				}
				prev_max = maxVal;

			}
			/*
			//imshow("before", img(roi));
			if (!prev_frame.empty()) {
				Ptr<DualTVL1OpticalFlow> tvl1 = cv::DualTVL1OpticalFlow::create();
				Mat split_prev_frame[3], split_img[3];
				split(prev_frame, split_prev_frame);
				split(img, split_img);
				tvl1->calc(split_prev_frame[1](roi2), split_img[1](roi2), flow);
				split(flow, splitflow);
				cv::Mat magnitude, angle;
				cv::cartToPolar(splitflow[0], splitflow[1], magnitude, angle, true);

				//translate magnitude to range [0;1]
				double mag_max;
				cv::minMaxLoc(magnitude, 0, &mag_max);
				magnitude.convertTo(magnitude, -1, 1.0 / mag_max);

				//build hsv image
				cv::Mat _hsv[3], hsv;
				_hsv[0] = angle;
				_hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
				_hsv[2] = magnitude;
				cv::merge(_hsv, 3, hsv);

				//convert to BGR and show
				cv::Mat bgr;//CV_32FC3 matrix
				cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
				imshow("flow", bgr);
				imshow("flowx", splitflow[0]);
				imshow("flowy", splitflow[1]);
				waitKey(0);
			}
			*/
			prev_frame = img.clone();
			Rect2d temproi;
			double offsetx = (maxLoc.x * (double)roi.width / (double)response.cols - roi.width / 2 + 1);
			double offsety = (maxLoc.y * (double)roi.height / (double)response.rows - roi.height / 2 + 1);
			temproi.x = roi.x + offsetx;
			temproi.y = roi.y + offsety;
			if ((temproi.x + roi.width / 2) < 0)return false;
			if ((temproi.y + roi.height / 2) < 0)return false;
			if ((temproi.x + roi.width / 2) > img.cols)return false;
			if ((temproi.y + roi.height / 2) > img.rows)return false;
			/*
			if (!flow.empty()) {
				Scalar scalar = mean(flow);
				if (scalar.val[0] * offsetx < 0 || scalar.val[1] * offsety < 0) {
					return false;
				}
			}*/
			roi.x = temproi.x;
			roi.y = temproi.y;
			//imshow("result", img(roi));
			//waitKey(0);
			// update the bounding box
			int temp_x = (resizeImage ? ceil(roi.x) * imageResizeFactor : roi.x) + (roi_rate_x - 1)*boundingBox.width / 2;
			int temp_y = (resizeImage ? ceil(roi.y) * imageResizeFactor : roi.y) + (roi_rate_y - 1)*boundingBox.height / 2;
			boundingBox.x = temp_x;
			boundingBox.y = temp_y;
			

			if (!learning_flag) {
				return true;
			}
		}

		// extract the patch for learning purpose
		// get non compressed descriptors
		for (unsigned i = 0; i<descriptors_npca.size() - extractor_npca.size(); i++) {
			if (!getSubWindow(img, roi, features_npca[i], img_Patch, descriptors_npca[i]))return false;
		}
		//get non-compressed custom descriptors
		for (unsigned i = 0, j = (unsigned)(descriptors_npca.size() - extractor_npca.size()); i<extractor_npca.size(); i++, j++) {
			if (!getSubWindow(img, roi, features_npca[j], extractor_npca[i]))return false;
		}
		if (features_npca.size()>0)merge(features_npca, X[1]);

		// get compressed descriptors
		for (unsigned i = 0; i<descriptors_pca.size() - extractor_pca.size(); i++) {
			if (!getSubWindow(img, roi, features_pca[i], img_Patch, descriptors_pca[i]))return false;
		}
		//get compressed custom descriptors
		for (unsigned i = 0, j = (unsigned)(descriptors_pca.size() - extractor_pca.size()); i<extractor_pca.size(); i++, j++) {
			if (!getSubWindow(img, roi, features_pca[j], extractor_pca[i]))return false;
		}
		if (features_pca.size()>0)merge(features_pca, X[0]);

		//update the training data
		if (frame == 0) {
			Z[0] = X[0].clone();
			Z[1] = X[1].clone();
		}
		else if(learning_flag){
			Z[0] = (1.0 - params.interp_factor)*Z[0] + params.interp_factor*X[0];
			Z[1] = (1.0 - params.interp_factor)*Z[1] + params.interp_factor*X[1];
		}

		if (params.desc_pca != 0 || use_custom_extractor_pca) {
			// initialize the vector of Mat variables
			if (frame == 0) {
				layers_pca_data.resize(Z[0].channels());
				average_data.resize(Z[0].channels());
			}

			// feature compression
			updateProjectionMatrix(Z[0], old_cov_mtx, proj_mtx, params.pca_learning_rate, params.compressed_size, layers_pca_data, average_data, data_pca, new_covar, w_data, u_data, vt_data);
			compress(proj_mtx, X[0], X[0], data_temp, compress_data);
		}

		// merge all features
		if (features_npca.size() == 0)
			x = X[0];
		else if (features_pca.size() == 0)
			x = X[1];
		else
			merge(X, 2, x);

		// initialize some required Mat variables
		if (frame == 0) {
			layers.resize(x.channels());
			vxf.resize(x.channels());
			vyf.resize(x.channels());
			vxyf.resize(vyf.size());
			new_alphaf = Mat_<Vec2d >(yf.rows, yf.cols);
		}

		// Kernel Regularized Least-Squares, calculate alphas
		denseGaussKernel(params.sigma, x, x, k, layers, vxf, vyf, vxyf, xy_data, xyf_data);

		// compute the fourier transform of the kernel and add a small value
		fft2(k, kf);
		kf_lambda = kf + params.lambda;

		double den;
		if (params.split_coeff) {
			mulSpectrums(yf, kf, new_alphaf, 0);
			mulSpectrums(kf, kf_lambda, new_alphaf_den, 0);
		}
		else {
			for (int i = 0; i<yf.rows; i++) {
				for (int j = 0; j<yf.cols; j++) {
					den = 1.0 / (kf_lambda.at<Vec2d>(i, j)[0] * kf_lambda.at<Vec2d>(i, j)[0] + kf_lambda.at<Vec2d>(i, j)[1] * kf_lambda.at<Vec2d>(i, j)[1]);

					new_alphaf.at<Vec2d>(i, j)[0] =
						(yf.at<Vec2d>(i, j)[0] * kf_lambda.at<Vec2d>(i, j)[0] + yf.at<Vec2d>(i, j)[1] * kf_lambda.at<Vec2d>(i, j)[1])*den;
					new_alphaf.at<Vec2d>(i, j)[1] =
						(yf.at<Vec2d>(i, j)[1] * kf_lambda.at<Vec2d>(i, j)[0] - yf.at<Vec2d>(i, j)[0] * kf_lambda.at<Vec2d>(i, j)[1])*den;
				}
			}
		}

		// update the RLS model
		if (frame == 0) {
			alphaf = new_alphaf.clone();
			if (params.split_coeff)alphaf_den = new_alphaf_den.clone();
		}
		else {
			alphaf = (1.0 - params.interp_factor)*alphaf + params.interp_factor*new_alphaf;
			if (params.split_coeff)alphaf_den = (1.0 - params.interp_factor)*alphaf_den + params.interp_factor*new_alphaf_den;
		}

		frame++;
		return true;
	}


	/*-------------------------------------
	|  implementation of the KCF functions
	|-------------------------------------*/

	/*
	* hann window filter
	*/
	void III_TrackerKCFImpl::createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) const {
		CV_Assert(type == CV_32FC1 || type == CV_64FC1);

		dest.create(winSize, type);
		Mat dst = dest.getMat();

		int rows = dst.rows, cols = dst.cols;

		AutoBuffer<double> _wc(cols);
		double * const wc = (double *)_wc;

		double coeff0 = 2.0 * CV_PI / (double)(cols - 1), coeff1 = 2.0f * CV_PI / (double)(rows - 1);
		for (int j = 0; j < cols; j++)
			wc[j] = 0.5 * (1.0 - cos(coeff0 * j));

		if (dst.depth() == CV_32F) {
			for (int i = 0; i < rows; i++) {
				float* dstData = dst.ptr<float>(i);
				double wr = 0.5 * (1.0 - cos(coeff1 * i));
				for (int j = 0; j < cols; j++)
					dstData[j] = (float)(wr * wc[j]);
			}
		}
		else {
			for (int i = 0; i < rows; i++) {
				double* dstData = dst.ptr<double>(i);
				double wr = 0.5 * (1.0 - cos(coeff1 * i));
				for (int j = 0; j < cols; j++)
					dstData[j] = wr * wc[j];
			}
		}

		// perform batch sqrt for SSE performance gains
		//cv::sqrt(dst, dst); //matlab do not use the square rooted version
	}

	/*
	* simplification of fourier transform function in opencv
	*/
	void inline III_TrackerKCFImpl::fft2(const Mat src, Mat & dest) const {
		dft(src, dest, DFT_COMPLEX_OUTPUT);
	}

	void inline III_TrackerKCFImpl::fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const {
		split(src, layers_data);

		for (int i = 0; i<src.channels(); i++) {
			dft(layers_data[i], dest[i], DFT_COMPLEX_OUTPUT);
		}
	}

	/*
	* simplification of inverse fourier transform function in opencv
	*/
	void inline III_TrackerKCFImpl::ifft2(const Mat src, Mat & dest) const {
		idft(src, dest, DFT_SCALE + DFT_REAL_OUTPUT);
	}

	/*
	* Point-wise multiplication of two Multichannel Mat data
	*/
	void inline III_TrackerKCFImpl::pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB) const {
		for (unsigned i = 0; i<src1.size(); i++) {
			mulSpectrums(src1[i], src2[i], dest[i], flags, conjB);
		}
	}

	/*
	* Combines all channels in a multi-channels Mat data into a single channel
	*/
	void inline III_TrackerKCFImpl::sumChannels(std::vector<Mat> src, Mat & dest) const {
		dest = src[0].clone();
		for (unsigned i = 1; i<src.size(); i++) {
			dest += src[i];
		}
	}

	/*
	* obtains the projection matrix using PCA
	*/
	void inline III_TrackerKCFImpl::updateProjectionMatrix(const Mat src, Mat & old_cov, Mat &  proj_matrix, double pca_rate, int compressed_sz,
		std::vector<Mat> & layers_pca, std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat vt) const {
		CV_Assert(compressed_sz <= src.channels());

		split(src, layers_pca);

		for (int i = 0; i<src.channels(); i++) {
			average[i] = mean(layers_pca[i]);
			layers_pca[i] -= average[i];
		}

		// calc covariance matrix
		merge(layers_pca, pca_data);
		pca_data = pca_data.reshape(1, src.rows*src.cols);

		new_cov = 1.0 / (double)(src.rows*src.cols - 1)*(pca_data.t()*pca_data);
		if (old_cov.rows == 0)old_cov = new_cov.clone();

		// calc PCA
		SVD::compute((1.0 - pca_rate)*old_cov + pca_rate*new_cov, w, u, vt);

		// extract the projection matrix
		proj_matrix = u(Rect(0, 0, compressed_sz, src.channels())).clone();
		Mat proj_vars = Mat::eye(compressed_sz, compressed_sz, proj_matrix.type());
		for (int i = 0; i<compressed_sz; i++) {
			proj_vars.at<double>(i, i) = w.at<double>(i);
		}

		// update the covariance matrix
		old_cov = (1.0 - pca_rate)*old_cov + pca_rate*proj_matrix*proj_vars*proj_matrix.t();
	}

	/*
	* compress the features
	*/
	void inline III_TrackerKCFImpl::compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const {
		data = src.reshape(1, src.rows*src.cols);
		compressed = data*proj_matrix;
		dest = compressed.reshape(proj_matrix.cols, src.rows).clone();
	}

	/*
	* obtain the patch and apply hann window filter to it
	*/
	bool III_TrackerKCFImpl::getSubWindow(const Mat img, const Rect _roi, Mat& feat, Mat& patch, MODE desc) const {

		Rect region = _roi;
		Rect _region = _roi;
		// return false if roi is outside the image
		if ((_roi.x + region.width<=0)
			|| (_roi.y + region.height<=0)
			|| (_roi.x >= img.cols)
			|| (_roi.y >= img.rows)
			)return false;

		// extract patch inside the image
		if (_roi.x<0) { region.x = 0; region.width += _roi.x; }
		if (_roi.y<0) { region.y = 0; region.height += _roi.y; }
		if (_roi.x + region.width>img.cols)region.width = img.cols - _roi.x;
		if (_roi.y + region.height>img.rows)region.height = img.rows - _roi.y;
		if (region.width>img.cols)region.width = img.cols;
		if (region.height>img.rows)region.height = img.rows;
		
		patch = img(region).clone();
		//imshow("patch",patch);
		//waitKey(0);
		resize(patch, patch, Size(ceil((double)patch.cols/ (double)templateResizeFactor), ceil((double)patch.rows/ (double)templateResizeFactor)));

		// add some padding to compensate when the patch is outside image border
		int addTop , addBottom, addLeft, addRight;
		bool addTopFlag, addBottomFlag, addLeftFlag, addRightFlag;
		addTopFlag = _region.y < 0;
		addBottomFlag = _roi.height + _roi.y > img.rows;
		addLeftFlag = _region.x < 0;
		addRightFlag = _roi.width + _roi.x > img.cols;
		if (addTopFlag || addBottomFlag) {
			if (addTopFlag&&addBottomFlag) {
				addTop = round((hann.rows - patch.rows)*(abs(_region.y)) / (abs(_region.y) + (_roi.height + _roi.y - img.rows)));
				addBottom = hann.rows - patch.rows - addTop;
			}
			else if (addTopFlag) {
				addTop = hann.rows - patch.rows;
				addBottom = 0;
			} else {
				addTop = 0;
				addBottom = hann.rows - patch.rows;
			}
		} else {
			addBottom = 0;
			addTop = 0;
		}

		if (addLeftFlag || addRightFlag) {
			if (addLeftFlag&&addRightFlag) {
				addLeft = round((hann.cols - patch.cols)*(abs(_region.x)) / (abs(_region.x) + (_roi.width + _roi.x - img.cols)));
				addRight = hann.cols - patch.cols - addLeft;
			}
			else if (addLeftFlag) {
				addLeft = hann.cols - patch.cols;
				addRight = 0;
			}
			else {
				addLeft = 0;
				addRight = hann.cols - patch.cols;
			}
		} else {
			addLeft = 0;
			addRight = 0;
		}

		copyMakeBorder(patch, patch, addTop, addBottom, addLeft, addRight, BORDER_REPLICATE);
		if (patch.rows == 0 || patch.cols == 0)return false;

		// extract the desired descriptors
		switch (desc) {
		case CN:
			CV_Assert(img.channels() == 3);
			extractCN(patch, feat);
			feat = feat.mul(hann_cn); // hann window filter
			break;
		default: // GRAY
			if (img.channels()>1)
				cvtColor(patch, feat, CV_BGR2GRAY);
			else
				feat = patch;
			feat.convertTo(feat, CV_64F);
			if (feat.size != hann.size) {
				printf("size are different");
			}
			feat = feat / 255.0 - 0.5; // normalize to range -0.5 .. 0.5
			feat = feat.mul(hann); // hann window filter
			break;
		}

		return true;

	}

	/*
	* get feature using external function
	*/
	bool III_TrackerKCFImpl::getSubWindow(const Mat img, const Rect _roi, Mat& feat, void(*f)(const Mat, const Rect, Mat&)) const {

		// return false if roi is outside the image
		if ((_roi.x + _roi.width<0)
			|| (_roi.y + _roi.height<0)
			|| (_roi.x >= img.cols)
			|| (_roi.y >= img.rows)
			)return false;

		f(img, _roi, feat);

		if (_roi.width != feat.cols || _roi.height != feat.rows) {
			printf("error in customized function of features extractor!\n");
			printf("Rules: roi.width==feat.cols && roi.height = feat.rows \n");
		}

		Mat hann_win;
		std::vector<Mat> _layers;

		for (int i = 0; i<feat.channels(); i++)
			_layers.push_back(hann);

		merge(_layers, hann_win);

		feat = feat.mul(hann_win); // hann window filter

		return true;
	}

	/* Convert BGR to ColorNames
	*/
	void III_TrackerKCFImpl::extractCN(Mat patch_data, Mat & cnFeatures) const {
		Vec3b & pixel = patch_data.at<Vec3b>(0, 0);
		unsigned index;

		if (cnFeatures.type() != CV_64FC(10))
			cnFeatures = Mat::zeros(patch_data.rows, patch_data.cols, CV_64FC(10));

		for (int i = 0; i<patch_data.rows; i++) {
			for (int j = 0; j<patch_data.cols; j++) {
				pixel = patch_data.at<Vec3b>(i, j);
				index = (unsigned)(floor((float)pixel[2] / 8) + 32 * floor((float)pixel[1] / 8) + 32 * 32 * floor((float)pixel[0] / 8));

				//copy the values
				for (int _k = 0; _k<10; _k++) {
					cnFeatures.at<Vec<double, 10> >(i, j)[_k] = ColorNames[index][_k];
				}
			}
		}

	}

	/*
	*  dense gauss kernel function
	*/
	void III_TrackerKCFImpl::denseGaussKernel(const double sigma, const Mat x_data, const Mat y_data, Mat & k_data,
		std::vector<Mat> & layers_data, std::vector<Mat> & xf_data, std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf) const {
		double normX, normY;

		fft2(x_data, xf_data, layers_data);
		fft2(y_data, yf_data, layers_data);

		normX = norm(x_data);
		normX *= normX;
		normY = norm(y_data);
		normY *= normY;
		if (abs(normY - normX) > (normY*0.4) &&
			(normX< normY)) {
		//printf("normX = %f\n", normX);
		//printf("normY = %f\n", normY);
		//system("pause");
	    }
		pixelWiseMult(xf_data, yf_data, xyf_v, 0, true);
		sumChannels(xyf_v, xyf);
		ifft2(xyf, xyf);

		if (params.wrap_kernel) {
			shiftRows(xyf, x_data.rows / 2);
			shiftCols(xyf, x_data.cols / 2);
		}

		//(xx + yy - 2 * xy) / numel(x)
		xy = (normX + normY - 2 * xyf) / (x_data.rows*x_data.cols*x_data.channels());

		// TODO: check wether we really need thresholding or not
		//threshold(xy,xy,0.0,0.0,THRESH_TOZERO);//max(0, (xx + yy - 2 * xy) / numel(x))
		for (int i = 0; i<xy.rows; i++) {
			for (int j = 0; j<xy.cols; j++) {
				if (xy.at<double>(i, j)<0.0)xy.at<double>(i, j) = 0.0;
			}
		}

		double sig = -1.0 / (sigma*sigma);
		xy = sig*xy;
		exp(xy, k_data);

	}

	/* CIRCULAR SHIFT Function
	* http://stackoverflow.com/questions/10420454/shift-like-matlab-function-rows-or-columns-of-a-matrix-in-opencv
	*/
	// circular shift one row from up to down
	void III_TrackerKCFImpl::shiftRows(Mat& mat) const {

		Mat temp;
		Mat m;
		int _k = (mat.rows - 1);
		mat.row(_k).copyTo(temp);
		for (; _k > 0; _k--) {
			m = mat.row(_k);
			mat.row(_k - 1).copyTo(m);
		}
		m = mat.row(0);
		temp.copyTo(m);

	}

	// circular shift n rows from up to down if n > 0, -n rows from down to up if n < 0
	void III_TrackerKCFImpl::shiftRows(Mat& mat, int n) const {
		if (n < 0) {
			n = -n;
			flip(mat, mat, 0);
			for (int _k = 0; _k < n; _k++) {
				shiftRows(mat);
			}
			flip(mat, mat, 0);
		}
		else {
			for (int _k = 0; _k < n; _k++) {
				shiftRows(mat);
			}
		}
	}

	//circular shift n columns from left to right if n > 0, -n columns from right to left if n < 0
	void III_TrackerKCFImpl::shiftCols(Mat& mat, int n) const {
		if (n < 0) {
			n = -n;
			flip(mat, mat, 1);
			transpose(mat, mat);
			shiftRows(mat, n);
			transpose(mat, mat);
			flip(mat, mat, 1);
		}
		else {
			transpose(mat, mat);
			shiftRows(mat, n);
			transpose(mat, mat);
		}
	}

	/*
	* calculate the detection response
	*/
	void III_TrackerKCFImpl::calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) const {
		//alpha f--> 2channels ; k --> 1 channel;
		mulSpectrums(alphaf_data, kf_data, spec_data, 0, false);
		ifft2(spec_data, response_data);
	}

	/*
	* calculate the detection response for splitted form
	*/
	void III_TrackerKCFImpl::calcResponse(const Mat alphaf_data, const Mat _alphaf_den, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) const {

		mulSpectrums(alphaf_data, kf_data, spec_data, 0, false);

		//z=(a+bi)/(c+di)=[(ac+bd)+i(bc-ad)]/(c^2+d^2)
		double den;
		for (int i = 0; i<kf_data.rows; i++) {
			for (int j = 0; j<kf_data.cols; j++) {
				den = 1.0 / (_alphaf_den.at<Vec2d>(i, j)[0] * _alphaf_den.at<Vec2d>(i, j)[0] + _alphaf_den.at<Vec2d>(i, j)[1] * _alphaf_den.at<Vec2d>(i, j)[1]);
				spec2_data.at<Vec2d>(i, j)[0] =
					(spec_data.at<Vec2d>(i, j)[0] * _alphaf_den.at<Vec2d>(i, j)[0] + spec_data.at<Vec2d>(i, j)[1] * _alphaf_den.at<Vec2d>(i, j)[1])*den;
				spec2_data.at<Vec2d>(i, j)[1] =
					(spec_data.at<Vec2d>(i, j)[1] * _alphaf_den.at<Vec2d>(i, j)[0] - spec_data.at<Vec2d>(i, j)[0] * _alphaf_den.at<Vec2d>(i, j)[1])*den;
			}
		}

		ifft2(spec2_data, response_data);
	}

	void III_TrackerKCFImpl::setFeatureExtractor(void(*f)(const Mat, const Rect, Mat&), bool pca_func) {
		if (pca_func) {
			extractor_pca.push_back(f);
			use_custom_extractor_pca = true;
		}
		else {
			extractor_npca.push_back(f);
			use_custom_extractor_npca = true;
		}
	}

	bool III_TrackerKCFImpl::init(const Mat& image, const Rect2d& boundingBox) {
		return initImpl(image, boundingBox);
	}

	bool III_TrackerKCFImpl::update(const Mat& image, Rect2d& boundingBox, bool ovverride_roi) {
		return updateImpl(image, boundingBox, ovverride_roi);
	}
	/*----------------------------------------------------------------------*/

	/*
	* Parameters
	*/
	III_TrackerKCFImpl::Params::Params() {
		sigma = 0.2;
		lambda = 0.01;
		interp_factor = 0.075;
		output_sigma_factor = 1.0 / 16.0;
		resize = true;
		max_patch_size = 10 * 10;
		split_coeff = true;
		wrap_kernel = false;
		desc_npca = GRAY;
		desc_pca = CN;

		//feature compression
		compress_feature = true;
		compressed_size = 2;
		pca_learning_rate = 0.15;

	}

	void III_TrackerKCFImpl::Params::read(const cv::FileNode& /*fn*/) {}

	void III_TrackerKCFImpl::Params::write(cv::FileStorage& /*fs*/) const {}

} /* namespace cv */


