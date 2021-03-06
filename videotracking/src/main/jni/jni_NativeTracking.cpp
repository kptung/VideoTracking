#include <jni_NativeTracking.h>
#include <stdio.h>
#include <stdlib.h>     /* NULL */
#include <assert.h>     /* assert */
#include <vector>
#include <opencv2/opencv.hpp>
#include <android/log.h>
#include "timer.h"

#include "HMD_AbstractTracker.hpp"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>

#define LOG_NDEBUG 1
#define LOG_TAG "JNI_NativeTracking"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

#define JNI_DBG 0
#define VIDEO_TRACKING_LIB_VERSION 0.1
#define MIN_RECT_VALUE 14

#ifndef JPG
#define JPG (std::string(".jpg"))
#endif

using namespace std;
using namespace cv;

template <typename T>
std::string ToString(const T& value)
{
	std::ostringstream stream;
	stream << value;
	return stream.str();
}

//// //// ************ //// ////
//// ////   DBG_INFO   //// ////
//// //// ************ //// ////
#ifdef ANDROID
Timer timer;
// For debug Image
static int debugid = 1;
bool writeDBGInfo(Rect rect, int opt) {
	const std::string &filename = "/sdcard/TrackingDebug/TrackingRect.txt";
	FILE *fp = fopen(filename.c_str(), "a");
	if (fp)
	{
		switch (opt) {
		case 0:
			fprintf(fp, "INIT ID: %d , Rect: %d %d %d %d , %lld ms\r\n", debugid, rect.x, rect.y, rect.width, rect.height, timer.get_count());
			break;
		case 1:
			fprintf(fp, "ADD ID: %d , Rect: %d %d %d %d , %lld ms\r\n", debugid, rect.x, rect.y, rect.width, rect.height, timer.get_count());
			break;
		case 2:
			fprintf(fp, "RUN ID: %d , Rect: %d %d %d %d , %lld ms\r\n", debugid, rect.x, rect.y, rect.width, rect.height, timer.get_count());
			break;
		}
		fclose(fp);
		return true;
	}
	return false;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

	static int imgHeight, imgWidth = 0;
	static map<int, Mat> trackingObjects = map<int, Mat>();

	bool operator ! (const Mat&m) { return m.empty(); }

	enum FrameFormat {
		FRAME_NV21 = 1,
		FRAME_YUV420P = 2,
		FRAME_GRAY = 3,
	};

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    createHandle
	* Signature: ()J
	*/
	JNIEXPORT jlong JNICALL Java_org_iii_snsi_videotracking_NativeTracking_createHandle
	(JNIEnv *env, jobject jNativeTracking) {
		trackingObjects.clear();
		if (JNI_DBG)
			LOGD("CreateVideoTracker");
		/********** tracking method ***********/
		// 0: TM //color
		// 1: KCF by jimchen transfered on OpenCV //color
		// 2: CMT //gray
		// 3: DAT //color
		// 5: CSK // gray
		// 6: SKCF // color
		return (jlong)CreateVideoTracker(2);
		/***********************************/
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    initTrackingObjects
	* Signature: (J[BII[I)[I
	*/

	JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_initTrackingObjectsJPG
	(JNIEnv *env, jobject jNativeTracking, jlong jhandle, jbyteArray jimage, jint jsize, jintArray jrects) {

		jbyte* frame = env->GetByteArrayElements(jimage, 0);
		Mat rawData = Mat(1, jsize, CV_8UC1, (uchar *)frame);
		Mat image = imdecode(rawData, IMREAD_COLOR);
		if (image.data == NULL)
		{
			if (JNI_DBG) {
				LOGD("image convert fail");
			}
			return NULL;
		}

		imgHeight = (int)image.rows;
		imgWidth = (int)image.cols;

		// For debug Image
		if (JNI_DBG) {
			cv::imwrite(std::string("/sdcard/TrackingDebug/DBG_") + ToString(debugid) + JPG, image);
		}

		/* Rect data (jint Array) */
		int jrectsLength = env->GetArrayLength(jrects);
		jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

		/* ID data (jint Array) */
		jintArray jids = env->NewIntArray(jrectsLength / 4);
		jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

		/* Rect data (Rect) */
		Rect rec = Rect((int)jrectsArrayData[0], (int)jrectsArrayData[1], (int)jrectsArrayData[2], (int)jrectsArrayData[3]);
		if (jrectsArrayData[2] < MIN_RECT_VALUE || jrectsArrayData[3] < MIN_RECT_VALUE) {
			if (JNI_DBG) {
				LOGD("Rect Object is too small, width %d height %d", rec.width, rec.height);
			}
			return NULL;
		}
		if (JNI_DBG)
			LOGD("initTrackingObjects, Rect width %d height %d", rec.width, rec.height);
		timer.Reset();
		timer.Start();
		jidsArrayData[0] = SetTrackingTarget((T_HANDLE)jhandle, image, rec);
		timer.Pause();
		trackingObjects.insert(make_pair(jidsArrayData[0], image(rec).clone()));
		/* return the init rect array*/
		int* buf_result = new int[5 * (jrectsLength / 4)];
		buf_result[0] = jidsArrayData[0];
		buf_result[1] = (int)jrectsArrayData[0];
		buf_result[2] = (int)jrectsArrayData[1];
		buf_result[3] = (int)jrectsArrayData[2];
		buf_result[4] = (int)jrectsArrayData[3];
		jintArray jIdsRects = env->NewIntArray(5);
		env->SetIntArrayRegion(jIdsRects, 0, 5, buf_result);

		env->ReleaseByteArrayElements(jimage, frame, 0);
		env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
		env->ReleaseIntArrayElements(jids, jidsArrayData, 0);
		env->DeleteLocalRef(jids);

		// For debug Image
		if (JNI_DBG) {
			writeDBGInfo(rec, 0);
			cv::rectangle(image, rec, Scalar(255, 0, 0));
			cv::imwrite(std::string("/sdcard/TrackingDebug/INIT_") + ToString(debugid) + JPG, image);
			debugid++;
		}

		return jIdsRects;
		//return 0;
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    initTrackingObjects
	* Signature: (J[BII[I)[I
	*/

	JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_initTrackingObjects
	(JNIEnv *env, jobject jNativeTracking, jlong jhandle, jbyteArray jimage, jint jwidth, jint jheight, jintArray jrects) {
		/* Image Size */
		imgHeight = (int)jheight;
		imgWidth = (int)jwidth;

		jbyte* frame = env->GetByteArrayElements(jimage, 0);
		Mat image;
		Mat myuv(imgHeight + imgHeight / 2, imgWidth, CV_8UC1, (uchar *)frame);
		cv::cvtColor(myuv, image, CV_YUV420sp2BGR);

		if (!image) {
			if (JNI_DBG) {
				LOGD("image convert fail");
			}
			return NULL;
		}

		// For debug Image
		if (JNI_DBG) {
			cv::imwrite(std::string("/sdcard/TrackingDebug/DBG_") + ToString(debugid) + JPG, image);
		}

		/* Rect data (jint Array) */
		int jrectsLength = env->GetArrayLength(jrects);
		jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

		/* ID data (jint Array) */
		jintArray jids = env->NewIntArray(jrectsLength / 4);
		jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

		/* Rect data (Rect) */
		Rect rec = Rect((int)jrectsArrayData[0], (int)jrectsArrayData[1], (int)jrectsArrayData[2], (int)jrectsArrayData[3]);
		if (jrectsArrayData[2] < MIN_RECT_VALUE || jrectsArrayData[3] < MIN_RECT_VALUE) {
			if (JNI_DBG) {
				LOGD("Rect Object is too small, width %d height %d", rec.width, rec.height);
			}
			return NULL;
		}
		if (JNI_DBG)
			LOGD("initTrackingObjects, Rect width %d height %d", rec.width, rec.height);
		timer.Reset();
		timer.Start();
		jidsArrayData[0] = SetTrackingTarget((T_HANDLE)jhandle, image, rec);
		timer.Pause();
		trackingObjects.insert(make_pair(jidsArrayData[0], myuv(rec).clone()));

		/* return the init rect array*/
		int* buf_result = new int[5 * (jrectsLength / 4)];
		buf_result[0] = jidsArrayData[0];
		buf_result[1] = (int)jrectsArrayData[0];
		buf_result[2] = (int)jrectsArrayData[1];
		buf_result[3] = (int)jrectsArrayData[2];
		buf_result[4] = (int)jrectsArrayData[3];
		jintArray jIdsRects = env->NewIntArray(5);
		env->SetIntArrayRegion(jIdsRects, 0, 5, buf_result);

		env->ReleaseByteArrayElements(jimage, frame, 0);
		env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
		env->ReleaseIntArrayElements(jids, jidsArrayData, 0);
		env->DeleteLocalRef(jids);

		// For debug Image
		if (JNI_DBG) {
			writeDBGInfo(rec, 0);
			cv::rectangle(image, rec, Scalar(255, 0, 0));
			cv::imwrite(std::string("/sdcard/TrackingDebug/INIT_") + ToString(debugid) + JPG, image);
			debugid++;
		}

		return jIdsRects;
		//return 0;
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    addTrackingObjects
	* Signature: (J[B[I)[I
	*/
	JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_addTrackingObjectsJPG
	(JNIEnv *env, jobject jNativeTracking, jlong jhandle, jbyteArray jimage, jint jsize, jintArray jrects) {

		jbyte* frame = env->GetByteArrayElements(jimage, 0);
		Mat rawData = Mat(1, jsize, CV_8UC1, (uchar *)frame);
		Mat image = imdecode(rawData, IMREAD_COLOR);
		if (image.data == NULL)
		{
			if (JNI_DBG) {
				LOGD("image convert fail");
			}

			env->ReleaseByteArrayElements(jimage, frame, 0);
			return NULL;
		}

		imgHeight = (int)image.rows;
		imgWidth = (int)image.cols;

		// For debug Image
		if (JNI_DBG) {
			cv::imwrite(std::string("/sdcard/TrackingDebug/DBG_") + ToString(debugid) + JPG, image);
		}

		/* Rect data (jint Array) */
		int jrectsLength = env->GetArrayLength(jrects);
		jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

		/* ID data (jint Array) */
		jintArray jids = env->NewIntArray(jrectsLength / 4);
		jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

		for (int i = 0, j = 0; i < jrectsLength; i += 4, j++) {
			// Rect data (Rect)
			const Rect& target = Rect((int)jrectsArrayData[i], (int)jrectsArrayData[i + 1], (int)jrectsArrayData[i + 2], (int)jrectsArrayData[i + 3]);
			if (jrectsArrayData[i + 2] < MIN_RECT_VALUE || jrectsArrayData[i + 3] < MIN_RECT_VALUE) {
				if (JNI_DBG) {
					LOGD("Rect Object is too small, width %d height %d", target.width, target.height);
				}

				env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
				env->ReleaseIntArrayElements(jids, jidsArrayData, 0);
				env->DeleteLocalRef(jids);
				return NULL;
			}
			if (JNI_DBG)
				LOGD("AddTrackingTarget, Rect width %d height %d", target.width, target.height);
			timer.Reset();
			timer.Start();
			jidsArrayData[j] = AddTrackingTarget((T_HANDLE)jhandle, image, target);
			timer.Pause();
			trackingObjects.insert(make_pair(jidsArrayData[j], image(target).clone()));


			// For debug Image
			if (JNI_DBG) {
				writeDBGInfo(target, 1);
				cv::rectangle(image, target, Scalar(255, 0, 0));
			}
		}

		/* return the init rect array*/
		int* buf_result = new int[5 * (jrectsLength / 4)];
		int data_count = 0;
		for (int i = 0, j = 0; i < jrectsLength; i += 4, j++) {
			buf_result[data_count] = jidsArrayData[j];
			buf_result[data_count + 1] = (int)jrectsArrayData[i];
			buf_result[data_count + 2] = (int)jrectsArrayData[i + 1];
			buf_result[data_count + 3] = (int)jrectsArrayData[i + 2];
			buf_result[data_count + 4] = (int)jrectsArrayData[i + 3];
			data_count = data_count + 5;
		}
		/* return the additional rect */
		jintArray jIdsRects = env->NewIntArray((jrectsLength / 4) * 5);
		env->SetIntArrayRegion(jIdsRects, 0, (jrectsLength / 4) * 5, buf_result);

		env->ReleaseByteArrayElements(jimage, frame, 0);
		env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
		env->ReleaseIntArrayElements(jids, jidsArrayData, 0);
		env->DeleteLocalRef(jids);

		// For debug Image
		if (JNI_DBG) {
			cv::imwrite(std::string("/sdcard/TrackingDebug/ADD_") + ToString(debugid) + JPG, image);
			debugid++;
		}

		return jIdsRects;
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    addTrackingObjects
	* Signature: (J[B[I)[I
	*/
	JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_addTrackingObjects
	(JNIEnv *env, jobject jNativeTracking, jlong jhandle, jbyteArray jimage, jintArray jrects) {

		jbyte* frame = env->GetByteArrayElements(jimage, 0);
		Mat image;
		Mat myuv(imgHeight + imgHeight / 2, imgWidth, CV_8UC1, (uchar *)frame);
		cv::cvtColor(myuv, image, CV_YUV420sp2BGR);

		if (!image) {
			if (JNI_DBG) {
				LOGD("image convert fail");
			}

			env->ReleaseByteArrayElements(jimage, frame, 0);
			return NULL;
		}

		// For debug Image
		if (JNI_DBG) {
			cv::imwrite(std::string("/sdcard/TrackingDebug/DBG_") + ToString(debugid) + JPG, image);
		}

		/* Rect data (jint Array) */
		int jrectsLength = env->GetArrayLength(jrects);
		jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

		/* ID data (jint Array) */
		jintArray jids = env->NewIntArray(jrectsLength / 4);
		jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

		for (int i = 0, j = 0; i < jrectsLength; i += 4, j++) {
			// Rect data (Rect)
			const Rect& target = Rect((int)jrectsArrayData[i], (int)jrectsArrayData[i + 1], (int)jrectsArrayData[i + 2], (int)jrectsArrayData[i + 3]);
			if (jrectsArrayData[i + 2] < MIN_RECT_VALUE || jrectsArrayData[i + 3] < MIN_RECT_VALUE) {
				if (JNI_DBG) {
					LOGD("Rect Object is too small, width %d height %d", target.width, target.height);
				}

				env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
				env->ReleaseIntArrayElements(jids, jidsArrayData, 0);
				env->DeleteLocalRef(jids);
				return NULL;
			}
			if (JNI_DBG)
				LOGD("AddTrackingTarget, Rect width %d height %d", target.width, target.height);

			timer.Reset();
			timer.Start();
			jidsArrayData[j] = AddTrackingTarget((T_HANDLE)jhandle, image, target);
			timer.Pause();
			trackingObjects.insert(make_pair(jidsArrayData[j], myuv(target).clone()));

			// For debug Image
			if (JNI_DBG) {
				writeDBGInfo(target, 1);
				cv::rectangle(image, target, Scalar(255, 0, 0));
			}
		}

		/* return the init rect array*/
		int* buf_result = new int[5 * (jrectsLength / 4)];
		int data_count = 0;
		for (int i = 0, j = 0; i < jrectsLength; i += 4, j++) {
			buf_result[data_count] = jidsArrayData[j];
			buf_result[data_count + 1] = (int)jrectsArrayData[i];
			buf_result[data_count + 2] = (int)jrectsArrayData[i + 1];
			buf_result[data_count + 3] = (int)jrectsArrayData[i + 2];
			buf_result[data_count + 4] = (int)jrectsArrayData[i + 3];
			data_count = data_count + 5;
		}
		/* return the additional rect */
		jintArray jIdsRects = env->NewIntArray((jrectsLength / 4) * 5);
		env->SetIntArrayRegion(jIdsRects, 0, (jrectsLength / 4) * 5, buf_result);

		env->ReleaseByteArrayElements(jimage, frame, 0);
		env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
		env->ReleaseIntArrayElements(jids, jidsArrayData, 0);
		env->DeleteLocalRef(jids);

		// For debug Image
		if (JNI_DBG) {
			cv::imwrite(std::string("/sdcard/TrackingDebug/ADD_") + ToString(debugid) + JPG, image);
			debugid++;
		}

		return jIdsRects;
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    removeTrackingObjects
	* Signature: (J[I)Z
	*/
	JNIEXPORT jboolean JNICALL Java_org_iii_snsi_videotracking_NativeTracking_removeTrackingObjects
	(JNIEnv *env, jobject jNativeTracking, jlong jhandle, jintArray jids) {
		bool result = false;

		/* ID data (jint Array) */
		int jidsLength = env->GetArrayLength(jids);
		jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

		for (int i = 0; i < jidsLength; i++) {
			/* Object ID */
			const int& object_id = jidsArrayData[i];
			if (JNI_DBG)
				LOGD("RemoveTrackingTarget");

			// Object ID = -1, remove all tracking object
			if (object_id == -1) {
				map<int, Mat>::iterator itr = trackingObjects.begin();
				for (; itr != trackingObjects.end(); itr++)
					result = RemoveTrackingTarget((T_HANDLE)jhandle, itr->first);
				trackingObjects.clear();
				break;
			}
			result = RemoveTrackingTarget((T_HANDLE)jhandle, object_id);
			if (!result)
				break;
		}

		env->ReleaseIntArrayElements(jids, jidsArrayData, 0);

		return (result == true ? JNI_TRUE : JNI_FALSE);
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    processTracking
	* Signature: (J[B)[I
	*/
	JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_processTracking
	(JNIEnv *env, jobject jNativeTracking, jlong jhandle, jbyteArray jimage, jint frame_format) {

		if (JNI_DBG) {
			LOGD("processTracking In");
		}

		if (trackingObjects.empty())
			return NULL;

		if (JNI_DBG) {
			LOGD("processTracking In2");
		}

		jbyte* frame = env->GetByteArrayElements(jimage, 0);
		Mat image;
		int data_rows;
		int data_cols;
		switch (frame_format) {
		case FRAME_NV21:
			data_rows = imgHeight * 3 / 2;
			break;
		case FRAME_YUV420P:
			data_rows = imgHeight * 3 / 2;
			break;
		case FRAME_GRAY:
			data_rows = imgHeight;
			break;
		}

		data_cols = imgWidth;
		Mat myuv(data_rows, data_cols, CV_8UC1, (uchar *)frame);
		switch (frame_format) {
		case FRAME_NV21:
			cv::cvtColor(myuv, image, CV_YUV420sp2GRAY);
			break;
		case FRAME_YUV420P:
			cv::cvtColor(myuv, image, CV_YUV420p2GRAY);
			break;
		case FRAME_GRAY:
			image = myuv.clone();
			  break;
		}

		if (JNI_DBG) {
			LOGD("data_rows = %d\n", data_rows);
			LOGD("data_cols = %d\n", data_cols);
		}

		if (!image) {
			if (JNI_DBG) {
				LOGD("image convert fail");
			}

			env->ReleaseByteArrayElements(jimage, frame, 0);
			return NULL;
		}

		// For debug Image
		if (JNI_DBG) {
			cv::imwrite(std::string("/sdcard/TrackingDebug/DBG_") + ToString(debugid) + JPG, image);
		}

		/* Map result */
		map<int, Rect> results;
		results.clear();

		if (JNI_DBG)
			LOGD("RunTargetTracking");
		//double t = (double)getTickCount();
		timer.Reset();
		timer.Start();
		RunTargetTracking((T_HANDLE)jhandle, image, results);
		timer.Pause();
		//t = ((double)getTickCount() - t) / getTickFrequency();

		/* Remove map to int[] */
		int* buf_result = new int[results.size() * 5];
		map<int, Rect>::iterator it;
		int data_count = 0;
		for (it = results.begin(); it != results.end(); it++) {
			/* Add ID & RECT */
			int id_element = it->first;
			Rect rect_element = it->second;
			buf_result[data_count] = id_element;
			buf_result[data_count + 1] = rect_element.x;
			buf_result[data_count + 2] = rect_element.y;
			buf_result[data_count + 3] = rect_element.width;
			buf_result[data_count + 4] = rect_element.height;
			data_count = data_count + 5;

			// For debug Image
			if (JNI_DBG) {
				writeDBGInfo(rect_element, 2);
				cv::rectangle(image, rect_element, Scalar(255, 0, 0));
			}
		}

		/* result data (jint Array) */
		jintArray jIdsRects = env->NewIntArray(results.size() * 5);
		env->SetIntArrayRegion(jIdsRects, 0, results.size() * 5, buf_result);
		env->ReleaseByteArrayElements(jimage, frame, 0);

		// For debug Image
		if (JNI_DBG) {
			cv::imwrite(std::string("/sdcard/TrackingDebug/RUN_") + ToString(debugid) + JPG, image);
			debugid++;
		}

		return jIdsRects;
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    releaseHandle
	* Signature: (J)Z
	*/
	JNIEXPORT jboolean JNICALL Java_org_iii_snsi_videotracking_NativeTracking_releaseHandle
	(JNIEnv *env, jobject jNativeTracking, jlong jhandle) {
		trackingObjects.clear();
		if (JNI_DBG)
			LOGD("DeleteVideoTracker");
		return (DeleteVideoTracker((T_HANDLE)jhandle) == true ? JNI_TRUE : JNI_FALSE);
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    getTrackingObjImg
	* Signature: (I)[B
	*/
	JNIEXPORT jbyteArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_getTrackingObjImg
	(JNIEnv *env, jobject jNativeTracking, jint jobjectID) {

		Mat image;
		map<int, Mat>::iterator itr = trackingObjects.find((int)jobjectID);
		if (itr != trackingObjects.end())
		{
			cvtColor(itr->second, image, CV_BGR2BGRA, 4);
			//image = itr->second.clone();
		}
		else {
			return NULL;
		}

		int size = image.total() * image.elemSize();
		jbyte* buf_result = new jbyte[size];  // you will have to delete[] that later
		memcpy(buf_result, image.data, size * sizeof(jbyte));

		jbyteArray imgData = env->NewByteArray(size);
		env->SetByteArrayRegion(imgData, 0, size, buf_result);

		if (JNI_DBG)
			LOGD("getTrackingObjImg");
		return imgData;
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    convertYUV2RGBA
	* Signature: (II[B)[I
	*/
	JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_convertYUV2RGBA
	(JNIEnv *env, jobject jNativeTracking, jint jwidth, jint jheight, jbyteArray jyuv) {

		jbyte* _yuv = env->GetByteArrayElements(jyuv, 0);
		int* _bgra = new int[jwidth * jheight];

		cv::Mat myuv(jheight + jheight / 2, jwidth, CV_8UC1, (unsigned char *)_yuv);
		cv::Mat mbgra(jheight, jwidth, CV_8UC4, (unsigned char *)_bgra);

		//Please make attention about BGRA byte order
		//ARGB stored in java as int array becomes BGRA at native level
		cv::cvtColor(myuv, mbgra, CV_YUV420sp2BGR, 4);

		env->ReleaseByteArrayElements(jyuv, _yuv, 0);

		/* return bgra array */
		jintArray bgra = env->NewIntArray(jwidth * jheight);
		env->SetIntArrayRegion(bgra, 0, jwidth * jheight, _bgra);

		return bgra;

	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    convertRGBA2YUV
	* Signature: (II[I)[B
	*/
	JNIEXPORT jbyteArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_convertRGBA2YUV
	(JNIEnv *env, jobject jNativeTracking, jint jwidth, jint jheight, jintArray jbgra) {

		jint* _bgra = env->GetIntArrayElements(jbgra, 0);
		int size = (jheight + jheight / 2) * jwidth;
		jbyte* _yuv = new jbyte[size];

		cv::Mat mbgra(jheight, jwidth, CV_8UC4, (unsigned char *)_bgra);
		cv::Mat myuv(jheight + jheight / 2, jwidth, CV_8UC1, (unsigned char *)_yuv);

		cv::cvtColor(mbgra, myuv, CV_BGR2YUV);

		env->ReleaseIntArrayElements(jbgra, _bgra, 0);

		/* return yuv array */
		jbyteArray yuv = env->NewByteArray(size);
		env->SetByteArrayRegion(yuv, 0, size, _yuv);

		return yuv;

	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    convertARGB2MAT
	* Signature: ([BIJ)Lorg/opencv/core/Mat;
	*/
	JNIEXPORT jobject JNICALL Java_org_iii_snsi_videotracking_NativeTracking_convertARGB2MAT
	(JNIEnv *env, jobject jNativeTracking, jbyteArray jargb, jint jsize) {
		// Get a class reference
		jclass classCvMat = env->FindClass("org/opencv/core/Mat");

		// Get the Field ID of the instance variables
		jmethodID matInit = env->GetMethodID(classCvMat, "<init>", "()V");
		jmethodID getPtrMethod = env->GetMethodID(classCvMat, "getNativeObjAddr", "()J");

		// Construct and return Mat
		jobject objCvMat = env->NewObject(classCvMat, matInit);
		Mat& matImage = *(Mat*)env->CallLongMethod(objCvMat, getPtrMethod);

		jbyte* frame = env->GetByteArrayElements(jargb, 0);
		Mat rawData = Mat(1, jsize, CV_8UC1, (uchar *)frame);
		matImage = imdecode(rawData, IMREAD_COLOR);

		imgHeight = (int)matImage.rows;
		imgWidth = (int)matImage.cols;

		// Release pointer
		env->DeleteLocalRef(classCvMat);
		env->ReleaseByteArrayElements(jargb, frame, 0);
		return objCvMat;
	}

	/*
	* Class:     org_iii_snsi_videotracking_NativeTracking
	* Method:    convertNV212MAT
	* Signature: ([BJ)Lorg/opencv/core/Mat;
	*/
	JNIEXPORT jobject JNICALL Java_org_iii_snsi_videotracking_NativeTracking_convertNV212MAT
	(JNIEnv *env, jobject jNativeTracking, jbyteArray jyuv) {
		// Get a class reference
		jclass classCvMat = env->FindClass("org/opencv/core/Mat");

		// Get the Field ID of the instance variables
		jmethodID matInit = env->GetMethodID(classCvMat, "<init>", "()V");
		jmethodID getPtrMethod = env->GetMethodID(classCvMat, "getNativeObjAddr", "()J");

		// Construct and return Mat
		jobject objCvMat = env->NewObject(classCvMat, matInit);
		Mat& matImage = *(Mat*)env->CallLongMethod(objCvMat, getPtrMethod);

		jbyte* frame = env->GetByteArrayElements(jyuv, 0);
		Mat image;
		Mat myuv(imgHeight + imgHeight / 2, imgWidth, CV_8UC1, (uchar *)frame);
		cv::cvtColor(myuv, image, CV_YUV420sp2BGR);
		cv::cvtColor(myuv, matImage, CV_YUV420sp2BGR);

		// Release pointer
		env->DeleteLocalRef(classCvMat);
		env->ReleaseByteArrayElements(jyuv, frame, 0);
		return objCvMat;
	}

#ifdef __cplusplus
}
#endif
