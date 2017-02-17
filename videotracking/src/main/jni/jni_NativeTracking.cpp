#include <jni_NativeTracking.h>
#include <stdio.h>
#include <stdlib.h>     /* NULL */
#include <assert.h>     /* assert */
#include <vector>
#include <opencv2/opencv.hpp>
#include <android/log.h>

#include "HMD_AbstractTracker.hpp"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>

#define LOG_NDEBUG 0
#define LOG_TAG "JNI_NativeTracking"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

#define JNI_DBG 0
#define VIDEO_TRACKING_LIB_VERSION 0.2
#define MIN_RECT_VALUE 13

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
// For debug Image
static int debugid = 1;
bool writeDBGInfo(Rect rect) {
	const std::string &filename = "/sdcard/TrackingDebug/TrackingRect.txt";
	FILE *fp = fopen(filename.c_str(), "a");
	if (fp)
	{
		fprintf(fp, "DBG ID: %d , Rect: %d %d %d %d\r\n", debugid, rect.x, rect.y, rect.width, rect.height);
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

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    createHandle
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_org_iii_snsi_videotracking_NativeTracking_createHandle
  (JNIEnv *env, jobject jNativeTracking) {
      trackingObjects.clear();
      if(JNI_DBG)
          LOGD("CreateVideoTracker");
      return (jlong)CreateVideoTracker();
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
      Mat myuv(imgHeight + imgHeight/2, imgWidth, CV_8UC1, (uchar *)frame);
      cv::cvtColor(myuv, image, CV_YUV420sp2BGR);

      if(!image) {
        if(JNI_DBG) {
            LOGD("image convert fail");
        }
        return NULL;
      }

      // For debug Image
      if(JNI_DBG) {
        cv::imwrite(std::string("/sdcard/TrackingDebug/DBG_")+ToString(debugid)+JPG, image);
      }

      /* Rect data (jint Array) */
      int jrectsLength = env->GetArrayLength(jrects);
      jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

      /* ID data (jint Array) */
      jintArray jids = env->NewIntArray(jrectsLength / 4);
      jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

      /* Rect data (Rect) */
      Rect rec = Rect((int)jrectsArrayData[0], (int)jrectsArrayData[1], (int)jrectsArrayData[2], (int)jrectsArrayData[3]);
      if(jrectsArrayData[2] < MIN_RECT_VALUE || jrectsArrayData[3] < MIN_RECT_VALUE) {
          if(JNI_DBG) {
            LOGD("Rect Object is too small, width %d height %d", rec.width, rec.height);
          }
          return NULL;
      }
      if(JNI_DBG)
          LOGD("initTrackingObjects, Rect width %d height %d", rec.width, rec.height);
      jidsArrayData[0] = SetTrackingTarget((T_HANDLE)jhandle, image, rec);
      trackingObjects.insert(make_pair(jidsArrayData[0], myuv(rec).clone()));
      /* return the init rect array*/
      int* buf_result = new int[ 5 * (jrectsLength / 4) ];
      buf_result[0]=jidsArrayData[0];
      buf_result[1]=(int)jrectsArrayData[0];
      buf_result[2]=(int)jrectsArrayData[1];
      buf_result[3]=(int)jrectsArrayData[2];
      buf_result[4]=(int)jrectsArrayData[3];
      jintArray jIdsRects = env->NewIntArray(5);
      env->SetIntArrayRegion(jIdsRects, 0 , 5, buf_result);

      env->ReleaseByteArrayElements(jimage, frame, 0);
      env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
      env->ReleaseIntArrayElements(jids, jidsArrayData, 0);
      env->DeleteLocalRef(jids);

      // For debug Image
      if(JNI_DBG) {
        writeDBGInfo(rec);
        cv::rectangle(image, rec, Scalar(255, 0, 0));
        cv::imwrite(std::string("/sdcard/TrackingDebug/INIT_")+ToString(debugid)+JPG, image);
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
JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_addTrackingObjects
  (JNIEnv *env, jobject jNativeTracking, jlong jhandle, jbyteArray jimage, jintArray jrects) {

      jbyte* frame = env->GetByteArrayElements(jimage, 0);
      Mat image;
      Mat myuv(imgHeight + imgHeight/2, imgWidth, CV_8UC1, (uchar *)frame);
      cv::cvtColor(myuv, image, CV_YUV420sp2BGR);

      if(!image) {
        if(JNI_DBG) {
            LOGD("image convert fail");
        }
        return NULL;
      }

      // For debug Image
      if(JNI_DBG) {
          cv::imwrite(std::string("/sdcard/TrackingDebug/DBG_")+ToString(debugid)+JPG, image);
      }

       /* Rect data (jint Array) */
      int jrectsLength = env->GetArrayLength(jrects);
      jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

       /* ID data (jint Array) */
      jintArray jids = env->NewIntArray(jrectsLength / 4);
      jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

      for (int i = 0, j = 0; i < jrectsLength; i += 4, j++) {
          // Rect data (Rect)
          const Rect& target = Rect((int)jrectsArrayData[i], (int)jrectsArrayData[i+1], (int)jrectsArrayData[i+2], (int)jrectsArrayData[i+3]);
          if(jrectsArrayData[i+2] < MIN_RECT_VALUE || jrectsArrayData[i+3] < MIN_RECT_VALUE) {
              if(JNI_DBG) {
                  LOGD("Rect Object is too small, width %d height %d", target.width, target.height);
              }
              return NULL;
          }
          if(JNI_DBG)
              LOGD("AddTrackingTarget, Rect width %d height %d", target.width, target.height);
          jidsArrayData[j] = AddTrackingTarget((T_HANDLE)jhandle, image, target);
          trackingObjects.insert(make_pair(jidsArrayData[j], myuv(target).clone()));

          // For debug Image
          if(JNI_DBG) {
            writeDBGInfo(target);
            cv::rectangle(image, target, Scalar(255, 0, 0));
          }
      }

      /* return the init rect array*/
      int* buf_result = new int[ 5 * (jrectsLength / 4) ];
      int data_count = 0;
      for (int i = 0, j = 0; i < jrectsLength; i += 4, j++) {
        buf_result[data_count]=jidsArrayData[j];
        buf_result[data_count + 1]=(int)jrectsArrayData[i];
        buf_result[data_count + 2]=(int)jrectsArrayData[i + 1];
        buf_result[data_count + 3]=(int)jrectsArrayData[i + 2];
        buf_result[data_count + 4]=(int)jrectsArrayData[i + 3];
        data_count = data_count + 5;
      }
      /* return the additional rect */
      jintArray jIdsRects = env->NewIntArray((jrectsLength / 4) * 5);
      env->SetIntArrayRegion(jIdsRects, 0 , (jrectsLength / 4) * 5, buf_result);

      env->ReleaseByteArrayElements(jimage, frame, 0);
      env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
      env->ReleaseIntArrayElements(jids, jidsArrayData, 0);
      env->DeleteLocalRef(jids);

      // For debug Image
      if(JNI_DBG) {
          cv::imwrite(std::string("/sdcard/TrackingDebug/ADD_")+ToString(debugid)+JPG, image);
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
          if(JNI_DBG)
              LOGD("RemoveTrackingTarget");

          // Object ID = -1, remove all tracking object
          if(object_id == -1) {
            map<int, Mat>::iterator itr = trackingObjects.begin();
            for ( ; itr != trackingObjects.end(); itr++ )
                result =  RemoveTrackingTarget((T_HANDLE)jhandle, itr->first);
            trackingObjects.clear();
            break;
          }
          result =  RemoveTrackingTarget((T_HANDLE)jhandle, object_id);
          if(!result)
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
  (JNIEnv *env, jobject jNativeTracking, jlong jhandle, jbyteArray jimage) {

      if(trackingObjects.empty())
          return NULL;

      jbyte* frame = env->GetByteArrayElements(jimage, 0);
      Mat image;
      Mat myuv(imgHeight + imgHeight/2, imgWidth, CV_8UC1, (uchar *)frame);
      cv::cvtColor(myuv, image, CV_YUV420sp2BGR);

      if(!image) {
        if(JNI_DBG) {
            LOGD("image convert fail");
        }
        return NULL;
      }

      // For debug Image
      if(JNI_DBG) {
          cv::imwrite(std::string("/sdcard/TrackingDebug/DBG_")+ToString(debugid)+JPG, image);
      }

      /* Map result */
      map<int, Rect> results;
      results.clear();

      if(JNI_DBG)
          LOGD("RunTargetTracking");
      double t = (double)getTickCount();
      RunTargetTracking((T_HANDLE)jhandle, image, results);
      t = ((double)getTickCount() - t) / getTickFrequency();

      /* Remove map to int[] */
      int* buf_result = new int[results.size() * 5];
      map<int, Rect>::iterator it;
      int data_count = 0;
      for(it = results.begin() ; it != results.end() ; it++) {
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
          if(JNI_DBG) {
            writeDBGInfo(rect_element);
            cv::rectangle(image, rect_element, Scalar(255, 0, 0));
          }
      }

      /* result data (jint Array) */
      jintArray jIdsRects = env->NewIntArray(results.size() * 5);
      env->SetIntArrayRegion(jIdsRects, 0 , results.size() * 5, buf_result);

      env->ReleaseByteArrayElements(jimage, frame, 0);

      // For debug Image
      if(JNI_DBG) {
          cv::imwrite(std::string("/sdcard/TrackingDebug/RUN_")+ToString(debugid)+JPG, image);
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
      if(JNI_DBG)
          LOGD("DeleteVideoTracker");
      return ( DeleteVideoTracker((T_HANDLE)jhandle) == true ? JNI_TRUE : JNI_FALSE);
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
      if(itr != trackingObjects.end())
      {
          cvtColor(itr->second, image, CV_BGR2BGRA, 4);
          //image = itr->second.clone();
      } else {
          return NULL;
      }

      int size = image.total() * image.elemSize();
      jbyte* buf_result = new jbyte[size];  // you will have to delete[] that later
      memcpy(buf_result, image.data, size*sizeof(jbyte));

      jbyteArray imgData = env->NewByteArray(size);
      env->SetByteArrayRegion(imgData, 0, size, buf_result);

      if(JNI_DBG)
          LOGD("getTrackingObjImg");
      return imgData;
  }

#ifdef __cplusplus
}
#endif
