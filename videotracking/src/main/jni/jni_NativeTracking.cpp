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

#define JNI_DBG 1
#define VIDEO_TRACKING_LIB_VERSION 0.2
#define MIN_RECT_VALUE 8

using namespace std;
using namespace cv;

#ifdef __cplusplus
extern "C" {
#endif

static int imgHeight, imgWidth = 0;
static vector<int> trackingObjects = vector<int>();

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

      if(!image)
        return NULL;

      /* Rect data (jint Array) */
      int jrectsLength = env->GetArrayLength(jrects);
      jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

      /* ID data (jint Array) */
      jintArray jids = env->NewIntArray(jrectsLength / 4);
      jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

      /* Rect data (Rect) */
      Rect rec = Rect((int)jrectsArrayData[0], (int)jrectsArrayData[1], (int)jrectsArrayData[2], (int)jrectsArrayData[3]);
      if(JNI_DBG)
          LOGD("initTrackingObjects");
      jidsArrayData[0] = SetTrackingTarget((T_HANDLE)jhandle, image, rec);
      trackingObjects.push_back(jidsArrayData[0]);
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

      if(!image)
        return NULL;

       /* Rect data (jint Array) */
      int jrectsLength = env->GetArrayLength(jrects);
      jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

       /* ID data (jint Array) */
      jintArray jids = env->NewIntArray(jrectsLength / 4);
      jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

      for (int i = 0, j = 0; i < jrectsLength; i += 4, j++) {
          // Rect data (Rect)
          const Rect& target = Rect((int)jrectsArrayData[i], (int)jrectsArrayData[i+1], (int)jrectsArrayData[i+2], (int)jrectsArrayData[i+3]);
          if(JNI_DBG)
              LOGD("AddTrackingTarget");
          jidsArrayData[j] = AddTrackingTarget((T_HANDLE)jhandle, image, target);
          trackingObjects.push_back(jidsArrayData[j]);
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
            for (int itr = 0 ; itr < trackingObjects.size(); ++itr )
                result =  RemoveTrackingTarget((T_HANDLE)jhandle, trackingObjects[itr]);
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

      jbyte* frame = env->GetByteArrayElements(jimage, 0);
      Mat image;
      Mat myuv(imgHeight + imgHeight/2, imgWidth, CV_8UC1, (uchar *)frame);
      cv::cvtColor(myuv, image, CV_YUV420sp2BGR);

      if(!image)
          return NULL;

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
      }

      /* result data (jint Array) */
      jintArray jIdsRects = env->NewIntArray(results.size() * 5);
      env->SetIntArrayRegion(jIdsRects, 0 , results.size() * 5, buf_result);

      env->ReleaseByteArrayElements(jimage, frame, 0);

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

#ifdef __cplusplus
}
#endif
