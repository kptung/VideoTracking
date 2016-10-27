#include <jni_NativeTracking.h>

#include <stdlib.h>     /* NULL */
#include <assert.h>     /* assert */
#include <vector>
#include <opencv2/opencv.hpp>
#include <android/log.h>

#include "HMD_AbstractTracker.hpp"

#define LOG_TAG "JNI_NativeTracking"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

#define JNI_DBG 1
#define VIDEO_TRACKING_LIB_VERSION 0.1

using namespace std;
using namespace cv;

#ifdef __cplusplus
extern "C" {
#endif

static int imgHeight, imgWidth = 0;

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    createHandle
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_org_iii_snsi_videotracking_NativeTracking_createHandle
  (JNIEnv *env, jobject jNativeTracking) {
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
      // Image Size
      imgHeight = (int)jheight;
      imgWidth = (int)jwidth;

      // Image data (Mat)
      jbyte* jimageArrayData = env->GetByteArrayElements(jimage, 0);
      const Mat& image = Mat(imgHeight, imgWidth, CV_8UC1, (unsigned char *)jimageArrayData);

      // Rect data (jint Array)
      int jrectsLength = env->GetArrayLength(jrects);
      jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

      // ID data (jint Array)
      jintArray jids = env->NewIntArray(jrectsLength / 4);
      jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

      for (int i = 0; i < jrectsLength; i + 4) {
          // Rect data (Rect)
          const Rect& target = Rect((int)jrectsArrayData[i], (int)jrectsArrayData[i+1], (int)jrectsArrayData[i+2], (int)jrectsArrayData[i+3]);
          if(JNI_DBG)
              LOGD("SetTrackingTarget");
          jidsArrayData[i] = SetTrackingTarget((T_HANDLE)jhandle, image, target);
      }

      env->ReleaseByteArrayElements(jimage, jimageArrayData, 0);
      env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
      env->ReleaseIntArrayElements(jids, jidsArrayData, 0);

      return jids;
  }

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    addTrackingObjects
 * Signature: (J[B[I)[I
 */
JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_addTrackingObjects
  (JNIEnv *env, jobject jNativeTracking, jlong jhandle, jbyteArray jimage, jintArray jrects) {
      // Image data (Mat)
      jbyte* jimageArrayData = env->GetByteArrayElements(jimage, 0);
      const Mat& image = Mat(imgHeight, imgWidth, CV_8UC1, (unsigned char *)jimageArrayData);

      // Rect data (jint Array)
      int jrectsLength = env->GetArrayLength(jrects);
      jint* jrectsArrayData = env->GetIntArrayElements(jrects, 0);

      // ID data (jint Array)
      jintArray jids = env->NewIntArray(jrectsLength / 4);
      jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

      for (int i = 0; i < jrectsLength; i + 4) {
          // Rect data (Rect)
          const Rect& target = Rect((int)jrectsArrayData[i], (int)jrectsArrayData[i+1], (int)jrectsArrayData[i+2], (int)jrectsArrayData[i+3]);
          if(JNI_DBG)
              LOGD("AddTrackingTarget");
          jidsArrayData[i] = AddTrackingTarget((T_HANDLE)jhandle, image, target);
      }

      env->ReleaseByteArrayElements(jimage, jimageArrayData, 0);
      env->ReleaseIntArrayElements(jrects, jrectsArrayData, 0);
      env->ReleaseIntArrayElements(jids, jidsArrayData, 0);

      return jids;
  }

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    removeTrackingObjects
 * Signature: (J[I)Z
 */
JNIEXPORT jboolean JNICALL Java_org_iii_snsi_videotracking_NativeTracking_removeTrackingObjects
  (JNIEnv *env, jobject jNativeTracking, jlong jhandle, jintArray jids) {
      bool result = false;

      // ID data (jint Array)
      int jidsLength = env->GetArrayLength(jids);
      jint* jidsArrayData = env->GetIntArrayElements(jids, 0);

      for (int i = 0; i < jidsLength; i++) {
          // Object ID
          const int& object_id = jidsArrayData[i];
          if(JNI_DBG)
              LOGD("RemoveTrackingTarget");
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
      // Image data (Mat)
      jbyte* jimageArrayData = env->GetByteArrayElements(jimage, 0);
      const Mat& image = Mat(imgHeight, imgWidth, CV_8UC1, (unsigned char *)jimageArrayData);

      // Map result
      map<int, Rect> results;
      results.clear();

      if(JNI_DBG)
          LOGD("RunTargetTracking");
      RunTargetTracking((T_HANDLE)jhandle, image, results);

      // Remove map to int[]
      int* buf_result = new int[results.size() * 5];
      map<int, Rect>::iterator it;
      int data_count = 0;
      for(it = results.begin() ; it != results.end() ; it++) {
          // Add ID & RECT
          int id_element = it->first;
          Rect rect_element = it->second;
          buf_result[data_count] = id_element;
          buf_result[data_count + 1] = rect_element.x;
          buf_result[data_count + 2] = rect_element.y;
          buf_result[data_count + 3] = rect_element.width;
          buf_result[data_count + 4] = rect_element.height;
          data_count = data_count + 5;
      }

      // result data (jint Array)
      jintArray jIdsRects = env->NewIntArray(results.size() * 5);
      env->SetIntArrayRegion(jIdsRects, 0 , results.size() * 5, buf_result);

      env->ReleaseByteArrayElements(jimage, jimageArrayData, 0);

      return jIdsRects;

  }

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    releaseHandle
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_org_iii_snsi_videotracking_NativeTracking_releaseHandle
  (JNIEnv *env, jobject jNativeTracking, jlong jhandle) {
      if(JNI_DBG)
          LOGD("DeleteVideoTracker");
      return ( DeleteVideoTracker((T_HANDLE)jhandle) == true ? JNI_TRUE : JNI_FALSE);
  }

#ifdef __cplusplus
}
#endif
