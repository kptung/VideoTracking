#include "tracking.h"
#include "opencv2/opencv.hpp"
#include "opencv2/stitching.hpp"

#include <opencv2/core/utility.hpp>
//#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>

#include <stdio.h>
#include <ctype.h>
#include "common.hpp"


using namespace std;
using namespace cv;

JNIEXPORT jlong JNICALL Java_org_iii_snsi_videotracking_NativeTracking_createHandle
(JNIEnv *env, jclass clazz) {
    return 0;
}

JNIEXPORT jint JNICALL Java_org_iii_snsi_videotracking_NativeTracking_initTrackingA
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jint width,
        jint height, jintArray rect) {
    return 0;
}

JNIEXPORT jint JNICALL Java_org_iii_snsi_videotracking_NativeTracking_initTrackingO
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jint width,
        jint height, jobjectArray rects) {
    return 0;
}

JNIEXPORT jint JNICALL Java_org_iii_snsi_videotracking_NativeTracking_addTrackingObjectA
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jintArray rect) {
    return 0;
}

JNIEXPORT jint JNICALL Java_org_iii_snsi_videotracking_NativeTracking_addTrackingObjectO
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jobject rect) {
    return 0;
}

JNIEXPORT void JNICALL Java_org_iii_snsi_videotracking_NativeTracking_removeTrackingObject
(JNIEnv *env, jclass clazz, jlong handle, jint id) {
}

JNIEXPORT jboolean JNICALL Java_org_iii_snsi_videotracking_NativeTracking_processTrackingA
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image,
        jintArray rects) {
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_org_iii_snsi_videotracking_NativeTracking_processTrackingO
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jobject rects) {
    return JNI_TRUE;
}

JNIEXPORT void JNICALL Java_org_iii_snsi_videotracking_NativeTracking_releaseHandle
(JNIEnv *env, jclass clazz, jlong handle) {
}
