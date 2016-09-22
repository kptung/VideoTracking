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

JNIEXPORT jlong JNICALL Java_org_iii_snsi_videotracking_NativeCamshift_initialize
(JNIEnv *env, jclass clazz) {
    return 0;
}

JNIEXPORT jint JNICALL Java_org_iii_snsi_videotracking_NativeCamshift_initCamshiftA
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jint width,
        jint height, jintArray rect) {
    return 0;
}

JNIEXPORT jint JNICALL Java_org_iii_snsi_videotracking_NativeCamshift_initCamshiftO
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jint width,
        jint height, jobject rect) {
    return 0;
}

JNIEXPORT jint JNICALL Java_org_iii_snsi_videotracking_NativeCamshift_processCamshiftA
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jint width,
        jint height, jintArray rects) {
    return 0;
}

JNIEXPORT jint JNICALL Java_org_iii_snsi_videotracking_NativeCamshift_processCamshiftO
(JNIEnv *env, jclass clazz, jlong handle, jbyteArray image, jint width,
        jint height, jobject rects) {
    return 0;
}

JNIEXPORT void JNICALL Java_org_iii_snsi_videotracking_NativeCamshift_deleteRectangle
(JNIEnv *env, jclass clazz, jlong handle, jint id) {

}

JNIEXPORT void JNICALL Java_org_iii_snsi_videotracking_NativeCamshift_releaseHandle
(JNIEnv *env, jclass clazz, jlong handle) {

}