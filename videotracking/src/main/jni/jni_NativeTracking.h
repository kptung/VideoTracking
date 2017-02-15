/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class org_iii_snsi_videotracking_NativeTracking */

#ifndef _Included_org_iii_snsi_videotracking_NativeTracking
#define _Included_org_iii_snsi_videotracking_NativeTracking
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    createHandle
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_org_iii_snsi_videotracking_NativeTracking_createHandle
  (JNIEnv *, jobject);

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    initTrackingObjects
 * Signature: (J[BII[I)[I
 */
JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_initTrackingObjects
  (JNIEnv *, jobject, jlong, jbyteArray, jint, jint, jintArray);

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    addTrackingObjects
 * Signature: (J[B[I)[I
 */
JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_addTrackingObjects
  (JNIEnv *, jobject, jlong, jbyteArray, jintArray);

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    removeTrackingObjects
 * Signature: (J[I)Z
 */
JNIEXPORT jboolean JNICALL Java_org_iii_snsi_videotracking_NativeTracking_removeTrackingObjects
  (JNIEnv *, jobject, jlong, jintArray);

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    processTracking
 * Signature: (J[B)[I
 */
JNIEXPORT jintArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_processTracking
  (JNIEnv *, jobject, jlong, jbyteArray);

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    releaseHandle
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_org_iii_snsi_videotracking_NativeTracking_releaseHandle
  (JNIEnv *, jobject, jlong);

/*
 * Class:     org_iii_snsi_videotracking_NativeTracking
 * Method:    getTrackingObjImg
 * Signature: (I)[B
 */
JNIEXPORT jbyteArray JNICALL Java_org_iii_snsi_videotracking_NativeTracking_getTrackingObjImg
  (JNIEnv *env, jobject jNativeTracking, jint jobjectID);

#ifdef __cplusplus
}
#endif
#endif
