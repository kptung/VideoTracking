package org.iii.snsi.videotracking;

import android.graphics.Rect;

import java.util.HashMap;

public class NativeTracking {

	static {
		System.loadLibrary("videotracking");
	}
	private long handle;

	/**
	 * Constructor
	 */
	public NativeTracking() {
		handle = createHandle();
	}

	/**
	 * The function to initialize Tracking algorithm
	 *
	 * @param image The NV21 image.
	 * @param width The image width.
	 * @param height The image height.
	 * @param rects A list of four elements integer array indecates retangles
	 * that will be track. </ br>
	 * The data format is [[x1, y1, w1, h1],[x2, y2, w2, h2], ...]
	 * @return A postive rectangle id, return -1 if error occurred.
	 */
	public int initTracking(byte[] image, int width, int height,
			int[] rects) {
		return 0;
		//return initTrackingA(handle, image, width, height, rects);
	}

	/**
	 * The function to initialize Tracking algorithm
	 *
	 * @param image The NV21 image.
	 * @param width The image width.
	 * @param height The image height.
	 * @param rects The retangles to be track.
	 * @return A postive rectangle id, return -1 if error occurred.
	 */
	public int initTracking(byte[] image, int width, int height,
			Rect[] rects) {
		return 0;
		//return initTrackingO(handle, image, width, height, rects);
	}

	/**
	 * Processing Camshit to track rectangles
	 *
	 * @param image The NV21 image.
	 * @param rect The integer array that indecate retangle.
	 * @return A postive rectangle id, return -1 if error occurred.
	 */
	public int addTrackingObject(byte[] image, int[] rect) {
		//return addTrackingObjectA(handle, image, rect);
		return 0;
	}

	/**
	 * Processing algorithm to track rectangles
	 *
	 * @param image The NV21 image.
	 * @param rect The retangle to be track
	 * @return A postive rectangle id, return -1 if error occurred.
	 */
	public int addTrackingObject(byte[] image, Rect rect) {
		return 0;
		//return addTrackingObjectO(handle, image, rect);
	}

	/**
	 * The function to release rectangle object
	 *
	 * @param ids The rectangle id that is returned by initTracking.
	 */
	public void removeTrackingObject(int[] ids) {
		removeTrackingObjects(handle, ids);
	}

	/**
	 * Processing Camshit to track rectangles
	 *
	 * @param image The NV21 image.
	 * @param rects The one dimension integer array to save results. <br />
	 * i.e. [0,122,20,45,78,1,23,23,100,20,2,-1,-1,-1,-1] means the following:
	 * RectangleID 0: x = 122, y = 20, w = 45, h = 78.<br />
	 * RectangleID 1: x = 23, y = 23, w = 100, h = 20.<br />
	 * RectangleID 2: False alarm.
	 * @return Return false if error occured, otherwise return true.
	 */
	public boolean processTracking(byte[] image, int[] rects) {
		return true;
		//return processTrackingA(handle, image, rects);
	}

	/**
	 * Processing Camshit to track rectangles
	 *
	 * @param image The NV21 image.
	 * @param rects The hash map to save results.
	 * @return Return false if error occured, otherwise return true.
	 */
	public boolean processTracking(byte[] image,
			HashMap<Integer, Rect> rects) {
		return true;
		//return processTrackingO(handle, image, rects);
	}

	/**
	 * The function to release Tracking algorithm
	 */
	public void releaseHandle() {
		releaseHandle(handle);
	}
	
	
	
	/**
	 * Native functions (will be implemented by C/C++)
	 */
	private native long createHandle();

	private native int[] initTrackingObjects(long handle, byte[] image, int width, int height, int[] rects);

	private native int[] addTrackingObjects(long handle, byte[] image, int[] rects);

	private native void removeTrackingObjects(long handle, int[] ids);

	private native boolean processTracking(long handle, byte[] image, int[] ids, int[] rects);

	private native void releaseHandle(long handle);
	
}
