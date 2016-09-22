package org.iii.snsi.videotracking;

import android.graphics.Rect;

import java.util.HashMap;

public class NativeCamshift {

	static {
		System.loadLibrary("opencv_java3");
		System.loadLibrary("videotracking");
	}
	private long handle;

	/**
	 * Constructor
	 */
	public NativeCamshift() {
		handle = initialize();
	}

	/**
	 * The function to initialize Camshift algorithm
	 *
	 * @param image The NV21 image.
	 * @param width The image width.
	 * @param height The image height.
	 * @param rect The four elements one dimension array that indecate a
	 * retangle. The data format is [x, y, w, h]
	 * @return A postive rectangle id, return -1 if error occurred.
	 */
	public int initrectangle(byte[] image, int width, int height,
			int[] rect) {
		return initCamshiftA(handle, image, width, height, rect);
	}

	/**
	 * The function to initialize Camshift algorithm
	 *
	 * @param image The NV21 image.
	 * @param width The image width.
	 * @param height The image height.
	 * @param rect The retangle to be track.
	 * @return A postive rectangle id, return -1 if error occurred.
	 */
	public int initCamshift(byte[] image, int width, int height,
			Rect rect) {
		return initCamshiftO(handle, image, width, height, rect);
	}

	/**
	 * Processing Camshit to track rectangles
	 *
	 * @param image The NV21 image.
	 * @param width The image width.
	 * @param height The image height.
	 * @param rects The one dimension integer array to save results. <br />
	 * i.e. [0,122,20,45,78,1,23,23,100,20,2,-1,-1,-1,-1] means the following:
	 * RectangleID 0: x = 122, y = 20, w = 45, h = 78.<br />
	 * RectangleID 1: x = 23, y = 23, w = 100, h = 20.<br />
	 * RectangleID 2: False alarm.
	 * @return Return false if error occured, otherwise return true.
	 */
	public boolean processCamshift(byte[] image, int width, int height,
			int[] rects) {
		return processCamshiftA(handle, image, width, height, rects);
	}

	/**
	 * Processing Camshit to track rectangles
	 *
	 * @param image The NV21 image.
	 * @param width The image width.
	 * @param height The image height.
	 * @param rects The hash map to save results.
	 * @return Return false if error occured, otherwise return true.
	 */
	public boolean processCamshift(byte[] image, int width, int height,
			HashMap<Integer, Rect> rects) {
		return processCamshiftO(handle, image, width, height, rects);
	}

	/**
	 * The function to release rectangle object
	 *
	 * @param id The rectangle id that is returned by initCamshift.
	 */
	public void deleteRectangle(int id) {
		deleteRectangle(handle, id);
	}

	/**
	 * The function to release Camshift algorithm
	 */
	public void releaseHandle() {
		releaseHandle(handle);
	}

	private native long initialize();

	private native int initCamshiftA(long handle, byte[] image, int width,
			int height, int[] rect);

	private native int initCamshiftO(long handle, byte[] image, int width,
			int height, Rect rect);

	private native boolean processCamshiftA(long handle, byte[] image,
			int width, int height, int[] rects);

	private native boolean processCamshiftO(long handle, byte[] image,
			int width, int height, HashMap<Integer, Rect> rects);

	private native void deleteRectangle(long handle, int id);

	private native void releaseHandle(long handle);
}
