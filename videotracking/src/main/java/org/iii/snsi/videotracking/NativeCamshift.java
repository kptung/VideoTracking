package org.iii.snsi.videotracking;

import android.graphics.Rect;

import java.util.HashMap;

public class NativeCamshift {

	static {
		System.loadLibrary("opencv_java3");
		System.loadLibrary("videotracking");
	}
	private long handle;

	public NativeCamshift() {
		handle = initialize();
	}

	public int initCamshift(byte[] imageNV21, int width, int height,
			int[] initRect) {
		return initCamshiftA(handle, imageNV21, width, height, initRect);
	}

	public int initCamshift(byte[] imageNV21, int width, int height,
			Rect rectangle) {
		return initCamshiftO(handle, imageNV21, width, height, rectangle);
	}

	public int processCamshift(byte[] imageNV21, int width, int height,
			int[] rectList) {
		return processCamshiftA(handle, imageNV21, width, height, rectList);
	}

	public int processCamshift(byte[] imageNV21, int width, int height,
			HashMap<Integer, Rect> rectList) {
		return processCamshiftO(handle, imageNV21, width, height, rectList);
	}

	public void deleteRectangle(int rectangleID) {
		deleteRectangle(handle, rectangleID);
	}

	public void releaseHandle() {
		releaseHandle(handle);
	}

	/**
	 * @return A structure pointer.
	 */
	private native long initialize();

	/**
	 * @param handle Structure pointer.
	 * @param image Input NV21image.
	 * @param width Image width.
	 * @param height Image height.
	 * @param rect A size = 4 int array that contains top left position of the
	 * rectangle and size. i.e. [x, y, w, h]
	 * @return Rectangle ID, ID should >=0, return -1 if initialize failed.
	 */
	private native int initCamshiftA(long handle, byte[] image, int width,
			int height, int[] rect);

	/**
	 * @param handle Structure pointer.
	 * @param image Input NV21image.
	 * @param width Image width.
	 * @param height Image height.
	 * @param rect Rectangle.
	 * @return Rectangle ID, ID should >=0, return -1 if initialize failed.
	 */
	private native int initCamshiftO(long handle, byte[] image, int width,
			int height, Rect rect);

	/**
	 * @param handle Structure pointer.
	 * @param image Input NV21image.
	 * @param width Image width.
	 * @param height Image height.
	 * @param rects i.e. [0,122,20,45,78,1,23,23,100,20,2,-1,-1,-1,-1] means the
	 * following: RectangleID 0: x = 122, y = 20, w = 45, h = 78. RectangleID 1:
	 * x = 23, y = 23, w = 100, h = 20. RectangleID 2: False alarm.
	 * @return Return -1 if error occured, 0 otherwise.
	 */
	private native int processCamshiftA(long handle, byte[] image, int width,
			int height, int[] rects);

	/**
	 * @param handle Structure pointer.
	 * @param image Input NV21image.
	 * @param width Image width.
	 * @param height Image height.
	 * @param rects Result list in hash map.
	 * @return Return -1 if error occured, 0 otherwise.
	 */
	private native int processCamshiftO(long handle, byte[] image, int width,
			int height, HashMap<Integer, Rect> rects);

	/**
	 * @param id The id is returned from initCamshift.
	 */
	private native void deleteRectangle(long handle, int id);

	/**
	 * @param handle Structure pointer.
	 */
	private native void releaseHandle(long handle);
}
