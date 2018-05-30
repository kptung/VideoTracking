package org.iii.snsi.videotracking;

import org.iii.snsi.tracking.Point3D;
import org.iii.snsi.tracking.Tracker;

public class NativeTracking implements Tracker {

	static {
		System.loadLibrary("videotracking");
	}

	protected long handle;
	protected boolean firstRun = true;

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
	 * @param rects A list of four elements integer array indicates rectangles
	 * that will be track. </ br>
	 * The data format is [[x1, y1, w1, h1],[x2, y2, w2, h2], ...]
	 * @return A positive rectangle id, return -1 if error occurred.
	 */
	private int[] initTrackingObjects(byte[] image, int width, int height,
			int[] rects) {
		return initTrackingObjects(handle, image, width, height, rects);
	}

	/**
	 * Processing Camshift to track rectangles
	 *
	 * @param image The NV21 image.
	 * @param width The image width.
	 * @param height The image height.
	 * @param rect The integer array that indicate rectangle.
	 * @return A positive rectangle id, return -1 if error occurred.
	 */
	@Override
	public int[] addTrackingObjects(byte[] image, int width, int height,
			int[] rect) {

		if (firstRun) {
			int[] result = initTrackingObjects(image, width, height, rect);
			if(result != null) {
				firstRun = false;
			}
			return result;
		}

		return addTrackingObjects(handle, image, rect);
	}

	/**
	 * Processing Camshift to track rectangles
	 *
	 * @param image The ARGB image.
	 * @param rect The integer array that indicate rectangle.
	 * @return A positive rectangle id, return -1 if error occurred.
	 */
	@Override
	public int[] addTrackingObjectsJPG(byte[] image, int[] rect) {
		if (firstRun) {
			int[] result = initTrackingObjectsJPG(handle, image, image.length,
					rect);
			if(result != null) {
				firstRun = false;
			}
			return result;
		}

		return addTrackingObjectsJPG(handle, image, image.length, rect);
	}

	@Override
	public int[] addTrackingObjectsWCS(Point3D[] objWCS, int w, int h) {
		return null;
	}

	/**
	 * The function to release rectangle object
	 *
	 * @param ids The rectangle id that is returned by initTracking.
	 * @return true if success, otherwise return false
	 */
    @Override
	public boolean removeTrackingObjects(int[] ids) {
		return removeTrackingObjects(handle, ids);
	}

	/**
	 * Processing Camshift to track rectangles
	 *
	 * @param image The NV21 image. i.e.
	 * [0,122,20,45,78,1,23,23,100,20,2,-1,-1,-1,-1] means the following:
	 * RectangleID 0: x = 122, y = 20, w = 45, h = 78.<br />
	 * RectangleID 1: x = 23, y = 23, w = 100, h = 20.<br />
	 * RectangleID 2: False alarm.
	 * @return Return false if error occurred, otherwise return true.
	 */
    @Override
	public Object processTracking(byte[] image) {
		return processTracking(handle, image);
	}

	/**
	 * The function to release Tracking algorithm
	 */
	@Override
	public void release() {
		releaseHandle(handle);
		handle = -1;
	}

    @Override
    public boolean isNativeLibAvailable() {
        return handle != -1;
    }

    /**
	 * Native functions (will be implemented by C/C++)
	 */
	private native synchronized long createHandle();

	private native synchronized int[] initTrackingObjects(long handle,
			byte[] image, int width, int height, int[] rects);

	protected native synchronized int[] initTrackingObjectsJPG(long handle,
			byte[] image, int size, int[] rects);

	private native synchronized int[] addTrackingObjects(long handle,
			byte[] image, int[] rects);

	protected native synchronized int[] addTrackingObjectsJPG(long handle,
			byte[] image, int size, int[] rects);

	protected native synchronized boolean removeTrackingObjects(long handle,
			int[] ids);

	protected native synchronized int[] processTracking(long handle, byte[] image);

	private native synchronized boolean releaseHandle(long handle);

	public native synchronized byte[] getTrackingObjImg(int objectID);

	public native synchronized int[] convertYUV2RGBA(int width, int height, byte[] yuv);

	public native synchronized byte[] convertRGBA2YUV(int width, int height, byte[] bgra);

//	protected native synchronized Mat convertARGB2MAT(byte[] argb, int size);
//
//	protected native synchronized Mat convertNV212MAT(byte[] yuv);
}
