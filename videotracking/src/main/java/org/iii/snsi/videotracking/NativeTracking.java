package org.iii.snsi.videotracking;

public class NativeTracking {

	static {
		System.loadLibrary("videotracking");
	}

	private long handle;
	private boolean firstRun = true;

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
	private int[] initTrackingObjects(byte[] image, int width, int height,
			int[] rects) {
		return initTrackingObjects(handle, image, width, height, rects);
	}

	/**
	 * Processing Camshit to track rectangles
	 *
	 * @param image The NV21 image.
	 * @param width The image width.
	 * @param height The image height.
	 * @param rect The integer array that indecate retangle.
	 * @return A postive rectangle id, return -1 if error occurred.
	 */
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
	 * Processing Camshit to track rectangles
	 *
	 * @param image The ARGB image.
	 * @param rect The integer array that indecate retangle.
	 * @return A postive rectangle id, return -1 if error occurred.
	 */
	public int[] addTrackingObjectsJPG(byte[] image, int size,
			int[] rect) {

		if (firstRun) {
			int[] result = initTrackingObjectsJPG(handle, image, size, rect);
			if(result != null) {
				firstRun = false;
			}
			return result;
		}

		return addTrackingObjectsJPG(handle, image, size, rect);
	}

	/**
	 * The function to release rectangle object
	 *
	 * @param ids The rectangle id that is returned by initTracking.
	 * @return true if success, otherwise return false
	 */
	public boolean removeTrackingObject(int[] ids) {
		return removeTrackingObjects(handle, ids);
	}

	/**
	 * Processing Camshit to track rectangles
	 *
	 * @param image The NV21 image. i.e.
	 * [0,122,20,45,78,1,23,23,100,20,2,-1,-1,-1,-1] means the following:
	 * RectangleID 0: x = 122, y = 20, w = 45, h = 78.<br />
	 * RectangleID 1: x = 23, y = 23, w = 100, h = 20.<br />
	 * RectangleID 2: False alarm.
	 * @return Return false if error occured, otherwise return true.
	 */
	public int[] processTracking(byte[] image) {
		return processTracking(handle, image);

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
	private native synchronized long createHandle();

	private native synchronized int[] initTrackingObjects(long handle,
			byte[] image, int width, int height, int[] rects);

	private native synchronized int[] initTrackingObjectsJPG(long handle,
			byte[] image, int size, int[] rects);

	private native synchronized int[] addTrackingObjects(long handle,
			byte[] image, int[] rects);

	private native synchronized int[] addTrackingObjectsJPG(long handle,
			byte[] image, int size, int[] rects);

	private native synchronized boolean removeTrackingObjects(long handle,
			int[] ids);

	private native synchronized int[] processTracking(long handle, byte[] image);

	private native synchronized boolean releaseHandle(long handle);

	public native synchronized byte[] getTrackingObjImg(int objectID);

	public native synchronized int[] convertYUV2RGBA(int width, int height, byte[] yuv);

	public native synchronized byte[] convertRGBA2YUV(int width, int height, byte[] bgra);

}
