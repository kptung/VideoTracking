package org.iii.snsi.videotracking;

import org.iii.snsi.irglass.library.IrMixedReality;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by randolphchen on 2017/5/11.
 */
public class MarkerTracking extends NativeTracking {

    private static final String TAG = "MarkerTracking";
    private static final int TRACKING_IMG_WIDTH = 320;
    private static final int TRACKING_IMG_HEIGHT = 240;
    private static int trackingObjId;
    private static int imgWidth, imgHeight;
    private static List<Point3[]> objWCSPts;
    private static IrMixedReality.ProjResult projResult;

    public MarkerTracking() {
        IrMixedReality.loadCalibration();
        trackingObjId = -1;
        imgWidth = 0;
        imgHeight = 0;
        objWCSPts = new ArrayList<>();
    }

    public static IrMixedReality.ProjResult getProjResult() {
        return projResult;
    }

    /**
     * Processing Camshit to track rectangles
     *
     * @param image The ARGB image.
     * @param rect The integer array that indecate retangle.
     * @return A postive rectangle id, return -1 if error occurred.
     */
    public int[] addTrackingObjectsJPG(byte[] image, int size, int[] rect) {
        // image
        Mat frame = convertARGB2MAT(image, size);
        Mat trackingImage = new Mat();
        imgWidth = frame.cols();
        imgHeight = frame.rows();
        Imgproc.resize(frame, trackingImage,
                new Size(TRACKING_IMG_WIDTH, TRACKING_IMG_HEIGHT));

        // image ratio
        double imgWidthRatio, imgHeightRatio;
        imgWidthRatio = (double)TRACKING_IMG_WIDTH / imgWidth;
        imgHeightRatio = (double)TRACKING_IMG_HEIGHT / imgHeight;

        // corner
        Point[] corner = new Point[rect.length];
        for (int i = 0; i < rect.length; i += 4) {
            corner[i] = new Point(rect[i] * imgWidthRatio,
                    rect[i + 1] * imgHeightRatio);
            corner[i + 1] = new Point((rect[i] + rect[i + 2]) * imgWidthRatio,
                    rect[i + 1] * imgHeightRatio);
            corner[i + 2] = new Point((rect[i] + rect[i + 2]) * imgWidthRatio,
                    (rect[i + 1] + rect[i + 3]) * imgHeightRatio);
            corner[i + 3] = new Point(rect[i] * imgWidthRatio,
                    (rect[i + 1] + rect[i + 3]) * imgHeightRatio);
        }

        Point3[] objWCS = IrMixedReality.addObjectTracking(trackingImage,
                corner);
        if (objWCS != null) {
            objWCSPts.add(objWCS);
            trackingObjId += 1;
        }

        return new int[] {trackingObjId};
    }

    /**
     * The function to release rectangle object
     *
     * @param ids The rectangle id that is returned by initTracking.
     * @return true if success, otherwise return false
     */
    public boolean removeTrackingObject(int[] ids) {
        // Remove all tracking object
        trackingObjId = -1;
        objWCSPts.clear();
        return true;
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
        // image
        Mat mBGR = convertNV212MAT(image);
        Mat trackingImage = new Mat();
        imgWidth = mBGR.cols();
        imgHeight = mBGR.rows();
        Imgproc.resize(mBGR, trackingImage,
                new Size(TRACKING_IMG_WIDTH, TRACKING_IMG_HEIGHT));

        // image ratio
        double imgWidthRatio, imgHeightRatio;
        imgWidthRatio = (double)imgWidth / TRACKING_IMG_WIDTH;
        imgHeightRatio = (double)imgHeight / TRACKING_IMG_HEIGHT;

        // Rect
        int[] rect = new int[objWCSPts.size() * 5];
        for (int i = 0; i < objWCSPts.size() * 5; i += 5) {
            projResult = IrMixedReality.getObjectProj(trackingImage,
                    objWCSPts.get(i / 5));
            if (projResult != null && projResult.imagePts != null) {
                rect[i] = trackingObjId;
                rect[i + 1] = (int) (projResult.imagePts[0].x * imgWidthRatio);
                rect[i + 2] = (int) (projResult.imagePts[0].y * imgHeightRatio);
                rect[i + 3] = (int) ((projResult.imagePts[2].x
                        - projResult.imagePts[0].x) * imgWidthRatio);
                rect[i + 4] = (int) ((projResult.imagePts[2].y
                        - projResult.imagePts[0].y) * imgHeightRatio);
            } else {
                rect[i] = -1;
                rect[i + 1] = -1;
                rect[i + 2] = -1;
                rect[i + 3] = -1;
                rect[i + 4] = -1;
            }
        }

        return rect;

    }

    /**
     * The function to release Tracking algorithm
     */
    public void releaseHandle() {
        // Do nothing
    }

}
