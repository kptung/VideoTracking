package org.iii.snsi.videotracking;

import org.iii.snsi.irglass.library.IrMixedReality;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by randolphchen on 2017/5/11.
 */
public class MarkerTracking extends NativeTracking {

    private static final String TAG = "MarkerTracking";
    private static int trackingObjId;
    private static List<Point3[]> objWCSPts;
    private static int imgWidth, imgHeight;

    public MarkerTracking() {
        IrMixedReality.loadCalibration();
        trackingObjId = 0;
        objWCSPts = new ArrayList<>();
        imgWidth = 0;
        imgHeight = 0;
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
        Mat rawData = new Mat(1, size, CvType.CV_8UC1);
        rawData.put(1, size, image);
        Mat frame = Imgcodecs.imdecode(rawData, Imgcodecs.IMREAD_COLOR);
        imgWidth = frame.cols();
        imgHeight = frame.rows();

        // corner
        Point[] corner = new Point[rect.length];
        for (int i = 0; i < rect.length; i += 4) {
            corner[i].x = rect[i];
            corner[i].y = rect[i + 1];
            corner[i + 1].x = rect[i] + rect[i + 2];
            corner[i + 1].y = rect[i + 1];
            corner[i + 2].x = rect[i] + rect[i + 2];
            corner[i + 2].y = rect[i + 1] + rect[i + 3];
            corner[i + 3].x = rect[i];
            corner[i + 3].y = rect[i + 1] + rect[i + 3];
        }

        Point3[] objWCS = IrMixedReality.addObjectTracking(frame, corner);
        objWCSPts.add(objWCS);
        if (objWCS != null) {
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
        trackingObjId = 0;
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
        Mat mYUV = new Mat(imgHeight + imgHeight / 2, imgWidth, CvType.CV_8UC1);
        mYUV.put(imgHeight + imgHeight / 2, imgWidth, image);
        Mat mBGR = new Mat();
        Imgproc.cvtColor(mYUV, mBGR, Imgproc.COLOR_YUV420sp2BGR);

        // Rect
        int[] rect = new int[objWCSPts.size() * 10];
        for (int i = 0; i < objWCSPts.size() * 10; i += 10) {
            Point[] screenPt = IrMixedReality.getObjectProj(mBGR,
                    objWCSPts.get(i / 10));
            rect[i] = trackingObjId;
            rect[i + 1] = (int)screenPt[0].x;
            rect[i + 2] = (int)screenPt[0].y;
            rect[i + 3] = (int)(screenPt[2].x - screenPt[0].x);
            rect[i + 4] = (int)(screenPt[2].y - screenPt[0].y);
            rect[i + 5] = trackingObjId + 1;
            rect[i + 6] = (int)screenPt[4].x;
            rect[i + 7] = (int)screenPt[4].y;
            rect[i + 8] = (int)(screenPt[6].x - screenPt[4].x);
            rect[i + 9] = (int)(screenPt[6].y - screenPt[4].y);
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
