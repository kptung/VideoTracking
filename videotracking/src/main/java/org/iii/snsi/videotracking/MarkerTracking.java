package org.iii.snsi.videotracking;

import android.os.Environment;
import android.util.Log;

import org.iii.snsi.irglass.library.IrMixedReality;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by randolphchen on 2017/5/11.
 */
public class MarkerTracking extends NativeTracking {

    static {
        System.loadLibrary("irmarker");
        System.loadLibrary("opencv_java3");
    }

    private static final String TAG = "MarkerTracking";
    private static final int TRACKING_IMG_WIDTH = 640;
    private static final int TRACKING_IMG_HEIGHT = 480;
    private static int trackingObjId;
    private static int imgWidth, imgHeight;
    private static List<Point3[]> objWCSPts;
    private static IrMixedReality.ProjResult projResult;

    public static final String DIR = Environment.getExternalStorageDirectory()
            .getAbsolutePath() + "/IR/";
    public static int num = 1;

    public MarkerTracking() {
        trackingObjId = -1;
        imgWidth = 0;
        imgHeight = 0;
        objWCSPts = new ArrayList<>();

        IrMixedReality.setDebugMode(true);
        IrMixedReality.loadCalibration();

        File f = new File(DIR);
        if (!f.exists()) {
            f.mkdirs();
        }
    }

    private int[] addTrackingObjects(Mat frame, int[] rect) {
        Mat trackingImage = new Mat();
        imgWidth = frame.cols();
        imgHeight = frame.rows();
        Imgproc.resize(frame, trackingImage,
                new Size(TRACKING_IMG_WIDTH, TRACKING_IMG_HEIGHT));

        double imgWidthRatio = (double) TRACKING_IMG_WIDTH / imgWidth;
        double imgHeightRatio = (double) TRACKING_IMG_HEIGHT / imgHeight;

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

        Log.e("MR", "addTrackingObjects = "
                + (objWCS != null ? "ok" : "failed"));

        return new int[] {trackingObjId};
    }

    @Override
    public int[] addTrackingObjectsJPG(byte[] image, int[] rect) {
        Mat frame = convertARGB2MAT(image, image.length);
        return addTrackingObjects(frame, rect);
    }

    @Override
    public int[] addTrackingObjects(byte[] image, int width, int height,
            int[] rect) {
        Mat matBGR = convertNV212MAT(image);
        return addTrackingObjects(matBGR, rect);
    }

    @Override
    public int[] addTrackingObjectsWCS(Point3[] objWCS, int w, int h) {
        return null;
    }

    @Override
    public boolean removeTrackingObjects(int[] ids) {
        trackingObjId = -1;
        objWCSPts.clear();
        return true;
    }

    @Override
    public Object processTracking(byte[] image) {
        Mat matBGR = convertNV212MAT(image);
        Mat trackingImage = new Mat();
        imgWidth = matBGR.cols();
        imgHeight = matBGR.rows();
        Imgproc.resize(matBGR, trackingImage,
                new Size(TRACKING_IMG_WIDTH, TRACKING_IMG_HEIGHT));

        // Imgcodecs.imwrite(DIR + num + ".jpg", trackingImage);
        // num++;

        double imgWidthRatio = (double) imgWidth / TRACKING_IMG_WIDTH;
        double imgHeightRatio = (double) imgHeight / TRACKING_IMG_HEIGHT;

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

    @Override
    public void release() {
    }

    @Override
    public boolean isNativeLibAvailable() {
        return true;
    }

    public static IrMixedReality.ProjResult getProjResult() {
        return projResult;
    }
}
