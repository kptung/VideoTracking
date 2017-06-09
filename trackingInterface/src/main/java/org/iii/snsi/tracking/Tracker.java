package org.iii.snsi.tracking;

import org.opencv.core.Point3;

/**
 * Created by liting on 2017/6/9.
 */
public interface Tracker {

    int[] addTrackingObjectsJPG(byte[] image, int[] rect);

    int[] addTrackingObjects(byte[] image, int w, int h, int[] rect);

    int[] addTrackingObjectsWCS(Point3[] objWCS, int w, int h);

    boolean removeTrackingObjects(int[] ids);

    Object processTracking(byte[] image);

    void release();

    boolean isNativeLibAvailable();

}
