package org.iii.snsi.trackingtest;

import android.content.Context;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.SurfaceHolder;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

/**
 * Created by jimchen on 2015/12/11.
 */
public class OldCamera implements SurfaceHolder.Callback {

    public Camera camera;
    public int frameAccumulator, rtFPS;
    private Camera.PreviewCallback getCamFrame;
    private Camera.Parameters camParameters;
    private String[] PreviewFormatList, pictureSizeList, fpsList;
    private int camW, camH;
    private int fpsIndex;
    // cameraState = 0 : stop
    // cameraState = 1 : configure OK
    // cameraState = 2 : playing
    // cameraState = 7 : surface mode, delay playing
    private int cameraState;
    private Thread fpsLog;
    private boolean fpsSwitch;
    private SurfaceHolder surfaceHolder;
    private SurfaceTexture surfaceTexture;

    private HandlerThread encodeHandlerThread;
    private Handler encodeHandler;
    private Runnable encodeRunnable;
    private boolean bufferMode;
    private byte[] data;
    private Context context;
    private CallbackFrameListener listener;

    public interface CallbackFrameListener {
        void onCallback(byte[] data);
    }

    public OldCamera() {
        frameAccumulator = 0;
        fpsIndex = 0;
        cameraState = 0;
        camW = 640;
        camH = 480;
        bufferMode = true;
        surfaceTexture = new SurfaceTexture(10);
        initializeList();
        initalizeRunnables();
    }

    private void initializeList() {
        getCameraParameters();
        getPreviewFormatList();
        getFPSList();
        getPictureSizeList();
    }

    public void setCallbackFunction(CallbackFrameListener listener) {
        this.listener = listener;
    }

    private void initalizeRunnables() {
        getCamFrame = new Camera.PreviewCallback() {
            @Override
            public void onPreviewFrame(byte[] dataCam, Camera camera) {
                data = dataCam;
                if (listener != null)
                    listener.onCallback(dataCam);
            }
        };
    }

    /**
     * Override method, shouldn't be called manually.
     */
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width,
                               int height) {
    }

    /**
     * Override method, shouldn't be called manually.
     */
    @Override
    public void surfaceCreated(SurfaceHolder holder) {
    }

    /**
     * Override method, shouldn't be called manually.
     */
    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        closeCamera();
    }

    public int getWidth() {
        return camW;
    }

    public int getHeight() {
        return camH;
    }

    /**
     * Changing the resolution of camera previewing.
     *
     * @param FormatIndex
     * @param IDCode
     */
    public synchronized void setPreviewFormat(int FormatIndex, int IDCode) {
        setPreviewSizeIndex(FormatIndex);
    }

    /**
     * Changing the resolution of taking picture.
     *
     * @param FormatIndex
     * @param IDCode      This number is for identifying the command which the
     *                    socket
     *                    receives.
     */
    public void setPhotoFormat(int FormatIndex, int IDCode) {
        if (camera == null) {
            return;
        }
        camParameters = camera.getParameters();
        List<Camera.Size> SizeList = camParameters.getSupportedPictureSizes();
        camParameters.setPictureSize(SizeList.get(FormatIndex).width,
                SizeList.get(FormatIndex).height);
        camera.setParameters(camParameters);

    }

    /**
     * Auto focus then taking picture.
     *
     * @param IDCode
     */
    public void takePicture(final int IDCode) {

        final Camera.PictureCallback PicCall = new Camera.PictureCallback() {
            @Override
            public void onPictureTaken(byte[] data, Camera camera) {
                String path =
                        Environment.getExternalStorageDirectory().toString();
                java.io.File file;
                FileOutputStream out = null;

                int picID = -1;
                try {
                    do {
                        picID++;
                        file = new java.io.File(path + String
                                .format("/DCIM/Streamer/Pic%08d.jpg", picID));

                    } while (file.exists());
                    out = new FileOutputStream(path + String
                            .format("/DCIM/Streamer/Pic%08d.jpg", picID));
                    int Count = 0;
                    while (data.length > Count * 100000) {
                        byte[] packet = new byte[100000];
                        if ((data.length - Count * 100000) >= 100000) {
                            System.arraycopy(data, Count * 100000, packet, 0,
                                    100000);
                        } else if ((data.length - Count * 100000) > 0) {
                            System.arraycopy(data, Count * 100000, packet, 0,
                                    data.length - Count * 100000);
                        }

                        Count = Count + 1;
                    }

                    out.write(data);

                } catch (Exception e) {
                    e.printStackTrace();
                } finally {
                    try {
                        out.close();
                    } catch (Throwable ignore) {
                    }
                }

                OldCamera.this.camera.startPreview();
            }
        };

        camera.autoFocus(new Camera.AutoFocusCallback() {
            @Override
            public void onAutoFocus(boolean success, Camera camera) {
                try {
                    OldCamera.this.camera.takePicture(null, null, null,
                            PicCall);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        });

    }

    /**
     * Open and display camera.
     */
    public synchronized boolean openCamera() {
        if (camera != null) {
            return false;
        }

        camera = Camera.open(0);
        startFPSLogThread();
        camParameters = camera.getParameters();
        camParameters.setPreviewSize(camW, camH);
        camParameters.setPreviewFpsRange(
                camParameters.getSupportedPreviewFpsRange().get(fpsIndex)[0],
                camParameters.getSupportedPreviewFpsRange().get(fpsIndex)[1]);
        camParameters.setPreviewFormat(ImageFormat.NV21);
        camParameters.setPictureFormat(ImageFormat.JPEG);
        camParameters.setRecordingHint(true);
        camParameters.setJpegQuality(100);
        if (camParameters.getSupportedFocusModes().contains("auto")) {
            camParameters.setFocusMode("auto");
        }

        camera.setParameters(camParameters);
        cameraState = 1;
        return true;
    }


    public synchronized boolean startPreview(boolean bufferMode) {
        if (cameraState == 0) {
            return false;
        }

        if (cameraState == 2) {
            return true;
        }

        this.bufferMode = bufferMode;
        try {
            if (!bufferMode) {
                camera.setPreviewTexture(surfaceTexture);
            } else if (surfaceHolder != null) {
                camera.setPreviewDisplay(surfaceHolder);
            } else {
                camera.setPreviewTexture(surfaceTexture);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }


        if (bufferMode) {
            camera.setPreviewCallback(getCamFrame);
            startEncodeThread();
        } else {
            camera.addCallbackBuffer(null);
            camera.setPreviewCallbackWithBuffer(null);
        }

        camera.startPreview();
        cameraState = 2;
        return true;
    }

    /**
     * Close the camera.
     */
    public synchronized void closeCamera() {
        if (camera == null) {
            return;
        }

        camera.stopPreview();
        camera.setPreviewCallbackWithBuffer(null);
        stopEncodeThread();
        stopFPSLogThread();


        camera.release();
        camera = null;
        cameraState = 0;
    }

    public String[] getPreviewFormatList() {
        if (PreviewFormatList != null) {
            return PreviewFormatList;
        }

        List<Camera.Size> SizeList = camParameters.getSupportedPreviewSizes();
        PreviewFormatList = new String[SizeList.size()];
        for (int i = 0; i < SizeList.size(); i++) {
            PreviewFormatList[i] = Integer.toString(SizeList.get(i).width) + "x"
                    + Integer.toString(SizeList.get(i).height);
        }

        return PreviewFormatList;
    }

    public String[] getFPSList() {
        if (fpsList != null) {
            return fpsList;
        }

        List<int[]> SizeList = camParameters.getSupportedPreviewFpsRange();
        fpsList = new String[SizeList.size()];
        for (int i = 0; i < SizeList.size(); i++) {
            fpsList[i] = Integer.toString(SizeList.get(i)[0] / 1000) + "~"
                    + Integer.toString(SizeList.get(i)[1] / 1000);
        }

        return fpsList;
    }

    public String[] getPictureSizeList() {
        if (pictureSizeList != null) {
            return pictureSizeList;
        }

        List<Camera.Size> pictureSizes =
                camParameters.getSupportedPictureSizes();
        pictureSizeList = new String[pictureSizes.size()];
        for (int i = 0; i < pictureSizes.size(); i++) {
            pictureSizeList[i] = Integer.toString(pictureSizes.get(i).width)
                    + "x" + Integer.toString(pictureSizes.get(i).height);
        }

        return pictureSizeList;
    }

    public void setFPSIndex(int index) {
        fpsIndex = index;
        if (fpsIndex < 0) {
            fpsIndex = 0;
        }

        if (fpsIndex >= camParameters.getSupportedPreviewFpsRange().size()) {
            fpsIndex = camParameters.getSupportedPreviewFpsRange().size() - 1;
        }
    }

    public void setPreviewSizeIndex(int index) {
        if (index < 0) {
            index = 0;
        }

        if (index >= camParameters.getSupportedPreviewSizes().size()) {
            index = camParameters.getSupportedPreviewSizes().size() - 1;
        }

        camW = camParameters.getSupportedPreviewSizes().get(index).width;
        camH = camParameters.getSupportedPreviewSizes().get(index).height;
    }

    /**
     * Change the brightness of the camera of this phone.
     *
     * @param value
     * @param ID
     */
    public void setExposureCompensation(int value, int ID) {
        int ExMax = camParameters.getMaxExposureCompensation();
        int ExMin = camParameters.getMinExposureCompensation();
        double percent = (double) value / 20;
        int setValue = (int) (Math.round((ExMax - ExMin) * percent) + ExMin);
        camParameters.setExposureCompensation(setValue);
        camera.setParameters(camParameters);
    }

    public int getCurrentFPS() {
        return rtFPS;
    }

    public void setSurfaceHolder(SurfaceHolder SurfaceViewHolder) {
        surfaceHolder = SurfaceViewHolder;
    }

    private void sleep(long time) {
        try {
            Thread.sleep(time);
        } catch (InterruptedException e) {
        }
    }

    private void startFPSLogThread() {
        fpsLog = new Thread(new Runnable() {
            @Override
            public void run() {
                while (fpsSwitch) {
                    sleep(1000);
                    rtFPS = frameAccumulator;
                    frameAccumulator = 0;
                }
            }
        });

        fpsSwitch = true;
        fpsLog.start();
    }

    private void stopFPSLogThread() {
        fpsSwitch = false;
        if (fpsLog != null) {
            fpsLog.interrupt();
            fpsLog = null;
        }
    }

    private void startEncodeThread() {
        if (encodeHandlerThread != null) {
            return;
        }
        encodeHandlerThread = new HandlerThread("encode");
        encodeHandlerThread.start();
        encodeHandler = new Handler(encodeHandlerThread.getLooper());
        encodeHandler.post(encodeRunnable);
    }

    private void stopEncodeThread() {
        if (encodeHandlerThread == null) {
            return;
        }
        encodeHandlerThread.getLooper().quit();
        encodeHandlerThread.interrupt();
        encodeHandlerThread = null;

    }

    private void getCameraParameters() {
        camera = Camera.open(0);
        camParameters = camera.getParameters();
        camera.release();
        camera = null;
    }
}
