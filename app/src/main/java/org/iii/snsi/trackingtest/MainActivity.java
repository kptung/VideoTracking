package org.iii.snsi.trackingtest;

import android.app.Activity;
import android.graphics.Rect;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;

import org.iii.snsi.multimedia.Camera2;
import org.iii.snsi.multimedia.OldCamera;
import org.iii.snsi.videotracking.NativeTracking;

import java.util.ArrayList;

import static android.content.ContentValues.TAG;

public class MainActivity extends Activity{

	// Button initialization
	private Button cameraButton;
	private Button saveButton;
	private Button trackButton;

	/// display classes;  sufaceView shows the camera captured screen preview;
	/// mCamera is to control camera; mView is to draw the rect when camera is opened
	private SurfaceView surfaceView;
	private OldCamera mCamera;
	private Camera2 mCamera2;
	private TouchView mView;

	///  show the available drawing rect on touchview
	private Rect rec = new Rect();

	// Thread initialization
	private Thread drawThread;
	private boolean threadFlag;

	// Tracking initialization
	private NativeTracking tracker;
	private int mScreenHeight;
	private int mScreenWidth;
	/// camera captured frame
	private byte[] preview;
	private int previewWidth;
	private int previewHeight;
	/// the ratio of the screen to camera captured frame
	private float wRatio;
	private float hRatio;
	/// a number of tracking objs
	private int objcount=0;
	private int trackcount=0;

	/// multi-objs tracking initialization
	ArrayList<Integer> rect = new ArrayList();
	ArrayList<Integer> other = new ArrayList();

	/// the flag to open camera
	private boolean cameraflag=false;
	/// the flag to process real-time video tracking
	private boolean camtrackflag = false;
	/// the flag to save real-time rectangle
	private boolean camsaveflag = false;
	/// the flag to process multi-objs tracking
	private boolean track_multiobj_flag = true;

	//
	int x=0;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		//Log.i(TAG, "onCreate()");
		tracker= new NativeTracking();

		requestWindowFeature(Window.FEATURE_NO_TITLE);
		getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

		// display our (only) XML layout - Views already ordered
		setContentView(R.layout.activity_main);

		// according to device screen size 2 get the window width and height to display buttons
		DisplayMetrics displaymetrics = new DisplayMetrics();
		getWindowManager().getDefaultDisplay().getMetrics(displaymetrics);
		mScreenHeight = displaymetrics.heightPixels;
		mScreenWidth = displaymetrics.widthPixels;

		// Show camera
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
			mCamera = new OldCamera();
			previewWidth = mCamera.getWidth();
			previewHeight = mCamera.getHeight();
		} else {
			mCamera2 = new Camera2(this);
			previewWidth = mCamera2.getWidth();
			previewHeight = mCamera2.getHeight();
		}

		wRatio=(float)previewWidth/mScreenWidth;
		hRatio=(float)previewHeight/mScreenHeight;
		/// draw camera view @ this surfaceview
		surfaceView = (SurfaceView)findViewById(R.id.surfaceView);
		/// button initialization
		cameraButton = (Button)findViewById(R.id.button1);
		saveButton = (Button)findViewById(R.id.button2);
		trackButton = (Button)findViewById(R.id.button3);

		cameraButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				Log.i(TAG, "Open camera");
				SurfaceHolder holder = surfaceView.getHolder();
				if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
					mCamera.openCamera();
					mCamera.setSurfaceHolder(holder);
					mCamera.setCallbackFrameListener(
							new OldCamera.CallbackFrameListener() {
								@Override
								public void onCallbackFrame(byte[] data,
										int width, int height) {
									preview = data;
								}
							});
					mCamera.startPreview(true, false);
				} else {
					mCamera2.setCallbackFrameListener(
							new Camera2.CallbackFrameListener() {
								@Override
								public void onCallbackFrame(byte[] data,
										int width, int height) {
									preview = data;
								}
							});
					mCamera2.setSurfaceHolder(holder);
					mCamera2.openCamera();
					mCamera2.startPreview(true, true, false);
				}

				tracker = new NativeTracking();
			}
		});

		saveButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				Log.i(TAG, "save roi position");
				objcount+=1;
				camsaveflag=true;
			}
		});

		trackButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				Log.i(TAG, "strat tracking");
				camtrackflag=true;
			}
		});

		//handle = tracker.createHandle();

		mView = (TouchView) findViewById(R.id.left_top_view);
		/// calcualte the rect moving distance
		rec.set((int) ((double) mScreenWidth),
				(int) ((double) mScreenHeight),
				(int) ((double) mScreenWidth),
				(int) ((double) mScreenHeight));
		/// set the rect position for boundary checking
		mView.setRec(rec);

		drawThread = new Thread(new Runnable(){
			@Override
			public void run(){
				while(threadFlag){
					if(camsaveflag) {
						// rect down-sampling : ( OpenCV Rect (left-top x, left-top y, width, height))
						int lx = Math.round(mView.getmLeftTopPosX() * wRatio);
						int ly = Math.round(mView.getmLeftTopPosY() * hRatio);
						int width=Math.round((mView.getmRightTopPosX() - mView.getmLeftTopPosX()) * wRatio);
						int height=Math.round((mView.getmRightBottomPosY() - mView.getmRightTopPosY()) * hRatio);

						// Specify a tracking target
						if(objcount==1)
						{
							rect.add(lx);
							rect.add(ly);
							rect.add(width);
							rect.add(height);
							int[] roi = convert2intArray(rect);
							// init tracking
							int[] initarr=tracker.addTrackingObjects(preview, previewWidth, previewHeight, roi);
						}
						else if(objcount>1)
						{
							other.add(lx);
							other.add(ly);
							other.add(width);
							other.add(height);
						}
						camsaveflag=false;
					}
					if(camtrackflag){
						trackcount+=1;
						if(objcount>1 && trackcount==1)
						{
							//all.addAll(other);
							int[] others = convert2intArray(other);
							int[] addarr = tracker.addTrackingObjects(preview, previewWidth, previewHeight, others);
						}
						/// tracking
						int[] rects = (int []) tracker.processTracking(preview);
						// rect up-sampling
						for(int i=0;i<rects.length;i+=5) {
							int lx=rects[i + 1];
							int ly=rects[i + 2];
							int w=rects[i + 3];
							int h=rects[i + 4];
							rects[i + 1] = Math.round(lx / wRatio);
							rects[i + 2] = Math.round(ly / hRatio);
							rects[i + 3] = Math.round((lx + w) / wRatio );
							rects[i + 4] = Math.round((ly + h) / hRatio );
						}

						/// update Rects
						mView.setRects(rects);
						/// update UI
						Message msg = new Message();
						msg.what = 1;
						mHandler.sendMessage(msg);
					}
					sleep(10);
				}
				tracker.release();
			}
		});
		threadFlag = true;
		drawThread.start();

	}

	private Handler mHandler = new Handler(){
		@Override
		public void handleMessage(Message msg) {
			switch(msg.what){
				case 1:
					mView.setVisibility(View.INVISIBLE);
					mView.setVisibility(View.VISIBLE);
					break;
			}
		}
	};

	@Override
	protected void onDestroy() {

		// start another activity
		super.onDestroy();
		if (Build.VERSION.SDK_INT < Build.VERSION_CODES.LOLLIPOP) {
			mCamera.closeCamera();
		} else {
			mCamera2.closeCamera();
		}
		threadFlag = false;
	}

	public void sleep(long time){
		try {
			Thread.sleep(time);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	public int[] convert2intArray(ArrayList<Integer> IntegerList) {
		int[] intArray = new int[IntegerList.size()];
		int count = 0;
		for(int i : IntegerList){
			intArray[count++] = i;
		}
		return intArray;
	}

}
