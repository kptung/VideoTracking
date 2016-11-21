package org.iii.snsi.trackingtest;

import android.app.Activity;
import android.graphics.Rect;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.DisplayMetrics;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;

import org.iii.snsi.videotracking.NativeTracking;

public class MainActivity extends Activity{

	// Button initialization
	private Button cameraButton;
	private Button clearButton;
	private Button saveButton;
	private Button trackButton;

	// Thread initialization
	private Thread drawThread;
	private SurfaceView surfaceView;
	private OldCamera mCamera;
	private boolean threadFlag;
	private int mScreenHeight;
	private int mScreenWidth;

	// Tracking initialization
	private long handle;
	private NativeTracking tracker = new NativeTracking();
	private boolean trackflag = false;
	private boolean saveflag = false;
	private TouchView mView;
	private Rect rec = new Rect();
	private byte[] pixels;
	int bmapWidth;
	int bmapHeight;
	float wRatio;
	float hRatio;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		//Log.i(TAG, "onCreate()");
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
		mCamera = new OldCamera();
		bmapWidth=mCamera.getWidth();
		bmapHeight=mCamera.getHeight();
		wRatio=(float)bmapWidth/mScreenWidth;
		hRatio=(float)bmapHeight/mScreenHeight;
		/// draw camera view @ this surfaceview
		surfaceView = (SurfaceView)findViewById(R.id.surfaceView);

		/// button initialization
		cameraButton = (Button)findViewById(R.id.button1);
		clearButton = (Button)findViewById(R.id.button2);
		saveButton = (Button)findViewById(R.id.button3);
		trackButton = (Button)findViewById(R.id.button4);

		cameraButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				mCamera.openCamera();
				SurfaceHolder holder = surfaceView.getHolder();
				mCamera.setSurfaceHolder(holder);
				mCamera.setCallbackFunction(new OldCamera.CallbackFrameListener(){
					@Override
					public void onCallback(byte[] data) {
						pixels=data;
					}
				});
				mCamera.startPreview(true);

			}
		});

		clearButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				mView.setmLeftTopPosX(0);
				mView.setmLeftTopPosY(0);
				mView.setmRightTopPosX(0);
				mView.setmRightTopPosY(0);
				mView.setmLeftBottomPosX(0);
				mView.setmLeftBottomPosY(0);
				mView.setmRightBottomPosX(0);
				mView.setmRightBottomPosY(0);
			}
		});

		saveButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				saveflag=true;
			}
		});

		trackButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				trackflag=true;
			}
		});

		handle = tracker.createHandle();

		mView = (TouchView) findViewById(R.id.left_top_view);
		/// calcualte the rect moving distance
		rec.set((int) ((double) mScreenWidth * .85),
				(int) ((double) mScreenHeight * .10),
				(int) ((double) mScreenWidth * .85),
				(int) ((double) mScreenHeight * .70));
		/// set the rect position for boundary checking
		mView.setRec(rec);

		drawThread = new Thread(new Runnable(){
			@Override
			public void run(){
				while(threadFlag){
					if(saveflag) {
						// rect down-sampling : ( OpenCV Rect (left-top x, left-top y, width, height))
						int lx = Math.round(mView.getmLeftTopPosX() );
						int ly = Math.round(mView.getmLeftTopPosY() * hRatio);
						int width=Math.round(mView.getmRightTopPosX() - mView.getmLeftTopPosX());
						int height=Math.round(mView.getmLeftBottomPosY() - mView.getmLeftTopPosY());
						int[] rects = new int [4];
						rects[0] = Math.round(lx * wRatio);
						rects[1] = Math.round(ly * hRatio);
						rects[2] = Math.round(width * wRatio);
						rects[3] = Math.round(height * hRatio);
						// init tracking
						tracker.initTrackingObjects(handle, pixels, bmapWidth, bmapHeight, rects);
						saveflag=false;
					}
					if(trackflag){
						/// tracking
						int[] rects = tracker.processTracking(handle, pixels);
						// rect up-sampling
						if (rects.length == 5) {
							int lx=rects[1];
							int ly=rects[2];
							int width=rects[3];
							int height=rects[4];
							mView.setmLeftTopPosX(lx / wRatio);
							mView.setmLeftTopPosY(ly / hRatio);
							mView.setmRightTopPosX((lx + width)/wRatio);
							mView.setmRightTopPosY(ly / hRatio);
							mView.setmLeftBottomPosX(lx / wRatio);
							mView.setmLeftBottomPosY((ly + height)/hRatio);
							mView.setmRightBottomPosX((lx + width) / wRatio);
							mView.setmRightBottomPosY((ly + height) / hRatio);
							/// update UI
							Message msg = new Message();
							msg.what = 1;
							mHandler.sendMessage(msg);
						}


					}

					sleep(10);
				}
				tracker.releaseHandle(handle);
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
		mCamera.closeCamera();
		threadFlag = false;
	}

	public void sleep(long time){
		try {
			Thread.sleep(time);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}


}
