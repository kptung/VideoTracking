package org.iii.snsi.trackingtest;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

public class MainActivity extends AppCompatActivity {

	private DisplayScreen displayScreen;
	private Bitmap bitmap;
	private Thread drawThread;
	private boolean threadFlag;
	private int resIdBase;
	private int resId;
	private static final int totalFrames = 100;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		displayScreen = (DisplayScreen)findViewById(R.id.surfaceView);
		resIdBase = R.raw.frame_0001;
		resId = 0;
		drawThread = new Thread(new Runnable(){
			@Override
			public void run() {
				while(threadFlag){
					bitmap = BitmapFactory.decodeResource(getResources(),
							resIdBase + resId);
					displayScreen.setBitmap(bitmap);
					displayScreen.setRect(50,50,200,200);
					displayScreen.paint();
					resId = (resId + 1)%100;
					sleep(10);
				}
			}
		});
		threadFlag = true;
		drawThread.start();


	}

	@Override
	protected void onDestroy() {

		// start another activity
		super.onDestroy();

	}

	public void sleep(long time){
		try {
			Thread.sleep(time);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
}
