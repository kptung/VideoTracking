package org.iii.snsi.trackingtest;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;

import java.nio.ByteBuffer;
import org.iii.snsi.videotracking.NativeTracking;

public class MainActivity extends AppCompatActivity {

	private DisplayScreen displayScreen;
	private Bitmap bitmap;
	private Thread drawThread;
	private boolean threadFlag;
	private int resIdBase;
	private int resId;
	private long handle;
	private NativeTracking tracker = new NativeTracking();
	private static final int totalFrames = 100;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		displayScreen = (DisplayScreen)findViewById(R.id.surfaceView);
		resIdBase = R.raw.frame_0001;
		resId = 223;
		handle = tracker.createHandle();
		
		drawThread = new Thread(new Runnable(){
			@Override
			public void run(){
				while(threadFlag){
					if(resId==223){
						bitmap = BitmapFactory.decodeResource(getResources(),
							resIdBase + resId);

						//calculate how many bytes our image consists of.
						int width = bitmap.getWidth();
						int height = bitmap.getHeight();
						int bytes = bitmap.getByteCount();
						//int bytes = width*height*4;
						// Create a new buffer
						ByteBuffer buffer = ByteBuffer.allocate(bytes);
						// Move the byte data to the buffer
						bitmap.copyPixelsToBuffer(buffer);
						// Get the underlying array containing the data.
						byte[] pixels = buffer.array();

						float wr = (float) width / 320.0f;
						float hr = (float) height / 240.0f;

						// Specify a tracking target
						int[] rects = new int[4];
						rects[0] = (int) (120.0f * wr); // x, left
						rects[1] = (int) (150.0f * hr); // y, top
						rects[2] = (int) (52.0f * wr); // w
						rects[3] = (int) (45.0f * hr); // h
						tracker.initTrackingObjects(handle, pixels, width, height, rects);

						int right = rects[0] + rects[2];
						int bottom = rects[1] + rects[3];
						// top, left, bottom, right
						displayScreen.setRect(rects[1], rects[0], bottom, right);
						displayScreen.setBitmap(bitmap);
					}
					else{
						bitmap = BitmapFactory.decodeResource(getResources(),
							resIdBase + resId);

						int width = bitmap.getWidth();
						int height = bitmap.getHeight();
						int bytes = width*height*4;
						ByteBuffer buffer = ByteBuffer.allocate(bytes);
						bitmap.copyPixelsToBuffer(buffer);
						byte[] pixels = buffer.array();

						int[] rects = tracker.processTracking(handle, pixels);
						if (rects.length == 5){
							int right = rects[1] + rects[3];
							int bottom = rects[2] + rects[4];
							displayScreen.setRect(rects[2], rects[1], bottom, right);
						}
						displayScreen.setBitmap(bitmap);
					}

					displayScreen.paint();
					resId = 223 + (resId - 222)%50;
					sleep(10);
				}
				tracker.releaseHandle(handle);
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
