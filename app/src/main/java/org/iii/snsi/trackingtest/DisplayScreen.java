package org.iii.snsi.trackingtest;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class DisplayScreen extends SurfaceView
		implements SurfaceHolder.Callback {

	Paint paint;
	Canvas canvas;
	RectF rf;
	Bitmap picture;

	public DisplayScreen(Context context, AttributeSet attrs) {
		super(context, attrs);
		this.getHolder().addCallback(this);
		this.setBackgroundColor(0xFFFFFFFF);
		paint = new Paint();
		paint.setAntiAlias(true);
		rf = new RectF(50, 100, 160, 180);
	}

	public DisplayScreen(Context context) {
		super(context);
		this.getHolder().addCallback(this);
		paint = new Paint();
		paint.setAntiAlias(true);
		rf = new RectF(50, 100, 160, 180);
	}

	public void setRect(float top, float left, float bottom, float right) {
		rf.top = top;
		rf.left = left;
		rf.bottom = bottom;
		rf.right = right;

	}

	public void setBitmap(Bitmap bmp) {
		picture = bmp.copy(bmp.getConfig(), true);
	}

	public void paint() {
		if (canvas != null) {
			draw(canvas);
		}
	}

	@Override
	protected void onDraw(Canvas canvas) {
		paint.setColor(Color.WHITE);
		canvas.drawRect(0, 0, getWidth(), getHeight(), paint);
		if (picture != null) {
			canvas.drawBitmap(picture, 0, 0, paint);
		}
		//a,r,g,b
		paint.setARGB(255, 0, 0, 255);
		//hollow
		paint.setStyle(Paint.Style.STROKE);
		paint.setStrokeWidth(5);
		canvas.drawRect(rf, paint);
		postInvalidate();
	}

	@Override
	public void surfaceCreated(SurfaceHolder holder) {
		canvas = holder.lockCanvas();
	}

	@Override
	public void surfaceDestroyed(SurfaceHolder holder) {
		if (canvas != null) {
			holder.unlockCanvasAndPost(canvas);
		}
	}

	@Override
	public void surfaceChanged(SurfaceHolder holder, int format, int width,
			int height) {}
}
