package xws.util;

/**
 * 二维点
 * 
 * @author 1036225283
 *
 */
public class Vec2 {

	private float x = 0;

	private float y = 0;
	
	public Vec2() {
		// TODO Auto-generated constructor stub
	}

	public Vec2(float[] vec) {
		// TODO Auto-generated constructor stub
		x = vec[0];
		y = vec[1];
	}

	public Vec2(float x, float y) {
		// TODO Auto-generated constructor stub
		this.x = x;
		this.y = y;
	}

	public float getX() {
		return x;
	}

	public void setX(float x) {
		this.x = x;
	}

	public float getY() {
		return y;
	}

	public void setY(float y) {
		this.y = y;
	}

}
