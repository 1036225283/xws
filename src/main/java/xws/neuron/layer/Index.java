package xws.neuron.layer;

/**
 * pool用来记录坐标信息，包括深度，高度，宽度
 * Created by xws on 2019/2/24.
 */
public class Index {

    private int depth;
    private int height;
    private int width;
    private double value;


    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
