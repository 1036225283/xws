package xws.neuron.layer;

/**
 * pool用来记录坐标信息，包括深度，高度，宽度
 * Created by xws on 2019/2/24.
 */
public class Index {

    private int depthFrom;//池化前的深度
    private int depthTo;//池化后的深度
    private int heightFrom;//池化前的高度坐标
    private int heightTo;//池化后的高度坐标
    private int widthFrom;//池化前的宽度坐标
    private int widthTo;//池化后的宽度坐标
    private double value;

    public int getDepthFrom() {
        return depthFrom;
    }

    public void setDepthFrom(int depthFrom) {
        this.depthFrom = depthFrom;
    }

    public int getDepthTo() {
        return depthTo;
    }

    public void setDepthTo(int depthTo) {
        this.depthTo = depthTo;
    }

    public int getHeightFrom() {
        return heightFrom;
    }

    public void setHeightFrom(int heightFrom) {
        this.heightFrom = heightFrom;
    }

    public int getHeightTo() {
        return heightTo;
    }

    public void setHeightTo(int heightTo) {
        this.heightTo = heightTo;
    }

    public int getWidthFrom() {
        return widthFrom;
    }

    public void setWidthFrom(int widthFrom) {
        this.widthFrom = widthFrom;
    }

    public int getWidthTo() {
        return widthTo;
    }

    public void setWidthTo(int widthTo) {
        this.widthTo = widthTo;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
