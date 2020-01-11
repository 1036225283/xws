package xws.util;

import xws.neuron.Tensor;

/**
 * 包含RBG和label
 * Created by xws on 2019/3/14.
 */
public class Cifar10 {

    private Tensor rgb;
    private Tensor label;
    private double value;
    private int index;

    public Tensor getRgb() {
        return rgb;
    }

    public void setRgb(Tensor rgb) {
        this.rgb = rgb;
    }

    public Tensor getLabel() {
        return label;
    }

    public void setLabel(Tensor label) {
        this.label = label;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    public int getIndex() {
        return index;
    }

    public void setIndex(int index) {
        this.index = index;
    }
}
