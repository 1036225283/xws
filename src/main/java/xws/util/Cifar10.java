package xws.util;

import xws.neuron.Tensor;

/**
 * 包含RBG和label
 * Created by xws on 2019/3/14.
 */
public class Cifar10 {

    private Tensor rgb;
    private int label;
    private double value;

    public Tensor getRgb() {
        return rgb;
    }

    public void setRgb(Tensor rgb) {
        this.rgb = rgb;
    }

    public int getLabel() {
        return label;
    }

    public void setLabel(int label) {
        this.label = label;
    }

    public double getValue() {
        return value;
    }

    public void setValue(double value) {
        this.value = value;
    }
}
