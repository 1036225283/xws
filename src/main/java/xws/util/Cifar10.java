package xws.util;

import xws.neuron.Tensor;

/**
 * 包含RBG和label
 * Created by xws on 2019/3/14.
 */
public class Cifar10 {

    private Tensor data;
    private Tensor expect;
    private int label;
    private double value;

    public Tensor getData() {
        return data;
    }

    public void setData(Tensor data) {
        this.data = data;
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

    public Tensor getExpect() {
        return expect;
    }

    public void setExpect(Tensor expect) {
        this.expect = expect;
    }
}
