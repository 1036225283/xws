package xws.util;

import xws.neuron.Tensor;

/**
 * 包含RBG和label
 * Created by xws on 2019/3/14.
 */
public class Cifar10 {

    private Tensor data;//data
    private Tensor expect;//expect for learn
    private double value;//
    private int index;

    public Tensor getData() {
        return data;
    }

    public void setData(Tensor data) {
        this.data = data;
    }

    public Tensor getExpect() {
        return expect;
    }

    public void setExpect(Tensor expect) {
        this.expect = expect;
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
