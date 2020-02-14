package xws.util;

import xws.neuron.Tensor;

/**
 * batch tensor
 * Created by xws on 2019/3/14.
 */
public class BatchData {

    private Tensor data;
    private Tensor expect;
    private Tensor value;
    private Tensor gamma;//policy gradient

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

    public Tensor getValue() {
        return value;
    }

    public void setValue(Tensor value) {
        this.value = value;
    }

    public Tensor getGamma() {
        return gamma;
    }

    public void setGamma(Tensor gamma) {
        this.gamma = gamma;
    }
}
