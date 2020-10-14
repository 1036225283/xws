package xws.neuron.layer.resnet;

import xws.neuron.Tensor;
import xws.neuron.layer.FullLayer;
import xws.neuron.layer.Layer;

import java.util.ArrayList;
import java.util.List;


/**
 * resNet
 * Created by xws on 2020/10/14.
 */
public class BottleneckFullLayer extends Layer {


    FullLayer full1;
    FullLayer full2;
    FullLayer full3;


    public BottleneckFullLayer() {
    }

    public BottleneckFullLayer(String name, String activationType, int num, double lambda) {
        super(BottleneckFullLayer.class.getSimpleName());
        full1 = new FullLayer(name, activationType, num, lambda);
        full2 = new FullLayer(name, activationType, num / 2, lambda);
        full3 = new FullLayer(name, activationType, num, lambda);

    }


    @Override
    public Tensor forward(Tensor tensor) {

        Tensor tensorInput = tensor;
        tensorInput = full1.forward(tensorInput);
        tensorInput = full2.forward(tensorInput);
        tensorInput = full3.forward(tensorInput);

        return tensorInput;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {

        Tensor tensorError = tensor;
        tensorError = full3.backPropagation(tensorError);
        tensorError = full2.backPropagation(tensorError);
        tensorError = full1.backPropagation(tensorError);
        tensorError.add(tensor);
        return tensorError;
    }

    public FullLayer getFull1() {
        return full1;
    }

    public void setFull1(FullLayer full1) {
        this.full1 = full1;
    }

    public FullLayer getFull2() {
        return full2;
    }

    public void setFull2(FullLayer full2) {
        this.full2 = full2;
    }

    public FullLayer getFull3() {
        return full3;
    }

    public void setFull3(FullLayer full3) {
        this.full3 = full3;
    }
}
