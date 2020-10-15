package xws.neuron.layer.resnet;

import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.Layer;
import xws.neuron.layer.Padding1DLayer;
import xws.neuron.layer.conv.Conv1DLayer;


/**
 * resNet
 * Created by xws on 2020/10/14.
 */
public class BottleneckConv1DLayer extends Layer {


    private Tensor tensorInput;
    private Tensor tensorOut;


    Padding1DLayer padding1DLayer = new Padding1DLayer(1);
    Conv1DLayer conv1;
    Conv1DLayer conv2;
    Conv1DLayer conv3;


    public BottleneckConv1DLayer() {
    }

    public BottleneckConv1DLayer(int num) {
        super(BottleneckConv1DLayer.class.getSimpleName());
        conv1 = new Conv1DLayer("", "relu", num, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001);
        conv2 = new Conv1DLayer("", "relu", num, 3, 1, 0, UtilNeuralNet.e() * 0.00000000001);
        conv3 = new Conv1DLayer("", "relu", num, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001);

    }


    @Override
    public Tensor forward(Tensor tensor) {

        tensorInput = tensor;
        tensor = conv1.forward(tensor);
        tensor = padding1DLayer.forward(tensor);
        tensor = conv2.forward(tensor);
        tensor = conv3.forward(tensor);

        tensor.add(tensorInput);
        return tensor;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {
        tensorOut = tensor;

        tensor = conv3.backPropagation(tensor);
        tensor = conv2.backPropagation(tensor);
        tensor = padding1DLayer.backPropagation(tensor);
        tensor = conv1.backPropagation(tensor);

        tensor.add(tensorOut);
        return tensor;
    }

    public Padding1DLayer getPadding1DLayer() {
        return padding1DLayer;
    }

    public void setPadding1DLayer(Padding1DLayer padding1DLayer) {
        this.padding1DLayer = padding1DLayer;
    }

    public Conv1DLayer getConv1() {
        return conv1;
    }

    public void setConv1(Conv1DLayer conv1) {
        this.conv1 = conv1;
    }

    public Conv1DLayer getConv2() {
        return conv2;
    }

    public void setConv2(Conv1DLayer conv2) {
        this.conv2 = conv2;
    }

    public Conv1DLayer getConv3() {
        return conv3;
    }

    public void setConv3(Conv1DLayer conv3) {
        this.conv3 = conv3;
    }
}
