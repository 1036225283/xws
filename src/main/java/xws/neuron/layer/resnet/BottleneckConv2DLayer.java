package xws.neuron.layer.resnet;

import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.Layer;
import xws.neuron.layer.Padding2DLayer;
import xws.neuron.layer.conv.ConvolutionLayer;


/**
 * resNet
 * Created by xws on 2020/10/14.
 */
public class BottleneckConv2DLayer extends Layer {


    private Tensor tensorInput;
    private Tensor tensorOut;


    Padding2DLayer padding2DLayer = new Padding2DLayer(1);
    ConvolutionLayer conv1;
    ConvolutionLayer conv2;
    ConvolutionLayer conv3;


    public BottleneckConv2DLayer() {
    }

    public BottleneckConv2DLayer(int num) {
        super(BottleneckConv2DLayer.class.getSimpleName());
        conv1 = new ConvolutionLayer("", "relu", num, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001);
        conv2 = new ConvolutionLayer("", "relu", num, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001);
        conv3 = new ConvolutionLayer("", "relu", num, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001);

    }


    @Override
    public Tensor forward(Tensor tensor) {

        tensorInput = tensor;
        tensor = conv1.forward(tensor);
        tensor = padding2DLayer.forward(tensor);
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
        tensor = padding2DLayer.backPropagation(tensor);
        tensor = conv1.backPropagation(tensor);

        tensor.add(tensorOut);
        return tensor;
    }

    public ConvolutionLayer getConv1() {
        return conv1;
    }

    public void setConv1(ConvolutionLayer conv1) {
        this.conv1 = conv1;
    }

    public ConvolutionLayer getConv2() {
        return conv2;
    }

    public void setConv2(ConvolutionLayer conv2) {
        this.conv2 = conv2;
    }

    public ConvolutionLayer getConv3() {
        return conv3;
    }

    public void setConv3(ConvolutionLayer conv3) {
        this.conv3 = conv3;
    }
}
