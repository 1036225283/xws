package xws.neuron.layer.resnet;

import xws.neuron.Tensor;
import xws.neuron.layer.Layer;
import xws.neuron.layer.conv.ConvolutionLayer;


/**
 * resNet
 * Created by xws on 2020/10/14.
 */
public class BottleneckConv1DLayer extends Layer {


    private Tensor tensorInput;
    private Tensor tensorOut;


    ConvolutionLayer conv1 = new ConvolutionLayer();
    ConvolutionLayer conv2 = new ConvolutionLayer();
    ConvolutionLayer conv3 = new ConvolutionLayer();


    public BottleneckConv1DLayer() {
    }

    public BottleneckConv1DLayer(int padding) {
        super(BottleneckConv1DLayer.class.getSimpleName());
    }

    public BottleneckConv1DLayer(String name, int padding) {
        super(BottleneckConv1DLayer.class.getSimpleName());
        setName(name);
    }

    //构造函数时，传入filters的构造
    public BottleneckConv1DLayer(int pTop, int pBottom, int pLeft, int pRight) {
        super(BottleneckConv1DLayer.class.getSimpleName());
    }

    public BottleneckConv1DLayer(String name, int pTop, int pBottom, int pLeft, int pRight) {
        super(BottleneckConv1DLayer.class.getSimpleName());
        setName(name);

    }

    @Override
    public Tensor forward(Tensor tensor) {

        tensorInput = tensor;
        Tensor tensorOutConv1 = conv1.forward(tensorInput);
        Tensor tensorOutConv2 = conv1.forward(tensorOutConv1);
        Tensor tensorOutConv3 = conv1.forward(tensorOutConv2);

        return tensorOutConv3;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {

        Tensor tensorErrorConv3 = conv3.backPropagation(tensor);
        Tensor tensorErrorConv2 = conv2.backPropagation(tensorErrorConv3);
        Tensor tensorErrorConv1 = conv1.backPropagation(tensorErrorConv2);

        tensorErrorConv1.add(tensor);
        return tensorErrorConv1;
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
