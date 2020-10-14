package xws.neuron.layer.resnet;

import xws.neuron.Tensor;
import xws.neuron.layer.FullLayer;
import xws.neuron.layer.Layer;
import xws.neuron.layer.conv.ConvolutionLayer;


/**
 * resNet
 * Created by xws on 2020/10/14.
 */
public class BottleneckFullLayer extends Layer {


    private Tensor tensorInput;
    private Tensor tensorOut;


    FullLayer conv1 = new FullLayer();
    FullLayer conv2 = new FullLayer();
    FullLayer conv3 = new FullLayer();


    public BottleneckFullLayer() {
    }

    public BottleneckFullLayer(int padding) {
        super(BottleneckFullLayer.class.getSimpleName());
    }

    public BottleneckFullLayer(String name, int padding) {
        super(BottleneckFullLayer.class.getSimpleName());
        setName(name);
    }

    //构造函数时，传入filters的构造
    public BottleneckFullLayer(int pTop, int pBottom, int pLeft, int pRight) {
        super(BottleneckFullLayer.class.getSimpleName());
    }

    public BottleneckFullLayer(String name, int pTop, int pBottom, int pLeft, int pRight) {
        super(BottleneckFullLayer.class.getSimpleName());
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


}
