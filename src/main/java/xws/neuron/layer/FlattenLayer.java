package xws.neuron.layer;

import xws.neuron.Tensor;


/**
 * 将多维数据压成1维
 * Created by xws on 2019/2/19.
 */
public class FlattenLayer extends Layer {


    private int inputDepth;
    private int inputHeight;
    private int inputWidth;


    public FlattenLayer() {
        super(FlattenLayer.class.getSimpleName());
    }

    @Override
    public Tensor forward(Tensor tensor) {
        inputDepth = tensor.getDepth();
        inputHeight = tensor.getHeight();
        inputWidth = tensor.getWidth();

        tensor.setWidth(tensor.getDepth() * tensor.getHeight() * tensor.getWidth());
        tensor.setDepth(1);
        tensor.setHeight(1);
        return tensor;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {
        tensor.setDepth(inputDepth);
        tensor.setHeight(inputHeight);
        tensor.setWidth(inputWidth);
        return tensor;
    }


}
