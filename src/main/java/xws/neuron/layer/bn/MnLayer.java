package xws.neuron.layer.bn;

import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.Layer;


/**
 * 最大化 layer 归一化
 * Created by xws on 2019/2/20.
 */
public class MnLayer extends Layer {


    private double max;

    public MnLayer() {
        super("MnLayer");
    }

    public MnLayer(String name) {
        super("MnLayer");
        setName(name);
    }

    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {

        double[] array = tensor.getArray();
        max = UtilNeuralNet.max(array);
        for (int i = 0; i < array.length; i++) {
            array[i] = array[i] / max;
        }

        return tensor;
    }

    //获取这一层神经网络的输出
    @Override
    public Tensor a() {
        return null;
    }


    /**
     * 反向传播
     */
    @Override
    public Tensor backPropagation(Tensor tensor) {

        double[] array = tensor.getArray();
        for (int i = 0; i < array.length; i++) {
            array[i] = array[i] * (1 / max);
        }

        return tensor;
    }

    //误差计算
    @Override
    public Tensor error() {
        return null;
    }

    public double getMax() {
        return max;
    }

    public void setMax(double max) {
        this.max = max;
    }
}
