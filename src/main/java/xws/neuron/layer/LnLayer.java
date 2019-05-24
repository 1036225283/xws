package xws.neuron.layer;

import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;


/**
 * LN 归一化
 * Created by xws on 2019/2/20.
 */
public class LnLayer extends Layer {


    private double standard;

    public LnLayer() {
        super("LnLayer");
    }

    public LnLayer(String name) {
        super("LnLayer");
        setName(name);
    }

    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {

        double[] array = tensor.getArray();

        //计算均值
        double average = UtilNeuralNet.average(array);

        //计算方差
        double variance = UtilNeuralNet.variance(array, average);

        //分母
        standard = Math.sqrt(variance + 0.000000000001);

        //归一化
        for (int i = 0; i < array.length; i++) {
            array[i] = (array[i] - average) / standard;
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
            array[i] = array[i] * (1 / standard);
        }

        return tensor;
    }

    //误差计算
    @Override
    public Tensor error() {
        return null;
    }

}
