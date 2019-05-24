package xws.neuron.layer;

import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;


/**
 * AN 归一化,每一个输入都有一个alpha
 * Created by xws on 2019/2/20.
 */
public class AnLayer extends Layer {


    private double average;//均值
    private double variance;//方差
    private double standard;//标准差
    private Double beta = null;//ß
    private Double alpha = null;//α
    private Tensor tensorInput;


    public AnLayer() {
        super("AnLayer");
    }

    public AnLayer(String name) {
        super("AnLayer");
        setName(name);
    }


    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {
        tensorInput = tensor;

        if (alpha == null) {
            beta = 0.0010000000474974513;
            alpha = UtilNeuralNet.initData();
        }

        double[] array = tensor.getArray();

        //计算均值
        average = UtilNeuralNet.average(array);

        //计算方差
        variance = UtilNeuralNet.variance(array, average);

        //标准差
        standard = Math.sqrt(variance + 0.000000000001);

        Tensor tensorOut = tensorInput.copy();
        array = tensorOut.getArray();
        //反向传播时，需要用来计算误差
        for (int i = 0; i < array.length; i++) {
            array[i] = (array[i] - average) / standard * alpha + beta;
        }

        return tensorOut;
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
        Tensor tensorError = tensorInput.copy();

        double[] array = tensor.getArray();
        double[] arrayInput = tensorInput.getArray();
        double[] arrayError = tensorError.getArray();


        //先计算输入误差
        for (int i = 0; i < arrayInput.length; i++) {
            arrayError[i] = array[i] / standard * alpha;
        }


        //计算beta误差
        double total = 0;
        for (int i = 0; i < array.length; i++) {
            total = total + array[i];
        }

        //更新beta误差
        beta = beta - getLearnRate() * total;

        //计算alpha误差
        total = 0;
        for (int i = 0; i < arrayInput.length; i++) {
            total = total + (arrayInput[i] - average) / standard * array[i];
        }

        //更新alpha误差
        alpha = alpha - getLearnRate() * total;

        return tensorError;
    }

    //误差计算
    @Override
    public Tensor error() {
        return null;
    }

    public Double getBeta() {
        return beta;
    }

    public void setBeta(Double beta) {
        this.beta = beta;
    }

    public Double getAlpha() {
        return alpha;
    }

    public void setAlpha(Double alpha) {
        this.alpha = alpha;
    }
}
