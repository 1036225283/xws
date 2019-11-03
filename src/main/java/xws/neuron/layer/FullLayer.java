package xws.neuron.layer;

import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.util.UtilFile;


/**
 * 全连接层,没有激活函数
 * 初始化时，需要指定有多少个神经元
 * Created by xws on 2019/2/20.
 */
public class FullLayer extends Layer {


    //查看w的变化
    UtilFile logW;
    //查看b的变化
    UtilFile logB;
    //查看a的变化
    UtilFile logA;
    //查看e的变化
    UtilFile logE;
    //查看z的变化
    UtilFile logZ;
    private Tensor tensorOut;//某一层的输出
    private Tensor tensorInput;//把上一层的输入也保存起来
    private Tensor tensorInputMultiplyWeight;//input * tensorWeight
    private Tensor tensorWeight;//存放权重信息
    private Tensor tensorBias;//每个神经元的偏置
    private Tensor tensorAddBias;//input * tensorWeight + tensorBias
    //正则化
    private double lambda = 0;
    //神经元的个数
    private int neuralNum;

    public FullLayer() {

    }

    //初始化神经网络层,num为神经元的数量，inputs为输入的数量
    public FullLayer(int neuralNum) {
        super(FullLayer.class.getSimpleName());
        this.neuralNum = neuralNum;
        //初始化每个神经元的权重和偏置
        init(neuralNum);

    }

    public FullLayer(String name, int neuralNum) {
        super(FullLayer.class.getSimpleName());
        this.neuralNum = neuralNum;

        setName(name);
        //初始化每个神经元的权重和偏置

        init(neuralNum);
    }

    public FullLayer(String name, int neuralNum, double lambda) {
        super(FullLayer.class.getSimpleName());
        this.neuralNum = neuralNum;
        setName(name);
        //初始化每个神经元的权重和偏置
        this.lambda = lambda;

        init(neuralNum);
    }

    private void init(int neuralNum) {
        //init bias
        tensorBias = new Tensor();
        tensorBias.setWidth(neuralNum);
        tensorBias.createArray();
    }

    private void initW(int inputs) {
        if (tensorWeight == null) {
            tensorWeight = new Tensor();
            tensorWeight.setHeight(neuralNum);
            tensorWeight.setWidth(inputs);
            tensorWeight.createArray();
            UtilNeuralNet.initWeight(tensorWeight.getArray());
            UtilNeuralNet.initBias(tensorBias.getArray());
        }
    }


    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {

        this.tensorInput = tensor;

        initW(tensor.size());
        tensorInputMultiplyWeight = tensorInput.multiplyW(tensorWeight);
        tensorOut = tensorInputMultiplyWeight.add(tensorBias);

        return tensorOut;
    }

    //获取这一层神经网络的输出
    @Override
    public Tensor a() {
        return tensorOut;
    }


    /**
     * 反向传播
     * 先计算计算∂C/∂I
     * 再计算∂C/∂W
     * 再计算∂C/∂B
     */
    @Override
    public Tensor backPropagation(Tensor tensor) {
        Tensor tensorInputPartialDerivative = tensorWeight.calculateInputPartialDerivative(tensor);
        if (!isTest()) {
            // update bias
            tensorBias.updateBias(tensor, getLearnRate());
            // update weight
            Tensor tensorWeightPartialDerivative = tensorWeight.calculateWeightPartialDerivative(tensor, tensorInput);
            tensorWeight.updateWeight(tensorWeightPartialDerivative, getLearnRate());
        }
        return tensorInputPartialDerivative;
    }

    //误差计算 ∂C/∂A
    @Override
    public Tensor error() {
        return null;
    }

    public Tensor getTensorWeight() {
        return tensorWeight;
    }

    public void setTensorWeight(Tensor tensorWeight) {
        this.tensorWeight = tensorWeight;
    }

    public Tensor getTensorBias() {
        return tensorBias;
    }

    public void setTensorBias(Tensor tensorBias) {
        this.tensorBias = tensorBias;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    private void initFile() {
        if (logA == null) {
            logA = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".tensorOut.csv");
        }

        if (logB == null) {
            logB = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".b.csv");
        }

        if (logW == null) {
            logW = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".tensorWeight.csv");
        }

        if (logE == null) {
            logE = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".e.csv");
        }

        if (logZ == null) {
            logZ = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".tensorAddBias.csv");
        }

    }
}
