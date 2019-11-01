package xws.neuron.layer.output;

import xws.neuron.ActivationFunction;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.Layer;
import xws.util.UtilFile;


/**
 * 均方误差
 * Created by xws on 2019/2/20.
 */
public class MseLayer extends Layer {


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
    //以下三个变量，在每次计算之前必须清空
    private Tensor pdi;//∂C/∂I - I是上一层的输入
    private Tensor pdw;//∂C/∂W
    private Tensor pdb;//∂C/∂Z = ∂C/∂B - Z是这一层的输出
    //输入数据和输出数据的维度
    private int inputDepth;
    private int inputHeight;
    private int inputWidth;
    //正则化
    private double lambda = 0;

    public MseLayer() {

    }

    //初始化神经网络层,num为神经元的数量，inputs为输入的数量
    public MseLayer(int neuralNum) {
        super(MseLayer.class.getSimpleName());

        //初始化每个神经元的权重和偏置
        init(neuralNum);

    }

    public MseLayer(String name, String activationType, int neuralNum) {
        super(MseLayer.class.getSimpleName());
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置

        init(neuralNum);
    }

    public MseLayer(String name, String activationType, int neuralNum, double lambda) {
        super(MseLayer.class.getSimpleName());
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置
        this.lambda = lambda;

        init(neuralNum);
    }

    private void init(int neuralNum) {
        tensorOut = new Tensor();
        tensorOut.setWidth(neuralNum);
        tensorOut.createArray();

        tensorBias = new Tensor();
        tensorBias.setWidth(neuralNum);
        tensorBias.createArray();

        tensorAddBias = new Tensor();
        tensorAddBias.setWidth(neuralNum);
        tensorAddBias.createArray();
    }

    private void initW(int inputs) {
        tensorWeight = new Tensor();
        tensorWeight.setHeight(tensorBias.getWidth());
        tensorWeight.setWidth(inputs);
        tensorWeight.createArray();
        UtilNeuralNet.initWeight(tensorWeight.getArray());
        UtilNeuralNet.initBias(tensorBias.getArray());
    }


    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {
        tensorInput = tensor;
        return tensor;
    }

    //获取这一层神经网络的输出
    @Override
    public Tensor a() {
        return tensorInput;
    }


    /**
     * 反向传播
     * 先计算计算∂C/∂I
     * 再计算∂C/∂W
     * 再计算∂C/∂B
     */
    @Override
    public Tensor backPropagation(Tensor tensor) {
        return null;
    }

    //误差计算 ∂C/∂A
    @Override
    public Tensor error() {


        Tensor pda = new Tensor();
        pda.setWidth(tensorBias.getWidth());
        pda.createArray();

        for (int i = 0; i < tensorWeight.getHeight(); i++) {
            pda.set(i, (tensorOut.get(i) - getExpect().get(i)));//二次损失函数
        }

        return pda;
    }

    //神经元计算∂C/∂A(l-1)，在这，好汇集所有输入的误差
    public void pdi() {
        for (int i = 0; i < pdi.getWidth(); i++) {
            for (int k = 0; k < tensorWeight.getHeight(); k++) {
                pdi.set(i, pdi.get(i) + pdb.get(k) * tensorWeight.get(k, i));
            }
        }
    }

    //神经元计算∂C/∂W
    public void pdw() {
//        for (int i = 0; i < tensorWeight.getHeight(); i++) {
//
//            double[] pdw = this.pdw[i];
//            double pdz = this.pdb.get(i);
//            for (int k = 0; k < tensorWeight.getWidth(); k++) {
//                pdw[k] = pdz * tensorInput.get(k);
//                double val = tensorWeight.get(i, k) - getLearnRate() * pdw[k] - lambda * tensorWeight.get(i, k);
//                tensorWeight.set(i, k, val);
//            }
//        }
    }

    //神经元计算∂C/∂B =
    public void pdb() {
        for (int i = 0; i < tensorBias.getWidth(); i++) {
            tensorBias.set(i, tensorBias.get(i) - getLearnRate() * pdb.get(i));
        }
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
