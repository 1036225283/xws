package xws.neuron.layer;

import xws.neuron.ActivationFunction;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.util.UtilFile;


/**
 * 全连接层,使用交叉熵作为损失函数
 * 初始化时，需要指定有多少个神经元
 * Created by xws on 2019/2/20.
 */
public class CrossEntropyLayer extends Layer {


    private Tensor tensorOut;//某一层的输出
    private Tensor tensorInput;//把上一层的输入也保存起来

    private Tensor w;//存放权重信息 一维是神经元的数量，二维是每个神经元的权重
    private Tensor bias;//每个神经元的偏置
    private Tensor z;//每个神经元的z值

    //一下三个变量，在每次计算之前必须清空
    private Tensor pdi;//∂C/∂I - I是上一层的输入
    private double[][] pdw;//∂C/∂W
    private double[] pdb;//∂C/∂Z = ∂C/∂B - Z是这一层的输出

    //输入数据和输出数据的维度
    private int inputDepth;
    private int inputHeight;
    private int inputWidth;

    //正则化
    private double lambda = 0;


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

    public CrossEntropyLayer() {

    }

    //初始化神经网络层,num为神经元的数量，inputs为输入的数量
    public CrossEntropyLayer(int num) {
        super("CrossEntropyLayer");

        //初始化每个神经元的权重和偏置
        init(num);


    }

    public CrossEntropyLayer(String name, String activationType, int num) {
        super("CrossEntropyLayer");
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置
        init(num);

    }

    public CrossEntropyLayer(String name, String activationType, int num, double lambda) {
        super("CrossEntropyLayer");
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置
        this.lambda = lambda;

        init(num);
    }

    private void initW(int inputs) {
        w = new Tensor();
        w.setHeight(bias.getWidth());
        w.setWidth(inputs);
        w.createArray();
        UtilNeuralNet.initWeight(w.getArray());
        UtilNeuralNet.initBias(bias.getArray());
    }

    private void init(int num) {
        tensorOut = new Tensor();
        tensorOut.setWidth(num);
        tensorOut.createArray();

        bias = new Tensor();
        bias.setWidth(num);
        bias.createArray();

        z = new Tensor();
        z.setWidth(num);
        z.createArray();
    }


    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {

        initFile();


        inputDepth = tensor.getDepth();
        inputHeight = tensor.getHeight();
        inputWidth = tensor.getWidth();

        this.tensorInput = tensor;

        //如果权重为空，则进行权重的初始化
        if (w == null) {
            initW(tensorInput.getWidth());
        }

        //如果z为空，则进行初始化
        if (z == null) {
            z = new Tensor();
            z.setWidth(bias.getWidth());
            z.createArray();
        }

        if (tensorOut == null) {
            tensorOut = new Tensor();
            tensorOut.setWidth(bias.getWidth());
            tensorOut.createArray();
        }


        //输出w
//        logW.append(w.toString());
//        输出b
//        StringBuffer sb = new StringBuffer();
//        for (int i = 0; i < bias.length; i++) {
//            sb.append(bias[i] + "\t");
//        }
//        sb.append("\n");
//        logB.append(sb.toString());

        for (int i = 0; i < w.getHeight(); i++) {
            //计算神经元的输出
            z.set(i, 0);
            tensorOut.set(i, 0);


            for (int k = 0; k < w.getWidth(); k++) {
                z.set(i, z.get(i) + w.get(i, k) * tensorInput.get(k));
            }
            z.set(i, z.get(i) + bias.get(i));
            tensorOut.set(i, ActivationFunction.activation(z.get(i), getActivationType()));


        }

//        StringBuffer sbz = new StringBuffer();
//        for (int i = 0; i < z.length; i++) {
//            sbz.append(z[i] + "\t");
//        }
//        sbz.append("\n");
//        logZ.append(sbz.toString());

//        logA.append(tensorOut.toString());
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

        //取出误差∂C/∂B
        pdb = tensor.getArray();
//        logE.append(tensor.toString());

        //inputs决定了有多少个输入，也就是这一层的神经元会有多少个w
        pdi = new Tensor();
        pdi.setDepth(inputDepth);
        pdi.setHeight(inputHeight);
        pdi.setWidth(inputWidth);
        pdi.createArray();

        pdw = new double[bias.getWidth()][w.getWidth()];

        pdi();
        if (!isTest()) {
            pdw();
            pdb();
        }
        return pdi;
    }

    //误差计算 ∂C/∂A
    @Override
    public Tensor error() {

        Tensor pda = new Tensor();
        pda.setWidth(bias.getWidth());
        pda.createArray();

        for (int i = 0; i < w.getHeight(); i++) {
            pda.set(i, tensorOut.get(i) - getExpect()[i]);//使用交叉熵作为损失函数，所以，这里传递的不是pda(l-1),而是pdb(l-1)
        }

        return pda;
    }

    //神经元计算∂C/∂A(l-1)，在这，好汇集所有输入的误差
    public void pdi() {
        for (int i = 0; i < pdi.getWidth(); i++) {
            for (int k = 0; k < w.getHeight(); k++) {
                pdi.set(i, pdi.get(i) + pdb[k] * w.get(k, i));
            }
        }
    }

    //神经元计算∂C/∂W
    public void pdw() {
        for (int i = 0; i < w.getHeight(); i++) {
            double[] pdw = this.pdw[i];
            double pdz = this.pdb[i];
            for (int k = 0; k < w.getWidth(); k++) {
                pdw[k] = pdz * tensorInput.get(k);
                double val = w.get(i, k) - getLearnRate() * pdw[k] - lambda * w.get(i, k);
                w.set(i, k, val);
            }
        }
    }

    //神经元计算∂C/∂B =
    public void pdb() {
        for (int i = 0; i < bias.getWidth(); i++) {
            bias.set(i, bias.get(i) - getLearnRate() * pdb[i]);
        }
    }

    public Tensor getW() {
        return w;
    }

    public void setW(Tensor w) {
        this.w = w;
    }

    public Tensor getBias() {
        return bias;
    }

    public void setBias(Tensor bias) {
        this.bias = bias;
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
            logW = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".w.csv");
        }

        if (logE == null) {
            logE = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".e.csv");
        }

        if (logZ == null) {
            logZ = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".z.csv");
        }

    }
}
