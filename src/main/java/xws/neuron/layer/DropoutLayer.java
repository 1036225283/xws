package xws.neuron.layer;

import xws.neuron.ActivationFunction;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.util.UtilFile;

import java.util.Set;
import java.util.TreeSet;


/**
 * 全连接层
 * 初始化时，需要指定有多少个神经元
 * 先放一放，有点问题！！！
 * Created by xws on 2019/2/20.
 */
public class DropoutLayer extends Layer {


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
    private Tensor w;//存放权重信息
    //    private double[][] w;//一维是神经元的数量，二维是每个神经元的权重
    private double[] bias;//每个神经元的偏置
    private double[] z;//每个神经元的z值
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
    private double dropoutRate = 0;//默认不丢任何权重0/0.5
    private Set<Integer> dropouts;

    public DropoutLayer() {

    }

    //初始化神经网络层,num为神经元的数量，inputs为输入的数量
    public DropoutLayer(int num) {
        super(PaddingLayer.class.getSimpleName());

        //初始化每个神经元的权重和偏置
        bias = new double[num];
        z = new double[num];

        init(num);

    }

    public DropoutLayer(String name, String activationType, int num) {
        super(PaddingLayer.class.getSimpleName());
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置
        bias = new double[num];
        z = new double[num];
        init(num);

    }

    public DropoutLayer(String name, String activationType, int num, double dropoutRate) {
        super(PaddingLayer.class.getSimpleName());
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置
        bias = new double[num];
        z = new double[num];
        this.dropoutRate = dropoutRate;
        init(num);
    }

    public DropoutLayer(String name, String activationType, int num, double dropoutRate, double lambda) {
        super(PaddingLayer.class.getSimpleName());
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置
        bias = new double[num];
        z = new double[num];
        this.dropoutRate = dropoutRate;
        this.lambda = lambda;

        init(num);
    }

    private void init(int num) {
        tensorOut = new Tensor();
        tensorOut.setWidth(num);
        tensorOut.createArray();
    }

    private void initW(int inputs) {
        w = new Tensor();
        w.setHeight(bias.length);
        w.setWidth(inputs);
        w.createArray();
        UtilNeuralNet.initWeight(w.getArray());
        UtilNeuralNet.initBias(bias);
    }

    //丢弃部分权重，丢弃比例为dropoutRate
    public void dropout() {
        //如果属于同一批次，丢弃的神经元还是上次丢弃的那批
        if (getPrevBatch() == getBatch()) {
            return;
        }

        //获取总的神经元个数
        int length = w.getHeight();

        //计算应该丢弃的神经元个数
        int dropoutNum = (int) (length * dropoutRate);
        dropouts = new TreeSet<>();
        //产生随机数
        while (dropouts.size() < dropoutNum) {
            int randomIndex = (int) (Math.random() * length);
            dropouts.add(randomIndex);
        }
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
            z = new double[bias.length];
        }

        if (tensorOut == null) {
            tensorOut = new Tensor();
            tensorOut.setWidth(bias.length);
            tensorOut.createArray();
        }
        tensorOut.zero();

        //使用神经元丢弃技术
        if (!isTest()) {
            dropout();
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
        //将权重进行处理
//        if (isTest()) {
//            double[] array = w.getArray();
//            for (int i = 0; i < array.length; i++) {
//                array[i] = array[i] * dropoutRate;
//            }
//        }

        for (int i = 0; i < w.getHeight(); i++) {
            //计算神经元的输出
            z[i] = 0;//把之前的数据清理掉
            if (!isTest()) {
                if (dropouts.contains(i)) {
                    continue;
                }
            }

            for (int k = 0; k < w.getWidth(); k++) {
                z[i] = z[i] + w.get(i, k) * tensorInput.get(k);
            }
            z[i] = z[i] + bias[i];
            tensorOut.set(i, ActivationFunction.activation(z[i], getActivationType()));

//            if (isTest() && dropoutRate != 0) {
//                tensorOut[i] = tensorOut[i] * dropoutRate;
//            }
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

        //取出误差∂C/∂A
        double[] pda = tensor.getArray();
//        logE.append(tensor.toString());

        //先把pdb清空
        pdb = new double[bias.length];
        //inputs决定了有多少个输入，也就是这一层的神经元会有多少个w
        pdi = new Tensor();
        pdi.setWidth(inputDepth * inputHeight * inputWidth);
        pdi.createArray();

        pdw = new double[bias.length][w.getWidth()];

        //计算pdz = ∂C/∂A * ∂A/∂Z
        for (int i = 0; i < pda.length; i++) {
            if (!isTest()) {
                if (dropouts.contains(i)) {
                    pdb[i] = 0;
                    continue;
                }
            }
            pdb[i] = pda[i] * ActivationFunction.derivation(z[i], getActivationType());
        }

        pdi();
        if (!isTest()) {
            pdw();
            pdb();
        }

        return pdi;
    }

    //误差计算
    @Override
    public Tensor error() {

        Tensor pda = new Tensor();
        pda.setWidth(bias.length);
        pda.createArray();

        for (int i = 0; i < w.getHeight(); i++) {
            pda.set(i, (tensorOut.get(i) - getExpect().get(i)) * getGamma().get(i));
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
            if (!isTest()) {
                if (dropouts.contains(i)) {
                    continue;
                }
            }
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
        for (int i = 0; i < bias.length; i++) {
            if (!isTest()) {
                if (dropouts.contains(i)) {
                    continue;
                }
            }
            bias[i] = bias[i] - getLearnRate() * pdb[i];
        }
    }

    public Tensor getW() {
        return w;
    }

    public void setW(Tensor w) {
        this.w = w;
    }

    public double[] getBias() {
        return bias;
    }

    public void setBias(double[] bias) {
        this.bias = bias;
    }

    public double getDropoutRate() {
        return dropoutRate;
    }

    public void setDropoutRate(double dropoutRate) {
        this.dropoutRate = dropoutRate;
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
