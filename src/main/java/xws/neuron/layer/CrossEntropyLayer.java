package xws.neuron.layer;

import xws.neuron.ActivationFunction;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.util.UtilFile;

import java.util.Set;
import java.util.TreeSet;


/**
 * 全连接层,使用交叉熵作为损失函数
 * 初始化时，需要指定有多少个神经元
 * Created by xws on 2019/2/20.
 */
public class CrossEntropyLayer extends Layer {


    private double[] a;//某一层的输出
    private double[] input;//把上一层的输入也保存起来

    private Tensor w;//存放权重信息
    //    private double[][] w;//一维是神经元的数量，二维是每个神经元的权重
    private double[] bias;//每个神经元的偏置
    private double[] z;//每个神经元的z值

    //一下三个变量，在每次计算之前必须清空
    private double[] pdi;//∂C/∂I - I是上一层的输入
    private double[][] pdw;//∂C/∂W
    private double[] pdb;//∂C/∂Z = ∂C/∂B - Z是这一层的输出
    private double[] pda;//∂C/∂A

    //输入数据和输出数据的维度
    private int inputDepth;
    private int inputHeight;
    private int inputWidth;

    //正则化
    private double lambda = 0;


    public CrossEntropyLayer() {

    }

    //初始化神经网络层,num为神经元的数量，inputs为输入的数量
    public CrossEntropyLayer(int num) {
        super("CrossEntropyLayer");

        //初始化每个神经元的权重和偏置
        bias = new double[num];
        a = new double[num];
        z = new double[num];

    }

    public CrossEntropyLayer(String name, String activationType, int num) {
        super("CrossEntropyLayer");
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置
        bias = new double[num];
        a = new double[num];
        z = new double[num];

    }

    public CrossEntropyLayer(String name, String activationType, int num, double lambda) {
        super("CrossEntropyLayer");
        setName(name);
        setActivationType(activationType);
        //初始化每个神经元的权重和偏置
        bias = new double[num];
        a = new double[num];
        z = new double[num];
        this.lambda = lambda;
    }

    private void initW(int inputs) {
        w = new Tensor();
        w.setHeight(bias.length);
        w.setWidth(inputs);
        w.createArray();
        UtilNeuralNet.initWeight(w.getArray());
        UtilNeuralNet.initBias(bias);
    }


    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {


        inputDepth = tensor.getDepth();
        inputHeight = tensor.getHeight();
        inputWidth = tensor.getWidth();

        this.input = tensor.getArray();

        //如果权重为空，则进行权重的初始化
        if (w == null) {
            initW(input.length);
        }

        //如果z为空，则进行初始化
        if (z == null) {
            z = new double[bias.length];
        }

        if (a == null) {
            a = new double[bias.length];
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
            z[i] = 0;//把之前的数据清理掉
            a[i] = 0;


            for (int k = 0; k < w.getWidth(); k++) {
                z[i] = z[i] + w.get(i, k) * input[k];
            }
            z[i] = z[i] + bias[i];
            a[i] = ActivationFunction.activation(z[i], getActivationType());

        }

//        StringBuffer sbz = new StringBuffer();
//        for (int i = 0; i < z.length; i++) {
//            sbz.append(z[i] + "\t");
//        }
//        sbz.append("\n");
//        logZ.append(sbz.toString());

        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(1);
        tensorOut.setHeight(1);
        tensorOut.setWidth(a.length);
        tensorOut.setArray(a);

        return tensorOut;
    }

    //获取这一层神经网络的输出
    @Override
    public Tensor a() {
        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(1);
        tensorOut.setHeight(1);
        tensorOut.setWidth(a.length);
        tensorOut.setArray(a);
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
        pdi = new double[inputDepth * inputHeight * inputWidth];
        pdw = new double[bias.length][w.getWidth()];

        pdi();
        if (!isTest()) {
            pdw();
            pdb();
        }


        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(1);
        tensorOut.setHeight(1);
        tensorOut.setWidth(pdi.length);
        tensorOut.setArray(pdi);

        return tensorOut;
    }

    //误差计算
    @Override
    public Tensor error() {

        pda = new double[bias.length];

        for (int i = 0; i < w.getHeight(); i++) {
            pda[i] = a[i] - getExpect()[i];//使用交叉熵作为损失函数，所以，这里传递的不是pda(l-1),而是pdb(l-1)
        }

        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(1);
        tensorOut.setHeight(1);
        tensorOut.setWidth(pda.length);
        tensorOut.setArray(pda);
        return tensorOut;
    }

    //神经元计算∂C/∂A(l-1)，在这，好汇集所有输入的误差
    public void pdi() {
        for (int i = 0; i < pdi.length; i++) {
            for (int k = 0; k < w.getHeight(); k++) {

                pdi[i] = pdi[i] + pdb[k] * w.get(k, i);
            }
        }
    }

    //神经元计算∂C/∂W
    public void pdw() {
        for (int i = 0; i < w.getHeight(); i++) {
            double[] pdw = this.pdw[i];
            double pdz = this.pdb[i];
            for (int k = 0; k < w.getWidth(); k++) {
                pdw[k] = pdz * input[k];
                double val = w.get(i, k) - getLearnRate() * pdw[k] - lambda * w.get(i, k);
                w.set(i, k, val);
            }
        }
    }

    //神经元计算∂C/∂B =
    public void pdb() {
        for (int i = 0; i < bias.length; i++) {
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

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

}
