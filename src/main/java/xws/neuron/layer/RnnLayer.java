package xws.neuron.layer;

import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.util.UtilFile;

import java.util.ArrayList;
import java.util.List;


/**
 * 全连接层转换成循环神经网络
 * 初始化时，需要指定有多少个神经元
 * <p>
 * 输入a(l-1)到z(l)的权重是公共的
 * 输出s(t-1)到s(t)的权重是共享的
 * 应该是一个神经元，一个Tensor，存放权重信息，这样计算的时候，使用Tensor的点乘就可以了
 * 神经元的个数，应该是在初始化layer的时候完成的
 * ResNets并没有残差权重
 * <p>
 * <p>
 * 就拿实际情况来说把，输入20天的数据，再加上换手率，量，那么一天的数据就是6条数据，
 * 隐含层有4个神经元，那么全连接权重=6
 * 上一个时刻，到这一个时刻的权重 = 4
 * skip collection 到这个时刻的权重也是4
 * Created by xws on 2019/5/15.
 */
public class RnnLayer extends Layer {


//    private double[] a;//某一层的输出

    //    private Tensor w;//这次的权重
    private int num;//神经元的个数

    private List<RnnTime> rnnTimes;//存放每一个序列的信息

    private List<Tensor> shareListInputWeight;//这次的权重
    private List<Tensor> shareListInputPreviousWeight;//上次的权重
    private List<Tensor> shareListInputResidualWeight;//邻残差权重

    private List<Tensor> shareListErrorInputWeight;//输入权重误差
    private List<Tensor> shareListErrorInputPreviousWeight;//上一时刻输入权重误差
    private List<Tensor> shareListErrorInputResidualWeight;//残差输入权重误差
    private Tensor shareErrorBias;//偏置的误差

    private boolean initFlag = false;

    private Tensor shareBias;//每个神经元的偏置

    private int step = 0;//当前序列

    private Tensor tensorOut;

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

    public RnnLayer() {

    }

    //初始化神经网络层,num为神经元的数量，inputs为输入的数量
    public RnnLayer(int num) {
        super("RnnLayer");
        //初始化神经元的个数
        this.num = num;
        init();

    }

    public RnnLayer(String name, String activationType, int num) {
        super("RnnLayer");
        this.num = num;
        setName(name);
        setActivationType(activationType);
        //初始化神经元的个数
        init();


    }

    public RnnLayer(String name, String activationType, int num, double lambda) {
        super("RnnLayer");
        this.num = num;
        setName(name);
        setActivationType(activationType);
        //初始化神经元的个数
        init();
        this.lambda = lambda;
    }


    //神经元的个数，应该是在初始化layer的时候完成的
    private void init() {
        shareBias = new Tensor();
        shareBias.setDepth(1);
        shareBias.setHeight(1);
        shareBias.setWidth(num);
        shareBias.createArray();
        UtilNeuralNet.initBias(shareBias.getArray());
        //初始化每个神经元的权重和偏置
        shareListInputWeight = new ArrayList<>();
        shareListInputPreviousWeight = new ArrayList<>();
        shareListInputResidualWeight = new ArrayList<>();
        for (int i = 0; i < num; i++) {
            shareListInputWeight.add(new Tensor());
        }

        for (int i = 0; i < num; i++) {
            shareListInputPreviousWeight.add(new Tensor());
        }

        for (int i = 0; i < num; i++) {
            shareListInputResidualWeight.add(new Tensor());
        }

        //根据神经元的数量，初始化上一次的权重信息
        initWeightList(shareListInputPreviousWeight);
        //根据神经元的数量，初始化邻残差的权重信息
        initWeightList(shareListInputResidualWeight);

    }

    //初始化权重信息
    private void initWeight(int inputs) {


        //初始化权重list，有多少个输入，那么，就有多少个神经元
        for (int i = 0; i < shareListInputWeight.size(); i++) {
            Tensor tensor = shareListInputWeight.get(i);
            tensor.setDepth(1);
            tensor.setHeight(1);
            tensor.setWidth(inputs);
            tensor.createArray();
            UtilNeuralNet.initWeight(tensor.getArray());
        }
    }


    private void initWeightList(List<Tensor> list) {
        for (int i = 0; i < list.size(); i++) {
            Tensor tensor = list.get(i);
            tensor.setDepth(1);
            tensor.setHeight(1);
            tensor.setWidth(shareBias.getWidth());
            tensor.createArray();
            UtilNeuralNet.initWeight(tensor.getArray());
        }
    }


    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {


        //判断权重是否初始化
        if (initFlag == false) {
            initWeight(tensor.getWidth());
            initFlag = true;
        }

        //如果step==0，那么，进行一下初始化工作
        if (step == 0) {
            rnnTimes = new ArrayList<>();

            //初始化共享误差
            shareErrorBias = new Tensor();
            shareErrorBias.setWidth(shareBias.getWidth());
            shareErrorBias.createArray();

            //有多少个神经元，就有多少个Tensor,Tensor的width就是权重和输入的数量
            shareListErrorInputWeight = new ArrayList<>();
            shareListErrorInputPreviousWeight = new ArrayList<>();
            shareListErrorInputResidualWeight = new ArrayList<>();
            for (int i = 0; i < shareBias.getWidth(); i++) {
                Tensor w = new Tensor();
                w.setWidth(num);
                w.createArray();
                shareListErrorInputWeight.add(w);

                Tensor p = new Tensor();
                p.setWidth(shareBias.getWidth());
                p.createArray();
                shareListErrorInputPreviousWeight.add(p);

                Tensor r = new Tensor();
                r.setWidth(shareBias.getWidth());
                r.createArray();
                shareListErrorInputResidualWeight.add(r);

            }

        }


        //创建RnnTime,一下训练完成后，重置step=0
        RnnTime rnnTime = new RnnTime();
        rnnTime.setStep(step);
        rnnTime.setShareBias(shareBias);
        rnnTime.setLayer(this);
        rnnTime.setRnnTimes(rnnTimes);
        rnnTime.setShareListInputWeight(shareListInputWeight);
        rnnTime.setShareListInputPreviousWeight(shareListInputPreviousWeight);
        rnnTime.setShareListInputResidualWeight(shareListInputResidualWeight);

        rnnTime.setShareErrorBias(shareErrorBias);
        rnnTime.setShareListErrorInputWeight(shareListErrorInputWeight);
        rnnTime.setShareListErrorInputPreviousWeight(shareListErrorInputPreviousWeight);
        rnnTime.setShareListErrorInputResidualWeight(shareListErrorInputResidualWeight);


        rnnTime.setTensorInput(tensor);
        if (step > 0) {
            rnnTime.setTensorInputPrevious(rnnTimes.get(step - 1).getA());
        }

        if (step - 2 >= 0) {
            rnnTime.setTensorInputResidual(rnnTimes.get(step - 2).getA());

        }
        rnnTimes.add(rnnTime);

        //输出
        tensorOut = rnnTime.forward();
        step = step + 1;
        return tensorOut;
    }

    //获取这一层神经网络的输出
    @Override
    public Tensor a() {
        return tensorOut;
    }


    /**
     * 反向传播,误差是针对每个神经元的
     * 先计算计算∂C/∂I
     * 再计算∂C/∂W
     * 再计算∂C/∂B
     */
    @Override
    public Tensor backPropagation(Tensor tensor) {

        //将误差累加到最后一个时刻
        RnnTime lastRnnTime = rnnTimes.get(rnnTimes.size() - 1);
        lastRnnTime.getTensorErrorOut().add(tensor);

        //进行循环，从最后一个RnnTime开始进行反向传播
        for (int i = rnnTimes.size() - 1; i >= 0; i--) {
            RnnTime rnnTime = rnnTimes.get(i);
            rnnTime.backPropagation();
        }

        //更新权重和误差
        updateError(shareListInputWeight, shareListErrorInputWeight);
        updateError(shareListInputPreviousWeight, shareListErrorInputPreviousWeight);
        updateError(shareListInputResidualWeight, shareListErrorInputResidualWeight);
        updateErrorBias();

        return lastRnnTime.getTensorInputError();
    }

    //更新误差
    private void updateError(List<Tensor> shareWeight, List<Tensor> shareError) {
        for (int i = 0; i < shareWeight.size(); i++) {
            Tensor tensorW = shareWeight.get(i);
            Tensor tensorE = shareError.get(i);

            for (int n = 0; n < tensorW.getWidth(); n++) {
                double w = tensorW.get(n);
                double e = tensorE.get(n);
                double v = w - getLearnRate() * e - lambda * w;
                tensorW.set(n, v);
            }

        }
    }


    //神经元计算∂C/∂B =
    public void updateErrorBias() {
        for (int i = 0; i < shareBias.getWidth(); i++) {
            double b = shareBias.get(i) - getLearnRate() * shareErrorBias.get(i);
            shareBias.set(i, b);
        }
    }


    public Tensor getShareBias() {
        return shareBias;
    }

    public void setShareBias(Tensor shareBias) {
        this.shareBias = shareBias;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    private void initFile() {
        if (logA == null) {
            logA = new UtilFile("/Users/xws/Desktop/xws/log/" + getName() + ".a.csv");
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

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public int getNum() {
        return num;
    }

    public void setNum(int num) {
        this.num = num;
    }

    public List<Tensor> getShareListInputWeight() {
        return shareListInputWeight;
    }

    public void setShareListInputWeight(List<Tensor> shareListInputWeight) {
        this.shareListInputWeight = shareListInputWeight;
    }

    public List<Tensor> getShareListInputPreviousWeight() {
        return shareListInputPreviousWeight;
    }

    public void setShareListInputPreviousWeight(List<Tensor> shareListInputPreviousWeight) {
        this.shareListInputPreviousWeight = shareListInputPreviousWeight;
    }

    public List<Tensor> getShareListInputResidualWeight() {
        return shareListInputResidualWeight;
    }

    public void setShareListInputResidualWeight(List<Tensor> shareListInputResidualWeight) {
        this.shareListInputResidualWeight = shareListInputResidualWeight;
    }

    public boolean isInitFlag() {
        return initFlag;
    }

    public void setInitFlag(boolean initFlag) {
        this.initFlag = initFlag;
    }
}
