package xws.neuron.layer.rnn;

import xws.neuron.layer.Layer;

public class BidirectionalRnnLayer extends Layer {

    private int num;//神经元的个数
    private double lambda = 0;
    private String activationType;

    private RnnLayer forwardRnn;
    private RnnLayer backwardRnn;

    public BidirectionalRnnLayer() {
        super(BidirectionalRnnLayer.class.getSimpleName());
    }

    //初始化神经网络层,num为神经元的数量，inputs为输入的数量
    public BidirectionalRnnLayer(int num) {
        super(BidirectionalRnnLayer.class.getSimpleName());
        //初始化神经元的个数
        this.num = num;
        init();

    }

    public BidirectionalRnnLayer(String name, String activationType, int num) {
        super(BidirectionalRnnLayer.class.getSimpleName());
        this.num = num;
        setName(name);
        setActivationType(activationType);
        //初始化神经元的个数
        init();


    }

    public BidirectionalRnnLayer(String name, String activationType, int num, double lambda) {
        super(BidirectionalRnnLayer.class.getSimpleName());
        this.num = num;
        setName(name);
        setActivationType(activationType);
        //初始化神经元的个数
        init();
        this.lambda = lambda;
    }

    private void init() {

    }

    public int getNum() {
        return num;
    }

    public void setNum(int num) {
        this.num = num;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }


}
