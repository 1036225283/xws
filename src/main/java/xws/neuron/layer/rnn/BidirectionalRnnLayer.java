package xws.neuron.layer.rnn;

import xws.neuron.Tensor;
import xws.neuron.layer.Layer;

import java.util.ArrayList;
import java.util.List;

public class BidirectionalRnnLayer extends Layer {

    private int num;//神经元的个数
    private double lambda = 0;

    private RnnLayer forwardRnn;
    private RnnLayer backwardRnn;
    private List<Tensor> listInput = new ArrayList<>();
    private Tensor tensorInput;
    private Tensor tensorOut;

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
        forwardRnn = new RnnLayer("", getActivationType(), num, lambda);
        backwardRnn = new RnnLayer("", getActivationType(), num, lambda);
    }

    @Override
    public Tensor forward(Tensor tensor) {
        tensorInput = tensor;
        // 每次forward的时候，都需要重新运行一次反向循环神经网络，拿到最新输出
        if (getStep() == 0) {
            forwardRnn.setStep(0);
            // 将输入清空，同时传入最新的
            listInput = new ArrayList<>();
        }
        listInput.add(tensor);

        Tensor tensorBackwardOut = null;
        backwardRnn.setStep(0);
        // 获取反向RNN的输出
        for (int i = listInput.size() - 1; i >= 0; i--) {
            Tensor tensorInput = listInput.get(i);
            tensorBackwardOut = backwardRnn.forward(tensorInput);
        }
        Tensor tensorOut = forwardRnn.forward(tensor);
        this.tensorOut = tensorOut;
        setStep(getStep() + 1);
        Tensor tmp = tensorOut.concat(tensorBackwardOut);
        return tmp;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {
        // 首先，将tensor切割成两个tensor
        Tensor tensorDiffForward = tensorOut.copy();
        System.arraycopy(tensor.getArray(), 0, tensorDiffForward.getArray(), 0, tensorDiffForward.getArray().length);
        Tensor tensorDiffBackward = tensorOut.copy();
        System.arraycopy(tensor.getArray(), tensorDiffBackward.getArray().length, tensorDiffBackward.getArray(), 0, tensorDiffForward.getArray().length);

        Tensor tensorErrForward = forwardRnn.backPropagation(tensorDiffForward);
        Tensor tensorErrBackward = backwardRnn.backPropagation(tensorDiffBackward);
        tensorErrForward.add(tensorErrBackward);

        return tensorErrForward.divide(2);
    }

    public int getNum() {
        return num;
    }

    public void setNum(int num) {
        this.num = num;
    }


}
