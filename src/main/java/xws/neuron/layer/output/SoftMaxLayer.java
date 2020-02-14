package xws.neuron.layer.output;

import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.Layer;
import xws.util.UtilFile;


/**
 * SoftMax
 * 初始化时，需要指定有多少个神经元
 * Created by xws on 2019/2/20.
 */
public class SoftMaxLayer extends Layer {


    private Tensor tensorOut;//某一层的输出
    private Tensor tensorInput;//把上一层的输入也保存起来

    public SoftMaxLayer() {
        super("SoftMaxLayer");
    }

    public SoftMaxLayer(String name) {
        super("SoftMaxLayer");
        setName(name);
    }

    //获取数组最大值
    public double max(double[] array) {
        double max = 0;
        for (int i = 0; i < array.length; i++) {
            max = Math.max(max, array[i]);
        }
        return max;
    }

    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {

        this.tensorInput = tensor;
        tensorOut = tensorInput.copy();
        tensorOut.zero();

        //先求输入数据的最大值
        double max = max(tensorInput.getArray());

        //求分子numerator
        for (int i = 0; i < tensorInput.getWidth(); i++) {
            tensorOut.set(i, Math.exp(tensorInput.get(i) - max));
        }
        //求分母denominator
        double denominator = 0;
        for (int i = 0; i < tensorOut.getWidth(); i++) {
            denominator = denominator + tensorOut.get(i);
        }
        //输出结果
        for (int i = 0; i < tensorOut.getWidth(); i++) {
            tensorOut.set(i, tensorOut.get(i) / denominator);
        }

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
        return tensor;
    }

    //误差计算 ∂C/∂A
    @Override
    public Tensor error() {

        return tensorInput.calculateOutPartialDerivativeByCrossEntropy(getExpect());

//        Tensor pda = new Tensor();
//        pda.setWidth(bias.length);
//        pda.createArray();
//
//        for (int i = 0; i < w.getHeight(); i++) {
//            pda.set(i, (tensorOut.get(i) - getExpect().get(i)) * getGamma());
////            if (getExpect()[i] == 1) {
////                pda[i] = tensorOut[i] - 1;
////            } else {
////                pda[i] = tensorOut[i];
////            }
//        }
//
//        return pda;
    }


}
