package xws.neuron.layer.activation;

import xws.neuron.ActivationFunction;
import xws.neuron.Tensor;
import xws.neuron.layer.Layer;


/**
 * ReLu激活函数
 * Created by xws on 2019/10/31.
 */
public class ReLuLayer extends Layer {


    private Tensor tensorInput;


    public ReLuLayer() {
        super(ReLuLayer.class.getSimpleName());
    }

    public ReLuLayer(String name) {
        super(ReLuLayer.class.getSimpleName());
        setName(name);
    }

    public static void main(String[] args) {
        new ReLuLayer();
    }

    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {
        tensorInput = tensor;

        Tensor tensorOut = tensorInput.copy();

        for (int i = 0; i < tensorInput.size(); i++) {
            tensorOut.set(i, ActivationFunction.relu(tensorOut.get(i)));
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

        Tensor tensorError = tensor.copy();

        //先计算输入误差
        for (int i = 0; i < tensorError.size(); i++) {
            tensorError.set(i, ActivationFunction.relu_d(tensorError.get(i)));
        }

        return tensorError;
    }

    //误差计算
    @Override
    public Tensor error() {
        return null;
    }

}
