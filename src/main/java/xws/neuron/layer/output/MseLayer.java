package xws.neuron.layer.output;

import xws.neuron.Tensor;
import xws.neuron.layer.Layer;


/**
 * 均方误差
 * Created by xws on 2019/2/20.
 */
public class MseLayer extends Layer {

    private Tensor tensorInput;//把上一层的输入也保存起来

    public MseLayer(String name) {
        super(MseLayer.class.getSimpleName());
        setName(name);
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
        return tensorInput.calculateOutPartialDerivativeByMse(getExpect());
    }

}
