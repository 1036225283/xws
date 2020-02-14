package xws.neuron.layer.output;

import xws.neuron.Tensor;
import xws.neuron.layer.Layer;


/**
 * 交叉熵损失函数
 * 初始化时，需要指定有多少个神经元
 * Created by xws on 2019/2/20.
 */
public class CrossEntropyLayer extends Layer {


    private Tensor tensorOut;//某一层的输出
    private Tensor tensorInput;//把上一层的输入也保存起来

    public CrossEntropyLayer() {
    }

    public CrossEntropyLayer(String name) {
        super("CrossEntropyLayer");
        setName(name);
    }


    //计算每一个神经元的输出值
    @Override
    public Tensor forward(Tensor tensor) {
        tensorInput = tensor;
        tensorOut = tensor;
        return tensor;
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

    }

}
