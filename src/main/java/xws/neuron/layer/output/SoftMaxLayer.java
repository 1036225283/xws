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

        //计算每个批次的最大值
        Tensor maxTensor = new Tensor();
        maxTensor.setWidth(tensor.getBatch());
        maxTensor.createArray();

        Tensor denominatorTensor = new Tensor();
        denominatorTensor.setWidth(tensor.getBatch());
        denominatorTensor.createArray();

        // get every batch max value
        for (int b = 0; b < tensor.getBatch(); b++) {
            maxTensor.set(b, tensor.batchMax(b));
        }

        // get numerator
        for (int b = 0; b < tensor.getBatch(); b++) {
            for (int w = 0; w < tensorInput.getWidth(); w++) {
                double val = Math.exp(tensorInput.get(b, 0, 0, w) - maxTensor.get(b));
                tensorOut.set(b, 0, 0, w, val);
            }
        }

        // get denominator
        for (int b = 0; b < tensor.getBatch(); b++) {
            double denominator = 0;
            for (int w = 0; w < tensorOut.getWidth(); w++) {
                denominator = denominator + tensorOut.get(b, 0, 0, w);
            }
            denominatorTensor.set(b, denominator);

        }

        // result
        for (int b = 0; b < tensor.getBatch(); b++) {
            for (int w = 0; w < tensorOut.getWidth(); w++) {
                tensorOut.set(b, 0, 0, w, tensorOut.get(b, 0, 0, w) / denominatorTensor.get(b));
            }
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
        return tensorInput.calculateOutPartialDerivativeBySoftmax(getExpect(), getGamma());
    }


}
