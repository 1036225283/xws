package xws.test;

import xws.neuron.Tensor;
import xws.neuron.layer.pool.MaxPoolLayer;
import xws.neuron.layer.pool.MeanPoolLayer;

/**
 * mean pool test
 * Created by xws on 2019/6/6.
 */
public class PoolTest {
    public static void main(String[] args) {
//        testMaxPool();
//        testPadding();
    }


    public static void testMaxPool() {
        int val = 5;
        Tensor tensor = new Tensor();
        tensor.setHeight(val);
        tensor.setWidth(val);
        tensor.createArray();

        for (int i = 0; i < val; i++) {
            for (int j = 0; j < val; j++) {
                tensor.set(i, j, 2);
            }
        }

        tensor.show("输入数据：");

        MaxPoolLayer poolLayer = new MaxPoolLayer(2, 2, 2, 2);
        Tensor tensorForward = poolLayer.forward(tensor);
        tensorForward.show("前向传播结果：");

        Tensor tensorB = poolLayer.backPropagation(tensorForward);
        tensorB.show("反向传播结果：");
    }

    public static void testMeanPool() {
        int val = 4;
        Tensor tensor = new Tensor();
        tensor.setHeight(val);
        tensor.setWidth(val);
        tensor.createArray();

        for (int i = 0; i < val; i++) {
            for (int j = 0; j < val; j++) {
                tensor.set(i, j, 2);
            }
        }

        tensor.show();

        MeanPoolLayer poolLayer = new MeanPoolLayer(2, 2, 1, 1);
        Tensor tensorForward = poolLayer.forward(tensor);
        tensorForward.show();

        Tensor tensorB = poolLayer.backPropagation(tensor);
        tensorB.show();
    }
}
