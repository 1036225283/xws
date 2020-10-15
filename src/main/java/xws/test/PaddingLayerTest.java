package xws.test;

import xws.neuron.Tensor;
import xws.neuron.layer.PaddingLayer;
import xws.neuron.layer.pool.MaxPoolLayer;
import xws.neuron.layer.pool.MeanPoolLayer;

/**
 * mean pool test
 * Created by xws on 2019/6/6.
 */
public class PaddingLayerTest {
    public static void main(String[] args) {
        testPadding();
    }


    //测试padding
    public static void testPadding() {
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

        PaddingLayer paddingLayer = new PaddingLayer(1, 1, 1, 1);
        Tensor tensorForward = paddingLayer.forward(tensor);
        tensorForward.show("前向传播结果：");

        Tensor tensorB = paddingLayer.backPropagation(tensorForward);
        tensorB.show("反向传播结果：");
    }


}
