package xws.test;

import xws.neuron.Tensor;
import xws.neuron.layer.Padding1DLayer;
import xws.neuron.layer.Padding2DLayer;

/**
 * mean pool test
 * Created by xws on 2019/6/6.
 */
public class PaddingLayerTest {
    public static void main(String[] args) {
        testPadding2D();
//        testPadding1D();
    }


    //测试padding
    public static void testPadding2D() {
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

        Padding2DLayer paddingLayer = new Padding2DLayer(1, 1);
        Tensor tensorForward = paddingLayer.forward(tensor);
        tensorForward.show("前向传播结果：");

        Tensor tensorB = paddingLayer.backPropagation(tensorForward);
        tensorB.show("反向传播结果：");
    }


    public static void testPadding1D() {
        int val = 5;
        Tensor tensor = new Tensor();
        tensor.setDepth(val);
        tensor.setWidth(val);
        tensor.createArray();

        for (int d = 0; d < val; d++) {
            for (int w = 0; w < val; w++) {
                tensor.set(d, 0, w, 2);
            }
        }


        tensor.show("输入数据：");

        Padding1DLayer paddingLayer = new Padding1DLayer(1);
        Tensor tensorForward = paddingLayer.forward(tensor);
        tensorForward.show("前向传播结果：");

        Tensor tensorB = paddingLayer.backPropagation(tensorForward);
        tensorB.show("反向传播结果：");
    }

}
