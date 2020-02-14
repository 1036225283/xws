package xws.test;

import xws.neuron.Tensor;

/**
 * Created by xws on 2019/6/18.
 */
public class TensorSubTest {

    public static void main(String[] args) {
        int val = 10;
        int depth = 4;
        Tensor tensor = new Tensor();
        tensor.setDepth(depth);
        tensor.setHeight(val);
        tensor.setWidth(val);
        tensor.createArray();

        int index = 0;
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < val; h++) {
                for (int w = 0; w < val; w++) {
                    tensor.set(d, h, w, index);
                    index++;
                }
            }
        }

        tensor.show("裁剪前：");
        tensor.subWidth(1, 0).show("**********************************************************************************************************************：");

    }
}
