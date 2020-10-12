package xws.test;

import xws.neuron.Tensor;
import xws.neuron.layer.conv.Conv1DLayer;
import xws.neuron.layer.pool.MaxPoolLayer;

public class Conv1DLayerTest {
    public static void main(String[] args) {
        Tensor tensor = createDepthTensor();
        Conv1DLayer conv1DLayer = new Conv1DLayer(10, 3, 1, 0);
        Tensor tensor1 = conv1DLayer.forward(tensor);
        MaxPoolLayer maxPoolLayer = new MaxPoolLayer(1, 2, 2, 1);
        tensor1 = maxPoolLayer.forward(tensor1);
        tensor1.show();

    }

    public static Tensor createTensor() {
        Tensor tensor = new Tensor();
        tensor.setWidth(10);
        tensor.createArray();

        for (int i = 0; i < tensor.getWidth(); i++) {
            tensor.set(i, i + 1);
        }
        return tensor;
    }

    public static Tensor createDepthTensor() {
        Tensor tensor = new Tensor();
        tensor.setDepth(3);
        tensor.setWidth(10);
        tensor.createArray();
        try {
            for (int d = 0; d < tensor.getDepth(); d++) {
                for (int w = 0; w < tensor.getWidth(); w++) {
                    tensor.set(d, 0, w, w + 1);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }


        return tensor;
    }


}
