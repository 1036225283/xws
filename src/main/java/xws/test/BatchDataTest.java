package xws.test;

import xws.neuron.Tensor;
import xws.neuron.data.BatchDataFactory;
import xws.util.BatchData;
import xws.util.Cifar10;
import xws.util.UtilMnist;

import java.util.ArrayList;
import java.util.List;

/**
 * batch data test
 */
public class BatchDataTest {

    public static void main(String[] args) {
        testBig();

    }


    public static void testBig() {
        List<Cifar10> listTest = UtilMnist.testData();
        BatchDataFactory batchDataFactory = new BatchDataFactory(5, listTest);
        BatchData batchData = batchDataFactory.batch();
        batchData.getData().show();
        batchData.getExpect().show("this is expect *****************************************************");
        listTest.get(2).getRgb().show("this is first *****************************************************");
    }

    public static void testSmall() {
        List<Cifar10> listTest = createData();
//        for (int i = 0; i < listTest.size(); i++) {
//            listTest.get(i).getRgb().show();
//        }
        BatchDataFactory batchDataFactory = new BatchDataFactory(5, listTest);
        BatchData batchData = batchDataFactory.batch();
        batchData.getData().show();
        batchData.getExpect().show("this is expect *****************************************************");
//        listTest.get(2).getRgb().show("this is first *****************************************************");
    }

    public static List<Cifar10> createData() {

        List<Cifar10> list = new ArrayList<>();
        for (int b = 0; b < 5; b++) {
            Cifar10 cifar10 = new Cifar10();
            cifar10.setRgb(createTensor(b));
            cifar10.setLabel(cifar10.getRgb());
            list.add(cifar10);
        }
        return list;
    }

    public static Tensor createTensor(int batch) {

        Tensor tensor = new Tensor();
        tensor.setHeight(4);
        tensor.setWidth(4);
        tensor.createArray();
        for (int h = 0; h < tensor.getHeight(); h++) {
            for (int w = 0; w < tensor.getWidth(); w++) {
                tensor.set(0, 0, h, w, (batch * 4 * 4) + (h * 4) + w + 1);
            }
        }
        return tensor;
    }


}
