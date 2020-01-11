package xws.test;

import xws.neuron.data.BatchDataFactory;
import xws.util.BatchData;
import xws.util.Cifar10;
import xws.util.UtilMnist;

import java.util.List;

/**
 * batch data test
 */
public class BatchDataTest {

    public static void main(String[] args) {
        List<Cifar10> listTest = UtilMnist.testData();
        BatchDataFactory batchDataFactory = new BatchDataFactory(5, listTest);
        BatchData batchData = batchDataFactory.batch();
        batchData.getData().show();
        batchData.getExpect().show("this is expect *****************************************************");
    }

    public static List<Cifar10> createData() {


        return null;
    }
}
