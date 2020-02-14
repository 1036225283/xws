package xws.neuron.data;

import xws.neuron.Tensor;
import xws.util.BatchData;
import xws.util.Cifar10;

import java.util.List;


public class BatchDataFactory {

    private int batch;
    private List<Cifar10> list;
    private int index = 0;

    public BatchDataFactory(int batch, List<Cifar10> list) {
        this.batch = batch;
        this.list = list;
    }


    public BatchData batch() {
        if ((index + 1) * batch >= list.size()) {
            index = 0;
        }

        BatchData batchData = createTensor();
        index = index + 1;

        return batchData;
    }

    private BatchData createTensor() {
        Tensor data = new Tensor();
        data.setBatch(batch);
        data.setDepth(list.get(0).getData().getDepth());
        data.setHeight(list.get(0).getData().getHeight());
        data.setWidth(list.get(0).getData().getWidth());
        data.createArray();

        Tensor expect = new Tensor();
        expect.setBatch(batch);
        expect.setWidth(list.get(0).getExpect().size());
        expect.createArray();

        Tensor gamma = new Tensor();
        gamma.setBatch(batch);
        gamma.createArray();

        for (int i = 0; i < batch; i++) {
            Cifar10 item = list.get(index * batch + i);
            Tensor single = item.getData();
            for (int size = 0; size < single.size(); size++) {
                data.set(i, 0, 0, size, single.get(size));
            }

//            batch expect
            Tensor singleExpect = item.getExpect();
            for (int size = 0; size < singleExpect.size(); size++) {
                expect.set(i, 0, 0, size, singleExpect.get(size));
            }

            gamma.set(i, 0, 0, item.getValue());

        }


        BatchData batchData = new BatchData();
        batchData.setData(data);
        batchData.setExpect(expect);
        return batchData;
    }

    public int getBatch() {
        return batch;
    }


    public List<Cifar10> getList() {
        return list;
    }

}
