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
        if (index * batch > list.size()) {
            index = 0;
        }

        BatchData batchData = createTensor();
        index = index + 1;

        return batchData;
    }

    private BatchData createTensor() {
        Tensor data = new Tensor();
        data.setBatch(batch);
        data.setDepth(list.get(0).getRgb().getDepth());
        data.setHeight(list.get(0).getRgb().getHeight());
        data.setWidth(list.get(0).getRgb().getWidth());
        data.createArray();

        Tensor expect = new Tensor();
        expect.setBatch(batch);
        expect.setWidth(list.get(0).getLabel().size());
        expect.createArray();

        for (int i = 0; i < batch; i++) {
            Tensor single = list.get(index * batch + i).getRgb();
            for (int size = 0; size < single.size(); size++) {
                data.set(i, 0, 0, size, single.get(size));
            }

//            batch expect
            Tensor singleExpect = list.get(index * batch + i).getLabel();
            for (int size = 0; size < singleExpect.size(); size++) {
                expect.set(i, 0, 0, size, singleExpect.get(size));
            }

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
