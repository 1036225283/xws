package xws.util;

import xws.neuron.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * RNN DATA
 * Created by xws on 2019/5/19.
 */
public class RnnSequence {


    private List<Cifar10> list = new ArrayList<>();


    public Cifar10 get(int index) {
        return list.get(index);
    }

    public Tensor getData(int index) {
        return list.get(index).getData();
    }


    public double[] getExpect(int index) {
        return new double[]{list.get(index).getValue()};
    }

    public void add(Cifar10 cifar10) {
        list.add(cifar10);
    }

    public void add(double[] key, double val) {
        Tensor tensor = new Tensor();
        tensor.setWidth(key.length);
        tensor.setArray(key);

        Cifar10 cifar10 = new Cifar10();
        cifar10.setData(tensor);
        cifar10.setValue(val);
        list.add(cifar10);
    }

    public int size() {
        return list.size();
    }

    public List<Cifar10> getList() {
        return list;
    }

    public void setList(List<Cifar10> list) {
        this.list = list;
    }
}
