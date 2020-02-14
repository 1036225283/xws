package xws.neuron;

import xws.neuron.layer.Layer;

import java.util.ArrayList;
import java.util.List;

/**
 * 卷积神经网络的输入层
 * Created by xws on 2019/2/19.
 */
public class Input2Layer extends Layer {

    private List<double[][]> list = new ArrayList<>();

    public Input2Layer() {
        super("input2");
    }

    //存放通道数据
    public void add(double[][] channel) {
        list.add(channel);
    }


    public List<double[][]> getList() {
        return list;
    }
}
