package xws.neuron.graph;

/**
 * 计算图
 * Created by xws on 2019/9/1.
 */
public class Graph {

    private int capacity = 100;//default value 100
    private int size = 0;//default value 0


    private Operation[][] array;

    public Graph() {

    }

    //    init graph
    private void init() {
        array = new Operation[capacity][capacity];
    }

//    add

}
