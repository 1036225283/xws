package xws.neuron.graph;

import com.alibaba.fastjson.JSON;
import xws.neuron.layer.FullLayer;
import xws.neuron.layer.Layer;
import xws.neuron.layer.bn.BnLayer;
import xws.neuron.layer.conv.ConvolutionLayer;
import xws.neuron.layer.pool.MaxPoolBackLayer;
import xws.neuron.layer.pool.MaxPoolLayer;

import java.util.ArrayList;

/**
 * 计算图
 * Created by xws on 2019/9/1.
 */
public class Graph {

    private ArrayList<Layer> layers = new ArrayList<>();
    private ArrayList<String> nodes = new ArrayList<>();


    private int capacity = 20;//default value 100
    private int size = 0;//default value 0

    private boolean[] inputDegree = new boolean[capacity];


    private int[][] edges;

    public Graph() {
        init();
    }

    // init graph
    private void init() {
        edges = new int[capacity][capacity];
    }

    // add vertex (layer)
    public void addVertex(Layer layer) {

        if (size == capacity) {
            resize(capacity + capacity / 2);
        }

        layers.add(layer);
        size = size + 1;
    }

    // add edge (forward)
    public void addEdge(Layer layerFrom, Layer layerTo) {
        int indexFrom = layers.indexOf(layerFrom);
        int indexTo = layers.indexOf(layerTo);
        edges[indexFrom][indexTo] = 1;
    }

    // depth first search
    public void depthSearch() {

    }

    //search input degree


    private void resize(int newSize) {
        int[][] newEdges = new int[newSize][newSize];
        for (int i = 0; i < edges.length; i++) {
            for (int k = 0; k < edges[0].length; k++) {
                newEdges[i][k] = edges[i][k];
            }
        }
        capacity = newSize;
        edges = newEdges;
        inputDegree = new boolean[capacity];
    }

    public int size() {
        return size;
    }

    public static void main(String[] args) {

        int capacity = 200;
        capacity = capacity + capacity / 2;
        System.out.println(capacity);

        Graph graph = new Graph();

        Layer bnLayer_1 = new BnLayer();
        Layer convLayer_1 = new ConvolutionLayer();
        Layer maxPollLayer_1 = new MaxPoolLayer();
        Layer fullLayer_1 = new FullLayer();


        graph.addVertex(bnLayer_1);
        graph.addVertex(convLayer_1);
        graph.addVertex(maxPollLayer_1);
        graph.addVertex(fullLayer_1);

        graph.addEdge(bnLayer_1, convLayer_1);
        graph.addEdge(convLayer_1, maxPollLayer_1);
        graph.addEdge(maxPollLayer_1, fullLayer_1);


        System.out.println(graph.size());
        System.out.println(JSON.toJSONString(graph.edges));

    }

}
