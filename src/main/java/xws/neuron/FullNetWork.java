package xws.neuron;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * 神经网络
 * 前向传播搞定了
 * 误差计算没什么用
 * 反向传播怎么搞
 * 先计算pda = ∂C/∂A
 * 在循环每一个层
 * 将神经元消除掉，只保留神经层就可以了
 * Created by xws on 2019/1/5.
 */
public class FullNetWork extends NeuralNetWork {


    private Layer[] layers;//层级结构
    private int[] struct;//存储神经网络结构
    private double[] input;//整个神经网络的输入
    private double[] expect;//整个神经网络的期望
    private double rate = 2.71828;//整个神经网络的学习速率


    /**
     * 外部可以操作的神经层
     */
    private Layer layerOperation;

    //外部可操作的神经元索引
    private int neuronIndex;


    //初始化神经网络
    public FullNetWork(int[] struct) {
        init(struct);
    }


    public FullNetWork(String str) {

        String[] strings = str.split("-");

        int[] struct = new int[strings.length];
        for (int i = 0; i < struct.length; i++) {
            struct[i] = Integer.valueOf(strings[i]);
        }

        init(struct);
    }

    private void init(int[] struct) {
        this.struct = struct;
        layers = new Layer[struct.length - 1];
        for (int i = 0; i < layers.length; i++) {
            Layer layer = new Layer(struct[i + 1], struct[i], i);//arr[i]表示这一层需要多少个神经元, arr[i-1]表示上一层有多少个输入
            this.layers[i] = layer;
        }
    }


    public double[] work(double[] input) {
        return forward(input, null);
    }

    //前向传播
    public double[] forward(double[] input, double[] expect) {


        if (expect != null && expect.length != struct[struct.length - 1]) {
            throw new RuntimeException("期望值的数量和输出值的数量不符");
        }

        this.expect = expect;

        for (int i = 0; i < this.layers.length; i++) {
            Layer layer = this.layers[i];
            layer.forward(input);
//            System.out.println(JSONObject.toJSONString(layer.a()));
            input = layer.a();
            if (UtilNeuralNet.checkDouble(input)) {
                throw new RuntimeException("checkDouble layerIndex = " + i);
            }

        }


        if (expect != null) {
            backPropagation();
        }

//        System.out.println("输出的误差是：" + JSON.toJSONString(error()));


        return layers[layers.length - 1].a();
    }


    //反向传播
    public void backPropagation() {
        //反向传播，从最后一层神经层向前传播到第一层神经层
        for (int i = layers.length - 1; i >= 0; i--) {
            Layer layer = layers[i];
            layer.backPropagation();
        }
    }

    //计算误差
    public double error() {


        double error = 0;
        //获取最后一层的输出
        Layer lastLayer = layers[layers.length - 1];
        double[] out = lastLayer.a();
        double[] errs = new double[out.length];


        if (out == null || out.length == 0 || expect == null || expect.length == 0) {
            return Double.MAX_VALUE;
        }

        for (int i = 0; i < out.length; i++) {
            errs[i] = Math.pow(expect[i] - out[i], 2);
            error = error + Math.pow(expect[i] - out[i], 2);
        }

        System.out.println("误差数组：" + JSON.toJSONString(errs));
        return error;

    }

    //测试用例
    public List<Double> calculate() {
        double[] input = new double[]{1};
        List<Double> vertex = new ArrayList<>();
        //创建顶点并绘制
        for (double x = -10; x <= 10; x = x + 0.01) {
            input[0] = x;
            double[] y = this.forward(input, null);
            vertex.add(x);
            vertex.add(y[0]);
            vertex.add(0d);
        }
        return vertex;
    }

    //选中某一个层的某一个神经元，然后拿到输出，然后查看这个神经元的输出函数图像
    public List<Double> calculate(int l, int n) {
        double[] input = new double[]{1};
        List<Double> vertex = new ArrayList<>();
        //创建顶点并绘制
        for (double x = -10; x <= 10; x = x + 0.01) {
            input[0] = x;
            this.forward(input, null);
            double y = layers[l].a()[n];

            vertex.add(x);
            vertex.add(y);
            vertex.add(0d);
        }
        return vertex;
    }

    public Layer[] getLayers() {
        return layers;
    }

    public int[] getStruct() {
        return struct;
    }

    //选取某个layer
    public FullNetWork selectLayer(int index) {

        if (index < layers.length) {
            layerOperation = layers[0];
        } else {
            layerOperation = layers[index];
        }
        return this;
    }

    //选取某个神经元
    public FullNetWork selectNeuron(int index) {
        this.neuronIndex = index;
        return this;
    }

    //设置某个神经元的某个w
    public FullNetWork setW(int index, double w) {
        layerOperation.w[neuronIndex][index] = w;
        return this;
    }

    //设置某个神经元的b
    public FullNetWork setB(double b) {
        layerOperation.bias[neuronIndex] = b;
        return this;
    }

    //获取输入
    public double[] getInput() {
        return input;
    }

    //获取输出
    public double[] a() {
        return layers[layers.length - 1].a();
    }

    //获取神经网络的唯一识别结果
    public int result() {
        double[] out = a();
        double max = 0;
        int maxIndex = 0;
        for (int i = 0; i < out.length; i++) {
            if (out[i] > max) {
                max = out[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    //加载神经网络
    public static FullNetWork load(String name) {
        try {

            JSONObject jsonObject = loadJson(name);

            int version = jsonObject.getIntValue("version");

            JSONArray structArr = jsonObject.getJSONArray("struct");

            int[] struct = new int[structArr.size()];
            for (int i = 0; i < struct.length; i++) {
                struct[i] = structArr.getIntValue(i);
            }

            FullNetWork netWork = new FullNetWork(struct);

            netWork.setVersion(version + 1);


            JSONArray layers = jsonObject.getJSONArray("layers");

            for (int i = 0; i < layers.size(); i++) {

                JSONObject item = layers.getJSONObject(i);

                JSONArray bias = item.getJSONArray("bias");

                double[] b = new double[bias.size()];
                for (int k = 0; k < bias.size(); k++) {
                    b[k] = bias.getDoubleValue(k);
                }
                netWork.layers[i].bias = b;

                JSONArray weight = item.getJSONArray("w");
                double[][] w = new double[weight.size()][];
                for (int k = 0; k < weight.size(); k++) {
                    JSONArray item_w = weight.getJSONArray(k);

                    w[k] = new double[item_w.size()];
                    for (int j = 0; j < item_w.size(); j++) {
                        w[k][j] = item_w.getDoubleValue(j);
                    }
                }

                netWork.layers[i].w = w;
            }
            return netWork;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

    }


    private class Layer {


        private int index;//表示在神经网络中，属于第几层

        private double[] a;//某一层的输出
        private double[] input;//把上一层的输入也保存起来

        private double[][] w;//一维是神经元的数量，二维是每个神经元的权重
        private double[] bias;//每个神经元的偏置
        private double[] z;//每个神经元的z值

        //一下三个变量，在每次计算之前必须清空
        private double[][] pdi;//∂C/∂I - I是上一层的输入
        private double[][] pdw;//∂C/∂W
        private double[] pdb;//∂C/∂Z = ∂C/∂B - Z是这一层的输出
        private double[] pda;//每层输出的偏导，最后一层不需要

        //初始化神经网络层
        public Layer(int num, int inputs, int index) {

            //index决定了这个layer在神经网络的哪一层
            this.index = index;


            //初始化每个神经元的权重和偏置
            w = new double[num][inputs];
            bias = new double[num];

            for (int i = 0; i < w.length; i++) {
                double[] weight = w[i];
//                for (int k = 0; k < weight.length; k++) {
//                    weight[k] = Math.random();
//                }
                UtilNeuralNet.initWeight(weight);
//                bias[i] = Math.random();
            }
            UtilNeuralNet.initBias(bias);

            a = new double[num];
            z = new double[num];


        }

        //计算每一个神经元的输出值
        public void forward(double[] input) {
            this.input = input;
            for (int i = 0; i < w.length; i++) {
                double[] weight = w[i];
                double bias = this.bias[i];
                //计算神经元的输出
                z[i] = 0;//把之前的数据清理掉
                for (int k = 0; k < weight.length; k++) {
                    z[i] = z[i] + weight[k] * input[k];
                }
                z[i] = z[i] + bias;
//                a[i] = sigmoid(z[i]);
                a[i] = ActivationFunction.activation(z[i], "sigmoid");
            }
        }

        //获取这一层神经网络的输出
        public double[] a() {
            return a;
        }


        /**
         * 反向传播
         * 先计算计算∂C/∂I
         * 再计算∂C/∂W
         * 再计算∂C/∂B
         */
        public void backPropagation() {

            //先把pdz,pdw,pdb,pdi清空
            pdb = new double[bias.length];
            pda = new double[bias.length];

            //inputs决定了有多少个输入，也就是这一层的神经元会有多少个w
            pdi = new double[bias.length][w[0].length];
            pdw = new double[bias.length][w[0].length];

            //首先计算pdz = ∂C/∂A * ∂A/∂Z
            //如果最后一层，那么f(c) = (a-y)^2
            if (index == layers.length - 1) {

                //∑(∂E/∂A)

//                double totalError = 0;
//                for (int i = 0; i < w.length; i++) {
//                    totalError = totalError + (a[i] - expect[i]);
//                }


                for (int i = 0; i < w.length; i++) {
//                    pdz[i] = totalError * d_sigmoid(a[i]);
                    pdb[i] = (a[i] - expect[i]) * ActivationFunction.derivation(z[i], "sigmoid");
                }
            } else {
                Layer nextLayer = layers[index + 1];
                for (int i = 0; i < w.length; i++) {
                    for (int k = 0; k < nextLayer.w.length; k++) {
                        pda[i] = pda[i] + nextLayer.pdi[k][i];
                    }
                    pdb[i] = pda[i] * ActivationFunction.derivation(z[i], "sigmoid");
                }
            }

//            System.out.println("pdz = " + JSON.toJSONString(pdz));

            pdi();
            pdw();
            pdb();
        }


        //神经元计算∂C/∂I
        public void pdi() {
            for (int i = 0; i < pdi.length; i++) {
                double[] is = pdi[i];
                double[] weight = w[i];
                for (int k = 0; k < weight.length; k++) {
                    is[k] = this.pdb[i] * weight[k];
                }
            }
        }

        //神经元计算∂C/∂W
        public void pdw() {
            for (int i = 0; i < w.length; i++) {
                double[] weight = w[i];
                double[] pdw = this.pdw[i];
                double pdz = this.pdb[i];
                for (int k = 0; k < weight.length; k++) {
//                    System.out.println("训练前w = " + weight[k]);
                    pdw[k] = pdz * input[k];
                    weight[k] = weight[k] - rate * pdw[k];
//                    System.out.println("训练后w = " + weight[k]);
                }
            }
        }

        //神经元计算∂C/∂B =
        public void pdb() {
            for (int i = 0; i < bias.length; i++) {
//                System.out.println("训练前b = " + bias[i]);
                bias[i] = bias[i] - rate * pdb[i];
//                System.out.println("训练后b = " + bias[i]);
            }
        }

        public double[][] getW() {
            return w;
        }

        public double[] getBias() {
            return bias;
        }
    }

}

