package xws.test;

import com.alibaba.fastjson.JSONObject;
import xws.neuron.CNNetWork;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.*;
import xws.neuron.layer.bn.LnLayer;
import xws.neuron.layer.conv.ConvolutionLayer;
import xws.neuron.layer.output.SoftMaxLayer;
import xws.neuron.layer.pool.MaxPoolLayer;
import xws.util.Cifar10;
import xws.util.UtilCifar10;

import java.util.List;

import static xws.test.FullNetWorkTest.*;

/**
 * Cifar10测试
 * Created by xws on 2019/1/22.
 */
public class Cifar10Test {

    static List<Cifar10> listTest = UtilCifar10.testData();
    static List<Cifar10> listLearn = UtilCifar10.learnData();

    //    private static String strName = "cifar10-test";
//    private static String strName = "cifar10-12-36-256-128-10";
//    private static String strName = "cifar10-depthwise-2";
    private static String strName = "cifar10-LN";


    public static void main(String[] args) {
        createCNNetWork();
        learnCifar10();
    }


    public static void createCNNetWork() {
        CNNetWork cnNetWork = new CNNetWork();


        //54.21%
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 16, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 16, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 16, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 10, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftMaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 16, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 16, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new LnLayer("bn2"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 16, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 16, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftMaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

        cnNetWork.addLayer(new LnLayer("ln0"));
        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 1, 1));
        cnNetWork.addLayer(new LnLayer("ln1"));
        cnNetWork.addLayer(new ConvolutionLayer("filter2", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 1, 1));
        cnNetWork.addLayer(new LnLayer("ln2"));
        cnNetWork.addLayer(new ConvolutionLayer("filter2", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 1, 1));
        cnNetWork.addLayer(new LnLayer("ln2"));
        cnNetWork.addLayer(new FullLayer("full2", 64, UtilNeuralNet.e() * 0.00000000001));
        cnNetWork.addLayer(new SoftMaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));


        //37.22%    ||  40.87%
//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.00000000001));

        //36.67%
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftMaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

        //39.87%    ||    47.36%    ||  47.58
//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter2", "relu", 16, 5, 5, 1, 1, 0));
//        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new LnLayer("bn3"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.0000001));


        //38.92%
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new ConvolutionLayer("filter2", "relu", 16, 5, 5, 1, 1, 0));
//        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftMaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));
//
        //32*32
//        cnNetWork.addLayer(new DepthSeparableLayer("depth1", "relu", 5, 5, 1, 1, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 6, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new DepthSeparableLayer("depth2", "relu", 3, 3, 1, 1, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new ConvolutionLayer("filter2", "relu", 6, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftMaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

        //41.73%
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new ConvolutionLayer("filter2", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftMaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));


        System.out.println(JSONObject.toJSONString(cnNetWork));

        Cifar10 cifar10 = listTest.get(0);
//        UtilNeuralNet.initMinst(cifar10.getRgb().getArray());
        cnNetWork.entryTest();
        cnNetWork.learn(cifar10.getRgb(), expectMNIST(cifar10.getLabel()));
        cnNetWork.save(strName);
    }


    //测试手写数组识别
    public static void learnCifar10() {

        //数据归一化
//        for (int i = 0; i < listLearn.size(); i++) {
//            UtilNeuralNet.initStock(listLearn.get(i).getRgb().getArray());
//        }
//
//        for (int i = 0; i < listTest.size(); i++) {
//            UtilNeuralNet.initStock(listTest.get(i).getRgb().getArray());
//        }

        double learnRate = UtilNeuralNet.e() * 0.001;
        for (int x = 0; x < 100; x++) {
            CNNetWork cnNetWork = CNNetWork.load(strName);
            cnNetWork.entryLearn();
            cnNetWork.setBatchSize(5);
            learnRate = learnRate * 0.9;
            cnNetWork.setLearnRate(learnRate);
            cnNetWork.setBatch(10);
            System.out.println("第 " + x + "次");
            //每次取一批数据
            int batch = 2000;
            for (int i = 0; i < listLearn.size(); i = i + batch) {
                //将这一批数据，反复喂给
                for (int k = 0; k < 3; k++) {
                    for (int j = 0; j < batch; j++) {
                        Cifar10 cifar10 = listLearn.get(i + j);
                        Tensor expect = expectMNIST(cifar10.getLabel());
                        cnNetWork.learn(cifar10.getRgb(), expect);
                    }
                }
                UtilCifar10.test(cnNetWork, listTest);
            }
            cnNetWork.save(strName);
        }


    }


}
