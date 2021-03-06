package xws.test;

import com.alibaba.fastjson.JSON;
import xws.neuron.CNNetWork;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.*;
import xws.util.Cifar10;
import xws.util.UtilCifar10;
import xws.util.UtilMnist;

import java.util.List;
import java.util.Map;

import static xws.test.FullNetWorkTest.*;

/**
 * 卷积神经网络的测试
 * Created by xws on 2019/1/22.
 */
public class MnistTest {

    private static String strName = "DROPOUT";
//    private static String strName = "XXX";
//    private static String strName = "LeNet-5";
//    private static String strName = "FC";


    public static void main(String[] args) {
//        testBN();

        createCNNetWork();
        learnMNIST();//训练手写字符识别
//        testMNIST();//识别手写字符


//        testCNNetWork();//测试卷积神经网络
//        testTensor();//测试张量二维和一维的相互转换

//        testFilterLayer();//测试卷积操作
//        testPoolLayer();//测试卷积层
//        System.out.println(ActivationFunction.sigmoid(2));
//        and();

//        xor();
//        and_or_xor();


    }


    //测试BN前后的变化
    public static void testBN() {
        Map<Double, double[]> test = loadMNIST();
        for (Map.Entry<Double, double[]> entry : test.entrySet()) {
            double[] input = entry.getValue();
            //首先，求均值和方差
            System.out.println("1.均值：" + UtilNeuralNet.average(input));
            System.out.println("2.方差：" + UtilNeuralNet.variance(input));

            for (int k = 0; k < input.length; k++) {
                input[k] = input[k] / 255;
            }

            System.out.println("3.均值：" + UtilNeuralNet.average(input));
            System.out.println("4.方差：" + UtilNeuralNet.variance(input));

            //归一化后，在求均值和方差
            System.out.println("    ");
        }
    }

    //产生Tensor
    public static Tensor createTensor() {
        Tensor tensor = new Tensor();
        tensor.setDepth(10);

        tensor.setHeight(10);
        tensor.setWidth(10);
        tensor.createArray();

        int i = 0;
        for (int d = 0; d < 10; d++) {
            for (int h = 0; h < 10; h++) {
                for (int w = 0; w < 10; w++) {
                    tensor.set(d, h, w, i++);
                }
            }
        }
        return tensor;
    }

    public static Tensor createTensorSimple() {
        Tensor tensor = new Tensor();
        tensor.setDepth(3);

        tensor.setHeight(10);
        tensor.setWidth(10);
        tensor.createArray();

        int i = 0;
        for (int d = 0; d < 3; d++) {
            for (int h = 0; h < 10; h++) {
                for (int w = 0; w < 10; w++) {
                    tensor.set(d, h, w, 1);
                }
            }
        }
        return tensor;
    }

    //测试池化操作
    public static void testPoolLayer() {

        Tensor tensor = createTensor();
        System.out.println("前向传播输入数据：");
        tensor.show();
        PoolLayer poolLayer = new PoolLayer("test", 2, 2, 2, 2);
        tensor = poolLayer.forward(tensor);
        System.out.println("前向传播结果数据");
        tensor.show();
        tensor = poolLayer.backPropagation(tensor);
        System.out.println("反向传播结果数据");

        tensor.show();
    }

    //测试卷积层的卷积操作
    public static void testFilterLayer() {

        Tensor tensor = createTensorSimple();
        System.out.println("前向传播输入数据：");
        tensor.show();

        FilterLayer filterLayer = new FilterLayer("test", "relu", 3, 3, 3, 1, 1, 0);
        tensor = filterLayer.forward(tensor);
        System.out.println("前向传播结果数据：");
        tensor.show();

        tensor = filterLayer.backPropagation(tensor);
        System.out.println("反向传播输出数据：");
        tensor.show();


    }

    public static void createCNNetWork() {
        CNNetWork cnNetWork = new CNNetWork();


        //97.07%
//        cnNetWork.addLayer(new DropoutLayer("drop1", "relu", 64, 0.5, UtilNeuralNet.e() * 0.000000000001));
//        cnNetWork.addLayer(new DropoutLayer("drop2", "relu", 64, 0.5, UtilNeuralNet.e() * 0.000000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("SoftMaxLayer", 10, UtilNeuralNet.e() * 0.000000000001));

        //96.81%
//        cnNetWork.addLayer(new DropoutLayer("drop1", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new DropoutLayer("drop2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.00000000001));

        //97.25%
//        cnNetWork.addLayer(new DropoutLayer("drop1", "relu", 64, UtilNeuralNet.e() * 0.000000000001));
//        cnNetWork.addLayer(new DropoutLayer("drop2", "relu", 32, UtilNeuralNet.e() * 0.000000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.000000000001));


        //97.56%
//        cnNetWork.addLayer(new FullLayer("full1", "relu", 64,                    UtilNeuralNet.e() * 0.000000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64,                    UtilNeuralNet.e() * 0.000000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.000000000001));
//

        //97.45%
//        cnNetWork.addLayer(new FullLayer("full1", "relu", 64, UtilNeuralNet.e() * 0.000000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 32, UtilNeuralNet.e() * 0.000000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.000000000001));


        //97.02
//        cnNetWork.addLayer(new FullLayer("full1", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() *    0.00000000001));
//


        //98.78%
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.00000000001));

        //98.60%
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

        //98.55%    ||  98.71%
//        cnNetWork.addLayer(new LnLayer("ln1"));
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new LnLayer("ln2"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));


        //98.91%    ||  98.87%  ||  99.02%
//        cnNetWork.addLayer(new BnLayer("ln1"));
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new BnLayer("ln1"));
//        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 16, 5, 5, 1, 1, 0));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new BnLayer("ln1"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.0000001));


        //98.86%
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 16, 5, 5, 1, 1, 0));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

        //99.05%    epoch=8     batch=10    learnRate=0.9   k=2 filter1=8   filter2=8
        //99.17%    epoch=8     batch=10    learnRate=0.9   k=2 filter1=9   filter2=9
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));


        //99.03%    ||  99.01%  ||  99.00%
        cnNetWork.addLayer(new BnLayer("bn1"));
        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 9, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
        cnNetWork.addLayer(new BnLayer("bn2"));
        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 9, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
        cnNetWork.addLayer(new BnLayer("bn3"));
        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
        cnNetWork.addLayer(new BnLayer("bn4"));
        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));


        //98.98%
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 10, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 10, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

        //98.81%
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new LnLayer("bn1"));
//        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new LnLayer("bn2"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

        //98.88%
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new BnLayer("bn1"));
//        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 10, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new BnLayer("bn2"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));


        //98.32%
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 1, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 6, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 6, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));

        //
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 9, 1, 1, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new MnLayer("ln"));
//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 9, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new MnLayer("ln"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));


//        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 16, 5, 5, 1, 1, 0));
//        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new FullLayer("full1", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.0000001));


//        最高96.38%
//        cnNetWork.addLayer(new FullLayer("full1", "relu", 32));
//        cnNetWork.addLayer(new FullLayer("full2", "sigmoid", 32));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10));


        List<Cifar10> list = UtilMnist.testData();
        Cifar10 cifar10 = list.get(0);
        UtilNeuralNet.initMinst(cifar10.getRgb().getArray());
        cnNetWork.entryTest();
        cnNetWork.learn(cifar10.getRgb(), expectMNIST(cifar10.getLabel()));
        cnNetWork.save(strName);
    }


    //测试手写数组识别
    public static void learnMNIST() {

        //加载数据
        List<Cifar10> list = UtilMnist.learnData();
        List<Cifar10> listTest = UtilMnist.testData();

        //对所有的数据进行归一化
        for (int i = 0; i < list.size(); i++) {
            UtilNeuralNet.initMinst(list.get(i).getRgb().getArray());
        }
//
        for (int i = 0; i < listTest.size(); i++) {
            UtilNeuralNet.initMinst(listTest.get(i).getRgb().getArray());
        }

        double learnRate = UtilNeuralNet.e() * 0.001;
        for (int x = 0; x < 100; x++) {
            CNNetWork cnNetWork = CNNetWork.load(strName);
            cnNetWork.entryLearn();
            cnNetWork.setBatchSize(5);
//            cnNetWork.setBatchSize(20);
            cnNetWork.setBatch(10);
            learnRate = learnRate * 0.9;
            cnNetWork.setLearnRate(learnRate);
            System.out.println("第 " + x + "次");
//            int correct = 0;
            //每次取一批数据
            int batch = 2000;
            for (int i = 0; i < list.size(); i = i + batch) {

                //将这一批数据，反复喂给
                for (int k = 0; k < 3; k++) {
                    for (int j = 0; j < batch; j++) {
                        Cifar10 cifar10 = list.get(i + j);
                        double[] expect = expectMNIST(cifar10.getLabel());
                        int result = cnNetWork.learn(cifar10.getRgb(), expect);
                    }
                }
                UtilCifar10.test(cnNetWork, listTest);
            }
            cnNetWork.save(strName);
        }


    }


    //手写识别成功率
    public static void testMNIST() {

        List<Cifar10> listTest = UtilMnist.testData();

        //对所有的数据进行归一化

        for (int i = 0; i < listTest.size(); i++) {
            UtilNeuralNet.initMinst(listTest.get(i).getRgb().getArray());
        }

        List<String> list = CNNetWork.loadAll("FC");
        for (String str : list) {
            //加载神经网络
            CNNetWork netWork = CNNetWork.load(str);
            UtilCifar10.test(netWork, listTest);
        }
    }


    //当两个都为1是才为1
    public static void and() {
        CNNetWork netWork = CNNetWork.load("and_cnn");
//        CNNetWork netWork = new CNNetWork();

//        netWork.addLayer(new FullLayer("full1", "sigmoid", 2));
//        netWork.addLayer(new FullLayer("full2", "sigmoid", 1));


        if (netWork == null) {
            return;
        }

        Tensor tensor11 = new Tensor();
        tensor11.setHeight(1);
        tensor11.setWidth(2);
        tensor11.createArray();
        tensor11.set(0, 1);
        tensor11.set(1, 1);

        Tensor tensor10 = new Tensor();
        tensor10.setHeight(1);
        tensor10.setWidth(2);
        tensor10.createArray();
        tensor10.set(0, 1);
        tensor10.set(1, 0);


        Tensor tensor01 = new Tensor();
        tensor01.setHeight(1);
        tensor01.setWidth(2);
        tensor01.createArray();
        tensor01.set(0, 0);
        tensor01.set(1, 1);

        Tensor tensor00 = new Tensor();
        tensor00.setHeight(1);
        tensor00.setWidth(2);
        tensor00.createArray();
        tensor00.set(0, 0);
        tensor00.set(1, 0);


        for (int i = 1; i < 1000; i++) {
            netWork.learn(tensor11, new double[]{1});
            System.out.println("error-11----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor10, new double[]{0});
            System.out.println("error-10----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor01, new double[]{0});
            System.out.println("error-01----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor00, new double[]{0});
            System.out.println("error-00----------------------------------------------" + netWork.totalError());


            double[] out = netWork.work(tensor11);
            System.out.println("11 = " + JSON.toJSONString(out));

            out = netWork.work(tensor10);
            System.out.println("10 = " + JSON.toJSONString(out));

            out = netWork.work(tensor01);
            System.out.println("01 = " + JSON.toJSONString(out));

            out = netWork.work(tensor00);
            System.out.println("00 = " + JSON.toJSONString(out));
        }

        netWork.save("and_cnn");

    }


    public static void xor() {
//        CNNetWork netWork = CNNetWork.load("xor_cnn");
        CNNetWork netWork = new CNNetWork();
//
        netWork.addLayer(new FullLayer("full1", "sigmoid", 2));
        netWork.addLayer(new FullLayer("full2", "sigmoid", 1));


        if (netWork == null) {
            return;
        }

        Tensor tensor11 = new Tensor();
        tensor11.setHeight(1);
        tensor11.setWidth(2);
        tensor11.createArray();
        tensor11.set(0, 1);
        tensor11.set(1, 1);

        Tensor tensor10 = new Tensor();
        tensor10.setHeight(1);
        tensor10.setWidth(2);
        tensor10.createArray();
        tensor10.set(0, 1);
        tensor10.set(1, 0);


        Tensor tensor01 = new Tensor();
        tensor01.setHeight(1);
        tensor01.setWidth(2);
        tensor01.createArray();
        tensor01.set(0, 0);
        tensor01.set(1, 1);

        Tensor tensor00 = new Tensor();
        tensor00.setHeight(1);
        tensor00.setWidth(2);
        tensor00.createArray();
        tensor00.set(0, 0);
        tensor00.set(1, 0);


        for (int i = 1; i < 1000; i++) {
            netWork.learn(tensor11, new double[]{0});
            System.out.println("error-11----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor10, new double[]{1});
            System.out.println("error-10----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor01, new double[]{1});
            System.out.println("error-01----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor00, new double[]{0});
            System.out.println("error-00----------------------------------------------" + netWork.totalError());


            double[] out = netWork.work(tensor11);
            System.out.println("11 = " + JSON.toJSONString(out));

            out = netWork.work(tensor10);
            System.out.println("10 = " + JSON.toJSONString(out));

            out = netWork.work(tensor01);
            System.out.println("01 = " + JSON.toJSONString(out));

            out = netWork.work(tensor00);
            System.out.println("00 = " + JSON.toJSONString(out));
        }

        netWork.save("xor_cnn");

    }


    public static void and_or_xor() {
//        CNNetWork netWork = CNNetWork.load("and_or_xor_cnn");
        CNNetWork netWork = new CNNetWork();
//
        netWork.addLayer(new FullLayer("full1", "sigmoid", 2));
        //用二次方，果然慢了很多
//        netWork.addLayer(new FullLayer("full2", "sigmoid", 3));
        //用交叉熵，果然快了很多
        netWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 3));


        if (netWork == null) {
            return;
        }

        Tensor tensor11 = new Tensor();
        tensor11.setHeight(1);
        tensor11.setWidth(2);
        tensor11.createArray();
        tensor11.set(0, 1);
        tensor11.set(1, 1);

        Tensor tensor10 = new Tensor();
        tensor10.setHeight(1);
        tensor10.setWidth(2);
        tensor10.createArray();
        tensor10.set(0, 1);
        tensor10.set(1, 0);


        Tensor tensor01 = new Tensor();
        tensor01.setHeight(1);
        tensor01.setWidth(2);
        tensor01.createArray();
        tensor01.set(0, 0);
        tensor01.set(1, 1);

        Tensor tensor00 = new Tensor();
        tensor00.setHeight(1);
        tensor00.setWidth(2);
        tensor00.createArray();
        tensor00.set(0, 0);
        tensor00.set(1, 0);


        for (int i = 1; i < 100; i++) {
            System.out.println("i = ---------------------------------------" + i);
            netWork.learn(tensor11, new double[]{1, 1, 0});
            System.out.println("error-11----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor10, new double[]{0, 1, 1});
            System.out.println("error-10----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor01, new double[]{0, 1, 1});
            System.out.println("error-01----------------------------------------------" + netWork.totalError());

            netWork.learn(tensor00, new double[]{0, 0, 0});
            System.out.println("error-00----------------------------------------------" + netWork.totalError());


            double[] out = netWork.work(tensor11);
            System.out.println("11 = " + JSON.toJSONString(out));

            out = netWork.work(tensor10);
            System.out.println("10 = " + JSON.toJSONString(out));

            out = netWork.work(tensor01);
            System.out.println("01 = " + JSON.toJSONString(out));

            out = netWork.work(tensor00);
            System.out.println("00 = " + JSON.toJSONString(out));
        }

        netWork.save("and_or_xor_cnn");

    }

}
