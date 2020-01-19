package xws.test;

import com.alibaba.fastjson.JSON;
import xws.neuron.CNNetWork;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.*;
import xws.neuron.layer.rnn.RnnLayer;
import xws.util.Cifar10;
import xws.util.RnnSequence;
import xws.util.UtilCifar10;
import xws.util.UtilMnist;

import java.util.ArrayList;
import java.util.List;

import static xws.test.FullNetWorkTest.oneHot;


/**
 * 循环神经网络测试
 * Created by xws on 2019/5/14.
 */
public class RnnTest {

    private static String strName = "RNN_ADD";

    private static List<Cifar10> list = createData();

    public static void main(String[] args) {
//        createCNNetWork();
//        learnMNIST();
//        XOR();
//        ADD();
        MNIST();
    }


    //第1个数+第2个数+第3个数+第n个数
    public static void createCNNetWork() {
        CNNetWork cnNetWork = new CNNetWork();
//        cnNetWork.addLayer(new BnLayer("bn1"));
        cnNetWork.addLayer(new RnnLayer("rnn1", "sigmoid", 3));
//        cnNetWork.addLayer(new BnLayer("bn2"));
        cnNetWork.addLayer(new FullLayer("full2", 1, UtilNeuralNet.e() * 0.00000000001));

        Cifar10 cifar10 = list.get(0);
        cnNetWork.entryTest();
        cnNetWork.setStep(0);
        for (int i = 0; i < list.size(); i++) {
            cnNetWork.learn(cifar10.getRgb(), new Tensor(new double[]{cifar10.getValue()}));
        }

        cnNetWork.save(strName);


    }

    public static List<Cifar10> createData() {
        double total = 0;
        List<Cifar10> list = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            Tensor tensor = new Tensor();
            tensor.setDepth(1);
            tensor.setHeight(1);
            tensor.setWidth(1);
            tensor.createArray();
            Cifar10 cifar10 = new Cifar10();
            cifar10.setRgb(tensor);
            double val = Math.random();
            tensor.set(0, val);
            total = total + tensor.get(0);
            cifar10.setValue(total);
            list.add(cifar10);
        }

        return list;
    }

    //创建mnist序列
    public static List<RnnSequence> createSequenceMNIST(List<Cifar10> list) {
        List<RnnSequence> rnnList = new ArrayList<>();
        for (int i = 0; i < list.size(); i++) {
            Cifar10 cifar10 = list.get(i);
//            UtilNeuralNet.initMinst(cifar10.getRgb().getArray());
            RnnSequence rnnSequence = new RnnSequence();
            for (int k = 0; k < 28; k++) {
                rnnSequence.add(cifar10.getRgb().data(k), cifar10.getIndex());
            }
            rnnList.add(rnnSequence);
        }
        return rnnList;
    }

    //mnist
    public static void MNIST() {
        List<RnnSequence> list = createSequenceMNIST(UtilMnist.learnData());
        List<RnnSequence> listTest = createSequenceMNIST(UtilMnist.testData());

        CNNetWork cnNetWork = CNNetWork.load("RNN_MNIST_LN");
//        CNNetWork cnNetWork = new CNNetWork();
        //93.92%
        //95.52%
//        cnNetWork.addLayer(new RnnLayer("rnn1", "relu", 28));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 32, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new SoftMaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));
//        double learnRate = UtilNeuralNet.e() * 0.00001;
//        double learnRate = UtilNeuralNet.e() * 0.000001;
//        double learnRate = UtilNeuralNet.e() * 0.0000001;
        double learnRate = UtilNeuralNet.e() * 0.00000001;


//
        cnNetWork.entryLearn();
        cnNetWork.setBatchSize(1);
        cnNetWork.setLearnRate(learnRate);

        int batch = 2000;
        for (int i = 0; i < list.size(); i = i + batch) {
            System.out.println("i = " + i);
            //将这一批数据，反复喂给
            for (int k = 0; k < 3; k++) {
                cnNetWork.entryLearn();
                for (int j = 0; j < batch; j++) {
                    RnnSequence rnnSequence = list.get(i + j);

                    for (int n = 0; n < 10; n++) {
                        cnNetWork.setStep(0);
                        for (int r = 0; r < rnnSequence.size() - 1; r++) {
                            cnNetWork.learn(rnnSequence.getData(r), null);
                        }
                        cnNetWork.learn(rnnSequence.getData(27), FullNetWorkTest.oneHot(rnnSequence.get(27).getValue()));
                    }
                }

                UtilCifar10.testRnn(cnNetWork, listTest);
            }
        }

        cnNetWork.save("RNN_MNIST_LN");

    }

    //创建加法序列
    public static List<RnnSequence> createSequenceADD() {
        List<RnnSequence> list = new ArrayList<>();

        for (int k = 0; k < 100; k++) {
            double total = 0;
            RnnSequence rnnSequence = new RnnSequence();

            for (int i = 0; i < 5; i++) {
                double val = Math.random();
                total = total + val;
                rnnSequence.add(new double[]{val}, total);
            }
            list.add(rnnSequence);
        }

        return list;
    }

    public static void ADD() {
        List<RnnSequence> list = createSequenceADD();
        CNNetWork cnNetWork = CNNetWork.load("RNN_ADD");
//        CNNetWork cnNetWork = new CNNetWork();
//        cnNetWork.addLayer(new RnnLayer("rnn1", "relu", 4));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 1, UtilNeuralNet.e() * 0.00000000001));

        cnNetWork.entryLearn();
        cnNetWork.setBatchSize(1);
        double learnRate = UtilNeuralNet.e() * 0.0001;
        cnNetWork.setLearnRate(learnRate);
        for (int e = 0; e < 1000; e++) {

            System.out.println("e = " + e);
            for (int i = 0; i < list.size(); i++) {
                RnnSequence rnnSequence = list.get(i);

                for (int k = 0; k < 5; k++) {
                    cnNetWork.setStep(0);
                    cnNetWork.learn(rnnSequence.getData(0), new Tensor(rnnSequence.getExpect(0)));
                    cnNetWork.learn(rnnSequence.getData(1), new Tensor(rnnSequence.getExpect(1)));
                    cnNetWork.learn(rnnSequence.getData(2), new Tensor(rnnSequence.getExpect(2)));
                    cnNetWork.learn(rnnSequence.getData(3), new Tensor(rnnSequence.getExpect(3)));
                    cnNetWork.learn(rnnSequence.getData(4), new Tensor(rnnSequence.getExpect(4)));
                }

            }
        }

        cnNetWork.entryTest();
        //测试
        for (int i = 0; i < list.size(); i++) {
            cnNetWork.setStep(0);
            RnnSequence rnnSequence = list.get(i);
            double[] result;
            cnNetWork.work(rnnSequence.getData(0));
            cnNetWork.work(rnnSequence.getData(1));
            cnNetWork.work(rnnSequence.getData(2));
            cnNetWork.work(rnnSequence.getData(3));
            result = cnNetWork.work(rnnSequence.getData(4));
            System.out.println("结果 = " + JSON.toJSONString(result) + " 期望值 = " + JSON.toJSONString(rnnSequence.getExpect(4)));

        }
        cnNetWork.save("RNN_ADD");

    }

    //创建异或的序列
    public static List<RnnSequence> createSequenceXOR() {
        List<RnnSequence> list = new ArrayList<>();
        RnnSequence rnnSequence_00 = new RnnSequence();
        rnnSequence_00.add(new double[]{0}, 0);
        rnnSequence_00.add(new double[]{0}, 0);
        list.add(rnnSequence_00);


        RnnSequence rnnSequence_01 = new RnnSequence();
        rnnSequence_01.add(new double[]{0}, 0);
        rnnSequence_01.add(new double[]{1}, 1);
        list.add(rnnSequence_01);

        RnnSequence rnnSequence_10 = new RnnSequence();
        rnnSequence_10.add(new double[]{1}, 0);
        rnnSequence_10.add(new double[]{0}, 1);
        list.add(rnnSequence_10);

        RnnSequence rnnSequence_11 = new RnnSequence();
        rnnSequence_11.add(new double[]{1}, 0);
        rnnSequence_11.add(new double[]{1}, 0);
        list.add(rnnSequence_11);
        return list;
    }


    //异或运算
    public static void XOR() {
        List<RnnSequence> list = createSequenceXOR();

        CNNetWork cnNetWork = CNNetWork.load("RNN_XOR");
//        CNNetWork cnNetWork = new CNNetWork();
//        cnNetWork.addLayer(new RnnLayer("rnn1", "sigmoid", 2));
//        cnNetWork.addLayer(new FullLayer("full2", "sigmoid", 1, UtilNeuralNet.e() * 0.00000000001));

        cnNetWork.entryLearn();
        cnNetWork.setBatchSize(1);
        double learnRate = UtilNeuralNet.e() * 0.1;
        cnNetWork.setLearnRate(learnRate);
        for (int e = 0; e < 10000; e++) {

            for (int i = 0; i < list.size(); i++) {
                RnnSequence rnnSequence = list.get(i);
                cnNetWork.setStep(0);
                cnNetWork.learn(rnnSequence.getData(0), new Tensor(rnnSequence.getExpect(0)));
                cnNetWork.learn(rnnSequence.getData(1), new Tensor(rnnSequence.getExpect(1)));

                cnNetWork.setStep(0);
                cnNetWork.learn(rnnSequence.getData(0), new Tensor(rnnSequence.getExpect(0)));
                cnNetWork.learn(rnnSequence.getData(1), new Tensor(rnnSequence.getExpect(1)));


                cnNetWork.setStep(0);
                cnNetWork.learn(rnnSequence.getData(0), new Tensor(rnnSequence.getExpect(0)));
                cnNetWork.learn(rnnSequence.getData(1), new Tensor(rnnSequence.getExpect(1)));


                cnNetWork.setStep(0);
                cnNetWork.learn(rnnSequence.getData(0), new Tensor(rnnSequence.getExpect(0)));
                cnNetWork.learn(rnnSequence.getData(1), new Tensor(rnnSequence.getExpect(1)));


                cnNetWork.setStep(0);
                cnNetWork.learn(rnnSequence.getData(0), new Tensor(rnnSequence.getExpect(0)));
                cnNetWork.learn(rnnSequence.getData(1), new Tensor(rnnSequence.getExpect(1)));


            }
        }

        cnNetWork.entryTest();
        //测试
        for (int i = 0; i < list.size(); i++) {
            cnNetWork.setStep(0);
            RnnSequence rnnSequence = list.get(i);
            double[] result;
            cnNetWork.work(rnnSequence.getData(0));
            result = cnNetWork.work(rnnSequence.getData(1));
            System.out.println("结果 = " + JSON.toJSONString(result) + " 期望值 = " + JSON.toJSONString(rnnSequence.getExpect(1)));

        }
        cnNetWork.save("RNN_XOR");

    }
}
