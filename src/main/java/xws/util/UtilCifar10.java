package xws.util;

import xws.neuron.CNNetWork;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.test.FullNetWorkTest;

import java.io.FileInputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;


/**
 * cifar10 工具类
 * Created by xws on 2019/3/14.
 */
public class UtilCifar10 {

    private static String path = "/Users/xws/Desktop/data/cifar10/";
    private static UtilFile fileLog = new UtilFile("/Users/xws/Desktop/data/log.txt");

    //读取数据
    public static void loadData(String fileName, List<Cifar10> list) {


        try {
            FileInputStream fileInputStream = new FileInputStream(fileName);
            for (int j = 0; j < 10000; j++) {
                Cifar10 cifar10 = new Cifar10();
                int label = fileInputStream.read();
                cifar10.setLabel(UtilNeuralNet.oneHot(label));
                cifar10.setIndex(label);
                byte[] buffer = new byte[32 * 32 * 3];  //32*32*3
                fileInputStream.read(buffer);
                Tensor tensor = new Tensor();
                tensor.setDepth(3);
                tensor.setHeight(32);
                tensor.setWidth(32);
                tensor.createArray();
                double[] array = tensor.getArray();
                for (int i = 0; i < array.length; i++) {
                    array[i] = buffer[i];
                }
                cifar10.setRgb(tensor);
                list.add(cifar10);

            }
            fileInputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    //获取训练数据

    public static List<Cifar10> learnData() {
        List<Cifar10> list = new ArrayList<>();
        loadData(path + "data_batch_1.bin", list);
        loadData(path + "data_batch_2.bin", list);
        loadData(path + "data_batch_3.bin", list);
        loadData(path + "data_batch_4.bin", list);
        loadData(path + "data_batch_5.bin", list);
        return list;
    }
    //获取测试数据

    public static List<Cifar10> testData() {
        List<Cifar10> list = new ArrayList<>();
        loadData(path + "test_batch.bin", list);
        return list;
    }

    //测试 test
    public static double test(CNNetWork netWork, List<Cifar10> list) {
        netWork.setBatchSize(1);
        netWork.entryTest();
        //初始化识别正确的数量
        int right = 0;
        for (int i = 0; i < list.size(); i++) {
            Cifar10 cifar10 = list.get(i);
            double[] out = netWork.work(cifar10.getRgb());
            int maxIndex = UtilNeuralNet.maxIndex(out);
            if (maxIndex == cifar10.getIndex()) {
                right = right + 1;
            }
        }
        //计算识别率
        double rate = right / (double) list.size();
        System.out.println(UtilDate.dateToyyyyMMddHHmmss(new Date()) + "\t" + netWork.getName() + "\t" + netWork.getVersion() + "\t" + rate);
//        fileLog.append(UtilDate.dateToyyyyMMddHHmmss(new Date()) + "\t" + netWork.getName() + "\t" + netWork.getVersion() + "\t" + rate + "\n");
        return rate;
    }

    //测试rnn序列-mnist
    public static double testRnn(CNNetWork netWork, List<RnnSequence> list) {
        netWork.setBatchSize(1);
        netWork.entryTest();
        //初始化识别正确的数量
        int right = 0;
        for (int i = 0; i < list.size(); i++) {
            RnnSequence rnnSequence = list.get(i);
            netWork.setStep(0);
            for (int r = 0; r < rnnSequence.size() - 1; r++) {
                netWork.work(rnnSequence.getData(r));
            }
            double[] out = netWork.work(rnnSequence.getData(rnnSequence.size() - 1));
            double maxIndex = UtilNeuralNet.maxIndex(out);
            if (maxIndex == rnnSequence.get(rnnSequence.size() - 1).getValue()) {
                right = right + 1;
            }
        }
        //计算识别率
        double rate = right / (double) list.size();
        System.out.println(UtilDate.dateToyyyyMMddHHmmss(new Date()) + "\t" + netWork.getName() + "\t" + netWork.getVersion() + "\t" + rate);
//        fileLog.append(UtilDate.dateToyyyyMMddHHmmss(new Date()) + "\t" + netWork.getName() + "\t" + netWork.getVersion() + "\t" + rate + "\n");
        return rate;
    }

    //测试所有
    public static void testAll(String strName, List<Cifar10> list) {
        List<String> listNet = CNNetWork.loadAll(strName);
        for (String str : listNet) {
            //加载神经网络
            CNNetWork netWork = CNNetWork.load(str);
            UtilCifar10.test(netWork, list);
        }
    }


    public static void main(String[] args) {
        List<Cifar10> list = testData();
        System.out.println(list);
    }
}
