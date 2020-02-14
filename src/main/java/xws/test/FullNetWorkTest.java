package xws.test;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import xws.neuron.FullNetWork;
import xws.neuron.MnistRead;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.util.UtilFile;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 神经网络的测试
 * Created by xws on 2019/1/22.
 */
public class FullNetWorkTest {

    public static String bimonthlyA = "[{x:174,y:111},{x:167.94444444444446,y:123.36777777777777},{x:162.57777777777778,y:135.2711111111111},{x:157.9,y:146.71},{x:153.9111111111111,y:157.68444444444444},{x:150.61111111111111,y:168.19444444444446},{x:148,y:178.24},{x:146.07777777777778,y:187.8211111111111},{x:144.84444444444443,y:196.9377777777778},{x:144.29999999999998,y:205.58999999999997},{x:144.44444444444443,y:213.77777777777777},{x:145.27777777777777,y:221.50111111111113},{x:146.8,y:228.76000000000002},{x:149.01111111111112,y:235.55444444444444},{x:151.9111111111111,y:241.88444444444445},{x:155.5,y:247.75},{x:159.77777777777777,y:253.1511111111111},{x:164.74444444444444,y:258.0877777777778},{x:170.4,y:262.56},{x:176.74444444444444,y:266.56777777777774},{x:183.77777777777777,y:270.11111111111114},{x:191.5,y:273.19},{x:199.9111111111111,y:275.8044444444444},{x:209.01111111111112,y:277.95444444444445},{x:218.8,y:279.64},{x:229.27777777777783,y:280.8611111111111},{x:240.44444444444446,y:281.6177777777778},{x:252.3,y:281.91},{x:264.84444444444443,y:281.73777777777775},{x:278.0777777777778,y:281.10111111111115},{x:292,y:280},{x:174,y:111},{x:168.65333333333334,y:122.9811111111111},{x:163.9466666666667,y:134.52444444444444},{x:159.88,y:145.63000000000002},{x:156.45333333333332,y:156.29777777777778},{x:153.66666666666669,y:166.52777777777777},{x:151.52,y:176.32},{x:150.01333333333332,y:185.67444444444445},{x:149.14666666666668,y:194.5911111111111},{x:148.92000000000002,y:203.07},{x:149.33333333333331,y:211.1111111111111},{x:150.38666666666668,y:218.71444444444444},{x:152.07999999999998,y:225.88},{x:154.41333333333333,y:232.60777777777778},{x:157.38666666666668,y:238.89777777777778},{x:161,y:244.75},{x:165.25333333333333,y:250.16444444444446},{x:170.14666666666668,y:255.14111111111112},{x:175.68,y:259.68}]";
    public static String bimonthlyB = "[{x:182,y:221},{x:192.41222222222223,y:219.35444444444445},{x:202.4488888888889,y:218.1511111111111},{x:212.11,y:217.39},{x:221.39555555555555,y:217.07111111111112},{x:230.30555555555554,y:217.19444444444446},{x:238.84,y:217.76},{x:246.9988888888889,y:218.76777777777778},{x:254.78222222222223,y:220.21777777777777},{x:262.19,y:222.11},{x:269.22222222222223,y:224.44444444444443},{x:275.87888888888887,y:227.2211111111111},{x:282.15999999999997,y:230.44},{x:288.0655555555556,y:234.10111111111112},{x:293.59555555555556,y:238.20444444444445},{x:298.75,y:242.75},{x:303.5288888888889,y:247.73777777777778},{x:307.9322222222222,y:253.16777777777776},{x:311.96,y:259.04},{x:315.6122222222222,y:265.3544444444444},{x:318.88888888888886,y:272.1111111111111},{x:321.79,y:279.31},{x:324.3155555555556,y:286.9511111111111},{x:326.4655555555555,y:295.0344444444445},{x:328.24,y:303.56},{x:329.6388888888889,y:312.52777777777777},{x:330.6622222222222,y:321.9377777777778},{x:331.31,y:331.78999999999996},{x:331.58222222222224,y:342.08444444444444},{x:331.4788888888889,y:352.8211111111111},{x:331,y:364},{x:182,y:221},{x:192.79888888888888,y:218.9677777777778},{x:203.19555555555556,y:217.40444444444444},{x:213.19,y:216.31},{x:222.78222222222223,y:215.68444444444444},{x:231.97222222222223,y:215.52777777777777},{x:240.76,y:215.84},{x:249.14555555555555,y:216.6211111111111},{x:257.12888888888887,y:217.8711111111111},{x:264.71,y:219.59},{x:271.8888888888889,y:221.77777777777777},{x:278.66555555555556,y:224.43444444444444},{x:285.04,y:227.56},{x:291.01222222222225,y:231.15444444444444},{x:296.58222222222224,y:235.21777777777777},{x:301.75,y:239.75},{x:306.5155555555555,y:244.7511111111111},{x:310.87888888888887,y:250.22111111111113},{x:314.84,y:256.15999999999997}]";

    public static void main(String[] args) {
//        t1();
//        and();
//        or();
//        xor();
//        and_or_xor();
//        x();

//        bimonthly();
//        t1();
//        testLoadFile();
//        testSaveFile();
//        init();
        learnMNIST();//训练手写字符识别
//        testMNIST();//识别手写字符

//        MNIST();//保存整理测试数据
//        loadMNIST();//测试加载测试数据
//        double b = 4.306488082239371E-10;
//        System.out.println(b>0.0000000001);

    }

    //1-1-1
    public static void t1() {
        FullNetWork netWork = new FullNetWork("1-2-1");
//        FullNetWork netWork = FullNetWork.load("1-2-1");
        for (int i = 1; i < 1000; i++) {
            double[] input = new double[]{1};
            netWork.forward(input, input);
            System.out.println("误差-----------------------------------------------" + netWork.error());

            double[] out = netWork.work(input);
            System.out.println("out = " + JSON.toJSONString(out));
        }

        netWork.save("1-1-1");


        double[] input = new double[]{1};
        double[] out = netWork.work(input);
        System.out.println("out = " + JSON.toJSONString(out));
    }

    //当两个都为1是才为1
    public static void and() {
        FullNetWork netWork =
                FullNetWork.load("2-1");

        if (netWork == null) {
            return;
        }


        double[] input11 = new double[]{1, 1};
        double[] input10 = new double[]{1, 0};
        double[] input01 = new double[]{0, 1};
        double[] input00 = new double[]{0, 0};

        for (int i = 1; i < 1000; i++) {
            netWork.forward(input11, new double[]{1});
            System.out.println("error-11----------------------------------------------" + netWork.error());

            netWork.forward(input10, new double[]{0});
            System.out.println("error-10----------------------------------------------" + netWork.error());

            netWork.forward(input01, new double[]{0});
            System.out.println("error-01----------------------------------------------" + netWork.error());

            netWork.forward(input00, new double[]{0});
            System.out.println("error-00----------------------------------------------" + netWork.error());


            double[] out = netWork.work(input11);
            System.out.println("11 = " + JSON.toJSONString(out));

            out = netWork.work(input10);
            System.out.println("10 = " + JSON.toJSONString(out));

            out = netWork.work(input01);
            System.out.println("01 = " + JSON.toJSONString(out));

            out = netWork.work(input00);
            System.out.println("00 = " + JSON.toJSONString(out));
        }

        netWork.save("and");


    }

    public static void or() {
        FullNetWork netWork =
//                new FullNetWork("2-1");
                FullNetWork.load("or");

        if (netWork == null) {
            return;
        }


        double[] input11 = new double[]{1, 1};
        double[] input10 = new double[]{1, 0};
        double[] input01 = new double[]{0, 1};
        double[] input00 = new double[]{0, 0};

        for (int i = 1; i < 1000; i++) {
            netWork.forward(input11, new double[]{1});
            System.out.println("error-11----------------------------------------------" + netWork.error());

            netWork.forward(input10, new double[]{1});
            System.out.println("error-10----------------------------------------------" + netWork.error());

            netWork.forward(input01, new double[]{1});
            System.out.println("error-01----------------------------------------------" + netWork.error());

            netWork.forward(input00, new double[]{0});
            System.out.println("error-00----------------------------------------------" + netWork.error());


            double[] out = netWork.work(input11);
            System.out.println("11 = " + JSON.toJSONString(out));

            out = netWork.work(input10);
            System.out.println("10 = " + JSON.toJSONString(out));

            out = netWork.work(input01);
            System.out.println("01 = " + JSON.toJSONString(out));

            out = netWork.work(input00);
            System.out.println("00 = " + JSON.toJSONString(out));
        }

        netWork.save("or");


    }

    public static void xor() {
        String name = "2-2-1";
//        FullNetWork netWork = new FullNetWork(name);
        FullNetWork netWork = FullNetWork.load(name);

        if (netWork == null) {
            return;
        }


        double[] input11 = new double[]{1, 1};
        double[] input10 = new double[]{1, 0};
        double[] input01 = new double[]{0, 1};
        double[] input00 = new double[]{0, 0};

        for (int i = 1; i < 100; i++) {
            netWork.forward(input11, new double[]{0});
            System.out.println("error-11----------------------------------------------" + netWork.error());

            netWork.forward(input10, new double[]{1});
            System.out.println("error-10----------------------------------------------" + netWork.error());

            netWork.forward(input01, new double[]{1});
            System.out.println("error-01----------------------------------------------" + netWork.error());

            netWork.forward(input00, new double[]{0});
            System.out.println("error-00----------------------------------------------" + netWork.error());


            double[] out = netWork.work(input11);
            System.out.println("11 = " + JSON.toJSONString(out));

            out = netWork.work(input10);
            System.out.println("10 = " + JSON.toJSONString(out));

            out = netWork.work(input01);
            System.out.println("01 = " + JSON.toJSONString(out));

            out = netWork.work(input00);
            System.out.println("00 = " + JSON.toJSONString(out));
        }

        netWork.save(name);


    }

    public static void and_or_xor() {
        FullNetWork netWork =
// new FullNetWork("2-2-3");
                FullNetWork.load("and_or_xor_fc");

        if (netWork == null) {
            return;
        }


        double[] input11 = new double[]{1, 1};
        double[] input10 = new double[]{1, 0};
        double[] input01 = new double[]{0, 1};
        double[] input00 = new double[]{0, 0};

        for (int i = 1; i < 1; i++) {
            netWork.forward(input11, new double[]{1, 1, 0});
            System.out.println("error-11----------------------------------------------" + netWork.error());

            netWork.forward(input10, new double[]{0, 1, 1});
            System.out.println("error-10----------------------------------------------" + netWork.error());

            netWork.forward(input01, new double[]{0, 1, 1});
            System.out.println("error-01----------------------------------------------" + netWork.error());

            netWork.forward(input00, new double[]{0, 0, 0});
            System.out.println("error-00----------------------------------------------" + netWork.error());


            double[] out = netWork.work(input11);
            System.out.println("11 = " + JSON.toJSONString(out));

            out = netWork.work(input10);
            System.out.println("10 = " + JSON.toJSONString(out));

            out = netWork.work(input01);
            System.out.println("01 = " + JSON.toJSONString(out));

            out = netWork.work(input00);
            System.out.println("00 = " + JSON.toJSONString(out));
        }

//        netWork.save("and_or_xor_fc");


    }


    //线性函数的逼近
    public static void x() {
        FullNetWork netWork =
//                new FullNetWork("1-1-1");
                FullNetWork.load("x");

        if (netWork == null) {
            return;
        }

        for (int i = 1; i < 100; i++) {
            double[] input = new double[]{i};
            netWork.forward(input, input);
            System.out.println("error-11----------------------------------------------" + netWork.error());

            double[] out = netWork.work(input);
            System.out.println("11 = " + JSON.toJSONString(out));
        }

        netWork.save("x");


    }

    //双月数据集
    public static void bimonthly() {

        //首先，对双月数据进行归一化
        JSONArray listA = JSON.parseArray(bimonthlyA);
        JSONArray listB = JSON.parseArray(bimonthlyB);


        FullNetWork netWork =
//                new FullNetWork("2-2-2");
                FullNetWork.load("bimonthly");

        if (netWork == null) {
            return;
        }

        double[] A = new double[]{1, 0};
        double[] B = new double[]{0, 1};

        for (int k = 0; k < 1000; k++) {
            for (int i = 1; i < listB.size(); i++) {
                JSONObject item = listB.getJSONObject(i);
                double x = item.getDouble("x") / 1000;
                double y = item.getDouble("y") / 1000;

                double[] inputB = new double[]{x, y};
//                netWork.forward(inputB, B);
                System.out.println("error-B----------------------------------------------" + netWork.error());

                double[] out = netWork.work(inputB);
                System.out.println("B = " + JSON.toJSONString(out));


                item = listA.getJSONObject(i);
                x = item.getDouble("x") / 1000;
                y = item.getDouble("y") / 1000;
                double[] inputA = new double[]{x, y};

//                netWork.forward(inputA, A);
                System.out.println("error-A----------------------------------------------" + netWork.error());

                out = netWork.work(inputA);
                System.out.println("A = " + JSON.toJSONString(out));

            }
        }


        netWork.save("bimonthly");
    }

    //测试加载神经网络文件时，找不到文件的情况
    public static void testLoadFile() {
        int[] a = new int[]{1, 1, 1};
        FullNetWork netWork = FullNetWork.load("1-1-1");
        System.out.println("netWork = " + netWork);
    }

    //测试持久化后的神经网络
    public static void testSaveFile() {
        int[] a = new int[]{1, 2, 1};
        FullNetWork netWork = new FullNetWork(a);
        netWork.save("test");
        System.out.println();
    }

    //初始化并持久化神经网络
    public static void init() {

        FullNetWork netWork = new FullNetWork("1-1-1");
        netWork.save("1-1-1");

        netWork = new FullNetWork("1-2-1");
        netWork.save("1-2-1");

        netWork = new FullNetWork("1-2-2");
        netWork.save("1-2-2");

        netWork = new FullNetWork("2-2-1");
        netWork.save("2-2-1");

        netWork = new FullNetWork("2-2-2");
        netWork.save("2-2-2");


        netWork = new FullNetWork("1-1");
        netWork.save("1-1");

        netWork = new FullNetWork("2-1");
        netWork.save("2-1");

        netWork = new FullNetWork("3-1");
        netWork.save("3-1");

        netWork = new FullNetWork("4-1");
        netWork.save("4-1");

        netWork = new FullNetWork("1-2");
        netWork.save("1-2");

        netWork = new FullNetWork("1-3");
        netWork.save("1-3");

        netWork = new FullNetWork("1-4");
        netWork.save("1-4");

        netWork = new FullNetWork("2-2");
        netWork.save("2-2");


    }

    //测试手写数组识别
    public static void learnMNIST() {
        //首先构建神经网络
        String name = "784-16-16-10";


        //加载数据
        double[][] images = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
        double[] labels = MnistRead.getLabels(MnistRead.TRAIN_LABELS_FILE);

        for (int x = 0; x < 30; x++) {
//            FullNetWork netWork = FullNetWork.load(name);
            FullNetWork netWork = new FullNetWork(name);
            int correct = 0;
            int max = 20000;
            if (netWork == null) {
                return;
            }
            System.out.println("第 " + x + "次");
            for (int i = 0; i < 60000; i++) {
//                int index = (int) (Math.random() * 59500);
                double[] image = images[i];
                //数据归一化
                for (int k = 0; k < image.length; k++) {
                    image[k] = image[k] / 255;
                }

                //组织输出期望数据
                double val = labels[i];
                double[] expect = expectMNIST(val).getArray();
                double[] out = netWork.forward(image, expect);
                int maxIndex = UtilNeuralNet.maxIndex(out);
                if (maxIndex == (int) val) {
                    correct = correct + 1;
                }
            }

            System.out.println("识别正确率：" + (correct / (double) max));


            //到最后，使用自己提取的测试数据，测一下误差
//            Map<Double, double[]> test = loadMNIST();
//            for (Map.Entry<Double, double[]> entry : test.entrySet()) {
//                double[] input = entry.getValue();
//                for (int k = 0; k < input.length; k++) {
//                    input[k] = input[k] / 255;
//                }
//                double[] out = netWork.work(entry.getValue());
//
//                //将误差打印到文件中去
//                appendErrorData(entry.getKey() + "", JSON.toJSONString(out), netWork.getVersion() + "");
//                appendErrorDataDetail(entry.getKey() + "", JSON.toJSONString(out[entry.getKey().intValue()]), netWork.getVersion() + "");
//
//            }

            netWork.save(name);
        }


    }

    //提取出0-9的字符的图片和标示，放到json里面去，训练完的时候，拿来检测一下
    public static void MNIST() {
        double[][] images = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
        double[] labels = MnistRead.getLabels(MnistRead.TRAIN_LABELS_FILE);

        String fileName = "/Users/xws/Desktop/xws/MNIST.txt";
        Map<Double, Object> map = new HashMap<>();

        for (int i = 0; i < images.length; i++) {
            double[] image = images[i];
            double label = labels[i];
            if (map.size() == 10) {
                String json = JSON.toJSONString(map);
                UtilFile.writeFile(json, fileName);
                return;
            }

            if (!map.containsKey(label)) {
                map.put(label, image);
//                saveMNIST(image,label);
            }
        }
    }


    //加载我自己弄好的测试数据
    public static Map<Double, double[]> loadMNIST() {
        String fileName = "/Users/xws/Desktop/xws/MNIST.txt";
        String json = UtilFile.readFile(fileName);
        JSONObject jsonObject = JSON.parseObject(json);
        Map<Double, double[]> map = new HashMap<>();
        for (int i = 0; i < 10; i++) {
            jsonObject.getString(i + ".0");
            JSONArray array = jsonObject.getJSONArray(i + ".0");
            double[] arr = new double[array.size()];
            for (int k = 0; k < arr.length; k++) {
                arr[k] = array.getDoubleValue(k);
            }
            map.put((double) i, arr);
//            saveMNIST(arr,i);
        }
        return map;
    }

    //追加误差数据
    public static void appendErrorData(String key, String value, String version) {
        UtilFile utilFile = new UtilFile("/Users/xws/Desktop/xws/mnist/" + key + ".txt");
        utilFile.append(value + "\t\t" + version + "\n");
    }

    public static void appendErrorDataDetail(String key, String value, String version) {
        UtilFile utilFile = new UtilFile("/Users/xws/Desktop/xws/mnist2/" + key + ".txt");
        utilFile.append(value + "\t\t" + version + "\n");
    }

    //加工手写字符的期望数据
    public static Tensor expectMNIST(double val) {
        return expectMNIST(val, 10);
    }

    public static Tensor expectMNIST(double val, int total) {
        double[] expect = new double[total];
        int index = (int) val;
        expect[index] = 1;
        return new Tensor(expect);
    }

    //保存图片到文件中
    public static void saveMNIST(double[] arr, Object i) {
        try {
            MnistRead.drawGrayPicture(arr, 28, 28, "/Users/xws/Desktop/xws/mnist/" + i + ".jpg");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    //手写识别成功率
    public static void testMNIST() {
        //加载测试数据
        double[][] images = MnistRead.getImages(MnistRead.TEST_IMAGES_FILE);
        double[] labels = MnistRead.getLabels(MnistRead.TEST_LABELS_FILE);

        UtilFile utilFile = new UtilFile("/Users/xws/Desktop/xws/mnist/result.txt");

        List<String> list = FullNetWork.loadAll("784-16-16-10");
        for (String str : list) {
            //加载神经网络
            FullNetWork netWork = FullNetWork.load(str);
            //初始化识别正确的数量
            int right = 0;
            for (int i = 0; i < images.length; i++) {
                netWork.work(images[i]);
                double result = netWork.result();
                if (result == labels[i]) {
                    right = right + 1;
                }
            }
            //计算识别率
            double rate = right / (double) images.length;
            System.out.println("识别正确率：" + rate);
            //将识别的结果写到文件中去
            utilFile.append(rate + "\t\t" + netWork.getVersion() + "\n");
        }
    }


}
