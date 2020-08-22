package xws.neuron;

import com.alibaba.fastjson.JSON;

import java.util.Random;

/**
 * 神经网络的工具类
 * Created by xws on 2019/2/19.
 */
public class UtilNeuralNet {

    public static double alpha = 0000000000000000000000000000001D;

    //扩展输入数据 如果步幅导致不能出现数据边界有未处理数据
    public static double[][] extension(double[][] input, int height, int width, int strideX, int strideY) {
        int widthInput = input.length;
        int heightInput = input[0].length;

        //如果filter的 widthInput/(width+strideX)!=0 那么，就需要进行数据补充，以实现全部的卷积化

        int modx = (widthInput - width) % strideX;
        int mody = (heightInput - height) % strideY;


        //新的width和height
        int widthNew = widthInput;
        int heightNew = heightInput;

        if (modx != 0) {
            widthNew = widthInput + strideX - modx;
            System.out.println("输入数据进行宽度扩展了");
        }

        if (mody != 0) {
            heightNew = heightInput + strideY - mody;
            System.out.println("输入数据进行高度扩展了");
        }


        //开辟新的输入数组
        double[][] newInput = new double[widthNew][heightNew];
        //然后开始复制数据
        for (int h = 0; h < heightInput; h++) {
            for (int w = 0; w < widthInput; w++) {
                newInput[h][w] = input[h][w];
            }
        }

        return newInput;
    }


    //卷积之前，对数据进行扩展
    public static double[][] padding(double[][] input, int padding) {

        int heightInput = input.length;
        int widthInput = input[0].length;


        //新的width和height
        int widthNew = widthInput + padding * 2;
        int heightNew = heightInput + padding * 2;


        //开辟新的输入数组
        double[][] newInput = new double[heightNew][widthNew];


        //然后开始复制数据
        for (int h = 0; h < heightInput; h++) {
            for (int w = 0; w < widthInput; w++) {
                newInput[h + padding][w + padding] = input[h][w];
            }
        }

        return newInput;
    }

    //1*1卷积运算
    public static double[][] convolution(double[][] input, double[][] input1) {
        double[][] result = new double[input.length][input[0].length];
        for (int h = 0; h < input.length; h++) {
            for (int w = 0; w < input[h].length; w++) {
                result[h][w] = input[h][w] + input1[h][w];
            }
        }
        return result;
    }


    //计算卷积之后的数据高度
    public static int afterHeight(int inputHeight, int padding, int strideY, int height) {
        return (inputHeight + 2 * padding - height) / strideY + 1;
    }

    //计算卷积之后的数据宽度
    public static int afterWidth(int inputWidth, int padding, int strideX, int width) {
        return (inputWidth + 2 * padding - width) / strideX + 1;
    }

    //宇宙常数e
    public static double e() {
        return 2.71828182845904523536;
    }

    public static double rate() {
        return e() * 0.0001;
    }

    //初始化单个数据
    public static double initData() {
        Random random = new Random();
        return random.nextGaussian() * 0.1;
    }

    //初始化bias
    public static void initBias(double[] bias) {
        for (int i = 0; i < bias.length; i++) {
            bias[i] = 0.0010000000474974513;
        }
    }

    //初始化weight
    public static void initWeight(double[] weight) {
        Random random = new Random();
        for (int i = 0; i < weight.length; i++) {
            weight[i] = (random.nextGaussian() * 0.1);
        }
    }

    //初始化手写字符数据
    public static void initMinst(double[] data) {
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] / 128d - 1d;
//            data[i] = data[i] / 256;
        }
    }

    //初始化stock数据
    public static void initStock(double[] data) {
//        double max = max(data);

        for (int i = 0; i < data.length; i++) {
//            data[i] = (data[i] / max) * 10;
            data[i] = data[i] / 128d - 1d;

        }


//        for (int i = 0; i < data.length; i++) {
//            data[i] = (data[i] - min) / (max - min);
//        }
    }

    //除以最大值，进行归一化
    public static void initMax(double[] data) {
        double max = max(data);
        for (int i = 0; i < data.length; i++) {
            data[i] = data[i] / max;
        }
    }

    // 标准化
    public void normalization(double[] data) {
        double max = max(data);
        double min = min(data);
        for (int i = 0; i < data.length; i++) {
            data[i] = (data[i] - min) / (max - min);
        }
    }

    // 标准化-平均
    public void normalization_mean(double[] data) {
        double max = max(data);
        double min = min(data);
        double mean = average(data);

        for (int i = 0; i < data.length; i++) {
            data[i] = (data[i] - mean) / (max - min);
        }
    }

    // 归一化
    public void standardization(double[] data) {
        double average = average(data);
        double variance = variance(data, average);
        double standard = Math.sqrt(variance + 0.000000000001);

        for (int i = 0; i < data.length; i++) {
            data[i] = (data[i] - average) / standard;
        }
    }


    //求均值
    public static double average(double[] val) {
        double total = 0;
        for (int i = 0; i < val.length; i++) {
            total = total + val[i];
        }
        return total / val.length;
    }

    //求方差
    public static double variance(double[] val) {
        double total = 0;
        double average = average(val);
        for (int i = 0; i < val.length; i++) {
            total = total + Math.pow(val[i] - average, 2);
        }
        return total / val.length;
    }


    //求方差
    public static double variance(double[] val, double average) {
        double total = 0;
        for (int i = 0; i < val.length; i++) {
            total = total + Math.pow(val[i] - average, 2);
        }
        return total / val.length;
    }

    //取最大值
    public static double max(double[] array) {
        double max = 0;
        for (int i = 0; i < array.length; i++) {
            max = Math.max(max, array[i]);
        }
        return max;
    }

    //取最小值
    public static double min(double[] array) {
        double min = 9999;
        for (double d : array) {
            if (d < min) {
                min = d;
            }
        }
        return min;
    }

    //取最大值所在索引
    public static int maxIndex(double[] array) {
        double max = 0;
        int maxIndex = 0;
        for (int i = 0; i < array.length; i++) {
            if (max < array[i]) {
                max = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    //取数据概率
    public static double[] probability(double[] array) {
        double max = max(array);

        //求每个元素的log值
        double[] log = new double[array.length];
        for (int i = 0; i < log.length; i++) {
            log[i] = Math.exp(array[i] - max);
        }

        double total = 0;
        //求分母
        for (int i = 0; i < log.length; i++) {
            total = total + log[i];
        }

        //求概率
        for (int i = 0; i < log.length; i++) {
            log[i] = log[i] / total;
        }

        return log;
    }

    //交并比
    public static void iou() {

    }

    // 检测单个double的值是否合理
    public static boolean checkDouble(double d) {
        if (!Double.isFinite(d) || Double.isInfinite(d) || Double.isNaN(d)) {
            return true;
        }
        return false;
    }

    // 检测单个double数组的值是否合理
    public static boolean checkDouble(double[] doubles) {
        for (int i = 0; i < doubles.length; i++) {
            if (checkDouble(doubles)) {
                return true;
            }
        }
        return false;
    }

    // 检测单个tensor的值是否合理
    public static boolean checkDouble(Tensor tensor) {
        for (int i = 0; i < tensor.getArray().length; i++) {
            if (checkDouble(tensor.getArray()[i])) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) {

        double[] a = new double[]{2, 2, 3, 3};
        double[] b = new double[]{0, 0, 5, 5};

        System.out.println(average(a));
        System.out.println(average(b));
        System.out.println(variance(a));
        System.out.println(variance(b));

        double[] weight = new double[10];
        initWeight(weight);
        System.out.println(JSON.toJSONString(weight));

        System.out.println("test double " + checkDouble(12.12));
        System.out.println("test double " + (Double.NaN * 10));


    }


}
