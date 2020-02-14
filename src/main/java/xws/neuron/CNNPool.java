package xws.neuron;

import com.alibaba.fastjson.JSON;
import xws.neuron.layer.pool.Index;

/**
 * 池化核
 * 不需要width和height
 * Created by xws on 2019/2/13.
 */
public class CNNPool {

    //默认3*3的卷积核
    private int width = 3;
    private int height = 3;

    //步长默认为1
    private int strideX = 1;
    private int strideY = 1;


    public static void main(String[] args) {

        //初始化输入数据
        double[][] test = new double[4][4];

        for (int k = 0; k < 4; k++) {
            test[0][k] = 1;
        }
        for (int k = 0; k < 4; k++) {
            test[1][k] = 2 + k;
        }
        for (int k = 0; k < 4; k++) {
            test[2][k] = 2 + k * 2;
        }
        for (int k = 0; k < 4; k++) {
            test[3][k] = 3 + k * 3;
        }

        System.out.println(JSON.toJSONString(test));
        CNNPool cnnPool = new CNNPool(2, 2, 2, 2);
        System.out.println(JSON.toJSONString(cnnPool));

    }


    //池化操作
    public double[][] pool(double[][] input) {
        //进行数据扩展
        input = UtilNeuralNet.extension(input, height, width, strideX, strideY);

        //获取输入数据的width和height
        int widthInput = input.length;
        int heightInput = input[0].length;
        //卷积和能卷多少次呢：widthInput-weight
        int widthNum = (widthInput - width) / strideX + 1;
        int heightNum = (heightInput - height) / strideY + 1;
        //输出结果初始化
        double[][] pool = new double[heightNum][widthNum];
        for (int h = 0; h < heightNum; h++) {
            for (int w = 0; w < widthNum; w++) {
                double max = maxPool(w * strideX, h * strideY, input);
                pool[h][w] = max;
            }
        }
        return pool;
    }


    //池化运算
    private double maxPool(int widthIndex, int heightIndex, double[][] input) {
        double max = 0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {

                double tmp = input[heightIndex + h][widthIndex + w];
                if (tmp > max) {
                    max = tmp;
                }
            }
        }
        return max;
    }


    //最大值池化
    public Index maxPool(int depth, int heightIndex, int widthIndex, Tensor tensor) {
        Index index = new Index();
        double max = 0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {

                double tmp = tensor.get(depth, heightIndex + h, widthIndex + w);
                if (tmp > max) {
                    max = tmp;
                    index.setDepthFrom(depth);
                    index.setHeightFrom(heightIndex + h);
                    index.setWidthFrom(widthIndex + w);
                    index.setValue(max);
                }
            }
        }
        return index;
    }


    //平均值池化
    public double meanPool(int depth, int heightIndex, int widthIndex, Tensor tensor) {
        double total = 0;
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                double tmp = tensor.get(depth, heightIndex + h, widthIndex + w);
                total = total + tmp;
            }
        }
        return total;
    }

    //平均值池化求导
    public void meanPool_d(int depth, int heightIndex, int widthIndex, Tensor tensor, double error) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                double value = tensor.get(depth, heightIndex + h, widthIndex + w);
                tensor.set(depth, heightIndex + h, widthIndex + w, value + error);
            }
        }
    }

    public CNNPool() {
        init();
    }

    public CNNPool(int height, int width, int strideX, int strideY) {
        this.height = height;
        this.width = width;
        this.strideX = strideX;
        this.strideY = strideY;
        init();
    }

    //初始化工作
    private void init() {

    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public int getStrideX() {
        return strideX;
    }

    public int getStrideY() {
        return strideY;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public void setStrideX(int strideX) {
        this.strideX = strideX;
    }

    public void setStrideY(int strideY) {
        this.strideY = strideY;
    }
}
