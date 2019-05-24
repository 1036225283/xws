package xws.neuron;

import com.alibaba.fastjson.JSON;
import xws.util.UtilFile;

import java.util.List;

/**
 * 卷积核
 * 需要width和height
 * Created by xws on 2019/2/13.
 */
public class CNNFilter {

    //默认3*3的卷积核
    private int depth = 1;
    private int width = 3;
    private int height = 3;

    //步长默认为1
    private int strideX = 1;
    private int strideY = 1;

    //偏置
    private double bias = 0.0010000000474974513;

    //正则化
    private double lambda = 0;


    //核数据
    private Tensor filter;

    //核偏导
    private Tensor tensorW;

    //卷积结果

    //权重文件输出
    private UtilFile[] files;


    public double convolution(int heightIndex, int widthIndex, Tensor tensor) {
        double total = 0;
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    total = total + filter.get(d, h, w) * tensor.get(d, heightIndex + h, widthIndex + w);
                }
            }
        }
        return total;
    }

    //卷积求导
    public void convolution_d(int heightIndex, int widthIndex, Tensor tensorError, double error) {
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    double total = tensorError.get(d, heightIndex + h, widthIndex + w);
                    total = total + filter.get(d, h, w) * error;
                    tensorError.set(d, heightIndex + h, widthIndex + w, total);
                }
            }
        }

    }

    public void initTensorW() {
        if (tensorW == null) {
            tensorW = new Tensor();
            tensorW.setDepth(filter.getDepth());
            tensorW.setHeight(filter.getHeight());
            tensorW.setWidth(filter.getWidth());
            tensorW.createArray();
        } else {
            tensorW.zero();
        }
    }

    public void pdw(int heightIndex, int widthIndex, Tensor tensorInput, double error) {
        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    double input = tensorInput.get(d, heightIndex + h, widthIndex + w);
                    double total = tensorW.get(d, h, w);
                    total = total + input * error;
                    tensorW.set(d, h, w, total);
                }
            }
        }

    }

    //更新w的误差
    public void updateErrorW(double learnRate) {

        for (int d = 0; d < depth; d++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    double val = filter.get(d, h, w) - learnRate * tensorW.get(d, h, w) - lambda * filter.get(d, h, w);
                    filter.set(d, h, w, val);
                }
            }
        }

    }

    //更新b的误差
    public void updateErrorB(Tensor tensorError, int d, double learnRate) {
        //先对误差求和
        double error = 0;
        for (int h = 0; h < tensorError.getHeight(); h++) {
            for (int w = 0; w < tensorError.getWidth(); w++) {
                error = error + tensorError.get(d, h, w);
            }
        }
        bias = bias - learnRate * error;
    }


    public CNNFilter() {
    }

    public CNNFilter(int height, int width, int strideX, int strideY) {
        this.height = height;
        this.width = width;
        this.strideX = strideX;
        this.strideY = strideY;
        init();
    }

    public CNNFilter(int depth, int height, int width, int strideX, int strideY) {
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.strideX = strideX;
        this.strideY = strideY;
        init();
    }

    public CNNFilter(int depth, int height, int width, int strideX, int strideY, double lambda) {
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.strideX = strideX;
        this.strideY = strideY;
        this.lambda = lambda;
        init();
    }

    //初始化工作
    private void init() {
        filter = new Tensor();
        filter.setDepth(depth);
        filter.setHeight(height);
        filter.setWidth(width);
        filter.createArray();
        UtilNeuralNet.initWeight(filter.getArray());
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


    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public Tensor getFilter() {
        return filter;
    }

    public void setFilter(Tensor filter) {
        this.filter = filter;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

    private void initFile() {

    }
}
