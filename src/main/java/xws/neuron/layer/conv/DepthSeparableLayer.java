package xws.neuron.layer.conv;

import xws.neuron.ActivationFunction;
import xws.neuron.CNNFilter;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.Layer;
import xws.util.UtilFile;


/**
 * 深度可分离卷积层
 * Created by xws on 2019/2/19.
 */
public class DepthSeparableLayer extends Layer {


    private CNNFilter[] filters;//卷积核
    private int height;
    private int width;
    private int strideX;
    private int strideY;

    //输出输入数据大小
    private int outDepth;
    private int outHeight;
    private int outWidth;

    //正则化
    private double lambda = 0;


    //输入数据存储
    private Tensor tensorInput;

    //z值数据存储
    private Tensor tensorZ;

    public DepthSeparableLayer() {
        super(DepthSeparableLayer.class.getSimpleName());
    }

    //构造函数时，传入filters的构造,total是特征
    public DepthSeparableLayer(int height, int width, int strideX, int strideY) {
        super(DepthSeparableLayer.class.getSimpleName());
        this.height = height;
        this.width = width;
        this.strideX = strideX;
        this.strideY = strideY;
    }


    public DepthSeparableLayer(String name, String activationType, int height, int width, int strideX, int strideY) {
        super(DepthSeparableLayer.class.getSimpleName());
        setName(name);
        setActivationType(activationType);
        this.height = height;
        this.width = width;
        this.strideX = strideX;
        this.strideY = strideY;
    }

    public DepthSeparableLayer(String name, String activationType, int height, int width, int strideX, int strideY, double lambda) {
        super(DepthSeparableLayer.class.getSimpleName());
        setName(name);
        setActivationType(activationType);
        this.height = height;
        this.width = width;
        this.strideX = strideX;
        this.strideY = strideY;
        this.lambda = lambda;

    }


    @Override
    public Tensor forward(Tensor tensor) {


        tensorInput = tensor;

        //需要计算出结果数据的维度和大小
        outDepth = tensorInput.getDepth();
        outHeight = UtilNeuralNet.afterHeight(tensor.getHeight(), 0, strideY, height);
        outWidth = UtilNeuralNet.afterWidth(tensor.getWidth(), 0, strideX, width);


        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(outDepth);
        tensorOut.setHeight(outHeight);
        tensorOut.setWidth(outWidth);
        tensorOut.createArray();

        //如果卷积核还没有进行初始化，先进行初始化
        if (filters == null) {
            filters = new CNNFilter[tensorInput.getDepth()];
            for (int i = 0; i < filters.length; i++) {
                CNNFilter filter = new CNNFilter(1, height, width, strideX, strideY, lambda);
                filters[i] = filter;
            }
        }


        /**
         * 32*32*3
         * 5*5*3*6
         * 28*28*6
         */
        //进行卷积运算,按核的数量来算
        for (int d = 0; d < filters.length; d++) {
            CNNFilter cnnFilter = filters[d];
            //有多少输出，意味着就要执行多少次卷积操作
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    double val = cnnFilter.convolution(h * strideY, w * strideX, tensorInput);
                    tensorOut.set(d, h, w, val);
                }
            }
        }

        activation(tensorOut);
        return tensorOut;
    }

    public void activation(Tensor tensor) {
        tensorZ = tensor.copy();
        for (int d = 0; d < tensor.getDepth(); d++) {
            double b = filters[d].getBias();
            for (int h = 0; h < tensor.getHeight(); h++) {
                for (int w = 0; w < tensor.getWidth(); w++) {
                    double z = tensor.get(d, h, w) + b;
                    tensorZ.set(d, h, w, z);
                    double a = ActivationFunction.activation(z, getActivationType());
                    tensor.set(d, h, w, a);
                }
            }
        }
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {

        //创建张量，存储误差数据
        Tensor tensorError = new Tensor();
        tensorError.setDepth(tensorInput.getDepth());
        tensorError.setHeight(tensorInput.getHeight());
        tensorError.setWidth(tensorInput.getWidth());
        tensorError.createArray();

        pdz(tensor);

        /**
         * 32*32*3
         */
        //计算上一层输入的误差
        for (int d = 0; d < filters.length; d++) {
            CNNFilter filter = filters[d];
            //有多少输出，意味着就要执行多少次卷积操作
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    //获取误差
                    double error = tensor.get(d, h, w);
                    filter.convolution_d(h * strideY, w * strideX, tensorError, error);
                }
            }
        }

        if (isTest()) {
            return tensorError;
        }

        //计算w的误差
        for (int f = 0; f < filters.length; f++) {
            CNNFilter filter = filters[f];
            filter.initTensorW();
            //有多少输出，意味着就要执行多少次卷积操作
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    //获取误差
                    double error = tensor.get(f, h, w);
                    filter.pdw(h * strideY, w * strideX, tensorInput, error);
                }
            }

        }

        //更新核误差
        for (int i = 0; i < filters.length; i++) {
            CNNFilter filter = filters[i];
            filter.updateErrorW(getLearnRate());
            filter.updateErrorB(tensor, i, getLearnRate());
        }

        return tensorError;
    }

    //求∂A/∂Z
    public void pdz(Tensor tensorError) {
        for (int d = 0; d < tensorError.getDepth(); d++) {
            for (int h = 0; h < tensorError.getHeight(); h++) {
                for (int w = 0; w < tensorError.getWidth(); w++) {
                    double error = tensorError.get(d, h, w);
                    double z = tensorZ.get(d, h, w);
                    error = ActivationFunction.derivation(z, getActivationType()) * error;
                    tensorError.set(d, h, w, error);
                }
            }
        }
    }

    public CNNFilter[] getFilters() {
        return filters;
    }

    public void setFilters(CNNFilter[] filters) {
        this.filters = filters;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getStrideX() {
        return strideX;
    }

    public void setStrideX(int strideX) {
        this.strideX = strideX;
    }

    public int getStrideY() {
        return strideY;
    }

    public void setStrideY(int strideY) {
        this.strideY = strideY;
    }

    public double getLambda() {
        return lambda;
    }

    public void setLambda(double lambda) {
        this.lambda = lambda;
    }

}
