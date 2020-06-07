package xws.neuron.layer.pool;

import xws.neuron.CNNPool;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.Layer;


/**
 * 平均池化层
 * 恢复数据的问题
 * 1.全连接层的一维数据恢复到
 * 2.池化层恢复到卷积层
 * Created by xws on 2019/6/6.
 */
public class MeanPoolLayer extends Layer {


    private CNNPool pool;//池化核

    private int outDepth;
    private int outHeight;
    private int outWidth;

    private int inputDepth;
    private int inputHeight;
    private int inputWidth;


    public MeanPoolLayer() {
    }

    //构造函数时，传入filters的构造
    public MeanPoolLayer(int height, int width, int strideX, int strideY) {
        super(MeanPoolLayer.class.getName());
        pool = new CNNPool(height, width, strideX, strideY);
    }

    public MeanPoolLayer(String name, int height, int width, int strideX, int strideY) {
        super(MeanPoolLayer.class.getName());
        pool = new CNNPool(height, width, strideX, strideY);
        setName(name);
    }


    @Override
    public Tensor forward(Tensor tensor) {

        inputDepth = tensor.getDepth();
        inputHeight = tensor.getHeight();
        inputWidth = tensor.getWidth();


        //需要计算出结果数据的维度和大小
        outHeight = UtilNeuralNet.afterHeight(tensor.getHeight(), 0, pool.getStrideY(), pool.getHeight());
        outWidth = UtilNeuralNet.afterWidth(tensor.getWidth(), 0, pool.getStrideX(), pool.getWidth());
        outDepth = tensor.getDepth();
        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(outDepth);
        tensorOut.setHeight(outHeight);
        tensorOut.setWidth(outWidth);
        tensorOut.createArray();

        for (int d = 0; d < tensor.getDepth(); d++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    double value = pool.meanPool(d, h * pool.getStrideY(), w * pool.getStrideX(), tensor);
                    tensorOut.set(d, h, w, value / pool.getHeight() / pool.getWidth());
                }
            }

        }

//        System.out.println(JSON.toJSONString(list));

        return tensorOut;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {
        //先创造和输入数据一样大小的虚拟数据
        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(inputDepth);
        tensorOut.setHeight(inputHeight);
        tensorOut.setWidth(inputWidth);
        tensorOut.createArray();


        //计算数量
        double total = pool.getHeight() * pool.getWidth();
        //根据前向传播进行逆运算，将误差数据平摊回去
        for (int d = 0; d < tensor.getDepth(); d++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    //首先，拿到该位置的误差
                    double error = tensor.get(d, h, w);
                    //平摊误差
                    double errorMean = error / total;
                    //累加误差
                    pool.meanPool_d(d, h * pool.getStrideY(), w * pool.getStrideX(), tensorOut, errorMean);
                }
            }
        }
        return tensorOut;
    }

    public CNNPool getPool() {
        return pool;
    }

    public void setPool(CNNPool pool) {
        this.pool = pool;
    }
}
