package xws.neuron.layer.pool;

import xws.neuron.CNNPool;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.Layer;


/**
 * 反池化层-平均池化
 * Created by xws on 2019/6/8.
 */
public class MeanPoolBackLayer extends Layer {


    private CNNPool pool;//池化核

    private int outDepth;
    private int outHeight;
    private int outWidth;

    private int inputDepth;
    private int inputHeight;
    private int inputWidth;


    public MeanPoolBackLayer() {
        super(MeanPoolBackLayer.class.getSimpleName());
    }

    //构造函数时，传入filters的构造
    public MeanPoolBackLayer(int height, int width, int strideX, int strideY) {
        super(MeanPoolBackLayer.class.getSimpleName());
        pool = new CNNPool(height, width, strideX, strideY);
    }

    public MeanPoolBackLayer(String name, int height, int width, int strideX, int strideY) {
        super(MeanPoolBackLayer.class.getSimpleName());
        pool = new CNNPool(height, width, strideX, strideY);
        setName(name);
    }


    @Override
    public Tensor forward(Tensor tensor) {

        inputDepth = tensor.getDepth();
        inputHeight = tensor.getHeight();
        inputWidth = tensor.getWidth();


        //首先，计算反池化后的大小
        outHeight = (tensor.getHeight() - 1) * pool.getStrideY() + pool.getHeight();
        outWidth = (tensor.getWidth() - 1) * pool.getStrideX() + pool.getWidth();
        outDepth = tensor.getDepth();
        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(outDepth);
        tensorOut.setHeight(outHeight);
        tensorOut.setWidth(outWidth);
        tensorOut.createArray();

        //填充数据
        for (int d = 0; d < tensor.getDepth(); d++) {
            for (int h = 0; h < outHeight; h++) {
                for (int w = 0; w < outWidth; w++) {
                    //取出池化值
                    double value = tensor.get(d, h, w);
                    //逆向操作
                    pool.meanPool_d(d, h * pool.getStrideY(), w * pool.getStrideX(), tensorOut, value);
                }
            }

        }

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
                    //累加误差
                    double error = pool.meanPool(d, h * pool.getStrideY(), w * pool.getStrideX(), tensor);
                    tensorOut.set(outHeight, outWidth, error);
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
