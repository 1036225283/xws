package xws.neuron.layer;

import com.alibaba.fastjson.JSON;
import xws.neuron.CNNPool;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;

import java.util.ArrayList;
import java.util.List;


/**
 * 池化层
 * 恢复数据的问题
 * 1.全连接层的一维数据恢复到
 * 2.池化层恢复到卷积层
 * 3.j
 * Created by xws on 2019/2/19.
 */
public class PoolLayer extends Layer {


    private CNNPool pool;//池化核

    private int outDepth;
    private int outHeight;
    private int outWidth;

    private int inputDepth;
    private int inputHeight;
    private int inputWidth;

    private List<Index> list;


    public PoolLayer() {
    }

    //构造函数时，传入filters的构造
    public PoolLayer(int height, int width, int strideX, int strideY) {
        super("pool");
        pool = new CNNPool(height, width, strideX, strideY);
    }

    public PoolLayer(String name, int height, int width, int strideX, int strideY) {
        super("pool");
        pool = new CNNPool(height, width, strideX, strideY);
        setName(name);
    }


    @Override
    public Tensor forward(Tensor tensor) {

        inputDepth = tensor.getDepth();
        inputHeight = tensor.getHeight();
        inputWidth = tensor.getWidth();

        list = new ArrayList<>();


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
                    Index index = pool.maxPool(d, h * pool.getStrideY(), w * pool.getStrideX(), tensor);
                    tensorOut.set(d, h, w, index.getValue());
                    list.add(index);
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


        //根据前向传播时记录的最大值的坐标，将误差数据恢复回去
        double[] error = tensor.getArray();
        for (int i = 0; i < error.length; i++) {
            Index index = list.get(i);
            tensorOut.set(index.getDepth(), index.getHeight(), index.getWidth(), error[i]);
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
