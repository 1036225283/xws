package xws.neuron.layer.pool;

import xws.neuron.CNNPool;
import xws.neuron.Tensor;
import xws.neuron.layer.Layer;

import java.util.List;


/**
 * 反池化层-最大池化
 * Created by xws on 2019/6/6.
 */
public class MaxPoolBackLayer extends Layer {


    private CNNPool pool;//池化核

    private List<Index> list;
    private MaxPoolLayer maxPoolLayer;

    private Tensor tensorInput;

    private String targetName;//反池化的层名称


    public MaxPoolBackLayer() {
    }

    //构造函数时，传入filters的构造
    public MaxPoolBackLayer(int height, int width, int strideX, int strideY) {
        super(MaxPoolBackLayer.class.getName());
        pool = new CNNPool(height, width, strideX, strideY);
    }

    public MaxPoolBackLayer(String name, int height, int width, int strideX, int strideY) {
        super(MaxPoolBackLayer.class.getName());
        pool = new CNNPool(height, width, strideX, strideY);
        setName(name);
    }


    @Override
    public Tensor forward(Tensor tensor) {

        //首先，拿到坐标信息
        list = maxPoolLayer.info();
        tensorInput = tensor;

        //首先计算反池化后的大小,也就是池化之前的大小
//        if (originalHeight != null && originalWidth != null) {
//            outHeight = originalHeight;
//            outWidth = originalWidth;
//        } else {
//            outHeight = (tensor.getHeight() - 1) * pool.getStrideY() + pool.getHeight();
//            outWidth = (tensor.getWidth() - 1) * pool.getStrideX() + pool.getWidth();
//        }


        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(tensorInput.getDepth());
        tensorOut.setHeight(maxPoolLayer.tensorInput().getHeight());
        tensorOut.setWidth(maxPoolLayer.tensorInput().getWidth());
        tensorOut.createArray();

        for (int i = 0; i < list.size(); i++) {
            Index index = list.get(i);
            tensorOut.set(index.getDepthFrom(), index.getHeightFrom(), index.getWidthFrom(), index.getValue());
        }


        return tensorOut;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {
        //先创造和输入数据一样大小的虚拟数据
        Tensor tensorOut = new Tensor();
        tensorOut.setDepth(tensorInput.getDepth());
        tensorOut.setHeight(tensorInput.getHeight());
        tensorOut.setWidth(tensorInput.getWidth());
        tensorOut.createArray();


        //根据前向传播时记录的最大值的坐标，将误差数据恢复回去
        double[] error = tensor.getArray();
        for (int i = 0; i < error.length; i++) {
            Index index = list.get(i);
            tensorOut.set(index.getDepthFrom(), index.getHeightFrom(), index.getWidthFrom(), error[i]);
        }
//
//        for (int d = 0; d < tensor.getDepth(); d++) {
//            for (int h = 0; h < outHeight; h++) {
//                for (int w = 0; w < outWidth; w++) {
//                    Index index = pool.maxPool(d, h * pool.getStrideY(), w * pool.getStrideX(), tensor);
//                    tensorOut.set(d, h, w, index.getValue());
//                    list.add(index);
//                }
//            }
//
//        }


        return tensorOut;
    }

    public CNNPool getPool() {
        return pool;
    }

    public void setPool(CNNPool pool) {
        this.pool = pool;
    }

    public void setMaxPoolLayer(MaxPoolLayer maxPoolLayer) {
        this.maxPoolLayer = maxPoolLayer;
    }

    public String getTargetName() {
        return targetName;
    }

    public void setTargetName(String targetName) {
        this.targetName = targetName;
    }
}
