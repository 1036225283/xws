package xws.neuron.layer;

import xws.neuron.Tensor;


/**
 * 填充空白
 * 左填充
 * 右填充
 * 上填充
 * 下填充
 * Created by xws on 2019/2/19.
 */
public class PaddingLayer extends Layer {


    private Tensor tensorInput;
    private Tensor tensorOut;

    private int pTop;
    private int pBottom;
    private int pLeft;
    private int pRight;


    public PaddingLayer() {
    }

    public PaddingLayer(int padding) {
        super(PaddingLayer.class.getSimpleName());
        this.pTop = padding;
        this.pBottom = padding;
        this.pLeft = padding;
        this.pRight = padding;
    }

    public PaddingLayer(String name, int padding) {
        super(PaddingLayer.class.getSimpleName());
        this.pTop = padding;
        this.pBottom = padding;
        this.pLeft = padding;
        this.pRight = padding;
        setName(name);
    }

    //构造函数时，传入filters的构造
    public PaddingLayer(int pTop, int pBottom, int pLeft, int pRight) {
        super(PaddingLayer.class.getSimpleName());
        this.pTop = pTop;
        this.pBottom = pBottom;
        this.pLeft = pLeft;
        this.pRight = pRight;
    }

    public PaddingLayer(String name, int pTop, int pBottom, int pLeft, int pRight) {
        super(PaddingLayer.class.getSimpleName());
        this.pTop = pTop;
        this.pBottom = pBottom;
        this.pLeft = pLeft;
        this.pRight = pRight;
        setName(name);

    }

    @Override
    public Tensor forward(Tensor tensor) {

        tensorInput = tensor;
        //首先，根据现有大小，创建新的大小
        tensorOut = new Tensor();
        tensorOut.setDepth(tensorInput.getDepth());
        tensorOut.setHeight(tensorInput.getHeight() + pTop + pBottom);
        tensorOut.setWidth(tensorInput.getWidth() + pLeft + pRight);
        tensorOut.createArray();


        //开始拷贝数据到新的tenser里面去
        for (int d = 0; d < tensorOut.getDepth(); d++) {
            for (int h = 0; h < tensorOut.getHeight(); h++) {
                for (int w = 0; w < tensorOut.getWidth(); w++) {
                    if (h < pTop || h > tensorInput.getHeight() + pTop - 1) {
                        continue;
                    }
                    if (w < pLeft || w > tensorInput.getWidth() + pLeft - 1) {
                        continue;
                    }

                    double val = tensor.get(d, h - pTop, w - pLeft);
                    tensorOut.set(d, h, w, val);
                }
            }

        }

        return tensorOut;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {

        //首先，创造和输入数据一样大小的虚拟数据
        Tensor tensorError = new Tensor();
        tensorError.setDepth(tensorInput.getDepth());
        tensorError.setHeight(tensorInput.getHeight());
        tensorError.setWidth(tensorInput.getWidth());
        tensorError.createArray();


        //其次，将误差数据拷贝回去
        for (int d = 0; d < tensorOut.getDepth(); d++) {
            for (int h = 0; h < tensorOut.getHeight(); h++) {
                for (int w = 0; w < tensorOut.getWidth(); w++) {

                    if (h < pTop || h > tensorInput.getHeight() + pTop - 1) {
                        continue;
                    }
                    if (w < pLeft || w > tensorInput.getWidth() + pLeft - 1) {
                        continue;
                    }

                    double val = tensor.get(d, h, w);
                    tensorError.set(d, h - pTop, w - pLeft, val);
                }
            }

        }

        return tensorError;
    }

    public int getpTop() {
        return pTop;
    }

    public void setpTop(int pTop) {
        this.pTop = pTop;
    }

    public int getpBottom() {
        return pBottom;
    }

    public void setpBottom(int pBottom) {
        this.pBottom = pBottom;
    }

    public int getpLeft() {
        return pLeft;
    }

    public void setpLeft(int pLeft) {
        this.pLeft = pLeft;
    }

    public int getpRight() {
        return pRight;
    }

    public void setpRight(int pRight) {
        this.pRight = pRight;
    }
}
