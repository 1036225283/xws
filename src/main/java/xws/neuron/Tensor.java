package xws.neuron;


/**
 * 张量,虚拟的一维数组对象
 * Created by xws on 2019/2/21.
 */
public class Tensor {

    private int height = 1;
    private int width = 1;
    private int depth = 1;
    private double[] array;


    public Tensor() {
    }

    public Tensor(int size) {
        array = new double[size];
    }

    public Tensor(double[] array) {
        this.array = array;
        width = array.length;
    }

    public static void main(String[] args) {
        Tensor tensor = new Tensor(10000);
        tensor.setWidth(10);
        tensor.setHeight(10);
        tensor.setDepth(100);
//        for (int i = 0; i < 10000; i++) {
//            tensor.set(i, i);
//        }
//        System.out.println(JSON.toJSONString(tensor));
//
//        for (int i = 0; i < 100; i++) {
//            for (int w = 0; w < 100; w++) {
//                System.out.print(tensor.get(i, w) + " | ");
//            }
//            System.out.println();
//        }

        for (int d = 0; d < 100; d++) {
            for (int h = 0; h < 10; h++) {
                for (int w = 0; w < 10; w++) {
                    tensor.set(d, h, w, w);
                }
            }
        }

//        System.out.println(JSON.toJSONString(tensor));

        for (int d = 0; d < 100; d++) {
            System.out.println("深度：" + d);
            for (int h = 0; h < 10; h++) {
                for (int w = 0; w < 10; w++) {
                    System.out.print(tensor.get(d, h, w) + " | ");
                }
                System.out.println();
            }
        }
    }

    //虚拟一维数组，根据索引获取值
    public double get(int index) {
        return array[index];
    }

    //虚拟二维数组，根据索引获取值
    public double get(int height, int width) {
        if (height >= this.height || width >= this.width) {
            return 0;
        }
        return array[this.width * height + width];
    }

    //虚拟三维数组，根据索引获取值
    public double get(int depth, int height, int width) {
        if (height >= this.height || width >= this.width) {
            return 0;
        }
        return array[depth * this.height * this.width + height * this.width + width];
    }

    //虚拟一维数组，根据索引设置值
    public void set(int index, double value) {
        array[index] = value;
    }

    //虚拟二维数组，根据索引设置值
    public void set(int height, int width, double value) {
        array[height * this.width + width] = value;
    }

    //虚拟三维数组，根据索引设置值
    public void set(int depth, int height, int width, double value) {
        array[depth * this.height * this.width + height * this.width + width] = value;
    }

    //获取实际索引
    public int index(int height, int width) {
        return height * this.width + width;
    }

    public int index(int depth, int height, int width) {
        return depth * this.height * this.width + height * this.width + width;
    }

    //创建一维数组
    public void createArray() {
        if (depth != 0 && height != 0 && width != 0) {
            array = new double[depth * height * width];
        } else if (height != 0 && width != 0) {
            array = new double[height * width];
        } else if (width != 0) {
            array = new double[width];
        } else {
            throw new RuntimeException("传教一维数组异常");
        }
    }

    //创建当前对象的一份拷贝
    public Tensor copy() {
        Tensor copy = new Tensor();
        copy.setDepth(depth);
        copy.setHeight(height);
        copy.setWidth(width);
        copy.createArray();
        double[] copyArr = copy.getArray();
        for (int i = 0; i < copyArr.length; i++) {
            copyArr[i] = array[i];
        }
        return copy;
    }

    //0填充
    public void zero() {
        for (int i = 0; i < array.length; i++) {
            array[i] = 0;
        }
    }

    //累加值
    public double val() {
        double total = 0;
        for (int i = 0; i < array.length; i++) {
            total = total + array[i];
        }
        return total;
    }

    //缩放数据
    public void scale(double val) {
        for (int i = 0; i < array.length; i++) {
            array[i] = array[i] / val;
        }
    }

    //展示数据
    public void show(String msg) {
        if (msg != null) {
            System.out.print(msg);
        }
        for (int d = 0; d < depth; d++) {
            System.out.print(" 深度" + d + " ");
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    System.out.print(this.get(d, h, w) + "\t");
                }
                System.out.println();
            }
        }
    }

    public void show() {
        show(null);
    }

    //一维点乘
    public double dot(Tensor tensor) {
        double total = 0;
        double[] array = this.getArray();
        double[] dest = tensor.getArray();
        for (int i = 0; i < array.length; i++) {
            total = total + array[i] * dest[i];
        }
        return total;
    }

    //向量加法
    public Tensor add(Tensor tensor) {

        Tensor out = tensor.copy();

        if (array.length != tensor.size()) {
            throw new RuntimeException("两个tensor的长度不一致");
        }

        for (int i = 0; i < array.length; i++) {
            out.set(i, out.get(i) + array[i]);
        }

        return out;
    }

    //tensorInput * tensorW
    public Tensor multiplyW(Tensor tensorW) {
        Tensor tensorInputMultiplyWeight = new Tensor();
        tensorInputMultiplyWeight.setWidth(tensorW.getHeight());
        tensorInputMultiplyWeight.createArray();
        for (int h = 0; h < tensorW.getHeight(); h++) {
            for (int w = 0; w < tensorW.getWidth(); w++) {
                tensorInputMultiplyWeight.set(h, tensorInputMultiplyWeight.get(h) + this.get(w) * tensorW.get(h, w));
            }
        }
        return tensorInputMultiplyWeight;
    }

    // please use the tensorBias call the method, update back propagation bias error
    public void updateBias(Tensor tensorPartialDerivative, double learnRate) {
        for (int i = 0; i < getWidth(); i++) {
            set(i, get(i) - learnRate * tensorPartialDerivative.get(i));
        }
    }

    //please use the tensorW call the method, update back propagation weight error
    public void updateWeight(Tensor tensorPartialDerivative, double learnRate) {
        for (int i = 0; i < size(); i++) {
            set(i, get(i) - learnRate * tensorPartialDerivative.get(i));
        }
    }


    // please use the tensorW call the method
    public Tensor calculateWeightPartialDerivative(Tensor tensorPartialDerivative, Tensor tensorInput) {
        Tensor tensorWeightPartialDerivative = this.copy();
        tensorWeightPartialDerivative.zero();
        for (int h = 0; h < this.getHeight(); h++) {
            for (int w = 0; w < this.getWidth(); w++) {
                tensorWeightPartialDerivative.set(h, w, tensorWeightPartialDerivative.get(h, w) + tensorPartialDerivative.get(h) * tensorInput.get(w));
            }
        }
        return tensorWeightPartialDerivative;
    }

    // please use the tensorW call the method
    public Tensor calculateInputPartialDerivative(Tensor tensorPartialDerivative) {
        Tensor tensorInputPartialDerivative = new Tensor();
        tensorInputPartialDerivative.setWidth(this.getWidth());
        tensorInputPartialDerivative.createArray();
        for (int h = 0; h < this.getHeight(); h++) {
            for (int w = 0; w < this.getWidth(); w++) {
                tensorInputPartialDerivative.set(w, tensorInputPartialDerivative.get(w) + tensorPartialDerivative.get(h) * this.get(h, w));
            }
        }
        return tensorInputPartialDerivative;
    }

    // please use the tensorOut call the method
    public Tensor calculateOutPartialDerivativeByMse(Tensor expect) {
        Tensor tensorPartialDerivative = expect.copy();
        tensorPartialDerivative.zero();
        for (int i = 0; i < expect.size(); i++) {
            tensorPartialDerivative.set(i, this.get(i) - expect.get(i));
        }
        return tensorPartialDerivative;
    }

    public double sum() {
        double total = 0;
        for (int i = 0; i < array.length; i++) {
            total = total + array[i];
        }
        return total;
    }


    public int size() {
        return array.length;
    }

    //根据高度获取数据
    public double[] data(int height) {
        double[] val = new double[this.width];
        for (int i = 0; i < this.width; i++) {
            val[i] = get(height, i);
        }
        return val;
    }

    //根据深度和高度获取数据
    public double[] data(int depth, int height) {
        double[] val = new double[this.width];
        for (int i = 0; i < this.width; i++) {
            val[i] = get(depth, height, i);
        }
        return val;
    }

    //从byte[]拷贝数据
    public void copy(byte[] bs) {
        if (bs.length != array.length) {
            throw new RuntimeException("数据长度不匹配");
        }

        for (int i = 0; i < array.length; i++) {
            array[i] = bs[i];
        }
    }

    //裁剪数据
    public Tensor subtensor(int startDepth, int endDepth, int startHeight, int endHeight, int startWidth, int endWidth) {

        if (startDepth < 0 || startHeight < 0 || startWidth < 0) {
            throw new RuntimeException("start must >=0 ");
        }

        if (endDepth >= depth || endHeight >= height || endWidth >= width) {
            throw new RuntimeException("end must < length");
        }

        if (startDepth >= (depth - endDepth) || startHeight >= (height - endHeight) || startWidth >= (width - endWidth)) {
            throw new RuntimeException("length - end must > start");
        }

        Tensor tensorNew = new Tensor();
        tensorNew.setDepth(depth - endDepth - startDepth);
        tensorNew.setHeight(height - endHeight - startHeight);
        tensorNew.setWidth(width - endWidth - startWidth);
        tensorNew.createArray();
        for (int d = 0; d < tensorNew.getDepth(); d++) {
            for (int h = 0; h < tensorNew.getHeight(); h++) {
                for (int w = 0; w < tensorNew.getWidth(); w++) {
                    double val = get(d + startDepth, h + startHeight, w + startWidth);
                    tensorNew.set(d, h, w, val);
                }
            }
        }
        return tensorNew;
    }

    //裁剪宽度
    public Tensor subWidth(int start, int end) {
        return subtensor(0, 0, 0, 0, start, end);
    }

    //裁剪高度
    public Tensor subHeight(int start, int end) {
        return subtensor(0, 0, start, end, 0, 0);
    }

    //裁剪深度
    public Tensor subDepth(int start, int end) {
        return subtensor(start, end, 0, 0, 0, 0);
    }

    public String toString() {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < array.length; i++) {
            sb.append(array[i] + "\t");
        }
        sb.append("\n");
        return sb.toString();
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

    public int getDepth() {
        return depth;
    }

    public void setDepth(int depth) {
        this.depth = depth;
    }

    public double[] getArray() {
        return array;
    }

    public void setArray(double[] array) {
        this.array = array;
    }

    public byte[] toByteArray() {
        byte[] bs = new byte[array.length];
        for (int i = 0; i < bs.length; i++) {
            bs[i] = (byte) array[i];
        }
        return bs;
    }


}
