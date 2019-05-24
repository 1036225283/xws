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

    //缩放数据
    public void scale(double val) {
        for (int i = 0; i < array.length; i++) {
            array[i] = array[i] / val;
        }
    }

    //展示数据
    public void show() {
        for (int d = 0; d < depth; d++) {
            System.out.println("深度：" + d);
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    System.out.print(this.get(d, h, w) + "\t");
                }
                System.out.println();
            }
        }
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
    public void add(Tensor tensor) {

        if (array.length != tensor.getArray().length) {
            throw new RuntimeException("两个tensor的长度不一致");
        }

        for (int i = 0; i < array.length; i++) {
            array[i] = array[i] + tensor.get(i);
        }
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

    public String toString() {
        StringBuffer sb = new StringBuffer();
        for (int i = 0; i < array.length; i++) {
            sb.append(array[i] + "\t");
        }
        sb.append("\n");
        return sb.toString();
    }

    public Tensor() {
    }

    public Tensor(int size) {
        array = new double[size];
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

    public byte[] toByteArray() {
        byte[] bs = new byte[array.length];
        for (int i = 0; i < bs.length; i++) {
            bs[i] = (byte) array[i];
        }
        return bs;
    }

    public void setArray(double[] array) {
        this.array = array;
    }


}
