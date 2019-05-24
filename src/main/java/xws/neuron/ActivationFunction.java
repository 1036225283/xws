package xws.neuron;

/**
 * 激活函数
 * Created by xws on 2019/2/18.
 */
public class ActivationFunction {


    public static double sigmoid(double z) {
        return 1 / (1 + Math.pow(Math.E, -z));
    }

    public static double sigmoid_d(double z) {
        double a = sigmoid(z);
        return a * (1 - a);
    }

    public static double tanh(double z) {
        double a = Math.pow(Math.E, z);
        double b = Math.pow(Math.E, -z);
        return (a - b) / (a + b);
    }

    public static double tanh_d(double z) {
        double a = tanh(z);
        return 1 - a * a;
    }

    public static double relu(double z) {
        if (z > 0) {
            return z;
        } else {
            return 0;
        }
    }

    public static double relu_d(double z) {
        if (z > 0) {
            return 1;
        } else {
            return 0;
        }
    }


    public static double activation(double val, String type) {
        if ("sigmoid".equals(type)) {
            return sigmoid(val);
        } else if ("tanh".equals(type)) {
            return tanh(val);
        } else if ("relu".equals(type)) {
            return relu(val);
        }
        throw new RuntimeException("没有找到激活函数");
    }

    public static double derivation(double val, String type) {
        if ("sigmoid".equals(type)) {
            return sigmoid_d(val);
        } else if ("tanh".equals(type)) {
            return tanh_d(val);
        } else if ("relu".equals(type)) {
            return relu_d(val);
        }
        throw new RuntimeException("没有找到激活函数");
    }

    public static void main(String[] args) {
        System.out.println(relu(1.12342342342d));
        System.out.println(relu(0f));
        System.out.println(relu(-1.2d));

        System.out.println(relu_d(1.1d));
        System.out.println(relu_d(0f));
        System.out.println(relu_d(-1.2d));
    }
}
