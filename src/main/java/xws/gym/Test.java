package xws.gym;

import com.alibaba.fastjson.JSON;
import xws.neuron.UtilNeuralNet;

import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * test
 * Created by xws on 2019/10/10.
 */
public class Test {
    static double gamma = 0.95;

    public static void main(String[] args) {
//        double ep_rs[] = new double[]{1.0, 2.0, 3.0, 4, 5, 6, 7, 8, 9, 10};
        double ep_rs[] = new double[]{1.0, 2.0, 3.0, 4, 5, 6, 1, 1, 1, 100};
        rewards(ep_rs);
        System.out.println(Math.sqrt(16));
        Test2();

    }

    public static void Test2() {
        Date time = new Date("Sun May 05 2019 10:45:35 GMT+0800");
        SimpleDateFormat sdf = new SimpleDateFormat("yyyy-MM-dd");
        String timeFormat = sdf.format(time);
        System.out.println(timeFormat);
    }

    public static double rewards(double[] array) {

        double[] rewards = new double[array.length];
        double reward = 0;
        for (int i = array.length - 1; i >= 0; i--) {
            rewards[i] = reward * gamma + array[i];
            reward = rewards[i];
        }
        System.out.println("gamma = " + JSON.toJSONString(rewards));

        double average = UtilNeuralNet.average(rewards);

        for (int i = 0; i < rewards.length; i++) {
            rewards[i] = rewards[i] - average;
        }
        System.out.println("average = " + JSON.toJSONString(rewards));

        double variance = UtilNeuralNet.variance(rewards);

        double standardDeviation = Math.sqrt(variance);
        for (int i = 0; i < rewards.length; i++) {
            rewards[i] = rewards[i] / standardDeviation;
        }
        System.out.println("variance = " + JSON.toJSONString(rewards));

        return 0;
    }
}
