package xws.util;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import xws.neuron.MnistRead;
import xws.neuron.Tensor;
import xws.test.FullNetWorkTest;

import java.util.ArrayList;
import java.util.List;

import static xws.test.FullNetWorkTest.oneHot;

/**
 * Created by xws on 2019/3/22.
 */
public class UtilMnist {

    public static List<Cifar10> learnData() {
        double[][] images = MnistRead.getImages(MnistRead.TRAIN_IMAGES_FILE);
        double[] labels = MnistRead.getLabels(MnistRead.TRAIN_LABELS_FILE);

        List<Cifar10> list = new ArrayList<>();
        for (int i = 0; i < images.length; i++) {
            Cifar10 cifar10 = new Cifar10();
            cifar10.setLabel(FullNetWorkTest.oneHot((int) labels[i]));
            cifar10.setRgb(createData(images[i]));
            cifar10.setIndex((int) labels[i]);
            list.add(cifar10);
        }
        return list;
    }
    //获取测试数据

    public static List<Cifar10> testData() {

        double[][] imagesTest = MnistRead.getImages(MnistRead.TEST_IMAGES_FILE);
        double[] labelsTest = MnistRead.getLabels(MnistRead.TEST_LABELS_FILE);

        List<Cifar10> list = new ArrayList<>();
        for (int i = 0; i < imagesTest.length; i++) {
            Cifar10 cifar10 = new Cifar10();
            cifar10.setLabel(FullNetWorkTest.oneHot((int) labelsTest[i]));
            cifar10.setRgb(createData(imagesTest[i]));
            cifar10.setIndex((int) labelsTest[i]);
            list.add(cifar10);
        }
        return list;
    }


    //size=10,width=28,height=28
    public static List<Cifar10> data10_28() {
        return dataFromJsonFile("/Users/xws/Desktop/xws/MNIST.txt", 28);

    }

    //size=10,width=50,height=50
    public static List<Cifar10> data10_100() {
        return dataFromJsonFile("/Users/xws/Desktop/xws/MNIST1.txt", 100);
    }

    public static List<Cifar10> dataFromJsonFile(String fileName, int size) {
        List<Cifar10> list = new ArrayList<>();
        String json = UtilFile.readFile(fileName);
        JSONObject jsonObject = JSON.parseObject(json);
        for (int i = 0; i < 10; i++) {
            jsonObject.getString(i + ".0");
            JSONArray array = jsonObject.getJSONArray(i + ".0");
            double[] arr = new double[array.size()];
            for (int k = 0; k < arr.length; k++) {
                arr[k] = array.getDoubleValue(k);
            }

            Cifar10 cifar10 = new Cifar10();
            cifar10.setLabel(FullNetWorkTest.oneHot((int) i));
            Tensor tensor = new Tensor();
            tensor.setDepth(1);
            tensor.setHeight(size);
            tensor.setWidth(size);
            tensor.setArray(arr);
            cifar10.setRgb(tensor);
            list.add(cifar10);
        }
        return list;
    }

    private static Tensor createData(double[] image) {
        Tensor tensor = new Tensor();
        tensor.setDepth(1);
        tensor.setHeight(28);
        tensor.setWidth(28);
        tensor.setArray(image);
        return tensor;
    }


}
