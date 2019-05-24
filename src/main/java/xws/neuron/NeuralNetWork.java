package xws.neuron;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONObject;
import xws.util.UtilFile;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * 抽象的神经网络，提供一些公共部分，比如持久化
 * Created by xws on 2019/2/22.
 */
public class NeuralNetWork {

    private static String path = "/Users/xws/Desktop/xws";
    private String type;//神经网络的类型
    private int version = 0;//神经网络持久化的版本信息
    private String name;
    private int batch = 0;//批次
    private int batchSize = 1;//批次大小


    public void save(String name) {
        this.name = name;
        String fileName = path + File.separator + name + File.separator + name;
        String fileVersion = path + File.separator + name + File.separator + name + "." + version;
        String json = JSON.toJSONString(this);
        UtilFile utilFile = new UtilFile(fileName);
        UtilFile.writeFile(json, fileName);
        UtilFile.writeFile(json, fileVersion);
    }

    //将神经网络json字符串从硬盘加载到内存中，形成JSONObject对象
    public static JSONObject loadJson(String name) {
        String strPath = name;
        if (name.contains(".")) {
            int index = name.indexOf(".");
            strPath = name.substring(0, index);
        }

        String json = UtilFile.readFile(path + File.separator + strPath + File.separator + name);
        JSONObject jsonObject = JSON.parseObject(json);
        return jsonObject;
    }

    //加载神经网络
    public static NeuralNetWork load(String name) {
        return null;
    }


    //把当前下面所有的神经网络持久化文件都列出来
    public static List<String> loadAll(String name) {
        String strPath = path + File.separator + name;
        String[] files = UtilFile.getFileNames(strPath);
        List<String> list = new ArrayList<>();
        for (String str : files) {
            if (str.contains(name)) {
                list.add(str);
            }
        }
        return list;
    }

    //总误差
    public double totalError() {
        return 0;
    }

    public static String getPath() {
        return path;
    }

    public static void setPath(String path) {
        NeuralNetWork.path = path;
    }

    public int getVersion() {
        return version;
    }

    public void setVersion(int version) {
        this.version = version;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getBatch() {
        return batch;
    }

    public void setBatch(int batch) {
        this.batch = batch;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
}
