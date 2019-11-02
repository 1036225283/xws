package xws.neuron;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import xws.neuron.layer.*;
import xws.neuron.layer.activation.ReLuLayer;
import xws.neuron.layer.activation.SigmoidLayer;
import xws.neuron.layer.activation.TanhLayer;
import xws.neuron.layer.bn.BnLayer;
import xws.neuron.layer.bn.LnLayer;
import xws.neuron.layer.bn.MnLayer;
import xws.neuron.layer.conv.ConvolutionLayer;
import xws.neuron.layer.conv.DepthSeparableLayer;
import xws.neuron.layer.output.CrossEntropyLayer;
import xws.neuron.layer.output.MseLayer;
import xws.neuron.layer.output.SoftMaxLayer;
import xws.neuron.layer.pool.MaxPoolBackLayer;
import xws.neuron.layer.pool.MaxPoolLayer;
import xws.neuron.layer.pool.MeanPoolLayer;
import xws.neuron.layer.rnn.RnnLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * 卷积神经网络
 * 前向传播搞定了
 * Created by xws on 2019/1/5.
 */
public class CNNetWork extends NeuralNetWork {

    private List<Layer> layers = new ArrayList<>();

    //学习率
    private double learnRate = UtilNeuralNet.e();

    //step rnn控制输入是否开始
    private int step = 0;

    //加载卷积神经网络从硬盘上
    public static CNNetWork load(String name) {
        JSONObject jsonObject = loadJson(name);


        CNNetWork cnNetWork = new CNNetWork();
        int version = jsonObject.getIntValue("version");
        cnNetWork.setVersion(version + 1);


        //从json中提取出各个层
        JSONArray layerArr = jsonObject.getJSONArray("layers");
        for (int i = 0; i < layerArr.size(); i++) {
//            System.out.println(layerArr.get(i));
            JSONObject layer = layerArr.getJSONObject(i);
            String strType = layer.getString("type");
            if (ConvolutionLayer.class.getSimpleName().equals(strType)) {
                ConvolutionLayer convolutionLayer = JSONObject.parseObject(layer.toString(), ConvolutionLayer.class);
                cnNetWork.addLayer(convolutionLayer);
            } else if (MaxPoolLayer.class.getSimpleName().equals(strType)) {
                MaxPoolLayer maxPoolLayer = JSONObject.parseObject(layer.toString(), MaxPoolLayer.class);
                cnNetWork.addLayer(maxPoolLayer);
            } else if (MeanPoolLayer.class.getSimpleName().equals(strType)) {
                MeanPoolLayer meanPoolLayer = JSONObject.parseObject(layer.toString(), MeanPoolLayer.class);
                cnNetWork.addLayer(meanPoolLayer);
            } else if (FullLayer.class.getSimpleName().equals(strType)) {
                FullLayer fullLayer = JSONObject.parseObject(layer.toString(), FullLayer.class);
                cnNetWork.addLayer(fullLayer);
            } else if (CrossEntropyLayer.class.getSimpleName().equals(strType)) {
                CrossEntropyLayer crossEntropyLayer = JSONObject.parseObject(layer.toString(), CrossEntropyLayer.class);
                cnNetWork.addLayer(crossEntropyLayer);
            } else if (DropoutLayer.class.getSimpleName().equals(strType)) {
                DropoutLayer dropoutLayer = JSONObject.parseObject(layer.toString(), DropoutLayer.class);
                cnNetWork.addLayer(dropoutLayer);
            } else if (SoftMaxLayer.class.getSimpleName().equals(strType)) {
                SoftMaxLayer softMaxLayer = JSONObject.parseObject(layer.toString(), SoftMaxLayer.class);
                cnNetWork.addLayer(softMaxLayer);
            } else if (DepthSeparableLayer.class.getSimpleName().equals(strType)) {
                DepthSeparableLayer depthSeparableLayer = JSONObject.parseObject(layer.toString(), DepthSeparableLayer.class);
                cnNetWork.addLayer(depthSeparableLayer);
            } else if (MnLayer.class.getSimpleName().equals(strType)) {
                MnLayer mnLayer = JSONObject.parseObject(layer.toString(), MnLayer.class);
                cnNetWork.addLayer(mnLayer);
            } else if (LnLayer.class.getSimpleName().equals(strType)) {
                LnLayer lnLayer = JSONObject.parseObject(layer.toString(), LnLayer.class);
                cnNetWork.addLayer(lnLayer);
            } else if (BnLayer.class.getSimpleName().equals(strType)) {
                BnLayer bnLayer = JSONObject.parseObject(layer.toString(), BnLayer.class);
                cnNetWork.addLayer(bnLayer);
            } else if (RnnLayer.class.getSimpleName().equals(strType)) {
                RnnLayer rnnLayer = JSONObject.parseObject(layer.toString(), RnnLayer.class);
                cnNetWork.addLayer(rnnLayer);
            } else if (PaddingLayer.class.getSimpleName().equals(strType)) {
                PaddingLayer paddingLayer = JSONObject.parseObject(layer.toString(), PaddingLayer.class);
                cnNetWork.addLayer(paddingLayer);
            }
            //激活层
            else if (ReLuLayer.class.getSimpleName().equals(strType)) {
                ReLuLayer reLuLayer = JSONObject.parseObject(layer.toString(), ReLuLayer.class);
                cnNetWork.addLayer(reLuLayer);
            } else if (SigmoidLayer.class.getSimpleName().equals(strType)) {
                SigmoidLayer sigmoidLayer = JSONObject.parseObject(layer.toString(), SigmoidLayer.class);
                cnNetWork.addLayer(sigmoidLayer);
            } else if (TanhLayer.class.getSimpleName().equals(strType)) {
                TanhLayer tanhLayer = JSONObject.parseObject(layer.toString(), TanhLayer.class);
                cnNetWork.addLayer(tanhLayer);
            }
            //输出层
            else if (MseLayer.class.getSimpleName().equals(strType)) {
                MseLayer mseLayer = JSONObject.parseObject(layer.toString(), MseLayer.class);
                cnNetWork.addLayer(mseLayer);
            } else if (CrossEntropyLayer.class.getSimpleName().equals(strType)) {
                CrossEntropyLayer crossEntropyLayer = JSONObject.parseObject(layer.toString(), CrossEntropyLayer.class);
                cnNetWork.addLayer(crossEntropyLayer);
            } else if (SoftMaxLayer.class.getSimpleName().equals(strType)) {
                SoftMaxLayer softMaxLayer = JSONObject.parseObject(layer.toString(), SoftMaxLayer.class);
                cnNetWork.addLayer(softMaxLayer);
            }


        }

        return cnNetWork;
    }

    //添加层到神经网络里面去
    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    //是否进入学习状态
    public void entryLearn() {
        //所有层进入learn状态
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.setTest(false);
        }
    }

    //是否进入工作状态
    public void entryTest() {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            layer.setTest(true);
        }
    }

    //卷积神经网络学习
    public int learn(Tensor tensorInput, Tensor expect, double gamma) {
//        tensorInput.show();


        int maxIndex = 0;
        int batchSize = getBatchSize();
        for (int b = 0; b < batchSize; b++) {
            //前向传播
            work(tensorInput);

            if (expect == null) {
                continue;
            }

            //误差计算
            Layer lastLayer = layers.get(layers.size() - 1);
            lastLayer.setExpect(expect);
            lastLayer.setGamma(gamma);
            //获取识别的值
            double[] result = lastLayer.a().getArray();
            maxIndex = UtilNeuralNet.maxIndex(result);
            //反向传播
            Tensor error = lastLayer.error();
            for (int i = layers.size() - 2; i >= 0; i--) {
                lastLayer = layers.get(i);
//                System.out.println(lastLayer.getName() + " back propagation ...\n");
//                error.show("error = ");
                error = lastLayer.backPropagation(error);
            }
        }

        setBatch(getBatch() + 1);//批次号自增

        return maxIndex;

    }

    public int learn(Tensor tensorInput, Tensor expect) {
        return learn(tensorInput, expect, 1);
    }

    //卷积神经网络工作
    public double[] work(Tensor tensorInput) {
        //前向传播
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
//            System.out.println(layer.getName() + " forward ...\n");
            layer.setBatch(getBatch());
            //需要判断layer的类型，如果是反池化，反卷积，或者合并和裁剪的时候，需要获取前面层的信息
            if (layer.getType().equals("MaxPoolBackLayer")) {
                MaxPoolBackLayer maxPoolBackLayer = (MaxPoolBackLayer) layer;
                String layerName = maxPoolBackLayer.getTargetName();
                Layer targetLayer = getLayer(layerName);
                maxPoolBackLayer.setMaxPoolLayer((MaxPoolLayer) targetLayer);
            }
            tensorInput = layer.forward(tensorInput);
        }
        double[] result = tensorInput.getArray();
        return result;
    }

    @Override
    public double totalError() {
        Layer lastLayer = layers.get(layers.size() - 1);
        //拿到期望值
        Tensor expect = lastLayer.getExpect();
        double[] a = lastLayer.a().getArray();
        double totalError = 0;
        for (int i = 0; i < expect.getWidth(); i++) {
            totalError = totalError + Math.pow(expect.get(i) - a[i], 2);
        }
        return totalError;
    }

    //获取layer根据层index
    public Layer getLayer(int index) {
        if (index >= layers.size()) {
            throw new RuntimeException(index + " must < layers.size()");
        }
        return layers.get(index);
    }


    //获取layer根据name
    public Layer getLayer(String name) {
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            if (name.equals(layer.getName())) {
                return layer;
            }
        }
        throw new RuntimeException(name + " not found");
    }

    public List<Layer> getLayers() {
        return layers;
    }


    public double getLearnRate() {
        return learnRate;
    }

    public void setLearnRate(double learnRate) {
        this.learnRate = learnRate;
        if (layers != null) {
            for (int i = 0; i < layers.size(); i++) {
                layers.get(i).setLearnRate(learnRate);
            }
        }
    }

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
        if (layers != null) {
            for (int i = 0; i < layers.size(); i++) {
                layers.get(i).setStep(step);
            }
        }
    }
}

