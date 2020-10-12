package xws.neuron;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import xws.neuron.json.LayerJson;
import xws.neuron.layer.*;
import xws.neuron.layer.bn.BnLayer;
import xws.neuron.layer.bn.LnLayer;
import xws.neuron.layer.bn.MnLayer;
import xws.neuron.layer.conv.Conv1DLayer;
import xws.neuron.layer.conv.ConvolutionLayer;
import xws.neuron.layer.conv.DepthSeparableLayer;
import xws.neuron.layer.pool.MaxPool1DLayer;
import xws.neuron.layer.pool.MaxPoolBackLayer;
import xws.neuron.layer.pool.MaxPoolLayer;
import xws.neuron.layer.pool.MeanPoolLayer;
import xws.neuron.layer.rnn.BidirectionalRnnLayer;
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
    public int learn(Tensor tensorInput, double[] expect, double gamma) {
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
            for (int i = layers.size() - 1; i >= 0; i--) {
                lastLayer = layers.get(i);
                error = lastLayer.backPropagation(error);
            }
        }

        setBatch(getBatch() + 1);//批次号自增

        return maxIndex;

    }

    public int learn(Tensor tensorInput, double[] expect) {
        return learn(tensorInput, expect, 1);
    }

    //卷积神经网络工作
    public double[] work(Tensor tensorInput) {
        //前向传播
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
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
                convolutionLayer.setNum(convolutionLayer.getFilters().length);
                cnNetWork.addLayer(convolutionLayer);
            } else if (Conv1DLayer.class.getSimpleName().equals(strType)) {
                Conv1DLayer conv1DLayer = JSONObject.parseObject(layer.toString(), Conv1DLayer.class);
                conv1DLayer.setNum(conv1DLayer.getFilters().length);
                cnNetWork.addLayer(conv1DLayer);
            } else if (MaxPoolLayer.class.getSimpleName().equals(strType)) {
                MaxPoolLayer maxPoolLayer = JSONObject.parseObject(layer.toString(), MaxPoolLayer.class);
                cnNetWork.addLayer(maxPoolLayer);
            } else if (MaxPool1DLayer.class.getSimpleName().equals(strType)) {
                MaxPool1DLayer maxPool1DLayer = JSONObject.parseObject(layer.toString(), MaxPool1DLayer.class);
                cnNetWork.addLayer(maxPool1DLayer);
            } else if (MeanPoolLayer.class.getSimpleName().equals(strType)) {
                MeanPoolLayer meanPoolLayer = JSONObject.parseObject(layer.toString(), MeanPoolLayer.class);
                cnNetWork.addLayer(meanPoolLayer);
            } else if ("full".equals(strType) || FullLayer.class.getSimpleName().equals(strType)) {
                FullLayer fullLayer = JSONObject.parseObject(layer.toString(), FullLayer.class);
                fullLayer.setType(FullLayer.class.getSimpleName());
                fullLayer.setNum(fullLayer.getBias().length);
                cnNetWork.addLayer(fullLayer);
            } else if (CrossEntropyLayer.class.getSimpleName().equals(strType)) {
                CrossEntropyLayer crossEntropyLayer = JSONObject.parseObject(layer.toString(), CrossEntropyLayer.class);
                cnNetWork.addLayer(crossEntropyLayer);
            } else if (DropoutLayer.class.getSimpleName().equals(strType)) {
                DropoutLayer dropoutLayer = JSONObject.parseObject(layer.toString(), DropoutLayer.class);
                cnNetWork.addLayer(dropoutLayer);
            } else if (SoftmaxLayer.class.getSimpleName().equals(strType)) {
                SoftmaxLayer softmaxLayer = JSONObject.parseObject(layer.toString(), SoftmaxLayer.class);
                softmaxLayer.setNum(softmaxLayer.getBias().length);
                cnNetWork.addLayer(softmaxLayer);
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
            } else if (BidirectionalRnnLayer.class.getSimpleName().equals(strType)) {
                BidirectionalRnnLayer bidirectionalRnnLayer = JSONObject.parseObject(layer.toString(), BidirectionalRnnLayer.class);
                cnNetWork.addLayer(bidirectionalRnnLayer);
            }


        }

        return cnNetWork;
    }


    @Override
    public double totalError() {
        Layer lastLayer = layers.get(layers.size() - 1);
        //拿到期望值
        double[] expect = lastLayer.getExpect();
        double[] a = lastLayer.a().getArray();
        double totalError = 0;
        for (int i = 0; i < expect.length; i++) {
            totalError = totalError + Math.pow(expect[i] - a[i], 2);
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

    public List<LayerJson> structure() {
        List<LayerJson> layerJsons = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            Layer layer = layers.get(i);
            LayerJson layerJson = new LayerJson();
            layerJson.setType(layer.getType());
            if (layer instanceof FullLayer) {
                FullLayer fullLayer = (FullLayer) layer;
                layerJson.setNum(fullLayer.getNum());
                layerJson.setActivation(fullLayer.getActivationType());
            } else if (layer instanceof RnnLayer) {
                RnnLayer rnnLayer = (RnnLayer) layer;
                layerJson.setNum(rnnLayer.getNum());
            } else if (layer instanceof ConvolutionLayer) {
                ConvolutionLayer convolutionLayer = (ConvolutionLayer) layer;
                layerJson.setNum(convolutionLayer.getNum());
                layerJson.setActivation(convolutionLayer.getActivationType());
                layerJson.setHeight(convolutionLayer.getHeight());
                layerJson.setWidth(convolutionLayer.getWidth());
                layerJson.setStrideX(convolutionLayer.getStrideX());
                layerJson.setStrideY(convolutionLayer.getStrideY());
            } else if (layer instanceof Conv1DLayer) {
                Conv1DLayer conv1DLayer = (Conv1DLayer) layer;
                layerJson.setNum(conv1DLayer.getNum());
                layerJson.setActivation(conv1DLayer.getActivationType());
                layerJson.setWidth(conv1DLayer.getWidth());
                layerJson.setStrideX(conv1DLayer.getStrideX());
                layerJson.setStrideY(conv1DLayer.getStrideY());
            } else if (layer instanceof MaxPoolLayer) {
                MaxPoolLayer maxPoolLayer = (MaxPoolLayer) layer;
                layerJson.setHeight(maxPoolLayer.getPool().getHeight());
                layerJson.setWidth(maxPoolLayer.getPool().getWidth());
                layerJson.setStrideX(maxPoolLayer.getPool().getStrideX());
                layerJson.setStrideY(maxPoolLayer.getPool().getStrideY());
            } else if (layer instanceof MaxPool1DLayer) {
                MaxPool1DLayer maxPool1DLayer = (MaxPool1DLayer) layer;
                layerJson.setHeight(maxPool1DLayer.getPool().getHeight());
                layerJson.setWidth(maxPool1DLayer.getPool().getWidth());
                layerJson.setStrideX(maxPool1DLayer.getPool().getStrideX());
                layerJson.setStrideY(maxPool1DLayer.getPool().getStrideY());
            } else if (layer instanceof SoftmaxLayer) {
                SoftmaxLayer softmaxLayer = (SoftmaxLayer) layer;
                layerJson.setNum(softmaxLayer.getNum());
            } else if (layer instanceof LnLayer) {
                LnLayer maxPoolLayer = (LnLayer) layer;
            } else if (layer instanceof PaddingLayer) {
                PaddingLayer maxPoolLayer = (PaddingLayer) layer;
            } else if (layer instanceof BnLayer) {
                BnLayer maxPoolLayer = (BnLayer) layer;
            } else {
                throw new RuntimeException("structure error");
            }


            layerJsons.add(layerJson);
        }
        return layerJsons;
    }

    public CNNetWork create(String json) {
        CNNetWork cnNetWork = this;

        List<LayerJson> layerJsons = JSONArray.parseArray(json, LayerJson.class);
        for (int i = 0; i < layerJsons.size(); i++) {
            LayerJson layerJson = layerJsons.get(i);
            if (ConvolutionLayer.class.getSimpleName().equals(layerJson.getType())) {
                ConvolutionLayer convolutionLayer = new ConvolutionLayer(layerJson.getName(), layerJson.getActivation(), layerJson.getNum(), layerJson.getHeight(), layerJson.getWidth(), layerJson.getStrideX(), layerJson.getStrideY(), 0);
                cnNetWork.addLayer(convolutionLayer);
            } else if (Conv1DLayer.class.getSimpleName().equals(layerJson.getType())) {
                Conv1DLayer conv1DLayer = new Conv1DLayer(layerJson.getName(), layerJson.getActivation(), layerJson.getNum(), layerJson.getWidth(), layerJson.getStrideX(), 0);
                cnNetWork.addLayer(conv1DLayer);
            } else if (MaxPoolLayer.class.getSimpleName().equals(layerJson.getType())) {
                MaxPoolLayer maxPoolLayer = new MaxPoolLayer(layerJson.getHeight(), layerJson.getWidth(), layerJson.getStrideX(), layerJson.getStrideY());
                cnNetWork.addLayer(maxPoolLayer);
            } else if (MaxPool1DLayer.class.getSimpleName().equals(layerJson.getType())) {
                MaxPool1DLayer maxPool1DLayer = new MaxPool1DLayer(layerJson.getWidth(), layerJson.getStrideX());
                cnNetWork.addLayer(maxPool1DLayer);
            } else if (MeanPoolLayer.class.getSimpleName().equals(layerJson.getType())) {
                MeanPoolLayer meanPoolLayer = new MeanPoolLayer(layerJson.getHeight(), layerJson.getWidth(), layerJson.getStrideX(), layerJson.getStrideY());
                cnNetWork.addLayer(meanPoolLayer);
            } else if ("full".equals(layerJson.getType()) || FullLayer.class.getSimpleName().equals(layerJson.getType())) {
                FullLayer fullLayer = new FullLayer(layerJson.getName(), layerJson.getActivation(), layerJson.getNum());
                cnNetWork.addLayer(fullLayer);
            } else if (CrossEntropyLayer.class.getSimpleName().equals(layerJson.getType())) {
                CrossEntropyLayer crossEntropyLayer = new CrossEntropyLayer();
                cnNetWork.addLayer(crossEntropyLayer);
            } else if (DropoutLayer.class.getSimpleName().equals(layerJson.getType())) {
                DropoutLayer dropoutLayer = new DropoutLayer();
                cnNetWork.addLayer(dropoutLayer);
            } else if (SoftmaxLayer.class.getSimpleName().equals(layerJson.getType())) {
                SoftmaxLayer softmaxLayer = new SoftmaxLayer(layerJson.getNum());
                cnNetWork.addLayer(softmaxLayer);
            } else if (DepthSeparableLayer.class.getSimpleName().equals(layerJson.getType())) {
                DepthSeparableLayer depthSeparableLayer = new DepthSeparableLayer();
                cnNetWork.addLayer(depthSeparableLayer);
            } else if (MnLayer.class.getSimpleName().equals(layerJson.getType())) {
                MnLayer mnLayer = new MnLayer();
                cnNetWork.addLayer(mnLayer);
            } else if (LnLayer.class.getSimpleName().equals(layerJson.getType())) {
                LnLayer lnLayer = new LnLayer();
                cnNetWork.addLayer(lnLayer);
            } else if (BnLayer.class.getSimpleName().equals(layerJson.getType())) {
                BnLayer bnLayer = new BnLayer();
                cnNetWork.addLayer(bnLayer);
            } else if (RnnLayer.class.getSimpleName().equals(layerJson.getType())) {
                RnnLayer rnnLayer = new RnnLayer(layerJson.getNum());
                rnnLayer.setInitFlag(false);
                cnNetWork.addLayer(rnnLayer);
            } else if (PaddingLayer.class.getSimpleName().equals(layerJson.getType())) {
                PaddingLayer paddingLayer = new PaddingLayer();
                cnNetWork.addLayer(paddingLayer);
            }
        }
        return cnNetWork;
    }
}

