package xws.neuron.layer;

import xws.neuron.Tensor;


/**
 * 抽象层的概念
 * 每一层单独计算，单独输出结果，单独计算误差，单独输出误差
 * Created by xws on 2019/2/19.
 */
public class Layer {


    private String type;//层的类型

    private String name;//层的名称-用户自定义

    private String activationType = "sigmoid";//激活函数类型

    private Tensor expect;//期望值

    private boolean isTest;//是否工作状态/学习状态

    private int batch;//当前批次

    private int prevBatch;//上一批次

    private double learnRate = 0;//学习率

    private int step = 0;//run step

    private double gamma = 1;//loss rate


    public Layer() {
        type = "full";
    }

    public Layer(String type) {
        this.type = type;
    }


    //正向传播
    public Tensor forward(Tensor tensor) {
        return null;
    }

    //反向传播
    public Tensor backPropagation(Tensor tensor) {
        return null;
    }

    //误差计算
    public Tensor error() {
        return null;
    }

    //获取输出值
    public Tensor a() {
        return null;
    }


    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public Tensor getExpect() {
        return expect;
    }

    public void setExpect(Tensor expect) {
        this.expect = expect;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }


    public String getActivationType() {
        return activationType;
    }

    public void setActivationType(String activationType) {
        this.activationType = activationType;
    }

    public boolean isTest() {
        return isTest;
    }

    public void setTest(boolean test) {
        isTest = test;
    }


    public int getBatch() {
        return batch;
    }

    public void setBatch(int batch) {
        prevBatch = this.batch;
        this.batch = batch;

    }

    public int getPrevBatch() {
        return prevBatch;
    }

    public void setPrevBatch(int prevBatch) {
        this.prevBatch = prevBatch;
    }

    public double getLearnRate() {
        return learnRate;
    }

    public void setLearnRate(double learnRate) {
        this.learnRate = learnRate;
    }

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public double getGamma() {
        return gamma;
    }

    public void setGamma(double gamma) {
        this.gamma = gamma;
    }
}
