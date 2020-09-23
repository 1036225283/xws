package xws.neuron.layer.rnn;

import xws.neuron.ActivationFunction;
import xws.neuron.Tensor;
import xws.neuron.layer.Layer;

import java.util.List;

/**
 * time layer 存储时序列层
 * Created by xws on 2019/5/17.
 */
public class RnnTime {


    private List<Tensor> shareListErrorInputWeight;//输入权重误差
    private List<Tensor> shareListErrorInputPreviousWeight;//上一时刻输入权重误差
    private List<Tensor> shareListErrorInputResidualWeight;//残差输入权重误差
    private Tensor shareErrorBias;//偏置的误差


    private List<Tensor> shareListInputWeight;//共享的输入权重
    private List<Tensor> shareListInputPreviousWeight;//共享的上一时刻输入权重
    private List<Tensor> shareListInputResidualWeight;//共享的残差输入权重
    private Tensor shareBias;//共享的偏置


    private Tensor tensorInput;//输入
    private Tensor tensorInputPrevious;//上一时刻输入
    private Tensor tensorInputResidual;//残差输入


    private double[] z;//激活之前的值
    private double[] pdz;//求激活结果求导
    private Tensor a;//激活之后的值
    private Tensor tensorErrorOut;//对输出求导
    private Tensor tensorInputError;//输入误差


    private List<RnnTime> rnnTimes;

    private Layer layer;

    private int step;//当前的时刻

    public RnnTime() {
    }

    //传入上下文，共享输入权重，共享time权重，共享残差权重
    public RnnTime(List<RnnTime> rnnTimes, List<Tensor> shareListInputWeight, List<Tensor> shareListInputPreviousWeight, List<Tensor> shareListInputResidualWeight) {
        this.rnnTimes = rnnTimes;
        this.shareListInputWeight = shareListInputWeight;
        this.shareListInputPreviousWeight = shareListInputPreviousWeight;
        this.shareListInputResidualWeight = shareListInputResidualWeight;
    }

    //前向传播
    public Tensor forward() {
        //进行初始化工作
        if (z == null) {
            z = new double[shareListInputWeight.size()];
            a = new Tensor();
            a.setWidth(shareListInputWeight.size());
            a.createArray();

            tensorErrorOut = new Tensor();
            tensorErrorOut.setWidth(shareListInputWeight.size());
            tensorErrorOut.createArray();

            tensorInputError = new Tensor();
            tensorInputError.setWidth(shareListInputWeight.get(0).getWidth());
            tensorInputError.createArray();

        }
        //每个神经元都计算一次
        for (int i = 0; i < shareListInputWeight.size(); i++) {
            //首先计算自己的输入
            Tensor w = shareListInputWeight.get(i);
            Tensor p = shareListInputPreviousWeight.get(i);
            Tensor r = shareListInputResidualWeight.get(i);

            double total = w.dot(tensorInput);
            if (step > 0) {
                total = total + p.dot(tensorInputPrevious);
            }

            if (step - 2 >= 0) {
                total = total + r.dot(tensorInputResidual);
            }

            total = total + shareBias.get(i);
            z[i] = total;
            a.set(i, ActivationFunction.activation(total, layer.getActivationType()));
        }

        return a;

    }

    //反向传播
    public void backPropagation() {

        //对误差进行转换
        pdz();

        //对每一个误差进行循环处理
        for (int i = 0; i < tensorErrorOut.getWidth(); i++) {


            //取出每一个神经元对呀的误差
            double neureError = pdz[i];
            //偏置的误差
            double errBias = shareErrorBias.get(i);
            errBias = errBias + neureError;
            shareErrorBias.set(i, errBias);


            //输入权重的误差 = 误差 * 输入

            //误差
            Tensor tensorErrorInputWeight = shareListErrorInputWeight.get(i);//每一个神经元的输入权重误差

            //对每一个神经元的输入权重进行循环处理，
            for (int w = 0; w < tensorInput.getWidth(); w++) {
                double widthError = tensorErrorInputWeight.get(w);
                widthError = widthError + neureError * tensorInput.get(w);
                tensorErrorInputWeight.set(w, widthError);
            }

            //输入的误差 = 误差 * 权重
            //权重
            Tensor shareTensorInputWeight = shareListInputWeight.get(i);
            for (int w = 0; w < tensorInput.getWidth(); w++) {
                double widthError = tensorInputError.get(w);
                widthError = widthError + neureError * shareTensorInputWeight.get(w);
                tensorInputError.set(w, widthError);
            }

            //其次，计算上一时刻的误差
            if (step > 0) {
                //计算上一时刻输入权重误差 = 误差 * 输入

                //误差
                Tensor shareTensorErrorInputPreviousWeight = shareListErrorInputPreviousWeight.get(i);

                //权重
                Tensor shareTensorInputPreviousWeight = shareListInputPreviousWeight.get(i);

                for (int w = 0; w < shareTensorInputPreviousWeight.getWidth(); w++) {
                    double widthError = shareTensorErrorInputPreviousWeight.get(w);
                    widthError = widthError + neureError * tensorInputPrevious.get(w);
                    shareTensorErrorInputPreviousWeight.set(w, widthError);
                }

                //计算上一时刻输入进行误差
                Tensor tensorErrorOut = rnnTimes.get(step - 1).getTensorErrorOut();
                for (int w = 0; w < shareTensorInputPreviousWeight.getWidth(); w++) {
                    double widthError = tensorErrorOut.get(w);
                    widthError = widthError + neureError * shareTensorInputPreviousWeight.get(w);
                    tensorErrorOut.set(w, widthError);
                }

            }

            //最后，计算skip connection 的误差
            if (step - 2 >= 0) {
                //计算共享权重的误差 = 误差 * 输入

                //误差
                Tensor shareTensorErrorInputResidualWeight = shareListErrorInputResidualWeight.get(i);
                //权重
                Tensor shareTensorInputResidualWeight = shareListInputResidualWeight.get(i);
                //计算误差
                for (int w = 0; w < shareTensorInputResidualWeight.getWidth(); w++) {
                    double widthError = shareTensorErrorInputResidualWeight.get(w);
                    widthError = widthError + neureError * tensorInputResidual.get(w);
                    shareTensorErrorInputResidualWeight.set(w, widthError);
                }

                //计算残差输入的误差 = 误差 * 权重

                //误差
                Tensor tensorErrorOut = rnnTimes.get(step - 2).getTensorErrorOut();
                for (int w = 0; w < shareTensorInputResidualWeight.getWidth(); w++) {
                    double widthError = tensorErrorOut.get(w);
                    widthError = widthError + neureError * shareTensorInputResidualWeight.get(w);
                    tensorErrorOut.set(w, widthError);
                }

            }


        }

    }

    //计算pdz
    private void pdz() {
        pdz = new double[z.length];
        for (int i = 0; i < tensorErrorOut.getWidth(); i++) {
            pdz[i] = ActivationFunction.derivation(z[i], layer.getActivationType()) * tensorErrorOut.get(i);
        }
    }

    public Tensor getTensorErrorOut() {
        return tensorErrorOut;
    }

    public void setTensorErrorOut(Tensor tensorErrorOut) {
        this.tensorErrorOut = tensorErrorOut;
    }


    public List<Tensor> getShareListErrorInputWeight() {
        return shareListErrorInputWeight;
    }

    public void setShareListErrorInputWeight(List<Tensor> shareListErrorInputWeight) {
        this.shareListErrorInputWeight = shareListErrorInputWeight;
    }

    public List<Tensor> getShareListErrorInputPreviousWeight() {
        return shareListErrorInputPreviousWeight;
    }

    public void setShareListErrorInputPreviousWeight(List<Tensor> shareListErrorInputPreviousWeight) {
        this.shareListErrorInputPreviousWeight = shareListErrorInputPreviousWeight;
    }

    public List<Tensor> getShareListErrorInputResidualWeight() {
        return shareListErrorInputResidualWeight;
    }

    public void setShareListErrorInputResidualWeight(List<Tensor> shareListErrorInputResidualWeight) {
        this.shareListErrorInputResidualWeight = shareListErrorInputResidualWeight;
    }

    public List<Tensor> getShareListInputWeight() {
        return shareListInputWeight;
    }

    public void setShareListInputWeight(List<Tensor> shareListInputWeight) {
        this.shareListInputWeight = shareListInputWeight;
    }

    public List<Tensor> getShareListInputPreviousWeight() {
        return shareListInputPreviousWeight;
    }

    public void setShareListInputPreviousWeight(List<Tensor> shareListInputPreviousWeight) {
        this.shareListInputPreviousWeight = shareListInputPreviousWeight;
    }

    public List<Tensor> getShareListInputResidualWeight() {
        return shareListInputResidualWeight;
    }

    public void setShareListInputResidualWeight(List<Tensor> shareListInputResidualWeight) {
        this.shareListInputResidualWeight = shareListInputResidualWeight;
    }

    public Tensor getTensorInput() {
        return tensorInput;
    }

    public void setTensorInput(Tensor tensorInput) {
        this.tensorInput = tensorInput;
    }

    public Tensor getTensorInputPrevious() {
        return tensorInputPrevious;
    }

    public void setTensorInputPrevious(Tensor tensorInputPrevious) {
        this.tensorInputPrevious = tensorInputPrevious;
    }

    public Tensor getTensorInputResidual() {
        return tensorInputResidual;
    }

    public void setTensorInputResidual(Tensor tensorInputResidual) {
        this.tensorInputResidual = tensorInputResidual;
    }


    public double[] getZ() {
        return z;
    }

    public void setZ(double[] z) {
        this.z = z;
    }

    public double[] getPdz() {
        return pdz;
    }

    public void setPdz(double[] pdz) {
        this.pdz = pdz;
    }

    public Tensor getA() {
        return a;
    }

    public void setA(Tensor a) {
        this.a = a;
    }

    public Tensor getShareBias() {
        return shareBias;
    }

    public void setShareBias(Tensor shareBias) {
        this.shareBias = shareBias;
    }

    public List<RnnTime> getRnnTimes() {
        return rnnTimes;
    }

    public void setRnnTimes(List<RnnTime> rnnTimes) {
        this.rnnTimes = rnnTimes;
    }

    public Layer getLayer() {
        return layer;
    }

    public void setLayer(Layer layer) {
        this.layer = layer;
    }

    public int getStep() {
        return step;
    }

    public void setStep(int step) {
        this.step = step;
    }

    public Tensor getShareErrorBias() {
        return shareErrorBias;
    }

    public void setShareErrorBias(Tensor shareErrorBias) {
        this.shareErrorBias = shareErrorBias;
    }

    public Tensor getTensorInputError() {
        return tensorInputError;
    }

    public void setTensorInputError(Tensor tensorInputError) {
        this.tensorInputError = tensorInputError;
    }
}
