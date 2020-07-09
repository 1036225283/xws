package xws.test;

import xws.neuron.CNNetWork;

public class CreateNetTest {

    public static void main(String[] args) {
//        直接创建神经网络

//        通过json创建神经网络
        String json = "[{\"height\":0,\"num\":28,\"strideX\":0,\"strideY\":0,\"type\":\"RnnLayer\",\"width\":0},{\"activation\":\"relu\",\"height\":0,\"num\":32,\"strideX\":0,\"strideY\":0,\"type\":\"FullLayer\",\"width\":0},{\"height\":0,\"num\":10,\"strideX\":0,\"strideY\":0,\"type\":\"SoftmaxLayer\",\"width\":0}]\n";
        CNNetWork cnNetWork = new CNNetWork();
        cnNetWork = cnNetWork.create(json);
        System.out.println(cnNetWork);
    }
}
