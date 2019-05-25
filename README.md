# XWS
A Java implement of Deep Neural Network.

# Build a CNN Network
```java       
        CNNetWork cnNetWork = new CNNetWork();
        cnNetWork.addLayer(new BnLayer("bn1"));
        cnNetWork.addLayer(new FilterLayer("filter1", "relu", 9, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
        cnNetWork.addLayer(new PoolLayer("pool1", 2, 2, 2, 2));
        cnNetWork.addLayer(new BnLayer("bn2"));
        cnNetWork.addLayer(new FilterLayer("filter2", "relu", 9, 3, 3, 1, 1, 0, UtilNeuralNet.e() * 0.0000000001));
        cnNetWork.addLayer(new PoolLayer("pool2", 2, 2, 2, 2));
        cnNetWork.addLayer(new BnLayer("bn3"));
        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
        cnNetWork.addLayer(new BnLayer("bn4"));
        cnNetWork.addLayer(new FullLayer("full2", "relu", 128, UtilNeuralNet.e() * 0.00000000001));
        cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));
        List<Cifar10> list = UtilMnist.testData();
        Cifar10 cifar10 = list.get(0);
        UtilNeuralNet.initMinst(cifar10.getRgb().getArray());
        cnNetWork.entryTest();
        cnNetWork.learn(cifar10.getRgb(), expectMNIST(cifar10.getLabel()));
        cnNetWork.save(strName);

```
# Build a RNN Network
```JAVA 
List<RnnSequence> list = createSequenceMNIST(UtilMnist.learnData());
         List<RnnSequence> listTest = createSequenceMNIST(UtilMnist.testData());
 
         CNNetWork cnNetWork = CNNetWork.load("RNN_MNIST_LN");
 //        CNNetWork cnNetWork = new CNNetWork();
         //93.92%
         //95.52%
         cnNetWork.addLayer(new RnnLayer("rnn1", "relu", 28));
         cnNetWork.addLayer(new FullLayer("full2", "relu", 32, UtilNeuralNet.e() * 0.00000000001));
         cnNetWork.addLayer(new SoftmaxLayer("softmax", 10, UtilNeuralNet.e() * 0.00000000001));
 //        double learnRate = UtilNeuralNet.e() * 0.00001;
 //        double learnRate = UtilNeuralNet.e() * 0.000001;
 //        double learnRate = UtilNeuralNet.e() * 0.0000001;
         double learnRate = UtilNeuralNet.e() * 0.00000001;
 
 
 //
         cnNetWork.entryLearn();
         cnNetWork.setBatchSize(1);
         cnNetWork.setLearnRate(learnRate);
 
         int batch = 2000;
         for (int i = 0; i < list.size(); i = i + batch) {
             System.out.println("i = " + i);
             //将这一批数据，反复喂给
             for (int k = 0; k < 3; k++) {
                 cnNetWork.entryLearn();
                 for (int j = 0; j < batch; j++) {
                     RnnSequence rnnSequence = list.get(i + j);
 
                     for (int n = 0; n < 10; n++) {
                         cnNetWork.setStep(0);
                         for (int r = 0; r < rnnSequence.size() - 1; r++) {
                             cnNetWork.learn(rnnSequence.getData(r), null);
                         }
                         cnNetWork.learn(rnnSequence.getData(27), expectMNIST(rnnSequence.get(27).getValue()));
                     }
                 }
 
                 UtilCifar10.testRnn(cnNetWork, listTest);
             }
         }
 
         cnNetWork.save("RNN_MNIST_LN");
```
# Pull Request
Pull request is welcome.
# communicate with
QQ group: 1036225283

# Features
1. Batch gradient descent is not used，Using online learning。
2. Instead of using LSTM, RNN USES the ResNet residual
3. without any dependency
4. Basic layer: input layer, dropout layer,filter layer, pooling layer(MAX), full connect layer, softmax layer, rnn layer 
5. Loss function: Cross Entropy,log like-hood ,MSE loss
6. active funcs:sigmod , tanh, relu
7. L2 regularization is supported.

# Test and Performance
## DNN
1. mnist recognition success rate is 99%
## CNN
1. mnist recognition success rate is 99%
2. cifar10 recognition success rate is 65%
## RNN
1. XOR test ok
2. ADD test ok
3. mnist recognition success rate is 95%

# License MIT

