package xws.gym;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import xws.neuron.CNNetWork;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.FullLayer;
import xws.neuron.layer.activation.ReLuLayer;
import xws.neuron.layer.output.MseLayer;

import java.util.ArrayList;
import java.util.List;

import static xws.test.FullNetWorkTest.expect;
import static xws.test.FullNetWorkTest.expectMNIST;


/**
 * 强化学习-Q learning
 * Created by xws on 2019/1/22.
 */
public class GymQlearning {

    private List<GymStore> stores = new ArrayList<>();
    private int storeIndex = 0;
    private int episode_count = 300;
    private float epsilon = 0.9f;//
    private CNNetWork cnNetWork;
    private int actionCount = 1;
    private String instanceId;
    private String envName = "CartPole-v0";
    private JSONArray observation;
    private double gamma = 0.95;


    public static void main(String[] args) {
        GymQlearning gymQlearning = new GymQlearning();
        gymQlearning.init();
        gymQlearning.run();
    }


    //    选择action
    public int chooseAction() {
        cnNetWork.entryTest();
        Tensor tensorInput = observationToTensor(observation);
        double[] out = cnNetWork.work(tensorInput);
        System.out.println("out" + JSON.toJSONString(out));
        return UtilNeuralNet.maxIndex(out);
    }

    public int greedyAction() {
        if (Math.random() < epsilon) {
            return chooseAction();
        } else {
            return (int) (Math.random() * (float) actionCount);
        }
    }

    public Tensor observationToTensor(JSONArray observation) {
        Tensor tensor = new Tensor();
        tensor.setWidth(observation.size());
        tensor.createArray();
        for (int i = 0; i < observation.size(); i++) {
            tensor.set(i, observation.getDouble(i));
        }
        return tensor;
    }

    //    store
    public void store(boolean done, JSONArray observation, double reward, int action, JSONArray nextObservation) {
        GymStore gymStore = new GymStore();
        gymStore.setDone(done);
        gymStore.setObservation(observation);
        gymStore.setReward(reward);
        gymStore.setAction(action);
        gymStore.setNextObservation(nextObservation);

        if (stores.size() < 200) {
            stores.add(gymStore);
            return;
        }


        if (storeIndex == 199) {
            storeIndex = 0;
        }

        stores.set(storeIndex, gymStore);
        storeIndex = storeIndex + 1;
    }

    //    初始化操作
    public void init() {
        //        String instanceId = GymClient.create("DemonAttack-v0");
        instanceId = GymClient.create(envName);
        JSONObject action = GymClient.actionSpace(instanceId);
        GymClient.monitorStart(instanceId, "/tmp/test/", true, false, false);
        observation = GymClient.reset(instanceId);
        actionCount = action.getIntValue("n");
        System.out.println("action = " + action);
//        创建神经网络
        createCNNetWork();
    }

    public void run() {

        int max_steps = 200;
//        UtilPanel utilPanel = new UtilPanel();

//        主循环开始了
        for (int episode = 0; episode < episode_count; episode++) {
            System.out.println("episode = " + episode);
            for (int nStep = 0; nStep < max_steps; nStep++) {
//                observationSpace(instanceId);
//                从sample请求数据，如果是自己训练ai的话，需要用ai的action代替sample的action
//                int nAction = GymClient.sampleAction(instanceId);
                int nAction = greedyAction();

//                执行单步操作，返回observation,reward,done,info四个信息
                JSONObject stepInfo = GymClient.step(instanceId, nAction, true);

//                如果done==true end
                boolean done = stepInfo.getBoolean("done");

//                拿到observation
                JSONArray nextObservation = stepInfo.getJSONArray("observation");
//                observation -> tensor -> image
//                Tensor tensor = GymClient.observationToTensor(observation);
//                utilPanel.show(tensor);

//                reward
                double reward = stepInfo.getDouble("reward");
                if (done) {
                    reward = -1;
                }
                System.out.println("reward = " + reward);
                store(done, observation, reward, nAction, nextObservation);
                observation = nextObservation;
                learn();
                if (done) {
                    GymClient.reset(instanceId);
                    break;
                }
            }

        }

//        监控结束
        GymClient.monitorClose(instanceId);

        GymClient.shutdown(instanceId);
        cnNetWork.save(envName);

    }


    public void createCNNetWork() {

        try {
            cnNetWork = CNNetWork.load(envName);
            System.out.println("load net success");
        } catch (Exception e) {
            //97.56%
            cnNetWork = new CNNetWork();

            cnNetWork.addLayer(new FullLayer("full0", 64, UtilNeuralNet.e() * 0.00000001));
            cnNetWork.addLayer(new ReLuLayer("sigmoid0"));
            cnNetWork.addLayer(new FullLayer("full1", 32, UtilNeuralNet.e() * 0.00000001));
            cnNetWork.addLayer(new ReLuLayer("sigmoid1"));
            cnNetWork.addLayer(new FullLayer("full2", 10, UtilNeuralNet.e() * 0.00000001));
            cnNetWork.addLayer(new ReLuLayer("sigmoid2"));
            cnNetWork.addLayer(new MseLayer("mse"));


            //98.91%    ||  98.87%  ||  99.02%
//        cnNetWork.addLayer(new BnLayer("ln1"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter1", "relu", 6, 5, 5, 1, 1, 0, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new MaxPoolLayer("pool1", 2, 2, 2, 2));
//        cnNetWork.addLayer(new BnLayer("ln1"));
//        cnNetWork.addLayer(new ConvolutionLayer("filter2", "relu", 16, 5, 5, 1, 1, 0));
//        cnNetWork.addLayer(new MaxPoolLayer("pool2", 2, 2, 2, 2));
//        cnNetWork.addLayer(new BnLayer("ln1"));
//        cnNetWork.addLayer(new FullLayer("full2", "relu", 64, UtilNeuralNet.e() * 0.00000000001));
//        cnNetWork.addLayer(new CrossEntropyLayer("cross-entropy", "sigmoid", 10, UtilNeuralNet.e() * 0.0000001));
            cnNetWork.entryTest();
            Tensor tensor = observationToTensor(observation);
            cnNetWork.work(tensor);
            cnNetWork.save(envName);
            System.out.println("create net success");
        }
    }


    //测试手写数组识别
    public void learn() {
        double learnRate = UtilNeuralNet.e() * 0.001;
        for (int x = 0; x < 10; x++) {
            cnNetWork.entryLearn();
            cnNetWork.setBatchSize(5);
            cnNetWork.setBatch(10);
            learnRate = learnRate * 0.9;
            cnNetWork.setLearnRate(learnRate);
            int batch = stores.size();
            for (int i = 0; i < stores.size(); i = i + batch) {
                GymStore gymStore = stores.get(i);

                JSONArray nextObservation = gymStore.getNextObservation();
                Tensor nextTensor = observationToTensor(nextObservation);


                JSONArray observation = gymStore.getObservation();
                Tensor tensor = observationToTensor(observation);


                double[] out = cnNetWork.work(nextTensor);
                double max = UtilNeuralNet.max(out);

                double reward = gymStore.getReward();
                if (!gymStore.isDone()) {
                    reward = reward + gamma * max;
                }
                int action = gymStore.getAction();
                Tensor expects = expect(action, reward, actionCount);

                int result = cnNetWork.learn(tensor, expects);

            }
        }
    }
}
