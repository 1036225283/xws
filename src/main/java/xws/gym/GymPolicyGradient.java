package xws.gym;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import xws.neuron.CNNetWork;
import xws.neuron.Tensor;
import xws.neuron.UtilNeuralNet;
import xws.neuron.layer.FullLayer;
import xws.neuron.layer.activation.ReLuLayer;
import xws.neuron.layer.output.SoftMaxLayer;

import java.util.ArrayList;
import java.util.List;


/**
 * 强化学习-Policy gradient
 * Created by xws on 2019/1/22.
 */
public class GymPolicyGradient {

    private List<GymStorePolicy> stores = new ArrayList<>();
    private List<GymEpisode> episodes = new ArrayList<>();
    private int episodeCount = 100;
    private CNNetWork cnNetWork;
    private int actionCount = 1;
    private String instanceId;
    private String envName = "CartPole-v0";
    private JSONArray observation;
    private double gamma = 0.99;
    private int episodeIndex = 0;


    public static void main(String[] args) {
        GymPolicyGradient gymQlearning = new GymPolicyGradient();
        gymQlearning.init();
        gymQlearning.run();
    }


    //    选择action
    public int chooseAction() {
        cnNetWork.entryTest();
        Tensor tensorInput = observationToTensor(observation);
        double[] out = cnNetWork.work(tensorInput);
//        先将概率转换成概率区间
        double[] actions = new double[out.length];
        double total = 0;
        for (int i = 0; i < actions.length; i++) {
            total = total + out[i];
            actions[i] = total;
        }
//        区间判断，选择action
        double random = Math.random();
        for (int i = 0; i < actions.length; i++) {
            if (random <= actions[i]) {
//                System.out.println("random = " + random + " action = " + i + " probability = " + JSON.toJSONString(out));
                return i;
            }
        }
//        System.out.println("选择概率出错了");
        return UtilNeuralNet.maxIndex(out);
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
    public void store(JSONArray observation, double reward, int action) {
        GymStorePolicy gymStore = new GymStorePolicy();
        gymStore.setObservation(observation);
        gymStore.setReward(reward);
        gymStore.setAction(action);
        stores.add(gymStore);
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

        int max_steps = 10000;
//        UtilPanel utilPanel = new UtilPanel();

//        主循环开始了
        for (int episode = 0; episode < 3000; episode++) {
            System.out.println("episode = " + episode);
            for (int nStep = 0; nStep < max_steps; nStep++) {
//                observationSpace(instanceId);
//                从sample请求数据，如果是自己训练ai的话，需要用ai的action代替sample的action
//                int nAction = GymClient.sampleAction(instanceId);
                int nAction = chooseAction();

//                执行单步操作，返回observation,reward,done,info四个信息
                JSONObject stepInfo = GymClient.step(instanceId, nAction, true);

//                如果done==true end
                boolean done = stepInfo.getBoolean("done");

//                拿到observation
                observation = stepInfo.getJSONArray("observation");
//                observation -> tensor -> image
//                Tensor tensor = GymClient.observationToTensor(observation);
//                utilPanel.show(tensor);

//                reward
                double reward = stepInfo.getDouble("reward");
                if (done) {
                    reward = -3;
                }
                store(observation, reward, nAction);
//                System.out.println("reward = " + reward);

                if (done) {
                    learn();
                    GymClient.reset(instanceId);
                    break;
                }

            }
            GymClient.reset(instanceId);
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
            cnNetWork.addLayer(new FullLayer("full2", actionCount, UtilNeuralNet.e() * 0.00000001));
            cnNetWork.addLayer(new SoftMaxLayer("softmax"));


            cnNetWork.entryTest();
            Tensor tensor = observationToTensor(observation);
            cnNetWork.work(tensor);
            cnNetWork.save(envName);
            System.out.println("create net success");
        }
    }

    //    测试手写数组识别
    public void learn() {
//        storeEpisode();
//        for (int i = 0; i < episodes.size(); i++) {
//            learn(episodes.get(i).getPolicies());
//        }
//
        rewards();
        learn(stores);
        stores = new ArrayList<>(); //  Policy Gradient every episode clean train data
    }

    public void storeEpisode() {
        double totalReward = rewards();

        GymEpisode episode = new GymEpisode();
        episode.setReward(totalReward);
        episode.setPolicies(stores);
        episodes.add(episode);


        removeSmallest();
//        if (episodes.size() < episodeCount) {
//            episodes.add(episode);
//            return;
//        }
//
//
//        if (episodes.size() >= episodeCount) {
//            episodes.set(episodeIndex, episode);
//        }
//
//        if (episodeIndex == episodeCount) {
//            episodeIndex = 0;
//        }
//
//        episodeIndex = episodeIndex + 1;

    }

    public void learn(List<GymStorePolicy> stores) {
        double learnRate = UtilNeuralNet.e() * 0.0000000000001;
        for (int x = 0; x < 1; x++) {
            cnNetWork.entryLearn();
            cnNetWork.setBatchSize(5);
            cnNetWork.setBatch(5);
//            learnRate = learnRate * 0.9;
            cnNetWork.setLearnRate(learnRate);
            for (int i = 0; i < stores.size(); i++) {
//                int index = (int) (Math.random() * stores.size());
                GymStorePolicy gymStore = stores.get(i);

                if (gymStore.getExpect() > 0) {
                    continue;
                }

                JSONArray observation = gymStore.getObservation();
                Tensor tensor = observationToTensor(observation);

                int action = gymStore.getAction();

                Tensor expects = UtilNeuralNet.oneHot(action, actionCount);
                int result = cnNetWork.learn(tensor, expects, gymStore.getExpect());
            }
        }
    }

    public boolean test() {
        double success = 0;
        for (int i = 0; i < stores.size(); i++) {
            GymStorePolicy gymStore = stores.get(i);
            JSONArray observation = gymStore.getObservation();
            Tensor tensor = observationToTensor(observation);
            int action = gymStore.getAction();
            int expect = (int) gymStore.getExpect();
            Tensor expects = UtilNeuralNet.oneHot(action, gymStore.getExpect(), actionCount);
            double[] result = cnNetWork.work(tensor);

        }
        return false;
    }

    //    total reward
    public double rewards() {
        double[] rewards = new double[stores.size()];
        double reward = 0;
        for (int i = stores.size() - 1; i >= 0; i--) {
            rewards[i] = reward * gamma + stores.get(i).getReward();
            reward = rewards[i];
        }

//        total reward
        double totalReward = 0;
        for (int i = 0; i < rewards.length; i++) {
            totalReward = rewards[i] + totalReward;
        }

        System.out.println("totalReward = " + totalReward);

        double average = UtilNeuralNet.average(rewards);
        for (int i = 0; i < rewards.length; i++) {
            rewards[i] = rewards[i] - average;
        }

//        variance
        double variance = UtilNeuralNet.variance(rewards);

        double standardDeviation = Math.sqrt(variance);
        for (int i = 0; i < rewards.length; i++) {
            rewards[i] = rewards[i] / standardDeviation;
        }

//        set expect
        for (int i = 0; i < rewards.length; i++) {
            stores.get(i).setExpect(0.01 * rewards[i]);
//
//            if (rewards[i] > 0) {
//                stores.get(i).setExpect(1);
//            } else {
//                stores.get(i).setExpect(-1);
//            }
        }

        System.out.println(JSON.toJSONString(rewards));

        return totalReward;

    }

    public void removeSmallest() {

        if (episodes.size() <= episodeCount) {
            return;
        }

        double reward = episodes.get(0).getReward();
        int index = 0;

        for (int i = 1; i < episodes.size(); i++) {
            GymEpisode episode = episodes.get(i);

            if (episode.getReward() < reward) {
                reward = episode.getReward();
                index = i;
            }
        }

        episodes.remove(index);
    }


}
