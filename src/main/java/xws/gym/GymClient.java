package xws.gym;

import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import xws.neuron.Tensor;

import java.util.HashMap;
import java.util.Map;

import static xws.gym.HttpClient.doGet;
import static xws.gym.HttpClient.doPost;


/**
 * Created by xws on 2019/9/21.
 */
public class GymClient {

    //    获取环境列表
    public static JSONObject getEnvs() {
        String result = doGet("http://127.0.0.1:5000/v1/envs/");
        JSONObject jsonObject = JSON.parseObject(result);
        JSONObject array = jsonObject.getJSONObject("all_envs");
        System.out.println(array);
        return array;
    }

    //    创建环境,返回唯一标示
    public static String create(String env_id) {
        Map<String, Object> map = new HashMap<>();
        map.put("action", 1);
        map.put("render", true);
        map.put("env_id", env_id);
        String result = doPost("http://127.0.0.1:5000/v1/envs/", JSON.toJSONString(map));
        JSONObject jsonObject = JSON.parseObject(result);
        System.out.println(result);
        return jsonObject.getString("instance_id");
    }

    //    Get information (name and dimensions/bounds) of the env's action_space
    public static JSONObject actionSpace(String instanceId) {
        String url = "http://127.0.0.1:5000/v1/envs/{instanceId}/action_space/";
        url = url.replace("{instanceId}", instanceId);
        String result = doGet(url);
        JSONObject jsonObject = JSON.parseObject(result);
        return jsonObject.getJSONObject("info");
    }

    //    Reset the state of the environment and return an initial observation
    //    return observation
    public static JSONArray reset(String instanceId) {
        String url = "http://127.0.0.1:5000/v1/envs/{instanceId}/reset/";
        url = url.replace("{instanceId}", instanceId);
        String result = doPost(url, "");
        JSONObject jsonObject = JSON.parseObject(result);
        return jsonObject.getJSONArray("observation");
    }


    public static JSONObject step(String instanceId, int action, boolean render) {
        String url = "http://127.0.0.1:5000/v1/envs/{instanceId}/step/";
        url = url.replace("{instanceId}", instanceId);

        Map<String, Object> map = new HashMap<>();
        map.put("action", action);
        map.put("render", render);

        String result = doPost(url, JSON.toJSONString(map));
        return JSON.parseObject(result);
    }

    //    Get information (name and dimensions/bounds) of the env's observation_space
    public static String observationSpace(String instanceId) {
        String url = "http://127.0.0.1:5000/v1/envs/{instanceId}/observation_space/";
        url = url.replace("{instanceId}", instanceId);
        String result = doGet(url);
        System.out.println(result);
        return null;
    }

    //    Start monitoring
    public static void monitorStart(String instanceId, String directory, boolean force, boolean resume, boolean video_callable) {
        String url = "http://127.0.0.1:5000/v1/envs/{instanceId}/monitor/start/";
        url = url.replace("{instanceId}", instanceId);

        Map<String, Object> map = new HashMap<>();
        map.put("directory", directory);
        map.put("force", force);
        map.put("resume", resume);
        map.put("video_callable", video_callable);
        doPost(url, JSON.toJSONString(map));
    }


    //    Flush all monitor data to disk
    public static void monitorClose(String instanceId) {
        String url = "http://127.0.0.1:5000/v1/envs/{instanceId}/monitor/close/";
        url = url.replace("{instanceId}", instanceId);
        doPost(url, "");
    }

    //    Flush all monitor data to disk
    public static void upload(String instanceId, String directory) {
        String url = "http://127.0.0.1:5000/v1/envs/{instanceId}/monitor/close/";
        url = url.replace("{instanceId}", instanceId);
        Map<String, Object> map = new HashMap<>();
        map.put("directory", directory);
        map.put("api_key", "api_key");
        map.put("algorithm_id", "");
        doPost(url, JSON.toJSONString(map));
    }

    //    Request a server shutdown
    public static void shutdown(String instanceId) {
        String url = "http://127.0.0.1:5000/v1/shutdown/";
        doPost(url, "");
    }

    //    Sample action
    public static int sampleAction(String instanceId) {
        String url = "http://127.0.0.1:5000/v1/envs/{instanceId}/action_space/sample";
        url = url.replace("{instanceId}", instanceId);
        String result = doGet(url);
        JSONObject jsonObject = JSON.parseObject(result);
        return jsonObject.getIntValue("action");
    }

    //    Observation to tensor
    public static Tensor observationToTensor(JSONArray observation) {
        Tensor tensor = new Tensor();
        int height = observation.size();
        int width = observation.getJSONArray(0).size();
        tensor.setDepth(3);
        tensor.setHeight(height);
        tensor.setWidth(width);
        tensor.createArray();

        for (int h = 0; h < height; h++) {
            JSONArray widthInfo = observation.getJSONArray(h);
            for (int w = 0; w < width; w++) {
                JSONArray depthInfo = widthInfo.getJSONArray(w);
                for (int d = 0; d < 3; d++) {
                    tensor.set(d, h, w, depthInfo.getDouble(d));
                }
            }
        }

        return tensor;
    }

    public static void main(String[] args) {

//        String instanceId = create("CartPole-v0");
        String instanceId = create("DemonAttack-v0");
//        getEnvs();
        JSONObject action = actionSpace(instanceId);
        System.out.println(action);
        monitorStart(instanceId, "/tmp/test/", true, false, false);

        int episode_count = 100;
        int max_steps = 200;

        reset(instanceId);

//        UtilPanel utilPanel = new UtilPanel();

        for (int episode = 0; episode < episode_count; episode++) {
            System.out.println("episode = " + episode);
            for (int nStep = 0; nStep < max_steps; nStep++) {
//                observationSpace(instanceId);

//                从sample请求数据，如果是自己训练ai的话，需要用ai的action代替sample的action
                int nAction = sampleAction(instanceId);
//                执行单步操作，返回observation,reward,done,info四个信息
                JSONObject stepInfo = step(instanceId, nAction, true);
//                如果done==true end
                boolean done = stepInfo.getBoolean("done");
                if (done) {
                    break;
                }
//                拿到observation
                JSONArray observation = stepInfo.getJSONArray("observation");
//                observation -> tensor -> image
                Tensor tensor = observationToTensor(observation);
//                utilPanel.show(tensor);

//                reward
                double reward = stepInfo.getDouble("reward");
                System.out.println("reward = " + reward);
            }

        }

//        监控结束
        monitorClose(instanceId);

        shutdown(instanceId);

    }
}
