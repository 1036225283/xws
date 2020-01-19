package xws.gym;

import com.alibaba.fastjson.JSONArray;

/**
 * store train data
 * Created by xws on 2019/10/1.
 */
public class GymStorePolicy {

    private int action;
    private JSONArray observation;
    private double reward;
    private double expect;

    public int getAction() {
        return action;
    }

    public void setAction(int action) {
        this.action = action;
    }

    public JSONArray getObservation() {
        return observation;
    }

    public void setObservation(JSONArray observation) {
        this.observation = observation;
    }

    public double getReward() {
        return reward;
    }

    public void setReward(double reward) {
        this.reward = reward;
    }

    public double getExpect() {
        return expect;
    }

    public void setExpect(double expect) {
        this.expect = expect;
    }
}
