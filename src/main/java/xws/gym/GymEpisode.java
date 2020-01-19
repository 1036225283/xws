package xws.gym;

import java.util.List;

/**
 * episode
 * Created by xws on 2019/10/19.
 */
public class GymEpisode {

    private double reward;//total reward
    private List<GymStorePolicy> policies;

    public double getReward() {
        return reward;
    }

    public void setReward(double reward) {
        this.reward = reward;
    }

    public List<GymStorePolicy> getPolicies() {
        return policies;
    }

    public void setPolicies(List<GymStorePolicy> policies) {
        this.policies = policies;
    }
}
