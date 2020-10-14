package xws.neuron.layer.resnet;

import xws.neuron.Tensor;
import xws.neuron.layer.FullLayer;
import xws.neuron.layer.Layer;

import java.util.ArrayList;
import java.util.List;


/**
 * resNet
 * Created by xws on 2020/10/14.
 */
public class ResnetFullLayer extends Layer {


    List<FullLayer> list = new ArrayList<>();


    public ResnetFullLayer() {
    }

    public ResnetFullLayer(int layers, String name, String activationType, int num, double lambda) {
        super(ResnetFullLayer.class.getSimpleName());
        for (int i = 0; i < layers; i++) {
            FullLayer full = new FullLayer(name, activationType, num, lambda);
            list.add(full);
        }
    }


    @Override
    public Tensor forward(Tensor tensor) {

        Tensor tensorInput = tensor;
        for (int i = 0; i < list.size(); i++) {
            FullLayer fullLayer = list.get(i);
            tensor = fullLayer.forward(tensor);
        }

        return tensor;
    }

    @Override
    public Tensor backPropagation(Tensor tensor) {

        Tensor tensorError = tensor;
        for (int i = list.size() - 1; i >= 0; i--) {
            FullLayer fullLayer = list.get(i);
            tensorError = fullLayer.backPropagation(tensorError);
        }

        tensorError.add(tensor);
        return tensorError;
    }

    public List<FullLayer> getList() {
        return list;
    }

    public void setList(List<FullLayer> list) {
        this.list = list;
    }
}
