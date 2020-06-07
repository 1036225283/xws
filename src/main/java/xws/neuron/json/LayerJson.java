package xws.neuron.json;

public class LayerJson {

    private String type;
    private String name;
    //full rnn
    private int num;
    private String activation;
    // convolution and maxPool
    private int height;
    private int width;
    private int strideX;
    private int strideY;

    public static void main(String[] args) {
        System.out.println(LayerType.FullLayer.name());
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public int getNum() {
        return num;
    }

    public void setNum(int num) {
        this.num = num;
    }

    public String getActivation() {
        return activation;
    }

    public void setActivation(String activation) {
        this.activation = activation;
    }

    public int getHeight() {
        return height;
    }

    public void setHeight(int height) {
        this.height = height;
    }

    public int getWidth() {
        return width;
    }

    public void setWidth(int width) {
        this.width = width;
    }

    public int getStrideX() {
        return strideX;
    }

    public void setStrideX(int strideX) {
        this.strideX = strideX;
    }

    public int getStrideY() {
        return strideY;
    }

    public void setStrideY(int strideY) {
        this.strideY = strideY;
    }
}
