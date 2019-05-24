package xws.neuron.panel;

import xws.neuron.Tensor;
import xws.util.UtilImage;
import xws.test.FullNetWorkTest;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.Map;

/**
 * 展示mnist图片
 */
public class MnistPanel extends JPanel {


    protected void paintComponent(Graphics g2) {

        Map<Double, double[]> map = FullNetWorkTest.loadMNIST();

        Map<Double, double[]> mapNew = new HashMap<>();

        super.paintComponent(g2);
        Graphics2D g2d = (Graphics2D) g2;


        BufferedImage image;

        int x = 0;


        for (Map.Entry<Double, double[]> entry : map.entrySet()) {
            double[] array = entry.getValue();
            Tensor tensor = new Tensor();
            tensor.setDepth(1);
            tensor.setHeight(28);
            tensor.setWidth(28);
            tensor.createArray();
            tensor.setArray(array);

            image = UtilImage.tensorToImage(tensor);

            Image tmp = image.getScaledInstance(100, 100, BufferedImage.SCALE_SMOOTH);
            BufferedImage te = UtilImage.imageToBufferedImage(tmp);

            tensor = UtilImage.imageToTensor(te);
            mapNew.put(entry.getKey(), tensor.getArray());

            image = UtilImage.tensorToImage(tensor);
            g2d.drawImage(image, x, 0, null);
            x = x + 103;

        }

        String fileName = "/Users/xws/Desktop/xws/MNIST1.txt";

//        String json = JSON.toJSONString(mapNew);
//        UtilFile.writeFile(json, fileName);

    }


    /**
     * @param args
     */
    public static void main(String[] args) {
        JFrame jf = new JFrame();
        jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jf.getContentPane().add(new MnistPanel());
        jf.setPreferredSize(new Dimension(1000, 1000));
        jf.pack();
        jf.setVisible(true);

    }

}

