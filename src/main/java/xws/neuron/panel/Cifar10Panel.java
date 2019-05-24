package xws.neuron.panel;

import xws.neuron.Tensor;
import xws.util.Cifar10;
import xws.util.UtilCifar10;
import xws.util.UtilImage;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;

/**
 * Cifar10图片展示
 */
public class Cifar10Panel extends JPanel {

    private List<Cifar10> list;

    protected void paintComponent(Graphics g2) {

        super.paintComponent(g2);
        Graphics2D g2d = (Graphics2D) g2;

        if (list == null) {
            list = UtilCifar10.testData();
        }


        //创建
        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_3BYTE_BGR);


        //展示 10 个图片
        for (int ih = 0; ih < 50; ih++) {
            for (int jw = 0; jw < 50; jw++) {
                Cifar10 cifar10 = list.get(ih * 50 + jw);
                Tensor rgb = cifar10.getRgb();
                image = UtilImage.tensorToImage3(rgb);
//                Image tmp = image.getScaledInstance(50, 50, BufferedImage.SCALE_SMOOTH);
//                BufferedImage te = toBufferedImage(tmp);
                g2d.drawImage(image, jw * 32, ih * 32, null);
            }

        }
    }


    /**
     * @param args
     */
    public static void main(String[] args) {
        JFrame jf = new JFrame();
        jf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jf.getContentPane().add(new Cifar10Panel());
        jf.setPreferredSize(new Dimension(1000, 1000));
        jf.pack();
        jf.setVisible(true);

    }

}

