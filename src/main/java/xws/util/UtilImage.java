package xws.util;

import xws.neuron.Tensor;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 * 图像工具类
 * Created by xws on 2019/3/22.
 */
public class UtilImage {

    //mnist BufferImage生成数组
    public static Tensor imageToTensor(BufferedImage image) {
        Tensor tensor = new Tensor();
        tensor.setDepth(1);
        tensor.setHeight(image.getHeight());
        tensor.setWidth(image.getWidth());
        tensor.createArray();

        for (int h = 0; h < tensor.getHeight(); h++) {
            for (int w = 0; w < tensor.getWidth(); w++) {
                int val = image.getRGB(w, h);
                tensor.set(h, w, val);
            }
        }

        return tensor;
    }


    public static Tensor image3ToTensor(BufferedImage image) {
        Tensor tensor = new Tensor();
        tensor.setDepth(3);
        tensor.setHeight(image.getHeight());
        tensor.setWidth(image.getWidth());
        tensor.createArray();

        for (int h = 0; h < tensor.getHeight(); h++) {
            for (int w = 0; w < tensor.getWidth(); w++) {

                int pixel = image.getRGB(w, h);
                tensor.set(0, h, w, (pixel & 0xff0000) >> 16);
                tensor.set(1, h, w, (pixel & 0xff00) >> 8);
                tensor.set(2, h, w, (pixel & 0xff));

            }
        }

        return tensor;
    }

    //mnist tensor转换成BufferImage
    public static BufferedImage tensorToImage(Tensor tensor) {
        BufferedImage image = new BufferedImage(tensor.getWidth(), tensor.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        for (int h = 0; h < tensor.getHeight(); h++) {
            for (int w = 0; w < tensor.getWidth(); w++) {
                int val = (int) tensor.get(h, w);
//                double r = val * 0.3;
//                double g = val * 0.59;
//                double b = val * 0.1;
                int value = val + (val << 8) + (val << 16);
//                int value = (r * 38 + g * 75 + val * 15) >> 7;
                image.setRGB(w, h, value);
            }
        }
        return image;
    }

    //3通道数据
    public static BufferedImage tensorToImage3(Tensor tensor) {
        BufferedImage image = new BufferedImage(tensor.getWidth(), tensor.getHeight(), BufferedImage.TYPE_3BYTE_BGR);
        for (int h = 0; h < tensor.getHeight(); h++) {
            for (int w = 0; w < tensor.getWidth(); w++) {
                int r = (int) tensor.get(0, h, w);
                int g = (int) tensor.get(1, h, w);
                int b = (int) tensor.get(2, h, w);
                int value = (r << 16) + (g << 8) + b;
                image.setRGB(w, h, value);
            }
        }
        return image;
    }

    //imageToBufferedImage
    public static BufferedImage imageToBufferedImage(Image image) {
        if (image instanceof BufferedImage) {
            return (BufferedImage) image;
        }
        // 加载所有像素
        image = new ImageIcon(image).getImage();
        BufferedImage bimage = null;
        GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
        try {
            int transparency = Transparency.OPAQUE;
            // 创建buffer图像
            GraphicsDevice gs = ge.getDefaultScreenDevice();
            GraphicsConfiguration gc = gs.getDefaultConfiguration();
            bimage = gc.createCompatibleImage(
                    image.getWidth(null), image.getHeight(null), transparency);
        } catch (HeadlessException e) {
            e.printStackTrace();
        }
        if (bimage == null) {
            int type = BufferedImage.TYPE_INT_RGB;
            bimage = new BufferedImage(image.getWidth(null), image.getHeight(null), type);
        }
        // 复制
        Graphics g = bimage.createGraphics();
        // 赋值
        g.drawImage(image, 0, 0, null);
        g.dispose();
        return bimage;
    }

    // 读取文件
    public static BufferedImage readImage(String strPath) {
        try {
            File file = new File(strPath);
            Image src = ImageIO.read(file); //构造Image对象
            int width = src.getWidth(null); //得到源图宽
            int height = src.getHeight(null); //得到源图长
            BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR);
            bufferedImage.getGraphics().drawImage(src, 0, 0, width, height, null);
            return bufferedImage;
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    // 绘制矩形
    public static void drawRect(Graphics g, int x, int y, int width, int height) {
        Graphics2D g2d = (Graphics2D) g.create();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setColor(Color.GRAY);
        // 1. 绘制一个矩形: 起点(30, 20), 宽80, 高100
        g2d.drawRect(x, y, width, height);
        g2d.dispose();
    }

    // 绘制点
    public static void drawPoint(Graphics g, int x, int y) {
        Graphics2D g2d = (Graphics2D) g.create();
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setColor(Color.RED);
        g2d.fillOval(x, y, 2, 2);
        g2d.dispose();
    }

    //加载数据

}
