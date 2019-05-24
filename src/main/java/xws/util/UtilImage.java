package xws.util;

import xws.neuron.Tensor;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

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


}
