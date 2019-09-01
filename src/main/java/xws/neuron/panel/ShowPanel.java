package xws.neuron.panel;


import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * 展示前端传递过来的图片
 */
public class ShowPanel extends JPanel {

    private BufferedImage image;

    protected void paintComponent(Graphics g2) {
        super.paintComponent(g2);
        Graphics2D g2d = (Graphics2D) g2;
        g2d.drawImage(image, 0, 0, null);
    }

    public BufferedImage getImage() {
        return image;
    }

    public void setImage(BufferedImage image) {
        this.image = image;
    }
}

