import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import com.sun.tools.javac.Main;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

private static final String IMAGE_NAME = "vase.jpg";
private static final String MODEL_NAME = "yolo26n.onnx";
private static final int IMG_SIZE = 640;
private static final float CONF_THRESHOLD = 0.8f;

private static final String[] classNames = {
        "person", "bicycle", "car", "motorcycle", "airplane",
        "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
};

void main () throws Exception {
    var imagePath = Main.class.getClassLoader().getResource( IMAGE_NAME ).getPath();
    var modelPath = Main.class.getClassLoader().getResource( MODEL_NAME ).getPath();

    var originalImage = ImageIO.read(new File(imagePath));

    var image = resize(originalImage, IMG_SIZE, IMG_SIZE);

    float [] inputData = toCHWArray(image);

    try (OrtEnvironment env = OrtEnvironment.getEnvironment();
         OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions())) {

        String inputName = session.getInputNames().iterator().next();
        // The shape the model expects: [batch=1, channels=3, height=640, width=640]
        long[] shape = {1, 3, IMG_SIZE, IMG_SIZE};
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputData), shape);
        OrtSession.Result result = session.run(Collections.singletonMap(inputName, inputTensor));

        float[][][] output = (float[][][]) result.get(0).getValue(); // shape=[1, 300, 6]
        float[][] detections = output[0]; // drop the batch dimension -> 300 rows of 6 numbers

        drawDetectionsOnImage(image, detections);
    }

    show(image);
}

private static void show(BufferedImage img) {
    SwingUtilities.invokeLater(() -> {
        JFrame f = new JFrame();
        f.add(new JLabel(new ImageIcon(img)));
        f.pack();
        f.setVisible(true);
    });
}

private static BufferedImage resize(BufferedImage src, int width, int height) {
    var dst = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    var g = dst.createGraphics();
    g.drawImage(src, 0, 0, width, height, null);
    g.dispose();
    return dst;
}

private static float[] toCHWArray(BufferedImage img) {
    int w = img.getWidth();
    int h = img.getHeight();
    var raster = img.getRaster();

    float[] chw = new float[3 * h * w];
    int i = 0;

    for (int channel = 0; channel < 3; channel++) {    // 0 = red, 1 = green, 2 = blue
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                chw[i++] = raster.getSample(x, y, channel) / 255f;
            }
        }
    }
    return chw;
}

private static void drawDetectionsOnImage(BufferedImage image, float [][] detections) {
    Graphics2D g = image.createGraphics();
    g.setColor(Color.WHITE);
    g.setStroke(new BasicStroke(2));
    g.setFont(new Font("Arial", Font.BOLD, 16));

    // Draw every row whose confidence clears our threshold.
    for (float[] row : detections) {
        float confidence = row[4];
        int classId = Math.round(row[5]);
        String className = classNames[classId];
        if (confidence < CONF_THRESHOLD) continue;

        int x1 = Math.round(row[0]);
        int y1 = Math.round(row[1]);
        int x2 = Math.round(row[2]);
        int y2 = Math.round(row[3]);

        int width = x2 - x1;
        int height = y2 - y1;

        // Bounding box
        g.drawRect(x1, y1, width, height);
        // Label + Confidence in %
        String label = className + " " + String.format("%.02f",(confidence * 100)) + "%";
        g.drawString(label, x1, Math.max(20, y1 - 5));
    }
}
