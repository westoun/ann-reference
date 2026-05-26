public class Main {

    public static void print(String prefix, double[] values) {
        System.out.println();
        System.out.print(prefix + "[");

        for (double value : values) {
            System.out.print((Math.round(value * 100) + 0.0) / 100 + ", ");
        }

        System.out.print("]");
    }

    public static void print(String prefix, double value) {
        System.out.println();
        System.out.print(prefix + value);
    }

    public static void print(String str) {
        System.out.println();
        System.out.print(str);
    }

    public static double[][] normalize(double[][] values) {
        int sampleCount = values.length;

        double[][] normalizedValues = new double[sampleCount][];

        for (int dim = 0; dim < values[0].length; dim++) {

            double min = values[0][dim];
            double max = values[0][dim];

            for (int sample_i = 1; sample_i < sampleCount; sample_i++) {

                if (values[sample_i][dim] < min) {
                    min = values[sample_i][dim];
                }

                if (values[sample_i][dim] > max) {
                    max = values[sample_i][dim];
                }
            }

            for (int sample_i = 0; sample_i < sampleCount; sample_i++) {
                if (normalizedValues[sample_i] == null) {
                    normalizedValues[sample_i] = new double[values[sample_i].length];
                }

                normalizedValues[sample_i][dim] = (normalizedValues[sample_i][dim] - min) / (max - min);
            }
        }

        return normalizedValues;
    }

    public static void main(String[] args) {

        double[][] X = new double[4][];
        X[0] = new double[] { 0.0, 0.0 };
        X[1] = new double[] { 0.0, 1.0 };
        X[2] = new double[] { 1.0, 0.0 };
        X[3] = new double[] { 1.0, 1.0 };

        double[][] Y_hat = new double[4][];
        Y_hat[0] = new double[] { 0.0 };
        Y_hat[1] = new double[] { 1.0 };
        Y_hat[2] = new double[] { 1.0 };
        Y_hat[3] = new double[] { 0.0 };

        // double[][] X = new double[4][];
        // X[0] = new double[]{3.0, -1.0, 7.0};
        // X[1] = new double[]{-2.0, -3.0, 4.0};
        // X[2] = new double[]{3.0, 2.0, 3.0};
        // X[3] = new double[]{1.0, 1.0, 1.0};

        // double[][] Y_hat = new double[4][];
        // Y_hat[0] = new double[]{17.0};
        // Y_hat[1] = new double[]{1.0};
        // Y_hat[2] = new double[]{10.0};
        // Y_hat[3] = new double[]{3.0};

        Network net = new Network(new int[] {
                X[0].length,
                10,
                // 10,
                Y_hat[0].length
            }
        );

        double learningRate = 1;
        int printEvery = 10000;
        int maxEpochs = 100000;

        for (int epoch = 1; epoch <= maxEpochs; epoch++) {

            double error = 0.0;

            double[][] Y = net.forward(X);

            double[][] deltas = new double[Y_hat.length][];

            for (int sample_i = 0; sample_i < X.length; sample_i++) {

                double[] y = Y[sample_i];
                double[] y_hat = Y_hat[sample_i];

                deltas[sample_i] = new double[y_hat.length];

                for (int y_i = 0; y_i < y.length; y_i++) {

                    error += 0.5 * Math.pow(y[y_i] - y_hat[y_i], 2) / X.length;
                    deltas[sample_i][y_i] = (y[y_i] - y_hat[y_i]) / X.length;
                }
            }

            net.backward(deltas, learningRate);

            if (epoch % printEvery == 0) {
                // learningRate *= 0.9;

                print("\nEpoch " + epoch + ":");
                // print("Output prediction: ", y);
                print("\t MSE: ", error);
                Y = net.forward(X);
                for (int sample_i = 0; sample_i < X.length; sample_i++) {
                    print("x = ", X[sample_i]);
                    print("y = ", Y[sample_i]);
                }
            }
        }
    }
}