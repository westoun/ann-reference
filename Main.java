public class Main {

    public static void main(String[] args) {

        // Data set for the XOR problem.
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

        // Network definition
        Network net = new Network(new int[] {
                X[0].length,
                10,
                Y_hat[0].length
        });

        // Hyper parameters
        double learningRate = 1;
        int printEvery = 10000;
        int maxEpochs = 100000;

        // Training loop
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
                Utils.print("\nEpoch " + epoch + ":");
                Utils.print("\t MSE: ", error);
                Y = net.forward(X);
                for (int sample_i = 0; sample_i < X.length; sample_i++) {
                    Utils.print("x = ", X[sample_i]);
                    Utils.print("y = ", Y[sample_i]);
                }
            }
        }
    }
}