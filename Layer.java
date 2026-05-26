public class Layer {
    int neurons;
    String activationFunction;

    double[][] weights;
    double[] biases;

    double[][] X;
    double[][] Y;

    int sampleCount;

    Layer(int neurons, String activationFunction) {
        this.neurons = neurons;
        this.activationFunction = activationFunction;
    }

    Layer(int neurons) {
        this(neurons, "sigmoid");
    }

    public double[][] forward(double[][] X) {
        if (weights == null) {
            weights = new double[neurons][];

            // Initialize weights based on the number of
            // neurons of the previous layer.
            for (int row_i = 0; row_i < neurons; row_i++) {
                weights[row_i] = new double[X[0].length];

                for (int cell_i = 0; cell_i < weights[row_i].length; cell_i++) {
                    weights[row_i][cell_i] = 2 * Math.random() - 1;
                }

            }

            // Initialize biases.
            biases = new double[neurons];
            for (int b_i = 0; b_i < biases.length; b_i++) {
                biases[b_i] = 2 * Math.random() - 1;
            }
        }

        sampleCount = X.length;

        // Store incoming activations for later use
        // in auto differentiation during back propagation.
        this.X = X;

        double[][] Y = new double[sampleCount][];

        // Compute activations (Y) for every sample in the
        // batch.
        for (int sample_i = 0; sample_i < sampleCount; sample_i++) {

            double[] x = X[sample_i];
            double[] y = new double[neurons];

            for (int row_i = 0; row_i < neurons; row_i++) {
                double[] row = weights[row_i];

                for (int i = 0; i < row.length; i++) {
                    y[row_i] += row[i] * x[i];
                }

                y[row_i] += biases[row_i];

                if (activationFunction.equals("sigmoid")) {
                    y[row_i] = sigmoid(y[row_i]);
                } else if (activationFunction.equals("relu")) {
                    y[row_i] = relu(y[row_i]);
                } else {
                    // do nothing
                }
            }

            Y[sample_i] = y;
        }

        this.Y = Y;
        return Y;
    }

    public double[][] backward(double[][] deltas, double stepsize) {
        double[][] backwardDeltas = new double[sampleCount][];

        // Compute deltas to backpropagate based on current weights
        // and biases.
        for (int sample_i = 0; sample_i < sampleCount; sample_i++) {

            double[] x = X[sample_i];
            double[] y = Y[sample_i];

            double[] delta = deltas[sample_i];
            for (int i = 0; i < neurons; i++) {

                if (activationFunction.equals("sigmoid")) {
                    delta[i] *= sigmoidDerivative(y[i]);
                } else if (activationFunction.equals("relu")) {
                    delta[i] *= reluDerivative(y[i]);
                } else {
                    // do nothing
                }

            }

            double[] backwardDelta = new double[x.length];
            for (int row_i = 0; row_i < neurons; row_i++) {
                for (int x_i = 0; x_i < x.length; x_i++) {
                    backwardDelta[x_i] += delta[row_i] * weights[row_i][x_i];
                }
            }
            backwardDeltas[sample_i] = backwardDelta;
        }

        // Update weights and biases.
        for (int sample_i = 0; sample_i < sampleCount; sample_i++) {

            double[] x = X[sample_i];

            double[] delta = deltas[sample_i];

            for (int row_i = 0; row_i < neurons; row_i++) {
                for (int x_i = 0; x_i < x.length; x_i++) {
                    weights[row_i][x_i] -= delta[row_i] * x[x_i] * stepsize;
                }
            }

            for (int b_i = 0; b_i < neurons; b_i++) {
                biases[b_i] -= delta[b_i] * stepsize;
            }
        }

        return backwardDeltas;
    }

    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double sigmoidDerivative(double y) {
        return y * (1 - y);
    }

    public static double relu(double x) {
        if (x > 0) {
            return x;
        } else {
            return 0;
        }
    }

    public static double reluDerivative(double y) {
        if (y > 0) {
            return 1;
        } else {
            return 0;
        }
    }
}
