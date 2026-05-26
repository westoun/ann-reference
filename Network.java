public class Network {
    Layer[] layers;

    Network(int[] neurons) {
        this.layers = new Layer[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            layers[i] = new Layer(neurons[i], "sigmoid");
        }
    }

    public double[][] forward(double[][] X) {
        double[][] Y = X;
        for (Layer layer : layers) {
            Y = layer.forward(Y);
        }

        return Y;
    }

    public void backward(double[][] deltas, double stepsize) {
        for (int i = layers.length - 1; i >= 0; i--) {
            deltas = layers[i].backward(deltas, stepsize);
        }
    }
}
