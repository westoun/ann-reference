public class Utils {
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
}
