package com.company.NeuralNetwork.Neuron;

import java.util.List;
import java.util.Random;

public class Neuron {

    private List<Double> weights;
    private Double out;
    private Double delta;

    public Neuron(Double out) {
        this.out = out;
    }

    public Neuron(int num_weights, boolean bias) {
        Random r = new Random();
        for(int i = 0; i < num_weights; i++) {
            this.weights.add(r.nextDouble());
        }
        if(bias) {
            this.weights.add(1.0);
        }
    }

    /**
     * Return the activation for a given node in the network
     *
     * @param neuron
     * @param inputs
     * @return
     */
    public static Double activate(Neuron neuron, List<Neuron> inputs) {
        List<Double> weights = neuron.getWeights();

        if(weights.size() != inputs.size() + 1) {
            throw new RuntimeException();
        }
        Double activation = weights.get(weights.size()-1);
        for (int i = 0; i <= weights.size(); i++) {
            activation += weights.get(i) * inputs.get(i).getOut();
        }
        return activation;
    }

    public static Double transfer_derivative(Double output) {
        return output * (1.0 - output);
    }

    /**
     * Sigmoid Activation Function
     *
     * @param activation
     * @return
     */
    public static Double sigmoid(Double activation) {
        return 1.0 / (1.0 + Math.exp(-1.0 * activation));
    }

    public List<Double> getWeights() {
        return weights;
    }

    public void setWeights(List<Double> weights) {
        this.weights = weights;
    }

    public Double getOut() {
        return out;
    }

    public void setOut(Double out) {
        this.out = out;
    }

    public Double getDelta() {
        return delta;
    }

    public void setDelta(Double delta) {
        this.delta = delta;
    }



}
