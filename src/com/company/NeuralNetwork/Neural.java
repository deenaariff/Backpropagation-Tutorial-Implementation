package com.company.NeuralNetwork;

import com.company.NeuralNetwork.Neuron.Neuron;

import java.util.*;

public class Neural {

    private List<List<Neuron>> network;
    private List<Neuron> inputs;

    /**
     * Constructor
     *
     * @param num_inputs
     * @param num_hidden
     * @param num_outputs
     */
    public Neural(int num_inputs, int num_hidden, int num_outputs) {

        Random r = new Random();
        for (int i = 0; i < num_inputs; i++) {
            inputs.add(new Neuron(r.nextDouble()));
        }

        List<Neuron> layer = new ArrayList<>();

        // Generate the Hidden Layers
        for (int i = 0; i < num_hidden; i++) {
            layer.add(new Neuron(num_inputs, true));
        }

        // Construct the network thus far
        this.network.add(layer);

        // Construct the output Layer

        layer = new ArrayList<>();

        // Generate the Output Layer
        for (int i = 0; i < num_outputs; i++) {
            layer.add(new Neuron(num_hidden, true));
        }

        this.network.add(layer);

    }

    /**
     * Return the neural network
     *
     * @return
     */
    private List<List<Neuron>> getNetwork() {
        return this.network;
    }


    /**
     * Perform Forward Propagation of the Neural Network
     *
     * @return List<Neuron> the calculated output layer
     */
    public List<Neuron> forward_propagation() {
        List<Neuron> input = this.inputs;
        for (List<Neuron> layer : this.network) {
            List<Neuron> tmp = new ArrayList<>();
            for (Neuron neuron : layer) {
                Double activation = Neuron.activate(neuron, input);
                neuron.setOut(Neuron.sigmoid(activation));
                tmp.add(neuron);
            }
            input = tmp;
        }
        return input;
    }

    /**
     * Perform Backward propogation to calculate all the deltas per node
     *
     * @param expected
     */
    public void backward_propagate_error(List<Double> expected) {
        List<Neuron> prev_layer = null;
        for(int count = this.network.size()-1; count >= 0; count--) {
            List<Neuron> layer = this.network.get(count);
            List<Double> errors =  new ArrayList<>();
            if(count != this.network.size()-1) {
                for(int j = 0; j < layer.size(); j++) {
                    double error = 0.0;
                    for (Neuron neuron : prev_layer) {
                        List<Double> weights = neuron.getWeights();
                        error += weights.get(j) * neuron.getDelta();
                    }
                    errors.add(error);
                }
            } else {
                for(int j = 0; j < layer.size(); j++) {
                    errors.add(expected.get(j) - layer.get(j).getOut());
                }
            }
            for(int j = 0; j < layer.size(); j++) {
                Neuron neuron = layer.get(j);
                neuron.setDelta(errors.get(j) * Neuron.transfer_derivative(neuron.getOut()));
            }
            count += 1;
            prev_layer = layer;
        }
    }

    public void update_weights() {
        for(int i = 0; i < this.network.size(); i++) {
            if i != 0:
            else:
                for j in range()
        }
    }

    /**
     *
     * @param args
     */
    public static void main(String[] args) {
        Neural nn = new Neural(2, 1, 2);
        for (List<Neuron> layer : nn.getNetwork()) {
            layer.toString();
        }
    }

}
