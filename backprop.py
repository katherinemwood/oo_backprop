from random import uniform, shuffle, gauss
from math import e, log, exp
import csv
from logger import *

class BackpropConnection():
    """
    A class representing the connection between two neurons.
    """
    def __init__(self, sender, recipient, eta, mu):
        """
        A connection needs to know its "sender" neuron, i.e. the neuron in the previous
        level, and its "recipient" neuron, i.e. the neuron in the next layer. A connection is
        initialized with a random Gaussian-distributed weight, and needs a learning rate and
        momentum argument to calculate the gradient.

        These connections batch-update, so they accumulate gradients from each training sample
        and then batch-update their weights once per epoch, after all samples have been seen.
        """
        self.sender = sender
        self.recipient = recipient
        self.weight = gauss(0, .01)
        self.delta_heap = 0.0
        self.delta = 0.0
        self.prev_delta = 0.0
        self.eta = eta
        self.mu = mu

    def accumulate(self):
        """
        Calculate the change in weight based on the error of the recipient neuron and the 
        activation of the sending neuron, weighted by learning rate and factoring in
        the previous gradient weighted by momentum.
        """
        self.delta = -self.eta * self.recipient.error * self.sender.activation + (self.mu * self.prev_delta)
        self.prev_delta = self.delta
        self.delta_heap += self.delta

    def update(self, samples):
        """
        Apply the accumulated weights, averaged over the number of samples in the 
        epoch, and reset the heap.
        """
        self.weight += self.delta_heap/samples
        self.delta_heap = 0.0

class BackpropUnit():
    """
    A class representing a neuron. These are ReLU-activated units.
    Neurons have a list of incoming and outgoing connections, each of
    which is an instance of a BackpropConnection class.

    A unit can be given a label and a desired activation if it is in
    the output layer.
    """
    def __init__(self, index, learning_rate, momentum):
        self.index = index
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.input = 0.0
        self.activation = 0.0
        self.error = 0.0
        self.inputs = []
        self.outputs = []
        self.desired_activation = None
        self.label = None

    def add_connection(self, recipient):
        """
        Instantiate a connection with this neuron as the sender and the provided neuron as the
        recipient. Add the connection to this unit's outgoing list, and the recipient unit's
        incoming list.
        """
        new_connection = BackpropConnection(sender=self, recipient=recipient, eta=self.learning_rate, mu=self.momentum)
        self.outputs.append(new_connection)
        recipient.inputs.append(new_connection)

    def update_input(self):
        """
        A unit's input is the sum of the activations of all of the units feeding up to it, weighed by the
        strength of the connection between them.
        """
        self.input = sum([connection.weight * connection.sender.activation for connection in self.inputs])

    def update_activation(self):
        """
        The activation function transforms the raw input. If the unit is
        in a hidden layer (i.e. its label attribute is None), it uses ReLU
        activation. Otherwise, it uses linear activation.
        """
        if self.label is None:
            self.activation = self.input if self.input > 0 else 0.0
        else:
            self.activation = self.input

    def update_error(self):
        """
        Update the error of a unit. If it has no incoming connections, such as a bias unit
        or a unit in the input layer, it has no error to update and is skipped. 

        If it is in a hidden layer, then the error is the sum of the error of each recipient neuron times 
        the connection weight between this neuron and the recipient across all connections, unless
        this neuron was not activated, in which case its error is 0 because it did not contribute
        to the activation of its recipients.

        If the neuron is in the output layer, its error is calculated relative to what the activation
        should have been for the class. If it should not have activated at all, the error is simply the
        amount of activation. If it should have activated, the error is the amount by which the unit
        under- or over-shot its desired activation level.
        """
        if not self.inputs:
            pass
        elif self.desired_activation is None:
            self.error = (sum([connection.recipient.error * connection.weight for connection in self.outputs])
                          * int(self.activation > 0.0))
        elif self.desired_activation == 1:
            self.error = self.activation - 1
        elif self.desired_activation == 0:
            self.error = self.activation

    def assign_label(self, lab):
        """
        Assign a human-understandable class label to an output unit.
        """
        self.label = lab

    def set_activation(self, activation):
        """
        Set the unit's activation to the provided activation.
        """
        self.activation = activation

    def set_desired_activation(self, desired_activation):
        """
        If the unit is in the output layer, set its desired activation.
        """
        self.desired_activation = desired_activation

    def predict(self):
        """
        If the unit is in the output layer, get its label and activation.
        """
        if not self.label is None:
            return (self.label, self.activation)

class BackpropLayer():
    """
    A class representing a layer in the network. A layer
    oversees the updating and organization of its units.
    """
    def __init__(self, units, learning_rate, momentum):
        """
        Initialize the layer to the specified size. Each unit is
        assigned the correct value for learning rate and momentum.
        """
        self.n_units = units
        self.units = [BackpropUnit(i, learning_rate, momentum) for i in range(units)]

    def update_units(self):
        """
        Update the input coming into each unit and apply the unit's
        activation function to transform the input.
        """
        [unit.update_input() for unit in self.units]
        [unit.update_activation() for unit in self.units]

    def update_error(self):
        """
        Update each unit's error.
        """
        [unit.update_error() for unit in self.units]

    def accumulate_weights(self):
        """
        Accumulate the gradient for the error on the given sample for each outgoing
        connection for each unit in the layer.
        """
        [connection.accumulate() for unit in self.units for connection in unit.outputs]

    def update_weights(self, samples):
        """
        Update all the outgoing connection weights for each unit in the layer. This
        happens at the end of each training epoch.
        """
        [connection.update(samples) for unit in self.units for connection in unit.outputs]

    def set_desired_activation(self, label):
        """
        If this is an output layer, set the expected activation. If the unit has the
        correct class label, it should have activation value 1, otherwise it should
        have value 0.
        """
        for unit in self.units:
            if unit.label == label:
                unit.set_desired_activation(1.0)
            else:
                unit.set_desired_activation(0.0)

    def softmax(self):
        """
        Transform the activations of the output layer units with
        softmax to get a probability distribution. Reassign the
        activations to be these probabilities.
        """
        f = [unit.activation for unit in self.units]
        all_z = sum([exp(a - max(f)) for a in f])
        [unit.set_activation(exp(unit.activation - max(f))/all_z) for unit in self.units]

    def set_activation(self, pattern):
        """
        Set the activation for the input units according to the sample.
        """
        [self.units[i].set_activation(pattern[i]) for i in range(len(pattern))]

    def label(self, labels):
        """
        If this is an output layer, assign class labels to each output unit.
        """
        [self.units[i].assign_label(labels[i]) for i in range(len(labels))]

    def predict(self):
        """
        Get a dictionary of class labels and activation values for the
        output layer.
        """
        prediction = ({unit.label: unit.activation for unit in self.units})
        return prediction

class BackpropNetwork():
    """
    A class representing a fully-connected network, with methods to train and
    evaluate its performance.
    """
    def __init__(self, layers, output_labels, learning_rate, momentum, logger, epochs=5000):
        """
        A network is initialized with an n-tuple of integers, where each integer is
        the number of neurons in that layer. The 0th and n-1th entries are assumed
        to be the input and output layers, respectively, with all layers in between
        assumed to be hidden layers.

        The network also takes an instance of the NetworkLogger class to record data.
        """
        self.layers = [BackpropLayer(u, learning_rate, momentum) for u in layers]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.training_epochs = 0
        self.has_bias_units = False
        self.total_epochs = epochs
        self.error_criterion = .01
        self.errors = []
        self.error = 1.0
        self.layers[-1].label(output_labels)
        self.logger = logger

    def connect(self):
        """
        Fully connect the layers to each other.
        """
        [sender_unit.add_connection(recipient_unit) for l in range(len(self.layers) - 1)
         for recipient_unit in self.layers[l + 1].units
         for sender_unit in self.layers[l].units]

    def add_bias_units(self):
        """
        If desired, add a bias unit to each layer (excluding the output layer), connect them
        up, and mark the has_bias_units attribute as True (used only for logging).
        """
        for l in range(len(self.layers[:-1])):
            bias_unit = BackpropUnit(self.layers[l].n_units, self.learning_rate, self.momentum)
            bias_unit.set_activation(1.0)
            [bias_unit.add_connection(unit) for unit in self.layers[l + 1].units]
            self.layers[l].units.append(bias_unit)
        self.has_bias_units = True

    def propagate_activation(self):
        """
        Pass the activation from the input layer forward through the network.
        """
        for layer in self.layers[1::]:
            layer.update_units()

    def backpropagate_error(self):
        """
        Backprogate the gradient through the network.
        """
        for layer in self.layers[::-1]:
            layer.update_error()

    def accumulate_weights(self):
        """
        Accumulate the gradient step for all connections in the network.
        """
        [layer.accumulate_weights() for layer in self.layers]

    def update_weights(self, samples):
        """
        Batch-update the weightl for all connections in the network.
        """
        [layer.update_weights(samples) for layer in self.layers]

    def evaluate(self):
        """
        Get the performance at the output layer for the network.
        """
        return ([-log(unit.activation) for unit in self.layers[-1].units if unit.desired_activation == 1])

    def train(self, patterns):
        """
        Train the network.

        This network uses a batch-training scheme. On each epoch, it sees all
        samples in a random order, accumulating weights, and then only updating
        the weights after all samples have been seen.
        """
        self.training_epochs = 0
        self.error = 1.0
        while self.training_epochs < self.total_epochs:
            self.error = 0
            self.errors = []
            shuffle(patterns)
            for pattern in patterns:
                self.layers[0].set_activation(pattern[0]) #feed in the input
                self.layers[-1].set_desired_activation(pattern[1]) #set the correct result at the output layer
                self.propagate_activation() #forward pass the activation through the network
                self.layers[-1].softmax() #convert the activation at the output layer with softmax
                self.errors += self.evaluate() #calculate the error at the output level
                self.backpropagate_error() #pass the gradient back through the network according to the loss function
                self.accumulate_weights() #update the weights on the connections between units
            self.error = float(sum(self.errors))/len(patterns) #get the average error for the epoch
            self.update_weights(len(patterns)) #batch-update the training weights
            self.training_epochs += 1
            if self.training_epochs % 100 == 0:
                print("iteration: %d; loss: %f" % (self.training_epochs, self.error))
        print("training complete")

    def test(self, pattern, data_split):
        """
        Test the network performance on an unseen pattern and write the results
        to the log file.
        """
        self.layers[0].set_activation(pattern[0])
        self.layers[-1].set_desired_activation(pattern[1])
        self.propagate_activation()
        self.layers[-1].softmax()
        predictions = self.layers[-1].predict()
        self.logger.log([self.learning_rate, self.momentum, self.training_epochs, self.has_bias_units] +
                       [pattern[1]] +
                        [data_split] +
                        [';'.join(str(lab) + ':' + str(predictions[lab]) for lab in sorted(predictions.keys()))])
        self.logger.write_data()