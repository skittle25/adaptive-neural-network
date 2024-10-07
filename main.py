import tensorflow as tf
import numpy as np
import math
import random


class LIFNeuronLayer:
    def __init__(self, num_inputs, num_neurons, tau=20.0, v_th=0.0, v_reset=0.0, name=None):
        # LIF model parameters
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.tau = tau
        self.v_th = v_th
        self.v_reset = v_reset

        self.name = name
        self.connections = {}
        self.connection_creating_probability = 0
        self.input_data = tf.constant([])

        # Weights of input-neuron connections
        self.weights = tf.Variable(tf.random.uniform((num_inputs, num_neurons), minval=0, maxval=1), trainable=True)

        # Initial state of neurons (membrane potential and output)
        self.v = tf.zeros(num_neurons)
        self.spikes = tf.zeros(num_neurons)

    def __call__(self, inputs, dt=1.0):
        # Change weights size if it is needed
        if inputs.shape[0] > self.num_inputs:
            new_weights = tf.concat(
                [
                    self.weights, tf.random.normal(
                    (inputs.shape[0] - self.weights[0], self.num_neurons)
                )
                ], axis=0
            )
            self.weights = new_weights
            self.num_inputs = inputs.shape[0]
        elif inputs.shape[0] < self.num_inputs:
            self.weights = (self.weights[:inputs.shape[0], :])
            self.num_inputs = inputs.shape[0]

        # Transforming input data by weights
        weighted_inputs = tf.matmul(inputs, self.weights)

        # Membrane potential update
        dv = (weighted_inputs - self.v) / self.tau
        self.v = self.v + dv * dt

        # Spikes generation
        self.spikes = tf.cast(self.v >= self.v_th, tf.float32)

        # Resetting the membrane potential for neurons that generated a spike
        self.v = tf.where(self.spikes > 0, self.v_reset, self.v)

        return self.spikes

    def get_config(self):
        return self.name, self.connections


class Map:
    def __init__(self, map_side_size, num_dims):
        self.map = np.zeros(tuple([map_side_size] * num_dims), dtype=object)

        self.num_inputs = 0
        self.num_outputs = 0
        self.next_coordinates = []
        self.activity = np.zeros(tuple([map_side_size] * num_dims))
        self.similar_loss = 0
        self.previous_loss = 0

        self.num_dims = num_dims

        self.last_input = tuple([0] * num_dims)
        self.last_output = tuple([map_side_size] * num_dims)

    def predict(self, inputs):
        self.activity = np.zeros(self.activity.shape)

        for input_data in inputs:
            for index, map_object in np.ndenumerate(self.map):
                try:
                    name, connections = map_object.get_config()
                except AttributeError:
                    name = None
                    connections = None

                if name == 'input' and input_data is not None:
                    x = map_object(input_data)

                    self.activity[index] = tf.reduce_sum(x).numpy()

                    for coordinates, weight in connections.items():
                        indices = tuple(float(num) if '.' in num else int(num) for num in coordinates.split(','))
                        self.next_coordinates.append(indices)

                        self.map[indices].input_data.concat([self.map[indices].input_data, x*int(weight)])

        for index, map_object in np.ndenumerate(self.map):
            try:
                name, connections = map_object.get_config()
            except AttributeError:
                name = None
                connections = None

            if name in('output', 'cell'):
                if map_object.input_data != tf.constant([]):
                    x = map_object(map_object.input_data)

                    self.activity[index] = tf.reduce_sum(x).numpy()

                    for coordinates, weight in connections.items():
                        indices = tuple(float(num) if '.' in num else int(num) for num in coordinates.split(','))
                        self.next_coordinates.append(indices)

                        self.map[indices].input_data.concat([self.map[indices].input_data, x*int(weight)])

                self.map[index].input_data = tf.constant([])

        outputs = []
        for index, map_object in np.ndenumerate(self.map):
            try:
                name, connections = map_object.get_config()
            except AttributeError:
                name = None

            reversed_index = [self.map.shape[0] - x for x in index]
            if name == 'output':
                outputs.append(self.activity[tuple(reversed_index)])

        return outputs

    def adapt(self, input_units, outputs_units):
        # Change number of input_units if it is needed
        if self.num_inputs < input_units:
            can_continue = False
            to_place = input_units - self.num_inputs
            for index, map_object in np.ndenumerate(self.map):
                if index == self.last_input:
                    can_continue = True
                if all(element % 3 == 0 for element in index) and can_continue:
                    self.map[index] = LIFNeuronLayer(1, 1, name='input')
                    to_place -= 1
                if to_place <= 0:
                    break
                self.last_input = index
        elif self.num_inputs > input_units:
            to_remove = self.num_inputs - input_units
            for index, map_object in np.ndenumerate(self.map):
                reversed_index = [self.map.shape[0] - x for x in index]
                try:
                    name, connections = self.map[tuple(reversed_index)].get_config()
                except AttributeError:
                    name = None
                if name == 'input' and to_remove > 0:
                    self.map[tuple(reversed_index)] = 0
                    to_remove -= 1
                if to_remove == 0 and name == 'input':
                    self.last_input = tuple(reversed_index)

        self.num_inputs = input_units

        # Change number of outputs if it is needed
        if self.num_outputs < outputs_units:
            can_continue = False
            to_place = outputs_units - self.num_outputs
            for index, map_object in np.ndenumerate(self.map):
                reversed_index = [self.map.shape[0] - x for x in index]
                if tuple(reversed_index) == self.last_input:
                    can_continue = True
                if all(element % 3 == 0 for element in tuple(reversed_index)) & 3 == 0 and can_continue:
                    self.map[tuple(reversed_index)] = LIFNeuronLayer(1, 1, name='output')
                    to_place -= 1
                if to_place <= 0:
                    break
                self.last_input = index
        elif self.num_outputs > outputs_units:
            to_remove = self.num_outputs - outputs_units
            for index, map_object in np.ndenumerate(self.map):
                try:
                    name, connections = self.map[tuple(index)].get_config()
                except AttributeError:
                    name = None
                if name == 'output' and to_remove > 0:
                    self.map[tuple(index)] = 0
                    to_remove -= 1
                if to_remove == 0 and name == 'output':
                    self.last_input = index

        self.num_outputs = outputs_units

        for index, map_object in np.ndenumerate(self.map):
            if map_object != 0:
                name, connections = map_object.get_config()

                if self.activity[index] != 0:
                    number = random.random()
                    if number < map_object.connection_creating_probability:
                        coordinates = [random.randint(0, self.map.shape[0]) for _ in range(self.num_dims)]

                        coords = np.array(np.nonzero(self.map != 0)).T

                        for coord in coords:
                            # Calculate the Euclidean distance from the reference point
                            euclidean_distance = np.linalg.norm(coord - coordinates)
                            try:
                                object_name, object_coordinates = self.map[coord].get_config()
                            except AttributeError:
                                object_name = None

                            if euclidean_distance <= 3 and object_name in ('output', 'cell'):
                                coordinates_as_str = ','.join(map(str, coord))
                                self.map[index].conncetions[coordinates_as_str] = '1'

                self.map[index].connection_creating_probability = 1 - math.exp(
                    -0.1 * self.map[index].connection_creating_probability
                )

                for connection, weight in connections.items():
                    if int(weight) > 0.001:
                        self.map[index].connections[connection] = str(int(weight) * 1.1)
                    else:
                        self.map[index].connections.pop(connection)
                else:
                    self.map[index].connection_creating_probability = self.map[index].connection_creating_probability *\
                                                                      math.exp(-0.3 * 5)

                    for connection, weight in connections.items():
                        if int(weight) > 0.001:
                            self.map[index].connections[connection] = str(int(weight) / 1.1)
                        else:
                            self.map[index].connections.pop(connection)

    def learn(self, alpha, loss, beta=8, gamma=0.03):
        for index, value in np.ndenumerate(self.activity):
            if value > 0 and self.map[index] != 0 and alpha:
                self.map[index].weights = 1 - tf.math.exp(
                    -0.1 * self.map[index].weights
                )
            elif value > 0 and self.map[index] != 0 and not alpha:
                self.map[index].weights = self.map[index].weights * math.exp(-0.3 * 5)

            if abs(self.previous_loss - loss) <= gamma:
                self.similar_loss += 1
            else:
                self.similar_loss = 0
            self.previous_loss = loss

            if self.similar_loss >= beta:
                coordinates = [random.randint(0, self.map.shape[0]) for _ in range(self.num_dims)]
                while self.map[tuple(coordinates)] != 0:
                    coordinates = [random.randint(0, self.map.shape[0]) for _ in range(self.num_dims)]

                self.map[coordinates] = LIFNeuronLayer(1, 1, name='cell')
