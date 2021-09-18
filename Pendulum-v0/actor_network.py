import tflearn
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Network_Actor(object):
    
    def __init__(self, session, state_space, action_space, action_threshold, rateLearning, tau_factor, sizeBatch):
        self.session = session
        self.state_space = state_space
        self.action_space = action_space
        self.action_threshold = action_threshold
        self.rateLearning = rateLearning
        self.tau_factor = tau_factor
        self.sizeBatch = sizeBatch

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_network()

        self.network_parameters = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_network()

        self.target_parameters = tf.trainable_variables()[
                                     len(self.network_parameters):]

        # Updating target network with network weights
        self.update_parameters_target = \
            [self.target_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau_factor) +
                                                  tf.multiply(self.target_parameters[i], 1. - self.tau_factor))
             for i in range(len(self.target_parameters))]

       
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_space])

        # Combinining Gradients
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_parameters, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.sizeBatch), self.unnormalized_actor_gradients))

        # Optimization Opeartion
        self.optimize = tf.train.AdamOptimizer(self.rateLearning). \
            apply_gradients(zip(self.actor_gradients, self.network_parameters))

        self.numberOfVariables = len(
            self.network_parameters) + len(self.target_parameters)
    #Training
    def train(self, inputs, a_gradient):
        self.session.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
    #predicitng based on inputs
    def predict(self, inputs):
        return self.session.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })
    #predicitng target based on inputs
    def predict_target(self, inputs):
        return self.session.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })
    #updating target network
    def update_targetNetwork(self):
        self.session.run(self.update_parameters_target)

    def get_numberOfVariables(self):
        return self.numberOfVariables
    #Creating Actor Network
    
    def create_network(self):
        inputs = tflearn.input_data(shape=[None, self.state_space])
        network = tflearn.fully_connected(inputs, 400)
        network = tflearn.layers.normalization.batch_normalization(network)
        network = tflearn.activations.relu(network)
        network = tflearn.fully_connected(network, 300)
        network = tflearn.layers.normalization.batch_normalization(network)
        network = tflearn.activations.relu(network)
       #We need the initialization for last layer of the Actor to be between -0.003 and 0.003 
       # as this prevents us from getting 1 or -1 output values in the initial stages,
       # which would squash our gradients to zero, as we use the tanh activation.
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        output = tflearn.fully_connected(
            network, self.action_space, activation='tanh', weights_init=w_init)
       
        outputScaled = tf.multiply(output, self.action_threshold)
        return inputs, output, outputScaled