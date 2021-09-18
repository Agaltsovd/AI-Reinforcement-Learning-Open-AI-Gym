import tflearn
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Network_Critic(object):
    
    def __init__(self, sess, state_space, action_space, rateLearning, tau_factor, gamma, num_actor_vars):
        self.sess = sess
        self.state_space = state_space
        self.action_space = action_space
        self.rateLearning = rateLearning
        self.tau_factor = tau_factor
        self.gamma = gamma

        # Creating  critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_parameters = tf.trainable_variables()[num_actor_vars:]

        # Creating Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_parameters = tf.trainable_variables()[(len(self.network_parameters) + num_actor_vars):]

        # Updating target network with network weight
        self.update_parameters_target = \
            [self.target_parameters[i].assign(tf.multiply(self.network_parameters[i], self.tau_factor) \
                                                  + tf.multiply(self.target_parameters[i], 1. - self.tau_factor))
             for i in range(len(self.target_parameters))]

        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Loss and Optimization 
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.rateLearning).minimize(self.loss)

        # Gradient of the network 
        self.action_grads = tf.gradients(self.out, self.action)
        
    #Training    
    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })
    #predicitng based on inputs 
    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })
    #Predicting target
    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })
    #gradients for actions
    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })
    #updating target network
    def update_targetNetwork(self):
        self.sess.run(self.update_parameters_target)
    #creating critic network
    def create_critic_network(self):

        inputs = tflearn.input_data(shape=[None, self.state_space])
        action = tflearn.input_data(shape=[None, self.action_space])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

      
        #getting weight and bias by using two temporary layers
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        #Q(s,a)
        #We need the initialization for last layer of the Actor to be between -0.003 and 0.003 
        # as this prevents us from getting 1 or -1 output values in the initial stages,
        # which would squash our gradients to zero, as we use the tanh activation.
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

