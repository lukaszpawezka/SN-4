import numpy as np
from enum import Enum
import matplotlib.pyplot as plt

def get_output_array(result, size):
    arr = np.zeros((size, 1))
    arr[result] = 1
    return arr


def cost_function(a, y):
    return a - y

class Optimalisation(Enum):
    MOMENTUM = "momentum"
    NESTROV_MOMENTUM = "nestrov_momentum"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    ADAM = "adam"


class Initialisation(Enum):
    RANDOM = "random"
    XAVIER = "xavier"
    HE = "he"

class ConvolutionLayerWithMaxPooling:
    def __init__(self, input_size=(28, 28), kernel=(3, 3), step=1, padding=1, convolutions=32, max_pooling_size=(2, 2), max_pooling_step = 2):
        self.input_size = input_size
        self.kernel = kernel
        self.step = step
        self.padding = padding
        self.convolutions = convolutions
        self.max_pooling_size = max_pooling_size
        self.max_pooling_step = max_pooling_step

        self.kernel_biases = [0 for k in range(0, convolutions)]
        self.kernels = [np.random.normal(size=(kernel[0], kernel[1]), loc=0, scale=np.sqrt(4 / (784 + 9))) for k in range(0, convolutions)]
        self.feature_matrices = [np.zeros((input_size[0], input_size[1])) for k in range(0, convolutions)]

        self.max_pooling_layers = [np.zeros((int(input_size[0]/max_pooling_step), int(input_size[1]/max_pooling_step))) for k in range(0, convolutions)]

    def do_convolutions(self, input_image):

        input_image = input_image.reshape(28, 28)
        image_with_padding = np.zeros((self.input_size[0]+self.padding*2, self.input_size[1]+self.padding*2))
        image_with_padding[self.padding:self.padding+self.input_size[0], self.padding:self.padding+self.input_size[1]] = input_image
        for kernel, feature_matrix in zip(self.kernels, self.feature_matrices):
            for x in range(0, np.shape(input_image)[0], self.step):
                for y in range(0, np.shape(input_image)[1], self.step):

                    segment = image_with_padding[
                              x-int(self.kernel[0]/2)+self.padding:x+int(self.kernel[0]/2)+self.padding+1,
                              y-int(self.kernel[1]/2)+self.padding:y+int(self.kernel[1]/2)+self.padding+1]

                    # print(np.sum(segment * kernel))
                    # print(np.shape(feature_matrix))
                    feature_matrix[x][y] = self.relu(np.sum(segment * kernel))
            # plt.imshow(feature_matrix)
            # plt.show()
        for max_pooling_layer, feature_matrix in zip(self.max_pooling_layers, self.feature_matrices):
            for x in range(0, np.shape(feature_matrix)[0], self.max_pooling_step):
                for y in range(0, np.shape(feature_matrix)[1], self.max_pooling_step):

                    # print()

                    max_pooling_layer[int(x/self.max_pooling_step)][int(y/self.max_pooling_step)] = np.max(feature_matrix[x:x+self.max_pooling_step, y:y+self.max_pooling_step])
            # plt.imshow(max_pooling_layer)
            # plt.show()

        # output_flatten_layer = self.max_pooling_layers[0].flatten()

        flatten_max_pooling_layers = [layer.flatten() for layer in self.max_pooling_layers]
        joined_max_pooling_layer = np.concatenate(flatten_max_pooling_layers, axis=0)
        # print(np.shape(joined_max_pooling_layer))
        # for max_pooling_layer in self.max_pooling_layers[1:]:
        #      output_flatten_layer = np.concatenate(max_pooling_layer, max_pooling_layer.flatten())

        return joined_max_pooling_layer

    def relu(self, x):
        return np.maximum(x, 0)

    def relu_derivative(self, x):
        return 1. * (x > 0)

class NeuronNetwork:
    def __init__(self,
                 layers,
                 init_range_scale,
                 activation_function,
                 momentum=0,
                 adadelta_y=0.9,
                 adam_b1=0.9,
                 adam_b2=0.999,
                 negative=True,
                 soft_max_output=False,
                 optimalisation: Optimalisation = Optimalisation.MOMENTUM,
                 initialisation: Initialisation = Initialisation.RANDOM,
                 convolution: ConvolutionLayerWithMaxPooling = ConvolutionLayerWithMaxPooling()
                 ):
        self.adadelta_y = adadelta_y
        self.adam_b1 = adam_b1
        self.adam_b2 = adam_b2
        self.adam_b1_pow = adam_b1
        self.adam_b2_pow = adam_b2
        self.convolution = convolution

        self.opt = optimalisation
        self.soft_max_output = soft_max_output
        self.function, self.prime = activation_function
        self.layers_details = layers
        self.momentum = momentum
        self.init_range_scale = init_range_scale
        self.z_values = []
        self.activations = []
        self.old_err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
        self.old_err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]
        self.init = False
        self.err_b_square = [np.zeros((l[1], 1)) for l in self.layers_details]
        self.err_w_square = [np.zeros((l[1], l[0])) for l in self.layers_details]
        self.err_b_sum = [np.zeros((l[1], 1)) for l in self.layers_details]
        self.err_w_sum = [np.zeros((l[1], l[0])) for l in self.layers_details]

        if initialisation == Initialisation.RANDOM:
            if negative:
                self.wights = [(np.random.random((layer[1], layer[0])) - 0.5) * 2 * self.init_range_scale for layer in
                           self.layers_details]
                self.biases = [(np.random.random((layer[1], 1)) - 0.5) * 2 * self.init_range_scale for layer in
                           self.layers_details]
            else:
                self.wights = [(np.random.random((layer[1], layer[0]))) * self.init_range_scale for layer in
                           self.layers_details]
                self.biases = [(np.random.random((layer[1], 1))) * self.init_range_scale for layer in
                           self.layers_details]
        elif initialisation == Initialisation.XAVIER:
            self.wights = [np.random.normal(size=(layer[1], layer[0]), loc=0, scale=np.sqrt(2 / (layer[0] + layer[1]))) for layer in
                           self.layers_details]
            self.biases = [np.zeros((layer[1], 1)) for layer in self.layers_details]
        elif initialisation == Initialisation.HE:
            self.wights = [np.random.normal(size=(layer[1], layer[0]), loc=0, scale=np.sqrt(4 / (layer[0] + layer[1])))
                           for layer in
                           self.layers_details]
            self.biases = [np.zeros((layer[1], 1)) for layer in self.layers_details]

    def forward_for_learning_fast(self, input):
        self.activations = [input.reshape((-1, 1))]
        self.z_values = []
        for w, b in zip(self.wights, self.biases):
            self.z_values.append(w @ self.activations[-1] + b)
            self.activations.append(self.function(self.z_values[-1]))

        # softmax
        if self.soft_max_output:
            exp_values = np.exp(self.activations[-1])
            output = exp_values / np.sum(exp_values, axis=0, keepdims=True)
            self.activations[-1] = output
        return self.activations[-1]

    def forward_for_testing(self, input):
        activation = input.reshape((-1, 1))

        for w, b in zip(self.wights, self.biases):
            z_value = w @ activation + b
            activation = self.function(z_value)

        # softmax
        if self.soft_max_output:
            exp_values = np.exp(activation)
            output = exp_values / np.sum(exp_values, axis=0, keepdims=True)
            return output
        return activation

    def backpropagate(self, input, result):
        convolution_output = self.convolution.do_convolutions(input)
        self.forward_for_learning_fast(convolution_output)

        y = get_output_array(result, self.layers_details[-1][1])
        cost = cost_function(self.activations[-1], y)

        delta = cost * self.prime(self.z_values[-1])
        err_b = [delta]
        err_w = [delta @ self.activations[-2].T]

        for layer_index in range(2, len(self.wights) + 1):
            z_value = self.z_values[-layer_index]
            prime = self.prime(z_value)
            delta = (self.wights[-layer_index + 1].T @ delta) * prime
            err_b.append(delta)
            err_w.append(delta @ self.activations[-1 - layer_index].T)
        err_b.reverse()
        err_w.reverse()
        return err_b, err_w

    def _train_mini_batch(self, mini_batch, mini_batch_results, eta):
        if self.opt == Optimalisation.MOMENTUM:
            self.momentum_train(mini_batch, mini_batch_results, eta)
        elif self.opt == Optimalisation.NESTROV_MOMENTUM:
            self.nestors_momentum_train(mini_batch, mini_batch_results, eta)
        elif self.opt == Optimalisation.ADAGRAD:
            self.adagrad_train(mini_batch, mini_batch_results, eta)
        elif self.opt == Optimalisation.ADADELTA:
            self.adadelta_train(mini_batch, mini_batch_results, eta)
        elif self.opt == Optimalisation.ADAM:
            self.adam_train(mini_batch, mini_batch_results, eta)

    def momentum_train(self,  mini_batch, mini_batch_results, eta):
        err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
        err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]
        for input_example, result in zip(mini_batch, mini_batch_results):
            err_b_new, err_w_new = self.backpropagate(input_example, result)
            err_b = [ob + nb for ob, nb in zip(err_b, err_b_new)]
            err_w = [ow + nw for ow, nw in zip(err_w, err_w_new)]

        size = len(mini_batch)
        err_b = [eta * b / size + ob * self.momentum for b, ob in zip(err_b, self.old_err_b)]
        err_w = [eta * w / size + ow * self.momentum for w, ow in zip(err_w, self.old_err_w)]

        self.wights = [w - nw for w, nw in zip(self.wights, err_w)]
        self.biases = [b - nb for b, nb in zip(self.biases, err_b)]

        self.old_err_w = err_w
        self.old_err_b = err_b

    def nestors_momentum_train(self, mini_batch, mini_batch_results, eta):
        err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
        err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]
        for input_example, result in zip(mini_batch, mini_batch_results):
            err_b_new, err_w_new = self.backpropagate(input_example, result)
            err_b = [ob + nb for ob, nb in zip(err_b, err_b_new)]
            err_w = [ow + nw for ow, nw in zip(err_w, err_w_new)]

        size = len(mini_batch)
        err_b = [eta * (b+self.momentum*ob) / size + ob * self.momentum for b, ob, sb in zip(err_b, self.old_err_b, self.biases)]
        err_w = [eta * (w+self.momentum*ow) / size + ow * self.momentum for w, ow, sw in zip(err_w, self.old_err_w, self.wights)]

        self.wights = [w - nw for w, nw in zip(self.wights, err_w)]
        self.biases = [b - nb for b, nb in zip(self.biases, err_b)]

        self.old_err_w = err_w
        self.old_err_b = err_b

    def adagrad_train(self, mini_batch, mini_batch_results, eta):
        err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
        err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]

        # err_b_adagrad = [np.zeros((l[1], 1)) for l in self.layers_details]
        # err_w_adagrad = [np.zeros((l[1], l[0])) for l in self.layers_details]

        for input_example, result in zip(mini_batch, mini_batch_results):
            err_b_new, err_w_new = self.backpropagate(input_example, result)
            err_b = [ob + nb for ob, nb in zip(err_b, err_b_new)]
            err_w = [ow + nw for ow, nw in zip(err_w, err_w_new)]
            self.err_b_square = [ob + np.square(nb) for ob, nb in zip(self.err_b_square, err_b_new)]
            self.err_w_square = [ow + np.square(nw) for ow, nw in zip(self.err_w_square, err_w_new)]

        # for l in self.wights:
        #     print(np.shape(l))
        #
        # for l in err_w_adagrad:
        #     print(np.shape(l))

        size = len(mini_batch)
        err_b = [eta * b / size for b in err_b]
        err_w = [eta * w / size for w in err_w]
        # err_b_adagrad = [b / size for b in err_b_adagrad]
        # err_w_adagrad = [w / size for w in err_w_adagrad]
        b_learning_rate = [1 / np.sqrt(b_adagrad+np.finfo(float).eps) for b_adagrad in self.err_b_square]
        w_learning_rate = [1 / np.sqrt(w_adagrad+np.finfo(float).eps) for w_adagrad in self.err_w_square]

        # for l in [nw * wl for w, nw, wl in zip(self.wights, err_w, b_learning_rate)]:
        #     print(np.shape(l))
        self.wights = [w - nw * wl for w, nw, wl in zip(self.wights, err_w, w_learning_rate)]
        self.biases = [b - nb * bl for b, nb, bl in zip(self.biases, err_b, b_learning_rate)]

    def adadelta_train(self, mini_batch, mini_batch_results, eta):
        err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
        err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]

        err_b_square = [np.zeros((l[1], 1)) for l in self.layers_details]
        err_w_square = [np.zeros((l[1], l[0])) for l in self.layers_details]

        for input_example, result in zip(mini_batch, mini_batch_results):
            err_b_new, err_w_new = self.backpropagate(input_example, result)
            err_b = [ob + nb for ob, nb in zip(err_b, err_b_new)]
            err_w = [ow + nw for ow, nw in zip(err_w, err_w_new)]
            err_b_square = [ob+np.square(nb) for ob, nb in zip(err_b_square, err_b_new)]
            err_w_square = [ow+np.square(nw) for ow, nw in zip(err_w_square, err_w_new)]
        if self.init:
            self.err_b_square = [self.adadelta_y * ob + (1 - self.adadelta_y) * nb for ob, nb in zip(self.err_b_square, err_b_square)]
            self.err_w_square = [self.adadelta_y * ow + (1 - self.adadelta_y) * nw for ow, nw in zip(self.err_w_square, err_w_square)]
        else:
            self.init = True
            self.err_b_square = err_b_square
            self.err_w_square = err_w_square
        # self.err_b_adagrad = [self.adadelta_y * ob + (1-self.adadelta_y)*nb for ob, nb in zip(self.err_b_adagrad, err_b_adagrad)]
        # self.err_w_adagrad = [self.adadelta_y * ow + (1-self.adadelta_y)*nw for ow, nw in zip(self.err_w_adagrad, err_w_adagrad)]
        # for l in self.wights:
        #     print(np.shape(l))
        #
        # for l in err_w_adagrad:
        #     print(np.shape(l))

        size = len(mini_batch)
        err_b = [eta * b / size for b in err_b]
        err_w = [eta * w / size for w in err_w]
        # err_b_adagrad = [b / size for b in err_b_adagrad]
        # err_w_adagrad = [w / size for w in err_w_adagrad]
        b_learning_rate = [1 / np.sqrt(b_adagrad+np.finfo(float).eps) for b_adagrad in self.err_b_square]
        w_learning_rate = [1 / np.sqrt(w_adagrad+np.finfo(float).eps) for w_adagrad in self.err_w_square]

        # for l in [nw * wl for w, nw, wl in zip(self.wights, err_w, b_learning_rate)]:
        #     print(np.shape(l))
        self.wights = [w - nw * wl for w, nw, wl in zip(self.wights, err_w, w_learning_rate)]
        self.biases = [b - nb * bl for b, nb, bl in zip(self.biases, err_b, b_learning_rate)]


    def adam_train(self, mini_batch, mini_batch_results, eta):
        # err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
        # err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]

        err_b_square = [np.zeros((l[1], 1)) for l in self.layers_details]
        err_w_square = [np.zeros((l[1], l[0])) for l in self.layers_details]
        err_b_sum = [np.zeros((l[1], 1)) for l in self.layers_details]
        err_w_sum = [np.zeros((l[1], l[0])) for l in self.layers_details]

        for input_example, result in zip(mini_batch, mini_batch_results):
            err_b_new, err_w_new = self.backpropagate(input_example, result)
            # err_b = [ob + nb for ob, nb in zip(err_b, err_b_new)]
            # err_w = [ow + nw for ow, nw in zip(err_w, err_w_new)]
            err_b_square = [ob+np.square(nb) for ob, nb in zip(err_b_square, err_b_new)]
            err_w_square = [ow+np.square(nw) for ow, nw in zip(err_w_square, err_w_new)]
            err_b_sum = [ob + nb for ob, nb in zip(err_b_sum, err_b_new)]
            err_w_sum = [ow + nw for ow, nw in zip(err_w_sum, err_w_new)]
        if self.init:
            self.err_b_square = [self.adam_b2 * ob + (1 - self.adam_b2) * nb for ob, nb in zip(self.err_b_square, err_b_square)]
            self.err_w_square = [self.adam_b2 * ow + (1 - self.adam_b2) * nw for ow, nw in zip(self.err_w_square, err_w_square)]
            self.err_b_sum = [self.adam_b1 * ob + (1 - self.adam_b1) * nb for ob, nb in zip(self.err_b_sum, err_b_sum)]
            self.err_w_sum = [self.adam_b1 * ow + (1 - self.adam_b1) * nw for ow, nw in zip(self.err_w_sum, err_w_sum)]
        else:
            self.init = True
            self.err_b_square = err_b_square
            self.err_w_square = err_w_square
            self.err_b_sum = err_b_sum
            self.err_w_sum = err_w_sum


        size = len(mini_batch)

        mb = [m / (1 - self.adam_b1_pow) for m in self.err_b_sum]
        mw = [w / (1 - self.adam_b1_pow) for w in self.err_w_sum]
        self.adam_b1_pow *= self.adam_b1
        vb = [m / (1 - self.adam_b2_pow) for m in self.err_b_square]
        vw = [w / (1 - self.adam_b2_pow) for w in self.err_w_square]
        self.adam_b2_pow *= self.adam_b2

        self.wights = [w - eta/(np.sqrt(v)+np.finfo(float).eps)*m/size for w, m, v in zip(self.wights, mw, vw)]
        self.biases = [b - eta/(np.sqrt(v)+np.finfo(float).eps)*m/size for b, m, v in zip(self.biases, mb, vb)]



    def train(self, input, values, epochs, mini_batch_size, eta):
        for epoch in range(epochs):
            print("Epoch: ", (epoch + 1), "/", epochs)
            state = np.random.get_state()
            np.random.shuffle(input)
            np.random.set_state(state)
            np.random.shuffle(values)

            mini_batches = [input[b:b + mini_batch_size] for b in range(0, int(np.shape(input)[0] / mini_batch_size))]
            mini_batches_results = [values[b:b + mini_batch_size] for b in
                                    range(0, int(np.shape(values)[0] / mini_batch_size))]
            num = 0
            size = len(mini_batches)
            for batch, result in zip(mini_batches, mini_batches_results):
                self._train_mini_batch(batch, result, eta)
                self.old_err_b = [np.zeros((l[1], 1)) for l in self.layers_details]
                self.old_err_w = [np.zeros((l[1], l[0])) for l in self.layers_details]
                num += 1
                print(f"{num / size * 100}%")

        pass

    def test(self, input, values):
        print("Testowanie")
        positive = 0
        iter = 0
        for example, value in zip(input, values):
            convolution_output = self.convolution.do_convolutions(example)
            result = np.argmax(self.forward_for_testing(convolution_output))
            if result == value:
                positive += 1
            iter += 1
        print(f"{positive}/{np.shape(values)[0]} = {positive / np.shape(values)[0] * 100}%")
        return positive / np.shape(values)[0] * 100

    def predict(self, input):
        return np.argmax(self.forward_for_testing(input))
