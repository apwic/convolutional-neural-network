import numpy as np
import json
from layers.Convolution import ConvolutionLayer
from layers.Flatten import FlattenLayer
from layers.Dense import DenseLayer

class Sequential:
    def __init__(self) -> None:
        self.conv_layers = []
        self.dense_layers = []
        self.flatten = None
        self.total_layers = 0

        self.input: np.ndarray = None
        self.output: np.ndarray = None

        self.targets: np.ndarray = None
        self.batch_size: int = 1
        self.num_epochs: int = 1

    def printSummary(self):
        convo_count = 0
        dense_count = 0
        print("———————————————————————————————————————————————————————————————————————")
        print("{:<30} {:<30} {:<10}".format(
            'Layer (type) ', 'Output Shape', 'Param #'))
        print("=======================================================================")
        sum_parameter = 0

        # Convolution Layer
        for layer in self.conv_layers:
            if(convo_count == 0):
                    postfix = " (Convo2D)"
            else:
                postfix = "_" + str(convo_count) + " (Convo2D)"
            convo_count += 1

            layerTypes = 'conv2d' + postfix
            shape = layer.getOutputShape()
            weight = layer.getParamsCount()
            sum_parameter += weight
            print("{:<30} {:<30} {:<10}".format(
                layerTypes, str(shape), weight))
            print("———————————————————————————————————————————————————————————————————————")
            
        # Flatten Layer
        print("{:<30} {:<30} {:<10}".format('flatten (Flatten)', str(self.flatten.getOutputShape()), 0))
        print("———————————————————————————————————————————————————————————————————————")

        # Dense Layer
        for layer in self.dense_layers:
            if(dense_count == 0):
                postfix = " (Dense)"
            else:
                postfix = "_" + str(dense_count) + " (Dense)"
            dense_count += 1

            layerTypes = 'dense' + postfix
            shape = layer.getOutputShape()
            weight = layer.getParamsCount()
            sum_parameter += weight
            print("{:<30} {:<30} {:<10}".format(
                layerTypes, str(shape), weight))
            if (layer != self.dense_layers[len(self.dense_layers)-1]):
                print(
                    "———————————————————————————————————————————————————————————————————————")
            else:
                print(
                    "=======================================================================")

        trainable_parameter = sum_parameter
        non_trainable_parameter = sum_parameter - trainable_parameter

        print("Total Params: {}".format(sum_parameter))
        print("Trainable Params: {}".format(trainable_parameter))
        print("Non-trainable Params: {}".format(non_trainable_parameter))
        print()

    def saveModel(self, filename):
        file = open(f'./output/{filename}.json' , 'w')
        data = []

        for layer in self.conv_layers:
            data += layer.getData()

        data.append(self.flatten.getData())

        for layer in self.dense_layers:
            data.append(layer.getData())


        file.write(json.dumps(data, indent=4))
        file.close()
        print("MODEL SAVED")

    # TODO: set each layer from data read, now it is only fetching the data
    def loadModel(self, filename):
        file = open(f'./output/{filename}.json', 'r')

        data = json.load(file)
        file.close()

        print("MODEL LOADED")
        return  data

    def setInput(self, input: np.ndarray):
        self.input = input

    def setTargets(self, targets: np.ndarray):
        self.targets = targets

    def setBatchSize(self, batchSize: int):
        self.batch_size = batchSize

    def setNumEpochs(self, numEpochs: int):
        self.num_epochs = numEpochs

    def setlearning_rate(self, learning_rate: float):
        for conv_layer in self.conv_layers:
            conv_layer.setlearning_rate(learning_rate)
        for dense_layer in self.dense_layers:
            dense_layer.setlearning_rate(learning_rate)

    def addConvLayer(self, layer: ConvolutionLayer):
        self.conv_layers.append(layer)
        self.total_layers += 1

    def addDenseLayer(self, layer: DenseLayer):
        self.dense_layers.append(layer)
        self.total_layers += 1

    def forwardProp(self):
        curr_input = self.input
        for layer in self.conv_layers:
            layer.setInput(curr_input)
            layer.calculate()
            curr_input = layer.getOutput()

        self.flatten = FlattenLayer(len(curr_input[0]))
        self.flatten.setInput(curr_input)
        self.flatten.flatten()
        curr_input = self.flatten.output
        
        for layer in self.dense_layers:
            layer.setInput(curr_input)
            layer.forward()
            curr_input = layer.getOutput()
        
        self.output = curr_input
        print(f"OUTPUT LAYER\n--------\n\nOutput: {self.output}\n")

    def backwardProp(self):
        dL_dFlattenedOutput = self.targets
        output_layer = True
        for layer in reversed(self.dense_layers):
            if (output_layer):
                dL_dFlattenedOutput = layer.backprop_output(dL_dFlattenedOutput)
                output_layer = False
            else:
                dL_dFlattenedOutput = layer.backprop_hidden(dL_dFlattenedOutput)

        dL_dConvOutput = self.flatten.backprop(dL_dFlattenedOutput)

        for layer in reversed(self.conv_layers):
            dL_dConvOutput = layer.backprop(dL_dConvOutput)
        
    def create_mini_batches(self):
        # shuffle inputs and targets in unison
        indices = np.arange(self.input.shape[0])
        np.random.shuffle(indices)
        inputs_shuffled = self.input[indices]
        targets_shuffled = self.targets[indices]

        # split into mini-batches
        num_batches = self.input.shape[0] // self.batch_size
        input_batches = np.array_split(inputs_shuffled, num_batches)
        target_batches = np.array_split(targets_shuffled, num_batches)

        return input_batches, target_batches

    def train(self):
        j = 0
        for i in range(self.num_epochs):
            print(f"\n\nEPOCH KE-{i}")
            input_batches, target_batches = self.create_mini_batches()
            for batch_inputs, batch_targets in zip(input_batches, target_batches):
                print(f"------------------------\nBATCH KE-{j}\n------------------------")
                for input_sample, _ in zip(batch_inputs, batch_targets):
                    self.setInput(input_sample[np.newaxis, :])
                    self.forwardProp()
                    self.backwardProp()
                for dense_layers in self.dense_layers:
                    dense_layers.update_weights_and_biases()
                for conv_layers in self.conv_layers:
                    conv_layers.update_weights_and_biases()

                self.reset()

    def reset(self):
        for conv in self.conv_layers:
            conv.reset()

        self.flatten.reset()

        for dense in self.dense_layers:
            dense.reset()
