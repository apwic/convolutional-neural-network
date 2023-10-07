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

        self.input_batch: np.ndarray = None
        self.target_batch: np.ndarray = None

        self.test_input: np.ndarray = None
        self.test_target: np.ndarray = None

        self.validation_input: np.ndarray = None
        self.validation_target: np.ndarray = None

        self.targets: np.ndarray = None
        self.batch_size: int = 1
        self.num_epochs: int = 1

    def print_separator(self, title=None, width=79):
        print("=" * width)
        if title:
            print(title.center(width))
        print("=" * width)

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

    def loadModel(self, filename):
        file = open(f'./output/{filename}.json', 'r')

        data = json.load(file)
        file.close()

        # clear the current model's layers
        self.conv_layers = []
        self.dense_layers = []

        # loop through the loaded data and reconstruct the layers
        for layer_data in data:
            layer_type = layer_data["type"]
            
            # Add convolutional layer
            if layer_type == "conv2d":
                params = layer_data["params"]
                convLayer = ConvolutionLayer(
                    num_of_input=params["num_of_input"],
                    input_size=params["input_size"],
                    filter_size_conv=params["filter_size_conv"],
                    number_of_filter_conv=params["number_of_filter_conv"],
                    filter_size_pool=params["filter_size_pool"],
                    stride_size_conv=params["stride_size_conv"],
                    stride_size_pool=params["stride_size_pool"],
                    padding_size=params["padding_size"],
                    mode=params["mode"],
                    learning_rate=params["learning_rate"],
                )
                convLayer.setWeights(np.array(params["kernel"]))
                convLayer.convolutionStage.setBiases(np.array(params["bias"]))
                self.addConvLayer(convLayer)
            
            # Add dense layer
            elif layer_type == "dense":
                params = layer_data["params"]
                denseLayer = DenseLayer(
                    units=params["units"],
                    activation_function=params["activation_function"],
                    learning_rate=params["learning_rate"],
                )
                denseLayer.setWeight(np.array(params["kernel"]))
                denseLayer.setBiases(np.array(params["biases"]))
                self.addDenseLayer(denseLayer)
            
            # Handle other layer types (max_pooling2d, flatten) if needed

        print("MODEL LOADED")
        self.printSummary()

    def setInput(self, input: np.ndarray):
        self.input = input

    def setTargets(self, targets: np.ndarray):
        self.targets = targets

    def setTest(self, input: np.ndarray, target: np.ndarray):
        self.test_input = input
        self.test_target = target

    def setValidation(self, input: np.ndarray, target: np.ndarray):
        self.validation_input = input
        self.validation_target = target

    def setBatchSize(self, batchSize: int):
        self.batch_size = batchSize

    def setNumEpochs(self, numEpochs: int):
        self.num_epochs = numEpochs

    def setLearningRate(self, learning_rate: float):
        for conv_layer in self.conv_layers:
            conv_layer.setLearningRate(learning_rate)
        for dense_layer in self.dense_layers:
            dense_layer.setLearningRate(learning_rate)

    def binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15  # to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def addConvLayer(self, layer: ConvolutionLayer):
        self.conv_layers.append(layer)
        self.total_layers += 1

        num_of_input, input_size, _ = layer.poolingStage.getOutputShape()
        self.flatten = FlattenLayer(num_of_input, input_size)

    def addDenseLayer(self, layer: DenseLayer):
        if (len(self.dense_layers) == 0):
            layer.setInputSize(self.flatten.getOutputShape())
        else:
            layer.setInputSize(self.dense_layers[-1].getOutputShape()[1])

        self.dense_layers.append(layer)
        self.total_layers += 1

    def forwardProp(self, input):
        curr_input = input
        for layer in self.conv_layers:
            layer.setInput(curr_input)
            layer.calculate()
            curr_input = layer.getOutput()

        self.flatten.setInput(curr_input)
        self.flatten.flatten()
        curr_input = self.flatten.output
        
        for layer in self.dense_layers:
            layer.setInput(curr_input)
            layer.forward()
            curr_input = layer.getOutput()
        
        self.output = curr_input

    def backwardProp(self, target):
        dL_dFlattenedOutput = target
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
        indices = np.arange(self.input_batch.shape[0])
        np.random.shuffle(indices)
        inputs_shuffled = self.input_batch[indices]
        targets_shuffled = self.targets[indices]

        # split into mini-batches
        num_batches = self.input_batch.shape[0] // self.batch_size
        input_batches = np.array_split(inputs_shuffled, num_batches)
        target_batches = np.array_split(targets_shuffled, num_batches)

        return input_batches, target_batches
    
    def create_k_folds(self, k=10):
        indices = np.arange(self.input.shape[0])
        np.random.shuffle(indices)
        
        inputs_shuffled = self.input[indices]
        targets_shuffled = self.targets[indices]

        input_folds = np.array_split(inputs_shuffled, k)
        target_folds = np.array_split(targets_shuffled, k)
        
        return input_folds, target_folds
    
    def train_and_test(self, k=10):
        self.train(k)

        # Now test on the 10% test set
        correct_predictions = 0
        for input_sample, target_sample in zip(self.validation_input, self.validation_target):
            self.forwardProp(input_sample)
            prediction = 1 if self.output >= 0.5 else 0
            if prediction == target_sample:
                correct_predictions += 1
        accuracy = correct_predictions / len(self.validation_input)
        print(f"\nTest Accuracy on the 10% Test Set: {accuracy * 100:.2f}%")

    def train(self, k=10):
        input_folds, target_folds = self.create_k_folds(k)

        # Store validation performances for each fold
        fold_performances = []

        for fold_idx in range(k):
            self.print_separator(f"Training on Fold {fold_idx + 1}/{k}")
            validation_input = input_folds[fold_idx]
            validation_target = target_folds[fold_idx]

            self.input_batch = np.concatenate([input_folds[i] for i in range(k) if i != fold_idx])
            self.target_batch = np.concatenate([target_folds[i] for i in range(k) if i != fold_idx])
            
            for i in range(self.num_epochs):
                print(f"\nEPOCH KE-{i+1}")
                input_batches, target_batches = self.create_mini_batches()
                j = 0
                for batch_inputs, batch_targets in zip(input_batches, target_batches):
                    for input_sample, target_sample in zip(batch_inputs, batch_targets):
                        self.forwardProp(input_sample)
                        self.backwardProp(target_sample)
                        self.resetOutput()
                    for dense_layers in self.dense_layers:
                        dense_layers.update_weights_and_biases()
                    for conv_layers in self.conv_layers:
                        conv_layers.update_weights_and_biases()

                    self.resetAll()
                    j += 1
                    
            self.resetAll()
            correct_predictions = 0
            for input_sample, target_sample in zip(validation_input, validation_target):
                self.forwardProp(input_sample)
                prediction = 1 if self.output >= 0.5 else 0
                if prediction == target_sample:
                    correct_predictions += 1
                self.resetAll()
            accuracy = correct_predictions / len(validation_input)
            fold_performances.append(accuracy)
            print(f"\nFold {fold_idx + 1} Validation Accuracy: {accuracy * 100:.2f}%\n")
            self.resetAll()

        avg_performance = np.mean(fold_performances)
        self.print_separator(f"Average Validation Accuracy: {avg_performance * 100:.2f}%")
        self.resetAll()

    def test(self):
        i = 1
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        correct_predictions = 0
    
        for input_sample, target_sample in zip(self.test_input, self.test_target):
            print(f"------------------------PENGUJIAN KE-{i}------------------------")
            self.forwardProp(input_sample)
            
            prediction = 1 if self.output >= 0.5 else 0
            
            if prediction == target_sample and target_sample == 0 :
                tn += 1
            elif prediction == target_sample and target_sample == 1 :
                tp += 1
            elif prediction != target_sample and target_sample == 0 :
                fp += 1
            else :
                fn += 1

            correct_predictions = tp + tn
                
            print(f"\nHasil Prediksi: {self.output}\tKelas Target: {target_sample}")
            print(f"Correct Predictions so far: {correct_predictions} / {i}\n")
            self.resetAll()
            i += 1
        
        accuracy = correct_predictions / len(self.test_input)
        print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

        confusion_matrix = [[tn, fp], [fn, tp]]

        print("Confusion Matrix:")
        print(confusion_matrix)

    def resetOutput(self):
        for conv in self.conv_layers:
            conv.resetOutput()

        self.flatten.resetOutput()

        for dense in self.dense_layers:
            dense.resetOutput()

    def resetAll(self):
        for conv in self.conv_layers:
            conv.resetAll()

        self.flatten.resetOutput()

        for dense in self.dense_layers:
            dense.resetAll()
