package Structures;


import Algorithms.GeneticAlgorithm;
import Lenses.MicroLens;
import Utilities.Utilities;
import com.amd.aparapi.Kernel;

import java.util.ArrayList;
import java.util.concurrent.Callable;

/**
 *
 * @author <Jared Nelsen>
 */

public class NeuralNetwork {

    private static final Utilities u = new Utilities();
    
    public final ArrayList<Layer> layers = new ArrayList<>();

    //Step function firing
    public final double firingThreshold = .5;
    public final int firingValue = 1;
    
    //types
    public enum NetworkOutputType{
        BINARY_APPROXIMATION,
        INTEGER_RGB_VALUE
    };
    
    private NetworkOutputType type = NetworkOutputType.BINARY_APPROXIMATION;

    public enum ThresholdFunctionID{
        SIGMOID_FUNCTION,
        RGBA_SCALED_SIGMOID_FUNCTION,
        STEP_FUNCTION,
        RGBA_PIECEWISE_INTERVAL_FUNCTION
    };
    
    public NeuralNetwork(){
        
    }
    
    public NeuralNetwork(ArrayList<Integer> layers, ThresholdFunctionID hiddenLayerThresholdID, ThresholdFunctionID outputThresholdID, NetworkOutputType networkType){
        this.type = networkType;
        
        int layerIndex = 0;
        boolean atLastLayer = false;
        for(Integer i : layers){
            Layer l = new Layer(hiddenLayerThresholdID);

            if(layerIndex == layers.size() - 1)
                atLastLayer = true;

            if(atLastLayer)
                l = new Layer(outputThresholdID);

            for (int j = 0; j < i; j++)
                l.neurons.add(l.createNewNeuron());

            //Creating a bias neuron in the last layer is necessary for these calculations to work
            //The memory increase is negligible
            Layer.Neuron bias = l.createNewNeuron();
            bias.isBias = true;
            l.neurons.add(bias);

            this.layers.add(l);
            layerIndex++;
        }

        //This is essentially a free parameter.
        //It would be wise to test different values here as I have seen research results
        //where high initial variance in weights is important
        //Of course, it would be wise to map these values to the weight limits in NetworkParameterEncoding
        this.fullyConnect(0, 0);
    }
    
    public NeuralNetwork copyNetwork(){
        NeuralNetwork copy = new NeuralNetwork();
        
        for (Layer l : this.layers) {
            Layer c = createNewLayer(l.thresholdFunctionID);
            for (Layer.Neuron n : l.neurons)
                c.neurons.add(l.createNewNeuron());
            copy.layers.add(c);
        }
        copy.fullyConnect(0, 0);
        
        return copy;
    }

//Network Structure
//*************************************************************************************************************

    public class Layer {

        public ArrayList<Neuron> neurons;

        public ThresholdFunctionID thresholdFunctionID = ThresholdFunctionID.STEP_FUNCTION;       

        public Layer(ThresholdFunctionID id) {
            neurons = new ArrayList<>();
            thresholdFunctionID = id;
        }

        public void loadActivations(float[] activations){
            int neuronIndex = 0;
            for(Neuron n : neurons){
                n.activation = activations[neuronIndex];
                neuronIndex++;
            }
        }

        public float[] getActivations(){
            float[] activations = new float[neurons.size()];
            for (int i = 0; i < neurons.size(); i++)
                activations[i] = neurons.get(i).activation;
            return activations;
        }

        public class Neuron {

            public ArrayList<NeuronConnection> incomingNeuronConnections;
            public float activation;
            public boolean isBias = false;

            public Neuron() {
                incomingNeuronConnections = new ArrayList<>();
            }

            public Neuron copyNeuron() {
                Neuron n = new Neuron();
                for (Layer.NeuronConnection conn : this.incomingNeuronConnections)
                    n.incomingNeuronConnections.add(conn);
                n.activation = this.activation;
                n.isBias = this.isBias;
                return n;
            }

        }

        public Neuron createNewNeuron() {
            return new Neuron();
        }

        public class NeuronConnection implements Cloneable {

            public Neuron sourceNeuron;
            public Neuron targetNeuron;
            public float weight;

            public NeuronConnection() {
                
            }

            public NeuronConnection(Neuron source, Neuron target, float wt) {
                sourceNeuron = source;
                targetNeuron = target;
                weight = wt;
            }

            public NeuronConnection copyNeuronConnection() {
                NeuronConnection n = new NeuronConnection();
                n.sourceNeuron = this.sourceNeuron;
                n.targetNeuron = this.targetNeuron;
                n.weight = this.weight;
                return n;
            }

        }

        public NeuronConnection createNewNeuronConnection(Neuron source, Neuron target, float weight) {
            return new NeuronConnection(source, target, weight);
        }

    }

    public Layer createNewLayer(ThresholdFunctionID id) {
        return new Layer(id);
    }

//Connection functions
//*************************************************************************************************************
    public void fullyConnect(float minConnectionWeight, float maxConnectionWeight) {

        //create the incoming connections for each neuron
        Object[] arr = layers.toArray();
        for (int i = 0; i < arr.length; i++) {
            //set this layer
            Layer thisLayer = (Layer) arr[i];

            //set the next layer checking to see if we are not at the last layer
            Layer nextLayer;
            try {
                nextLayer = (Layer) arr[i + 1];
            } catch (Exception e) {
                nextLayer = null;
            }

            //if we are looking at the last layer
            if (nextLayer != null) {
                Object[] neuronsInThisLayer = thisLayer.neurons.toArray();
                //for each neuron n in the next layer except the bias
                int nextLayerNeuronIndex = 0;
                for (Layer.Neuron nextLayerNeuron : nextLayer.neurons) {
                    if(nextLayerNeuronIndex == nextLayer.neurons.size() - 1)
                        break;
                    //create incoming connections to this neuron
                    for (int k = 0; k < neuronsInThisLayer.length; k++) {
                        //get the neuron in this layer
                        Layer.Neuron thisLayerNeuron = (Layer.Neuron) neuronsInThisLayer[k];
                        //add a connection from this to next
                        float connectionWeight = u.random.randomFloatInARange(minConnectionWeight, maxConnectionWeight);
                        nextLayerNeuron.incomingNeuronConnections.add(nextLayer.createNewNeuronConnection(thisLayerNeuron, nextLayerNeuron, connectionWeight));
                    }
                    nextLayerNeuronIndex++;
                }
            }

        }
    }
    
    public NetworkOutputType getNetworkOutputType(){
        return this.type;
    }

//IO
//*************************************************************************************************************
    public void loadSingularInput(Integer input){
        this.layers.get(0).neurons.get(0).activation = input;
    }
    
    public void loadDoubleInput(double[] input) {
        int i = 0;
        for (Layer.Neuron neuron : this.layers.get(0).neurons) {
            neuron.activation = (float)input[i];
            i++;
        }
    }
    
    public int[] readIntegerOutput(){
        ArrayList<Layer.Neuron> outputLayer = this.layers.get(layers.size() - 1).neurons;
        int[] output = new int[outputLayer.size()];
        
        int index = 0;
        for(Layer.Neuron neuron : outputLayer){
            output[index] = (int)neuron.activation;
            index++;
        }
        
        return output;
    }

    public double[] readDoubleArrayOutput() {
        ArrayList<Layer.Neuron> outputLayer = this.layers.get(layers.size() - 1).neurons;
        double[] output = new double[outputLayer.size()];

        int i = 0;
        for (Layer.Neuron neuron : outputLayer) {
            output[i] = neuron.activation;
            i++;
        }

        return output;
    }
    
    public int[] returnIntegerOutputForSingularInput(Integer input){
        loadSingularInput(input);
        runNetwork();
        return readIntegerOutput();
    }
    
    public double[] returnDoubleOutputsForInputs(double[] input) {
        loadDoubleInput(input);
        runNetwork();
        return readDoubleArrayOutput();
    }

//Network functions
//*************************************************************************************************************

    public void resetFiringValues() {
        for (Layer layer : this.layers)
            for (Layer.Neuron neuron : layer.neurons)
                neuron.activation = 0;
    }

//Network running
//*************************************************************************************************************
    public void runNetwork() {

        boolean firstLayerSkipped = false;
        int connectionCount = 0;
        
        //for each layer...
        for (Layer thisLayer : layers) {
            //skip the first layer. There are no incoming connections to the input layer...
            if(!firstLayerSkipped){
                firstLayerSkipped = true;
                continue;
            }
            
            //for each neuron in the layer...
            for (Layer.Neuron n : thisLayer.neurons) {
                
                float actionPotential = 0;
                //for each incoming connection to the layer...
                for (Layer.NeuronConnection conn : n.incomingNeuronConnections) {
                    //accumulate the action potential

                    //bias neurons always have an activation of 1
                    if(connectionCount == n.incomingNeuronConnections.size() - 1)
                        actionPotential += 1 * conn.weight;
                    else
                        actionPotential += conn.sourceNeuron.activation * conn.weight;

                    connectionCount++;
                }
                connectionCount = 0;

                //feed the action potential through the activation function to calculate the activation
                switch (thisLayer.thresholdFunctionID) {
                    case STEP_FUNCTION: n.activation = stepFunction(actionPotential); break;
                    case RGBA_PIECEWISE_INTERVAL_FUNCTION: n.activation = pieceWiseRGBIntervalFunction(actionPotential); break;
                    case SIGMOID_FUNCTION: n.activation = sigmoidFunction(actionPotential); break;
                    case RGBA_SCALED_SIGMOID_FUNCTION: n.activation = scaledSigmoidFunction(actionPotential); break;
                }

            }
        }

    }

//firing functions
//*************************************************************************************************************
    private float sigmoidFunction(float activation) {
        //Here, Dr. Donaldson used 100 but I am playing around with values currently
        double lambda = 1;
        //Note that through my research various sources have said that using doubles in
        //this function, as required by Java, will always return the same result as if
        //I had implemented it with floats
        return (float)(firingValue / (1 + (1 / Math.exp(-(lambda * (double)activation)))));
    }
    
    private int scaledSigmoidFunction(double activation){
        double lambda = 1;
        double functionValue = (1.0 / (1.0 + Math.exp(-(lambda * activation)))) * 255.0;
        //round to nearest integer value
        int actual = (int)functionValue;
        double base = functionValue - actual;
        int adjusted;
        if(base >= .5)
            adjusted = ((int)functionValue) + 1;
        else
            adjusted = (int)functionValue;

        return adjusted;
    }

    private int stepFunction(double activation) {
        if(activation >= firingThreshold)
            return firingValue;
        return 0;
    }
    
    private int pieceWiseRGBIntervalFunction(double activation){
        int actual = (int)activation;
        double base = activation - actual;
        if(base >= .5)
            return ((int)activation) + 1;
        else
            return (int)activation;
    }

//Weight modification
//*************************************************************************************************************
    public ArrayList<Float> getWeightsInOrder() {

        ArrayList<Float> weights = new ArrayList<>();
        for (int i = 1; i < layers.size(); i++) {
            Layer l = layers.get(i);
            for (Layer.Neuron n : l.neurons)
                for (Layer.NeuronConnection conn : n.incomingNeuronConnections)
                    weights.add(conn.weight);
        }

        return weights;
    }

    public float[] getWeightArrayInOrder(){
        return u.listManipulator.arrayListToFloatArray(getWeightsInOrder());
    }

    public int getNetworkWeightCount() {
        return getWeightsInOrder().size();
    }

    public void setNetworkWeights(float[] weights) {
        int weightIndex = 0;
        for (int i = 1; i < layers.size(); i++) { //notice the change in limit. There are no incoming connections to the input layer to set weights for...
            Layer l = (Layer) layers.get(i);
            for (Layer.Neuron n : l.neurons) {
                for (Layer.NeuronConnection conn : n.incomingNeuronConnections) {
                    conn.weight = weights[weightIndex];
                    weightIndex++;
                }
            }
        }
    }
    
//Printing Functions
//*************************************************************************************************************
    
    public String constructNetworkEncoding(){
        StringBuilder b = new StringBuilder();
        for (Layer l : this.layers){
            b.append("["); b.append(l.neurons.size()); b.append("]");
        }
        return new String(b);
    }

//Genetic Algorithm Training
//*************************************************************************************************************

    //Note that the fitness is evaluated using the convention that 0 is optimal
    //Thus this routine counts the number of bits that are NOT CORRECT
    //This routine only evaluates using the CPU
    private long evaluateBinaryTypeNetworkParameterEncoding(NetworkParameterEncoding encoding, MicroLens lense, GeneticAlgorithm ga) {
        
        //get a network instance
        NeuralNetwork net = ga.getNetworkThreadHandler().get();
        //set the weights of the network to the evolved weights in the encoding
        net.setNetworkWeights(u.listManipulator.arrayListToFloatArray(encoding.inOrderWeights));
        //set the inputs to the evolved inputs in the encoding
        ArrayList<Integer> inputs = encoding.getInputEncoding();
        
        //set the correct singular Integer outputs here  
        ArrayList<Integer> correctOutputs = lense.getBinaryOutputs();
        //record the actual singular Integer outputs of the network here
        ArrayList<Integer> actualIntegerOutputs = new ArrayList<>();
        
        //evaluate
        for (Integer input : inputs) {
            //record the singular outputs here by running the network against the input and
            //translating from the network's binary to an integer between 0 -> 255
            int[] binaryOutputs = net.returnIntegerOutputForSingularInput(input);
            int byteSizeIndex = 7;
            for (int i = 0; i < binaryOutputs.length; i += 8) {
                int[] arr = u.listManipulator.subIntArray(binaryOutputs, i, byteSizeIndex);
                String s = u.listManipulator.intArrayToString(arr);
                actualIntegerOutputs.add(Integer.parseInt(s, 2));
                byteSizeIndex += 8;
            }
            //actualIntegerOutputs.add(binaryToInteger(u.listManipulator.arrayListOfIntegersToString(net.returnOutputForSingularInput(input))));
            //**NOTE : May want to do some performance testing between this methodology and my own custom algorithm
            net.resetFiringValues(); //probably not necessary for this implementation but will keep for safety
        }
        
        //**NOTE : could possibly get a slight performance improvement by combining the above and below

        //calculate the global error of the network
        long globalError = 0;
        for (int i = 0; i < inputs.size(); i++)
            globalError += Math.abs(actualIntegerOutputs.get(i) - correctOutputs.get(i));
        
        return globalError;
    }

    private long evaluatePieceWiseRGBIntervalTypeNetworkParameterEncoding(NetworkParameterEncoding encoding, MicroLens lense, GeneticAlgorithm ga) {
        
        //get a network instance
        //if the ga is null that means we are using PSO so just use the network out of the lens
        //I may want to change this into a flag in the lens because I hate nulls
        NeuralNetwork net = null;
        if(ga == null)
            net = lense.getNetwork();
        else
            net = ga.getNetworkThreadHandler().get();

        //set the weights of the network to the evolved weights in the encoding
        net.setNetworkWeights(u.listManipulator.arrayListToFloatArray(encoding.getWeightEncoding()));
        //set the inputs to the evolved inputs in the encoding
        ArrayList<Integer> inputs = encoding.getInputEncoding();

        //set the correct Integer vector outputs here  
        ArrayList<int[]> correctOutputs = lense.getRGBOutputs();
        //record the actual Integer vector outputs of the network here
        ArrayList<int[]> actualOutputs = new ArrayList<>();

        //calculate the global error of the network
        long globalError = 0;
        if(ga != null)
            switch(ga.getEvaluationProtocol()){
                case SINGLE_THREAD_CPU:
                    //Fall through to Multi-Threaded which is the same for single
                case MULTI_THREAD_CPU:
                    //evaluate network on CPU
                    for (Integer input : inputs)
                        //record the singular outputs here by running the network against the input
                        actualOutputs.add(net.returnIntegerOutputForSingularInput(input));

                    //calculate global error
                    for (int vectorElementIndex = 0; vectorElementIndex < correctOutputs.size(); vectorElementIndex++) {
                        int[] correctElement = correctOutputs.get(vectorElementIndex);
                        int[] actualElement = actualOutputs.get(vectorElementIndex);
                        for (int elementVectorIndex = 0; elementVectorIndex < correctElement.length; elementVectorIndex++) 
                            globalError += Math.abs(correctElement[elementVectorIndex] - actualElement[elementVectorIndex]);
                    }
                break;
                case GPU:
                    for(Integer input : inputs) {
                        //load the singular input
                        net.loadSingularInput(input);
                        //throw the loaded network into a new evaluator and add it to the list of evaluators
                        //this is necessary to use invokeAll() to wait while the net is evaluated
                        ArrayList<NetworkMatrixEvaluator> evaluators = new ArrayList<>();
                        evaluators.add(new NetworkMatrixEvaluator(net));
                        //evaluate and add the result to the actual outputs
                        try {
                            //Invoke the evaluator and record the output
                            ga.getGPUEvaluatorService().invokeAll(evaluators);
                            actualOutputs.add(evaluators.get(0).getResult());
                            //and dispose of this evaluator's resources on this thread
                            ga.getGPUEvaluatorService().execute(new Runnable(){
                                @Override
                                public void run(){
                                    evaluators.get(0).disposeOf();
                                }
                            });
                        } catch (InterruptedException ex) {
                            System.out.println("GPU evaluations interrupted.\n\nExiting...");
                            System.exit(0);
                        }
                    }
                    for (int vectorElementIndex = 0; vectorElementIndex < correctOutputs.size(); vectorElementIndex++) {
                        int[] correctElement = correctOutputs.get(vectorElementIndex);
                        int[] actualElement = actualOutputs.get(vectorElementIndex);
                        for (int elementVectorIndex = 0; elementVectorIndex < correctElement.length; elementVectorIndex++)
                            globalError += Math.abs(correctElement[elementVectorIndex] - actualElement[elementVectorIndex]);
                    }
                break;
            }
        else
        //Particle Swarm
            for (int vectorElementIndex = 0; vectorElementIndex < correctOutputs.size(); vectorElementIndex++) {
                int[] correctElement = correctOutputs.get(vectorElementIndex);
                int[] actualElement = actualOutputs.get(vectorElementIndex);
                for (int elementVectorIndex = 0; elementVectorIndex < correctElement.length; elementVectorIndex++)
                    globalError += Math.abs(correctElement[elementVectorIndex] - actualElement[elementVectorIndex]);
            }

        return globalError;
    }
    
    public long evaluateNetworkParameterEncoding(NetworkParameterEncoding encoding, MicroLens lense, GeneticAlgorithm ga){
        
        switch(lense.getNetwork().getNetworkOutputType()){
            case BINARY_APPROXIMATION: return evaluateBinaryTypeNetworkParameterEncoding(encoding, lense, ga);
            case INTEGER_RGB_VALUE: return evaluatePieceWiseRGBIntervalTypeNetworkParameterEncoding(encoding, lense, ga);
        }
        
        u.dialoger.printMessage("Error in selection of evaluation function");
        
        return Integer.MIN_VALUE;
    }

    private static class NetworkMatrixEvaluator implements Callable<Object> {

        private NeuralNetwork neuralNetwork;
        private float[] resultantMatrix;

        private ArrayList<GPUMatrixMultiplier> multipliers = new ArrayList<>();
        private ArrayList<GPUMatrixMultiplier> consumed = new ArrayList<>();

        public NetworkMatrixEvaluator(NeuralNetwork network){
            neuralNetwork = network;
            float[] weights = network.getWeightArrayInOrder();
            //the weight sub-sectioning could be a flaw or cause confusion between the network model itself
            //and the matrix model. Though in practice it may not matter iff the evaluation method stays the same as either
            //GPU or CPU
            int weightSubsectionBeginIndex = 0;
            for (int i = 0; i < network.layers.size(); i++) {
                if(i == network.layers.size() - 1)
                    break;
                Layer thisLayer = network.layers.get(i);
                Layer nextLayer = network.layers.get(i + 1);
                float[] weightSubsection = u.listManipulator.subFloatArray(weights, weightSubsectionBeginIndex, weightSubsectionBeginIndex + (thisLayer.neurons.size() - 1) * (nextLayer.neurons.size() - 1));

                int thresholdFunctionConstant = 0;
                //These mappings may later encourage me to switch away from enums and go with constants...
                switch (nextLayer.thresholdFunctionID){
                    case STEP_FUNCTION: thresholdFunctionConstant = 1; break;
                    case SIGMOID_FUNCTION: thresholdFunctionConstant = 2; break;
                    case RGBA_SCALED_SIGMOID_FUNCTION: thresholdFunctionConstant = 3; break;
                    case RGBA_PIECEWISE_INTERVAL_FUNCTION: thresholdFunctionConstant = 4; break;
                }

                multipliers.add(new GPUMatrixMultiplier(thresholdFunctionConstant, thisLayer.neurons.size(), thisLayer.getActivations(), weightSubsection, nextLayer.neurons.size() - 1));

                weightSubsectionBeginIndex += (thisLayer.neurons.size() - 1) * (nextLayer.neurons.size() - 1);
            }
        }

        public int[] getResult(){
            int[] result = new int[neuralNetwork.layers.get(neuralNetwork.layers.size() - 1).neurons.size()];
            for (int i = 0; i < resultantMatrix.length; i++)
                result[i] = (int)resultantMatrix[i];
            return result;
        }

        public Object call(){
            resultantMatrix = seriallyMultiply(multipliers);
            return null;
        }

        private float[] seriallyMultiply(ArrayList<GPUMatrixMultiplier> multipliers){
            if(multipliers.size() == 1){
                GPUMatrixMultiplier finalMatrix = multipliers.remove(0);
                finalMatrix.execute(finalMatrix.getColumns2());
//                finalMatrix.execute(640);
                consumed.add(finalMatrix);
                return finalMatrix.getResultantMatrix();
            }

            //Execute and dispose of the 0th matrix
            GPUMatrixMultiplier current = multipliers.remove(0);
            current.execute(current.getColumns2());
//            current.execute(640);
            consumed.add(current);
            //load the next with the result
            multipliers.get(0).loadInputMatrix(current.getResultantMatrix());

            return seriallyMultiply(multipliers);
        }

        private void disposeOf(){
            for(GPUMatrixMultiplier toDispose : consumed)
                toDispose.dispose();
        }

        private class GPUMatrixMultiplier extends Kernel {

            private int rows, columns1, columns2, resultantSize;

            private float[] matrixA, matrixB, matrixC;

            private int thresholdFunctionID = 1;

            public GPUMatrixMultiplier(int thresholdFunctionID, int columns1Size, float[] inputMatrix, float[] weightSubsection, int columns2Size){

                //# rows will always be static at 1 because these are matrix tensors
                this.rows = 1;
                this.columns1 = columns1Size;
                this.columns2 = columns2Size;
                this.thresholdFunctionID = thresholdFunctionID;

                //Matrix A is always treated as the input matrix, whether it is the actual input layer or a subsequent layer in the serialization
                //so load it with the layer's activations
                matrixA = new float[rows * columns1];
                for (int i = 0; i < inputMatrix.length - 1; i++)
                    matrixA[i] = inputMatrix[i];
                //Set the bias value
                matrixA[matrixA.length - 1] = 1;

                //Matrix B is always the weight matrix so it is to be loaded with the subsection of the weight vector
                //that this layer corresponds to
                matrixB = new float[weightSubsection.length];
                for (int i = 0; i < matrixB.length; i++)
                    matrixB[i] = weightSubsection[i];

                //Matrix C is always the resultant matrix.
                matrixC = new float[columns2];

                this.resultantSize = matrixC.length;
            }

            private float[] getResultantMatrix(){
                return matrixC;
            }

            @Override
            public void run(){
                int rowsA = getGlobalId();
                //not sure about matrixB length
                //http://www.jppf.org/samples-pack/GPU/src/org/jppf/example/aparapi/MatrixKernel.java.html
                for (int columnsB = 0; columnsB < resultantSize; columnsB++)
                    multiply(rowsA, columnsB);
            }

            private void multiply(final int rowsA, final int columnsB){
                float sum = 0f;
                for (int i = 0; i < resultantSize; i++)
                    sum += matrixA[rowsA * resultantSize + i] * matrixB[i * resultantSize + columnsB];
                matrixC[columnsB * resultantSize + rowsA] = fireThresholdFunction(sum);
            }

            private float fireThresholdFunction(float activation){
                //Step
                if(thresholdFunctionID == 1)
                    if(activation >= .5)
                        return 1;
                    else
                        return 0;
                //Sigmoid
                if(thresholdFunctionID == 2)
                    return (float)(1 / (1 + (1 / exp(-(100 * (double)activation)))));
                //Scaled Sigmoid
                if(thresholdFunctionID == 3){
                    activation = (float)((1 / (1 + (1 / exp(-(100 * (double)activation))))) * 255.0);
                    int actual = (int)activation;
                    double base = activation - actual;
                    if(base >= .5)
                        return ((int)activation) + 1;
                    else
                        return (int)activation;
                }
                //Piecewise RGB Interval
                if(thresholdFunctionID == 4){
                    int actual = (int)activation;
                    double base = activation - actual;
                    if(base >= .5)
                        return ((int)activation) + 1;
                    else
                        return (int)activation;
                }
                return Float.MAX_VALUE;
            }

            public void loadInputMatrix(float[] input){
                for(int i = 0; i < input.length; i++)
                    matrixA[i] = input[i];
            }

            public int getColumns2(){
                return columns2;
            }
        }

    }
    
}
