package Evaluators;

import Lenses.MicroLens;
import Structures.NetworkParameterEncoding;
import Structures.NeuralNetwork;
import Utilities.Utilities;
import com.amd.aparapi.Kernel;

import java.util.ArrayList;

/**
 * Created by jared on 8/20/16.
 */
public class GpuNeuralNetworkEvaluator {

    //Utilities
    private Utilities u = new Utilities();

    private MicroLens lens;

    //Class level GPU Kernel
    private GPUKernelEvaluator gpuKernel;

    public GpuNeuralNetworkEvaluator(MicroLens lens){
        this.lens = lens;
        this.gpuKernel = new GPUKernelEvaluator();
        this.gpuKernel.setExecutionMode(Kernel.EXECUTION_MODE.SEQ);
    }

    public void shutdown(){
        this.gpuKernel.dispose();
    }

    public ArrayList<NetworkParameterEncoding> evaluatePopulation(ArrayList<NetworkParameterEncoding> population){

        //Construct and evaluate a Vectorized Neural Network for each weight encoding population member
        for(NetworkParameterEncoding encoding : population) {
            //Construct the  Vectorized Neural Network that corresponds to this encoding
            VectorizedNeuralNetwork vectorizedNeuralNetwork = new VectorizedNeuralNetwork(this.lens.getNetwork(), encoding.getWeightEncodingBackingArray());


            //Now serially evaluate this vectorized neural network on the GPU by:
            //1. Allocating layer computation features to the GPU
            //2. Executing the computation of the GPU


            //Keep a record of the inputs for computation in the immediately subsequent layer
            //The first layer will be the inputs to the network itself
            float[] vectorizedlayerInputs = u.listManipulator.intArrayToFloatArray(encoding.getInputEncodingBackingArray());

            //Record each float vector output the GPU computed network for comparison later
            ArrayList<float[]> actualOutputs = new ArrayList<>();

            boolean atFirstLayer = true;
            for(Float input : vectorizedlayerInputs) {

                //If we are at the first layer in the vectorized neural network then we need to load the input to the network into the first layer
                if(atFirstLayer){
                    //Note we must add the bias index too
                    vectorizedNeuralNetwork.getVectorizedLayers().get(0).loadInputMatrix(new float[]{input, 1});
                    atFirstLayer = false;
                }

                int vectorizedLayerCount = 0;
                for (VectorizedNeuralNetwork.VectorizedLayer vectorizedLayer : vectorizedNeuralNetwork.getVectorizedLayers()) {
                    //Pass the vectorized layer features to the GPU
                    gpuKernel.allocate(vectorizedLayer.getMatrixA(), vectorizedLayer.getMatrixB(), vectorizedLayer.getThresholdFunctionID(), vectorizedLayer.getMatrixCTensorDimension());

                    //Execute the model on the GPU where the integer input is the number of threads
                    gpuKernel.execute(1);

                    //I believe I read that my dell's gpu thread count was 640
                    //I should look for a way to read that from each graphics card to be able to run efficiently on other systems

                    //Get and record the result of the computation
                    vectorizedlayerInputs = gpuKernel.getResultantMatrix();

                    //If we are at the last layer in the vectorized neural network we finish this routine
                    //and do not load the result into a next layer
                    if (vectorizedLayerCount == vectorizedNeuralNetwork.getVectorizedLayers().size() - 1)
                        break;

                    //But if we are not yet at the last vectorized layer in the vectorized neural network
                    //we load the next vectorized neural network layer in the vectorized neural network
                    vectorizedNeuralNetwork.getVectorizedLayers().get(vectorizedLayerCount + 1).loadInputMatrix(vectorizedlayerInputs);

                    vectorizedLayerCount++;
                }

                //Record the outputs of this vectorized neuralNetwork
                actualOutputs.add(vectorizedNeuralNetwork.getNetworkOutput());
            }


            //Now we evaluate the fitness of the network
            //Set the correct Integer vector outputs here
            long globalError = 0;
            ArrayList<int[]> correctOutputs = lens.getRGBOutputs();
            for (int vectorElementIndex = 0; vectorElementIndex < correctOutputs.size(); vectorElementIndex++) {
                int[] correctElement = correctOutputs.get(vectorElementIndex);
                float[] actualElement = actualOutputs.get(vectorElementIndex);
                for (int elementVectorIndex = 0; elementVectorIndex < correctElement.length; elementVectorIndex++)
                    globalError += Math.abs(correctElement[elementVectorIndex] - actualElement[elementVectorIndex]);
            }

            //And set that global error back onto the encoding as its fitness
            encoding.setFitness(globalError);
        }

        return population;
    }

    private class VectorizedNeuralNetwork {

        private ArrayList<VectorizedLayer> vectorizedLayers = new ArrayList<>();

        public VectorizedNeuralNetwork(NeuralNetwork neuralNetworkReference, float[] weightArray){
            //First we will make a reference to the weight array.
            float[] weights = weightArray;

            //Now we must split the weight array into subsections. Each subsection represents the weights connecting
            //some layer A to the immediately subsequent layer B. Thus we calculate the split based in the features of
            //layer A and layer B.

            int weightSubsectionBeginIndex = 0;
            for (int i = 0; i < neuralNetworkReference.layers.size(); i++) {
                //We break the subsectioning at the last layer because there is no layer following this one.
                if(i == neuralNetworkReference.layers.size() - 1)
                    break;

                //Designate the layers in question.
                NeuralNetwork.Layer layerA = neuralNetworkReference.layers.get(i);
                NeuralNetwork.Layer layerB = neuralNetworkReference.layers.get(i + 1);

                //Calculate the ending index of this subsection
                int weightSubsectionEndIndex = weightSubsectionBeginIndex + weightSubsectionSizeBetween(layerA, layerB);

                //Carry out the subsectioning
                float[] weightSubsection = u.listManipulator.subFloatArray(weights, weightSubsectionBeginIndex, weightSubsectionEndIndex);

                //Designate the neuron threshold activation function to be used on the calculated activations in layer B
                //Map the enumerated value onto the designated integer indicator so the GPU kernel will accept it
                int thresholdFunctionConstant = 0;
                //We may not use a switch here because the enum namespace is not local
                if(layerB.thresholdFunctionID.equals(layerB.thresholdFunctionID.STEP_FUNCTION))
                    thresholdFunctionConstant = 1;
                else if(layerB.thresholdFunctionID.equals(layerB.thresholdFunctionID.SIGMOID_FUNCTION))
                    thresholdFunctionConstant = 2;
                else if(layerB.thresholdFunctionID.equals(layerB.thresholdFunctionID.RGBA_SCALED_SIGMOID_FUNCTION))
                    thresholdFunctionConstant = 3;
                else if(layerB.thresholdFunctionID.equals(layerB.thresholdFunctionID.RGBA_PIECEWISE_INTERVAL_FUNCTION))
                    thresholdFunctionConstant = 4;

                //Construct a Vectorized Layer representing these facets
                //Note the -1 offset in this statement correcting the size value
                VectorizedLayer vectorizedLayer = new VectorizedLayer(thresholdFunctionConstant, layerA.getActivations(), weightSubsection, layerB.neurons.size() - 1);
                //Add the Vectorized Layer to this Vectorized Network Model
                vectorizedLayers.add(vectorizedLayer);

                //Advance the subsection index
                weightSubsectionBeginIndex += weightSubsectionSizeBetween(layerA, layerB);
            }
        }

        public ArrayList<VectorizedLayer> getVectorizedLayers(){
            return vectorizedLayers;
        }

        private int weightSubsectionSizeBetween(NeuralNetwork.Layer layerA, NeuralNetwork.Layer layerB){
            return (layerA.neurons.size() - 1) * (layerB.neurons.size() - 1);
        }

        private float[] getNetworkOutput(){
            return vectorizedLayers.get(vectorizedLayers.size() - 1).getMatrixC();
        }

        private class VectorizedLayer{

            //The input layer representation
            private float[] matrixA;
            //The weight layer representation
            private float[] matrixB;
            //The output layer representation
            private float[] matrixC;

            //The signature for which threshold function to use on matrixB
            //Aparapi does not support enums so it is better to make this
            //indicator concrete here by converting it from the enum on
            //the network level.
            //The mappings follow:
            // 1 = Step Function
            // 2 = Sigmoid Function
            // 3 = RGBA Scaled Sigmoid Function
            // 4 = RGBA Piecewise Interval Function
            private int thresholdFunctionID = 0;

            public VectorizedLayer(int thresholdID, float[] inputMatrix, float[] weightSubsection, int outputMatrixDimension){

                //Set the signature for the threshold function to run against the resultant matrix activation values once calculated
                thresholdFunctionID = thresholdID;

                //Matrix A represents the input matrix layer's activations
                //This may or may not need to be copied But I am trying to avoid excessive copying
                matrixA = inputMatrix;

                //Matrix B represents the weight matrix so loaded it with the designated subsection of the weight vector.
                //Recall that a weight subsection is defined as the slice of the network-expansive weight tensor that lies
                //between these layer representations
                matrixB = new float[weightSubsection.length];
                for (int i = 0; i < matrixB.length; i++)
                    matrixB[i] = weightSubsection[i];

                //Matrix C represents the output layer or the resultant matrix calculated by the matrix multiplication
                matrixC = new float[outputMatrixDimension];

            }

            public void loadInputMatrix(float[] input){
                for (int i = 0; i < input.length; i++)
                    matrixA[i] = input[i];
            }

            public float[] getMatrixA(){
                return matrixA;
            }

            public float[] getMatrixB(){
                return matrixB;
            }

            public float[] getMatrixC() { return matrixC; };

            public int getMatrixCTensorDimension(){
                return matrixC.length;
            }

            public int getThresholdFunctionID(){
                return thresholdFunctionID;
            }

        }

    }

    private class GPUKernelEvaluator extends Kernel {

        //The matrix tensors
        private float[] matrixA, matrixB, matrixC;

        //The size of the resultant matrix tensor
        private int matrixASize;

        //The weight matrix tensor size
        private int weightMatrixTensorSize;

        //The resultant matrix size
        private int resultantMatrixSize;

        //The threshold function ID
        private int thresholdFunctionID = 0;

        public GPUKernelEvaluator(){

        }

        public void allocate(float[] matrixA, float[] matrixB, int thresholdFunctionID, int matrixCDimension){

            //Assign the proper dimensions to the matrices
            this.matrixASize = matrixA.length;
            this.weightMatrixTensorSize = matrixB.length; //Note that matrixB size should be equivalent to matrixCDimension
            this.resultantMatrixSize = matrixCDimension;

            //Assign the threshold function ID
            this.thresholdFunctionID = thresholdFunctionID;

            //Matrix A is loaded as the input matrix
            this.matrixA = new float[matrixA.length];
            for(int i = 0; i < matrixA.length; i++)
                this.matrixA[i] = matrixA[i];

            //Explicitly set the bias value
            this.matrixA[this.matrixA.length - 1] = 1;

            //Matrix B is loaded as the intermediate matrix
            this.matrixB = new float[matrixB.length];
            for (int i = 0; i < matrixB.length; i++)
                this.matrixB[i] = matrixB[i];

            //Matrix C is the resultant matrix
            matrixC = new float[this.matrixASize];

        }

        private float[] getResultantMatrix(){
            return matrixC;
        }

        @Override
        public void run(){
            int rowsA = getGlobalId();

            for (int columnsB = 0; columnsB < resultantMatrixSize; columnsB++)
                multiply(rowsA, columnsB);
        }

        private void multiply(int rowsA, int columnsB){
            float sum = 0f;
            for (int i = 0; i < matrixASize; i++)
                sum += matrixA[i] * matrixB[columnsB];
            matrixC[columnsB] = fireThresholdFunction(sum);
        }

//        @Override
//        public void run(){
//            int i = getGlobalId();
//            int j = getPassId();
//            float value = 0f;
//            for (int k = 0; k < matrixASize; k++)
//                value += matrixA[k + i * matrixASize] * matrixB[k + j];
//            matrixC[i + matrixASize - 1 + j] = fireThresholdFunction(value);
//        }

        private float fireThresholdFunction(float activation){
            //Step Function
            if(thresholdFunctionID == 1)
                if(activation >= .5)
                    return 1;
                else
                    return 0;
            //Sigmoid Function
            if(thresholdFunctionID == 2)
                return (float)(1 / (1 + (1 / exp(-(100 * (double)activation)))));
            //RGBA Scaled Sigmoid Function
            if(thresholdFunctionID == 3){
                activation = (float)((1 / (1 + (1 / exp(-(100 * (double)activation))))) * 255.0);
                int actual = (int)activation;
                double base = activation - actual;
                if(base >= .5)
                    return ((int)activation) + 1;
                else
                    return (int)activation;
            }
            //RGBA Piecewise Interval Function
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
    }

}