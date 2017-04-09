package Lenses;


import Algorithms.EnsembleMachine;
import Algorithms.GeneticAlgorithm;
import Structures.NeuralNetwork;

import java.util.ArrayList;

/**
 *
 * @author jared
 */
public class MicroLens {
    
    //The in-order set of singular input values between 0 -> 255
    private ArrayList<Integer> I;
    private final ReconciliationAlgorithm algorithm;
    private final NeuralNetwork lens;
    private final GeneticAlgorithm.EvaluationProtocol evaluationProtocol;
    private final GeneticAlgorithm.ComponentsToEvolve componentsToEvolve;
    private final GeneticAlgorithm.MutationProtocol mutationProtocol;
    //The in-order set of singular output values between 0 -> 255
    //The is actively translated from the binary output of the lens in the case of a binary output type net
    private ArrayList<Integer> O;
    //The in-order set of actual RGB values from a net in the form of a set of int values of a size no less than 4
    private ArrayList<int[]> RGB;
    
    public enum LenseType{
        FOUR,
        EIGHT,
        SIXTEEN,
        THIRTY_TWO
    }

    public enum ReconciliationAlgorithm{
        GENETIC_ALGORTIHM,
        PARTICLE_SWARM_OPTIMIZATION,
        EMSEMBLE_MACHINE
    }
    
    public MicroLens(ReconciliationAlgorithm algorithm, LenseType type, GeneticAlgorithm.EvaluationProtocol evaluationProtocol, GeneticAlgorithm.ComponentsToEvolve componentsToEvolve, GeneticAlgorithm.MutationProtocol mutationProtocol, NeuralNetwork.NetworkOutputType outputType){
        
        I = new ArrayList<>();
        
        ArrayList<Integer> layers = new ArrayList<>();

        layers.add(1);

        switch(type){
            case FOUR: layers.add(2); layers.add(4); break;
            case EIGHT: layers.add(4); layers.add(8); break;
            case SIXTEEN: layers.add(8); layers.add(16); break;
            case THIRTY_TWO: layers.add(16); layers.add(32); break;
        }

        this.algorithm = algorithm;
        this.evaluationProtocol = evaluationProtocol;
        this.componentsToEvolve = componentsToEvolve;
        this.mutationProtocol = mutationProtocol;
        
        this.lens = new NeuralNetwork(layers, NeuralNetwork.ThresholdFunctionID.SIGMOID_FUNCTION,
                                               NeuralNetwork.ThresholdFunctionID.RGBA_SCALED_SIGMOID_FUNCTION, outputType);
        O = new ArrayList<>();
    }
    
    public NeuralNetwork getNetwork(){
        return lens;
    }
    
    public ArrayList<Integer> getBinaryOutputs(){
        return O;
    }    
    
    public void setBinaryOutputs(ArrayList<Integer> outputs){
        O = outputs;
    }
    
    public ArrayList<int[]> getRGBOutputs(){
        return RGB;
    }
    
    public void setRGBOutputs(ArrayList<int[]> rgb) {
        RGB = rgb;
    }
    
    public ArrayList<Integer> getInputs(){
        return I;
    }
    
    //** May want to convert inputs to Arraylist<int[]> after primitve implementation
    public void setInputs(ArrayList<Integer> inputs){
        I = inputs;
    }
    
    public void reconcileInputsWithOutputs(){
        int vectorDimension = 0;
        
        switch(lens.getNetworkOutputType()){
            case BINARY_APPROXIMATION: vectorDimension = this.O.size(); break;
            case INTEGER_RGB_VALUE: vectorDimension = this.RGB.size(); break;                
        }

        switch(algorithm){
            case GENETIC_ALGORTIHM:
                setInputs(new GeneticAlgorithm(this, componentsToEvolve, mutationProtocol, evaluationProtocol, vectorDimension).run().getInputEncoding());
            break;
            case PARTICLE_SWARM_OPTIMIZATION:
                System.out.println("PSO Not Operational");
//                ParticleSwarmNetworkTrainer psot = new ParticleSwarmNetworkTrainer();
//                setInputs(psot.trainNetwork(this, new NetworkParameterEncoding().generateRandomEncoding(vectorDimension, lens.getNetworkWeightCount())).getInputEncoding());
            break;
            case EMSEMBLE_MACHINE:
                EnsembleMachine machine = new EnsembleMachine(this, componentsToEvolve, mutationProtocol, evaluationProtocol, vectorDimension);
                setInputs(machine.runEnsemble().getInputEncoding());
            break;
        }
    }
    
    public void runLense(){
        
    }
    
}
