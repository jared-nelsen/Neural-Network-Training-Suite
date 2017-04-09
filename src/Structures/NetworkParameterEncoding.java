package Structures;


import Algorithms.GeneticAlgorithm;
import Utilities.Utilities;

import java.util.ArrayList;

/**
 *
 * @author <Jared Nelsen>
 */

public class NetworkParameterEncoding{
    
    private Utilities u = new Utilities();

    //Input Vector
    private final int minInput = 0;
    private final int maxInput = 255;
    private final int inputBumpValue = 1;

    public ArrayList<Integer> inputVector = new ArrayList<>();

    //Weight Parameters
    public float minWeight = -1;
    public float maxWeight = 1;
    //This fractional bump is a rational guess
    private float weightBumpValue = maxWeight / 100;

    public ArrayList<Float> inOrderWeights = new ArrayList<>();

    public long fitness = Long.MAX_VALUE;
    
    public NetworkParameterEncoding(){
        
    }
    
    public NetworkParameterEncoding(ArrayList<Integer> inputEncoding, ArrayList<Float> weights) {
        inputVector = inputEncoding;
        inOrderWeights = weights;
    }
    
    public NetworkParameterEncoding generateRandomEncoding(ArrayList<Integer> inputVector, int numWeights) {
        
        ArrayList<Float> newWeights = new ArrayList<>();
        for (int i = 0; i < numWeights; i++)
            newWeights.add(u.random.randomFloatInARange(minWeight, maxWeight));
        
        NetworkParameterEncoding newEncoding = this.copyEncoding();
        newEncoding.setEncoding(inputVector, newWeights);
        return newEncoding;
    }

    public void setFitness(long fitness){
        this.fitness = fitness;
    }

    //does this have to be public?
    public ArrayList<Integer> getInputEncoding(){
        return inputVector;
    }

    public int[] getInputEncodingBackingArray(){
        return u.listManipulator.arrayListToIntArray(getInputEncoding());
    }

    //does this have to be public?
    public ArrayList<Float> getWeightEncoding(){
        return inOrderWeights;
    }

    public float[] getWeightEncodingBackingArray(){
        return u.listManipulator.arrayListToFloatArray(getWeightEncoding());
    }
    
    public void setEncoding(ArrayList<Integer> inputEncoding, ArrayList<Float> weightEncoding){
        setInputEncoding(inputEncoding);
        setWeightEncoding(weightEncoding);
    }
    
    public void setInputEncoding(ArrayList<Integer> inputEncoding){
        inputVector = inputEncoding;
    }
    
    public void setWeightEncoding(ArrayList<Float> weightEncoding){
        inOrderWeights = weightEncoding;
    }
    
    public void clearEncoding(){
        inputVector.clear();
        inOrderWeights.clear();
    }

    public NetworkParameterEncoding copyEncoding() {
        NetworkParameterEncoding w = new NetworkParameterEncoding();
        w.inputVector = new ArrayList(this.inputVector);
        w.fitness = this.fitness;
        w.minWeight = this.minWeight;
        w.maxWeight = this.maxWeight;
        w.inOrderWeights = new ArrayList(this.inOrderWeights);
        return w;
    }
    
    public String generateEncodingStringToWrite(){
        StringBuilder b = new StringBuilder();
        for (Integer in : this.inputVector) {
            b.append("["); b.append(in); b.append("]");
        }
        b.append("\n");
        for (Float weight : this.inOrderWeights) {
            b.append("["); b.append(weight); b.append("]");
        }
        return new String(b);
    }
    
    public void mutateEncoding(double mutationRate, GeneticAlgorithm.MutationProtocol mutationProtocol, GeneticAlgorithm.ComponentsToEvolve components){
        switch(components){
            case INPUTS: 
                mutateInputs(mutationRate, mutationProtocol);
                break;
            case WEIGHTS:
                mutateWeights(mutationRate, mutationProtocol);
                break;
            case BOTH: 
                mutateInputs(mutationRate, mutationProtocol);
                mutateWeights(mutationRate, mutationProtocol);
                break;
        }
    }
    
    private void mutateInputs(double mutationRate, GeneticAlgorithm.MutationProtocol mutationProtocol){
        for (int i = 0; i < inputVector.size(); i++) 
            if(Math.random() <= mutationRate)
                switch (mutationProtocol) {
                    case VALUE_BUMPING:
                        int newValue = inputVector.get(i);
                        if(u.random.randomIntInARange(0, 1) == 0)
                            newValue = newValue - inputBumpValue;
                        else
                            newValue = newValue + inputBumpValue;
                        if(newValue >= minInput && newValue <= maxInput)
                            inputVector.set(i, newValue);
                        break;
                    case VALUE_JUMPING:
                        inputVector.set(i, u.random.randomIntInARange(minInput, maxInput));
                        break;
                }
    }
    
    private void mutateWeights(double mutationRate, GeneticAlgorithm.MutationProtocol mutationProtocol){
        for (int i = 0; i < inOrderWeights.size(); i++)
            if(Math.random() <= mutationRate)
                switch (mutationProtocol) {
                    case VALUE_BUMPING:
                        float newValue = inOrderWeights.get(i);
                        if(u.random.randomIntInARange(0, 1) == 0)
                            newValue = newValue - weightBumpValue;
                        else
                            newValue = newValue + weightBumpValue;
                        if(newValue >= minWeight && newValue <= maxWeight)
                            inOrderWeights.set(i, newValue);
                        break;
                    case VALUE_JUMPING:
                        inOrderWeights.set(i, u.random.randomFloatInARange(minWeight, maxWeight));
                        break;
                }

    }

    public NetworkParameterEncoding averageWithThisEncoding(NetworkParameterEncoding toAverageWith){
        NetworkParameterEncoding copy = this.copyEncoding();
        for (int i = 0; i < copy.getWeightEncoding().size(); i++)
            copy.getWeightEncoding().set(i, ((copy.getWeightEncoding().get(i) + toAverageWith.getWeightEncoding().get(i)) / 2));
        return copy;
    }

    public int getMinInput() {
        return minInput;
    }

    public int getMaxInput() {
        return maxInput;
    }
}
