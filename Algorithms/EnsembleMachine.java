package Algorithms;

//Jared Nelsen

import Lenses.MicroLens;
import Structures.NetworkParameterEncoding;
import Utilities.Utilities;

import java.util.ArrayList;

public class EnsembleMachine {

    private final Utilities u = new Utilities();

    //Experts
    private final ArrayList<Expert> experts = new ArrayList<>();

    //Expert Properties
    private final int expertGenerationsToRun = 1000;
    private final int expertPopulationSize = 8;

    //Ensemble Machine Properties
    private final int numberOfExperts = expertPopulationSize;

    public EnsembleMachine(MicroLens reference, GeneticAlgorithm.ComponentsToEvolve components, GeneticAlgorithm.MutationProtocol mutationProtocol, GeneticAlgorithm.EvaluationProtocol evalMethod, int vectorDimension){
        u.dialoger.printMessage("The Ensemble Machine Algorithm was chosen");

        u.dialoger.printLn("Generating Ensemble");

        for(int i = 0; i < numberOfExperts; i++)
            experts.add(new Expert(new GeneticAlgorithm(reference, components, mutationProtocol, evalMethod, vectorDimension)));

        u.dialoger.newLines(1);
        u.dialoger.printLn("Ensemble generated");
        u.dialoger.newLines(1);
    }

    public NetworkParameterEncoding runEnsemble(){
        u.dialoger.printLn("Beginning Ensemble Expert Training");
        u.dialoger.newLines(1);

        long lastBestFitness = Integer.MAX_VALUE;
        long currentBestFitness = Integer.MAX_VALUE;
        int rehearsalCount = 0;
        while(currentBestFitness > 0){
            trainExperts();
            reconstituteAndDistributePopulationInOrder();
            currentBestFitness = getBestFitnessInEnsemble();
            if(currentBestFitness < lastBestFitness) {
                lastBestFitness = currentBestFitness;
                printBestFitness(rehearsalCount, currentBestFitness);
            }
            rehearsalCount++;
        }
        return getFittestIndividualInEnsemble();
    }

    private void trainExperts(){
        for (int i = 0; i < experts.size(); i++) {
            Expert expert = experts.get(i);
            expert.train();
        }
    }

    private void reconstituteAndDistributePopulationInOrder(){
        //Reconstitute
        ArrayList<NetworkParameterEncoding> reconstituted = new ArrayList<>();
        for (int i = 0; i < expertPopulationSize; i++) {
            ArrayList<NetworkParameterEncoding> inOrderEncodings = new ArrayList<>();
            for(Expert expert : experts)
                inOrderEncodings.add(expert.getGA().getPopulation().get(i));
            reconstituted.add(averageExperts(inOrderEncodings));
        }

        //Distribute
        for(Expert expert : experts)
            for (int i = 0; i < reconstituted.size(); i++)
                expert.getGA().getPopulation().set(i, reconstituted.get(i).copyEncoding());
    }

    private void printBestFitness(int rehearsalCount, long bestFitness){
        u.dialoger.printLn("Rehearsal " + rehearsalCount + ": Best fitness: " + bestFitness);
    }

    private long getBestFitnessInEnsemble(){
        return getFittestIndividualInEnsemble().fitness;
    }

    private NetworkParameterEncoding getFittestIndividualInEnsemble(){
        NetworkParameterEncoding fittest = null;
        long min = Long.MAX_VALUE;
        for(Expert expert : experts)
            if(expert.getGA().getPopulation().get(0).fitness < min) {
                fittest = expert.getGA().getPopulation().get(0).copyEncoding();
                min = fittest.fitness;
            }
        return fittest;
    }

    private NetworkParameterEncoding averageExperts(ArrayList<NetworkParameterEncoding> experts){
        if(experts.size() == 1)
            return experts.remove(0);

        return experts.remove(0).averageWithThisEncoding(experts.get(0));
    }

    private class Expert{

        private final GeneticAlgorithm geneticAlgorithm;

        public Expert(GeneticAlgorithm geneticAlgorithm){
            geneticAlgorithm.setToEnsembleMode(expertGenerationsToRun, expertPopulationSize);
            this.geneticAlgorithm = geneticAlgorithm;
        }

        public void train(){
            this.geneticAlgorithm.run();
        }

        public GeneticAlgorithm getGA(){
            return this.geneticAlgorithm;
        }

    }

}
