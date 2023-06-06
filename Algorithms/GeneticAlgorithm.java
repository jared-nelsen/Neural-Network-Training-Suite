package Algorithms;


import Evaluators.GpuNeuralNetworkEvaluator;
import Lenses.MicroLens;
import Structures.NetworkParameterEncoding;
import Structures.NeuralNetwork;
import Utilities.Utilities;

import java.util.ArrayList;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 *
 * @author <Jared Nelsen>
 */

public class GeneticAlgorithm {

    //Utilities
    public final Utilities u = new Utilities();
    
    //Printing
    public boolean silent = false;
    public final boolean printOnlyNewBestFitnesses = true;

    //Generations
    private int generationCount = 0;
    private int maxGenerations = Integer.MAX_VALUE;

    //Population
    public int populationSize = 8;
    public ArrayList<NetworkParameterEncoding> population;

    //Fitnesses
    //0 is optimal so start at max to decrease
    public double bestFitness = Double.MAX_VALUE;
    private double currentBestFitness = Double.MAX_VALUE;
    
    //Rates
    private final double crossoverRate = .9;
    private final double mutationRate = .01;
    private final double spontaneousDeathRate = .01;
    
    //Elitism
    private final boolean elitismOn = true;
    private final int numberOfEliteToSave = (int)(populationSize * .3);
    private final ArrayList<NetworkParameterEncoding> elites = new ArrayList();

    //Encodings
    private ArrayList<Integer> inputEncoding = new ArrayList<>();
        
    //Methodologies
    private enum CrossoverMethod {
        HALVING,
        RANDOM_ELEMENT,
        RANDOM_METHOD;
    }
    public CrossoverMethod crossoverMethodID = CrossoverMethod.RANDOM_METHOD;

    public enum MutationProtocol {
        VALUE_BUMPING,
        VALUE_JUMPING
    }
    public MutationProtocol mutationProtocol = MutationProtocol.VALUE_JUMPING;
    
    public enum ComponentsToEvolve {
        INPUTS,
        WEIGHTS,
        BOTH
    }
    public ComponentsToEvolve componentsToEvolveID = ComponentsToEvolve.WEIGHTS;

    //Computational Performance Parameters
    public enum EvaluationProtocol {
        SINGLE_THREAD_CPU,
        MULTI_THREAD_CPU,
        GPU
    }
    private EvaluationProtocol evaluationProtocol = EvaluationProtocol.SINGLE_THREAD_CPU;
    public EvaluationProtocol getEvaluationProtocol(){ return this.evaluationProtocol; }

    //Multithreaded indicator
    private final int cores = Runtime.getRuntime().availableProcessors();
    
    //Services
    private ExecutorService evaluatorService = null;
    private ExecutorService gpuEvaluatorService;
    public ExecutorService getGPUEvaluatorService(){
        return this.gpuEvaluatorService;
    }
    
    //Concurrent CPU Services
    //**One used for concurrent CPU sorting, may be removed if I can successfully sort via GPU
    private ExecutorService sorterService = null;
    private ExecutorService mergerService = null;
    
    //Neural Network operatives
    private final MicroLens lens;
    private final NetworkThreadHandler threadHandler;
    public NetworkThreadHandler getNetworkThreadHandler(){ return threadHandler; }

    //The instance of the CPU Neural Network Evaluator
    private GpuNeuralNetworkEvaluator gpuPopulationEvaluator;
    
    //Input and Output vector dimension (stays same throughout the ENTIRE program)
    //This is pass compatible with Binary Type or RGB type output networks
    //**I wonder if I can get around storing this and get it implicitly
    private final int ioVectorDimension;

    public GeneticAlgorithm(MicroLens reference, ComponentsToEvolve components, MutationProtocol mutationProtocol, EvaluationProtocol evaluationProtocol, int vectorDimension){
        this.lens = reference;
        this.componentsToEvolveID = components;
        this.mutationProtocol = mutationProtocol;
        this.evaluationProtocol = evaluationProtocol;
        this.ioVectorDimension = vectorDimension;
        this.threadHandler = new NetworkThreadHandler(lens.getNetwork());
        this.gpuPopulationEvaluator = new GpuNeuralNetworkEvaluator(lens);
    }
    
    public NetworkParameterEncoding run() {

        ArrayList<NetworkParameterEncoding> population = createPopulation(populationSize);

        u.timer.start();
        while (generationCount <= maxGenerations && currentBestFitness > 0) {

            population = saveTheElite(sortPopulation(evaluatePopulation(injectElites(spontaneousDeath(mutate(crossover(select(population))))))));

            //sortPopulationConcurrently() ??


            currentBestFitness = population.get(0).fitness;
            boolean newBest = false;
            if (currentBestFitness < bestFitness) {
                newBest = true;
                bestFitness = currentBestFitness;
            }
            
            String logString = "Generation " + generationCount + ": Best fitness = " + bestFitness + ": Found at second : " + u.timer.getSeconds();
            if(!silent)
                if (!printOnlyNewBestFitnesses)
                    System.out.println(logString);
                else if (printOnlyNewBestFitnesses && newBest)
                    System.out.println(logString);
            
            generationCount++;            
        }

        if(!silent)
            u.dialoger.printMessage("The lens has been successfully reconciled!");

        generationCount = 0;
        
        return population.get(0);
    }

    public int setToEnsembleMode(int generationsToRun, int populationSize){
        this.maxGenerations = generationsToRun;
        this.populationSize = populationSize;
        this.silent = true;
        return this.populationSize;
    }
    
    public ArrayList<NetworkParameterEncoding> createPopulation(int size) {
        if(!silent)
            u.dialoger.printLn("Generating Population");

        NetworkParameterEncoding encoding = new NetworkParameterEncoding();
        for (int i = 0; i < ioVectorDimension; i++)
            inputEncoding.add(u.random.randomIntInARange(encoding.getMinInput(), encoding.getMaxInput()));

        ArrayList<NetworkParameterEncoding> newPop = new ArrayList();
        for (int i = 0; i < size; i++)
            newPop.add(encoding.generateRandomEncoding(u.listManipulator.copyIntegerArrayList(inputEncoding), lens.getNetwork().getNetworkWeightCount()));

        return newPop;        
    }

    public ArrayList<NetworkParameterEncoding> evaluatePopulation(ArrayList<NetworkParameterEncoding> population){

        switch (evaluationProtocol){

            case SINGLE_THREAD_CPU: return evaluatePopulationOnSingleThread(population);
            case MULTI_THREAD_CPU: return evaluatePopulationMultiThreaded(population);
            case GPU: return gpuPopulationEvaluator.evaluatePopulation(population);

        }
        System.out.println("Population Not Evaluated Correctly");
        return population;
    }

    public ArrayList<NetworkParameterEncoding> evaluatePopulationOnSingleThread(ArrayList<NetworkParameterEncoding> pop){

        ArrayList<NetworkParameterEncoding> evaluatedPop = new ArrayList();
        for (NetworkParameterEncoding encoding : pop) {
            encoding.fitness = lens.getNetwork().evaluateNetworkParameterEncoding(encoding, lens, this);
            evaluatedPop.add(encoding);
        }

        return evaluatedPop;
    }

    public ArrayList<NetworkParameterEncoding> evaluatePopulationMultiThreaded(ArrayList<NetworkParameterEncoding> population) {
        
        //double check the cores are correct or the program will crash
        if(populationSize % cores != 0){
            System.out.println("The population size is not perfectly divisible by the number of cores...");
            System.exit(0);
        }

        if(evaluatorService == null){
            evaluatorService = Executors.newFixedThreadPool(1);
            if(evaluationProtocol == EvaluationProtocol.MULTI_THREAD_CPU)
                gpuEvaluatorService = Executors.newFixedThreadPool(cores);
        }
        
        //split the population
        int splitMagnitude = population.size() / cores;
        ArrayList<Evaluator> evaluators = new ArrayList<>();
        for (int i = 0; i < population.size(); i += splitMagnitude)
            evaluators.add(new Evaluator(new ArrayList(population.subList(i, i + splitMagnitude))));
        
        //evaluate the population
        try{
            evaluatorService.invokeAll(evaluators);
        }catch(Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        
        //gather the evaluated populations into the working population
        population.clear();
        for(Evaluator e : evaluators)
            population.addAll(e.getEvaluated());
        
        //Safety Net
        if(population.isEmpty() || population.size() < population.size()){
            System.out.println("");
            System.out.println("ERROR!");
            System.out.println("The evaluation of the population has failed!");
            System.out.println("This is generally caused by a failure in the Weight Evaluation Routine in the Neural Network Class.");
            System.out.println("\nExiting...");
            System.exit(0);
        }

        return population;
    }

    //This implementation uses Roulette Wheel Selection. This routine returns a population double the size of
    //the population passed to it for crossover. Selection puts the members to be crossed over in order such that
    //a member of the selected population will be crossed over with the next member in the population in pairs.
    // i.e : [crossover A][crossover A]*[crossover B][crossover B]*...*[crossover N][crossover N].
    //Roulette Wheel selection is a preferred method in Genetic Algorithms because though the members with the best
    //fitness have a greater chance of being selected for crossover, the poorer members still have some chance which is
    //important for retaining as much information in the population as possible as well as to effectively explore the
    //search space.
    public ArrayList<NetworkParameterEncoding> select(ArrayList<NetworkParameterEncoding> population) {

        ArrayList<NetworkParameterEncoding> selectedPopulation = new ArrayList();

        double fitnessSumOfPopulation = 0;
        for (NetworkParameterEncoding enc : population)
            fitnessSumOfPopulation += enc.fitness;
        
        for (int i = 0; i < population.size() * 2; i++) { //Notice the doubling of the population. Two make an N numbered child generation N * 2 parents are needed...
            double randomFitness = u.random.randomDoubleInARange(0, fitnessSumOfPopulation);
            double localFitnessSum = 0;
            for (int j = 0; j < population.size(); j++) {
                localFitnessSum += population.get(j).fitness;
                if(localFitnessSum > randomFitness){
                    selectedPopulation.add(population.get(j));
                    break;
                }
            }
        }
        
        return selectedPopulation;
    }

    //Crossover occurs based on the specified crossover rate. Crossover rates are meant to be high and usually always
    //higher than mutation rates. Some researched-backed values are .8 - .99. Crossover is representative of the exploratory
    //element of the algorithm. It seeks to deviate enough from the current best solutions in order to find 'distant' solutions
    //that may be better. It does this by recombinating the encodings of the population of solutions with each other. Solutions
    //with a higher fitness have a commensurately better chance of passing on their information. This logic is carried out in the
    //select routine. However, there is still a chance that a better solution lies 'closer' to a current poor one. The GA takes
    //this into account by recombinating solutions with each other. This methodology allows these properties: a graceful approch
    //to a better solution, consideration that a poorer solution may contain effective components, consideration that it is
    //logical that better solutions working together will probably result in an even better solution, extreme explorations should
    //be mitigated, and exploration should be made on logical (good fitness) bases. Note that the population size passed to this
    //routine should be exactly double that of the working population because the generation of N childred requires N * 2 parents.
    //Also, the exact method of crossover is still up for debate: random chromasome selection vs. halving. To compromise we just
    //flip a coin here.
    public ArrayList<NetworkParameterEncoding> crossover(ArrayList<NetworkParameterEncoding> doubleSizePop) {

        ArrayList<NetworkParameterEncoding> newPopulation = new ArrayList<>();

        for (int i = 0; i < doubleSizePop.size(); i++) {
            NetworkParameterEncoding newMember;
            NetworkParameterEncoding mem1 = doubleSizePop.get(i);
            i++;
            NetworkParameterEncoding mem2 = doubleSizePop.get(i);
            
            if(Math.random() <= crossoverRate){
                //first make a clone of member 1 (or member 2 it doesn't matter)
                newMember = mem1.copyEncoding();
                
                //now create the new member's encoding by crossover
                switch(crossoverMethodID){
                    case HALVING:
                        switch(componentsToEvolveID){
                            case INPUTS:
                                newMember.inputVector = u.listManipulator.combineFirstHalfOfVAndSecondHalfOfB(mem1.inputVector, mem2.inputVector);
                                break;
                            case WEIGHTS:
                                newMember.inOrderWeights = u.listManipulator.combineFirstHalfOfVAndSecondHalfOfB(mem1.inOrderWeights, mem2.inOrderWeights);
                                break;
                            case BOTH:
                                newMember.inputVector = u.listManipulator.combineFirstHalfOfVAndSecondHalfOfB(mem1.inputVector, mem2.inputVector);
                                newMember.inOrderWeights = u.listManipulator.combineFirstHalfOfVAndSecondHalfOfB(mem1.inOrderWeights, mem2.inOrderWeights);
                                break;
                        }
                        break;
                    case RANDOM_ELEMENT:
                        switch(componentsToEvolveID){
                            case INPUTS:
                                newMember.inputVector = u.listManipulator.createListRandomlyFromTwoLists(newMember.inputVector.size(), mem1.inputVector, mem2.inputVector);
                                break;
                            case WEIGHTS:
                                newMember.inOrderWeights = u.listManipulator.createListRandomlyFromTwoLists(newMember.inOrderWeights.size(), mem1.inOrderWeights, mem2.inOrderWeights);
                                break;
                            case BOTH:
                                newMember.inputVector = u.listManipulator.createListRandomlyFromTwoLists(newMember.inputVector.size(), mem1.inputVector, mem2.inputVector);
                                newMember.inOrderWeights = u.listManipulator.createListRandomlyFromTwoLists(newMember.inOrderWeights.size(), mem1.inOrderWeights, mem2.inOrderWeights);
                                break;
                        }
                        break;
                    case RANDOM_METHOD:
                        //Halving
                        if(u.random.randomIntInARange(0, 1) == 1){
                            switch(componentsToEvolveID){
                                case INPUTS:
                                    newMember.inputVector = u.listManipulator.combineFirstHalfOfVAndSecondHalfOfB(mem1.inputVector, mem2.inputVector);
                                    break;
                                case WEIGHTS:
                                    newMember.inOrderWeights = u.listManipulator.combineFirstHalfOfVAndSecondHalfOfB(mem1.inOrderWeights, mem2.inOrderWeights);
                                    break;
                                case BOTH:
                                    newMember.inputVector = u.listManipulator.combineFirstHalfOfVAndSecondHalfOfB(mem1.inputVector, mem2.inputVector);
                                    newMember.inOrderWeights = u.listManipulator.combineFirstHalfOfVAndSecondHalfOfB(mem1.inOrderWeights, mem2.inOrderWeights);
                                    break;
                            }
                        //Random Element selection
                        } else{
                            switch(componentsToEvolveID){
                                case INPUTS:
                                    newMember.inputVector = u.listManipulator.createListRandomlyFromTwoLists(newMember.inputVector.size(), mem1.inputVector, mem2.inputVector);
                                    break;
                                case WEIGHTS:
                                    newMember.inOrderWeights = u.listManipulator.createListRandomlyFromTwoLists(newMember.inOrderWeights.size(), mem1.inOrderWeights, mem2.inOrderWeights);
                                    break;
                                case BOTH:
                                    newMember.inputVector = u.listManipulator.createListRandomlyFromTwoLists(newMember.inputVector.size(), mem1.inputVector, mem2.inputVector);
                                    newMember.inOrderWeights = u.listManipulator.createListRandomlyFromTwoLists(newMember.inOrderWeights.size(), mem1.inOrderWeights, mem2.inOrderWeights);
                                    break;
                            }
                        }
                        break;
                }
            }
            else{
                newMember = mem1.copyEncoding();
            }

            newPopulation.add(newMember);
        }

        return newPopulation;
    }

    //Mutation occurs based opon the specified mutation rate. It serves to exploit the
    //solution members of the population by exploring the search space that is very local
    //to the member being mutated. In this implementation, mutation occurs at the same rate
    //globally to the popluation as it does locally to the member. If the member is
    //selected to be mutated then the mutation the member undergoes is still dictated
    //by the same rate. For a selected member to mutate: the member's encoding N units
    //wide will be mutated with a probability of 1 - (1 - MutationRate). Mutation rates
    //are generally always smaller than crossover rates and usually very, very small.
    //Some researched-backed values are .01 to .1 .
    public ArrayList<NetworkParameterEncoding> mutate(ArrayList<NetworkParameterEncoding> population) {

        for(NetworkParameterEncoding encoding : population)
            encoding.mutateEncoding(mutationRate, mutationProtocol, componentsToEvolveID);

        return population;
    }
    
    //Spontaneous Death should occur VERY sparsely. It helps to keep a robustly diverse genetic pool.
    //We kill off the weakest 30% (a value shown by research to be effective) and replace the "dead"
    //solutions with random ones. Essentially we get tired of having to account for the weakest
    //solutions in the population so we replace them with random solutions that may be better
    //or that will eventually contribute to a better global solution.
    public ArrayList<NetworkParameterEncoding> spontaneousDeath(ArrayList<NetworkParameterEncoding> population){

        if(Math.random() < spontaneousDeathRate){
            int numberToDie = (int)(population.size() * .3);
            int populationIndex = population.size() - 1;
            for (int i = 0; i < numberToDie; i++){
                population.set(populationIndex, new NetworkParameterEncoding().generateRandomEncoding(inputEncoding, lens.getNetwork().getNetworkWeightCount()));
                populationIndex--;
            }
        }

        return population;
    }
    
    // **Note on the following elitism functions: elitism should be used sparingly so as to retain genetic diversity.
    //                                            It should constitute a very small proportion of the population.
    
    //On each generation we remember the best solutions in the population by saving them. This is called
    //elitism and serves to continually exploit the best solutions to the problem we have.
    public ArrayList<NetworkParameterEncoding> saveTheElite(ArrayList<NetworkParameterEncoding> population){

        if(elitismOn){
            elites.clear();
            for (int i = 0; i < numberOfEliteToSave; i++)
                elites.add(population.get(i).copyEncoding());
        }

        return population;
    }
    
    //On the end of each generations we replace the weakest solutions on the population by
    //inserting the elites of the generation in the weak's place.
    public ArrayList<NetworkParameterEncoding> injectElites(ArrayList<NetworkParameterEncoding> population){

        if(elitismOn && !elites.isEmpty()){
            int populationIndex = population.size() - 1;
            for (int i = 0; i < elites.size(); i++){
                population.set(populationIndex, elites.get(i));
                populationIndex--;
            }
        }

        return population;
    }

    public ArrayList<NetworkParameterEncoding> getPopulation(){
        return this.population;
    }

    public void setPopulation(ArrayList<NetworkParameterEncoding> population){
        this.population = population;
    }
    
    public ArrayList<NetworkParameterEncoding> sortPopulation(ArrayList<NetworkParameterEncoding> population) {
        return mergeSort(population);
    }
    
    //An attempt to implement multithreaded merge sort
    private ArrayList<NetworkParameterEncoding> sortPopulationConcurrently(ArrayList<NetworkParameterEncoding> population){
        
        //this only supports even numbered threaded machines (most all machines)
        //so return if the cores arent even numbered
        if(cores % 2 != 0){
            sortPopulation(population);
            return null;
        }
        
        //the check for non-perfect division of the population into the cores has been take care of in the evaluation routine
        
        if(sorterService == null)
            sorterService = Executors.newFixedThreadPool(cores);
        
        //split the population
        int splitMagnitude = population.size() / cores;
        ArrayList<Sorter> sorters = new ArrayList<>();
        for (int i = 0; i < population.size(); i += splitMagnitude)
            sorters.add(new Sorter(new ArrayList(population.subList(i, i + splitMagnitude))));
        
        //evaluate the population
        try{
            sorterService.invokeAll(sorters);
        }catch(Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        
        if(mergerService == null)
            mergerService = Executors.newFixedThreadPool(cores);
        
        //delegate mergers for each in-order two tuple in the set of sorters
        ArrayList<Merger> mergers = new ArrayList<>();
        for (int i = 0; i < sorters.size(); i += 2)
            mergers.add(new Merger(sorters.get(i).getSorted(), sorters.get(i + 1).getSorted()));
        
        //merge the sorted and delegated subpopulations
        try{
            mergerService.invokeAll(mergers);
        }catch(Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        
        //reconstitute the population
        population.clear();
        for(Merger m : mergers)
            population.addAll(m.getMerged());

        return population;
    }
    
    private ArrayList<NetworkParameterEncoding> mergeSort(ArrayList<NetworkParameterEncoding> population){
        if(population.size() == 1)
            return population;
        
        ArrayList<NetworkParameterEncoding> a = new ArrayList(population.subList(0, (population.size() / 2)));
        ArrayList<NetworkParameterEncoding> b = new ArrayList(population.subList((population.size() / 2), population.size()));
        
        a = mergeSort(a);
        b = mergeSort(b);
        
        return merge(a, b);
    }

    //This implementation of Merge sort may be slightly slower than it needs to be because I am always removing at the 0th position...
    private ArrayList<NetworkParameterEncoding> merge(ArrayList<NetworkParameterEncoding> a, ArrayList<NetworkParameterEncoding> b){
        ArrayList<NetworkParameterEncoding> merged = new ArrayList<>();
        
        while(!a.isEmpty() && !b.isEmpty())
            if(a.get(0).fitness > b.get(0).fitness)
                merged.add(b.remove(0));
            else
                merged.add(a.remove(0));
        
        while(!a.isEmpty())
            merged.add(a.remove(0));
        
        while(!b.isEmpty())
            merged.add(b.remove(0));
        
        return merged;
    }
    
    private class Evaluator implements Callable<Object>{

        private ArrayList<NetworkParameterEncoding> toEvaluate = new ArrayList<>();
        private ArrayList<NetworkParameterEncoding> evaluated = new ArrayList<>();

        public Evaluator(ArrayList<NetworkParameterEncoding> toEvaluate){
            this.toEvaluate = toEvaluate;
        }
        
        @Override
        public Object call(){
            evaluated = evaluatePopulationOnSingleThread(toEvaluate);
            return null;
        }

        public ArrayList<NetworkParameterEncoding> getEvaluated(){
            return evaluated;
        }  
        
    }
    
    private class Sorter implements Callable<Object>{
        
        private ArrayList<NetworkParameterEncoding> toSort = new ArrayList<>();
        private ArrayList<NetworkParameterEncoding> sorted = new ArrayList<>();
        
        public Sorter(ArrayList<NetworkParameterEncoding> toSort){
            this.toSort = toSort;
        }
        
        @Override
        public Object call(){
            sorted = mergeSort(toSort);
            return null;
        }
        
        public ArrayList<NetworkParameterEncoding> getSorted(){
            return sorted;
        }
    }
    
    private class Merger implements Callable<Object>{
        
        private ArrayList<NetworkParameterEncoding> mergeSetA = new ArrayList<>();
        private ArrayList<NetworkParameterEncoding> mergeSetB = new ArrayList<>();
        
        private ArrayList<NetworkParameterEncoding> merged = new ArrayList<>();
        
        public Merger(ArrayList<NetworkParameterEncoding> a, ArrayList<NetworkParameterEncoding> b){
            mergeSetA = a;
            mergeSetB = b;
        }
        
        @Override
        public Object call(){
            merged = merge(mergeSetA, mergeSetB);
            return null;
        }
        
        public ArrayList<NetworkParameterEncoding> getMerged(){
            return merged;
        }
    }
    
    public class NetworkThreadHandler extends ThreadLocal<NeuralNetwork>{
        
        private final NeuralNetwork network;
        
        public NetworkThreadHandler(NeuralNetwork net){
            this.network = net;
        }
        
        @Override
        protected NeuralNetwork initialValue(){
            return network.copyNetwork();
        }
    }
    
}
  
