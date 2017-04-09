package Main;

import Algorithms.GeneticAlgorithm;
import IO.NetworkIO;
import Lenses.MicroLens;
import Structures.NeuralNetwork;

import java.util.ArrayList;
import java.util.Date;


/**
 *
 * @author jared
 */
public class Driver {
    
    public static void main(String[] args) {
        //KernelManager.setKernelManager(KernelManagers.SEQUENTIAL_ONLY);
        new Driver();
        System.exit(0); //Active threads from executor services prevent shutdown
    }
    
    private Driver(){

        //NetworkIO io = new NetworkIO("/home/jared/Desktop/Sync/Outputs/", new Date().toString().replace(':','-'));
        NetworkIO io = new NetworkIO("C:\\Users\\Jared\\Desktop\\Sync\\Outputs\\", new Date().toString().replace(':','-'));
        io.generateRandomizedVideoFile();
        io.parseVideoFile();
        ArrayList<MicroLens> outputLenses = io.constructOutputMicroLenses(MicroLens.ReconciliationAlgorithm.GENETIC_ALGORTIHM,
                                                                           MicroLens.LenseType.THIRTY_TWO,
                                                                           GeneticAlgorithm.EvaluationProtocol.MULTI_THREAD_CPU,
                                                                           GeneticAlgorithm.ComponentsToEvolve.WEIGHTS,
                                                                           GeneticAlgorithm.MutationProtocol.VALUE_JUMPING,
                                                                           NeuralNetwork.NetworkOutputType.INTEGER_RGB_VALUE);
        for(MicroLens lense : outputLenses)
            lense.reconcileInputsWithOutputs();

    }

    
}
