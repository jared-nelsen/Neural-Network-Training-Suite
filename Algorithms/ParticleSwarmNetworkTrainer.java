
package Algorithms;

import Lenses.MicroLens;
import Structures.NetworkParameterEncoding;
import Utilities.Utilities;

import java.util.ArrayList;

public class ParticleSwarmNetworkTrainer {
    
    Utilities u = new Utilities();
    
    public long globalBestFitness = Long.MAX_VALUE;
    public ArrayList<ParticleSwarmOptimizer> swarms = new ArrayList<>();
    
    public ParticleSwarmNetworkTrainer(){
        u.dialoger.printMessage("Particle swarm Optimization has been chosen as the training algorithm.");
    }
    
    public NetworkParameterEncoding trainNetwork(MicroLens lense, NetworkParameterEncoding encoding){
        
        u.dialoger.printLn("Initializing Swarms");
        for (int i = 0; i < lense.getNetwork().getNetworkWeightCount(); i++)
            swarms.add(new ParticleSwarmOptimizer());
        u.dialoger.printLn("Swarms Initialized");
        
        u.timer.start();
        int swarmIterations = 0;
        while(globalBestFitness > 0){
            
            //Optimize each weight using its own Particle Swarm
            int networkWeightIndex = 0;
            for(ParticleSwarmOptimizer swarm : swarms)
                swarm.swarm(lense, encoding, networkWeightIndex++); //trying the ++
            
            //now evaluate the network using the new encoding for a global fitness
            long evaluatedFitness = lense.getNetwork().evaluateNetworkParameterEncoding(encoding, lense, null); //pass in null for the ga to use the lense's network reference
            if(evaluatedFitness < globalBestFitness){
                System.out.println("Best fitness = " + evaluatedFitness + " Found at second " + u.timer.getSeconds() + " and Swarm Iteration " + swarmIterations);
                globalBestFitness = evaluatedFitness;
            }
            swarmIterations++;
        }
        
        return encoding;
    }
    
    private class ParticleSwarmOptimizer {
        
        private Particle gBest = new Particle();
        private double c1 = 2.05;
        private double c2 = 2.05;
        private double learningRate = .7298;

        private int numberOfParticles = 100;
        private ArrayList<Particle> particles = new ArrayList<>();

        private int numberOfIterations = 10;

        public ParticleSwarmOptimizer(){
            for (int i = 0; i < numberOfParticles; i++)
                particles.add(new Particle());
        }

        private void swarm(MicroLens lense, NetworkParameterEncoding encoding, int swarmWeightIndex){

            int iterations = 0;
            do{

                for(Particle particle : particles){
                    //set the weight in the encoding to the particles current value
                    ArrayList<Float> weights = encoding.getWeightEncoding();
                    weights.set(swarmWeightIndex, particle.getCurrent());
                    encoding.setWeightEncoding(weights);
                    
//                    System.out.println("");
//                    for(Double d : weights){
//                        System.out.print("[" + d + "]");
//                    }
                    
                    //evaluate the network based on the encoding
                    long fitness = lense.getNetwork().evaluateNetworkParameterEncoding(encoding, lense, null);
                    particle.setFitness(fitness);
                    if(fitness < particle.getFitness())
                        particle.setpBest(particle.getCurrent());
                }

                for(Particle particle : particles)
                    if(particle.getFitness() < gBest.getFitness())
                        gBest = particle;

                for(Particle particle : particles){
                    //This is patchy and should be redone
                    float velocity = particle.getVelocity() + (float)c1 * (float)Math.random() * (particle.getpBest() - particle.getCurrent()) +
                            (float)c2 * (float)Math.random() * (gBest.getCurrent() - particle.getCurrent());
                    
                    velocity = velocity * (float)learningRate;
                    
                    particle.setVelocity(velocity);
                    particle.setCurrent(particle.getCurrent() + particle.getVelocity());
                }

                iterations++;
            }while(iterations < numberOfIterations);

            //set the weight to the gBest
            ArrayList<Float> weights = encoding.getWeightEncoding();
                    weights.set(swarmWeightIndex, gBest.getCurrent());
                    encoding.setWeightEncoding(weights);
        }

        private class Particle implements Cloneable {

            private Long fitness = Long.MAX_VALUE;
            private Float current = u.random.randomFloatInARange(-1, 1);
            private Float pBest = current;
            private Float velocity = u.random.randomFloatInARange(-1, 1);

            private Particle(){

            }

            public Long getFitness() {
                return fitness;
            }

            public void setFitness(Long fitness) {
                this.fitness = fitness;
            }

            public Float getpBest() {
                return pBest;
            }

            public void setpBest(Float pBest) {
                this.pBest = pBest;
            }

            public Float getVelocity() {
                return velocity;
            }

            public void setVelocity(Float velocity) {
                this.velocity = velocity;
            }

            public Float getCurrent() {
                return current;
            }

            public void setCurrent(Float current) {
                this.current = current;
            }

        }

    }
    
}
