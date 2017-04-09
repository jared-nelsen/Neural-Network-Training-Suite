package Structures;


import Lenses.MicroLens;

import java.util.ArrayList;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author jared
 */
public class LayerMap {
    
    private ArrayList<Layer> layers;
    
    public LayerMap(){
        layers = new ArrayList<>();
    }
    
    public ArrayList<Layer> getLayers(){
        return layers;
    }
    
    public void addLense(MicroLens lense){
        layers.get(0).addLense(lense);
    }
    
    public void addLayer(){
        if(layers.isEmpty())
            layers.add(new Layer());
        else
            layers.add(0, new Layer());
    }
    
    public int getNumberOfLayers(){
        return layers.size();
    }
    
    public class Layer{
        
        private ArrayList<Integer> I;
        private ArrayList<MicroLens> lenses;
        private ArrayList<Integer> O; //make sure this is right after writing the read in code
        
        public Layer(){
            this.lenses = new ArrayList<>();
        }
        
        public void setInputVector(ArrayList<Integer> input){
            I = input;
        }
        
        public void setOutputVector(ArrayList<Integer> output){
            O = output;
        }
        
        public void addLense(MicroLens lense){
            lenses.add(lense);
        }
        
        public ArrayList<MicroLens> getLenses(){
            return lenses;
        }       
        
    }
    
}
