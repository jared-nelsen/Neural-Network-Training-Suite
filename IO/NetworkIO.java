package IO;

import Algorithms.GeneticAlgorithm;
import Lenses.MicroLens;
import Structures.NeuralNetwork;
import Utilities.Utilities;
import org.apache.commons.lang3.StringUtils;

import java.util.ArrayList;
import java.util.LinkedList;

/**
 *
 * @author <Jared Nelsen>
 */
public class NetworkIO {

    public Utilities u = new Utilities();

//--------------------------------------------------------------------------------------------------
    //This output matrix O will constitute F frames by P pixels by 4 Integer Values V defined by:
    
    //[Frame F0],[Frame F1],...,[Frame FF]
    
        //Where each Frame F is defined by:
    
        //[Pixel 00],[Pixel 11],...,[Pixel PP]
    
            //Where each Pixel P is defined by:
    
            //[Value V0],[Value V1],[Value V2],[Value V3]

    private ArrayList<ArrayList<ArrayList<Integer>>> outputData = new ArrayList<>();
    //      FrameSet<PixelSet<ValueSet<Integer Value>>>
    
    //Each MicroLens is responsible for one value V in a Vector of Pixels called PV. The set of PV is in-order
    //with P[]. PV is defined by:
    
        //[PV 0],[PV1],...,[PVPV]
    
            //Where each Pixel vector PV is defined by:
    
            //[PV PVO : P[P0 : F[F0,F1,...,FF]],[PV PV1 : P[P1 : F[F0,F1,...,FF]], ...,[PV PVPV : P[PP : F[F0,F1,...,FF]]
    
    //So to delegate output values to a MicroLens we take a cross section of the frame vector FV at each Pixel P in each F
    //and take a cross section of this Pixel Vector PV at each Integer Value
    //Ex: V[MicroLens] = PV0 :: F[0,1,...,FF] :: F[F[P0]]
    
    //Thus in the case of 32 bit output type MicroLenses there a 4 MicroLenses per Pixel
    //So the number of MicroLenses in a parsed video file is:
    // #ML = F * P * 4
    private int microLenseCount = 0;
    private int frameCount = 0;
    private int pixelsPerFrame = 0;
    private int microLensesPerPixel = 1;
//--------------------------------------------------------------------------------------------------
    
    public NetworkIO(String parentDirectory, String fileName) {
        u.setCurrentFile(u.addTextFile(parentDirectory, fileName));
    }
    
    public void generateRandomizedVideoFile(){
        
//        int perFrame = u.dialoger.promptForInt("How many pixels per frame?");
//        int numFrames = u.dialoger.promptForInt("How many frames?");
        int perFrame = 1;
        int numFrames = 10;
        
        u.dialoger.printLn("Generating Random Pixels");
        
        //ArrayList<String> frames = new ArrayList<>();
        
        LinkedList<String> pixelStrings = new LinkedList<>();
        
        for (int i = 0; i < numFrames; i++) {
            StringBuilder pixels = new StringBuilder();
            for (int j = 0; j < perFrame; j++) {
                StringBuilder pixel = new StringBuilder();
                pixel.append("[");
                for (int k = 0; k < 4; k++) {
                    pixel.append(u.random.randomIntInARange(0, 255));
                    pixel.append(",");
                }
                pixel.deleteCharAt(pixel.lastIndexOf(","));
                pixel.append("]");
                pixels.append(pixel);
            }
            pixelStrings.add(pixels.toString());
            //u.getCurrentFile().writeLine(pixels.toString());
            //frames.add(pixels.toString());
        }
        
        u.getCurrentFile().writeLines(pixelStrings);
        
        u.dialoger.printMessage(numFrames + " frames of " + perFrame + " pixels have been randomly generated and written to the current file");
    }
    
    public void parseVideoFile(){
        
        outputData.clear();
//        int frameCount = 0;
//        int pixelsPerFrameCount = 0;
//        for(String frame : u.getCurrentFile().getEntireListOfLines()){
//            ArrayList<ArrayList<Integer>> framePixelVector = new ArrayList<>();
//            while(!frame.isEmpty()){
//                ArrayList<Integer> pixel = new ArrayList<>();
//                int split = pixel.indexOf(']');
//                if(split == -1){
//                    return;
//                }
//                String pixelString = frame.substring(0, split - 1);
//                frame = frame.substring(split + 2, frame.length());
//                pixelString = pixelString.substring(2, pixelString.length());
//                String[] values = new String[4];
//                int valueIndex = 0;
//                while(!pixelString.isEmpty()){
//                    int subSplit = pixelString.indexOf(',');
//                    values[valueIndex] = pixelString.substring(0, subSplit - 1);
//                    pixelString = pixelString.substring(2, pixelString.length() - 1);
//                    valueIndex++;
//                }
//                for (String value : values) {
//                    pixel.add(Integer.parseInt(value));
//                }
//                framePixelVector.add(pixel);
//                pixelsPerFrameCount = framePixelVector.size();
//            }
//            outputData.add(framePixelVector);
//            frameCount++;
//        }
        
        for(String frame : u.getCurrentFile().getEntireListOfLines()){
            String[] pixels = StringUtils.substringsBetween(frame, "[", "]");
            ArrayList<ArrayList<Integer>> pixelSet = new ArrayList<>();
            for(String pixel : pixels){
                ArrayList<Integer> pixelValueSet = new ArrayList<>();
                String[] pixelValueArray = StringUtils.split(pixel, ',');
                for(String value : pixelValueArray){
                    pixelValueSet.add(Integer.parseInt(value));
                }
                pixelSet.add(pixelValueSet);
            }
            this.outputData.add(pixelSet);
        }
        
        frameCount = outputData.size();
        pixelsPerFrame = outputData.get(0).size();
        microLenseCount = frameCount * pixelsPerFrame * 4;
    }
    
    public ArrayList<MicroLens> constructOutputMicroLenses(MicroLens.ReconciliationAlgorithm algorithm,
                                                           MicroLens.LenseType lenseType,
                                                           GeneticAlgorithm.EvaluationProtocol evaluationProtocol,
                                                           GeneticAlgorithm.ComponentsToEvolve componentsToEvolve,
                                                           GeneticAlgorithm.MutationProtocol mutationProtocol,
                                                           NeuralNetwork.NetworkOutputType networkOutputType){
        ArrayList<MicroLens> lenses = new ArrayList<>();
        
        //At this point, in order to facilitate the ease of switching between the integer RGB type model to the binary model
        //I will still generate the pixels in an integer value form but will switch them to binary on the fly, making sure
        //to always use 32 bit output networks to do so
        
        //parse the pixels in RGBA Integer format
        for (int i = 0; i < pixelsPerFrame; i++) {
            for (int j = 0; j < microLensesPerPixel; j++) {
                MicroLens lense = new MicroLens(algorithm, lenseType, evaluationProtocol, componentsToEvolve, mutationProtocol, networkOutputType);
                ArrayList<int[]> pixelOutputs = new ArrayList<>();
                for(ArrayList<ArrayList<Integer>> pixel : outputData){
                    for(ArrayList<Integer> pixelValues : pixel){
                        int[] primitiveValues = new int[pixelValues.size()];
                        int primitiveIndex = 0;
                        for(Integer value : pixelValues){
                            primitiveValues[primitiveIndex] = value;
                            primitiveIndex++;
                        }
                        pixelOutputs.add(primitiveValues);
                    }
                    lense.setRGBOutputs(pixelOutputs);
                }
                lenses.add(lense);
            }
        }
        
        //and convert the lenses to binary 32 bit type if we are using the binary model
        if(networkOutputType == NeuralNetwork.NetworkOutputType.BINARY_APPROXIMATION){
            ArrayList<MicroLens> binaryLenses = new ArrayList<>();
            //for each RGBA lense that we have
            for(MicroLens lense : lenses){
                //designate a new output buffer
                ArrayList<Integer> binaryOutput = new ArrayList<>();
                //and for each RGBA value associated with the lense (which should at this point be only 1 item [R,G,B,A] but this SHOULD be flexible
                ArrayList<int[]> RGBAs = lense.getRGBOutputs();
                for(int[] rgbaValues : RGBAs){
                    String binaryBuffer = "";
                    for(int value : rgbaValues)
                        binaryBuffer += Integer.toBinaryString(value);
                    for (int i = 0; i < binaryBuffer.length(); i++)
                        if(binaryBuffer.charAt(i) == '1')
                            binaryOutput.add(1);
                        else
                            binaryOutput.add(0);
                }
                //create the equivalent 32 bit microlense
                MicroLens thirtyTwoBitLense = new MicroLens(algorithm, lenseType, evaluationProtocol, componentsToEvolve, mutationProtocol, networkOutputType);
                //and set its outputs to the derived binary
                thirtyTwoBitLense.setBinaryOutputs(binaryOutput);
                //and record it
                binaryLenses.add(thirtyTwoBitLense);
            }
            //now that we are done converting the lenses from RGBA to Binary we can replace the RGBA lenses with the Binary ones to return
            lenses = binaryLenses;
        }
        
        return lenses;
    }
    
}
