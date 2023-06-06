package Utilities;

import java.io.*;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Utilities {

    public ListManipulator listManipulator = new ListManipulator();
    public Pauser pauser = new Pauser();
    public Timer timer = new Timer();
    public TextInterface dialoger = new TextInterface(1, 3);
    private HashMap<String, TextFileReaderAndWriter> files = new HashMap<String, TextFileReaderAndWriter>();
    private TextFileReaderAndWriter currentFile = null;
    public Random random = new Random();

    public Utilities() {

    }

    public class ListManipulator {

        public ListManipulator() {

        }

        public ArrayList combineFirstHalfOfVAndSecondHalfOfB(ArrayList a, ArrayList b) {
            ArrayList vec = new ArrayList();

            int size1 = a.size() / 2;
            int size2 = b.size() / 2;
            for (int i = 0; i < size1; i++) {
                vec.add(a.get(i));
            }
            for (int i = size2; i < b.size(); i++) {
                vec.add(b.get(i));
            }

            return vec;
        }

        public ArrayList createListRandomlyFromTwoLists(int newListSize, ArrayList v1, ArrayList v2) {
            ArrayList newList = new ArrayList();

            for (int i = 0; i < newListSize; i++) {
                Object x = null;
                if (i % 2 == 0) {
                    x = v1.get(random.randomIntInARange(0, v1.size() - 1));
                } else {
                    x = v2.get(random.randomIntInARange(0, v2.size() - 1));
                }
                newList.add(x);
            }

            return newList;
        }

        public float[] arrayListToFloatArray(ArrayList<Float> list){
            float[] arr = new float[list.size()];

            int i = 0;
            for(Float f : list){
                arr[i] = f;
                i++;
            }

            return arr;
        }

        public double[] arrayListToDoubleArray(ArrayList<Double> list) {
            double[] arr = new double[list.size()];

            int i = 0;
            for (Double d : list) {
                arr[i] = d;
                i++;
            }

            return arr;
        }

        public int[] arrayListToIntArray(ArrayList<Integer> list){
            int[] arr = new int[list.size()];

            int i = 0;
            for(Integer x : list){
                arr[i] = x;
                i++;
            }

            return arr;
        }

        public float[] intArrayToFloatArray(int[] list){
            float[] arr = new float[list.length];

            for (int i = 0; i < list.length; i++)
                arr[i] = list[i];

            return arr;
        }

        public String arrayListOfIntegersToString(ArrayList<Integer> list) {
            StringBuilder b = new StringBuilder();

            for (Integer i : list)
                b.append(i);

            return new String(b);
        }
        
        public String intArrayToString(int[] arr){
            StringBuilder b = new StringBuilder();
            
            for (int i = 0; i < arr.length; i++)
                b.append(arr[i]);
            
            return new String(b);
        }
        
        public int[] subIntArray(int[] arr, int begin, int end){
            ArrayList<Integer> ls = new ArrayList<>();
            
            for (int i = begin; i < end; i++) 
                ls.add(arr[i]);
            
            int[] sub = new int[ls.size()];
            for (int i = 0; i < sub.length; i++)
                sub[i] = ls.get(i);
            
            return sub;
        }

        public double[] subDoubleArray(double[] arr, int begin, int end){
            ArrayList<Double> ls = new ArrayList<>();

            for (int i = begin; i < end; i++)
                ls.add(arr[i]);

            double[] sub = new double[ls.size()];
            for (int i = 0; i < sub.length; i++)
                sub[i] = ls.get(i);

            return sub;
        }

        public float[] subFloatArray(float[] arr, int begin, int end){
            ArrayList<Float> ls = new ArrayList<>();

            for (int i = begin; i < end; i++)
                ls.add(arr[i]);

            float[] sub = new float[ls.size()];
            for (int i = 0; i < sub.length; i++)
                sub[i] = ls.get(i);

            return sub;
        }
        
        public int binaryToInteger(String binary) {
            char[] numbers = binary.toCharArray();
            int result = 0;
            for(int i = numbers.length - 1; i >= 0; i--)
                if(numbers[i] == '1')
                    result += Math.pow(2, (numbers.length - i - 1));
            return result;
        }

        public ArrayList<Integer> copyIntegerArrayList(ArrayList<Integer> toCopy){
            ArrayList<Integer> copy = new ArrayList<>();
            for(Integer i : toCopy)
                copy.add(i);
            return copy;
        }
    }

    public class Pauser {

        public Pauser() {

        }

        public void waitXSeconds(int x) {
            int count = 0;
            while (count < x) {
                pauseXSeconds(1);
                System.out.print(".");
                count++;
            }
            System.out.println("\n");
        }

        public void pauseXSeconds(int x) {
            long scalar = x * 1000;
            long currentTime = System.currentTimeMillis();
            long waitToTime = currentTime + scalar;

            while (System.currentTimeMillis() < waitToTime) {
                //wait
            }
        }

    }

    public class Timer {

        private long startTime;
        private Date startDate;
        private long endTime;
        private Date endDate;

        public Timer() {

        }

        public void start() {
            startTime = System.nanoTime();
        }

        public void stop() {
            endTime = System.nanoTime();
        }

        public long getMillis() {
            stop();
            return TimeUnit.SECONDS.convert((endTime - startTime), TimeUnit.MILLISECONDS);
        }
        
        public long getSeconds(){
            stop();
            return TimeUnit.SECONDS.convert((endTime - startTime), TimeUnit.NANOSECONDS);
        }

        public String getMinutes() {
            endTime = System.nanoTime();
            long delta = endTime - startTime;
            //convert to seconds
            delta = delta / 1000000 / 1000;
            double time = delta / 60;
            return new Double(time).toString();
        }

        public String getResult() {
            long delta = endTime - startTime;
            startDate = new Date(startTime);
            endDate = new Date(endTime);
            StringBuilder builder = new StringBuilder();
            builder.append("Start: ");
            builder.append(startDate.toString());
            builder.append("\nEnd: ");
            builder.append(endDate.toString());
            builder.append("\nFor a total of:\n");
            builder.append(delta + " nanoseconds or " + (delta /= 1000000) + " milliseconds or " + (delta /= 1000) + " seconds or \n"
                    + (delta /= 60) + " minutes or " + (delta /= 60) + " hours or " + (delta /= 12) + " days...");

            return new String(builder);
        }
    }

    public Timer getNewTimer(){
        return new Timer();
    }

    public class TextInterface {

        private KeyboardInputClass in = new KeyboardInputClass();
        private int spaceCount;
        private int pauseCount;

        public TextInterface(int spaceCount, int pauseCount) {
            this.spaceCount = spaceCount;
            this.pauseCount = pauseCount;
        }

        public void print(String message) {
            System.out.print(message);
        }
        
        public void printLn(String message){
            System.out.print(message + "...");
            System.out.println("");
        }

        public void printMessage(String message) {
            print(message);
            wait(pauseCount);
            newLines(spaceCount);
        }

        public void wait(int waitCount) {
            for (int i = 0; i < waitCount; i++) {
                System.out.print(".");
                pauser.pauseXSeconds(1);
            }
            newLines(1);
        }

        public void countDownFrom(int x) {
            newLines(1);
            for (int i = x; i >= 0; i--) {
                print(new Integer(i).toString() + "...");
                pauser.pauseXSeconds(1);
            }
            newLines(1);
        }

        public void newLines(int spaceCount) {
            for (int i = 0; i < spaceCount; i++) {
                System.out.println("");
            }
        }

        public boolean yes(String question) {
            String prompt = question + "\ny = Yes\nN = No";
            return in.getCharacter(true, 'y', "yn", 2, prompt) == 'y';
        }

        public boolean no(String question) {
            return yes(question) == false;
        }

        public int optionPrompt(String question, List<String> options) {
            String prompt = question;
            int count = 0;
            for (String op : options) {
                prompt = prompt.concat("\n" + count + " = " + op);
                count++;
            }
            return in.getInteger(true, 0, 0, options.size() - 1, prompt);
        }

        public char getChar() {
            return in.getCharacter(true, 'a', "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890`~!@#$%^&*()-=_+[]{}\\\"|:';,./<>?", 0, "Input a character:");
        }

        public char promptForChar(String prompt) {
            newLines(1);
            print(prompt);
            return getChar();
        }

        public int getInt() {
            return in.getInteger(true, 0, Integer.MIN_VALUE, Integer.MAX_VALUE, "\nInput an integer:");
        }

        public int promptForInt(String prompt) {
            print(prompt);
            return getInt();
        }

        public double getDouble() {
            return in.getDouble(true, 0, Double.MIN_VALUE, Double.MAX_VALUE, "\nInput a double:");
        }

        public double promptForDouble(String prompt) {
            newLines(1);
            print(prompt);
            return getDouble();
        }

        public String getString() {
            return in.getString("", "Input a String:");
        }

        public String promptForString(String prompt) {
            newLines(1);
            print(prompt);
            newLines(1);
            return getString();
        }

    }

    public TextFileReaderAndWriter getCurrentFile() {
        return currentFile;
    }

    public void setCurrentFile(TextFileReaderAndWriter file) {
        currentFile = file;
    }

    public TextFileReaderAndWriter addTextFile(String fullPath) {
        String parentPath = "";
        String name = "";
        if (!fullPath.endsWith(".txt")) {
            dialoger.printMessage("The file you passed in is not a text file!");
            dialoger.printMessage("Exiting");
            System.exit(0);
        } else {
            File f = new File(fullPath);
            parentPath = f.getParent();
            name = f.getName();
        }
        return addTextFile(parentPath, name);
    }

    public TextFileReaderAndWriter addTextFile(String parentPath, String fileName) {
        dialoger.printMessage(fileName + " added");
        TextFileReaderAndWriter writer = new TextFileReaderAndWriter(fileName, parentPath);
        files.put(fileName, writer);
        return writer;
    }

    public TextFileReaderAndWriter getTextFile(String fileName) {
        if (!files.containsKey(fileName)) {
            dialoger.printMessage("No record of a file with the name " + fileName);
            return null;
        }
        return files.get(fileName);
    }

    public ArrayList<String> getTextFileNames() {
        if (files.isEmpty()) {
            dialoger.printMessage("The file list is empty");
            return new ArrayList<>();
        } else {
            return new ArrayList<>(files.keySet());
        }
    }

    public void removeTextFile(String fileName) {
        if (files.containsKey(fileName)) {
            files.remove(fileName);
            dialoger.printMessage(fileName + " has been removed");
        } else {
            dialoger.printMessage("No record of a file with the name " + fileName);
        }
    }

    public void deleteTextFile(String fileName) {
        if (files.containsKey(fileName)) {
            if (files.get(fileName).deleteFile()) {
                dialoger.printMessage(fileName + " has been deleted");
            }
        } else {
            dialoger.printMessage("No record of a file with the name: " + fileName);
        }
    }

    public void clearTextFiles() {
        files.clear();
        dialoger.printMessage("Text files cleared...");
    }

    public class TextFileReaderAndWriter {

        private BufferedReader reader;
        private BufferedWriter writer;

        private ArrayList<String> listOfLines = new ArrayList();

        private String parentPathString;
        private String fileExtension = ".txt";
        private final String title;

        public TextFileReaderAndWriter(String title, String parentPath) {

//            if (!parentPath.endsWith("\\") && !parentPath.endsWith("/")) {
//                dialoger.printMessage("Warning! the file path passed in does not end with a '\\' or a /");
//                dialoger.printMessage("Exiting");
//                System.exit(0);
//            }

            this.title = title;
            this.parentPathString = parentPath;

            createIfDoesNotExist();

            try {
                readAndLoadListOfStrings();
            } catch (IOException i) {
                System.out.println("Error in readAndLoadListOfStrings() in TextFileReaderAndWriter constructor");
                i.printStackTrace();
            }

        }

        public String getTitle() {
            return title;
        }

        public String getCompletePath() {
            return parentPathString + title + fileExtension;
        }

        private void setReader() {
            try {
                this.reader = new BufferedReader(new FileReader(getCompletePath()));
            } catch (FileNotFoundException f) {
                System.out.println("Error in setting Reader for " + getCompletePath());
                File file = new File(getCompletePath());
                if (!file.exists()) {
                    System.out.println(getCompletePath() + " does not exist!");
                }
                f.printStackTrace();
            }
        }

        private void setWriter() {
            try {
                this.writer = new BufferedWriter(new FileWriter(getCompletePath()));
            } catch (IOException i) {
                System.out.println("Error in setting Writer for " + getCompletePath());
                File file = new File(getCompletePath());
                if (!file.exists()) {
                    System.out.println(getCompletePath() + " does not exist!");
                }
                i.printStackTrace();
            }
        }

        public boolean compileAndWriteIndexFile() {
            boolean success = true;

            this.setWriter();

            try {
                //clear file
                this.writer.write("");
                this.writer.flush();
                //write new contents
                for (String toWrite : this.listOfLines) {
                    this.writer.write(toWrite);
                    this.writer.newLine();
                }
            } catch (Exception e) {
                System.out.println("Failed to compile and write index\n");
                e.printStackTrace();
                success = false;
            } finally {
                try {
                    this.writer.flush();
                    this.writer.close();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            return success;
        }

        public void readAndLoadListOfStrings() throws IOException {
            String line = null;

            this.setReader();

            try {
                while ((line = this.reader.readLine()) != null) {
                    listOfLines.add(line);
                }
            } catch (Exception e) {
                System.out.println("Error in readAndLoadListOfStrings()");
                e.printStackTrace();
            } finally {
                reader.close();
            }

        }

        public void clearListOfLines() {
            this.listOfLines.clear();
            compileAndWriteIndexFile();
        }

        public String getLineAt(int position) {
            return this.listOfLines.get(position);
        }

        public ArrayList<String> getSubListOfLines(int startPosition, int endPosition) {
            ArrayList<String> listOfLinesToReturn = new ArrayList();

            for (int i = startPosition; i <= endPosition; i++) {
                listOfLinesToReturn.add(this.listOfLines.get(i));
            }

            return listOfLinesToReturn;
        }

        public ArrayList<String> getEntireListOfLines() {
            return this.listOfLines;
        }

        public ArrayList<String> getListOfLinesContainingSubString(String subString) {
            ArrayList<String> found = new ArrayList();
            for (String line : this.listOfLines) {
                if (line.contains(subString)) {
                    found.add(line);
                }
            }
            return found;
        }

        public int getLength() {
            return this.listOfLines.size();
        }

        public void deleteLine(int position) {
            this.listOfLines.remove(position);
            compileAndWriteIndexFile();
        }

        public void deleteListOfLines(int startPosition, int endPosition) {
            int iterations = endPosition - startPosition;
            for (int i = 0; i < iterations + 1; i++) {
                deleteLine(startPosition);
            }
        }

        public void writeLine(String toWrite) {
            this.listOfLines.add(toWrite);
            compileAndWriteIndexFile();
        }

        public void writeLines(List<String> listToWrite) {
            for(String s : listToWrite)
                this.listOfLines.add(s);
            compileAndWriteIndexFile();
        }

        public void writeLineAt(int position, String toWrite) {
            this.listOfLines.add(position, toWrite);
            compileAndWriteIndexFile();
        }

        public void writeLinesAt(int position, List<String> linesToWrite) {
            if (linesToWrite.isEmpty()) {
                return;
            }
            writeLine(linesToWrite.remove(0));
            writeLinesAt(position++, linesToWrite);
        }

        public void replaceAtLine(int position, String toWrite) {
            this.listOfLines.set(position, toWrite);
            compileAndWriteIndexFile();
        }

        //File Operations
        //***********************************************************************************************
        public void createIfDoesNotExist() {
            if (getCompletePath() == null) {
                System.out.println("The file path is null! Can not create the file...");
                return;
            }
            System.out.println("Complete path = " + getCompletePath());
            File f = new File(getCompletePath());
            if (!f.exists()) {
                try {
                    f.createNewFile();
                    System.out.println(getCompletePath() + " successfully created...");
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        public boolean deleteFile() {
            if (getCompletePath() == null) {
                System.out.println("The file path is null! Can not delete the file...");
                return false;
            }
            File f = new File(getCompletePath());
            boolean deleted = false;
            if (!f.exists()) {
                System.out.println(getCompletePath() + " does not exist! Can not delete the file...");
            } else {
                if (!f.isDirectory()) {
                    deleted = f.delete();
                    System.out.println(getCompletePath() + " successfully deleted...");
                }
            }
            return deleted;
        }
    }

    public class KeyboardInputClass {

        public String getKeyboardInput(String prompt) {
            String inputString = "";
            System.out.println(prompt);
            try {
                InputStreamReader reader = new InputStreamReader(System.in);
                BufferedReader buffer = new BufferedReader(reader);
                inputString = buffer.readLine();
            } catch (Exception e) {
                e.printStackTrace();
            }
            return inputString;
        }

        public char getCharacter(boolean validateInput, char defaultResult, String validEntries, int caseConversionMode, String prompt) {
            if (validateInput) {
                if (caseConversionMode == 1) {
                    validEntries = validEntries.toUpperCase();
                    defaultResult = Character.toUpperCase(defaultResult);
                } else if (caseConversionMode == 2) {
                    validEntries = validEntries.toLowerCase();
                    defaultResult = Character.toLowerCase(defaultResult);
                }
                if ((validEntries.indexOf(defaultResult) < 0)) { //if default not in validEntries
                    validEntries = (new Character(defaultResult)).toString() + validEntries;//then add it
                }
            }
            String inputString = "";
            char result = defaultResult;
            boolean entryAccepted = false;
            while (!entryAccepted) {
                result = defaultResult;
                entryAccepted = true;
                inputString = getKeyboardInput(prompt);
                if (inputString.length() > 0) {
                    result = (inputString.charAt(0));
                    if (caseConversionMode == 1) {
                        result = Character.toUpperCase(result);
                    } else if (caseConversionMode == 2) {
                        result = Character.toLowerCase(result);
                    }
                }
                if (validateInput) {
                    if (validEntries.indexOf(result) < 0) {
                        entryAccepted = false;
                        System.out.println("Invalid entry. Select an entry from the characters shown in brackets: [" + validEntries + "]");
                    }
                }
            }
            return result;
        }

        public int getInteger(boolean validateInput, int defaultResult, int minAllowableResult, int maxAllowableResult, String prompt) {
            String inputString = "";
            int result = defaultResult;
            boolean entryAccepted = false;
            while (!entryAccepted) {
                result = defaultResult;
                entryAccepted = true;
                inputString = getKeyboardInput(prompt);
                if (inputString.length() > 0) {
                    try {
                        result = Integer.parseInt(inputString);
                    } catch (Exception e) {
                        entryAccepted = false;
                        System.out.println("Invalid entry...");
                    }
                }
                if (entryAccepted && validateInput) {
                    if ((result != defaultResult) && ((result < minAllowableResult) || (result > maxAllowableResult))) {
                        entryAccepted = false;
                        System.out.println("Invalid entry. Allowable range is " + minAllowableResult + "..." + maxAllowableResult + " (default = " + defaultResult + ").");
                    }
                }
            }
            return result;
        }

        public double getDouble(boolean validateInput, double defaultResult, double minAllowableResult, double maxAllowableResult, String prompt) {
            String inputString = "";
            double result = defaultResult;
            boolean entryAccepted = false;
            while (!entryAccepted) {
                result = defaultResult;
                entryAccepted = true;
                inputString = getKeyboardInput(prompt);
                if (inputString.length() > 0) {
                    try {
                        result = Double.parseDouble(inputString);
                    } catch (Exception e) {
                        entryAccepted = false;
                        System.out.println("Invalid entry...");
                    }
                }
                if (entryAccepted && validateInput) {
                    if ((result != defaultResult) && ((result < minAllowableResult) || (result > maxAllowableResult))) {
                        entryAccepted = false;
                        System.out.println("Invalid entry. Allowable range is " + minAllowableResult + "..." + maxAllowableResult + " (default = " + defaultResult + ").");
                    }
                }
            }
            return result;
        }

        public String getString(String defaultResult, String prompt) {
            String result = getKeyboardInput(prompt);
            if (result.length() == 0) {
                result = defaultResult;
            }
            return result;
        }

    }

    public class Random {

        public Random() {

        }

        public int randomIntInARange(int min, int max) {
            return (min + (int) (Math.random() * ((max - min) + 1)));
        }

        public double randomDoubleInARange(double min, double max) {
            return (min + (max - min) * new java.util.Random().nextDouble());
        }

        public float randomFloatInARange(float min, float max){
            return (min + (max - min) * new java.util.Random().nextFloat());
        }

    }

}
