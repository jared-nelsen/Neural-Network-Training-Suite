package Utilities;


import java.io.*;
import java.util.ArrayList;

/**
 *
 * @author <Jared Nelsen>
 */
public class TextFileReaderAndWriter {

    private BufferedReader reader;
    private BufferedWriter writer;

    private ArrayList<String> listOfLines = new ArrayList();
    private ArrayList<String> lineBuffer = new ArrayList();

    private String indexFilePathString;
    public String title;

    public TextFileReaderAndWriter(String indexFileString) {

        this.title = indexFileString;
        this.indexFilePathString = indexFileString;

        try {
            readAndLoadListOfStrings();
        } catch (IOException i) {
            System.out.println("Error in readAndLoadListOfStrings() in TextFileReaderAndWriter constructor");
            i.printStackTrace();
        }

    }

    private void setReader() {
        try {
            this.reader = new BufferedReader(new FileReader(indexFilePathString));
        } catch (FileNotFoundException f) {
            System.out.println("Error in setting Reader for " + indexFilePathString);
            File file = new File(indexFilePathString);
            if (!file.exists()) {
                System.out.println(indexFilePathString + " does not exist!");
            }
            f.printStackTrace();
        }
    }

    private void setWriter() {
        try {
            this.writer = new BufferedWriter(new FileWriter(indexFilePathString));
        } catch (IOException i) {
            System.out.println("Error in setting Writer for " + indexFilePathString);
            File file = new File(indexFilePathString);
            if (!file.exists()) {
                System.out.println(indexFilePathString + " does not exist!");
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
        this.setReader();

        try {
            String line = reader.readLine();
            while(line != null){
                listOfLines.add(line);
                line = reader.readLine();
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
        return new ArrayList(this.listOfLines.subList(startPosition, endPosition));
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

    public void writeLines(ArrayList<String> listToWrite) {
        if (listToWrite.isEmpty()) {
            return;
        }
        writeLine(listToWrite.remove(0));
        writeLines(listToWrite);
    }

    public void writeLineAt(int position, String toWrite) {
        this.listOfLines.add(position, toWrite);
        compileAndWriteIndexFile();
    }

    public void writeLinesAt(int position, ArrayList<String> linesToWrite) {
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

    //Buffer functions for speed
    public void addLineToBuffer(String line) {
        lineBuffer.add(line);
        if (lineBuffer.size() >= 100) {
            writeBufferToFile();
        }
    }

    public void clearBuffer() {
        lineBuffer.clear();
    }

    public void writeBufferToFile() {
        writeLines(lineBuffer);
        clearBuffer();
    }

}
