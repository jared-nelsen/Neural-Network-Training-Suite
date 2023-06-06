package Utilities;


import java.io.BufferedReader;
import java.io.InputStreamReader;

/**
 *
 * @author <Jared Nelsen>
 */

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
