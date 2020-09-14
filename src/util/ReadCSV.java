package util;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;

import com.csvreader.CsvReader;
import javafx.beans.binding.IntegerBinding;

/**
 * �ο� https://blog.51cto.com/lavasoft/265821��https://blog.csdn.net/galen2016/article/details/78119658
 */
public class ReadCSV {

    private ArrayList<String[]> csvList = new ArrayList<String[]>(); //��װ�ļ�����
    private int dimensions;  //����ά��
    private double [][]inputdata;
    private String featureName[];

    public void readCsvFile(String filePath) {

        try {
            CsvReader reader = new CsvReader(filePath, ',', Charset.forName("GBK"));
            //reader.readHeaders(); //������ͷ,��������ע�͵�

            while (reader.readRecord()) {
                csvList.add(reader.getValues()); //���ж�ȡ������ÿһ�е�������ӵ�list����
            }
            reader.close();
        }catch (Exception e) {
            e.printStackTrace();
        }
    }

    public double[][] getInputData(String filePath){

        readCsvFile(filePath);

        dimensions = csvList.get(0).length;
        featureName = new String[dimensions];
        inputdata = new double[csvList.size() - 1][dimensions];

        for (int i = 0; i < dimensions; i++) {
            featureName[i] = csvList.get(0)[i];
        }


        for (int i = 1; i < csvList.size(); i++) {
            for(int j = 0; j < dimensions; ++j){
                //System.out.print(csvList.get(i)[j] + ',');

                //���ַ�������ת�������ͣ���װ����ά����
                inputdata[i-1][j] = Double.parseDouble(csvList.get(i)[j]);
            }
                //System.out.println();
        }

        return inputdata;
    }

    public String[] getFeatureName() {
        return featureName;
    }

    public static void main(String[] args) {
//        String filePath = "resource/dataset/failureData.csv";
        ReadCSV readCSV = new ReadCSV();
//        readCSV.readCsvFile(filePath);

        double[][] inputData = readCSV.getInputData("resource/dataset/data.csv");

        for (int i = 0; i < inputData.length; i++) {
            for (int j = 0; j < inputData[0].length; j++) {
                System.out.print(inputData[i][j] + " ");
            }
            System.out.println();
        }
    }
}