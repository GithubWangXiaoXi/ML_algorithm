package util;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;

import com.csvreader.CsvReader;
import javafx.beans.binding.IntegerBinding;

/**
 * 参考 https://blog.51cto.com/lavasoft/265821，https://blog.csdn.net/galen2016/article/details/78119658
 */
public class ReadCSV {

    private ArrayList<String[]> csvList = new ArrayList<String[]>(); //封装文件数据
    private int dimensions;  //数据维度
    private double [][]inputdata;
    private String featureName[];

    public void readCsvFile(String filePath) {

        try {
            CsvReader reader = new CsvReader(filePath, ',', Charset.forName("GBK"));
            //reader.readHeaders(); //跳过表头,不跳可以注释掉

            while (reader.readRecord()) {
                csvList.add(reader.getValues()); //按行读取，并把每一行的数据添加到list集合
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

                //将字符串数据转换成整型，封装进二维数组
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