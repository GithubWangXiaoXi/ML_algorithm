package code_001_AdaBoost;

import util.ReadCSV;
import util.StrHandler;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;
import java.util.Scanner;

public class Main1 {


    public static void main(String[] args) {

        /**
         * step1：导入数据
         */
        ReadCSV readCSV = new ReadCSV();
        double[][] inputData = readCSV.getInputData("resource/dataset/data.csv");
        String[] featureName = readCSV.getFeatureName();

        System.out.print("数据字段:  ");
        for (String str: featureName) {
            System.out.print(str + "  ");
        }
        System.out.println();
        System.out.println("----------------------------------------------");

        /**
         * step2：选择特征字段和预测字段
         */
        int []XIndex = new int[inputData.length];
        int YIndex;

        System.out.println("请筛选你要的特征字段(eg：1,2): 1-x, 2-y");
        Scanner scanner = new Scanner(System.in);

        //对用户输入进行验证和处理
        boolean flag = false;
        while(!flag){
            String in = scanner.next();
            String[] split = in.split(",");
            int i = 0;

            if(StrHandler.hasNonNum(split)){
                System.out.println("请输入数字");
            }
            else if (split.length > inputData.length - 1){
                System.out.println("筛选超出范围");
            }
            else if(StrHandler.isDuplicate(split)){
                System.out.println("筛选重复");
            }
            else{
                XIndex = new int[split.length];
                for(String str : split){
                    XIndex[i++] = Integer.parseInt(str);
                }
                flag = true;
            }
        }

        System.out.print("你筛选特征字段是：");
        for (int j = 0; j < XIndex.length; j++) {
            System.out.print(featureName[XIndex[j] - 1] + "  ");
        }
        System.out.println();

        System.out.println("请筛选你要的预测字段(只能选一个): 1-x, 2-y");
        int in1 = scanner.nextInt();
        YIndex = in1;

        System.out.print("你筛选预测字段是：");
        System.out.println(featureName[YIndex - 1]);
        System.out.println("----------------------------------------------");

        /**
         * step3: 得到训练集，测试集
         */
        double[][] x_train, x_test;
        int []y_train, y_test;

        int test_size = inputData.length;
        int train_size = inputData.length;

        //测试集
        x_test = new double[test_size][XIndex.length];
        y_test = new int[test_size];
        int k = 0;
        for (int i = 0; i < test_size; i++) {
            for (int l = 0; l < XIndex.length; l++){
                if(inputData[i][YIndex - 1] == 0.0){
                    y_test[k] = -1;
                }else{
                    y_test[k] = (int)inputData[i][YIndex - 1];
                }
                x_test[k][l] = inputData[i][XIndex[l] - 1];
            }
            k++;
        }

        //训练集
        k = 0;
        x_train = new double[train_size][XIndex.length];
        y_train = new int[train_size];
        for (int i = 0; i < train_size; i++) {
            for (int l = 0; l < XIndex.length; l++){
                if(inputData[i][YIndex - 1] == 0.0){
                    y_train[k] = -1;
                }else{
                    y_train[k] = (int)inputData[i][YIndex - 1];
                }
                x_train[k][l] = inputData[i][XIndex[l] - 1];
            }
            k++;
        }

        System.out.println("你的测试数据集如下");
        for (int i = 0; i < test_size; i++) {
            for (int l = 0; l < XIndex.length; l++) {
                System.out.print(x_test[i][l] + "  ");
            }
            System.out.println(y_test[i]);
        }

        System.out.println();
        System.out.println("你的训练数据集如下");
        for (int i = 0; i < train_size; i++) {
            for (int l = 0; l < XIndex.length; l++) {
                System.out.print(x_train[i][l] + "  ");
            }
            System.out.println(y_train[i]);
        }

        /**
         * step4：将数据导入AdaBoost模型
         */
        AdaBoost adaBoost = new AdaBoost();
        adaBoost.inputdata(x_train,y_train,x_test,y_test);
        adaBoost.calculate();

        System.out.println("----------------------------------------------");
        System.out.println("输出最终强分类器");
        String g_x = adaBoost.getG_x();
        System.out.println(g_x);

        int[] outdata = adaBoost.getoutdata();

        for (int l = 0; l < XIndex.length; l++) {
            System.out.print(featureName[XIndex[l] - 1] + "  ");
        }
        System.out.println(featureName[YIndex - 1] + "  " + "预测值");

        for (int i = 0; i < x_test.length; i++) {
            for (int l = 0; l < XIndex.length; l++) {
                System.out.print(x_test[i][XIndex[l] - 1] + "   ");
            }
            System.out.print(y_test[i] + "  ");

            if (outdata[i] == 0){
                int num = -1;
                System.out.println(num);
            }else{
                System.out.println(outdata[i]);
            }
        }

        double correctRate = adaBoost.getCorrectRate();
        System.out.println("准确率：" + correctRate * 100 + "%");
    }

    public static void getNum(int[] array,int size){

        int []temp = new int[size];

        Random r = new Random();
        for (int i = 0; i < array.length; i++) {

            int number = r.nextInt(size);

            while(temp[number] == 1){
                number = r.nextInt(size);
            }

            array[i] = number;
            temp[number] = 1;
        }

        Arrays.sort(array);
    }
}
