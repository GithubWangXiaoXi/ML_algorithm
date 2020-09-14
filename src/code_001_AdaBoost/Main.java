package code_001_AdaBoost;

import util.ReadCSV;
import util.StrHandler;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;
import java.util.Scanner;

public class Main {


    public static void main(String[] args) {

        /**
         * step1����������
         */
        ReadCSV readCSV = new ReadCSV();
        double[][] inputData = readCSV.getInputData("resource/dataset/failureData.csv");
        String[] featureName = readCSV.getFeatureName();

        System.out.print("�����ֶ�:  ");
        for (String str: featureName) {
            System.out.print(str + "  ");
        }
        System.out.println();
        System.out.println("----------------------------------------------");

        /**
         * step2��ѡ�������ֶκ�Ԥ���ֶ�
         */
        int []XIndex = new int[inputData.length];
        int YIndex;

        System.out.println("��ɸѡ��Ҫ�������ֶ�(eg��1,2): 1-time, 2-failureCount, 3-label");
        Scanner scanner = new Scanner(System.in);

        //���û����������֤�ʹ���
        boolean flag = false;
        while(!flag){
            String in = scanner.next();
            String[] split = in.split(",");
            int i = 0;

            if(StrHandler.hasNonNum(split)){
                System.out.println("����������");
            }
            else if (split.length > inputData.length - 1){
                System.out.println("ɸѡ������Χ");
            }
            else if(StrHandler.isDuplicate(split)){
                System.out.println("ɸѡ�ظ�");
            }
            else{
                XIndex = new int[split.length];
                for(String str : split){
                    XIndex[i++] = Integer.parseInt(str);
                }
                flag = true;
            }
        }

        System.out.print("��ɸѡ�����ֶ��ǣ�");
        for (int j = 0; j < XIndex.length; j++) {
            System.out.print(featureName[XIndex[j] - 1] + "  ");
        }
        System.out.println();

        System.out.println("��ɸѡ��Ҫ��Ԥ���ֶ�(ֻ��ѡһ��): 1-time, 2-failureCount, 3-label");
        int in1 = scanner.nextInt();
        YIndex = in1;

        System.out.print("��ɸѡԤ���ֶ��ǣ�");
        System.out.println(featureName[YIndex - 1]);
        System.out.println("----------------------------------------------");

        /**
         * step3: �������ݱ�������
         */
        System.out.println("��������Ҫ���в������ݻ��ֵı�������λ%��");
        flag = false;

        int j = 0;
        while(!flag){
            int i = scanner.nextInt();
            if (i > 100 || i < 0){
                System.out.println("������Χ����׼ȷ��д");
            }else{
                j = i;
                flag = true;
            }
        }
        System.out.println("������Ĳ������ݻ��ֱ����ǣ�" + j + "%");
        //System.out.println(rate);
        System.out.println("----------------------------------------------");

        /**
         * step4����������зŻصĳ����õ�ѵ���������Լ�
         */
        double[][] x_train, x_test;
        int []y_train, y_test;
        DecimalFormat dF = new DecimalFormat("0.00000000");
        double rate = Double.parseDouble(dF.format((double)j/100));

        int test_size = (int) Math.floor(rate * inputData.length);
        int train_size = inputData.length - test_size;

        //����зŻصĳ���������index˳������
        int []test_RandomWithOrder = new int[test_size];
        int []train_RandomWithOrder = new int[train_size];
        getNum(test_RandomWithOrder,inputData.length);
        getNum(train_RandomWithOrder,inputData.length);

        //���Լ�
        x_test = new double[test_size][XIndex.length];
        y_test = new int[test_size];
        int k = 0;
        for (int i = 0; i < test_size; i++) {
            for (int l = 0; l < XIndex.length; l++){
                if(inputData[test_RandomWithOrder[i]][YIndex - 1] == 0.0){
                    y_test[k] = -1;
                }else{
                    y_test[k] = (int)inputData[test_RandomWithOrder[i]][YIndex - 1];
                }
                x_test[k][l] = inputData[test_RandomWithOrder[i]][XIndex[l] - 1];
            }
            k++;
        }

        //ѵ����
        k = 0;
        x_train = new double[train_size][XIndex.length];
        y_train = new int[train_size];
        for (int i = 0; i < train_size; i++) {
            for (int l = 0; l < XIndex.length; l++){
                if(inputData[train_RandomWithOrder[i]][YIndex - 1] == 0.0){
                    y_train[k] = -1;
                }else{
                    y_train[k] = (int)inputData[train_RandomWithOrder[i]][YIndex - 1];
                }
                x_train[k][l] = inputData[train_RandomWithOrder[i]][XIndex[l] - 1];
            }
            k++;
        }

        System.out.println("��Ĳ������ݼ�����");
        for (int i = 0; i < test_size; i++) {
            for (int l = 0; l < XIndex.length; l++) {
                System.out.print(x_test[i][l] + "  ");
            }
            System.out.println(y_test[i]);
        }

        System.out.println();
        System.out.println("���ѵ�����ݼ�����");
        for (int i = 0; i < train_size; i++) {
            for (int l = 0; l < XIndex.length; l++) {
                System.out.print(x_train[i][l] + "  ");
            }
            System.out.println(y_train[i]);
        }

        /**
         * step5�������ݵ���AdaBoostģ��
         */
        AdaBoost adaBoost = new AdaBoost();
        adaBoost.inputdata(x_train,y_train,x_test,y_test);
        adaBoost.calculate();

        System.out.println("----------------------------------------------");
        System.out.println("�������ǿ������");
        String g_x = adaBoost.getG_x();
        System.out.println(g_x);

        int[] outdata = adaBoost.getoutdata();

        for (int l = 0; l < XIndex.length; l++) {
            System.out.print(featureName[XIndex[l] - 1] + "  ");
        }
        System.out.println(featureName[YIndex - 1] + "  " + "Ԥ��ֵ");

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
        System.out.println("׼ȷ�ʣ�" + correctRate * 100 + "%");
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
