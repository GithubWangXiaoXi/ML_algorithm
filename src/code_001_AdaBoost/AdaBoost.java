package code_001_AdaBoost;

import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

//��ʵʵ��һ��AdaBoostû��Ҫʵ��BasicModel��������ôд����ͨ��BasicModel�ӿڻ�ȡ�ӿڵ�ʵ���ࣨ���ڶ������������չ
public class AdaBoost implements BasicModel{

    private double [][] x_train;
    private int [] y_train;
    private double [][] x_test;
    private int [] y_test;
    private double []w;  //�������ݵ�Ȩֵ
    private double []e = new double[TIMES];  //ÿ������������Ӧ����С�������
    private double []a = new double[TIMES];  //ÿ������������Ӧ��ϵ��
    private double borderIndex_m[]; //������¼ÿһ�����������ı߽�����
    private double borderleft_m[];  //������¼ÿһ��������������߽߱����
    Map<Double,Integer> xMap = new HashMap<>();
    private LinkedList<Double> x_borderList = new LinkedList<>();   //�ڵ������·ֶκ����У�����һϵ�м�ϵ�
    private LinkedList<Double> y_borderList = new LinkedList<>();  //�ڵ������·ֶκ����У�����һϵ�м�ϵ�����ĺ���ֵ
    private double threshold = 0.01;  //���õ���������������ʵ���ֵ
    private final static int TIMES = 2000;  //�㷨��������
    private String f_x;  //���ÿһ���Ļ������������������
    private String G_x;  //������յ�ǿ������
    private int[] outdata;  //���Լ���Ԥ��ֵ

    /**
     * ͨ��ʱ������Ԥ�����
     */
    public void calculate(){

        DecimalFormat dF = new DecimalFormat("0.00000000");

        /**
         * 1����ʼ��Ȩֵw��һ��ʼ���е㶼δͨ������Ԥ����࣬���Ը���������Ȩֵ��ͬ����Ϊ1/N
         */
        w = new double[x_train.length];
        for (int i = 0; i < x_train.length; i++) {
            w[i] = Double.parseDouble(dF.format((float)1/x_train.length));
        }

        int m = 0;  //��m��
        e[0] = x_train.length;//��ʼ����һ�������������������e[0]

        //��������� = 0, ������ѭ��
        BigDecimal e_m = new BigDecimal(e[0]);
        BigDecimal thres = new BigDecimal(threshold);
        while(e_m.compareTo(thres) > 0){

            if(m >= TIMES) break;   //���Ƶ�������Ϊ50��

            /**
             * 2�������ڸ�������������С�ķ��������e_m
             *   ������Ǹ����⣬���²������£�
             *  a���ҵ�����ı߽磬�����Ϊ1��-1���ұ�Ϊ-1��1�������ҵ��߽��  ��Ϊk��
             *  b������ñ߽磨q�����Ϊ1���ұ�Ϊ-1������ຯ��Ϊ: x < qΪ1��x>qΪ-1�����ڱ߽磨q�����Ϊ-1���ұ�Ϊ1�����ຯ��������ͬ��
             *  c������k��ѭ���������������Сֵe_m
             */
            int h = 0;
            double borderIndex[] = new double[x_train.length];  //���ܴ��ڵı߽�
            int borderleft[] = new int[x_train.length];  //��߽�
            for (int i = 0; i < x_train.length; i++) {
                  if(i!= x_train.length - 1 && y_train[i] != y_train[i + 1])  {
                      borderleft[h] = y_train[i];
                      borderIndex[h++] = x_train[i][0] + 0.5;  //ֻѡ��һλ�������м��㣨ʱ�����У�
                  }
            }

            double e_temp[] = new double[h]; //���б߽����ȡ���ķ��������
            for (int i = 0; i < h; i++) {

                double temp = 0.0;
                //�ۼ����w��������
                for(int j = 0; x_train[j][0] < borderIndex[i]; j++){
                    if(y_train[j] != borderleft[i]){
                        temp += w[j] * 1;
                    }
                }

                //�ۼ��ұ�w��������
                for(int j = (int)Math.ceil(borderIndex[i]); j < x_train.length; j++){
                    if(y_train[j] != borderleft[i] * (-1)){
                        temp += w[j] * 1;
                    }
                }
                e_temp[i] = temp;
            }

            //����õ�������������Сe_m,�Լ��߽��������߽���ߵ�ֵ
            double min  = e_temp[0];
            borderIndex_m[m] = borderIndex[0];
            borderleft_m[m] = borderleft[0];
            for (int i = 1; i < h; i++) {
                if(e_temp[i] < min){
                    min = e_temp[i];
                    borderIndex_m[m] = borderIndex[i];
                    borderleft_m[m] =  borderleft[i];
                }
            }

            e[m] = min;

            /**
             * 3��ͨ��e_m����ÿһ���ӷ�ģ��G(x)��ϵ����_m
             */
            a[m] = 0.5 * Math.log((double)(1 - e[m])/e[m]);

            /**
             * 4������Ȩֵw�����������ݽ���Ȩ�ص����ţ�
             */
            double Z_m = 0.0;  //�淶������
            for (int i = 0; i < x_train.length; i++) {
                if(i + 1 < borderIndex_m[m]){
                    Z_m += w[i] * Math.exp(-a[m] * y_train[i] * borderleft_m[m]);
                }else{
                    Z_m += w[i] * Math.exp(-a[m] * y_train[i] * borderleft_m[m] * (-1));
                }
            }

            for (int i = 0; i < x_train.length; i++) {
                if(i + 1 < borderIndex_m[m]){
                    w[i] = (double)(w[i] * Math.exp(-a[m] * y_train[i] * borderleft_m[m]))/Z_m;
                }else{
                    w[i] = (double)(w[i] * Math.exp(-a[m] * y_train[i] * borderleft_m[m] * (-1)))/Z_m;
                }
            }

            //�ж�w���Ƿ�Ϊ1
            double d = 0.0;
            for (int i = 0; i < x_train.length; i++) {
                d += w[i];
            }
            System.out.println("w��Ϊ��" + d);

            /**
             * 5��������������������Ϲ���f_x��������Ҫ�Ǹ���x_borderList��y_borderList�������б�
             */
            func_LinearCombination(a[m], borderIndex_m[m], borderleft_m[m], (-1) * borderleft_m[m]);

            f_x = getF_x();
            G_x = getG_x();

            System.out.print("m = " + m + "  ");
            System.out.print("�����e_m = " + e[m] + "   ");
            System.out.print("������ϵ��a_m = " + a[m] + "   ");
            System.out.print("f(x) = {" + f_x + ")" + "   ");

            //ͨ��G_x�ж�ѵ�����ݣ���������������㣬��G_xΪ���յ�ǿ������
            int count = getMissPointCount();
            if(count == 0){
                System.out.println("G_x�����������Ϊ" + count + "��");
                break;
            }
            System.out.println("G_x�����������Ϊ" + count + "��");

            e_m = new BigDecimal(e[m]);
            m++;
        }
    }

    @Override
    public void inputdata(double[][] x_trainP, int[] y_trainP, double[][] x_testP, int[] y_testP) {
        this.x_train = x_trainP;
        this.y_train = y_trainP;
        this.x_test = x_testP;
        this.y_test = y_testP;
        this.borderIndex_m = new double[TIMES];
        this.borderleft_m = new double[TIMES];
    }

    @Override
    public int[] getoutdata() {
        outdata = new int[x_test.length];

        for (int i = 0; i < x_test.length; i++) {

            if(x_test[i][0] < x_borderList.getFirst()) {

                double val = y_borderList.getFirst();
                if(val > 0.0){
                    outdata[i] = 1;
                }
                //ע�������д���-1 == 0����������ʵ��-1
                else{
                    outdata[i] = -1;
                }
            }
            else if(x_test[i][0] > x_borderList.getLast()){
                double val = y_borderList.getLast();
                if(val > 0.0){
                    outdata[i] = 1;
                }
                //ע�������д���-1 == 0����������ʵ��-1
                else{
                    outdata[i] = -1;
                }
            }
            else{
                for (int j = 1; j < x_borderList.size(); j++) {
                    if(x_borderList.get(j) > x_test[i][0]){
                        double val = y_borderList.get(j);
                        //System.out.println(">>>>>>>>>>>" + val);
                        if(val > 0.0){
                            outdata[i] = 1;
                        }
                        //ע�������д���-1 == 0����������ʵ��-1
                        else{
                            outdata[i] = -1;
                        }
                        break;
                    }
                }
            }
        }
        return outdata;
    }

    @Override
    public double getCorrectRate() {
        double correctRate = 0.0;

        int num = 0;
        for (int i = 0; i < x_test.length; i++) {
            if (y_test[i] == -1 && outdata[i] == 0){
                num++;
            }else if(y_test[i] == outdata[i]){
                num++;
            }
        }
        System.out.println(num + "/" + x_test.length);

        DecimalFormat dF = new DecimalFormat("0.0000");

        return Double.parseDouble(dF.format((float) num / x_test.length));
    }

    //���������������
    public void func_LinearCombination(double a_m, double borderIndex, double borderleft, double borderRight){

        //��һ�ε���
        borderleft = a_m * borderleft;
        borderRight = a_m * borderRight;
        if(x_borderList.size() == 0){
            x_borderList.add(new Double(borderIndex));
            y_borderList.add(new Double(borderleft));
            y_borderList.add(new Double(borderRight));
            xMap.put(borderIndex,0);
        }
        //�ǵ�һ�ε����������ԭ�еķֶκ���
        else{
            //�ҵ������λ��
            int insertIndex = 0;

            System.out.print("borderIndex: " + borderIndex + ",   ");

            //�ڱ�ͷ����
            if(borderIndex < x_borderList.getFirst()){
                insertIndex = 0;
                // borderleft + ��һ��y��borderRight + �б�����y
                y_borderList.add(insertIndex,borderleft + y_borderList.getFirst());

                for (int i = 1; i < y_borderList.size(); i++) {
                    y_borderList.set(i,(borderRight * a_m) + y_borderList.get(i));
                }
            }
            //�ڱ�ĩ����
            else if(borderIndex > x_borderList.getLast()){
                insertIndex = x_borderList.size();

                // borderleft + ����y��borderRight + ���һ��y
                y_borderList.add(insertIndex + 1,borderRight + y_borderList.getLast());

                for (int i = 0; i < y_borderList.size() - 1; i++) {
                    y_borderList.set(i,borderleft + y_borderList.get(i));
                }
            }
            //�ڱ��ڲ���
            else{
                //�µķֶκ�����ϵ����ۼӵķֶκ����д���
                if(xMap.get(borderIndex) != null){

                    //borderleft + y����λ��ǰ����Ԫ�أ���������λ�ã�
                    for (int i = 0; i <= xMap.get(borderIndex); i++) {
                        y_borderList.set(i, borderleft + y_borderList.get(i));
                    }

                    //borderRight + y����λ�ú�����Ԫ��
                    for (int i = xMap.get(borderIndex); i < x_borderList.size(); i++) {
                        y_borderList.set(i + 1, borderRight + y_borderList.get(i));
                    }
                }
                else{
                    for (int i = 1; i < x_borderList.size(); i++) {
                        if(x_borderList.get(i) > borderIndex){
                            insertIndex = i;
                            // borderleft + [y_first...y_index]��borderRight + [y_index...y_end]
                            for (int j = 0; j <= insertIndex; j++) {
                                y_borderList.set(j,borderleft + y_borderList.get(j));
                            }

                            y_borderList.add(insertIndex + 1, borderRight + y_borderList.get(insertIndex));

                            for (int j = insertIndex + 2 ; j <= x_borderList.size(); j++) {
                                y_borderList.set(j,borderRight + y_borderList.get(j));
                            }

                            break;
                        }
                    }
                }
            }
            //����borderIndex
            if(xMap.get(borderIndex) == null){
                x_borderList.add(insertIndex,borderIndex);
                xMap.put(borderIndex,insertIndex);
            }
        }

        System.out.print("x�߽��б�:");
        for (Double d: x_borderList) {
            System.out.print(d + ",");
        }

        System.out.print("    ");
        System.out.print("y�߽��б�:");
        for (Double d: y_borderList) {
            System.out.print(d + ",");
        }
        System.out.println();
    }

    public String getF_x(){

        String str = "";

        str += "x < " + x_borderList.get(0) + ", y = " + y_borderList.get(0) + "; ";

        for (int i = 1; i < x_borderList.size(); i++) {
            str += x_borderList.get(i-1) + "< x <" + x_borderList.get(i) + ", y = " + y_borderList.get(i) + "; ";
        }

        str += "x > " + x_borderList.get(x_borderList.size() - 1) + ", y = " + y_borderList.getLast() + "; ";

        return str;
    }

    public String getG_x(){
        return "sign[ ( " + f_x + " ) ]";
    }

    public int getMissPointCount() {
        int missPointCount = 0;

        for (int i = 0; i < x_train.length; i++) {

            //����xֵ��G_x�ֶκ������ײ�
            if(x_train[i][0] < x_borderList.getFirst()) {
                if(isMissPoint(y_train[i],y_borderList.getFirst())) {
                    missPointCount++;
                }
            }
            //����xֵ��G_x�ֶκ�����ĩβ
            else if(x_train[i][0] > x_borderList.getLast()) {
                if(isMissPoint(y_train[i],y_borderList.getLast())) {
                    missPointCount++;
                }
            }
            //����xֵ��G_x�ֶκ������м�
            else {
                for(int j = 0; j < x_borderList.size(); j++) {
                    if (x_borderList.get(j) > x_train[i][0]) {
                        double val = y_borderList.get(j);
                        if(isMissPoint(y_train[i],val)){
                            missPointCount++;
                        }
                        break;
                    }
                }
            }
        }

        return missPointCount;
    }


    public boolean isMissPoint(int y_real, double y_predict) {

        int predict = 0;
        if(y_predict > 0.0){
            predict = 1;
        }else{
            predict = -1;
        }

        if (y_real == predict) return false;
        return true;
    }

}
