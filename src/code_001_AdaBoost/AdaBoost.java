package code_001_AdaBoost;

import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

//其实实现一个AdaBoost没必要实现BasicModel，但是这么写方便通过BasicModel接口获取接口的实现类（存在多个），便于拓展
public class AdaBoost implements BasicModel{

    private double [][] x_train;
    private int [] y_train;
    private double [][] x_test;
    private int [] y_test;
    private double []w;  //样本数据的权值
    private double []e = new double[TIMES];  //每个弱分类器对应的最小误分类率
    private double []a = new double[TIMES];  //每个弱分类器对应的系数
    private double borderIndex_m[]; //用来记录每一步弱分类器的边界索引
    private double borderleft_m[];  //用来记录每一步弱分类器的左边边界类别
    Map<Double,Integer> xMap = new HashMap<>();
    private LinkedList<Double> x_borderList = new LinkedList<>();   //在迭代更新分段函数中，保存一系列间断点
    private LinkedList<Double> y_borderList = new LinkedList<>();  //在迭代更新分段函数中，保存一系列间断点区间的函数值
    private double threshold = 0.01;  //设置迭代过程中误分类率的阈值
    private final static int TIMES = 2000;  //算法迭代次数
    private String f_x;  //输出每一步的基本分类器的线性组合
    private String G_x;  //输出最终的强分类器
    private int[] outdata;  //测试集的预测值

    /**
     * 通过时间序列预测分析
     */
    public void calculate(){

        DecimalFormat dF = new DecimalFormat("0.00000000");

        /**
         * 1、初始化权值w：一开始所有点都未通过函数预测分类，所以各样本数据权值相同，均为1/N
         */
        w = new double[x_train.length];
        for (int i = 0; i < x_train.length; i++) {
            w[i] = Double.parseDouble(dF.format((float)1/x_train.length));
        }

        int m = 0;  //第m步
        e[0] = x_train.length;//初始化第一个弱分类器的误分类率e[0]

        //分类误差率 = 0, 则跳出循环
        BigDecimal e_m = new BigDecimal(e[0]);
        BigDecimal thres = new BigDecimal(threshold);
        while(e_m.compareTo(thres) > 0){

            if(m >= TIMES) break;   //限制迭代次数为50次

            /**
             * 2、计算在该弱分类器上最小的分类误差率e_m
             *   如何求是个问题，大致步骤如下：
             *  a、找到分类的边界，即左边为1或-1，右边为-1或1，记能找到边界个  数为k。
             *  b、如果该边界（q）左边为1，右边为-1，则分类函数为: x < q为1，x>q为-1。对于边界（q）左边为-1，右边为1，分类函数定义则同理
             *  c、经过k次循环，计算误分类最小值e_m
             */
            int h = 0;
            double borderIndex[] = new double[x_train.length];  //可能存在的边界
            int borderleft[] = new int[x_train.length];  //左边界
            for (int i = 0; i < x_train.length; i++) {
                  if(i!= x_train.length - 1 && y_train[i] != y_train[i + 1])  {
                      borderleft[h] = y_train[i];
                      borderIndex[h++] = x_train[i][0] + 0.5;  //只选第一位特征进行计算（时间序列）
                  }
            }

            double e_temp[] = new double[h]; //所有边界可能取到的分类误差率
            for (int i = 0; i < h; i++) {

                double temp = 0.0;
                //累加左边w·误分类点
                for(int j = 0; x_train[j][0] < borderIndex[i]; j++){
                    if(y_train[j] != borderleft[i]){
                        temp += w[j] * 1;
                    }
                }

                //累加右边w·误分类点
                for(int j = (int)Math.ceil(borderIndex[i]); j < x_train.length; j++){
                    if(y_train[j] != borderleft[i] * (-1)){
                        temp += w[j] * 1;
                    }
                }
                e_temp[i] = temp;
            }

            //计算得到该弱分类器最小e_m,以及边界索引，边界左边的值
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
             * 3、通过e_m计算每一步加法模型G(x)的系数α_m
             */
            a[m] = 0.5 * Math.log((double)(1 - e[m])/e[m]);

            /**
             * 4、更新权值w，对样本数据进行权重的重排：
             */
            double Z_m = 0.0;  //规范化因子
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

            //判断w和是否为1
            double d = 0.0;
            for (int i = 0; i < x_train.length; i++) {
                d += w[i];
            }
            System.out.println("w和为：" + d);

            /**
             * 5、基本分类器的线性组合构造f_x：这里主要是更新x_borderList，y_borderList这两个列表
             */
            func_LinearCombination(a[m], borderIndex_m[m], borderleft_m[m], (-1) * borderleft_m[m]);

            f_x = getF_x();
            G_x = getG_x();

            System.out.print("m = " + m + "  ");
            System.out.print("误差率e_m = " + e[m] + "   ");
            System.out.print("基函数系数a_m = " + a[m] + "   ");
            System.out.print("f(x) = {" + f_x + ")" + "   ");

            //通过G_x判断训练数据，如果不存在误分类点，即G_x为最终的强分类器
            int count = getMissPointCount();
            if(count == 0){
                System.out.println("G_x误分类样本点为" + count + "个");
                break;
            }
            System.out.println("G_x误分类样本点为" + count + "个");

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
                //注意数组中存入-1 == 0，而不是真实的-1
                else{
                    outdata[i] = -1;
                }
            }
            else if(x_test[i][0] > x_borderList.getLast()){
                double val = y_borderList.getLast();
                if(val > 0.0){
                    outdata[i] = 1;
                }
                //注意数组中存入-1 == 0，而不是真实的-1
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
                        //注意数组中存入-1 == 0，而不是真实的-1
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

    //基函数的线性组合
    public void func_LinearCombination(double a_m, double borderIndex, double borderleft, double borderRight){

        //第一次迭代
        borderleft = a_m * borderleft;
        borderRight = a_m * borderRight;
        if(x_borderList.size() == 0){
            x_borderList.add(new Double(borderIndex));
            y_borderList.add(new Double(borderleft));
            y_borderList.add(new Double(borderRight));
            xMap.put(borderIndex,0);
        }
        //非第一次迭代，则插入原有的分段函数
        else{
            //找到插入的位置
            int insertIndex = 0;

            System.out.print("borderIndex: " + borderIndex + ",   ");

            //在表头插入
            if(borderIndex < x_borderList.getFirst()){
                insertIndex = 0;
                // borderleft + 第一个y，borderRight + 列表所有y
                y_borderList.add(insertIndex,borderleft + y_borderList.getFirst());

                for (int i = 1; i < y_borderList.size(); i++) {
                    y_borderList.set(i,(borderRight * a_m) + y_borderList.get(i));
                }
            }
            //在表末插入
            else if(borderIndex > x_borderList.getLast()){
                insertIndex = x_borderList.size();

                // borderleft + 所有y，borderRight + 最后一个y
                y_borderList.add(insertIndex + 1,borderRight + y_borderList.getLast());

                for (int i = 0; i < y_borderList.size() - 1; i++) {
                    y_borderList.set(i,borderleft + y_borderList.get(i));
                }
            }
            //在表内插入
            else{
                //新的分段函数间断点在累加的分段函数中存在
                if(xMap.get(borderIndex) != null){

                    //borderleft + y插入位置前所有元素（包含插入位置）
                    for (int i = 0; i <= xMap.get(borderIndex); i++) {
                        y_borderList.set(i, borderleft + y_borderList.get(i));
                    }

                    //borderRight + y插入位置后所有元素
                    for (int i = xMap.get(borderIndex); i < x_borderList.size(); i++) {
                        y_borderList.set(i + 1, borderRight + y_borderList.get(i));
                    }
                }
                else{
                    for (int i = 1; i < x_borderList.size(); i++) {
                        if(x_borderList.get(i) > borderIndex){
                            insertIndex = i;
                            // borderleft + [y_first...y_index]，borderRight + [y_index...y_end]
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
            //插入borderIndex
            if(xMap.get(borderIndex) == null){
                x_borderList.add(insertIndex,borderIndex);
                xMap.put(borderIndex,insertIndex);
            }
        }

        System.out.print("x边界列表:");
        for (Double d: x_borderList) {
            System.out.print(d + ",");
        }

        System.out.print("    ");
        System.out.print("y边界列表:");
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

            //样本x值在G_x分段函数的首部
            if(x_train[i][0] < x_borderList.getFirst()) {
                if(isMissPoint(y_train[i],y_borderList.getFirst())) {
                    missPointCount++;
                }
            }
            //样本x值在G_x分段函数的末尾
            else if(x_train[i][0] > x_borderList.getLast()) {
                if(isMissPoint(y_train[i],y_borderList.getLast())) {
                    missPointCount++;
                }
            }
            //样本x值在G_x分段函数的中间
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
