package code_001_AdaBoost;


public interface BasicModel
{
    public void inputdata(double[][] x_trainP, int[] y_trainP, double[][] x_testP, int[] y_testP);  //获取原生数据
    public int[] getoutdata();  //预测测试数据集
    public double getCorrectRate();  //预测准确率
}
