package code_001_AdaBoost;


public interface BasicModel
{
    public void inputdata(double[][] x_trainP, int[] y_trainP, double[][] x_testP, int[] y_testP);  //��ȡԭ������
    public int[] getoutdata();  //Ԥ��������ݼ�
    public double getCorrectRate();  //Ԥ��׼ȷ��
}
