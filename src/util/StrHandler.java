package util;

public class StrHandler {

    public static boolean isDuplicate(String[] strings){

        boolean flag = false;
        for (int i = 0; i < strings.length; i++){
            for(int j = i + 1; j < strings.length; j++){
                if(strings[i].equals(strings[j])){
                    flag = true;
                    break;
                }
            }
        }
        return flag;
    }

    public static boolean hasNonNum(String[] split) {

        for (int i = 0; i < split.length; i++) {
            char[] chars = split[i].toCharArray();
            for (char c : chars){
                if(!('0' < c && c < '9')){
                    return true;
                }
            }
        }
        return false;
    }
}
